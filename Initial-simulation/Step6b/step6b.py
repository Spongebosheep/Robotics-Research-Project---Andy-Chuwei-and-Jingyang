import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# =========================
# Step 6B script
# Forward model: arrival-time only
#
# ti = t0 + d_path(ri, s) / v_eff + eps_i
# eps_i ~ N(0, sigma_t^2)
#
# Version B:
#   d_path = surface path / geodesic approximation
#
# Implementation idea:
# - discretize terrain into a grid
# - build a graph on the surface
# - edge weights = 3D segment lengths on the terrain surface
# - run Dijkstra from each deployed sensor node
# - interpolate shortest-path distance fields
#
# Notes:
# - Source is constrained on terrain surface:
#       s(q) = (x, y, h(x, y))
# - No amplitude threshold yet
# - Layouts fixed to 8 nodes
# - Deployed geometry is terrain-conforming and stable
# =========================

# -------------------------
# Global parameters
# -------------------------
L = 1.0
V_EFF = 1.0
SIGMA_T = 0.002

# 6B is heavier than 6A, so defaults are a bit smaller
N_EVENTS = 150
N_SEEDS = 3

SAVE_CSV = False
PLOT_TERRAINS = False
PLOT_DEPLOYMENTS = False
PLOT_SUMMARY = True
PLOT_GEODESIC_FIELD = False
PLOT_SEED = 0

# -------------------------
# Stable deployment weights
# -------------------------
K_EDGE = 1.0
K_CENTROID = 40.0
K_ORIENT = 20.0
K_DRIFT = 0.10

# -------------------------
# Terrain parameters
# -------------------------
HILL_AMP = 0.14
HILL_SIGMA = 0.22
HILL_XC = 0.55
HILL_YC = 0.50

ROUGH_RMS = 0.03
ROUGH_CORR_LEN = 0.18
ROUGH_GRID_N = 129

# -------------------------
# Arrival-time settings
# -------------------------
T0_MIN = 0.0
T0_MAX = 0.2

# -------------------------
# 6B geodesic settings
# -------------------------
GEO_GRID_N = 41         # grid resolution for surface shortest-path approximation
GEO_CONNECTIVITY = 8    # 4 or 8; 8 is smoother
SEARCH_GRID_N = 15      # coarse search grid before local least-squares
N_MULTI_START = 6


# =========================================================
# Layouts
# =========================================================
def normalize_layout(pts, L=1.0, target_center=(0.5, 0.5), target_span=0.72):
    """
    Normalize a 2D layout so that:
      1) centroid is at target_center * L
      2) max span in x/y is target_span * L
    """
    pts = np.asarray(pts, dtype=float)

    c = pts.mean(axis=0)
    q = pts - c

    span_x = q[:, 0].max() - q[:, 0].min()
    span_y = q[:, 1].max() - q[:, 1].min()
    current_span = max(span_x, span_y)

    if current_span <= 1e-12:
        raise ValueError("Degenerate layout with zero span.")

    scale = (target_span * L) / current_span
    q = q * scale

    center = np.array(target_center, dtype=float) * L
    q = q + center
    return q


def make_layout(name, L=1.0):
    """
    Return 8 nominal sensor coordinates in a planar L x L domain.
    """
    if name == "square":
        pts = np.array([
            [-1.5, -0.5],
            [-0.5, -0.5],
            [ 0.5, -0.5],
            [ 1.5, -0.5],
            [-1.5,  0.5],
            [-0.5,  0.5],
            [ 0.5,  0.5],
            [ 1.5,  0.5],
        ], dtype=float)

    elif name == "staggered":
        pts = np.array([
            [-1.5, -0.60],
            [-0.5, -0.60],
            [ 0.5, -0.60],
            [ 1.5, -0.60],
            [-1.0,  0.60],
            [ 0.0,  0.60],
            [ 1.0,  0.60],
            [ 2.0,  0.60],
        ], dtype=float)

    elif name in ["boundary", "boundary_focused"]:
        pts = np.array([
            [-1.5, -1.3],
            [ 0.0, -1.5],
            [ 1.5, -1.3],
            [ 1.7,  0.0],
            [ 1.5,  1.3],
            [ 0.0,  1.5],
            [-1.5,  1.3],
            [-1.7,  0.0],
        ], dtype=float)

    else:
        raise ValueError(f"Unknown layout: {name}")

    return normalize_layout(pts, L=L, target_center=(0.5, 0.5), target_span=0.72)


def delaunay_edges(pts):
    """Use Delaunay triangulation to define net connectivity."""
    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for a, b in combinations(simplex, 2):
            edges.add(tuple(sorted((a, b))))
    return sorted(edges)


# =========================================================
# Terrain models
# =========================================================
def make_terrain(
    kind,
    L=1.0,
    seed=0,
    hill_amp=0.14,
    hill_sigma=0.22,
    hill_xc=0.55,
    hill_yc=0.50,
    rough_rms=0.03,
    rough_corr_len=0.18,
    rough_grid_n=129,
):
    """Terrain model: flat / hill / rough."""
    if kind == "flat":
        def h(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            return np.zeros(np.broadcast(x, y).shape, dtype=float)
        return h

    if kind == "hill":
        A_h = hill_amp * L
        sigma_h = hill_sigma * L
        xc = hill_xc * L
        yc = hill_yc * L

        def h(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            return A_h * np.exp(
                -(((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma_h ** 2))
            )
        return h

    if kind == "rough":
        rng = np.random.default_rng(seed)
        xs = np.linspace(0.0, L, rough_grid_n)
        ys = np.linspace(0.0, L, rough_grid_n)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        white = rng.normal(0.0, 1.0, size=(rough_grid_n, rough_grid_n))

        sigma_r = rough_rms * L
        ell_r = rough_corr_len * L

        kx = 2.0 * np.pi * np.fft.fftfreq(rough_grid_n, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(rough_grid_n, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K2 = KX**2 + KY**2

        Hk = np.exp(-0.25 * (ell_r ** 2) * K2)
        Zk = np.fft.fft2(white) * Hk
        Z = np.fft.ifft2(Zk).real

        Z -= np.mean(Z)
        Z_std = np.std(Z)
        if Z_std < 1e-12:
            Z = np.zeros_like(Z)
        else:
            Z = Z * (sigma_r / Z_std)

        interp = RegularGridInterpolator(
            (xs, ys),
            Z,
            bounds_error=False,
            fill_value=None,
        )

        def h(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            shape = np.broadcast(x, y).shape
            xx = np.broadcast_to(x, shape).ravel()
            yy = np.broadcast_to(y, shape).ravel()
            xx = np.clip(xx, 0.0, L)
            yy = np.clip(yy, 0.0, L)
            pts = np.column_stack([xx, yy])
            zz = interp(pts).reshape(shape)
            return zz

        return h

    raise ValueError(f"Unknown terrain: {kind}")


# =========================================================
# Stable deployment solver
# =========================================================
def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def select_orientation_pair(u):
    """
    Pick the farthest pair of nominal nodes to define a robust global direction.
    """
    n = len(u)
    best_pair = None
    best_dist = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(u[j] - u[i])
            if d > best_dist:
                best_dist = d
                best_pair = (i, j)
    return best_pair


def deployment_reference_data(u):
    """Compute nominal reference quantities for stable deployment."""
    edges = delaunay_edges(u)
    l0 = np.array([np.linalg.norm(u[i] - u[j]) for i, j in edges], dtype=float)
    centroid0 = np.mean(u, axis=0)

    i_ref, j_ref = select_orientation_pair(u)
    v0 = u[j_ref] - u[i_ref]
    theta0 = np.arctan2(v0[1], v0[0])

    return {
        "edges": edges,
        "l0": l0,
        "centroid0": centroid0,
        "orient_pair": (i_ref, j_ref),
        "theta0": theta0,
    }


def deploy_to_terrain_stable(
    u,
    h,
    L=1.0,
    k_edge=1.0,
    k_centroid=40.0,
    k_orient=20.0,
    k_drift=0.10,
):
    """
    Stable terrain-conforming deployment solver.

    Optimization variables: only x_i, y_i
    z_i = h(x_i, y_i)

    Objective:
      1) preserve nominal edge lengths in 3D
      2) centroid close to nominal centroid
      3) orientation close to nominal orientation
      4) light drift regularization
    """
    ref = deployment_reference_data(u)
    edges = ref["edges"]
    l0 = ref["l0"]
    centroid0 = ref["centroid0"]
    i_ref, j_ref = ref["orient_pair"]
    theta0 = ref["theta0"]

    def residuals(v):
        xy = v.reshape(-1, 2)
        x = xy[:, 0]
        y = xy[:, 1]
        z = h(x, y)
        r = np.column_stack([x, y, z])

        edge_now = np.array([
            np.linalg.norm(r[i] - r[j])
            for i, j in edges
        ], dtype=float)
        edge_res = np.sqrt(k_edge) * (edge_now - l0)

        centroid = np.mean(xy, axis=0)
        centroid_res = np.sqrt(k_centroid) * (centroid - centroid0)

        vij = xy[j_ref] - xy[i_ref]
        theta = np.arctan2(vij[1], vij[0])
        dtheta = wrap_to_pi(theta - theta0)
        orient_res = np.array([np.sqrt(k_orient) * dtheta], dtype=float)

        drift_res = np.sqrt(k_drift) * (xy - u).ravel()

        return np.concatenate([edge_res, centroid_res, orient_res, drift_res])

    x0 = u.ravel().copy()
    lower = np.full_like(x0, 0.02 * L)
    upper = np.full_like(x0, 0.98 * L)

    sol = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=8000,
    )

    xy = sol.x.reshape(-1, 2)
    z = h(xy[:, 0], xy[:, 1])
    r = np.column_stack([xy[:, 0], xy[:, 1], z])

    centroid = np.mean(xy, axis=0)
    centroid_shift = np.linalg.norm(centroid - centroid0)

    vij = xy[j_ref] - xy[i_ref]
    theta = np.arctan2(vij[1], vij[0])
    orient_error_rad = wrap_to_pi(theta - theta0)
    orient_error_deg = np.degrees(np.abs(orient_error_rad))

    deploy_resnorm = np.linalg.norm(residuals(sol.x))

    info = {
        "edges": edges,
        "l0": l0,
        "centroid_shift": float(centroid_shift),
        "orient_error_deg": float(orient_error_deg),
        "deploy_resnorm": float(deploy_resnorm),
        "success": bool(sol.success),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "orient_pair": (int(i_ref), int(j_ref)),
    }
    return r, info


def geometry_metrics(r, edges, l0):
    """Return geometry distortion indicators."""
    edge_now = np.array([np.linalg.norm(r[i] - r[j]) for i, j in edges], dtype=float)
    delta_edge = np.mean(np.abs(edge_now - l0) / np.maximum(l0, 1e-12))
    sigma_z = np.std(r[:, 2])
    return float(delta_edge), float(sigma_z)


# =========================================================
# Surface source model
# =========================================================
def surface_source_point(q, h):
    """Map 2D source parameter q=(x,y) to 3D terrain surface point."""
    return np.array([q[0], q[1], float(h(q[0], q[1]))], dtype=float)


# =========================================================
# Geodesic approximation on the terrain surface
# =========================================================
def flatten_idx(ix, iy, ny):
    return ix * ny + iy


def nearest_grid_index(q, xs, ys):
    ix = int(np.argmin(np.abs(xs - q[0])))
    iy = int(np.argmin(np.abs(ys - q[1])))
    return ix, iy


def build_surface_graph(h, L=1.0, grid_n=41, connectivity=8):
    """
    Build a grid graph on the terrain surface.

    Each grid node corresponds to a surface point:
        (x_i, y_j, h(x_i, y_j))

    Edge weights are 3D segment lengths between neighboring nodes.
    """
    xs = np.linspace(0.0, L, grid_n)
    ys = np.linspace(0.0, L, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = h(X, Y)

    rows = []
    cols = []
    data = []

    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        nbrs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    ny = grid_n

    for ix in range(grid_n):
        for iy in range(grid_n):
            u = flatten_idx(ix, iy, ny)
            p = np.array([xs[ix], ys[iy], Z[ix, iy]], dtype=float)

            for dx, dy in nbrs:
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx < grid_n and 0 <= jy < grid_n:
                    v = flatten_idx(jx, jy, ny)
                    q = np.array([xs[jx], ys[jy], Z[jx, jy]], dtype=float)
                    w = np.linalg.norm(q - p)

                    rows.append(u)
                    cols.append(v)
                    data.append(w)

    G = csr_matrix((data, (rows, cols)), shape=(grid_n * grid_n, grid_n * grid_n))

    return {
        "xs": xs,
        "ys": ys,
        "Z": Z,
        "graph": G,
        "grid_n": grid_n,
    }


def build_sensor_geodesic_model(r, h, L=1.0, grid_n=41, connectivity=8):
    """
    For each deployed sensor, compute the shortest-path distance field
    over the terrain surface grid, then build an interpolator.
    """
    surf = build_surface_graph(h, L=L, grid_n=grid_n, connectivity=connectivity)
    xs = surf["xs"]
    ys = surf["ys"]
    G = surf["graph"]
    nxy = surf["grid_n"]

    n_sensors = r.shape[0]
    distance_fields = np.empty((n_sensors, nxy, nxy), dtype=float)
    interpolators = []
    sensor_nodes = []

    sensor_xy = r[:, :2]

    for k in range(n_sensors):
        ix, iy = nearest_grid_index(sensor_xy[k], xs, ys)
        node = flatten_idx(ix, iy, nxy)
        sensor_nodes.append(node)

        dist_vec = dijkstra(G, directed=True, indices=node)
        field = dist_vec.reshape(nxy, nxy)
        distance_fields[k] = field

        interpolators.append(
            RegularGridInterpolator(
                (xs, ys),
                field,
                bounds_error=False,
                fill_value=None,
            )
        )

    return {
        "xs": xs,
        "ys": ys,
        "distance_fields": distance_fields,
        "interpolators": interpolators,
        "sensor_nodes": sensor_nodes,
        "sensor_xy": sensor_xy,
        "grid_n": nxy,
        "connectivity": connectivity,
    }


def evaluate_geodesic_distances(q, geo_model, L=1.0):
    """
    Evaluate source-to-sensor surface-path distances at a continuous q=(x,y)
    using the precomputed interpolated shortest-path fields.
    """
    qq = np.array([np.clip(q[0], 0.0, L), np.clip(q[1], 0.0, L)], dtype=float)

    d = np.array([
        geo_model["interpolators"][k](qq).item()
        for k in range(len(geo_model["interpolators"]))
    ], dtype=float)

    return d


# =========================================================
# Step 6B: Forward model
# =========================================================
def simulate_arrival_times_versionB(
    r,
    q_true,
    h,
    geo_model,
    timing_noise,
    v_eff=1.0,
    t0=0.0,
    L=1.0,
):
    """
    Version B forward model:
        ti = t0 + d_path(ri, s) / v_eff + eps_i

    where d_path is the shortest-path / geodesic approximation
    along the terrain surface.

    Returns:
        t       : absolute arrival times
        dt      : TDoA relative to the earliest arrival node
        ref     : reference node index (earliest arrival)
        s_true  : true 3D source on the terrain surface
        d_path  : source-to-sensor surface path distances
    """
    s_true = surface_source_point(q_true, h)
    d_path = evaluate_geodesic_distances(q_true, geo_model, L=L)

    timing_noise = np.asarray(timing_noise, dtype=float)
    if timing_noise.shape[0] != r.shape[0]:
        raise ValueError("timing_noise must have one entry per sensor.")

    t = t0 + d_path / v_eff + timing_noise
    ref = int(np.argmin(t))
    dt = t - t[ref]

    return t, dt, ref, s_true, d_path


def estimate_surface_source_xy_versionB(
    dt_obs,
    ref,
    h,
    geo_model,
    L=1.0,
    v_eff=1.0,
    search_grid_n=15,
    n_multi_start=6,
):
    """
    Estimate a surface-constrained source in 2D parameter space q=(x,y),
    using Version B surface-path / geodesic approximation.

    Predicted TDoA:
        dt_pred_i = (d_path_i(q) - d_path_ref(q)) / v_eff
    """
    n_sensors = len(geo_model["interpolators"])
    use_idx = np.array([i for i in range(n_sensors) if i != ref], dtype=int)

    def resid(q):
        d_path = evaluate_geodesic_distances(q, geo_model, L=L)
        dt_pred = (d_path - d_path[ref]) / v_eff
        return dt_obs[use_idx] - dt_pred[use_idx]

    xs = np.linspace(0.05 * L, 0.95 * L, search_grid_n)
    ys = np.linspace(0.05 * L, 0.95 * L, search_grid_n)

    candidates = []
    for x in xs:
        for y in ys:
            q = np.array([x, y], dtype=float)
            loss = np.sum(resid(q) ** 2)
            candidates.append((loss, q))

    candidates.sort(key=lambda t: t[0])
    start_points = [q for _, q in candidates[:n_multi_start]]

    best_sol = None
    best_cost = np.inf

    for q0 in start_points:
        sol = least_squares(
            resid,
            q0,
            bounds=([0.0, 0.0], [L, L]),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=1000,
        )
        cost = np.sum(sol.fun ** 2)
        if sol.success and cost < best_cost:
            best_cost = cost
            best_sol = sol

    if best_sol is None:
        return None, False, np.inf

    return best_sol.x, True, float(np.sqrt(best_cost))


# =========================================================
# Visualization helpers
# =========================================================
def plot_terrain_examples(L=1.0):
    terrains = ["flat", "hill", "rough"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    x = np.linspace(0.0, L, 160)
    y = np.linspace(0.0, L, 160)
    X, Y = np.meshgrid(x, y, indexing="xy")

    for ax, terrain in zip(axes, terrains):
        h = make_terrain(
            terrain,
            L=L,
            seed=0,
            hill_amp=HILL_AMP,
            hill_sigma=HILL_SIGMA,
            hill_xc=HILL_XC,
            hill_yc=HILL_YC,
            rough_rms=ROUGH_RMS,
            rough_corr_len=ROUGH_CORR_LEN,
            rough_grid_n=ROUGH_GRID_N,
        )
        Z = h(X, Y)

        im = ax.imshow(Z.T, origin="lower", extent=[0, L, 0, L], aspect="equal")
        ax.set_title(f"{terrain}\nmean={np.mean(Z):.4f}, std={np.std(Z):.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Step 6B terrain models")
    plt.show()


def plot_deployment_maps(L=1.0, seed=0):
    terrains = ["flat", "hill", "rough"]
    layouts = ["square", "staggered", "boundary"]

    x = np.linspace(0.0, L, 180)
    y = np.linspace(0.0, L, 180)
    X, Y = np.meshgrid(x, y, indexing="xy")

    fig, axes = plt.subplots(len(terrains), len(layouts), figsize=(14, 12), constrained_layout=True)

    for i, terrain in enumerate(terrains):
        h = make_terrain(
            terrain,
            L=L,
            seed=seed,
            hill_amp=HILL_AMP,
            hill_sigma=HILL_SIGMA,
            hill_xc=HILL_XC,
            hill_yc=HILL_YC,
            rough_rms=ROUGH_RMS,
            rough_corr_len=ROUGH_CORR_LEN,
            rough_grid_n=ROUGH_GRID_N,
        )
        Z = h(X, Y)

        for j, layout in enumerate(layouts):
            ax = axes[i, j]
            u = make_layout(layout, L=L)
            r, info = deploy_to_terrain_stable(
                u, h, L=L,
                k_edge=K_EDGE,
                k_centroid=K_CENTROID,
                k_orient=K_ORIENT,
                k_drift=K_DRIFT,
            )
            edges = info["edges"]
            l0 = info["l0"]
            delta_edge, sigma_z = geometry_metrics(r, edges, l0)

            im = ax.imshow(Z.T, origin="lower", extent=[0, L, 0, L], aspect="equal", alpha=0.88)

            for a, b in edges:
                ax.plot(
                    [u[a, 0], u[b, 0]],
                    [u[a, 1], u[b, 1]],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    color="white",
                )

            for a, b in edges:
                ax.plot(
                    [r[a, 0], r[b, 0]],
                    [r[a, 1], r[b, 1]],
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.95,
                    color="red",
                )

            dx = r[:, 0] - u[:, 0]
            dy = r[:, 1] - u[:, 1]
            ax.quiver(
                u[:, 0], u[:, 1],
                dx, dy,
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.004,
                color="black",
                alpha=0.75,
            )

            ax.scatter(
                u[:, 0], u[:, 1],
                s=55,
                facecolors="none",
                edgecolors="white",
                linewidths=1.8,
                zorder=5,
            )
            ax.scatter(
                r[:, 0], r[:, 1],
                s=45,
                c="red",
                marker="x",
                linewidths=2.0,
                zorder=6,
            )

            ax.set_title(
                f"{terrain} | {layout}\n"
                f"res={info['deploy_resnorm']:.3e}, "
                f"δedge={delta_edge:.4f}, σz={sigma_z:.4f}\n"
                f"Δc={info['centroid_shift']:.4f}, θerr={info['orient_error_deg']:.2f}°"
            )
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.set_ylabel("height", rotation=90)

    fig.suptitle("Step 6B deployment visualization", fontsize=15)
    plt.show()


def plot_one_geodesic_field(layout="square", terrain="hill", seed=0, sensor_id=0, L=1.0):
    h = make_terrain(
        terrain,
        L=L,
        seed=seed,
        hill_amp=HILL_AMP,
        hill_sigma=HILL_SIGMA,
        hill_xc=HILL_XC,
        hill_yc=HILL_YC,
        rough_rms=ROUGH_RMS,
        rough_corr_len=ROUGH_CORR_LEN,
        rough_grid_n=ROUGH_GRID_N,
    )
    u = make_layout(layout, L=L)
    r, _ = deploy_to_terrain_stable(
        u, h, L=L,
        k_edge=K_EDGE,
        k_centroid=K_CENTROID,
        k_orient=K_ORIENT,
        k_drift=K_DRIFT,
    )
    geo_model = build_sensor_geodesic_model(
        r, h, L=L, grid_n=GEO_GRID_N, connectivity=GEO_CONNECTIVITY
    )

    field = geo_model["distance_fields"][sensor_id]
    xs = geo_model["xs"]
    ys = geo_model["ys"]

    plt.figure(figsize=(6, 5))
    plt.imshow(field.T, origin="lower", extent=[0, L, 0, L], aspect="equal")
    plt.scatter(r[:, 0], r[:, 1], c="white", s=45, marker="x")
    plt.scatter(r[sensor_id, 0], r[sensor_id, 1], c="red", s=60, marker="o")
    plt.colorbar(label="surface path distance")
    plt.title(f"Geodesic field | {terrain} | {layout} | sensor {sensor_id + 1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_step6b_summary(summary_df):
    terrains = ["flat", "hill", "rough"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for ax, terrain in zip(axes, terrains):
        sub = summary_df[summary_df["terrain"] == terrain].copy()
        sub = sub.sort_values("mean_err_3d")
        ax.bar(sub["layout"], sub["mean_err_3d"])
        ax.set_title(f"{terrain}: mean 3D localization error")
        ax.set_xlabel("layout")
        ax.set_ylabel("mean_err_3d")

    plt.show()


# =========================================================
# Main experiment runner
# =========================================================
def run_step6_versionB(
    L=1.0,
    M=150,
    n_seeds=3,
    sigma_t=0.002,
    v_eff=1.0,
    geo_grid_n=41,
    search_grid_n=15,
    n_multi_start=6,
):
    layouts = ["square", "staggered", "boundary"]
    terrains = ["flat", "hill", "rough"]
    rows = []

    n_sensors = make_layout("square", L=L).shape[0]

    for terrain in terrains:
        for seed in range(n_seeds):
            h = make_terrain(
                terrain,
                L=L,
                seed=seed,
                hill_amp=HILL_AMP,
                hill_sigma=HILL_SIGMA,
                hill_xc=HILL_XC,
                hill_yc=HILL_YC,
                rough_rms=ROUGH_RMS,
                rough_corr_len=ROUGH_CORR_LEN,
                rough_grid_n=ROUGH_GRID_N,
            )

            rng = np.random.default_rng(1000 + seed)

            # same event set shared across layouts for fair comparison
            source_params_xy = rng.uniform(0.05 * L, 0.95 * L, size=(M, 2))
            shared_timing_noise = rng.normal(0.0, sigma_t, size=(M, n_sensors))
            shared_t0 = rng.uniform(T0_MIN, T0_MAX, size=M)

            for layout in layouts:
                u = make_layout(layout, L=L)
                r, info = deploy_to_terrain_stable(
                    u, h, L=L,
                    k_edge=K_EDGE,
                    k_centroid=K_CENTROID,
                    k_orient=K_ORIENT,
                    k_drift=K_DRIFT,
                )

                geo_model = build_sensor_geodesic_model(
                    r,
                    h,
                    L=L,
                    grid_n=geo_grid_n,
                    connectivity=GEO_CONNECTIVITY,
                )

                edges = info["edges"]
                l0 = info["l0"]
                delta_edge, sigma_z = geometry_metrics(r, edges, l0)

                err_3d = []
                err_xy = []
                fit_resnorms = []
                arrival_spreads = []
                mean_path_lengths = []
                n_failed_localization = 0

                for event_id, q_true in enumerate(source_params_xy):
                    t, dt, ref, s_true, d_path = simulate_arrival_times_versionB(
                        r=r,
                        q_true=q_true,
                        h=h,
                        geo_model=geo_model,
                        timing_noise=shared_timing_noise[event_id],
                        v_eff=v_eff,
                        t0=shared_t0[event_id],
                        L=L,
                    )

                    q_hat, ok, fit_resnorm = estimate_surface_source_xy_versionB(
                        dt_obs=dt,
                        ref=ref,
                        h=h,
                        geo_model=geo_model,
                        L=L,
                        v_eff=v_eff,
                        search_grid_n=search_grid_n,
                        n_multi_start=n_multi_start,
                    )

                    arrival_spreads.append(float(np.max(t) - np.min(t)))
                    mean_path_lengths.append(float(np.mean(d_path)))

                    if not ok or q_hat is None:
                        n_failed_localization += 1
                        continue

                    s_hat = surface_source_point(q_hat, h)
                    err_3d.append(np.linalg.norm(s_hat - s_true))
                    err_xy.append(np.linalg.norm(q_hat - q_true))
                    fit_resnorms.append(fit_resnorm)

                n_valid = len(err_3d)
                valid_rate = n_valid / M

                rows.append({
                    "terrain": terrain,
                    "layout": layout,
                    "seed": seed,
                    "deploy_ok": bool(info["success"]),
                    "deploy_resnorm": float(info["deploy_resnorm"]),
                    "centroid_shift": float(info["centroid_shift"]),
                    "orient_error_deg": float(info["orient_error_deg"]),
                    "delta_edge": float(delta_edge),
                    "sigma_z": float(sigma_z),
                    "geo_grid_n": int(geo_grid_n),
                    "valid_rate": float(valid_rate),   # still solver-valid only; no detection threshold yet
                    "failed_localization": int(n_failed_localization),
                    "mean_arrival_spread": float(np.mean(arrival_spreads)),
                    "mean_path_length": float(np.mean(mean_path_lengths)),
                    "mean_fit_resnorm": float(np.mean(fit_resnorms)) if fit_resnorms else np.nan,
                    "mean_err_3d": float(np.mean(err_3d)) if err_3d else np.nan,
                    "median_err_3d": float(np.median(err_3d)) if err_3d else np.nan,
                    "p95_err_3d": float(np.percentile(err_3d, 95)) if err_3d else np.nan,
                    "mean_err_xy": float(np.mean(err_xy)) if err_xy else np.nan,
                })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["terrain", "layout"], as_index=False)
        .agg(
            deploy_ok_rate=("deploy_ok", "mean"),
            deploy_resnorm=("deploy_resnorm", "mean"),
            centroid_shift=("centroid_shift", "mean"),
            orient_error_deg=("orient_error_deg", "mean"),
            delta_edge=("delta_edge", "mean"),
            sigma_z=("sigma_z", "mean"),
            geo_grid_n=("geo_grid_n", "mean"),
            valid_rate=("valid_rate", "mean"),
            mean_arrival_spread=("mean_arrival_spread", "mean"),
            mean_path_length=("mean_path_length", "mean"),
            mean_fit_resnorm=("mean_fit_resnorm", "mean"),
            mean_err_3d=("mean_err_3d", "mean"),
            median_err_3d=("median_err_3d", "mean"),
            p95_err_3d=("p95_err_3d", "mean"),
            mean_err_xy=("mean_err_xy", "mean"),
        )
        .sort_values(["terrain", "mean_err_3d"])
        .reset_index(drop=True)
    )

    return df, summary


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    raw_df, summary_df = run_step6_versionB(
        L=L,
        M=N_EVENTS,
        n_seeds=N_SEEDS,
        sigma_t=SIGMA_T,
        v_eff=V_EFF,
        geo_grid_n=GEO_GRID_N,
        search_grid_n=SEARCH_GRID_N,
        n_multi_start=N_MULTI_START,
    )

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)

    print("\n=== Step 6 summary (Version B: arrival-time only, surface path / geodesic approximation) ===")
    print(summary_df.to_string(index=False))

    if SAVE_CSV:
        raw_df.to_csv("step6_versionB_raw_results.csv", index=False)
        summary_df.to_csv("step6_versionB_summary.csv", index=False)
        print("\nSaved: step6_versionB_raw_results.csv, step6_versionB_summary.csv")

    if PLOT_TERRAINS:
        plot_terrain_examples(L=L)

    if PLOT_DEPLOYMENTS:
        plot_deployment_maps(L=L, seed=PLOT_SEED)

    if PLOT_GEODESIC_FIELD:
        plot_one_geodesic_field(
            layout="square",
            terrain="hill",
            seed=PLOT_SEED,
            sensor_id=0,
            L=L,
        )

    if PLOT_SUMMARY:
        plot_step6b_summary(summary_df)