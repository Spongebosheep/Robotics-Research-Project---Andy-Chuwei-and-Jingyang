import sys
import time

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
# Step 9 script
# Freeze the Monte Carlo design with shared event banks for fair layout comparison
# =========================

# -------------------------
# Global parameters
# -------------------------
L = 1.0
V_EFF = 1.0
SIGMA_T = 0.002

N_EVENTS = 2000
N_SEEDS = 10

SAVE_CSV = False
PLOT_TERRAINS = False

# -------------------------
# Step 9 fixed Monte Carlo design
# -------------------------
MONTE_CARLO_BASE_SEED = 20260317
ROUGH_TERRAIN_BASE_SEED = 880000
SHARE_EVENT_BANK_ACROSS_TERRAINS = True

PLOT_DEPLOYMENTS = False
PLOT_SUMMARY = True
PLOT_SEED = 0

# -------------------------
# Step 9 reference / weighting
# -------------------------
REF_STRATEGY = "max_snr"        # "earliest_time", "max_amplitude", "max_snr"
WEIGHT_STRATEGY = "snr"         # "uniform", "amplitude", "snr"
MIN_DETECT = 4

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
# Geodesic settings
# -------------------------
GEO_GRID_N = 41
GEO_CONNECTIVITY = 8
SEARCH_GRID_N = 15
N_MULTI_START = 6

# -------------------------
# Step 9 amplitude / detection settings
# A_i = A0 / (d_i + d0)^p * exp(-alpha d_i) * eta_i
# detect if SNR_i = A_i / A_noise > gamma
# -------------------------
A0 = 0.40
D0 = 0.05
SPREAD_P = 1.35
ALPHA_ATTEN = 2.00
A_NOISE = 0.20
SNR_THRESHOLD = 2.00

LOCAL_COUPLING_MODE = "none"   # "none", "lognormal"
ETA_LOG_STD = 0.15              # used only when LOCAL_COUPLING_MODE == "lognormal"

# -------------------------
# Runtime / optimization settings
# -------------------------
SHOW_PROGRESS = True
EVENT_PROGRESS_EVERY = 50        # update every N events within each combo
PROGRESS_BAR_WIDTH = 28
LEAST_SQUARES_MAX_NFEV = 250
LEAST_SQUARES_TOL = 1e-8



# =========================================================
# Layouts
# =========================================================
def normalize_layout(pts, L=1.0, target_center=(0.5, 0.5), target_span=0.72):
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
            return A_h * np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma_h ** 2)))
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

        interp = RegularGridInterpolator((xs, ys), Z, bounds_error=False, fill_value=None)

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
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def select_orientation_pair(u):
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

        edge_now = np.array([np.linalg.norm(r[i] - r[j]) for i, j in edges], dtype=float)
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
    edge_now = np.array([np.linalg.norm(r[i] - r[j]) for i, j in edges], dtype=float)
    delta_edge = np.mean(np.abs(edge_now - l0) / np.maximum(l0, 1e-12))
    sigma_z = np.std(r[:, 2])
    return float(delta_edge), float(sigma_z)


# =========================================================
# Surface source model
# =========================================================
def surface_source_point(q, h):
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
    return {"xs": xs, "ys": ys, "Z": Z, "graph": G, "grid_n": grid_n}


def build_sensor_geodesic_model(r, h, L=1.0, grid_n=41, connectivity=8):
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

    return {
        "xs": xs,
        "ys": ys,
        "distance_fields": distance_fields,
        "sensor_nodes": sensor_nodes,
        "sensor_xy": sensor_xy,
        "grid_n": nxy,
        "connectivity": connectivity,
    }


def _bilinear_interp_stacked(fields, xs, ys, qx, qy):
    """Fast bilinear interpolation of stacked distance fields at a single point.
    fields: (n_sensors, nxy, nxy), xs/ys: 1D uniform grids.
    Returns array of shape (n_sensors,).
    """
    nxy = len(xs)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    fx = (qx - xs[0]) / dx
    fy = (qy - ys[0]) / dy
    ix = int(min(max(fx, 0.0), nxy - 2))
    iy = int(min(max(fy, 0.0), nxy - 2))
    tx = fx - ix
    ty = fy - iy
    return ((1 - tx) * (1 - ty) * fields[:, ix, iy]
            + tx * (1 - ty) * fields[:, ix + 1, iy]
            + (1 - tx) * ty * fields[:, ix, iy + 1]
            + tx * ty * fields[:, ix + 1, iy + 1])


def _bilinear_interp_stacked_batch(fields, xs, ys, qs):
    """Vectorized bilinear interpolation at N query points.
    fields: (n_sensors, nxy, nxy), qs: (N, 2).
    Returns array of shape (n_sensors, N).
    """
    nxy = len(xs)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    qx = np.clip(qs[:, 0], xs[0], xs[-1])
    qy = np.clip(qs[:, 1], ys[0], ys[-1])
    fx = (qx - xs[0]) / dx
    fy = (qy - ys[0]) / dy
    ix = np.clip(fx.astype(int), 0, nxy - 2)
    iy = np.clip(fy.astype(int), 0, nxy - 2)
    tx = fx - ix
    ty = fy - iy
    return ((1 - tx) * (1 - ty) * fields[:, ix, iy]
            + tx * (1 - ty) * fields[:, ix + 1, iy]
            + (1 - tx) * ty * fields[:, ix, iy + 1]
            + tx * ty * fields[:, ix + 1, iy + 1])


def evaluate_geodesic_distances(q, geo_model, L=1.0):
    xs = geo_model["xs"]
    ys = geo_model["ys"]
    fields = geo_model["distance_fields"]
    qx = float(np.clip(q[0], 0.0, L))
    qy = float(np.clip(q[1], 0.0, L))
    return _bilinear_interp_stacked(fields, xs, ys, qx, qy)


# =========================================================
# Step 9 forward model
# =========================================================
def sample_local_coupling(n_sensors, rng, mode="none", log_std=0.15):
    if mode == "none":
        return np.ones(n_sensors, dtype=float)
    if mode == "lognormal":
        eta = rng.lognormal(mean=0.0, sigma=log_std, size=n_sensors)
        return eta / np.mean(eta)
    raise ValueError(f"Unknown LOCAL_COUPLING_MODE: {mode}")



def build_shared_event_bank(
    M,
    n_sensors,
    seed,
    L=1.0,
    sigma_t=0.002,
    t0_min=0.0,
    t0_max=0.2,
    A0=0.40,
    source_strength_jitter=None,
    local_coupling_mode="none",
    eta_log_std=0.15,
):
    rng = np.random.default_rng(seed)

    q_true_bank = rng.uniform(0.05 * L, 0.95 * L, size=(M, 2))
    timing_noise_stdnormal_bank = rng.normal(0.0, 1.0, size=(M, n_sensors))
    timing_noise_bank = sigma_t * timing_noise_stdnormal_bank
    t0_bank = rng.uniform(t0_min, t0_max, size=M)

    if source_strength_jitter is None:
        A0_bank = np.full(M, A0, dtype=float)
    else:
        A0_bank = A0 * np.asarray(source_strength_jitter(rng, M), dtype=float)
    if local_coupling_mode == "none":
        eta_bank = np.ones((M, n_sensors), dtype=float)
    else:
        eta_bank = np.vstack([
            sample_local_coupling(
                n_sensors=n_sensors,
                rng=np.random.default_rng(seed + 100000 + k),
                mode=local_coupling_mode,
                log_std=eta_log_std,
            )
            for k in range(M)
        ])

    return {
        "q_true_bank": q_true_bank,
        "timing_noise_stdnormal_bank": timing_noise_stdnormal_bank,
        "timing_noise_bank": timing_noise_bank,
        "t0_bank": t0_bank,
        "A0_bank": A0_bank,
        "eta_bank": eta_bank,
        "event_bank_seed": int(seed),
        "M": int(M),
        "n_sensors": int(n_sensors),
    }



def precompute_layout_event_bank(
    geo_model,
    h,
    q_true_bank,
    timing_noise_bank,
    t0_bank,
    A0_bank,
    eta_bank,
    v_eff=1.0,
    d0=0.05,
    spread_p=1.35,
    alpha=2.0,
    A_noise=0.20,
    snr_threshold=2.0,
):
    q_true_bank = np.asarray(q_true_bank, dtype=float)
    timing_noise_bank = np.asarray(timing_noise_bank, dtype=float)
    t0_bank = np.asarray(t0_bank, dtype=float)
    A0_bank = np.asarray(A0_bank, dtype=float)
    eta_bank = np.asarray(eta_bank, dtype=float)

    all_d = _bilinear_interp_stacked_batch(
        geo_model["distance_fields"],
        geo_model["xs"],
        geo_model["ys"],
        q_true_bank,
    )  # (n_sensors, M)

    t_bank = (t0_bank[None, :] + all_d / v_eff + timing_noise_bank.T).T
    amplitude_bank = (
        (A0_bank[None, :] / np.power(all_d + d0, spread_p))
        * np.exp(-alpha * all_d)
        * eta_bank.T
    ).T
    snr_bank = amplitude_bank / A_noise
    detected_mask_bank = snr_bank > snr_threshold

    z_true = h(q_true_bank[:, 0], q_true_bank[:, 1])
    s_true_bank = np.column_stack([q_true_bank, z_true])

    return {
        "d_path_bank": all_d.T,
        "t_bank": t_bank,
        "amplitude_bank": amplitude_bank,
        "snr_bank": snr_bank,
        "detected_mask_bank": detected_mask_bank,
        "s_true_bank": s_true_bank,
    }


def _format_seconds(seconds):
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:4.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}"


class SimpleProgressBar:
    def __init__(self, total, prefix="", width=28, enabled=True, min_interval=0.15):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.width = max(int(width), 10)
        self.enabled = bool(enabled)
        self.min_interval = float(min_interval)
        self.start_time = time.perf_counter()
        self.last_print = -1e18

    def update(self, current, extra=""):
        if not self.enabled:
            return
        current = min(max(int(current), 0), self.total)
        now = time.perf_counter()
        if current < self.total and (now - self.last_print) < self.min_interval:
            return
        frac = current / self.total
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start_time
        eta = (elapsed / current) * (self.total - current) if current > 0 else 0.0
        msg = (
            f"\r{self.prefix} [{bar}] {current:>4d}/{self.total:<4d} "
            f"{100.0 * frac:5.1f}%  elapsed={_format_seconds(elapsed)}  ETA={_format_seconds(eta)}"
        )
        if extra:
            msg += f"  {extra}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        self.last_print = now
        if current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def simulate_event_step9(
    r,
    q_true,
    h,
    geo_model,
    timing_noise,
    rng,
    v_eff=1.0,
    t0=0.0,
    A0=0.40,
    d0=0.05,
    spread_p=1.35,
    alpha=2.0,
    A_noise=0.20,
    snr_threshold=2.00,
    local_coupling_mode="none",
    eta_log_std=0.15,
    eta_override=None,
    L=1.0,
):
    s_true = surface_source_point(q_true, h)
    d_path = evaluate_geodesic_distances(q_true, geo_model, L=L)

    timing_noise = np.asarray(timing_noise, dtype=float)
    if timing_noise.shape[0] != r.shape[0]:
        raise ValueError("timing_noise must have one entry per sensor.")

    t = t0 + d_path / v_eff + timing_noise

    if eta_override is None:
        if rng is None:
            raise ValueError("rng must be provided when eta_override is None.")
        eta = sample_local_coupling(
            n_sensors=r.shape[0],
            rng=rng,
            mode=local_coupling_mode,
            log_std=eta_log_std,
        )
    else:
        eta = np.asarray(eta_override, dtype=float)
        if eta.shape != (r.shape[0],):
            raise ValueError("eta_override must have shape (n_sensors,).")
    amplitude = A0 / np.power(d_path + d0, spread_p) * np.exp(-alpha * d_path) * eta
    snr = amplitude / A_noise
    detected_mask = snr > snr_threshold

    return {
        "t": t,
        "s_true": s_true,
        "d_path": d_path,
        "amplitude": amplitude,
        "snr": snr,
        "detected_mask": detected_mask,
        "eta": eta,
    }


# =========================================================
# Dynamic reference-node selection on detecting set D
# =========================================================
def make_detected_mask(n_sensors, detected_mask=None):
    if detected_mask is None:
        return np.ones(n_sensors, dtype=bool)
    detected_mask = np.asarray(detected_mask, dtype=bool)
    if detected_mask.shape != (n_sensors,):
        raise ValueError("detected_mask must have shape (n_sensors,).")
    return detected_mask


def choose_reference_node(t, detected_idx, strategy="max_snr", amplitude=None, snr=None):
    detected_idx = np.asarray(detected_idx, dtype=int)
    if detected_idx.size == 0:
        raise ValueError("Cannot choose a reference node from an empty detecting set.")

    if strategy == "earliest_time":
        return int(detected_idx[np.argmin(t[detected_idx])])
    if strategy == "max_amplitude":
        if amplitude is None:
            raise ValueError("amplitude is required when strategy='max_amplitude'.")
        amplitude = np.asarray(amplitude, dtype=float)
        return int(detected_idx[np.argmax(amplitude[detected_idx])])
    if strategy == "max_snr":
        if snr is None:
            raise ValueError("snr is required when strategy='max_snr'.")
        snr = np.asarray(snr, dtype=float)
        return int(detected_idx[np.argmax(snr[detected_idx])])

    raise ValueError(f"Unknown reference strategy: {strategy}")


def make_tdoa_weights(use_idx, strategy="snr", amplitude=None, snr=None):
    use_idx = np.asarray(use_idx, dtype=int)
    if use_idx.size == 0:
        return np.array([], dtype=float)

    if strategy == "uniform":
        return np.ones(use_idx.size, dtype=float)
    if strategy == "amplitude":
        if amplitude is None:
            raise ValueError("amplitude is required when weight_strategy='amplitude'.")
        raw = np.asarray(amplitude, dtype=float)[use_idx]
        raw = np.maximum(raw, 1e-12)
        return raw / np.mean(raw)
    if strategy == "snr":
        if snr is None:
            raise ValueError("snr is required when weight_strategy='snr'.")
        raw = np.asarray(snr, dtype=float)[use_idx]
        raw = np.maximum(raw, 1e-12)
        return raw / np.mean(raw)

    raise ValueError(f"Unknown weight strategy: {strategy}")


def build_tdoa_observation(
    t,
    detected_mask=None,
    ref_strategy="max_snr",
    weight_strategy="snr",
    amplitude=None,
    snr=None,
    min_detect=4,
):
    n_sensors = len(t)
    detected_mask = make_detected_mask(n_sensors, detected_mask=detected_mask)
    detected_idx = np.flatnonzero(detected_mask)
    n_detect = detected_idx.size

    if n_detect < min_detect:
        return None, False

    ref = choose_reference_node(
        t=t,
        detected_idx=detected_idx,
        strategy=ref_strategy,
        amplitude=amplitude,
        snr=snr,
    )

    use_idx = detected_idx[detected_idx != ref]
    dt_obs = t[use_idx] - t[ref]
    weights = make_tdoa_weights(
        use_idx,
        strategy=weight_strategy,
        amplitude=amplitude,
        snr=snr,
    )

    obs = {
        "detected_idx": detected_idx,
        "n_detect": int(n_detect),
        "ref": int(ref),
        "use_idx": use_idx,
        "dt_obs": dt_obs,
        "weights": weights,
    }
    return obs, True


def estimate_surface_source_xy_step9(
    obs,
    geo_model,
    L=1.0,
    v_eff=1.0,
    search_grid_n=15,
    n_multi_start=6,
    lsq_max_nfev=250,
    lsq_tol=1e-8,
):
    ref = int(obs["ref"])
    use_idx = np.asarray(obs["use_idx"], dtype=int)
    dt_obs = np.asarray(obs["dt_obs"], dtype=float)
    weights = np.asarray(obs["weights"], dtype=float)

    if use_idx.size == 0:
        return None, False, np.inf

    sqrt_w = np.sqrt(np.maximum(weights, 1e-12))

    def resid(q):
        d_path = evaluate_geodesic_distances(q, geo_model, L=L)
        dt_pred = (d_path[use_idx] - d_path[ref]) / v_eff
        return sqrt_w * (dt_obs - dt_pred)

    xs_grid = np.linspace(0.05 * L, 0.95 * L, search_grid_n)
    ys_grid = np.linspace(0.05 * L, 0.95 * L, search_grid_n)
    XX, YY = np.meshgrid(xs_grid, ys_grid, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (n_grid, 2)

    # Batch-evaluate all grid points at once
    all_d = _bilinear_interp_stacked_batch(
        geo_model["distance_fields"], geo_model["xs"], geo_model["ys"], pts
    )  # (n_sensors, n_grid)
    dt_pred_grid = (all_d[use_idx, :] - all_d[ref, :]) / v_eff  # (n_use, n_grid)
    residuals_grid = sqrt_w[:, None] * (dt_obs[:, None] - dt_pred_grid)  # (n_use, n_grid)
    losses = np.sum(residuals_grid ** 2, axis=0)  # (n_grid,)
    n_starts = max(1, min(int(n_multi_start), pts.shape[0]))
    top_idx = np.argpartition(losses, n_starts - 1)[:n_starts]
    top_idx = top_idx[np.argsort(losses[top_idx])]
    start_points = pts[top_idx]

    best_sol = None
    best_cost = np.inf
    for q0 in start_points:
        sol = least_squares(
            resid,
            q0,
            bounds=([0.0, 0.0], [L, L]),
            xtol=lsq_tol,
            ftol=lsq_tol,
            gtol=lsq_tol,
            max_nfev=lsq_max_nfev,
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

    plt.suptitle("Step 9 terrain models")
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
                ax.plot([u[a, 0], u[b, 0]], [u[a, 1], u[b, 1]], linestyle="--", linewidth=1.0, alpha=0.65, color="white")
            for a, b in edges:
                ax.plot([r[a, 0], r[b, 0]], [r[a, 1], r[b, 1]], linestyle="-", linewidth=1.2, alpha=0.95, color="red")

            dx = r[:, 0] - u[:, 0]
            dy = r[:, 1] - u[:, 1]
            ax.quiver(u[:, 0], u[:, 1], dx, dy, angles="xy", scale_units="xy", scale=1.0, width=0.004, color="black", alpha=0.75)
            ax.scatter(u[:, 0], u[:, 1], s=55, facecolors="none", edgecolors="white", linewidths=1.8, zorder=5)
            ax.scatter(r[:, 0], r[:, 1], s=45, c="red", marker="x", linewidths=2.0, zorder=6)

            ax.set_title(
                f"{terrain} | {layout}\n"
                f"res={info['deploy_resnorm']:.3e}, δedge={delta_edge:.4f}, σz={sigma_z:.4f}\n"
                f"Δc={info['centroid_shift']:.4f}, θerr={info['orient_error_deg']:.2f}°"
            )
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.set_ylabel("height", rotation=90)

    fig.suptitle("Step 9 deployment visualization", fontsize=15)
    plt.show()



def plot_step9_summary(summary_df):
    terrains = ["flat", "hill", "rough"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, terrain in zip(axes, terrains):
        sub = summary_df[summary_df["terrain"] == terrain].copy().sort_values("mean_err_3d_mean")
        x = np.arange(len(sub))
        labels = sub["layout"].tolist()

        ax.bar(x, sub["mean_err_3d_mean"], yerr=sub["mean_err_3d_std"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{terrain}: mean 3D localization error")
        ax.set_xlabel("layout")
        ax.set_ylabel("mean_err_3d_mean ± std over seeds")

    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, terrain in zip(axes, terrains):
        sub = summary_df[summary_df["terrain"] == terrain].copy().sort_values("valid_localization_rate_mean", ascending=False)
        x = np.arange(len(sub))
        labels = sub["layout"].tolist()

        ax.bar(x, sub["valid_localization_rate_mean"], yerr=sub["valid_localization_rate_std"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{terrain}: valid localization rate")
        ax.set_xlabel("layout")
        ax.set_ylabel("valid_localization_rate_mean ± std")

    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, terrain in zip(axes, terrains):
        sub = summary_df[summary_df["terrain"] == terrain].copy().sort_values("mean_detected_count_all_mean", ascending=False)
        x = np.arange(len(sub))
        labels = sub["layout"].tolist()

        ax.bar(x, sub["mean_detected_count_all_mean"], yerr=sub["mean_detected_count_all_std"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{terrain}: mean detected count")
        ax.set_xlabel("layout")
        ax.set_ylabel("mean_detected_count_all_mean ± std")

    plt.show()


# =========================================================
# Main experiment runner
# =========================================================

def run_step9_monte_carlo(
    L=1.0,
    M=2000,
    n_seeds=10,
    sigma_t=0.002,
    v_eff=1.0,
    geo_grid_n=41,
    search_grid_n=15,
    n_multi_start=6,
    ref_strategy="max_snr",
    weight_strategy="snr",
    min_detect=4,
    A0=0.40,
    d0=0.05,
    spread_p=1.35,
    alpha=2.0,
    A_noise=0.20,
    snr_threshold=2.00,
    local_coupling_mode="none",
    eta_log_std=0.15,
    monte_carlo_base_seed=20260317,
    rough_terrain_base_seed=880000,
    share_event_bank_across_terrains=True,
    show_progress=True,
    event_progress_every=50,
    progress_bar_width=28,
    lsq_max_nfev=250,
    lsq_tol=1e-8,
):
    layouts = ["square", "staggered", "boundary"]
    terrains = ["flat", "hill", "rough"]
    rows = []

    n_sensors = make_layout("square", L=L).shape[0]

    shared_event_banks = {}
    for seed in range(n_seeds):
        event_seed = monte_carlo_base_seed + seed
        shared_event_banks[seed] = build_shared_event_bank(
            M=M,
            n_sensors=n_sensors,
            seed=event_seed,
            L=L,
            sigma_t=sigma_t,
            t0_min=T0_MIN,
            t0_max=T0_MAX,
            A0=A0,
            source_strength_jitter=None,
            local_coupling_mode=local_coupling_mode,
            eta_log_std=eta_log_std,
        )

    for seed in range(n_seeds):
        if share_event_bank_across_terrains:
            event_bank_by_terrain = {terrain: shared_event_banks[seed] for terrain in terrains}
        else:
            event_bank_by_terrain = {}
            for terrain_idx, terrain in enumerate(terrains):
                event_bank_by_terrain[terrain] = build_shared_event_bank(
                    M=M,
                    n_sensors=n_sensors,
                    seed=monte_carlo_base_seed + 1000 * terrain_idx + seed,
                    L=L,
                    sigma_t=sigma_t,
                    t0_min=T0_MIN,
                    t0_max=T0_MAX,
                    A0=A0,
                    source_strength_jitter=None,
                    local_coupling_mode=local_coupling_mode,
                    eta_log_std=eta_log_std,
                )

        terrain_models = {}
        for terrain in terrains:
            terrain_seed = rough_terrain_base_seed + seed
            terrain_models[terrain] = make_terrain(
                terrain,
                L=L,
                seed=terrain_seed,
                hill_amp=HILL_AMP,
                hill_sigma=HILL_SIGMA,
                hill_xc=HILL_XC,
                hill_yc=HILL_YC,
                rough_rms=ROUGH_RMS,
                rough_corr_len=ROUGH_CORR_LEN,
                rough_grid_n=ROUGH_GRID_N,
            )

        for terrain in terrains:
            h = terrain_models[terrain]
            event_bank = event_bank_by_terrain[terrain]

            source_params_xy = event_bank["q_true_bank"]
            shared_timing_noise = event_bank["timing_noise_bank"]
            shared_t0 = event_bank["t0_bank"]
            shared_A0 = event_bank["A0_bank"]
            shared_eta = event_bank["eta_bank"]

            n_combos = len(layouts)
            for layout_idx, layout in enumerate(layouts):
                combo_label = f"seed={seed} terrain={terrain} layout={layout} ({layout_idx + 1}/{n_combos})"
                print(f"\n[{combo_label}]", flush=True)

                combo_tic = time.perf_counter()
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

                pre = precompute_layout_event_bank(
                    geo_model=geo_model,
                    h=h,
                    q_true_bank=source_params_xy,
                    timing_noise_bank=shared_timing_noise,
                    t0_bank=shared_t0,
                    A0_bank=shared_A0,
                    eta_bank=shared_eta,
                    v_eff=v_eff,
                    d0=d0,
                    spread_p=spread_p,
                    alpha=alpha,
                    A_noise=A_noise,
                    snr_threshold=snr_threshold,
                )

                t_bank = pre["t_bank"]
                s_true_bank = pre["s_true_bank"]
                d_path_bank = pre["d_path_bank"]
                amplitude_bank = pre["amplitude_bank"]
                snr_bank = pre["snr_bank"]
                detected_mask_bank = pre["detected_mask_bank"]

                arrival_spreads = (t_bank.max(axis=1) - t_bank.min(axis=1)).astype(float)
                mean_path_lengths = d_path_bank.mean(axis=1).astype(float)
                detected_counts_all = detected_mask_bank.sum(axis=1).astype(int)
                mean_snr_all = snr_bank.mean(axis=1).astype(float)
                mean_amp_all = amplitude_bank.mean(axis=1).astype(float)

                err_3d = []
                err_xy = []
                fit_resnorms = []
                detected_counts_usable = []
                usable_ref_nodes = []

                n_failed_detection = 0
                n_failed_localization = 0

                progress = SimpleProgressBar(
                    total=M,
                    prefix="events",
                    width=progress_bar_width,
                    enabled=show_progress,
                )

                for event_id, q_true in enumerate(source_params_xy):
                    t = t_bank[event_id]
                    s_true = s_true_bank[event_id]
                    d_path = d_path_bank[event_id]
                    amplitude = amplitude_bank[event_id]
                    snr = snr_bank[event_id]
                    detected_mask = detected_mask_bank[event_id]

                    obs, usable = build_tdoa_observation(
                        t=t,
                        detected_mask=detected_mask,
                        ref_strategy=ref_strategy,
                        weight_strategy=weight_strategy,
                        amplitude=amplitude,
                        snr=snr,
                        min_detect=min_detect,
                    )

                    if not usable or obs is None:
                        n_failed_detection += 1
                    else:
                        detected_counts_usable.append(obs["n_detect"])
                        usable_ref_nodes.append(obs["ref"])

                        q_hat, ok, fit_resnorm = estimate_surface_source_xy_step9(
                            obs=obs,
                            geo_model=geo_model,
                            L=L,
                            v_eff=v_eff,
                            search_grid_n=search_grid_n,
                            n_multi_start=n_multi_start,
                            lsq_max_nfev=lsq_max_nfev,
                            lsq_tol=lsq_tol,
                        )

                        if not ok or q_hat is None:
                            n_failed_localization += 1
                        else:
                            s_hat = surface_source_point(q_hat, h)
                            err_3d.append(np.linalg.norm(s_hat - s_true))
                            err_xy.append(np.linalg.norm(q_hat - q_true))
                            fit_resnorms.append(fit_resnorm)

                    if show_progress and (((event_id + 1) % max(1, event_progress_every) == 0) or (event_id + 1 == M)):
                        progress.update(
                            event_id + 1,
                            extra=(
                                f"valid={len(err_3d)} det_fail={n_failed_detection} "
                                f"loc_fail={n_failed_localization}"
                            ),
                        )

                n_usable = M - n_failed_detection
                n_solver_valid = len(err_3d)
                combo_runtime_sec = time.perf_counter() - combo_tic

                usable_event_rate = n_usable / M
                valid_localization_rate = n_solver_valid / M
                solver_success_given_usable = (n_solver_valid / n_usable) if n_usable > 0 else np.nan

                ref_counts = np.bincount(usable_ref_nodes, minlength=n_sensors) if usable_ref_nodes else np.zeros(n_sensors, dtype=int)

                rows.append({
                    "terrain": terrain,
                    "layout": layout,
                    "seed": seed,
                    "M": int(M),
                    "event_bank_seed": int(event_bank["event_bank_seed"]),
                    "rough_terrain_seed": int(rough_terrain_base_seed + seed),
                    "share_event_bank_across_terrains": bool(share_event_bank_across_terrains),
                    "ref_strategy": ref_strategy,
                    "weight_strategy": weight_strategy,
                    "deploy_ok": bool(info["success"]),
                    "deploy_resnorm": float(info["deploy_resnorm"]),
                    "centroid_shift": float(info["centroid_shift"]),
                    "orient_error_deg": float(info["orient_error_deg"]),
                    "delta_edge": float(delta_edge),
                    "sigma_z": float(sigma_z),
                    "geo_grid_n": int(geo_grid_n),
                    "A0": float(A0),
                    "d0": float(d0),
                    "spread_p": float(spread_p),
                    "alpha": float(alpha),
                    "A_noise": float(A_noise),
                    "snr_threshold": float(snr_threshold),
                    "usable_event_rate": float(usable_event_rate),
                    "valid_localization_rate": float(valid_localization_rate),
                    "solver_success_given_usable": float(solver_success_given_usable) if np.isfinite(solver_success_given_usable) else np.nan,
                    "failed_detection": int(n_failed_detection),
                    "failed_localization": int(n_failed_localization),
                    "mean_detected_count_all": float(np.mean(detected_counts_all)) if detected_counts_all.size > 0 else np.nan,
                    "mean_detected_count_usable": float(np.mean(detected_counts_usable)) if len(detected_counts_usable) > 0 else np.nan,
                    "mean_snr": float(np.mean(mean_snr_all)) if mean_snr_all.size > 0 else np.nan,
                    "mean_amplitude": float(np.mean(mean_amp_all)) if mean_amp_all.size > 0 else np.nan,
                    "mean_arrival_spread": float(np.mean(arrival_spreads)) if arrival_spreads.size > 0 else np.nan,
                    "mean_path_length": float(np.mean(mean_path_lengths)) if mean_path_lengths.size > 0 else np.nan,
                    "mean_fit_resnorm": float(np.mean(fit_resnorms)) if len(fit_resnorms) > 0 else np.nan,
                    "mean_err_3d": float(np.mean(err_3d)) if len(err_3d) > 0 else np.nan,
                    "median_err_3d": float(np.median(err_3d)) if len(err_3d) > 0 else np.nan,
                    "p95_err_3d": float(np.percentile(err_3d, 95)) if len(err_3d) > 0 else np.nan,
                    "mean_err_xy": float(np.mean(err_xy)) if len(err_xy) > 0 else np.nan,
                    "combo_runtime_sec": float(combo_runtime_sec),
                    **{f"ref_count_node_{k + 1}": int(ref_counts[k]) for k in range(n_sensors)},
                })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["terrain", "layout", "ref_strategy", "weight_strategy"], as_index=False)
        .agg(
            n_seed=("seed", "count"),
            M_per_seed=("M", "first"),
            deploy_ok_rate=("deploy_ok", "mean"),
            deploy_resnorm_mean=("deploy_resnorm", "mean"),
            deploy_resnorm_std=("deploy_resnorm", "std"),
            centroid_shift_mean=("centroid_shift", "mean"),
            centroid_shift_std=("centroid_shift", "std"),
            orient_error_deg_mean=("orient_error_deg", "mean"),
            orient_error_deg_std=("orient_error_deg", "std"),
            delta_edge_mean=("delta_edge", "mean"),
            delta_edge_std=("delta_edge", "std"),
            sigma_z_mean=("sigma_z", "mean"),
            sigma_z_std=("sigma_z", "std"),
            geo_grid_n=("geo_grid_n", "first"),
            usable_event_rate_mean=("usable_event_rate", "mean"),
            usable_event_rate_std=("usable_event_rate", "std"),
            valid_localization_rate_mean=("valid_localization_rate", "mean"),
            valid_localization_rate_std=("valid_localization_rate", "std"),
            solver_success_given_usable_mean=("solver_success_given_usable", "mean"),
            solver_success_given_usable_std=("solver_success_given_usable", "std"),
            mean_detected_count_all_mean=("mean_detected_count_all", "mean"),
            mean_detected_count_all_std=("mean_detected_count_all", "std"),
            mean_detected_count_usable_mean=("mean_detected_count_usable", "mean"),
            mean_detected_count_usable_std=("mean_detected_count_usable", "std"),
            mean_snr_mean=("mean_snr", "mean"),
            mean_snr_std=("mean_snr", "std"),
            mean_amplitude_mean=("mean_amplitude", "mean"),
            mean_amplitude_std=("mean_amplitude", "std"),
            mean_arrival_spread_mean=("mean_arrival_spread", "mean"),
            mean_arrival_spread_std=("mean_arrival_spread", "std"),
            mean_path_length_mean=("mean_path_length", "mean"),
            mean_path_length_std=("mean_path_length", "std"),
            mean_fit_resnorm_mean=("mean_fit_resnorm", "mean"),
            mean_fit_resnorm_std=("mean_fit_resnorm", "std"),
            mean_err_3d_mean=("mean_err_3d", "mean"),
            mean_err_3d_std=("mean_err_3d", "std"),
            median_err_3d_mean=("median_err_3d", "mean"),
            median_err_3d_std=("median_err_3d", "std"),
            p95_err_3d_mean=("p95_err_3d", "mean"),
            p95_err_3d_std=("p95_err_3d", "std"),
            mean_err_xy_mean=("mean_err_xy", "mean"),
            mean_err_xy_std=("mean_err_xy", "std"),
            combo_runtime_sec_mean=("combo_runtime_sec", "mean"),
            combo_runtime_sec_std=("combo_runtime_sec", "std"),
        )
        .sort_values(["terrain", "mean_err_3d_mean"])
        .reset_index(drop=True)
    )

    summary["mean_err_3d_sem"] = summary["mean_err_3d_std"] / np.sqrt(np.maximum(summary["n_seed"], 1))
    summary["valid_localization_rate_sem"] = summary["valid_localization_rate_std"] / np.sqrt(np.maximum(summary["n_seed"], 1))
    summary["mean_detected_count_all_sem"] = summary["mean_detected_count_all_std"] / np.sqrt(np.maximum(summary["n_seed"], 1))
    summary["rank_mean_err_within_terrain"] = summary.groupby("terrain")["mean_err_3d_mean"].rank(method="dense")
    summary["rank_valid_rate_within_terrain"] = summary.groupby("terrain")["valid_localization_rate_mean"].rank(method="dense", ascending=False)

    fairness_table = (
        df.groupby(["seed", "terrain"], as_index=False)
        .agg(
            n_layouts=("layout", "nunique"),
            n_unique_event_banks=("event_bank_seed", "nunique"),
            M_per_seed=("M", "first"),
        )
        .sort_values(["seed", "terrain"])
        .reset_index(drop=True)
    )

    return df, summary, fairness_table



# =========================================================
# Step 10 metric post-processing
# =========================================================
def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if (not np.isfinite(sd)) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def add_step10_metrics(raw_df, w_edge=0.5, w_height=0.5):
    df = raw_df.copy()
    df["edge_distortion"] = df["delta_edge"].astype(float)
    df["height_spread"] = df["sigma_z"].astype(float)

    z_edge = safe_zscore(df["edge_distortion"].to_numpy())
    z_height = safe_zscore(df["height_spread"].to_numpy())
    df["dgeom_star"] = w_edge * z_edge + w_height * z_height
    return df


def build_step10_summary(step10_raw_df):
    summary = (
        step10_raw_df.groupby(["terrain", "layout", "ref_strategy", "weight_strategy"], as_index=False)
        .agg(
            n_seed=("seed", "count"),
            M_per_seed=("M", "first"),
            mean_localization_error_mean=("mean_err_3d", "mean"),
            mean_localization_error_std=("mean_err_3d", "std"),
            median_localization_error_mean=("median_err_3d", "mean"),
            median_localization_error_std=("median_err_3d", "std"),
            p95_localization_error_mean=("p95_err_3d", "mean"),
            p95_localization_error_std=("p95_err_3d", "std"),
            valid_localization_rate_mean=("valid_localization_rate", "mean"),
            valid_localization_rate_std=("valid_localization_rate", "std"),
            normalized_deformation_index_mean=("dgeom_star", "mean"),
            normalized_deformation_index_std=("dgeom_star", "std"),
            edge_distortion_mean=("edge_distortion", "mean"),
            edge_distortion_std=("edge_distortion", "std"),
            height_spread_mean=("height_spread", "mean"),
            height_spread_std=("height_spread", "std"),
            deploy_ok_rate=("deploy_ok", "mean"),
            deploy_resnorm_mean=("deploy_resnorm", "mean"),
            deploy_resnorm_std=("deploy_resnorm", "std"),
            usable_event_rate_mean=("usable_event_rate", "mean"),
            usable_event_rate_std=("usable_event_rate", "std"),
            solver_success_given_usable_mean=("solver_success_given_usable", "mean"),
            solver_success_given_usable_std=("solver_success_given_usable", "std"),
            mean_detected_count_all_mean=("mean_detected_count_all", "mean"),
            mean_detected_count_all_std=("mean_detected_count_all", "std"),
            mean_fit_resnorm_mean=("mean_fit_resnorm", "mean"),
            mean_fit_resnorm_std=("mean_fit_resnorm", "std"),
            combo_runtime_sec_mean=("combo_runtime_sec", "mean"),
            combo_runtime_sec_std=("combo_runtime_sec", "std"),
        )
        .sort_values(["terrain", "mean_localization_error_mean", "valid_localization_rate_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    std_cols = [
        "mean_localization_error_std",
        "median_localization_error_std",
        "p95_localization_error_std",
        "valid_localization_rate_std",
        "normalized_deformation_index_std",
        "edge_distortion_std",
        "height_spread_std",
    ]
    for c in std_cols:
        summary[c] = summary[c].fillna(0.0)

    sqrt_n = np.sqrt(np.maximum(summary["n_seed"].to_numpy(dtype=float), 1.0))
    summary["mean_localization_error_sem"] = summary["mean_localization_error_std"] / sqrt_n
    summary["median_localization_error_sem"] = summary["median_localization_error_std"] / sqrt_n
    summary["p95_localization_error_sem"] = summary["p95_localization_error_std"] / sqrt_n
    summary["valid_localization_rate_sem"] = summary["valid_localization_rate_std"] / sqrt_n
    summary["normalized_deformation_index_sem"] = summary["normalized_deformation_index_std"] / sqrt_n
    summary["edge_distortion_sem"] = summary["edge_distortion_std"] / sqrt_n
    summary["height_spread_sem"] = summary["height_spread_std"] / sqrt_n

    summary["rank_mean_error_within_terrain"] = summary.groupby("terrain")["mean_localization_error_mean"].rank(method="dense", ascending=True)
    summary["rank_valid_rate_within_terrain"] = summary.groupby("terrain")["valid_localization_rate_mean"].rank(method="dense", ascending=False)
    summary["rank_dgeom_within_terrain"] = summary.groupby("terrain")["normalized_deformation_index_mean"].rank(method="dense", ascending=True)

    main_cols = [
        "terrain", "layout", "n_seed", "M_per_seed",
        "mean_localization_error_mean", "mean_localization_error_std",
        "median_localization_error_mean", "median_localization_error_std",
        "p95_localization_error_mean", "p95_localization_error_std",
        "valid_localization_rate_mean", "valid_localization_rate_std",
        "normalized_deformation_index_mean", "normalized_deformation_index_std",
        "edge_distortion_mean", "edge_distortion_std",
        "height_spread_mean", "height_spread_std",
        "rank_mean_error_within_terrain", "rank_valid_rate_within_terrain", "rank_dgeom_within_terrain",
    ]
    main_table = summary[main_cols].copy()
    return summary, main_table


def _plot_metric_by_terrain(summary_df, mean_col, std_col, title_prefix, ylabel, sort_ascending=True):
    terrains = ["flat", "hill", "rough"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, terrain in zip(axes, terrains):
        sub = summary_df[summary_df["terrain"] == terrain].copy()
        sub = sub.sort_values(mean_col, ascending=sort_ascending)
        x = np.arange(len(sub))
        ax.bar(x, sub[mean_col], yerr=sub[std_col])
        ax.set_xticks(x)
        ax.set_xticklabels(sub["layout"].tolist())
        ax.set_title(f"{terrain}: {title_prefix}")
        ax.set_xlabel("layout")
        ax.set_ylabel(ylabel)
    plt.show()


def plot_step10_core_metrics(summary_df):
    _plot_metric_by_terrain(
        summary_df,
        mean_col="mean_localization_error_mean",
        std_col="mean_localization_error_std",
        title_prefix="mean localization error",
        ylabel="mean ± std over seeds",
        sort_ascending=True,
    )
    _plot_metric_by_terrain(
        summary_df,
        mean_col="median_localization_error_mean",
        std_col="median_localization_error_std",
        title_prefix="median localization error",
        ylabel="median ± std over seeds",
        sort_ascending=True,
    )
    _plot_metric_by_terrain(
        summary_df,
        mean_col="p95_localization_error_mean",
        std_col="p95_localization_error_std",
        title_prefix="P95 localization error",
        ylabel="P95 ± std over seeds",
        sort_ascending=True,
    )
    _plot_metric_by_terrain(
        summary_df,
        mean_col="valid_localization_rate_mean",
        std_col="valid_localization_rate_std",
        title_prefix="valid localization rate",
        ylabel="rate ± std over seeds",
        sort_ascending=False,
    )
    _plot_metric_by_terrain(
        summary_df,
        mean_col="normalized_deformation_index_mean",
        std_col="normalized_deformation_index_std",
        title_prefix="normalized deformation index",
        ylabel="D_geom* ± std over seeds",
        sort_ascending=True,
    )


def plot_step10_geometry_components(summary_df):
    _plot_metric_by_terrain(
        summary_df,
        mean_col="edge_distortion_mean",
        std_col="edge_distortion_std",
        title_prefix="edge distortion",
        ylabel="edge distortion ± std",
        sort_ascending=True,
    )
    _plot_metric_by_terrain(
        summary_df,
        mean_col="height_spread_mean",
        std_col="height_spread_std",
        title_prefix="height spread",
        ylabel="height spread ± std",
        sort_ascending=True,
    )


# =========================================================
# Step 10 configuration
# =========================================================
PLOT_STEP10_CORE = True
PLOT_STEP10_GEOMETRY = True
SAVE_STEP10_CSV = False
DGEOM_W_EDGE = 0.50
DGEOM_W_HEIGHT = 0.50



# =========================================================
# Step 11 robustness experiment configuration
# =========================================================
TIMING_NOISE_LEVELS = [0.0, 0.001, 0.002, 0.004]
PLOT_STEP11_MEAN_ERROR = True
PLOT_STEP11_P95_ERROR = True
PLOT_STEP11_VALID_RATE = True
SAVE_STEP11_CSV = False


# =========================================================
# Step 11 robustness helpers
# =========================================================
def precompute_layout_static_event_bank(
    geo_model,
    h,
    q_true_bank,
    t0_bank,
    A0_bank,
    eta_bank,
    v_eff=1.0,
    d0=0.05,
    spread_p=1.35,
    alpha=2.0,
    A_noise=0.20,
    snr_threshold=2.0,
):
    q_true_bank = np.asarray(q_true_bank, dtype=float)
    t0_bank = np.asarray(t0_bank, dtype=float)
    A0_bank = np.asarray(A0_bank, dtype=float)
    eta_bank = np.asarray(eta_bank, dtype=float)

    all_d = _bilinear_interp_stacked_batch(
        geo_model["distance_fields"],
        geo_model["xs"],
        geo_model["ys"],
        q_true_bank,
    )  # (n_sensors, M)

    base_t_bank = (t0_bank[None, :] + all_d / v_eff).T
    amplitude_bank = (
        (A0_bank[None, :] / np.power(all_d + d0, spread_p))
        * np.exp(-alpha * all_d)
        * eta_bank.T
    ).T
    snr_bank = amplitude_bank / A_noise
    detected_mask_bank = snr_bank > snr_threshold

    z_true = h(q_true_bank[:, 0], q_true_bank[:, 1])
    s_true_bank = np.column_stack([q_true_bank, z_true])

    return {
        "d_path_bank": all_d.T,
        "base_t_bank": base_t_bank,
        "amplitude_bank": amplitude_bank,
        "snr_bank": snr_bank,
        "detected_mask_bank": detected_mask_bank,
        "s_true_bank": s_true_bank,
    }



def evaluate_precomputed_bank_at_noise_level(
    q_true_bank,
    s_true_bank,
    t_bank,
    d_path_bank,
    amplitude_bank,
    snr_bank,
    detected_mask_bank,
    h,
    geo_model,
    ref_strategy="max_snr",
    weight_strategy="snr",
    min_detect=4,
    L=1.0,
    v_eff=1.0,
    search_grid_n=15,
    n_multi_start=6,
    lsq_max_nfev=250,
    lsq_tol=1e-8,
    show_progress=True,
    event_progress_every=50,
    progress_prefix="events",
    progress_bar_width=28,
):
    M = q_true_bank.shape[0]
    arrival_spreads = (t_bank.max(axis=1) - t_bank.min(axis=1)).astype(float)
    mean_path_lengths = d_path_bank.mean(axis=1).astype(float)
    detected_counts_all = detected_mask_bank.sum(axis=1).astype(int)
    mean_snr_all = snr_bank.mean(axis=1).astype(float)
    mean_amp_all = amplitude_bank.mean(axis=1).astype(float)

    err_3d = []
    err_xy = []
    fit_resnorms = []
    detected_counts_usable = []
    usable_ref_nodes = []

    n_failed_detection = 0
    n_failed_localization = 0

    progress = SimpleProgressBar(
        total=M,
        prefix=progress_prefix,
        width=progress_bar_width,
        enabled=show_progress,
    )

    for event_id, q_true in enumerate(q_true_bank):
        t = t_bank[event_id]
        s_true = s_true_bank[event_id]
        amplitude = amplitude_bank[event_id]
        snr = snr_bank[event_id]
        detected_mask = detected_mask_bank[event_id]

        obs, usable = build_tdoa_observation(
            t=t,
            detected_mask=detected_mask,
            ref_strategy=ref_strategy,
            weight_strategy=weight_strategy,
            amplitude=amplitude,
            snr=snr,
            min_detect=min_detect,
        )

        if not usable or obs is None:
            n_failed_detection += 1
        else:
            detected_counts_usable.append(obs["n_detect"])
            usable_ref_nodes.append(obs["ref"])

            q_hat, ok, fit_resnorm = estimate_surface_source_xy_step9(
                obs=obs,
                geo_model=geo_model,
                L=L,
                v_eff=v_eff,
                search_grid_n=search_grid_n,
                n_multi_start=n_multi_start,
                lsq_max_nfev=lsq_max_nfev,
                lsq_tol=lsq_tol,
            )

            if not ok or q_hat is None:
                n_failed_localization += 1
            else:
                s_hat = surface_source_point(q_hat, h)
                err_3d.append(np.linalg.norm(s_hat - s_true))
                err_xy.append(np.linalg.norm(q_hat - q_true))
                fit_resnorms.append(fit_resnorm)

        if show_progress and (((event_id + 1) % max(1, event_progress_every) == 0) or (event_id + 1 == M)):
            progress.update(
                event_id + 1,
                extra=(
                    f"valid={len(err_3d)} det_fail={n_failed_detection} "
                    f"loc_fail={n_failed_localization}"
                ),
            )

    n_usable = M - n_failed_detection
    n_solver_valid = len(err_3d)
    usable_event_rate = n_usable / M
    valid_localization_rate = n_solver_valid / M
    solver_success_given_usable = (n_solver_valid / n_usable) if n_usable > 0 else np.nan

    n_sensors = t_bank.shape[1]
    ref_counts = np.bincount(usable_ref_nodes, minlength=n_sensors) if usable_ref_nodes else np.zeros(n_sensors, dtype=int)

    return {
        "usable_event_rate": float(usable_event_rate),
        "valid_localization_rate": float(valid_localization_rate),
        "solver_success_given_usable": float(solver_success_given_usable) if np.isfinite(solver_success_given_usable) else np.nan,
        "failed_detection": int(n_failed_detection),
        "failed_localization": int(n_failed_localization),
        "mean_detected_count_all": float(np.mean(detected_counts_all)) if detected_counts_all.size > 0 else np.nan,
        "mean_detected_count_usable": float(np.mean(detected_counts_usable)) if len(detected_counts_usable) > 0 else np.nan,
        "mean_snr": float(np.mean(mean_snr_all)) if mean_snr_all.size > 0 else np.nan,
        "mean_amplitude": float(np.mean(mean_amp_all)) if mean_amp_all.size > 0 else np.nan,
        "mean_arrival_spread": float(np.mean(arrival_spreads)) if arrival_spreads.size > 0 else np.nan,
        "mean_path_length": float(np.mean(mean_path_lengths)) if mean_path_lengths.size > 0 else np.nan,
        "mean_fit_resnorm": float(np.mean(fit_resnorms)) if len(fit_resnorms) > 0 else np.nan,
        "mean_err_3d": float(np.mean(err_3d)) if len(err_3d) > 0 else np.nan,
        "median_err_3d": float(np.median(err_3d)) if len(err_3d) > 0 else np.nan,
        "p95_err_3d": float(np.percentile(err_3d, 95)) if len(err_3d) > 0 else np.nan,
        "mean_err_xy": float(np.mean(err_xy)) if len(err_xy) > 0 else np.nan,
        **{f"ref_count_node_{k + 1}": int(ref_counts[k]) for k in range(n_sensors)},
    }



def run_step11_robustness_experiment(
    sigma_t_levels,
    L=1.0,
    M=2000,
    n_seeds=10,
    v_eff=1.0,
    geo_grid_n=41,
    search_grid_n=15,
    n_multi_start=6,
    ref_strategy="max_snr",
    weight_strategy="snr",
    min_detect=4,
    A0=0.40,
    d0=0.05,
    spread_p=1.35,
    alpha=2.0,
    A_noise=0.20,
    snr_threshold=2.00,
    local_coupling_mode="none",
    eta_log_std=0.15,
    monte_carlo_base_seed=20260317,
    rough_terrain_base_seed=880000,
    share_event_bank_across_terrains=True,
    show_progress=True,
    event_progress_every=50,
    progress_bar_width=28,
    lsq_max_nfev=250,
    lsq_tol=1e-8,
):
    sigma_t_levels = [float(x) for x in sigma_t_levels]
    layouts = ["square", "staggered", "boundary"]
    terrains = ["flat", "hill", "rough"]
    rows = []

    n_sensors = make_layout("square", L=L).shape[0]

    shared_event_banks = {}
    for seed in range(n_seeds):
        event_seed = monte_carlo_base_seed + seed
        shared_event_banks[seed] = build_shared_event_bank(
            M=M,
            n_sensors=n_sensors,
            seed=event_seed,
            L=L,
            sigma_t=1.0,
            t0_min=T0_MIN,
            t0_max=T0_MAX,
            A0=A0,
            source_strength_jitter=None,
            local_coupling_mode=local_coupling_mode,
            eta_log_std=eta_log_std,
        )

    for seed in range(n_seeds):
        if share_event_bank_across_terrains:
            event_bank_by_terrain = {terrain: shared_event_banks[seed] for terrain in terrains}
        else:
            event_bank_by_terrain = {}
            for terrain_idx, terrain in enumerate(terrains):
                event_bank_by_terrain[terrain] = build_shared_event_bank(
                    M=M,
                    n_sensors=n_sensors,
                    seed=monte_carlo_base_seed + 1000 * terrain_idx + seed,
                    L=L,
                    sigma_t=1.0,
                    t0_min=T0_MIN,
                    t0_max=T0_MAX,
                    A0=A0,
                    source_strength_jitter=None,
                    local_coupling_mode=local_coupling_mode,
                    eta_log_std=eta_log_std,
                )

        terrain_models = {}
        for terrain in terrains:
            terrain_seed = rough_terrain_base_seed + seed
            terrain_models[terrain] = make_terrain(
                terrain,
                L=L,
                seed=terrain_seed,
                hill_amp=HILL_AMP,
                hill_sigma=HILL_SIGMA,
                hill_xc=HILL_XC,
                hill_yc=HILL_YC,
                rough_rms=ROUGH_RMS,
                rough_corr_len=ROUGH_CORR_LEN,
                rough_grid_n=ROUGH_GRID_N,
            )

        for terrain in terrains:
            h = terrain_models[terrain]
            event_bank = event_bank_by_terrain[terrain]
            q_true_bank = event_bank["q_true_bank"]
            timing_noise_stdnormal_bank = event_bank["timing_noise_stdnormal_bank"]
            t0_bank = event_bank["t0_bank"]
            A0_bank = event_bank["A0_bank"]
            eta_bank = event_bank["eta_bank"]

            for layout in layouts:
                combo_label = f"seed={seed} terrain={terrain} layout={layout}"
                print(f"\n[{combo_label}]", flush=True)
                combo_tic = time.perf_counter()

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

                static_pre = precompute_layout_static_event_bank(
                    geo_model=geo_model,
                    h=h,
                    q_true_bank=q_true_bank,
                    t0_bank=t0_bank,
                    A0_bank=A0_bank,
                    eta_bank=eta_bank,
                    v_eff=v_eff,
                    d0=d0,
                    spread_p=spread_p,
                    alpha=alpha,
                    A_noise=A_noise,
                    snr_threshold=snr_threshold,
                )

                base_t_bank = static_pre["base_t_bank"]
                d_path_bank = static_pre["d_path_bank"]
                amplitude_bank = static_pre["amplitude_bank"]
                snr_bank = static_pre["snr_bank"]
                detected_mask_bank = static_pre["detected_mask_bank"]
                s_true_bank = static_pre["s_true_bank"]

                for sigma_t in sigma_t_levels:
                    sigma_t = float(sigma_t)
                    sigma_tic = time.perf_counter()
                    t_bank = base_t_bank + sigma_t * timing_noise_stdnormal_bank
                    eval_stats = evaluate_precomputed_bank_at_noise_level(
                        q_true_bank=q_true_bank,
                        s_true_bank=s_true_bank,
                        t_bank=t_bank,
                        d_path_bank=d_path_bank,
                        amplitude_bank=amplitude_bank,
                        snr_bank=snr_bank,
                        detected_mask_bank=detected_mask_bank,
                        h=h,
                        geo_model=geo_model,
                        ref_strategy=ref_strategy,
                        weight_strategy=weight_strategy,
                        min_detect=min_detect,
                        L=L,
                        v_eff=v_eff,
                        search_grid_n=search_grid_n,
                        n_multi_start=n_multi_start,
                        lsq_max_nfev=lsq_max_nfev,
                        lsq_tol=lsq_tol,
                        show_progress=show_progress,
                        event_progress_every=event_progress_every,
                        progress_prefix=f"σt={sigma_t:.4f}",
                        progress_bar_width=progress_bar_width,
                    )
                    sigma_runtime_sec = time.perf_counter() - sigma_tic

                    rows.append({
                        "terrain": terrain,
                        "layout": layout,
                        "seed": seed,
                        "M": int(M),
                        "sigma_t": sigma_t,
                        "event_bank_seed": int(event_bank["event_bank_seed"]),
                        "rough_terrain_seed": int(rough_terrain_base_seed + seed),
                        "share_event_bank_across_terrains": bool(share_event_bank_across_terrains),
                        "ref_strategy": ref_strategy,
                        "weight_strategy": weight_strategy,
                        "deploy_ok": bool(info["success"]),
                        "deploy_resnorm": float(info["deploy_resnorm"]),
                        "centroid_shift": float(info["centroid_shift"]),
                        "orient_error_deg": float(info["orient_error_deg"]),
                        "delta_edge": float(delta_edge),
                        "sigma_z": float(sigma_z),
                        "geo_grid_n": int(geo_grid_n),
                        "A0": float(A0),
                        "d0": float(d0),
                        "spread_p": float(spread_p),
                        "alpha": float(alpha),
                        "A_noise": float(A_noise),
                        "snr_threshold": float(snr_threshold),
                        "combo_runtime_sec": float(time.perf_counter() - combo_tic),
                        "sigma_runtime_sec": float(sigma_runtime_sec),
                        **eval_stats,
                    })

    raw_df = pd.DataFrame(rows)
    raw_df = add_step10_metrics(raw_df, w_edge=DGEOM_W_EDGE, w_height=DGEOM_W_HEIGHT)
    summary_df = build_step11_summary(raw_df)
    ranking_flip_df = build_step11_ranking_flip_table(summary_df)
    fairness_df = build_step11_fairness_table(raw_df)
    return raw_df, summary_df, ranking_flip_df, fairness_df



def build_step11_summary(step11_raw_df):
    summary = (
        step11_raw_df.groupby(["terrain", "layout", "sigma_t", "ref_strategy", "weight_strategy"], as_index=False)
        .agg(
            n_seed=("seed", "count"),
            M_per_seed=("M", "first"),
            mean_localization_error_mean=("mean_err_3d", "mean"),
            mean_localization_error_std=("mean_err_3d", "std"),
            median_localization_error_mean=("median_err_3d", "mean"),
            median_localization_error_std=("median_err_3d", "std"),
            p95_localization_error_mean=("p95_err_3d", "mean"),
            p95_localization_error_std=("p95_err_3d", "std"),
            valid_localization_rate_mean=("valid_localization_rate", "mean"),
            valid_localization_rate_std=("valid_localization_rate", "std"),
            normalized_deformation_index_mean=("dgeom_star", "mean"),
            normalized_deformation_index_std=("dgeom_star", "std"),
            edge_distortion_mean=("edge_distortion", "mean"),
            edge_distortion_std=("edge_distortion", "std"),
            height_spread_mean=("height_spread", "mean"),
            height_spread_std=("height_spread", "std"),
            deploy_ok_rate=("deploy_ok", "mean"),
            usable_event_rate_mean=("usable_event_rate", "mean"),
            usable_event_rate_std=("usable_event_rate", "std"),
            solver_success_given_usable_mean=("solver_success_given_usable", "mean"),
            solver_success_given_usable_std=("solver_success_given_usable", "std"),
            mean_detected_count_all_mean=("mean_detected_count_all", "mean"),
            mean_detected_count_all_std=("mean_detected_count_all", "std"),
            mean_arrival_spread_mean=("mean_arrival_spread", "mean"),
            mean_arrival_spread_std=("mean_arrival_spread", "std"),
            combo_runtime_sec_mean=("combo_runtime_sec", "mean"),
            combo_runtime_sec_std=("combo_runtime_sec", "std"),
            sigma_runtime_sec_mean=("sigma_runtime_sec", "mean"),
            sigma_runtime_sec_std=("sigma_runtime_sec", "std"),
        )
        .sort_values(["terrain", "sigma_t", "mean_localization_error_mean", "valid_localization_rate_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    std_cols = [
        "mean_localization_error_std",
        "median_localization_error_std",
        "p95_localization_error_std",
        "valid_localization_rate_std",
        "normalized_deformation_index_std",
        "edge_distortion_std",
        "height_spread_std",
        "usable_event_rate_std",
        "solver_success_given_usable_std",
        "mean_detected_count_all_std",
        "mean_arrival_spread_std",
    ]
    for c in std_cols:
        summary[c] = summary[c].fillna(0.0)

    sqrt_n = np.sqrt(np.maximum(summary["n_seed"].to_numpy(dtype=float), 1.0))
    summary["mean_localization_error_sem"] = summary["mean_localization_error_std"] / sqrt_n
    summary["median_localization_error_sem"] = summary["median_localization_error_std"] / sqrt_n
    summary["p95_localization_error_sem"] = summary["p95_localization_error_std"] / sqrt_n
    summary["valid_localization_rate_sem"] = summary["valid_localization_rate_std"] / sqrt_n
    summary["usable_event_rate_sem"] = summary["usable_event_rate_std"] / sqrt_n
    summary["mean_detected_count_all_sem"] = summary["mean_detected_count_all_std"] / sqrt_n

    summary["rank_mean_error_within_terrain_sigma"] = summary.groupby(["terrain", "sigma_t"])["mean_localization_error_mean"].rank(method="dense", ascending=True)
    summary["rank_p95_error_within_terrain_sigma"] = summary.groupby(["terrain", "sigma_t"])["p95_localization_error_mean"].rank(method="dense", ascending=True)
    summary["rank_valid_rate_within_terrain_sigma"] = summary.groupby(["terrain", "sigma_t"])["valid_localization_rate_mean"].rank(method="dense", ascending=False)
    return summary



def build_step11_ranking_flip_table(step11_summary_df, baseline_sigma=None):
    sigma_levels = sorted(step11_summary_df["sigma_t"].dropna().unique().tolist())
    if len(sigma_levels) == 0:
        return pd.DataFrame()
    if baseline_sigma is None:
        baseline_sigma = sigma_levels[0]

    rows = []
    terrains = sorted(step11_summary_df["terrain"].dropna().unique().tolist())
    for terrain in terrains:
        base = step11_summary_df[
            (step11_summary_df["terrain"] == terrain)
            & np.isclose(step11_summary_df["sigma_t"], baseline_sigma)
        ].copy()
        if base.empty:
            continue

        base_error_order = tuple(base.sort_values(["mean_localization_error_mean", "layout"])["layout"].tolist())
        base_p95_order = tuple(base.sort_values(["p95_localization_error_mean", "layout"])["layout"].tolist())
        base_valid_order = tuple(base.sort_values(["valid_localization_rate_mean", "layout"], ascending=[False, True])["layout"].tolist())

        for sigma_t in sigma_levels:
            sub = step11_summary_df[
                (step11_summary_df["terrain"] == terrain)
                & np.isclose(step11_summary_df["sigma_t"], sigma_t)
            ].copy()
            if sub.empty:
                continue

            error_order = tuple(sub.sort_values(["mean_localization_error_mean", "layout"])["layout"].tolist())
            p95_order = tuple(sub.sort_values(["p95_localization_error_mean", "layout"])["layout"].tolist())
            valid_order = tuple(sub.sort_values(["valid_localization_rate_mean", "layout"], ascending=[False, True])["layout"].tolist())

            rows.append({
                "terrain": terrain,
                "baseline_sigma_t": float(baseline_sigma),
                "sigma_t": float(sigma_t),
                "baseline_error_order": " > ".join(base_error_order),
                "error_order": " > ".join(error_order),
                "mean_error_ranking_changed": bool(error_order != base_error_order),
                "baseline_p95_order": " > ".join(base_p95_order),
                "p95_order": " > ".join(p95_order),
                "p95_ranking_changed": bool(p95_order != base_p95_order),
                "baseline_valid_order": " > ".join(base_valid_order),
                "valid_order": " > ".join(valid_order),
                "valid_rate_ranking_changed": bool(valid_order != base_valid_order),
            })

    return pd.DataFrame(rows).sort_values(["terrain", "sigma_t"]).reset_index(drop=True)



def build_step11_fairness_table(step11_raw_df):
    fairness = (
        step11_raw_df.groupby(["seed", "terrain", "sigma_t"], as_index=False)
        .agg(
            n_layouts=("layout", "nunique"),
            n_unique_event_banks=("event_bank_seed", "nunique"),
            M_per_seed=("M", "first"),
        )
        .sort_values(["seed", "terrain", "sigma_t"])
        .reset_index(drop=True)
    )
    return fairness



def _plot_step11_curve(summary_df, mean_col, sem_col, ylabel, title_prefix):
    terrains = ["flat", "hill", "rough"]
    layouts = ["square", "staggered", "boundary"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    for ax, terrain in zip(axes, terrains):
        sub = summary_df[summary_df["terrain"] == terrain].copy()
        for layout in layouts:
            ss = sub[sub["layout"] == layout].sort_values("sigma_t")
            if ss.empty:
                continue
            ax.errorbar(
                ss["sigma_t"],
                ss[mean_col],
                yerr=ss[sem_col],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=layout,
            )
        ax.set_title(f"{terrain}: {title_prefix}")
        ax.set_xlabel("timing noise σt")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.show()



def plot_step11_mean_error_curves(summary_df):
    _plot_step11_curve(
        summary_df,
        mean_col="mean_localization_error_mean",
        sem_col="mean_localization_error_sem",
        ylabel="mean localization error ± SEM",
        title_prefix="mean error vs timing noise",
    )



def plot_step11_p95_error_curves(summary_df):
    _plot_step11_curve(
        summary_df,
        mean_col="p95_localization_error_mean",
        sem_col="p95_localization_error_sem",
        ylabel="P95 localization error ± SEM",
        title_prefix="P95 error vs timing noise",
    )



def plot_step11_valid_rate_curves(summary_df):
    _plot_step11_curve(
        summary_df,
        mean_col="valid_localization_rate_mean",
        sem_col="valid_localization_rate_sem",
        ylabel="valid localization rate ± SEM",
        title_prefix="valid rate vs timing noise",
    )


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    wall_tic = time.perf_counter()
    step11_raw_df, step11_summary_df, step11_ranking_flip_df, step11_fairness_df = run_step11_robustness_experiment(
        sigma_t_levels=TIMING_NOISE_LEVELS,
        L=L,
        M=N_EVENTS,
        n_seeds=N_SEEDS,
        v_eff=V_EFF,
        geo_grid_n=GEO_GRID_N,
        search_grid_n=SEARCH_GRID_N,
        n_multi_start=N_MULTI_START,
        ref_strategy=REF_STRATEGY,
        weight_strategy=WEIGHT_STRATEGY,
        min_detect=MIN_DETECT,
        A0=A0,
        d0=D0,
        spread_p=SPREAD_P,
        alpha=ALPHA_ATTEN,
        A_noise=A_NOISE,
        snr_threshold=SNR_THRESHOLD,
        local_coupling_mode=LOCAL_COUPLING_MODE,
        eta_log_std=ETA_LOG_STD,
        monte_carlo_base_seed=MONTE_CARLO_BASE_SEED,
        rough_terrain_base_seed=ROUGH_TERRAIN_BASE_SEED,
        share_event_bank_across_terrains=SHARE_EVENT_BANK_ACROSS_TERRAINS,
        show_progress=SHOW_PROGRESS,
        event_progress_every=EVENT_PROGRESS_EVERY,
        progress_bar_width=PROGRESS_BAR_WIDTH,
        lsq_max_nfev=LEAST_SQUARES_MAX_NFEV,
        lsq_tol=LEAST_SQUARES_TOL,
    )
    total_runtime_sec = time.perf_counter() - wall_tic

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 200)

    main_cols = [
        "terrain", "layout", "sigma_t", "n_seed", "M_per_seed",
        "mean_localization_error_mean", "mean_localization_error_std",
        "median_localization_error_mean", "median_localization_error_std",
        "p95_localization_error_mean", "p95_localization_error_std",
        "valid_localization_rate_mean", "valid_localization_rate_std",
        "rank_mean_error_within_terrain_sigma", "rank_p95_error_within_terrain_sigma", "rank_valid_rate_within_terrain_sigma",
    ]

    print("\n=== Step 11 robustness main table ===")
    print(step11_summary_df[main_cols].to_string(index=False))

    print("\n=== Step 11 ranking-flip summary ===")
    print(step11_ranking_flip_df.to_string(index=False))

    print("\n=== Step 11 fairness check ===")
    print(step11_fairness_df.to_string(index=False))

    print(f"\nTotal runtime: {total_runtime_sec:.2f} s")

    if SAVE_STEP11_CSV:
        step11_raw_df.to_csv("step11_robustness_raw_results.csv", index=False)
        step11_summary_df.to_csv("step11_robustness_summary.csv", index=False)
        step11_ranking_flip_df.to_csv("step11_robustness_ranking_flip.csv", index=False)
        step11_fairness_df.to_csv("step11_robustness_fairness.csv", index=False)
        print("\nSaved: step11_robustness_raw_results.csv, step11_robustness_summary.csv, step11_robustness_ranking_flip.csv, step11_robustness_fairness.csv")

    if PLOT_STEP11_MEAN_ERROR:
        plot_step11_mean_error_curves(step11_summary_df)

    if PLOT_STEP11_P95_ERROR:
        plot_step11_p95_error_curves(step11_summary_df)

    if PLOT_STEP11_VALID_RATE:
        plot_step11_valid_rate_curves(step11_summary_df)
