import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.optimize import least_squares
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator

# =========================
# Step 3 script
# Step 1: minimal geometry -> TDoA localization
# Step 2: source is explicitly constrained to the terrain surface
# Step 3: terrain model is made explicit as:
#         flat / hill / rough (Gaussian random field)
#
# Added before Step 4:
#   deployment visualization
#   - nominal node positions
#   - deployed node positions
#   - displacement arrows
#   - edges before and after deployment
# =========================

# -------------------------
# Global parameters
# -------------------------
L = 1.0
V_EFF = 1.0
SIGMA_T = 0.002
N_EVENTS = 300
N_SEEDS = 5

BETA_ANCHOR = 0.15
GRID_N = 21
N_MULTI_START = 5

SAVE_CSV = False
PLOT_TERRAINS = True
PLOT_DEPLOYMENTS = True

# choose one seed for deployment visualization
PLOT_SEED = 0

# -------------------------
# Step 3 terrain parameters
# -------------------------
HILL_AMP = 0.14         # A_h as fraction of L
HILL_SIGMA = 0.22       # sigma_h as fraction of L
HILL_XC = 0.55          # hill center x as fraction of L
HILL_YC = 0.50          # hill center y as fraction of L

ROUGH_RMS = 0.03        # sigma_r as fraction of L
ROUGH_CORR_LEN = 0.18   # ell_r as fraction of L
ROUGH_GRID_N = 129      # interpolation grid resolution


def make_layout(name, L=1.0):
    """Return 8 nominal sensor coordinates in a planar L x L domain."""
    if name == "square":
        xs = np.array([0.18, 0.39, 0.60, 0.81]) * L
        ys = np.array([0.35, 0.65]) * L
        pts = np.array([[x, y] for y in ys for x in xs], dtype=float)

    elif name == "staggered":
        row1 = np.array([0.16, 0.36, 0.56, 0.76]) * L
        row2 = row1 + 0.10 * L
        y1, y2 = 0.33 * L, 0.67 * L
        pts = np.vstack([
            np.column_stack([row1, np.full(4, y1)]),
            np.column_stack([row2, np.full(4, y2)]),
        ])

    elif name == "boundary":
        pts = np.array([
            [0.15, 0.15],
            [0.50, 0.12],
            [0.85, 0.15],
            [0.88, 0.50],
            [0.85, 0.85],
            [0.50, 0.88],
            [0.15, 0.85],
            [0.12, 0.50],
        ]) * L

    else:
        raise ValueError(f"Unknown layout: {name}")

    return pts


def delaunay_edges(pts):
    """Use Delaunay triangulation to define the net connectivity."""
    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for a, b in combinations(simplex, 2):
            edges.add(tuple(sorted((a, b))))
    return sorted(edges)


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
    """
    Step 3 terrain model:
      flat  : h(x, y) = 0
      hill  : Gaussian hill
      rough : Gaussian random field with prescribed
              RMS height and correlation length
    """

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


def deploy_to_terrain(u, h, beta_anchor=0.15, L=1.0):
    """
    Minimal terrain-conforming deployment solver.

    Sensors start from nominal planar positions u = [(x_i, y_i)].
    They are allowed to adjust in x-y while z is constrained by terrain:
        r_i = (x_i, y_i, h(x_i, y_i))

    Objective:
    - approximately preserve nominal edge lengths
    - avoid rigid drift via anchor regularization
    """
    edges = delaunay_edges(u)
    l0 = np.array([np.linalg.norm(u[i] - u[j]) for i, j in edges])

    def residuals(v):
        xy = v.reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
        z = h(x, y)
        r = np.column_stack([x, y, z])

        edge_res = [np.linalg.norm(r[i] - r[j]) - lij for (i, j), lij in zip(edges, l0)]
        anchor_res = np.sqrt(beta_anchor) * (xy - u).ravel()
        return np.concatenate([np.array(edge_res), anchor_res])

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
        max_nfev=5000,
    )

    xy = sol.x.reshape(-1, 2)
    r = np.column_stack([xy[:, 0], xy[:, 1], h(xy[:, 0], xy[:, 1])])

    deploy_resnorm = np.linalg.norm(residuals(sol.x))
    return r, edges, l0, sol.success, deploy_resnorm


def geometry_metrics(r, edges, l0):
    """Return simple geometry distortion indicators."""
    edge_now = np.array([np.linalg.norm(r[i] - r[j]) for i, j in edges])
    delta_edge = np.mean(np.abs(edge_now - l0) / l0)
    sigma_z = np.std(r[:, 2])
    return delta_edge, sigma_z


def surface_source_point(q, h):
    """
    Step 2 core definition:
        q = (x_s, y_s) is the 2D source parameter
        s(q) = (x_s, y_s, h(x_s, y_s)) is the actual 3D source on the terrain
    """
    return np.array([q[0], q[1], float(h(q[0], q[1]))], dtype=float)


def simulate_tdoa_surface(r, q_true, h, timing_noise, v_eff=1.0):
    """
    Forward model for a surface-constrained source.

    The source is not a free 3D point.
    It is parameterized by q_true = (x_s, y_s), then mapped to:
        s_true = (x_s, y_s, h(x_s, y_s))

    Arrival times:
        t_i = ||r_i - s_true|| / v_eff + noise_i

    Reference node:
        earliest arrival node
    """
    s_true = surface_source_point(q_true, h)
    d = np.linalg.norm(r - s_true, axis=1)
    t = d / v_eff + timing_noise

    ref = int(np.argmin(t))
    dt = t - t[ref]
    return dt, ref, s_true


def estimate_surface_source_xy(r, dt_obs, ref, h, L=1.0, v_eff=1.0, grid_n=21, n_multi_start=5):
    """
    Estimate a surface-constrained source.

    Optimization variable:
        q = (x, y)

    Recovered 3D source:
        s(q) = (x, y, h(x, y))

    Objective:
        minimize TDoA mismatch in 2D parameter space
    """
    use_idx = np.array([i for i in range(len(r)) if i != ref], dtype=int)

    def resid(q):
        s = surface_source_point(q, h)
        d = np.linalg.norm(r - s, axis=1)
        dt_pred = (d - d[ref]) / v_eff
        return dt_obs[use_idx] - dt_pred[use_idx]

    xs = np.linspace(0.05 * L, 0.95 * L, grid_n)
    ys = np.linspace(0.05 * L, 0.95 * L, grid_n)

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
            max_nfev=500,
        )
        cost = np.sum(sol.fun ** 2)

        if sol.success and cost < best_cost:
            best_cost = cost
            best_sol = sol

    if best_sol is None:
        return None, False, np.inf

    return best_sol.x, True, np.sqrt(best_cost)


def plot_terrain_examples(L=1.0):
    """Visual sanity check for Step 3 terrain models."""
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

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[0, L, 0, L],
            aspect="equal",
        )
        ax.set_title(
            f"{terrain}\nmean={np.mean(Z):.4f}, std={np.std(Z):.4f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Step 3 terrain models")
    plt.show()


def plot_deployment_maps(L=1.0, seed=0, beta_anchor=0.15):
    """
    Visualize nominal vs deployed node positions
    for each terrain × layout combination.
    """
    terrains = ["flat", "hill", "rough"]
    layouts = ["square", "staggered", "boundary"]

    x = np.linspace(0.0, L, 180)
    y = np.linspace(0.0, L, 180)
    X, Y = np.meshgrid(x, y, indexing="xy")

    fig, axes = plt.subplots(
        len(terrains),
        len(layouts),
        figsize=(14, 12),
        constrained_layout=True
    )

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
            r, edges, l0, deploy_ok, deploy_resnorm = deploy_to_terrain(
                u, h, beta_anchor=beta_anchor, L=L
            )
            delta_edge, sigma_z = geometry_metrics(r, edges, l0)

            im = ax.imshow(
                Z,
                origin="lower",
                extent=[0, L, 0, L],
                aspect="equal",
                alpha=0.88,
            )

            # nominal edges (dashed)
            for a, b in edges:
                ax.plot(
                    [u[a, 0], u[b, 0]],
                    [u[a, 1], u[b, 1]],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.65,
                    color="white",
                )

            # deployed edges (solid)
            for a, b in edges:
                ax.plot(
                    [r[a, 0], r[b, 0]],
                    [r[a, 1], r[b, 1]],
                    linestyle="-",
                    linewidth=1.2,
                    alpha=0.95,
                    color="red",
                )

            # displacement arrows
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

            # nominal nodes
            ax.scatter(
                u[:, 0], u[:, 1],
                s=55,
                facecolors="none",
                edgecolors="white",
                linewidths=1.8,
                label="nominal" if (i == 0 and j == 0) else None,
                zorder=5,
            )

            # deployed nodes
            ax.scatter(
                r[:, 0], r[:, 1],
                s=45,
                c="red",
                marker="x",
                linewidths=2.0,
                label="deployed" if (i == 0 and j == 0) else None,
                zorder=6,
            )

            # sensor ids
            for k in range(len(u)):
                ax.text(
                    r[k, 0] + 0.01 * L,
                    r[k, 1] + 0.01 * L,
                    str(k + 1),
                    fontsize=8,
                    color="black",
                    weight="bold",
                )

            ax.set_title(
                f"{terrain} | {layout}\n"
                f"res={deploy_resnorm:.4e}, "
                f"δedge={delta_edge:.4f}, σz={sigma_z:.4f}"
            )
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.ax.set_ylabel("height", rotation=90)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)

    fig.suptitle("Deployment visualization: nominal vs deployed node positions", fontsize=15)
    plt.show()


def run_step3(
    L=1.0,
    M=300,
    n_seeds=5,
    sigma_t=0.002,
    v_eff=1.0,
    beta_anchor=0.15,
    grid_n=21,
    n_multi_start=5,
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
            source_params_xy = rng.uniform(0.05 * L, 0.95 * L, size=(M, 2))
            shared_timing_noise = rng.normal(0.0, sigma_t, size=(M, n_sensors))

            for layout in layouts:
                u = make_layout(layout, L=L)

                r, edges, l0, deploy_ok, deploy_resnorm = deploy_to_terrain(
                    u, h, beta_anchor=beta_anchor, L=L
                )
                delta_edge, sigma_z = geometry_metrics(r, edges, l0)

                err_3d = []
                err_xy = []
                fit_resnorms = []
                n_failed_localization = 0

                for event_id, q_true in enumerate(source_params_xy):
                    dt, ref, s_true = simulate_tdoa_surface(
                        r=r,
                        q_true=q_true,
                        h=h,
                        timing_noise=shared_timing_noise[event_id],
                        v_eff=v_eff,
                    )

                    q_hat, ok, fit_resnorm = estimate_surface_source_xy(
                        r=r,
                        dt_obs=dt,
                        ref=ref,
                        h=h,
                        L=L,
                        v_eff=v_eff,
                        grid_n=grid_n,
                        n_multi_start=n_multi_start,
                    )

                    if not ok or q_hat is None:
                        n_failed_localization += 1
                        continue

                    s_hat = surface_source_point(q_hat, h)

                    err_3d.append(np.linalg.norm(s_hat - s_true))
                    err_xy.append(np.linalg.norm(q_hat - q_true))
                    fit_resnorms.append(fit_resnorm)

                n_valid = len(err_3d)
                valid_rate = n_valid / M

                if n_valid == 0:
                    rows.append({
                        "terrain": terrain,
                        "layout": layout,
                        "seed": seed,
                        "deploy_ok": deploy_ok,
                        "deploy_resnorm": deploy_resnorm,
                        "delta_edge": delta_edge,
                        "sigma_z": sigma_z,
                        "valid_rate": 0.0,
                        "failed_localization": n_failed_localization,
                        "mean_fit_resnorm": np.nan,
                        "mean_err_3d": np.nan,
                        "median_err_3d": np.nan,
                        "p95_err_3d": np.nan,
                        "mean_err_xy": np.nan,
                    })
                    continue

                rows.append({
                    "terrain": terrain,
                    "layout": layout,
                    "seed": seed,
                    "deploy_ok": bool(deploy_ok),
                    "deploy_resnorm": float(deploy_resnorm),
                    "delta_edge": float(delta_edge),
                    "sigma_z": float(sigma_z),
                    "valid_rate": float(valid_rate),
                    "failed_localization": int(n_failed_localization),
                    "mean_fit_resnorm": float(np.mean(fit_resnorms)),
                    "mean_err_3d": float(np.mean(err_3d)),
                    "median_err_3d": float(np.median(err_3d)),
                    "p95_err_3d": float(np.percentile(err_3d, 95)),
                    "mean_err_xy": float(np.mean(err_xy)),
                })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["terrain", "layout"], as_index=False)
        .agg(
            deploy_ok_rate=("deploy_ok", "mean"),
            deploy_resnorm=("deploy_resnorm", "mean"),
            delta_edge=("delta_edge", "mean"),
            sigma_z=("sigma_z", "mean"),
            valid_rate=("valid_rate", "mean"),
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


if __name__ == "__main__":
    raw_df, summary_df = run_step3(
        L=L,
        M=N_EVENTS,
        n_seeds=N_SEEDS,
        sigma_t=SIGMA_T,
        v_eff=V_EFF,
        beta_anchor=BETA_ANCHOR,
        grid_n=GRID_N,
        n_multi_start=N_MULTI_START,
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("\n=== Step 3 summary ===")
    print(summary_df.to_string(index=False))

    if SAVE_CSV:
        raw_df.to_csv("step3_raw_results.csv", index=False)
        summary_df.to_csv("step3_summary.csv", index=False)
        print("\nSaved: step3_raw_results.csv, step3_summary.csv")

    if PLOT_TERRAINS:
        plot_terrain_examples(L=L)

    if PLOT_DEPLOYMENTS:
        plot_deployment_maps(
            L=L,
            seed=PLOT_SEED,
            beta_anchor=BETA_ANCHOR,
        )