import numpy as np
import pandas as pd
from itertools import combinations
from scipy.optimize import least_squares
from scipy.spatial import Delaunay

# =========================
# Step 1 minimal experiment
# Only test: geometry -> TDoA localization error
# No amplitude model, no detection threshold, no prototype
# =========================

L = 1.0              # domain size (normalized)
V_EFF = 1.0          # effective propagation speed
SIGMA_T = 0.002      # timing noise std
N_EVENTS = 300       # Monte Carlo events per terrain seed
N_SEEDS = 5          # number of seeds per terrain
BETA_ANCHOR = 0.15   # regularization to remove rigid drift in deployment solver
GRID_N = 11          # coarse grid for source initialization


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


def make_terrain(kind, L=1.0, seed=0):
    """Return a terrain height function z = h(x, y)."""
    if kind == "flat":
        return lambda x, y: np.zeros_like(np.asarray(x, dtype=float) + 0.0 * np.asarray(y, dtype=float))

    if kind == "hill":
        A = 0.14 * L
        xc, yc = 0.55 * L, 0.50 * L
        sigma = 0.22 * L
        return lambda x, y: A * np.exp(-(((np.asarray(x) - xc) ** 2 + (np.asarray(y) - yc) ** 2) / (2 * sigma ** 2)))

    if kind == "rough":
        rng = np.random.default_rng(seed)
        K = 5
        amps = rng.uniform(0.015 * L, 0.040 * L, size=K)
        wx = rng.integers(1, 4, size=K) * 2 * np.pi / L
        wy = rng.integers(1, 4, size=K) * 2 * np.pi / L
        ph = rng.uniform(0, 2 * np.pi, size=K)

        def h(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            z = np.zeros(np.broadcast(x, y).shape, dtype=float)
            for a, kx, ky, p in zip(amps, wx, wy, ph):
                z += a * np.sin(kx * x + p) * np.cos(ky * y - 0.5 * p)
            return z

        return h

    raise ValueError(f"Unknown terrain: {kind}")


def deploy_to_terrain(u, h, beta_anchor=0.15, L=1.0):
    """
    Minimal terrain-conforming deployment solver.
    It adjusts x,y while keeping the net close to its nominal shape,
    and sets z by the terrain height field.
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
    return r, edges, l0


def geometry_metrics(r, edges, l0):
    edge_now = np.array([np.linalg.norm(r[i] - r[j]) for i, j in edges])
    delta_edge = np.mean(np.abs(edge_now - l0) / l0)
    sigma_z = np.std(r[:, 2])
    return delta_edge, sigma_z


def source_point(q, h):
    return np.array([q[0], q[1], float(h(q[0], q[1]))], dtype=float)


def simulate_tdoa(r, q_true, h, v_eff=1.0, sigma_t=0.002, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    s_true = source_point(q_true, h)
    d = np.linalg.norm(r - s_true, axis=1)
    t = d / v_eff + rng.normal(0.0, sigma_t, size=len(r))

    # Dynamic reference: earliest arrival
    ref = int(np.argmin(t))
    dt = t - t[ref]
    return dt, ref, s_true


def estimate_source_xy(r, dt_obs, ref, h, L=1.0, v_eff=1.0, grid_n=11):
    # Coarse grid for a stable initial guess
    xs = np.linspace(0.05 * L, 0.95 * L, grid_n)
    ys = np.linspace(0.05 * L, 0.95 * L, grid_n)
    best_q = None
    best_loss = np.inf

    for x in xs:
        for y in ys:
            s = source_point((x, y), h)
            d = np.linalg.norm(r - s, axis=1)
            dt_pred = (d - d[ref]) / v_eff
            loss = np.sum((dt_obs - dt_pred) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_q = np.array([x, y], dtype=float)

    def resid(q):
        s = source_point(q, h)
        d = np.linalg.norm(r - s, axis=1)
        dt_pred = (d - d[ref]) / v_eff
        return dt_obs - dt_pred

    sol = least_squares(
        resid,
        best_q,
        bounds=([0.0, 0.0], [L, L]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=200,
    )

    return sol.x, sol.success


def run_step1(L=1.0, M=300, n_seeds=5, sigma_t=0.002, v_eff=1.0):
    layouts = ["square", "staggered", "boundary"]
    terrains = ["flat", "hill", "rough"]
    rows = []

    for terrain in terrains:
        for seed in range(n_seeds):
            h = make_terrain(terrain, L=L, seed=seed)

            # Same source set shared across layouts for fairness
            rng = np.random.default_rng(1000 + seed)
            sources_xy = rng.uniform(0.05 * L, 0.95 * L, size=(M, 2))

            for layout in layouts:
                u = make_layout(layout, L=L)
                r, edges, l0 = deploy_to_terrain(u, h, beta_anchor=BETA_ANCHOR, L=L)
                delta_edge, sigma_z = geometry_metrics(r, edges, l0)

                err_3d = []
                err_xy = []
                for q_true in sources_xy:
                    dt, ref, s_true = simulate_tdoa(r, q_true, h, v_eff=v_eff, sigma_t=sigma_t, rng=rng)
                    q_hat, _ = estimate_source_xy(r, dt, ref, h, L=L, v_eff=v_eff, grid_n=GRID_N)
                    s_hat = source_point(q_hat, h)
                    err_3d.append(np.linalg.norm(s_hat - s_true))
                    err_xy.append(np.linalg.norm(q_hat - q_true))

                rows.append({
                    "terrain": terrain,
                    "layout": layout,
                    "seed": seed,
                    "delta_edge": delta_edge,
                    "sigma_z": sigma_z,
                    "mean_err_3d": float(np.mean(err_3d)),
                    "median_err_3d": float(np.median(err_3d)),
                    "p95_err_3d": float(np.percentile(err_3d, 95)),
                    "mean_err_xy": float(np.mean(err_xy)),
                })

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["terrain", "layout"], as_index=False)
          .agg(
              mean_err_3d=("mean_err_3d", "mean"),
              median_err_3d=("median_err_3d", "mean"),
              p95_err_3d=("p95_err_3d", "mean"),
              mean_err_xy=("mean_err_xy", "mean"),
              delta_edge=("delta_edge", "mean"),
              sigma_z=("sigma_z", "mean"),
          )
          .sort_values(["terrain", "mean_err_3d"])
          .reset_index(drop=True)
    )
    return df, summary


if __name__ == "__main__":
    raw_df, summary_df = run_step1(L=L, M=N_EVENTS, n_seeds=N_SEEDS, sigma_t=SIGMA_T, v_eff=V_EFF)
    print("\n=== Step 1 summary ===")
    print(summary_df.to_string(index=False))

    raw_df.to_csv("step1_raw_results.csv", index=False)
    summary_df.to_csv("step1_summary.csv", index=False)
    print("\nSaved: step1_raw_results.csv, step1_summary.csv")
