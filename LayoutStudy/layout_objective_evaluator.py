import json
from itertools import combinations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import ConvexHull, Delaunay


DEFAULT_CONFIG = {
    "L": 1.0,
    "V_EFF": 1.0,
    "SIGMA_T": 0.002,
    "N_EVENTS": 2000,
    "N_SEEDS": 10,
    "MONTE_CARLO_BASE_SEED": 20260317,
    "SHARE_EVENT_BANK_ACROSS_TERRAINS": True,
    "REF_STRATEGY": "max_snr",
    "WEIGHT_STRATEGY": "snr",
    "MIN_DETECT": 4,
    "K_EDGE": 1.0,
    "K_CENTROID": 40.0,
    "K_ORIENT": 20.0,
    "K_DRIFT": 0.10,
    "HILL_AMP": 0.14,
    "HILL_SIGMA": 0.22,
    "HILL_XC": 0.55,
    "HILL_YC": 0.50,
    "HILL_RANDOMIZE": True,
    "HILL_AMP_RANGE": (0.08, 0.20),
    "HILL_SIGMA_RANGE": (0.16, 0.30),
    "HILL_XC_RANGE": (0.30, 0.70),
    "HILL_YC_RANGE": (0.30, 0.70),
    "ROUGH_RMS": 0.03,
    "ROUGH_CORR_LEN": 0.18,
    "ROUGH_GRID_N": 129,
    "ROUGH_RANDOMIZE": True,
    "TERRAIN_MASTER_SEED": None,
    "T0_MIN": 0.0,
    "T0_MAX": 0.2,
    "GEO_GRID_N": 41,
    "GEO_CONNECTIVITY": 8,
    "SEARCH_GRID_N": 15,
    "N_MULTI_START": 6,
    "A0": 0.40,
    "D0": 0.05,
    "SPREAD_P": 1.35,
    "ALPHA_ATTEN": 2.00,
    "A_NOISE": 0.20,
    "SNR_THRESHOLD": 2.00,
    "LOCAL_COUPLING_MODE": "none",
    "ETA_LOG_STD": 0.15,
    "LEAST_SQUARES_MAX_NFEV": 250,
    "LEAST_SQUARES_TOL": 1e-8,
    "DGEOM_W_EDGE": 0.50,
    "DGEOM_W_HEIGHT": 0.50,
    "DGEOM_EDGE_MEAN": 0.0,
    "DGEOM_EDGE_STD": 0.03,
    "DGEOM_HEIGHT_MEAN": 0.0,
    "DGEOM_HEIGHT_STD": 0.03,
    "VALIDATION_MIN_PAIRWISE_SEP": 0.08,
    "VALIDATION_BOUNDARY_MARGIN": 0.02,
    "VALIDATION_MIN_HULL_AREA": 0.08,
    "FAILURE_PENALTY_ERR_SCALE": 2.0,
    "INVALID_LAYOUT_RETURNS_PENALTY": True,
    "INVALID_LAYOUT_PENALTY": 1e6,
    "OBJECTIVE_NAN_PENALTY": 1e6,
}


def theta_to_layout_xy(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        if theta.size % 2 != 0:
            raise ValueError("theta length must be even so it can be reshaped into (n_nodes, 2).")
        return theta.reshape(-1, 2)
    if theta.ndim == 2 and theta.shape[1] == 2:
        return theta.astype(float, copy=False)
    raise ValueError("theta must be shape (2*n_nodes,) or (n_nodes, 2).")


def hull_area_2d(layout_xy):
    layout_xy = theta_to_layout_xy(layout_xy)
    if layout_xy.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(layout_xy)
        return float(hull.volume)
    except Exception:
        return 0.0


def min_pairwise_distance(layout_xy):
    layout_xy = theta_to_layout_xy(layout_xy)
    n = layout_xy.shape[0]
    if n < 2:
        return np.inf
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(layout_xy[i] - layout_xy[j])
            if d < best:
                best = d
    return float(best)


def validate_layout_xy(
    layout_xy,
    L=1.0,
    min_pairwise_sep=0.08,
    boundary_margin=0.02,
    min_hull_area=0.08,
):
    layout_xy = theta_to_layout_xy(layout_xy)
    if layout_xy.shape[0] < 4:
        raise ValueError("layout must contain at least 4 nodes.")
    if np.any(~np.isfinite(layout_xy)):
        raise ValueError("layout contains NaN or inf.")
    if boundary_margin < 0.0 or boundary_margin >= 0.5 * L:
        raise ValueError("boundary_margin must be non-negative and smaller than half of L.")

    lo = boundary_margin
    hi = L - boundary_margin
    if np.any(layout_xy < lo) or np.any(layout_xy > hi):
        raise ValueError(f"layout coordinates must lie inside [{lo}, {hi}] after applying boundary margin.")

    if min_pairwise_distance(layout_xy) < float(min_pairwise_sep):
        raise ValueError(f"layout violates minimum pairwise separation of {float(min_pairwise_sep):.6f}.")

    area = hull_area_2d(layout_xy)
    if area < float(min_hull_area):
        raise ValueError(f"layout convex hull area {area:.6f} is below minimum {float(min_hull_area):.6f}.")

    try:
        _ = Delaunay(layout_xy)
    except Exception as exc:
        raise ValueError("layout must not be degenerate/collinear for Delaunay triangulation.") from exc
    return layout_xy


def _spawn_rng(master_rng):
    child_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(child_seed), child_seed


def make_terrain(
    kind,
    L=1.0,
    rng=None,
    hill_amp=0.14,
    hill_sigma=0.22,
    hill_xc=0.55,
    hill_yc=0.50,
    hill_randomize=True,
    hill_amp_range=(0.08, 0.20),
    hill_sigma_range=(0.16, 0.30),
    hill_xc_range=(0.30, 0.70),
    hill_yc_range=(0.30, 0.70),
    rough_rms=0.03,
    rough_corr_len=0.18,
    rough_grid_n=129,
):
    if kind == "flat":
        def h(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            return np.zeros(np.broadcast(x, y).shape, dtype=float)
        return h, {"kind": "flat"}

    if rng is None:
        rng = np.random.default_rng(None)

    if kind == "hill":
        if hill_randomize:
            hill_amp = float(rng.uniform(*hill_amp_range))
            hill_sigma = float(rng.uniform(*hill_sigma_range))
            hill_xc = float(rng.uniform(*hill_xc_range))
            hill_yc = float(rng.uniform(*hill_yc_range))

        A_h = hill_amp * L
        sigma_h = hill_sigma * L
        xc = hill_xc * L
        yc = hill_yc * L

        def h(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            return A_h * np.exp(-(((x - xc) ** 2 + (y - yc) ** 2) / (2.0 * sigma_h ** 2)))

        return h, {
            "kind": "hill",
            "hill_amp": float(hill_amp),
            "hill_sigma": float(hill_sigma),
            "hill_xc": float(hill_xc),
            "hill_yc": float(hill_yc),
        }

    if kind == "rough":
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

        return h, {
            "kind": "rough",
            "rough_rms": float(rough_rms),
            "rough_corr_len": float(rough_corr_len),
            "rough_grid_n": int(rough_grid_n),
        }

    raise ValueError(f"Unknown terrain: {kind}")


def delaunay_edges(pts):
    tri = Delaunay(pts)
    edges = set()
    for simplex in tri.simplices:
        for a, b in combinations(simplex, 2):
            edges.add(tuple(sorted((a, b))))
    return sorted(edges)


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


def deploy_to_terrain_stable(u, h, L=1.0, k_edge=1.0, k_centroid=40.0, k_orient=20.0, k_drift=0.10):
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
    orient_error_deg = np.degrees(np.abs(wrap_to_pi(theta - theta0)))
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
    }
    return r, info


def geometry_metrics(r, edges, l0):
    edge_now = np.array([np.linalg.norm(r[i] - r[j]) for i, j in edges], dtype=float)
    delta_edge = np.mean(np.abs(edge_now - l0) / np.maximum(l0, 1e-12))
    sigma_z = np.std(r[:, 2])
    return float(delta_edge), float(sigma_z)


def surface_source_point(q, h):
    return np.array([q[0], q[1], float(h(q[0], q[1]))], dtype=float)


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

    rows, cols, data = [], [], []
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
    for k in range(n_sensors):
        ix, iy = nearest_grid_index(r[k, :2], xs, ys)
        node = flatten_idx(ix, iy, nxy)
        dist_vec = dijkstra(G, directed=True, indices=node)
        distance_fields[k] = dist_vec.reshape(nxy, nxy)

    return {"xs": xs, "ys": ys, "distance_fields": distance_fields, "search_grid_cache": {}}


def _bilinear_interp_stacked(fields, xs, ys, qx, qy):
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
    return _bilinear_interp_stacked(
        geo_model["distance_fields"],
        geo_model["xs"],
        geo_model["ys"],
        float(np.clip(q[0], 0.0, L)),
        float(np.clip(q[1], 0.0, L)),
    )


def get_search_grid_cache(geo_model, L=1.0, search_grid_n=15):
    cache = geo_model.setdefault("search_grid_cache", {})
    key = (float(L), int(search_grid_n))
    if key in cache:
        return cache[key]

    xs_grid = np.linspace(0.05 * L, 0.95 * L, int(search_grid_n))
    ys_grid = np.linspace(0.05 * L, 0.95 * L, int(search_grid_n))
    XX, YY = np.meshgrid(xs_grid, ys_grid, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    all_d_grid = _bilinear_interp_stacked_batch(geo_model["distance_fields"], geo_model["xs"], geo_model["ys"], pts)
    payload = {"pts": pts, "all_d_grid": all_d_grid}
    cache[key] = payload
    return payload


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
    local_coupling_mode="none",
    eta_log_std=0.15,
):
    rng = np.random.default_rng(seed)
    q_true_bank = rng.uniform(0.05 * L, 0.95 * L, size=(M, 2))
    timing_noise_stdnormal_bank = rng.normal(0.0, 1.0, size=(M, n_sensors))
    timing_noise_bank = sigma_t * timing_noise_stdnormal_bank
    t0_bank = rng.uniform(t0_min, t0_max, size=M)
    A0_bank = np.full(M, A0, dtype=float)

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

    all_d = _bilinear_interp_stacked_batch(geo_model["distance_fields"], geo_model["xs"], geo_model["ys"], q_true_bank)

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
        return int(detected_idx[np.argmax(np.asarray(amplitude, dtype=float)[detected_idx])])
    if strategy == "max_snr":
        return int(detected_idx[np.argmax(np.asarray(snr, dtype=float)[detected_idx])])
    raise ValueError(f"Unknown reference strategy: {strategy}")


def make_tdoa_weights(use_idx, strategy="snr", amplitude=None, snr=None):
    use_idx = np.asarray(use_idx, dtype=int)
    if use_idx.size == 0:
        return np.array([], dtype=float)
    if strategy == "uniform":
        return np.ones(use_idx.size, dtype=float)
    if strategy == "amplitude":
        raw = np.asarray(amplitude, dtype=float)[use_idx]
        raw = np.maximum(raw, 1e-12)
        return raw / np.mean(raw)
    if strategy == "snr":
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
    if detected_idx.size < min_detect:
        return None, False

    ref = choose_reference_node(t=t, detected_idx=detected_idx, strategy=ref_strategy, amplitude=amplitude, snr=snr)
    use_idx = detected_idx[detected_idx != ref]
    dt_obs = t[use_idx] - t[ref]
    weights = make_tdoa_weights(use_idx, strategy=weight_strategy, amplitude=amplitude, snr=snr)
    return {
        "ref": int(ref),
        "use_idx": use_idx,
        "dt_obs": dt_obs,
        "weights": weights,
        "n_detect": int(detected_idx.size),
    }, True


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

    search_cache = get_search_grid_cache(geo_model, L=L, search_grid_n=search_grid_n)
    pts = search_cache["pts"]
    all_d_grid = search_cache["all_d_grid"]
    dt_pred_grid = (all_d_grid[use_idx, :] - all_d_grid[ref, :]) / v_eff
    residuals_grid = sqrt_w[:, None] * (dt_obs[:, None] - dt_pred_grid)
    losses = np.einsum("ij,ij->j", residuals_grid, residuals_grid)

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
        cost = np.dot(sol.fun, sol.fun)
        if sol.success and cost < best_cost:
            best_sol = sol
            best_cost = cost

    if best_sol is None:
        return None, False, np.inf
    return best_sol.x, True, float(np.sqrt(best_cost))


def resolve_dgeom_stats(dgeom_stats, cfg):
    if dgeom_stats is not None:
        required = ("edge_mean", "edge_std", "height_mean", "height_std")
        missing = [k for k in required if k not in dgeom_stats]
        if missing:
            raise ValueError(f"dgeom_stats is missing keys: {missing}")
        return {
            "edge_mean": float(dgeom_stats["edge_mean"]),
            "edge_std": float(dgeom_stats["edge_std"]),
            "height_mean": float(dgeom_stats["height_mean"]),
            "height_std": float(dgeom_stats["height_std"]),
            "source": "user_provided",
        }
    return {
        "edge_mean": float(cfg["DGEOM_EDGE_MEAN"]),
        "edge_std": float(cfg["DGEOM_EDGE_STD"]),
        "height_mean": float(cfg["DGEOM_HEIGHT_MEAN"]),
        "height_std": float(cfg["DGEOM_HEIGHT_STD"]),
        "source": "fixed_config",
    }


def add_dgeom_star(rows, dgeom_stats, w_edge=0.5, w_height=0.5):
    if len(rows) == 0:
        return rows

    delta_edge = np.array([row["delta_edge"] for row in rows], dtype=float)
    sigma_z = np.array([row["sigma_z"] for row in rows], dtype=float)

    edge_mean = float(dgeom_stats["edge_mean"])
    edge_std = float(dgeom_stats["edge_std"])
    height_mean = float(dgeom_stats["height_mean"])
    height_std = float(dgeom_stats["height_std"])

    z_edge = np.zeros_like(delta_edge) if edge_std < 1e-12 else (delta_edge - edge_mean) / edge_std
    z_height = np.zeros_like(sigma_z) if height_std < 1e-12 else (sigma_z - height_mean) / height_std

    dgeom = w_edge * z_edge + w_height * z_height
    for row, value in zip(rows, dgeom):
        row["dgeom_star"] = float(value)
    return rows


def compute_dgeom_stats_from_rows(rows):
    if len(rows) == 0:
        raise ValueError("rows must be non-empty.")
    delta_edge = np.array([row["delta_edge"] for row in rows], dtype=float)
    sigma_z = np.array([row["sigma_z"] for row in rows], dtype=float)
    return {
        "edge_mean": float(np.mean(delta_edge)),
        "edge_std": float(max(np.std(delta_edge), 1e-12)),
        "height_mean": float(np.mean(sigma_z)),
        "height_std": float(max(np.std(sigma_z), 1e-12)),
    }


def collapse_error_metrics(err_3d, penalty_err):
    err_3d = np.asarray(err_3d, dtype=float)
    if err_3d.size == 0:
        return float(penalty_err), float(penalty_err), float(penalty_err), 0
    return float(np.mean(err_3d)), float(np.median(err_3d)), float(np.percentile(err_3d, 95)), int(err_3d.size)


def summarize_rows_by_terrain(rows):
    terrains = sorted({row["terrain"] for row in rows})
    terrain_summary = {}
    for terrain in terrains:
        sub = [row for row in rows if row["terrain"] == terrain]
        terrain_summary[terrain] = {
            "n_seed": len(sub),
            "mean_err_3d_post": float(np.nanmean([r["mean_err_3d"] for r in sub])),
            "median_err_3d_post": float(np.nanmean([r["median_err_3d"] for r in sub])),
            "p95_err_3d_post": float(np.nanmean([r["p95_err_3d"] for r in sub])),
            "valid_localization_rate_post": float(np.nanmean([r["valid_localization_rate"] for r in sub])),
            "dgeom_star": float(np.nanmean([r["dgeom_star"] for r in sub])),
            "delta_edge": float(np.nanmean([r["delta_edge"] for r in sub])),
            "sigma_z": float(np.nanmean([r["sigma_z"] for r in sub])),
            "deploy_resnorm": float(np.nanmean([r["deploy_resnorm"] for r in sub])),
            "usable_event_rate": float(np.nanmean([r["usable_event_rate"] for r in sub])),
        }
    return terrain_summary


def compute_objective_terms(terrain_summary, nan_penalty=1e6):
    terrain_scores = {}
    for terrain, stat in terrain_summary.items():
        J_t = (
            0.45 * stat["mean_err_3d_post"]
            + 0.30 * stat["p95_err_3d_post"]
            + 0.15 * (1.0 - stat["valid_localization_rate_post"])
            + 0.10 * stat["dgeom_star"]
        )
        if not np.isfinite(J_t):
            J_t = float(nan_penalty)
        terrain_scores[terrain] = {**stat, "J_t": float(J_t)}

    if not terrain_scores:
        return {}, float(nan_penalty)
    jt_values = np.array([x["J_t"] for x in terrain_scores.values()], dtype=float)
    if np.any(~np.isfinite(jt_values)):
        return terrain_scores, float(nan_penalty)
    J_universal = 0.75 * float(np.mean(jt_values)) + 0.25 * float(np.max(jt_values))
    if not np.isfinite(J_universal):
        J_universal = float(nan_penalty)
    return terrain_scores, float(J_universal)


def make_penalty_result(cfg, terrain_kinds, reason, return_raw=False, raw_rows=None):
    terrain_kinds = tuple(terrain_kinds)
    penalty = float(cfg["INVALID_LAYOUT_PENALTY"])
    terrain_scores = {
        terrain: {
            "n_seed": 0,
            "mean_err_3d_post": penalty,
            "median_err_3d_post": penalty,
            "p95_err_3d_post": penalty,
            "valid_localization_rate_post": 0.0,
            "dgeom_star": penalty,
            "delta_edge": penalty,
            "sigma_z": penalty,
            "deploy_resnorm": penalty,
            "usable_event_rate": 0.0,
            "J_t": penalty,
        }
        for terrain in terrain_kinds
    }
    result = {
        "J_universal": penalty,
        "terrain_scores": terrain_scores,
        "config": cfg,
        "valid": False,
        "failure_reason": str(reason),
        "dgeom_stats_mode": "penalty",
    }
    if return_raw:
        result["raw_rows"] = [] if raw_rows is None else raw_rows
    return result


def evaluate_layout_objective(layout_xy, terrain_kinds=("flat", "hill", "rough"), dgeom_stats=None, return_raw=False, **overrides):
    cfg = {**DEFAULT_CONFIG, **overrides}
    L = cfg["L"]
    terrain_kinds = tuple(terrain_kinds)
    penalty_err = float(cfg["FAILURE_PENALTY_ERR_SCALE"]) * float(L)

    try:
        layout_xy = validate_layout_xy(
            layout_xy,
            L=L,
            min_pairwise_sep=float(cfg["VALIDATION_MIN_PAIRWISE_SEP"]),
            boundary_margin=float(cfg["VALIDATION_BOUNDARY_MARGIN"]),
            min_hull_area=float(cfg["VALIDATION_MIN_HULL_AREA"]),
        )
    except Exception as exc:
        if cfg["INVALID_LAYOUT_RETURNS_PENALTY"]:
            return make_penalty_result(cfg, terrain_kinds, f"invalid_layout: {exc}", return_raw=return_raw)
        raise

    n_sensors = layout_xy.shape[0]
    rows = []
    resolved_dgeom_stats = resolve_dgeom_stats(dgeom_stats, cfg)

    terrain_master_seed = cfg.get("TERRAIN_MASTER_SEED", None)
    terrain_master_rng = np.random.default_rng(terrain_master_seed if terrain_master_seed is not None else None)

    shared_event_banks = {}
    for seed in range(int(cfg["N_SEEDS"])):
        event_seed = int(cfg["MONTE_CARLO_BASE_SEED"]) + seed
        shared_event_banks[seed] = build_shared_event_bank(
            M=int(cfg["N_EVENTS"]),
            n_sensors=n_sensors,
            seed=event_seed,
            L=L,
            sigma_t=cfg["SIGMA_T"],
            t0_min=cfg["T0_MIN"],
            t0_max=cfg["T0_MAX"],
            A0=cfg["A0"],
            local_coupling_mode=cfg["LOCAL_COUPLING_MODE"],
            eta_log_std=cfg["ETA_LOG_STD"],
        )

    terrain_realizations = {terrain: [] for terrain in terrain_kinds}

    for seed in range(int(cfg["N_SEEDS"])):
        if cfg["SHARE_EVENT_BANK_ACROSS_TERRAINS"]:
            event_bank_by_terrain = {terrain: shared_event_banks[seed] for terrain in terrain_kinds}
        else:
            event_bank_by_terrain = {}
            for terrain_idx, terrain in enumerate(terrain_kinds):
                event_bank_by_terrain[terrain] = build_shared_event_bank(
                    M=int(cfg["N_EVENTS"]),
                    n_sensors=n_sensors,
                    seed=int(cfg["MONTE_CARLO_BASE_SEED"]) + 1000 * terrain_idx + seed,
                    L=L,
                    sigma_t=cfg["SIGMA_T"],
                    t0_min=cfg["T0_MIN"],
                    t0_max=cfg["T0_MAX"],
                    A0=cfg["A0"],
                    local_coupling_mode=cfg["LOCAL_COUPLING_MODE"],
                    eta_log_std=cfg["ETA_LOG_STD"],
                )

        terrain_models = {}
        for terrain in terrain_kinds:
            terrain_rng, terrain_child_seed = _spawn_rng(terrain_master_rng)
            h, terrain_meta = make_terrain(
                terrain,
                L=L,
                rng=terrain_rng,
                hill_amp=cfg["HILL_AMP"],
                hill_sigma=cfg["HILL_SIGMA"],
                hill_xc=cfg["HILL_XC"],
                hill_yc=cfg["HILL_YC"],
                hill_randomize=bool(cfg["HILL_RANDOMIZE"]),
                hill_amp_range=tuple(cfg["HILL_AMP_RANGE"]),
                hill_sigma_range=tuple(cfg["HILL_SIGMA_RANGE"]),
                hill_xc_range=tuple(cfg["HILL_XC_RANGE"]),
                hill_yc_range=tuple(cfg["HILL_YC_RANGE"]),
                rough_rms=cfg["ROUGH_RMS"],
                rough_corr_len=cfg["ROUGH_CORR_LEN"],
                rough_grid_n=int(cfg["ROUGH_GRID_N"]),
            )
            terrain_meta = {**terrain_meta, "seed_index": int(seed), "terrain_child_seed": int(terrain_child_seed)}
            terrain_realizations[terrain].append(terrain_meta)
            terrain_models[terrain] = (h, terrain_meta)

        for terrain in terrain_kinds:
            h, terrain_meta = terrain_models[terrain]
            event_bank = event_bank_by_terrain[terrain]

            r, info = deploy_to_terrain_stable(
                layout_xy,
                h,
                L=L,
                k_edge=cfg["K_EDGE"],
                k_centroid=cfg["K_CENTROID"],
                k_orient=cfg["K_ORIENT"],
                k_drift=cfg["K_DRIFT"],
            )
            geo_model = build_sensor_geodesic_model(
                r,
                h,
                L=L,
                grid_n=int(cfg["GEO_GRID_N"]),
                connectivity=int(cfg["GEO_CONNECTIVITY"]),
            )
            delta_edge, sigma_z = geometry_metrics(r, info["edges"], info["l0"])

            pre = precompute_layout_event_bank(
                geo_model=geo_model,
                h=h,
                q_true_bank=event_bank["q_true_bank"],
                timing_noise_bank=event_bank["timing_noise_bank"],
                t0_bank=event_bank["t0_bank"],
                A0_bank=event_bank["A0_bank"],
                eta_bank=event_bank["eta_bank"],
                v_eff=cfg["V_EFF"],
                d0=cfg["D0"],
                spread_p=cfg["SPREAD_P"],
                alpha=cfg["ALPHA_ATTEN"],
                A_noise=cfg["A_NOISE"],
                snr_threshold=cfg["SNR_THRESHOLD"],
            )

            err_3d = []
            n_failed_detection = 0
            n_failed_localization = 0
            detected_counts_usable = []

            for event_id in range(len(event_bank["q_true_bank"])):
                t = pre["t_bank"][event_id]
                s_true = pre["s_true_bank"][event_id]
                amplitude = pre["amplitude_bank"][event_id]
                snr = pre["snr_bank"][event_id]
                detected_mask = pre["detected_mask_bank"][event_id]

                obs, usable = build_tdoa_observation(
                    t=t,
                    detected_mask=detected_mask,
                    ref_strategy=cfg["REF_STRATEGY"],
                    weight_strategy=cfg["WEIGHT_STRATEGY"],
                    amplitude=amplitude,
                    snr=snr,
                    min_detect=int(cfg["MIN_DETECT"]),
                )
                if (not usable) or (obs is None):
                    n_failed_detection += 1
                    continue

                detected_counts_usable.append(obs["n_detect"])
                q_hat, ok, _ = estimate_surface_source_xy_step9(
                    obs=obs,
                    geo_model=geo_model,
                    L=L,
                    v_eff=cfg["V_EFF"],
                    search_grid_n=int(cfg["SEARCH_GRID_N"]),
                    n_multi_start=int(cfg["N_MULTI_START"]),
                    lsq_max_nfev=int(cfg["LEAST_SQUARES_MAX_NFEV"]),
                    lsq_tol=cfg["LEAST_SQUARES_TOL"],
                )
                if (not ok) or (q_hat is None):
                    n_failed_localization += 1
                    continue

                s_hat = surface_source_point(q_hat, h)
                err_3d.append(np.linalg.norm(s_hat - s_true))

            M = int(cfg["N_EVENTS"])
            n_usable = M - n_failed_detection
            mean_err, median_err, p95_err, n_valid = collapse_error_metrics(err_3d, penalty_err=penalty_err)
            solver_success_given_usable = np.nan if n_usable <= 0 else (n_valid / n_usable)

            row = {
                "seed": int(seed),
                "terrain": terrain,
                "mean_err_3d": mean_err,
                "median_err_3d": median_err,
                "p95_err_3d": p95_err,
                "valid_localization_rate": float(n_valid / M),
                "usable_event_rate": float(n_usable / M),
                "solver_success_given_usable": float(solver_success_given_usable) if np.isfinite(solver_success_given_usable) else np.nan,
                "failed_detection": int(n_failed_detection),
                "failed_localization": int(n_failed_localization),
                "mean_detected_count_usable": float(np.mean(detected_counts_usable)) if detected_counts_usable else np.nan,
                "delta_edge": float(delta_edge),
                "sigma_z": float(sigma_z),
                "deploy_resnorm": float(info["deploy_resnorm"]),
                "centroid_shift": float(info["centroid_shift"]),
                "orient_error_deg": float(info["orient_error_deg"]),
                "deploy_ok": bool(info["success"]),
                "terrain_meta_json": json.dumps(terrain_meta, ensure_ascii=False),
            }
            rows.append(row)

    rows = add_dgeom_star(rows, dgeom_stats=resolved_dgeom_stats, w_edge=cfg["DGEOM_W_EDGE"], w_height=cfg["DGEOM_W_HEIGHT"])
    terrain_summary = summarize_rows_by_terrain(rows)
    terrain_scores, J_universal = compute_objective_terms(terrain_summary, nan_penalty=float(cfg["OBJECTIVE_NAN_PENALTY"]))

    if not np.isfinite(J_universal):
        return make_penalty_result(cfg, terrain_kinds, reason="nonfinite_objective_after_summary", return_raw=return_raw, raw_rows=rows)

    result = {
        "J_universal": float(J_universal),
        "terrain_scores": terrain_scores,
        "config": cfg,
        "valid": True,
        "failure_reason": None,
        "dgeom_stats_mode": resolved_dgeom_stats.get("source", "unknown"),
        "resolved_dgeom_stats": {
            "edge_mean": float(resolved_dgeom_stats["edge_mean"]),
            "edge_std": float(resolved_dgeom_stats["edge_std"]),
            "height_mean": float(resolved_dgeom_stats["height_mean"]),
            "height_std": float(resolved_dgeom_stats["height_std"]),
        },
        "terrain_realizations": terrain_realizations,
    }
    if return_raw:
        result["raw_rows"] = rows
    return result


def evaluate_layout_theta(theta, **kwargs):
    return evaluate_layout_objective(theta_to_layout_xy(theta), **kwargs)
