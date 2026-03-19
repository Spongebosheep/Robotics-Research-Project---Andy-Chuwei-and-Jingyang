import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layout_objective_evaluator import (
    DEFAULT_CONFIG,
    evaluate_layout_objective,
    hull_area_2d,
    min_pairwise_distance,
    theta_to_layout_xy,
    validate_layout_xy,
)


# ============================================================
# Helpers
# ============================================================


def auto_find_seed_csv(result_dir: str) -> str:
    candidates = [
        os.path.join(result_dir, "reeval_top50_summary.csv"),
        os.path.join(result_dir, "reeval_top20_summary.csv"),
        os.path.join(result_dir, "reeval_top10_summary.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    matches = glob.glob(os.path.join(result_dir, "random_layout_*_all_results.csv"))
    if matches:
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]

    raise FileNotFoundError(
        f"No seed CSV found under {os.path.abspath(result_dir)}. "
        f"Expected reeval_top*_summary.csv or random_layout_*_all_results.csv"
    )


def detect_sort_column(df: pd.DataFrame) -> Tuple[str, bool]:
    if "mean_J" in df.columns:
        return "mean_J", True
    if "J_universal" in df.columns:
        return "J_universal", True
    if "coarse_J" in df.columns:
        return "coarse_J", True
    if "display_score" in df.columns:
        return "display_score", False
    raise ValueError("Cannot determine ranking column from CSV.")


def load_seed_layouts(csv_path: str, top_k: int) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    df = pd.read_csv(csv_path)
    if "layout_json" not in df.columns:
        raise ValueError("Seed CSV must contain layout_json column.")

    sort_col, ascending = detect_sort_column(df)
    if "valid_repeat_rate" in df.columns:
        df = df[df["valid_repeat_rate"] > 0].copy()
    elif "valid" in df.columns:
        df = df[df["valid"].astype(bool)].copy()

    df = df.sort_values(by=[sort_col], ascending=[ascending]).reset_index(drop=True)
    top_k = min(int(top_k), len(df))
    if top_k <= 0:
        raise ValueError("No valid seed layouts available.")
    df = df.head(top_k).copy()

    layouts: List[np.ndarray] = []
    for s in df["layout_json"].tolist():
        arr = np.asarray(json.loads(s), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("layout_json must decode to shape (n_nodes, 2).")
        layouts.append(arr)

    return df, layouts


@dataclass
class EvalConfig:
    terrain_kinds: Tuple[str, ...]
    n_events: int
    n_seeds: int
    geo_grid_n: int
    search_grid_n: int
    n_multi_start: int
    rough_grid_n: int
    sigma_t: float
    terrain_master_seed: int
    mc_base_seed: int

    def to_overrides(self) -> dict:
        return {
            "N_EVENTS": int(self.n_events),
            "N_SEEDS": int(self.n_seeds),
            "GEO_GRID_N": int(self.geo_grid_n),
            "SEARCH_GRID_N": int(self.search_grid_n),
            "N_MULTI_START": int(self.n_multi_start),
            "ROUGH_GRID_N": int(self.rough_grid_n),
            "SIGMA_T": float(self.sigma_t),
            "TERRAIN_MASTER_SEED": int(self.terrain_master_seed),
            "MONTE_CARLO_BASE_SEED": int(self.mc_base_seed),
            "HILL_RANDOMIZE": True,
            "ROUGH_RANDOMIZE": True,
        }


# ============================================================
# Feasibility repair
# ============================================================


def clip_layout(layout_xy: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(layout_xy, lo, hi)


def repair_pairwise(layout_xy: np.ndarray, min_sep: float, lo: float, hi: float, rng: np.random.Generator, n_pass: int = 12) -> np.ndarray:
    x = layout_xy.copy()
    n = x.shape[0]
    for _ in range(n_pass):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dvec = x[j] - x[i]
                dist = float(np.linalg.norm(dvec))
                if dist >= min_sep:
                    continue
                if dist < 1e-12:
                    angle = rng.uniform(0.0, 2.0 * np.pi)
                    dvec = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                    dist = 1.0
                push = 0.5 * (min_sep - dist) * (dvec / dist)
                x[i] -= push
                x[j] += push
                changed = True
        x = clip_layout(x, lo, hi)
        if not changed:
            break
    return x


def repair_hull_area(layout_xy: np.ndarray, min_area: float, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    x = layout_xy.copy()
    area = hull_area_2d(x)
    if area >= min_area:
        return x

    c = np.mean(x, axis=0)
    centered = x - c
    for scale in np.linspace(1.05, 2.40, 20):
        cand = c + scale * centered
        cand = clip_layout(cand, lo, hi)
        if hull_area_2d(cand) >= min_area:
            return cand

    for jitter_scale in np.linspace(0.01, 0.08, 12):
        cand = x + rng.normal(scale=jitter_scale, size=x.shape)
        cand = clip_layout(cand, lo, hi)
        if hull_area_2d(cand) >= min_area:
            return cand
    return x


def repair_layout(layout_xy: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    L = float(cfg["L"])
    lo = float(cfg["VALIDATION_BOUNDARY_MARGIN"])
    hi = L - lo
    min_sep = float(cfg["VALIDATION_MIN_PAIRWISE_SEP"])
    min_area = float(cfg["VALIDATION_MIN_HULL_AREA"])

    x = theta_to_layout_xy(layout_xy).copy()
    x = clip_layout(x, lo, hi)
    x = repair_pairwise(x, min_sep=min_sep, lo=lo, hi=hi, rng=rng)
    x = repair_hull_area(x, min_area=min_area, lo=lo, hi=hi, rng=rng)
    x = repair_pairwise(x, min_sep=min_sep, lo=lo, hi=hi, rng=rng)
    x = clip_layout(x, lo, hi)
    return x


# ============================================================
# CMA-ES core
# ============================================================


class CMAES:
    def __init__(self, x0: np.ndarray, sigma0: float, rng: np.random.Generator, popsize: Optional[int] = None, C0: Optional[np.ndarray] = None):
        self.rng = rng
        self.m = np.asarray(x0, dtype=float).copy()
        self.n = self.m.size
        self.sigma = float(sigma0)
        self.lam = int(popsize) if popsize is not None else int(4 + math.floor(3 * np.log(self.n)))
        self.mu = self.lam // 2

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = float((np.sum(self.weights) ** 2) / np.sum(self.weights ** 2))

        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        self.C = np.eye(self.n) if C0 is None else np.asarray(C0, dtype=float).copy()
        self.C = 0.5 * (self.C + self.C.T) + 1e-12 * np.eye(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self._update_eigensystem()
        self.iteration = 0

    def _update_eigensystem(self):
        Csym = 0.5 * (self.C + self.C.T)
        evals, evecs = np.linalg.eigh(Csym)
        evals = np.maximum(evals, 1e-20)
        self.D = np.sqrt(evals)
        self.B = evecs
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T

    def ask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arz = self.rng.normal(size=(self.lam, self.n))
        ary = arz @ (self.B * self.D).T
        arx = self.m[None, :] + self.sigma * ary
        return arx, ary, arz

    def tell(self, arx: np.ndarray, fitness: np.ndarray):
        idx = np.argsort(fitness)
        xsel = arx[idx[:self.mu]]

        m_old = self.m.copy()
        ysel = (xsel - m_old[None, :]) / self.sigma
        y_w = np.sum(self.weights[:, None] * ysel, axis=0)
        self.m = m_old + self.sigma * y_w

        c_sigma_factor = math.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + c_sigma_factor * (self.invsqrtC @ y_w)

        norm_ps = np.linalg.norm(self.ps)
        hsig_num = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * (self.iteration + 1)))
        hsig = float(hsig_num / self.chi_n < (1.4 + 2 / (self.n + 1)))

        c_c_factor = math.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc = (1 - self.cc) * self.pc + hsig * c_c_factor * y_w

        artmp = ysel
        rank_mu = np.zeros_like(self.C)
        for w_i, y_i in zip(self.weights, artmp):
            rank_mu += w_i * np.outer(y_i, y_i)

        delta_hsig = (1 - hsig) * self.cc * (2 - self.cc)
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + delta_hsig * self.C)
            + self.cmu * rank_mu
        )
        self.C = 0.5 * (self.C + self.C.T)

        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chi_n - 1))
        self.iteration += 1

        if self.iteration % max(1, self.n // 2) == 0:
            self._update_eigensystem()


# ============================================================
# Objective wrapper
# ============================================================


class ObjectiveWrapper:
    def __init__(self, eval_cfg: EvalConfig, base_cfg: dict, seed_rng: np.random.Generator):
        self.eval_cfg = eval_cfg
        self.base_cfg = base_cfg
        self.seed_rng = seed_rng
        self.cache: Dict[bytes, Tuple[float, np.ndarray, bool, str]] = {}
        self.n_eval = 0

    def evaluate(self, theta: np.ndarray) -> Tuple[float, np.ndarray, bool, str]:
        layout_xy = theta_to_layout_xy(theta)
        repaired = repair_layout(layout_xy, self.base_cfg, self.seed_rng)
        key = np.round(repaired.ravel(), 6).astype(np.float64).tobytes()
        if key in self.cache:
            return self.cache[key]

        result = evaluate_layout_objective(
            repaired,
            terrain_kinds=self.eval_cfg.terrain_kinds,
            return_raw=False,
            **self.eval_cfg.to_overrides(),
        )
        self.n_eval += 1
        J = float(result["J_universal"])
        valid = bool(result.get("valid", False))
        reason = result.get("failure_reason") or ""
        out = (J, repaired.copy(), valid, reason)
        self.cache[key] = out
        return out


# ============================================================
# Optimization and reevaluation
# ============================================================


@dataclass
class RunResult:
    start_idx: int
    seed_metric: float
    seed_layout: np.ndarray
    best_layout: np.ndarray
    seed_J: float
    best_J_opt: float
    n_eval: int
    history: List[dict]


@dataclass
class ReevalResult:
    start_idx: int
    best_J_opt: float
    mean_J: float
    std_J: float
    ci95_low: float
    ci95_high: float
    best_layout: np.ndarray



def make_initial_cov(seed_layouts: Sequence[np.ndarray], floor: float = 1e-4) -> np.ndarray:
    thetas = np.array([x.ravel() for x in seed_layouts], dtype=float)
    n = thetas.shape[1]
    if thetas.shape[0] <= 1:
        return np.eye(n)
    cov = np.cov(thetas.T)
    cov = np.atleast_2d(cov)
    cov = 0.5 * (cov + cov.T)
    cov += floor * np.eye(n)
    scale = float(np.mean(np.diag(cov)))
    if not np.isfinite(scale) or scale <= 0:
        return np.eye(n)
    return cov / scale


def optimize_from_seed(
    start_idx: int,
    seed_layout: np.ndarray,
    seed_metric: float,
    eval_cfg: EvalConfig,
    base_cfg: dict,
    rng: np.random.Generator,
    max_gens: int,
    popsize: Optional[int],
    sigma0: float,
    init_cov: Optional[np.ndarray],
    patience: int,
) -> RunResult:
    obj = ObjectiveWrapper(eval_cfg=eval_cfg, base_cfg=base_cfg, seed_rng=rng)

    seed_J, seed_repaired, _, _ = obj.evaluate(seed_layout.ravel())
    cma = CMAES(x0=seed_repaired.ravel(), sigma0=sigma0, rng=rng, popsize=popsize, C0=init_cov)

    best_theta = seed_repaired.ravel().copy()
    best_J = float(seed_J)
    no_improve = 0
    history: List[dict] = []

    for gen in range(1, max_gens + 1):
        xs, _, _ = cma.ask()
        fit = np.empty(xs.shape[0], dtype=float)
        repaired_pop: List[np.ndarray] = []

        for i in range(xs.shape[0]):
            f_i, repaired_i, _, _ = obj.evaluate(xs[i])
            fit[i] = f_i
            repaired_pop.append(repaired_i.ravel())
            if f_i < best_J:
                best_J = float(f_i)
                best_theta = repaired_i.ravel().copy()
                no_improve = 0

        repaired_pop_arr = np.vstack(repaired_pop)
        cma.tell(repaired_pop_arr, fit)
        no_improve += 1

        gen_row = {
            "start_idx": start_idx,
            "generation": gen,
            "best_gen_J": float(np.min(fit)),
            "median_gen_J": float(np.median(fit)),
            "best_so_far_J": float(best_J),
            "sigma": float(cma.sigma),
            "n_eval_total": int(obj.n_eval),
        }
        history.append(gen_row)

        print(
            f"[start {start_idx:02d}] gen={gen:03d}/{max_gens} "
            f"seed_J={seed_J:.6f} gen_best={gen_row['best_gen_J']:.6f} best={best_J:.6f} sigma={cma.sigma:.4f}"
        )

        if gen_row["best_gen_J"] + 1e-12 < best_J + 1e-12:
            no_improve = 0

        if no_improve >= patience:
            print(f"[start {start_idx:02d}] early stop: no improvement for {patience} generations")
            break
        if cma.sigma < 1e-4:
            print(f"[start {start_idx:02d}] early stop: sigma too small")
            break
        condC = float(np.max(cma.D) / max(np.min(cma.D), 1e-20))
        if condC > 1e7:
            print(f"[start {start_idx:02d}] early stop: covariance ill-conditioned")
            break

    return RunResult(
        start_idx=start_idx,
        seed_metric=float(seed_metric),
        seed_layout=seed_repaired.copy(),
        best_layout=theta_to_layout_xy(best_theta).copy(),
        seed_J=float(seed_J),
        best_J_opt=float(best_J),
        n_eval=int(obj.n_eval),
        history=history,
    )



def summarize(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.inf, "std": math.inf, "ci95_low": math.inf, "ci95_high": math.inf}
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    sem = float(std / np.sqrt(finite.size)) if finite.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - 1.96 * sem),
        "ci95_high": float(mean + 1.96 * sem),
    }



def reevaluate_layouts(
    run_results: Sequence[RunResult],
    terrain_kinds: Tuple[str, ...],
    n_repeats: int,
    n_events: int,
    n_seeds: int,
    geo_grid_n: int,
    search_grid_n: int,
    n_multi_start: int,
    rough_grid_n: int,
    sigma_t: float,
    terrain_seed_base: int,
    mc_seed_base: int,
) -> Tuple[pd.DataFrame, List[ReevalResult]]:
    rows = []
    reeval_results: List[ReevalResult] = []
    for rr in run_results:
        vals = []
        for rep in range(n_repeats):
            result = evaluate_layout_objective(
                rr.best_layout,
                terrain_kinds=terrain_kinds,
                return_raw=False,
                N_EVENTS=int(n_events),
                N_SEEDS=int(n_seeds),
                GEO_GRID_N=int(geo_grid_n),
                SEARCH_GRID_N=int(search_grid_n),
                N_MULTI_START=int(n_multi_start),
                ROUGH_GRID_N=int(rough_grid_n),
                SIGMA_T=float(sigma_t),
                TERRAIN_MASTER_SEED=int(terrain_seed_base + rep * 100003 + rr.start_idx * 1009),
                MONTE_CARLO_BASE_SEED=int(mc_seed_base + rep * 100003 + rr.start_idx * 1009),
                HILL_RANDOMIZE=True,
                ROUGH_RANDOMIZE=True,
            )
            vals.append(float(result["J_universal"]))
        stats = summarize(vals)
        rows.append(
            {
                "start_idx": rr.start_idx,
                "seed_J": rr.seed_J,
                "best_J_opt": rr.best_J_opt,
                "reeval_mean_J": stats["mean"],
                "reeval_std_J": stats["std"],
                "reeval_ci95_low": stats["ci95_low"],
                "reeval_ci95_high": stats["ci95_high"],
                "layout_json": json.dumps(rr.best_layout.tolist()),
            }
        )
        reeval_results.append(
            ReevalResult(
                start_idx=rr.start_idx,
                best_J_opt=rr.best_J_opt,
                mean_J=stats["mean"],
                std_J=stats["std"],
                ci95_low=stats["ci95_low"],
                ci95_high=stats["ci95_high"],
                best_layout=rr.best_layout.copy(),
            )
        )
    df = pd.DataFrame(rows).sort_values(by=["reeval_mean_J", "reeval_ci95_high", "best_J_opt"], ascending=[True, True, True]).reset_index(drop=True)
    df.insert(0, "final_rank", np.arange(1, len(df) + 1))
    return df, reeval_results


# ============================================================
# Output
# ============================================================


def save_layout_grid(layouts: Sequence[np.ndarray], titles: Sequence[str], output_path: str, L: float):
    n = len(layouts)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, xy, title in zip(axes, layouts, titles):
        ax.scatter(xy[:, 0], xy[:, 1], s=35)
        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=10)
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(i), fontsize=8, ha="left", va="bottom")
    for ax in axes[len(layouts):]:
        ax.axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def build_parser():
    p = argparse.ArgumentParser(description="Multi-start CMA-ES from best random-search layouts.")
    p.add_argument("--seed-csv", type=str, default=None, help="Default: auto-find reeval_top50_summary.csv or latest random_layout_*_all_results.csv")
    p.add_argument("--result-dir", type=str, default="Result")
    p.add_argument("--top-k-seeds", type=int, default=5, help="Use best K layouts from seed CSV as CMA-ES starts")
    p.add_argument("--terrains", nargs="+", default=["flat", "hill", "rough"])

    p.add_argument("--max-gens", type=int, default=25)
    p.add_argument("--popsize", type=int, default=None)
    p.add_argument("--sigma0", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260318)

    # optimization-time evaluator: moderately cheap, low-noise due to fixed seeds
    p.add_argument("--n-events", type=int, default=100)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--geo-grid-n", type=int, default=21)
    p.add_argument("--search-grid-n", type=int, default=7)
    p.add_argument("--n-multi-start", type=int, default=3)
    p.add_argument("--rough-grid-n", type=int, default=65)
    p.add_argument("--sigma-t", type=float, default=float(DEFAULT_CONFIG["SIGMA_T"]))
    p.add_argument("--terrain-seed", type=int, default=310001)
    p.add_argument("--mc-base-seed", type=int, default=910001)

    # final reevaluation after optimization
    p.add_argument("--final-reeval-repeats", type=int, default=3)
    p.add_argument("--final-n-events", type=int, default=200)
    p.add_argument("--final-n-seeds", type=int, default=5)
    p.add_argument("--final-geo-grid-n", type=int, default=21)
    p.add_argument("--final-search-grid-n", type=int, default=7)
    p.add_argument("--final-n-multi-start", type=int, default=3)
    p.add_argument("--final-rough-grid-n", type=int, default=65)
    p.add_argument("--final-terrain-seed-base", type=int, default=510001)
    p.add_argument("--final-mc-seed-base", type=int, default=1110001)
    return p



def main():
    args = build_parser().parse_args()
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    seed_csv = args.seed_csv or auto_find_seed_csv(result_dir)
    seed_csv = os.path.abspath(seed_csv)
    seed_df, seed_layouts = load_seed_layouts(seed_csv, top_k=int(args.top_k_seeds))

    terrain_kinds = tuple(args.terrains)
    rng = np.random.default_rng(int(args.seed))
    base_cfg = dict(DEFAULT_CONFIG)
    init_cov = make_initial_cov(seed_layouts)

    eval_cfg = EvalConfig(
        terrain_kinds=terrain_kinds,
        n_events=int(args.n_events),
        n_seeds=int(args.n_seeds),
        geo_grid_n=int(args.geo_grid_n),
        search_grid_n=int(args.search_grid_n),
        n_multi_start=int(args.n_multi_start),
        rough_grid_n=int(args.rough_grid_n),
        sigma_t=float(args.sigma_t),
        terrain_master_seed=int(args.terrain_seed),
        mc_base_seed=int(args.mc_base_seed),
    )

    sort_col, _ = detect_sort_column(seed_df)
    run_results: List[RunResult] = []
    history_rows: List[dict] = []

    print(f"Seed CSV: {seed_csv}")
    print(f"Using top {len(seed_layouts)} seed layouts as CMA-ES starts")
    print(f"Optimization evaluator: n_events={args.n_events}, n_seeds={args.n_seeds}, terrains={terrain_kinds}")
    print(f"CMA-ES: max_gens={args.max_gens}, popsize={args.popsize}, sigma0={args.sigma0}")

    for i, layout in enumerate(seed_layouts, start=1):
        seed_metric = float(seed_df.iloc[i - 1][sort_col])
        print("\n" + "=" * 80)
        print(f"Start {i}/{len(seed_layouts)} | seed metric ({sort_col}) = {seed_metric:.6f}")
        print("=" * 80)
        rr = optimize_from_seed(
            start_idx=i,
            seed_layout=layout,
            seed_metric=seed_metric,
            eval_cfg=eval_cfg,
            base_cfg=base_cfg,
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
            max_gens=int(args.max_gens),
            popsize=args.popsize,
            sigma0=float(args.sigma0),
            init_cov=init_cov,
            patience=int(args.patience),
        )
        run_results.append(rr)
        history_rows.extend(rr.history)

    # sort by optimization J
    run_results_sorted = sorted(run_results, key=lambda r: (r.best_J_opt, r.seed_J))

    best_rows = []
    for rr in run_results_sorted:
        best_rows.append(
            {
                "opt_rank": len(best_rows) + 1,
                "start_idx": rr.start_idx,
                "seed_metric": rr.seed_metric,
                "seed_J": rr.seed_J,
                "best_J_opt": rr.best_J_opt,
                "improvement": rr.seed_J - rr.best_J_opt,
                "n_eval": rr.n_eval,
                "layout_json": json.dumps(rr.best_layout.tolist()),
            }
        )
    df_best = pd.DataFrame(best_rows)
    df_hist = pd.DataFrame(history_rows)

    # final reevaluation on the optimized elites
    df_final, reeval_results = reevaluate_layouts(
        run_results=run_results_sorted,
        terrain_kinds=terrain_kinds,
        n_repeats=int(args.final_reeval_repeats),
        n_events=int(args.final_n_events),
        n_seeds=int(args.final_n_seeds),
        geo_grid_n=int(args.final_geo_grid_n),
        search_grid_n=int(args.final_search_grid_n),
        n_multi_start=int(args.final_n_multi_start),
        rough_grid_n=int(args.final_rough_grid_n),
        sigma_t=float(args.sigma_t),
        terrain_seed_base=int(args.final_terrain_seed_base),
        mc_seed_base=int(args.final_mc_seed_base),
    )

    # outputs
    best_csv = os.path.join(result_dir, "cmaes_multistart_best.csv")
    hist_csv = os.path.join(result_dir, "cmaes_multistart_history.csv")
    final_csv = os.path.join(result_dir, "cmaes_multistart_final_reeval.csv")
    best_png = os.path.join(result_dir, "cmaes_multistart_best_layouts.png")
    final_png = os.path.join(result_dir, "cmaes_multistart_final_layouts.png")
    best_npz = os.path.join(result_dir, "cmaes_multistart_best_layouts.npz")
    meta_json = os.path.join(result_dir, "cmaes_multistart_meta.json")

    df_best.to_csv(best_csv, index=False)
    df_hist.to_csv(hist_csv, index=False)
    df_final.to_csv(final_csv, index=False)

    L = float(DEFAULT_CONFIG["L"])
    save_layout_grid(
        [rr.best_layout for rr in run_results_sorted],
        [f"start {rr.start_idx} | optJ={rr.best_J_opt:.4f}" for rr in run_results_sorted],
        best_png,
        L=L,
    )
    reeval_map = {rr.start_idx: rr for rr in reeval_results}
    final_sorted_layouts = []
    final_titles = []
    for _, row in df_final.iterrows():
        start_idx = int(row["start_idx"])
        rr = reeval_map[start_idx]
        final_sorted_layouts.append(rr.best_layout)
        final_titles.append(f"rank {int(row['final_rank'])} | meanJ={float(row['reeval_mean_J']):.4f}")
    save_layout_grid(final_sorted_layouts, final_titles, final_png, L=L)
    np.savez(best_npz, **{f"layout_start_{rr.start_idx}": rr.best_layout for rr in run_results_sorted})

    meta = {
        "seed_csv": seed_csv,
        "top_k_seeds": int(args.top_k_seeds),
        "terrain_kinds": list(terrain_kinds),
        "max_gens": int(args.max_gens),
        "popsize": None if args.popsize is None else int(args.popsize),
        "sigma0": float(args.sigma0),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "opt_eval": eval_cfg.to_overrides(),
        "final_reeval": {
            "repeats": int(args.final_reeval_repeats),
            "n_events": int(args.final_n_events),
            "n_seeds": int(args.final_n_seeds),
            "geo_grid_n": int(args.final_geo_grid_n),
            "search_grid_n": int(args.final_search_grid_n),
            "n_multi_start": int(args.final_n_multi_start),
            "rough_grid_n": int(args.final_rough_grid_n),
            "terrain_seed_base": int(args.final_terrain_seed_base),
            "mc_seed_base": int(args.final_mc_seed_base),
        },
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nOptimization best (on fixed optimization seeds):")
    print(df_best[["opt_rank", "start_idx", "seed_J", "best_J_opt", "improvement", "n_eval"]].to_string(index=False))
    print("\nFinal ranking after reevaluation:")
    print(df_final[["final_rank", "start_idx", "seed_J", "best_J_opt", "reeval_mean_J", "reeval_std_J"]].to_string(index=False))
    print(f"\nSaved best-per-start CSV: {best_csv}")
    print(f"Saved generation history CSV: {hist_csv}")
    print(f"Saved final reevaluation CSV: {final_csv}")
    print(f"Saved optimization layout figure: {best_png}")
    print(f"Saved final ranking layout figure: {final_png}")
    print(f"Saved layouts NPZ: {best_npz}")
    print(f"Saved meta JSON: {meta_json}")


if __name__ == "__main__":
    main()
