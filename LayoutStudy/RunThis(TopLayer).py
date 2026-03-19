import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layout_objective_evaluator import (
    DEFAULT_CONFIG,
    evaluate_layout_objective,
    hull_area_2d,
    min_pairwise_distance,
)


@dataclass
class SampleResult:
    rank_key: float
    score: float
    J_universal: float
    valid: bool
    failure_reason: str
    layout_xy: np.ndarray
    terrain_scores: dict
    min_pairwise_sep: float
    hull_area: float


def random_feasible_layout(
    rng: np.random.Generator,
    n_nodes: int,
    L: float,
    boundary_margin: float,
    min_pairwise_sep: float,
    min_hull_area: float,
    max_tries: int = 20000,
) -> np.ndarray:
    lo = boundary_margin
    hi = L - boundary_margin
    if lo >= hi:
        raise ValueError("boundary margin too large for the given L")

    for _ in range(max_tries):
        pts = []
        inner_tries = 0
        while len(pts) < n_nodes and inner_tries < max_tries:
            p = rng.uniform(lo, hi, size=2)
            if all(np.linalg.norm(p - q) >= min_pairwise_sep for q in pts):
                pts.append(p)
            inner_tries += 1
        if len(pts) < n_nodes:
            continue
        layout = np.asarray(pts, dtype=float)
        area = hull_area_2d(layout)
        if area >= min_hull_area:
            return layout

    raise RuntimeError("failed to sample a feasible random layout; relax constraints or increase max_tries")


def flatten_layout(layout_xy: np.ndarray) -> dict:
    out = {}
    for i, (x, y) in enumerate(layout_xy):
        out[f"x{i}"] = float(x)
        out[f"y{i}"] = float(y)
    return out


def format_terrain_value(terrain_scores: dict, terrain: str, key: str):
    if terrain not in terrain_scores:
        return np.nan
    return terrain_scores[terrain].get(key, np.nan)


def evaluate_random_layouts(
    n_layouts: int,
    n_nodes: int,
    terrain_kinds: Tuple[str, ...],
    seed: Optional[int],
    eval_overrides: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[SampleResult]]:
    rng = np.random.default_rng(seed if seed is not None else None)

    L = float(eval_overrides.get("L", DEFAULT_CONFIG["L"]))
    boundary_margin = float(eval_overrides.get("VALIDATION_BOUNDARY_MARGIN", DEFAULT_CONFIG["VALIDATION_BOUNDARY_MARGIN"]))
    min_pairwise_sep = float(eval_overrides.get("VALIDATION_MIN_PAIRWISE_SEP", DEFAULT_CONFIG["VALIDATION_MIN_PAIRWISE_SEP"]))
    min_hull_area = float(eval_overrides.get("VALIDATION_MIN_HULL_AREA", DEFAULT_CONFIG["VALIDATION_MIN_HULL_AREA"]))

    all_results: List[SampleResult] = []

    for idx in range(n_layouts):
        layout_xy = random_feasible_layout(
            rng=rng,
            n_nodes=n_nodes,
            L=L,
            boundary_margin=boundary_margin,
            min_pairwise_sep=min_pairwise_sep,
            min_hull_area=min_hull_area,
        )

        result = evaluate_layout_objective(
            layout_xy,
            terrain_kinds=terrain_kinds,
            return_raw=False,
            **eval_overrides,
        )

        J = float(result["J_universal"])
        score = 1.0 / (1.0 + max(J, 0.0)) if math.isfinite(J) else 0.0
        all_results.append(
            SampleResult(
                rank_key=J,
                score=score,
                J_universal=J,
                valid=bool(result.get("valid", False)),
                failure_reason=result.get("failure_reason") or "",
                layout_xy=layout_xy.copy(),
                terrain_scores=result.get("terrain_scores", {}),
                min_pairwise_sep=min_pairwise_distance(layout_xy),
                hull_area=hull_area_2d(layout_xy),
            )
        )

        if (idx + 1) % 25 == 0 or idx == 0 or (idx + 1) == n_layouts:
            valid_count = sum(r.valid for r in all_results)
            current_best = min(r.J_universal for r in all_results)
            print(f"[{idx + 1:4d}/{n_layouts}] valid={valid_count:4d} best_J={current_best:.6f}")

    rows = []
    for i, r in enumerate(all_results, start=1):
        row = {
            "sample_id": i,
            "valid": r.valid,
            "failure_reason": r.failure_reason,
            "J_universal": r.J_universal,
            "display_score": r.score,
            "min_pairwise_sep": r.min_pairwise_sep,
            "hull_area": r.hull_area,
        }
        for terrain in terrain_kinds:
            row[f"{terrain}_J_t"] = format_terrain_value(r.terrain_scores, terrain, "J_t")
            row[f"{terrain}_mean_err"] = format_terrain_value(r.terrain_scores, terrain, "mean_err_3d_post")
            row[f"{terrain}_p95_err"] = format_terrain_value(r.terrain_scores, terrain, "p95_err_3d_post")
            row[f"{terrain}_valid_rate"] = format_terrain_value(r.terrain_scores, terrain, "valid_localization_rate_post")
            row[f"{terrain}_dgeom"] = format_terrain_value(r.terrain_scores, terrain, "dgeom_star")
        row.update(flatten_layout(r.layout_xy))
        row["layout_json"] = json.dumps(r.layout_xy.tolist())
        rows.append(row)

    df_all = pd.DataFrame(rows).sort_values(by=["J_universal", "display_score"], ascending=[True, False]).reset_index(drop=True)
    df_all.insert(0, "rank", np.arange(1, len(df_all) + 1))
    df_top10 = df_all.head(10).copy()

    top10_results = []
    for _, row in df_top10.iterrows():
        sample_idx = int(row["sample_id"]) - 1
        top10_results.append(all_results[sample_idx])

    return df_all, df_top10, top10_results


def save_top10_plot(top10_results: List[SampleResult], output_path: str, L: float):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    axes = np.ravel(axes)

    for rank, (ax, res) in enumerate(zip(axes, top10_results), start=1):
        xy = res.layout_xy
        ax.scatter(xy[:, 0], xy[:, 1], s=35)
        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"#{rank} J={res.J_universal:.4f}\nscore={res.score:.4f}", fontsize=10)
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(i), fontsize=8, ha="left", va="bottom")

    for ax in axes[len(top10_results):]:
        ax.axis("off")

    fig.suptitle("Top 10 random layouts (ranked by lowest J_universal)", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top10_npz(top10_results: List[SampleResult], output_path: str):
    arrays = {f"layout_rank_{i + 1}": res.layout_xy for i, res in enumerate(top10_results)}
    np.savez(output_path, **arrays)


def make_output_paths(output_dir: str, n_layouts: int):
    all_csv = os.path.join(output_dir, f"random_layout_{n_layouts}_all_results.csv")
    top10_csv = os.path.join(output_dir, f"random_layout_{n_layouts}_top10.csv")
    top10_png = os.path.join(output_dir, f"random_layout_{n_layouts}_top10.png")
    top10_npz = os.path.join(output_dir, f"random_layout_{n_layouts}_top10_layouts.npz")
    return all_csv, top10_csv, top10_png, top10_npz


def make_parser():
    parser = argparse.ArgumentParser(description="Randomly sample layouts and show the top 10.")
    parser.add_argument("--n-layouts", type=int, default=5000)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for layout sampling. Default: None, so each run is different.")
    parser.add_argument("--terrain-seed", type=int, default=None, help="Shared terrain seed for this run. Default: None, so each run uses a new random terrain batch.")
    parser.add_argument("--output-dir", type=str, default="Result", help="Directory to save outputs. Default is the current working directory.")
    parser.add_argument("--terrains", nargs="+", default=["flat", "hill", "rough"])
    parser.add_argument("--n-events", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--geo-grid-n", type=int, default=21)
    parser.add_argument("--search-grid-n", type=int, default=7)
    parser.add_argument("--n-multi-start", type=int, default=3)
    parser.add_argument("--rough-grid-n", type=int, default=65)
    parser.add_argument("--sigma-t", type=float, default=float(DEFAULT_CONFIG["SIGMA_T"]))
    return parser


def main():
    args = make_parser().parse_args()
    output_dir = os.path.abspath(args.output_dir or ".")
    os.makedirs(output_dir, exist_ok=True)

    terrain_seed = args.terrain_seed
    if terrain_seed is None:
        terrain_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])

    eval_overrides = {
        "N_EVENTS": int(args.n_events),
        "N_SEEDS": int(args.n_seeds),
        "GEO_GRID_N": int(args.geo_grid_n),
        "SEARCH_GRID_N": int(args.search_grid_n),
        "N_MULTI_START": int(args.n_multi_start),
        "ROUGH_GRID_N": int(args.rough_grid_n),
        "SIGMA_T": float(args.sigma_t),
        "TERRAIN_MASTER_SEED": int(terrain_seed),
        "HILL_RANDOMIZE": True,
        "ROUGH_RANDOMIZE": True,
    }

    print(f"layout_seed={args.seed}")
    print(f"terrain_seed={terrain_seed}")

    df_all, df_top10, top10_results = evaluate_random_layouts(
        n_layouts=int(args.n_layouts),
        n_nodes=int(args.n_nodes),
        terrain_kinds=tuple(args.terrains),
        seed=args.seed,
        eval_overrides=eval_overrides,
    )

    all_csv_path, top10_csv_path, plot_path, npz_path = make_output_paths(output_dir=output_dir, n_layouts=int(args.n_layouts))
    df_all.to_csv(all_csv_path, index=False)
    df_top10.to_csv(top10_csv_path, index=False)

    L = float(eval_overrides.get("L", DEFAULT_CONFIG["L"]))
    save_top10_plot(top10_results, plot_path, L=L)
    save_top10_npz(top10_results, npz_path)

    summary_cols = ["rank", "sample_id", "J_universal", "display_score", "min_pairwise_sep", "hull_area"]
    print("\nTop 10 layouts (best = lowest J_universal):")
    print(df_top10[summary_cols].to_string(index=False))
    print(f"\nSaved all results to: {all_csv_path}")
    print(f"Saved top 10 table to: {top10_csv_path}")
    print(f"Saved top 10 figure to: {plot_path}")
    print(f"Saved top 10 layouts npz to: {npz_path}")


if __name__ == "__main__":
    main()
