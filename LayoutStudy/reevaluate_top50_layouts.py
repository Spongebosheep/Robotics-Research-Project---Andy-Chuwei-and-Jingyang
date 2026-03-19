import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layout_objective_evaluator import DEFAULT_CONFIG, evaluate_layout_objective


@dataclass
class ReevalResult:
    original_rank: int
    sample_id: int
    layout_xy: np.ndarray
    coarse_J: float
    coarse_display_score: float
    mean_J: float
    std_J: float
    median_J: float
    p95_J: float
    sem_J: float
    ci95_low: float
    ci95_high: float
    valid_repeat_rate: float
    mean_valid_layout_flag: float
    failure_count: int
    repeat_Js: List[float]
    repeat_valid_flags: List[bool]
    repeat_failure_reasons: List[str]
    terrain_metric_means: Dict[str, float]


def auto_find_input_csv(result_dir: str) -> str:
    pattern = os.path.join(result_dir, "random_layout_*_all_results.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No coarse-search CSV found under: {os.path.abspath(result_dir)}\n"
            f"Expected something like: {pattern}"
        )
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def load_layout_xy_from_row(row: pd.Series) -> np.ndarray:
    if "layout_json" in row and pd.notna(row["layout_json"]):
        arr = np.asarray(json.loads(row["layout_json"]), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr

    xy_cols = [c for c in row.index if c.startswith("x") or c.startswith("y")]
    if xy_cols:
        x_cols = sorted([c for c in row.index if c.startswith("x")], key=lambda s: int(s[1:]))
        y_cols = sorted([c for c in row.index if c.startswith("y")], key=lambda s: int(s[1:]))
        if len(x_cols) != len(y_cols):
            raise ValueError("Mismatched x/y columns in coarse CSV.")
        xy = np.column_stack([[float(row[c]) for c in x_cols], [float(row[c]) for c in y_cols]])
        return xy

    raise ValueError("Cannot recover layout coordinates from row; need layout_json or x*/y* columns.")


def make_repeat_overrides(args: argparse.Namespace, repeat_idx: int) -> dict:
    terrain_seed = int(args.terrain_master_base_seed + repeat_idx * args.terrain_master_stride)
    mc_seed = int(args.mc_base_seed + repeat_idx * args.mc_base_seed_stride)
    return {
        "N_EVENTS": int(args.n_events),
        "N_SEEDS": int(args.n_seeds),
        "GEO_GRID_N": int(args.geo_grid_n),
        "SEARCH_GRID_N": int(args.search_grid_n),
        "N_MULTI_START": int(args.n_multi_start),
        "ROUGH_GRID_N": int(args.rough_grid_n),
        "SIGMA_T": float(args.sigma_t),
        "TERRAIN_MASTER_SEED": terrain_seed,
        "MONTE_CARLO_BASE_SEED": mc_seed,
        "HILL_RANDOMIZE": True,
        "ROUGH_RANDOMIZE": True,
    }


def terrain_summary_to_flat(terrain_scores: dict, terrain_kinds: Tuple[str, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for terrain in terrain_kinds:
        stat = terrain_scores.get(terrain, {})
        out[f"{terrain}_J_t"] = float(stat.get("J_t", np.nan))
        out[f"{terrain}_mean_err"] = float(stat.get("mean_err_3d_post", np.nan))
        out[f"{terrain}_median_err"] = float(stat.get("median_err_3d_post", np.nan))
        out[f"{terrain}_p95_err"] = float(stat.get("p95_err_3d_post", np.nan))
        out[f"{terrain}_valid_rate"] = float(stat.get("valid_localization_rate_post", np.nan))
        out[f"{terrain}_dgeom"] = float(stat.get("dgeom_star", np.nan))
        out[f"{terrain}_usable_event_rate"] = float(stat.get("usable_event_rate", np.nan))
    return out


def summarize_numbers(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "mean": float("inf"),
            "std": float("inf"),
            "median": float("inf"),
            "p95": float("inf"),
            "sem": float("inf"),
            "ci95_low": float("inf"),
            "ci95_high": float("inf"),
        }
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    median = float(np.median(finite))
    p95 = float(np.percentile(finite, 95))
    sem = float(std / np.sqrt(finite.size)) if finite.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "p95": p95,
        "sem": sem,
        "ci95_low": float(mean - 1.96 * sem),
        "ci95_high": float(mean + 1.96 * sem),
    }


def reevaluate_one_layout(
    row: pd.Series,
    args: argparse.Namespace,
    terrain_kinds: Tuple[str, ...],
) -> Tuple[ReevalResult, List[dict]]:
    layout_xy = load_layout_xy_from_row(row)
    original_rank = int(row["rank"]) if "rank" in row else -1
    sample_id = int(row["sample_id"]) if "sample_id" in row else -1
    coarse_J = float(row["J_universal"])
    coarse_display_score = float(row["display_score"]) if "display_score" in row else (1.0 / (1.0 + max(coarse_J, 0.0)))

    repeat_Js: List[float] = []
    repeat_valid_flags: List[bool] = []
    repeat_failure_reasons: List[str] = []
    detail_rows: List[dict] = []
    terrain_flat_rows: List[Dict[str, float]] = []

    for repeat_idx in range(int(args.n_repeats)):
        overrides = make_repeat_overrides(args, repeat_idx)
        result = evaluate_layout_objective(
            layout_xy,
            terrain_kinds=terrain_kinds,
            return_raw=False,
            **overrides,
        )
        J = float(result.get("J_universal", np.nan))
        valid_flag = bool(result.get("valid", False))
        failure_reason = result.get("failure_reason") or ""
        terrain_scores = result.get("terrain_scores", {})
        terrain_flat = terrain_summary_to_flat(terrain_scores, terrain_kinds)
        terrain_flat_rows.append(terrain_flat)

        repeat_Js.append(J)
        repeat_valid_flags.append(valid_flag)
        repeat_failure_reasons.append(failure_reason)

        detail_row = {
            "original_rank": original_rank,
            "sample_id": sample_id,
            "repeat_idx": repeat_idx + 1,
            "coarse_J": coarse_J,
            "reeval_J": J,
            "valid": valid_flag,
            "failure_reason": failure_reason,
            "terrain_master_seed": overrides["TERRAIN_MASTER_SEED"],
            "mc_base_seed": overrides["MONTE_CARLO_BASE_SEED"],
            "n_events": overrides["N_EVENTS"],
            "n_seeds": overrides["N_SEEDS"],
            "layout_json": json.dumps(layout_xy.tolist()),
        }
        detail_row.update(terrain_flat)
        detail_rows.append(detail_row)

    stats = summarize_numbers(repeat_Js)
    terrain_metric_means: Dict[str, float] = {}
    if terrain_flat_rows:
        terrain_df = pd.DataFrame(terrain_flat_rows)
        terrain_metric_means = {k: float(terrain_df[k].mean()) for k in terrain_df.columns}

    summary = ReevalResult(
        original_rank=original_rank,
        sample_id=sample_id,
        layout_xy=layout_xy.copy(),
        coarse_J=coarse_J,
        coarse_display_score=coarse_display_score,
        mean_J=stats["mean"],
        std_J=stats["std"],
        median_J=stats["median"],
        p95_J=stats["p95"],
        sem_J=stats["sem"],
        ci95_low=stats["ci95_low"],
        ci95_high=stats["ci95_high"],
        valid_repeat_rate=float(np.mean(repeat_valid_flags)) if repeat_valid_flags else 0.0,
        mean_valid_layout_flag=float(np.mean(repeat_valid_flags)) if repeat_valid_flags else 0.0,
        failure_count=int(sum(0 if flag else 1 for flag in repeat_valid_flags)),
        repeat_Js=repeat_Js,
        repeat_valid_flags=repeat_valid_flags,
        repeat_failure_reasons=repeat_failure_reasons,
        terrain_metric_means=terrain_metric_means,
    )
    return summary, detail_rows


def make_summary_row(res: ReevalResult) -> dict:
    row = {
        "original_rank": res.original_rank,
        "sample_id": res.sample_id,
        "coarse_J": res.coarse_J,
        "coarse_display_score": res.coarse_display_score,
        "mean_J": res.mean_J,
        "std_J": res.std_J,
        "median_J": res.median_J,
        "p95_J": res.p95_J,
        "sem_J": res.sem_J,
        "ci95_low": res.ci95_low,
        "ci95_high": res.ci95_high,
        "valid_repeat_rate": res.valid_repeat_rate,
        "failure_count": res.failure_count,
        "layout_json": json.dumps(res.layout_xy.tolist()),
    }
    row.update(res.terrain_metric_means)
    return row


def save_top10_plot(results_sorted: List[ReevalResult], output_path: str, L: float):
    top = results_sorted[:10]
    n_show = len(top)
    ncols = 5
    nrows = max(1, math.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for rank, (ax, res) in enumerate(zip(axes, top), start=1):
        xy = res.layout_xy
        ax.scatter(xy[:, 0], xy[:, 1], s=35)
        ax.set_xlim(0.0, L)
        ax.set_ylim(0.0, L)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"#{rank} meanJ={res.mean_J:.4f}\nstd={res.std_J:.4f}  coarse={res.coarse_J:.4f}",
            fontsize=10,
        )
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(i), fontsize=8, ha="left", va="bottom")

    for ax in axes[n_show:]:
        ax.axis("off")

    fig.suptitle("Top 10 reevaluated layouts (ranked by mean J)", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_topk_npz(results_sorted: List[ReevalResult], top_k: int, output_path: str):
    arrays = {f"layout_rank_{i + 1}": res.layout_xy for i, res in enumerate(results_sorted[:top_k])}
    np.savez(output_path, **arrays)


def make_output_paths(output_dir: str, top_k: int):
    summary_csv = os.path.join(output_dir, f"reeval_top{top_k}_summary.csv")
    detail_csv = os.path.join(output_dir, f"reeval_top{top_k}_detail.csv")
    top10_png = os.path.join(output_dir, f"reeval_top{top_k}_top10.png")
    topk_npz = os.path.join(output_dir, f"reeval_top{top_k}_layouts.npz")
    meta_json = os.path.join(output_dir, f"reeval_top{top_k}_meta.json")
    return summary_csv, detail_csv, top10_png, topk_npz, meta_json


def build_parser():
    parser = argparse.ArgumentParser(description="Reevaluate top-K layouts from an existing coarse-search CSV.")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to coarse-search CSV. Default: auto-find latest Result/random_layout_*_all_results.csv")
    parser.add_argument("--result-dir", type=str, default="Result", help="Directory containing coarse CSV and for saving reevaluation outputs.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--terrains", nargs="+", default=["flat", "hill", "rough"])

    parser.add_argument("--n-events", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=5, help="Per repeat. One call to evaluator uses this many seeds and terrain realizations.")
    parser.add_argument("--n-repeats", type=int, default=2, help="Independent reevaluation repeats. Total effective realizations ~= n_seeds * n_repeats.")

    parser.add_argument("--geo-grid-n", type=int, default=21)
    parser.add_argument("--search-grid-n", type=int, default=7)
    parser.add_argument("--n-multi-start", type=int, default=3)
    parser.add_argument("--rough-grid-n", type=int, default=65)
    parser.add_argument("--sigma-t", type=float, default=float(DEFAULT_CONFIG["SIGMA_T"]))

    parser.add_argument("--terrain-master-base-seed", type=int, default=310001)
    parser.add_argument("--terrain-master-stride", type=int, default=100003)
    parser.add_argument("--mc-base-seed", type=int, default=910001)
    parser.add_argument("--mc-base-seed-stride", type=int, default=100003)
    return parser


def main():
    args = build_parser().parse_args()
    result_dir = os.path.abspath(args.result_dir or ".")
    os.makedirs(result_dir, exist_ok=True)

    input_csv = args.input_csv if args.input_csv else auto_find_input_csv(result_dir)
    input_csv = os.path.abspath(input_csv)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "J_universal" not in df.columns:
        raise ValueError("Input CSV must contain J_universal column.")

    if "valid" in df.columns:
        df = df[df["valid"].astype(bool)].copy()
    df = df.sort_values(by=["J_universal"], ascending=[True]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid layouts found in the input CSV.")

    top_k = min(int(args.top_k), len(df))
    df_top = df.head(top_k).copy()
    terrain_kinds = tuple(args.terrains)

    print(f"Using coarse CSV: {input_csv}")
    print(f"Taking top {top_k} layouts from coarse search")
    print(f"High-precision config: n_events={args.n_events}, n_seeds={args.n_seeds}, n_repeats={args.n_repeats}")
    print(f"Effective independent reevaluation blocks ~= {args.n_seeds * args.n_repeats}")

    summary_results: List[ReevalResult] = []
    detail_rows_all: List[dict] = []

    for i, (_, row) in enumerate(df_top.iterrows(), start=1):
        summary, detail_rows = reevaluate_one_layout(row=row, args=args, terrain_kinds=terrain_kinds)
        summary_results.append(summary)
        detail_rows_all.extend(detail_rows)

        print(
            f"[{i:3d}/{top_k}] coarse_rank={summary.original_rank:4d} "
            f"sample_id={summary.sample_id:4d} coarse_J={summary.coarse_J:.6f} "
            f"mean_J={summary.mean_J:.6f} std={summary.std_J:.6f}"
        )

    summary_rows = [make_summary_row(r) for r in summary_results]
    df_summary = pd.DataFrame(summary_rows).sort_values(
        by=["mean_J", "ci95_high", "std_J", "coarse_J"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    df_summary.insert(0, "reeval_rank", np.arange(1, len(df_summary) + 1))

    id_to_result = {(r.original_rank, r.sample_id): r for r in summary_results}
    results_sorted: List[ReevalResult] = []
    for _, row in df_summary.iterrows():
        key = (int(row["original_rank"]), int(row["sample_id"]))
        results_sorted.append(id_to_result[key])

    df_detail = pd.DataFrame(detail_rows_all)
    if not df_detail.empty:
        rank_map = {
            (r.original_rank, r.sample_id): new_rank
            for new_rank, r in enumerate(results_sorted, start=1)
        }
        df_detail.insert(
            0,
            "reeval_rank",
            [rank_map[(int(r.original_rank), int(r.sample_id))] for r in df_detail.itertuples(index=False)],
        )
        df_detail = df_detail.sort_values(by=["reeval_rank", "repeat_idx"]).reset_index(drop=True)

    summary_csv, detail_csv, top10_png, topk_npz, meta_json = make_output_paths(result_dir, top_k)
    df_summary.to_csv(summary_csv, index=False)
    df_detail.to_csv(detail_csv, index=False)

    L = float(DEFAULT_CONFIG["L"])
    save_top10_plot(results_sorted, top10_png, L=L)
    save_topk_npz(results_sorted, top_k=top_k, output_path=topk_npz)

    meta = {
        "input_csv": input_csv,
        "result_dir": result_dir,
        "top_k": top_k,
        "terrains": list(terrain_kinds),
        "n_events": int(args.n_events),
        "n_seeds": int(args.n_seeds),
        "n_repeats": int(args.n_repeats),
        "effective_blocks": int(args.n_seeds * args.n_repeats),
        "geo_grid_n": int(args.geo_grid_n),
        "search_grid_n": int(args.search_grid_n),
        "n_multi_start": int(args.n_multi_start),
        "rough_grid_n": int(args.rough_grid_n),
        "sigma_t": float(args.sigma_t),
        "terrain_master_base_seed": int(args.terrain_master_base_seed),
        "terrain_master_stride": int(args.terrain_master_stride),
        "mc_base_seed": int(args.mc_base_seed),
        "mc_base_seed_stride": int(args.mc_base_seed_stride),
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nTop 10 after reevaluation:")
    cols = [
        "reeval_rank",
        "original_rank",
        "sample_id",
        "coarse_J",
        "mean_J",
        "std_J",
        "ci95_low",
        "ci95_high",
        "valid_repeat_rate",
    ]
    print(df_summary[cols].head(10).to_string(index=False))

    print(f"\nSaved summary CSV to: {summary_csv}")
    print(f"Saved detail CSV to:  {detail_csv}")
    print(f"Saved top10 figure to: {top10_png}")
    print(f"Saved top{top_k} layouts npz to: {topk_npz}")
    print(f"Saved meta json to: {meta_json}")


if __name__ == "__main__":
    main()
