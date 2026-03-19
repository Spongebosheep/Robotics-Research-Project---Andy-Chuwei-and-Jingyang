# -*- coding: utf-8 -*-
"""
单独评估一个手工对称 layout 的脚本。

默认 layout 取自用户给出的示意图：
- 外层 4 个点接近正方形四角
- 内层 4 个点接近中心菱形

脚本会：
1) 调用 layout_objective_evaluator.py 里的 evaluate_layout_objective
2) 输出 J_universal 和各 terrain 的 J_t
3) 保存原始逐 seed 结果 CSV
4) 保存布局图 PNG
5) 保存 summary JSON

快速预览：
python run_symmetric_layout_candidate.py --quick

完整评估：
python run_symmetric_layout_candidate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layout_objective_evaluator import evaluate_layout_objective


# 根据你发的图大致读出来的 8 个节点坐标，已归一化到 [0, 1]×[0, 1]
DEFAULT_LAYOUT = np.array(
    [
        [0.1502, 0.8352],  # outer top-left
        [0.8513, 0.8310],  # outer top-right
        [0.4988, 0.7016],  # inner top
        [0.2990, 0.4959],  # inner left
        [0.7241, 0.4867],  # inner right
        [0.5078, 0.2888],  # inner bottom
        [0.1515, 0.1477],  # outer bottom-left
        [0.8525, 0.1438],  # outer bottom-right
    ],
    dtype=float,
)


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def polygon_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    nxt = np.roll(points, -1, axis=0)
    return float(np.sum(np.linalg.norm(nxt - points, axis=1)))


def extract_geometry_features(layout_xy: np.ndarray, L: float = 1.0) -> Dict[str, float]:
    xy = np.asarray(layout_xy, dtype=float)
    n = xy.shape[0]
    centroid = xy.mean(axis=0)
    r = np.linalg.norm(xy - centroid, axis=1)

    D = pairwise_distances(xy)
    pair = D[np.triu_indices(n, k=1)]
    D_no_diag = D + np.eye(n) * 1e9
    nn = D_no_diag.min(axis=1)

    dist_left = xy[:, 0]
    dist_right = L - xy[:, 0]
    dist_bottom = xy[:, 1]
    dist_top = L - xy[:, 1]
    dist_boundary = np.min(np.column_stack([dist_left, dist_right, dist_bottom, dist_top]), axis=1)

    angles = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
    angles = np.sort((angles + 2.0 * np.pi) % (2.0 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])

    cov = np.cov(xy.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    return {
        "n_nodes": int(n),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "mean_radius": float(np.mean(r)),
        "std_radius": float(np.std(r)),
        "min_radius": float(np.min(r)),
        "max_radius": float(np.max(r)),
        "min_pairwise_dist": float(np.min(pair)),
        "mean_pairwise_dist": float(np.mean(pair)),
        "nearest_neighbor_mean": float(np.mean(nn)),
        "nearest_neighbor_std": float(np.std(nn)),
        "mean_dist_to_boundary": float(np.mean(dist_boundary)),
        "min_dist_to_boundary": float(np.min(dist_boundary)),
        "fraction_near_boundary_0p1L": float(np.mean(dist_boundary < 0.10 * L)),
        "fraction_near_boundary_0p15L": float(np.mean(dist_boundary < 0.15 * L)),
        "angular_gap_mean": float(np.mean(gaps)),
        "angular_gap_std": float(np.std(gaps)),
        "max_angular_gap": float(np.max(gaps)),
        "lambda1": float(eigvals[0]),
        "lambda2": float(eigvals[1]),
        "anisotropy": float(eigvals[0] / max(eigvals[1], 1e-12)),
    }


def build_candidate_layout(layout_scale: float = 1.0, center: float = 0.5) -> np.ndarray:
    xy = DEFAULT_LAYOUT.copy()
    if layout_scale != 1.0:
        xy = (xy - center) * layout_scale + center
    return xy


def plot_layout(layout_xy: np.ndarray, outpath: Path, L: float = 1.0) -> None:
    xy = np.asarray(layout_xy, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    boundary = np.array([[0, 0], [L, 0], [L, L], [0, L], [0, 0]], dtype=float)
    ax.plot(boundary[:, 0], boundary[:, 1], linewidth=2)

    # 画辅助线：外框 + 内菱形，只是为了接近你的示意图
    outer_idx = [0, 1, 7, 6, 0]
    inner_idx = [2, 4, 5, 3, 2]
    ax.plot(xy[outer_idx, 0], xy[outer_idx, 1], linewidth=2)
    ax.plot(xy[inner_idx, 0], xy[inner_idx, 1], linewidth=2)

    ax.scatter(xy[:, 0], xy[:, 1], s=180, facecolors="none", edgecolors="red", linewidths=2.5)
    for i, (x, y) in enumerate(xy, start=1):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9)

    ax.set_xlim(-0.03 * L, 1.03 * L)
    ax.set_ylim(-0.03 * L, 1.03 * L)
    ax.set_title("Candidate symmetric layout")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="单独评估手工对称 layout")
    p.add_argument("--outdir", type=str, default="single_layout_symmetric_eval")
    p.add_argument("--quick", action="store_true", help="快速预览模式，明显更快")
    p.add_argument("--terrain-master-seed", type=int, default=20260318)
    p.add_argument("--mc-base-seed", type=int, default=20260317)
    p.add_argument("--n-events", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--geo-grid-n", type=int, default=None)
    p.add_argument("--rough-grid-n", type=int, default=None)
    p.add_argument("--search-grid-n", type=int, default=None)
    p.add_argument("--n-multi-start", type=int, default=None)
    p.add_argument("--layout-scale", type=float, default=1.0, help="以 0.5 为中心做缩放，1.0 表示原样")
    return p


def main() -> None:
    args = make_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = script_dir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    layout_xy = build_candidate_layout(layout_scale=float(args.layout_scale))
    plot_layout(layout_xy, outdir / "candidate_layout.png")

    if args.quick:
        overrides = {
            "N_EVENTS": 400,
            "N_SEEDS": 4,
            "GEO_GRID_N": 31,
            "ROUGH_GRID_N": 81,
            "SEARCH_GRID_N": 11,
            "N_MULTI_START": 4,
            "TERRAIN_MASTER_SEED": int(args.terrain_master_seed),
            "MONTE_CARLO_BASE_SEED": int(args.mc_base_seed),
        }
    else:
        overrides = {
            "N_EVENTS": 2000,
            "N_SEEDS": 10,
            "GEO_GRID_N": 41,
            "ROUGH_GRID_N": 129,
            "SEARCH_GRID_N": 15,
            "N_MULTI_START": 6,
            "TERRAIN_MASTER_SEED": int(args.terrain_master_seed),
            "MONTE_CARLO_BASE_SEED": int(args.mc_base_seed),
        }

    # 手动覆盖优先
    if args.n_events is not None:
        overrides["N_EVENTS"] = int(args.n_events)
    if args.n_seeds is not None:
        overrides["N_SEEDS"] = int(args.n_seeds)
    if args.geo_grid_n is not None:
        overrides["GEO_GRID_N"] = int(args.geo_grid_n)
    if args.rough_grid_n is not None:
        overrides["ROUGH_GRID_N"] = int(args.rough_grid_n)
    if args.search_grid_n is not None:
        overrides["SEARCH_GRID_N"] = int(args.search_grid_n)
    if args.n_multi_start is not None:
        overrides["N_MULTI_START"] = int(args.n_multi_start)

    print("[INFO] output dir:", outdir)
    print("[INFO] candidate layout:\n", np.array2string(layout_xy, precision=4, suppress_small=True))
    print("[INFO] overrides:", overrides)

    result = evaluate_layout_objective(
        layout_xy,
        terrain_kinds=("flat", "hill", "rough"),
        return_raw=True,
        **overrides,
    )

    geom = extract_geometry_features(layout_xy)

    rows = result.get("raw_rows", [])
    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        df_rows.to_csv(outdir / "raw_rows.csv", index=False, encoding="utf-8-sig")

    terrain_rows: List[Dict[str, float]] = []
    for terrain, stat in result.get("terrain_scores", {}).items():
        terrain_rows.append({"terrain": terrain, **stat})
    df_terrain = pd.DataFrame(terrain_rows)
    if not df_terrain.empty:
        df_terrain.to_csv(outdir / "terrain_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "layout_xy": layout_xy.tolist(),
        "geometry_features": geom,
        "valid": bool(result.get("valid", False)),
        "failure_reason": result.get("failure_reason"),
        "J_universal": float(result.get("J_universal", np.nan)),
        "terrain_scores": result.get("terrain_scores", {}),
        "config_used": result.get("config", {}),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("Symmetric candidate evaluation")
    print("=" * 72)
    print(f"valid            : {summary['valid']}")
    print(f"failure_reason   : {summary['failure_reason']}")
    print(f"J_universal      : {summary['J_universal']:.6f}")
    print("\n[Geometry features]")
    for key in [
        "min_pairwise_dist",
        "fraction_near_boundary_0p1L",
        "std_radius",
        "anisotropy",
        "angular_gap_std",
        "min_dist_to_boundary",
    ]:
        print(f"  {key:28s} = {geom[key]:.6f}")

    print("\n[Terrain scores]")
    for terrain, stat in result.get("terrain_scores", {}).items():
        print(
            f"  {terrain:6s} | J_t={stat['J_t']:.6f} | "
            f"mean_err={stat['mean_err_3d_post']:.6f} | "
            f"p95={stat['p95_err_3d_post']:.6f} | "
            f"valid_rate={stat['valid_localization_rate_post']:.4f} | "
            f"usable_rate={stat['usable_event_rate']:.4f}"
        )

    if not df_rows.empty:
        agg = (
            df_rows.groupby("terrain", as_index=False)[
                [
                    "mean_err_3d",
                    "median_err_3d",
                    "p95_err_3d",
                    "valid_localization_rate",
                    "usable_event_rate",
                    "failed_detection",
                    "failed_localization",
                    "delta_edge",
                    "sigma_z",
                ]
            ]
            .mean(numeric_only=True)
        )
        agg.to_csv(outdir / "raw_rows_grouped_mean.csv", index=False, encoding="utf-8-sig")

    print(f"\n[INFO] files written to: {outdir}")


if __name__ == "__main__":
    main()
