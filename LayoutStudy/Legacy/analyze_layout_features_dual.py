# -*- coding: utf-8 -*-
"""
同时读取：
1) random_layout_5000_all_results.csv   （5000 粗筛）
2) reeval_top50_summary.csv             （top50 复评）

输出：
- 5000 样本的几何特征表
- top50 复评样本的几何特征表
- 两套标签下的特征相关性对比
- top10 vs 其余样本的均值对比
- 若干散点图与条形图

用法：
python analyze_layout_features_dual.py

或者指定路径：
python analyze_layout_features_dual.py \
  --all-csv random_layout_5000_all_results.csv \
  --reeval-csv reeval_top50_summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


# ============================================================
# 几何特征
# ============================================================

def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def polygon_perimeter(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    nxt = np.roll(points, -1, axis=0)
    return float(np.sum(np.linalg.norm(nxt - points, axis=1)))


def load_layout_xy_from_row(row: pd.Series) -> np.ndarray:
    if "layout_json" in row.index and pd.notna(row["layout_json"]):
        arr = np.asarray(json.loads(row["layout_json"]), dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr

    x_cols = sorted([c for c in row.index if c.startswith("x") and c[1:].isdigit()], key=lambda s: int(s[1:]))
    y_cols = sorted([c for c in row.index if c.startswith("y") and c[1:].isdigit()], key=lambda s: int(s[1:]))
    if x_cols and len(x_cols) == len(y_cols):
        arr = np.column_stack([[float(row[c]) for c in x_cols], [float(row[c]) for c in y_cols]])
        return arr

    raise ValueError("无法从行数据中恢复 layout 坐标，需要 layout_json 或 x*/y* 列")


def extract_layout_features(layout_xy: np.ndarray, L: float = 1.0) -> Dict[str, float]:
    xy = np.asarray(layout_xy, dtype=float)
    n = xy.shape[0]
    feat: Dict[str, float] = {}

    # 质心与半径
    centroid = xy.mean(axis=0)
    r = np.linalg.norm(xy - centroid, axis=1)
    feat["n_nodes"] = float(n)
    feat["centroid_x"] = float(centroid[0])
    feat["centroid_y"] = float(centroid[1])
    feat["mean_radius"] = float(np.mean(r))
    feat["std_radius"] = float(np.std(r))
    feat["min_radius"] = float(np.min(r))
    feat["max_radius"] = float(np.max(r))

    # pairwise / nearest neighbor
    D = pairwise_distances(xy)
    pairwise = D[np.triu_indices(n, k=1)]
    D_no_diag = D + np.eye(n) * 1e9
    nn = D_no_diag.min(axis=1)
    feat["min_pairwise_dist"] = float(np.min(pairwise))
    feat["mean_pairwise_dist"] = float(np.mean(pairwise))
    feat["std_pairwise_dist"] = float(np.std(pairwise))
    feat["nearest_neighbor_mean"] = float(np.mean(nn))
    feat["nearest_neighbor_std"] = float(np.std(nn))

    # hull
    if n >= 3:
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
        feat["convex_hull_area"] = float(hull.volume)   # 2D 中 volume=面积
        feat["convex_hull_perimeter"] = polygon_perimeter(hull_pts)
    else:
        feat["convex_hull_area"] = 0.0
        feat["convex_hull_perimeter"] = 0.0
    feat["area_per_node"] = float(feat["convex_hull_area"] / max(n, 1))

    # boundary
    dist_left = xy[:, 0]
    dist_right = L - xy[:, 0]
    dist_bottom = xy[:, 1]
    dist_top = L - xy[:, 1]
    dist_boundary = np.min(np.column_stack([dist_left, dist_right, dist_bottom, dist_top]), axis=1)
    feat["mean_dist_to_boundary"] = float(np.mean(dist_boundary))
    feat["min_dist_to_boundary"] = float(np.min(dist_boundary))
    feat["fraction_near_boundary_0p1L"] = float(np.mean(dist_boundary < 0.1 * L))
    feat["fraction_near_boundary_0p15L"] = float(np.mean(dist_boundary < 0.15 * L))

    # angle coverage
    angles = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
    angles = np.sort((angles + 2 * np.pi) % (2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    feat["angular_gap_mean"] = float(np.mean(gaps))
    feat["angular_gap_std"] = float(np.std(gaps))
    feat["max_angular_gap"] = float(np.max(gaps))

    # anisotropy
    cov = np.cov(xy.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    feat["lambda1"] = float(eigvals[0])
    feat["lambda2"] = float(eigvals[1])
    feat["anisotropy"] = float(eigvals[0] / max(eigvals[1], 1e-12))

    return feat


FEATURE_COLS = [
    "mean_radius",
    "std_radius",
    "min_pairwise_dist",
    "mean_pairwise_dist",
    "nearest_neighbor_mean",
    "nearest_neighbor_std",
    "convex_hull_area",
    "convex_hull_perimeter",
    "mean_dist_to_boundary",
    "min_dist_to_boundary",
    "fraction_near_boundary_0p1L",
    "fraction_near_boundary_0p15L",
    "angular_gap_std",
    "max_angular_gap",
    "anisotropy",
]


# ============================================================
# 构表 / 统计
# ============================================================

def build_feature_table(df: pd.DataFrame, L: float) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        xy = load_layout_xy_from_row(row)
        feat = extract_layout_features(xy, L=L)
        meta = row.to_dict()
        meta.update(feat)
        rows.append(meta)
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df[target_col], errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
        rows.append({"feature": col, f"corr_with_{target_col}": corr, f"abs_corr_with_{target_col}": abs(corr) if pd.notna(corr) else np.nan})
    out = pd.DataFrame(rows).sort_values(by=f"abs_corr_with_{target_col}", ascending=False).reset_index(drop=True)
    return out


def top_vs_rest_table(df: pd.DataFrame, target_col: str, feature_cols: List[str], top_k: int, ascending: bool = True) -> pd.DataFrame:
    ranked = df.sort_values(by=target_col, ascending=ascending).reset_index(drop=True)
    top = ranked.head(top_k)
    rest = ranked.iloc[top_k:]
    rows = []
    for col in feature_cols:
        rows.append({
            "feature": col,
            "top_mean": float(top[col].mean()),
            "rest_mean": float(rest[col].mean()) if len(rest) > 0 else np.nan,
            "difference_top_minus_rest": float(top[col].mean() - rest[col].mean()) if len(rest) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def validation_compare_table(merged: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    coarse_corr = correlation_table(merged, "J_universal", feature_cols).rename(columns={"corr_with_J_universal": "corr_coarse_J", "abs_corr_with_J_universal": "abs_corr_coarse_J"})
    reeval_corr = correlation_table(merged, "mean_J", feature_cols).rename(columns={"corr_with_mean_J": "corr_reeval_mean_J", "abs_corr_with_mean_J": "abs_corr_reeval_mean_J"})
    out = coarse_corr.merge(reeval_corr, on="feature", how="outer")
    out["same_sign"] = np.sign(out["corr_coarse_J"]) == np.sign(out["corr_reeval_mean_J"])
    out["corr_diff"] = out["corr_reeval_mean_J"] - out["corr_coarse_J"]
    out["abs_strength_shift"] = out["abs_corr_reeval_mean_J"] - out["abs_corr_coarse_J"]
    out = out.sort_values(by=["same_sign", "abs_corr_reeval_mean_J"], ascending=[False, False]).reset_index(drop=True)
    return out


# ============================================================
# 画图
# ============================================================

def save_bar(corr_df: pd.DataFrame, corr_col: str, out_path: str, title: str, top_n: int = 12):
    plot_df = corr_df.dropna(subset=[corr_col]).copy().sort_values(by=corr_col)
    if len(plot_df) > top_n:
        head = plot_df.head(top_n // 2)
        tail = plot_df.tail(top_n - len(head))
        plot_df = pd.concat([head, tail], axis=0)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df[corr_col])
    plt.title(title)
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_scatter_grid(df: pd.DataFrame, target_col: str, features: List[str], out_path: str):
    n = len(features)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, features):
        ax.scatter(df[col], df[target_col], s=10)
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        ax.grid(True, alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================

def build_parser():
    p = argparse.ArgumentParser(description="同时读取 5000 粗筛和 top50 复评结果，提几何特征并做验证分析。")
    p.add_argument("--all-csv", type=str, default="random_layout_5000_all_results.csv")
    p.add_argument("--reeval-csv", type=str, default="reeval_top50_summary.csv")
    p.add_argument("--outdir", type=str, default="feature_analysis_dual")
    p.add_argument("--L", type=float, default=1.0, help="监测区域边长，默认 1.0")
    p.add_argument("--top-k", type=int, default=10, help="top-vs-rest 比较中的 top K")
    return p


def resolve_input_csv(path_str: str, script_dir: Path) -> Path:
    p = Path(path_str)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(p)
        candidates.append(script_dir / p)
        candidates.append(script_dir / "Result" / p.name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    tried = "\n".join(str(x.resolve()) if not x.is_absolute() else str(x) for x in candidates)
    raise FileNotFoundError(f"找不到 CSV: {path_str}\n已尝试:\n{tried}")


def main():
    args = build_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = script_dir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    all_csv = resolve_input_csv(args.all_csv, script_dir)
    reeval_csv = resolve_input_csv(args.reeval_csv, script_dir)

    df_all_raw = pd.read_csv(all_csv)
    df_reeval_raw = pd.read_csv(reeval_csv)

    # 只保留有效粗筛样本
    if "valid" in df_all_raw.columns:
        df_all_raw = df_all_raw[df_all_raw["valid"].astype(bool)].copy()

    print(f"[INFO] 5000 粗筛样本数: {len(df_all_raw)}")
    print(f"[INFO] top50 复评样本数: {len(df_reeval_raw)}")

    df_all_feat = build_feature_table(df_all_raw, L=float(args.L))
    df_reeval_feat = build_feature_table(df_reeval_raw, L=float(args.L))

    # 保存特征表
    all_feat_csv = outdir / "all5000_features.csv"
    reeval_feat_csv = outdir / "reeval_top50_features.csv"
    df_all_feat.to_csv(all_feat_csv, index=False, encoding="utf-8-sig")
    df_reeval_feat.to_csv(reeval_feat_csv, index=False, encoding="utf-8-sig")

    # 5000 粗筛：特征 vs J_universal
    corr_all = correlation_table(df_all_feat, "J_universal", FEATURE_COLS)
    corr_all_csv = outdir / "corr_all5000_vs_coarse_J.csv"
    corr_all.to_csv(corr_all_csv, index=False, encoding="utf-8-sig")

    # top50 复评：特征 vs mean_J
    corr_reeval = correlation_table(df_reeval_feat, "mean_J", FEATURE_COLS)
    corr_reeval_csv = outdir / "corr_reeval_top50_vs_mean_J.csv"
    corr_reeval.to_csv(corr_reeval_csv, index=False, encoding="utf-8-sig")

    # top vs rest
    top_vs_rest_all = top_vs_rest_table(df_all_feat, "J_universal", FEATURE_COLS, top_k=int(args.top_k), ascending=True)
    top_vs_rest_reeval = top_vs_rest_table(df_reeval_feat, "mean_J", FEATURE_COLS, top_k=min(int(args.top_k), len(df_reeval_feat)), ascending=True)
    top_vs_rest_all.to_csv(outdir / "top_vs_rest_all5000.csv", index=False, encoding="utf-8-sig")
    top_vs_rest_reeval.to_csv(outdir / "top_vs_rest_reeval_top50.csv", index=False, encoding="utf-8-sig")

    # merge：用复评 top50 去验证粗筛关系
    merge_keys = [k for k in ["sample_id", "layout_json"] if k in df_all_feat.columns and k in df_reeval_feat.columns]
    if not merge_keys:
        raise ValueError("5000 粗筛和 top50 复评之间没有可用 merge key（至少要有 sample_id 或 layout_json）")

    merged = df_reeval_feat.merge(
        df_all_feat[[*merge_keys, "J_universal", *FEATURE_COLS]],
        on=merge_keys,
        how="left",
        suffixes=("", "_from_all"),
    )

    validation = validation_compare_table(merged, FEATURE_COLS)
    validation_csv = outdir / "validation_coarse_vs_reeval.csv"
    validation.to_csv(validation_csv, index=False, encoding="utf-8-sig")

    # 简单摘要
    summary_txt = outdir / "summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== 5000 粗筛：与 coarse J 相关性最强的前 8 个特征 ===\n")
        f.write(corr_all.head(8).to_string(index=False))
        f.write("\n\n=== top50 复评：与 mean_J 相关性最强的前 8 个特征 ===\n")
        f.write(corr_reeval.head(8).to_string(index=False))
        f.write("\n\n=== 复评验证：粗筛关系与复评关系同号的特征 ===\n")
        f.write(validation[validation["same_sign"]].head(12).to_string(index=False))
        f.write("\n")

    # 画图
    save_bar(corr_all, "corr_with_J_universal", str(outdir / "bar_corr_all5000_vs_coarseJ.png"), "All 5000: feature correlation with coarse J")
    save_bar(corr_reeval, "corr_with_mean_J", str(outdir / "bar_corr_reeval_top50_vs_meanJ.png"), "Reeval top50: feature correlation with mean J")

    scatter_features = [
        "min_pairwise_dist",
        "mean_dist_to_boundary",
        "convex_hull_area",
        "angular_gap_std",
        "anisotropy",
        "mean_radius",
    ]
    save_scatter_grid(df_all_feat, "J_universal", scatter_features, str(outdir / "scatter_all5000.png"))
    save_scatter_grid(df_reeval_feat, "mean_J", scatter_features, str(outdir / "scatter_reeval_top50.png"))

    print("\n=== 5000 粗筛：特征 vs coarse J（前 8） ===")
    print(corr_all.head(8).to_string(index=False))

    print("\n=== top50 复评：特征 vs mean_J（前 8） ===")
    print(corr_reeval.head(8).to_string(index=False))

    print("\n=== 粗筛关系 vs 复评关系（同号优先） ===")
    print(validation.head(12).to_string(index=False))

    print("\n[INFO] 输出文件已保存到:")
    print(outdir.resolve())
    print(f"  - {all_feat_csv.name}")
    print(f"  - {reeval_feat_csv.name}")
    print(f"  - {corr_all_csv.name}")
    print(f"  - {corr_reeval_csv.name}")
    print(f"  - {validation_csv.name}")
    print("  - top_vs_rest_all5000.csv")
    print("  - top_vs_rest_reeval_top50.csv")
    print("  - bar_corr_all5000_vs_coarseJ.png")
    print("  - bar_corr_reeval_top50_vs_meanJ.png")
    print("  - scatter_all5000.png")
    print("  - scatter_reeval_top50.png")
    print("  - summary.txt")


if __name__ == "__main__":
    main()
