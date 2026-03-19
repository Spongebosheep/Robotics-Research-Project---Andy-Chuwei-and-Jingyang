# -*- coding: utf-8 -*-
"""
增强版：同时读取
1) random_layout_5000_all_results.csv   （5000 粗筛）
2) reeval_top50_summary.csv             （top50 复评）

新增功能：
- Pearson + Spearman 相关性
- 粗筛 outlier 过滤版相关性
- top-vs-rest 效应量（Cohen's d）
- top50 复评 bootstrap 相关系数置信区间
- 二维特征着色散点图
- terrain-specific 标签分析（若 CSV 中存在 flat_J_t / hill_J_t / rough_J_t）
- 自动写 summary.txt

用法：
python analyze_layout_features_dual_enhanced.py

或者：
python analyze_layout_features_dual_enhanced.py \
  --all-csv Result/random_layout_5000_all_results.csv \
  --reeval-csv Result/reeval_top50_summary.csv \
  --outdir feature_analysis_dual_enhanced
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Geometry features
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

    centroid = xy.mean(axis=0)
    r = np.linalg.norm(xy - centroid, axis=1)
    feat["n_nodes"] = float(n)
    feat["centroid_x"] = float(centroid[0])
    feat["centroid_y"] = float(centroid[1])
    feat["mean_radius"] = float(np.mean(r))
    feat["std_radius"] = float(np.std(r))
    feat["min_radius"] = float(np.min(r))
    feat["max_radius"] = float(np.max(r))

    D = pairwise_distances(xy)
    D_no_diag = D + np.eye(n) * 1e9
    pair = D[np.triu_indices(n, k=1)]
    nn = D_no_diag.min(axis=1)
    feat["min_pairwise_dist"] = float(np.min(pair))
    feat["mean_pairwise_dist"] = float(np.mean(pair))
    feat["std_pairwise_dist"] = float(np.std(pair))
    feat["nearest_neighbor_mean"] = float(np.mean(nn))
    feat["nearest_neighbor_std"] = float(np.std(nn))

    if n >= 3:
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
        feat["convex_hull_area"] = float(hull.volume)
        feat["convex_hull_perimeter"] = float(polygon_perimeter(hull_pts))
    else:
        feat["convex_hull_area"] = 0.0
        feat["convex_hull_perimeter"] = 0.0
    feat["area_per_node"] = feat["convex_hull_area"] / max(n, 1)

    dist_left = xy[:, 0]
    dist_right = L - xy[:, 0]
    dist_bottom = xy[:, 1]
    dist_top = L - xy[:, 1]
    dist_boundary = np.min(np.column_stack([dist_left, dist_right, dist_bottom, dist_top]), axis=1)
    feat["mean_dist_to_boundary"] = float(np.mean(dist_boundary))
    feat["min_dist_to_boundary"] = float(np.min(dist_boundary))
    feat["fraction_near_boundary_0p1L"] = float(np.mean(dist_boundary < 0.10 * L))
    feat["fraction_near_boundary_0p15L"] = float(np.mean(dist_boundary < 0.15 * L))

    angles = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
    angles = np.sort((angles + 2.0 * np.pi) % (2.0 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    feat["angular_gap_mean"] = float(np.mean(gaps))
    feat["angular_gap_std"] = float(np.std(gaps))
    feat["max_angular_gap"] = float(np.max(gaps))

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

DEFAULT_PAIR_PLOTS = [
    ("min_pairwise_dist", "fraction_near_boundary_0p1L"),
    ("min_pairwise_dist", "anisotropy"),
    ("std_radius", "fraction_near_boundary_0p1L"),
    ("mean_dist_to_boundary", "min_pairwise_dist"),
]


# ============================================================
# Stats utilities
# ============================================================


def safe_corr(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 3:
        return {
            "n": int(len(z)),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pr, pp = pearsonr(z["x"], z["y"])
    sr, sp = spearmanr(z["x"], z["y"])
    return {
        "n": int(len(z)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }



def bootstrap_corr(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_boot: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    z = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    n = len(z)
    if n < 5:
        return {
            "boot_pearson_low": np.nan,
            "boot_pearson_high": np.nan,
            "boot_spearman_low": np.nan,
            "boot_spearman_high": np.nan,
        }

    pear_vals = []
    spear_vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sub = z.iloc[idx]
        try:
            pear_vals.append(float(pearsonr(sub[feature], sub[target])[0]))
        except Exception:
            pear_vals.append(np.nan)
        try:
            spear_vals.append(float(spearmanr(sub[feature], sub[target])[0]))
        except Exception:
            spear_vals.append(np.nan)

    pear_vals = np.asarray(pear_vals, dtype=float)
    spear_vals = np.asarray(spear_vals, dtype=float)
    pear_vals = pear_vals[np.isfinite(pear_vals)]
    spear_vals = spear_vals[np.isfinite(spear_vals)]
    return {
        "boot_pearson_low": float(np.quantile(pear_vals, 0.025)) if len(pear_vals) else np.nan,
        "boot_pearson_high": float(np.quantile(pear_vals, 0.975)) if len(pear_vals) else np.nan,
        "boot_spearman_low": float(np.quantile(spear_vals, 0.025)) if len(spear_vals) else np.nan,
        "boot_spearman_high": float(np.quantile(spear_vals, 0.975)) if len(spear_vals) else np.nan,
    }



def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1)
    if pooled <= 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))



def correlation_table(df: pd.DataFrame, target_col: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        if feat not in df.columns or target_col not in df.columns:
            continue
        stats = safe_corr(df[feat], df[target_col])
        rows.append({"feature": feat, **stats})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["spearman_rho"], key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return out



def correlation_table_with_bootstrap(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    n_boot: int,
    rng_seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(rng_seed)
    for i, feat in enumerate(feature_cols):
        if feat not in df.columns or target_col not in df.columns:
            continue
        stats = safe_corr(df[feat], df[target_col])
        boot = bootstrap_corr(df, feat, target_col, n_boot=n_boot, rng=np.random.default_rng(rng.integers(0, 2**31 - 1)))
        rows.append({"feature": feat, **stats, **boot})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["spearman_rho"], key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return out



def top_vs_rest_table(df: pd.DataFrame, target_col: str, feature_cols: Sequence[str], top_k: int, ascending: bool = True) -> pd.DataFrame:
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    d = d.sort_values(by=target_col, ascending=ascending).reset_index(drop=True)
    top_k = min(int(top_k), len(d))
    top = d.iloc[:top_k]
    rest = d.iloc[top_k:]
    rows = []
    for feat in feature_cols:
        if feat not in d.columns:
            continue
        a = top[feat].to_numpy(dtype=float)
        b = rest[feat].to_numpy(dtype=float)
        rows.append(
            {
                "feature": feat,
                "top_mean": float(np.nanmean(a)) if len(a) else np.nan,
                "rest_mean": float(np.nanmean(b)) if len(b) else np.nan,
                "mean_diff_top_minus_rest": float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else np.nan,
                "cohen_d": cohen_d(a, b),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="cohen_d", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return out



def filter_upper_quantile(df: pd.DataFrame, target_col: str, q: float) -> pd.DataFrame:
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    cutoff = float(d[target_col].quantile(q))
    return d[d[target_col] <= cutoff].copy()


# ============================================================
# IO and feature tables
# ============================================================


def build_feature_table(df_raw: pd.DataFrame, L: float) -> pd.DataFrame:
    rows = []
    for _, row in df_raw.iterrows():
        rec = dict(row)
        xy = load_layout_xy_from_row(row)
        rec.update(extract_layout_features(xy, L=L))
        rows.append(rec)
    return pd.DataFrame(rows)



def resolve_input_csv(path_str: str, script_dir: Path) -> Path:
    p = Path(path_str)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            p,
            script_dir / p,
            script_dir / "Result" / p.name,
            script_dir / "Result" / p,
        ])
    for c in candidates:
        if c.exists():
            return c.resolve()
    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(f"找不到 CSV: {path_str}\n已尝试:\n{tried}")


# ============================================================
# Plots
# ============================================================


def plot_bar_correlation(df_corr: pd.DataFrame, value_col: str, title: str, output_path: Path):
    if df_corr.empty:
        return
    plot_df = df_corr.sort_values(by=value_col)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], plot_df[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def plot_scatter_grid(df: pd.DataFrame, x_cols: Sequence[str], y_col: str, title: str, output_path: Path):
    cols = [c for c in x_cols if c in df.columns and y_col in df.columns]
    if not cols:
        return
    n = len(cols)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, x_col in zip(axes, cols):
        z = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(z[x_col], z[y_col], s=16)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def plot_pair_colored(df: pd.DataFrame, x_col: str, y_col: str, color_col: str, title: str, output_path: Path):
    if x_col not in df.columns or y_col not in df.columns or color_col not in df.columns:
        return
    z = df[[x_col, y_col, color_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(z[x_col], z[y_col], c=z[color_col], s=26)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ============================================================
# Terrain-specific analysis
# ============================================================


def available_target_cols(df: pd.DataFrame, preferred: Sequence[str]) -> List[str]:
    return [c for c in preferred if c in df.columns]


# ============================================================
# Summary writer
# ============================================================


def top_feature_lines(df_corr: pd.DataFrame, value_col: str, k: int = 6) -> List[str]:
    if df_corr.empty or value_col not in df_corr.columns:
        return ["  (none)"]
    d = df_corr.copy()
    d = d.sort_values(by=value_col, key=lambda s: np.abs(s), ascending=False).head(k)
    return [f"  - {r.feature}: {value_col}={getattr(r, value_col):.4f}" for r in d.itertuples(index=False)]



def build_validation_table(merged: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        if feat not in merged.columns:
            continue
        c1 = safe_corr(merged[feat], merged["J_universal"]) if "J_universal" in merged.columns else {}
        c2 = safe_corr(merged[feat], merged["mean_J"]) if "mean_J" in merged.columns else {}
        rows.append(
            {
                "feature": feat,
                "coarse_pearson_r": c1.get("pearson_r", np.nan),
                "coarse_spearman_rho": c1.get("spearman_rho", np.nan),
                "reeval_pearson_r": c2.get("pearson_r", np.nan),
                "reeval_spearman_rho": c2.get("spearman_rho", np.nan),
                "same_sign_pearson": np.sign(c1.get("pearson_r", np.nan)) == np.sign(c2.get("pearson_r", np.nan)),
                "same_sign_spearman": np.sign(c1.get("spearman_rho", np.nan)) == np.sign(c2.get("spearman_rho", np.nan)),
                "abs_spearman_sum": abs(c1.get("spearman_rho", np.nan)) + abs(c2.get("spearman_rho", np.nan)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="abs_spearman_sum", ascending=False).reset_index(drop=True)
    return out


# ============================================================
# Main
# ============================================================


def build_parser():
    p = argparse.ArgumentParser(description="增强版 layout 几何特征分析")
    p.add_argument("--all-csv", type=str, default="random_layout_5000_all_results.csv")
    p.add_argument("--reeval-csv", type=str, default="reeval_top50_summary.csv")
    p.add_argument("--outdir", type=str, default="feature_analysis_dual_enhanced")
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--coarse-filter-quantile", type=float, default=0.995, help="粗筛标签过滤上分位数，用于去掉极端 penalty/outlier")
    p.add_argument("--bootstrap", type=int, default=2000, help="top50 bootstrap 次数")
    p.add_argument("--seed", type=int, default=20260318)
    return p



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

    if "valid" in df_all_raw.columns:
        df_all_raw = df_all_raw[df_all_raw["valid"].astype(bool)].copy()

    print(f"[INFO] coarse CSV: {all_csv}")
    print(f"[INFO] reeval CSV: {reeval_csv}")
    print(f"[INFO] coarse rows: {len(df_all_raw)}")
    print(f"[INFO] reeval rows: {len(df_reeval_raw)}")

    df_all_feat = build_feature_table(df_all_raw, L=float(args.L))
    df_reeval_feat = build_feature_table(df_reeval_raw, L=float(args.L))

    df_all_feat.to_csv(outdir / "all5000_features.csv", index=False, encoding="utf-8-sig")
    df_reeval_feat.to_csv(outdir / "reeval_top50_features.csv", index=False, encoding="utf-8-sig")

    # raw correlations
    corr_all_raw = correlation_table(df_all_feat, "J_universal", FEATURE_COLS)
    corr_all_raw.to_csv(outdir / "corr_all5000_vs_coarse_J_raw.csv", index=False, encoding="utf-8-sig")

    # filtered coarse correlations
    df_all_filtered = filter_upper_quantile(df_all_feat, "J_universal", float(args.coarse_filter_quantile))
    corr_all_filtered = correlation_table(df_all_filtered, "J_universal", FEATURE_COLS)
    corr_all_filtered.to_csv(outdir / "corr_all5000_vs_coarse_J_filtered.csv", index=False, encoding="utf-8-sig")

    # reeval correlations + bootstrap
    corr_reeval = correlation_table_with_bootstrap(
        df_reeval_feat,
        "mean_J",
        FEATURE_COLS,
        n_boot=int(args.bootstrap),
        rng_seed=int(args.seed),
    )
    corr_reeval.to_csv(outdir / "corr_reeval_top50_vs_mean_J_bootstrap.csv", index=False, encoding="utf-8-sig")

    # top-vs-rest effect sizes
    top_vs_rest_all = top_vs_rest_table(df_all_feat, "J_universal", FEATURE_COLS, top_k=int(args.top_k), ascending=True)
    top_vs_rest_reeval = top_vs_rest_table(df_reeval_feat, "mean_J", FEATURE_COLS, top_k=min(int(args.top_k), len(df_reeval_feat)), ascending=True)
    top_vs_rest_all.to_csv(outdir / "top_vs_rest_all5000.csv", index=False, encoding="utf-8-sig")
    top_vs_rest_reeval.to_csv(outdir / "top_vs_rest_reeval_top50.csv", index=False, encoding="utf-8-sig")

    # validation on merged subset
    merge_keys = [k for k in ["sample_id", "layout_json"] if k in df_all_feat.columns and k in df_reeval_feat.columns]
    if merge_keys:
        merged = df_reeval_feat.merge(
            df_all_feat[[*merge_keys, "J_universal", *FEATURE_COLS]],
            on=merge_keys,
            how="left",
            suffixes=("", "_from_all"),
        )
        validation = build_validation_table(merged, FEATURE_COLS)
        validation.to_csv(outdir / "validation_coarse_vs_reeval.csv", index=False, encoding="utf-8-sig")
    else:
        merged = pd.DataFrame()
        validation = pd.DataFrame()

    # terrain-specific label analysis if available
    terrain_targets = ["flat_J_t", "hill_J_t", "rough_J_t"]
    for name, df_cur in [("all5000", df_all_feat), ("reeval_top50", df_reeval_feat)]:
        for target in available_target_cols(df_cur, terrain_targets):
            ct = correlation_table(df_cur, target, FEATURE_COLS)
            ct.to_csv(outdir / f"corr_{name}_vs_{target}.csv", index=False, encoding="utf-8-sig")

    # plots
    plot_scatter_grid(
        df_all_filtered,
        ["min_pairwise_dist", "mean_dist_to_boundary", "convex_hull_area", "angular_gap_std", "anisotropy", "mean_radius"],
        "J_universal",
        f"All 5000 filtered (q<={args.coarse_filter_quantile}): feature vs coarse J",
        outdir / "scatter_all5000_filtered.png",
    )
    plot_scatter_grid(
        df_reeval_feat,
        ["min_pairwise_dist", "mean_dist_to_boundary", "convex_hull_area", "angular_gap_std", "anisotropy", "mean_radius"],
        "mean_J",
        "Reeval top50: feature vs mean J",
        outdir / "scatter_reeval_top50.png",
    )
    plot_bar_correlation(corr_all_raw, "spearman_rho", "All 5000 raw: Spearman correlation with coarse J", outdir / "bar_corr_all5000_raw_spearman.png")
    plot_bar_correlation(corr_all_filtered, "spearman_rho", f"All 5000 filtered(q<={args.coarse_filter_quantile}): Spearman correlation with coarse J", outdir / "bar_corr_all5000_filtered_spearman.png")
    plot_bar_correlation(corr_reeval, "spearman_rho", "Reeval top50: Spearman correlation with mean J", outdir / "bar_corr_reeval_top50_spearman.png")

    for name, df_cur, target in [
        ("all5000", df_all_filtered, "J_universal"),
        ("reeval_top50", df_reeval_feat, "mean_J"),
    ]:
        for xcol, ycol in DEFAULT_PAIR_PLOTS:
            plot_pair_colored(
                df_cur,
                xcol,
                ycol,
                target,
                f"{name}: {xcol} vs {ycol}, colored by {target}",
                outdir / f"pair_{name}_{xcol}_vs_{ycol}.png",
            )

    # summary text
    summary_path = outdir / "summary.txt"
    lines: List[str] = []
    lines.append("Enhanced layout feature analysis summary")
    lines.append("=" * 70)
    lines.append(f"coarse_csv: {all_csv}")
    lines.append(f"reeval_csv: {reeval_csv}")
    lines.append(f"coarse_rows_raw: {len(df_all_feat)}")
    lines.append(f"coarse_rows_filtered: {len(df_all_filtered)}")
    lines.append(f"reeval_rows: {len(df_reeval_feat)}")
    lines.append("")

    lines.append("Top features | all5000 raw | Spearman")
    lines.extend(top_feature_lines(corr_all_raw, "spearman_rho"))
    lines.append("")

    lines.append("Top features | all5000 filtered | Spearman")
    lines.extend(top_feature_lines(corr_all_filtered, "spearman_rho"))
    lines.append("")

    lines.append("Top features | reeval top50 | Spearman")
    lines.extend(top_feature_lines(corr_reeval, "spearman_rho"))
    lines.append("")

    lines.append("Top top-vs-rest effect sizes | all5000")
    if not top_vs_rest_all.empty:
        for r in top_vs_rest_all.head(6).itertuples(index=False):
            lines.append(f"  - {r.feature}: cohen_d={r.cohen_d:.4f}, top_mean={r.top_mean:.4f}, rest_mean={r.rest_mean:.4f}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Top top-vs-rest effect sizes | reeval top50")
    if not top_vs_rest_reeval.empty:
        for r in top_vs_rest_reeval.head(6).itertuples(index=False):
            lines.append(f"  - {r.feature}: cohen_d={r.cohen_d:.4f}, top_mean={r.top_mean:.4f}, rest_mean={r.rest_mean:.4f}")
    else:
        lines.append("  (none)")
    lines.append("")

    lines.append("Bootstrap-stable reeval features (95% CI for Spearman excludes 0)")
    if not corr_reeval.empty:
        stable = corr_reeval[(corr_reeval["boot_spearman_low"] > 0) | (corr_reeval["boot_spearman_high"] < 0)]
        if len(stable) == 0:
            lines.append("  (none)")
        else:
            for r in stable.head(10).itertuples(index=False):
                lines.append(
                    f"  - {r.feature}: rho={r.spearman_rho:.4f}, CI=[{r.boot_spearman_low:.4f}, {r.boot_spearman_high:.4f}]"
                )
    lines.append("")

    if not validation.empty:
        lines.append("Features with same sign in coarse and reeval Spearman")
        same = validation[validation["same_sign_spearman"] == True]
        if len(same) == 0:
            lines.append("  (none)")
        else:
            for r in same.head(10).itertuples(index=False):
                lines.append(
                    f"  - {r.feature}: coarse_rho={r.coarse_spearman_rho:.4f}, reeval_rho={r.reeval_spearman_rho:.4f}"
                )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] outputs saved to: {outdir}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
