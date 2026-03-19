# -*- coding: utf-8 -*-
"""
完整增强版：layout 几何特征分析

相比 analyze_layout_features_dual_enhanced.py，补齐以下缺失部分：
1. 二维特征图（保留并系统化输出）
2. 随机森林 importance（用于特征优先级，不强调预测）
3. terrain-specific 分析（相关性 + 随机森林 + 二维图 + 非线性分箱）
4. top10 layout 真正画出来，并配套几何分布对比图
5. 非线性区间分析（默认至少对 min_pairwise_dist 做分箱）

默认输入：
- random_layout_5000_all_results.csv
- reeval_top50_summary.csv

默认运行：
python analyze_layout_features_complete.py

也可以显式指定：
python analyze_layout_features_complete.py \
  --all-csv Result/random_layout_5000_all_results.csv \
  --reeval-csv Result/reeval_top50_summary.csv \
  --outdir feature_analysis_complete
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Geometry utilities
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

    raise ValueError("无法从行数据恢复 layout：需要 layout_json 或 x*/y* 列")



def compute_layout_profiles(layout_xy: np.ndarray) -> Dict[str, np.ndarray]:
    xy = np.asarray(layout_xy, dtype=float)
    centroid = xy.mean(axis=0)
    radii = np.linalg.norm(xy - centroid, axis=1)

    n = len(xy)
    D = pairwise_distances(xy)
    pair = D[np.triu_indices(n, k=1)] if n >= 2 else np.asarray([], dtype=float)

    angles = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
    angles = np.sort((angles + 2.0 * np.pi) % (2.0 * np.pi))
    angular_gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi]) if len(angles) >= 2 else np.asarray([], dtype=float)

    return {
        "radii": np.asarray(radii, dtype=float),
        "pairwise": np.asarray(pair, dtype=float),
        "angular_gaps": np.asarray(angular_gaps, dtype=float),
    }



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
    pair = D[np.triu_indices(n, k=1)] if n >= 2 else np.asarray([], dtype=float)
    nn = D_no_diag.min(axis=1) if n >= 2 else np.asarray([0.0], dtype=float)
    feat["min_pairwise_dist"] = float(np.min(pair)) if len(pair) else np.nan
    feat["mean_pairwise_dist"] = float(np.mean(pair)) if len(pair) else np.nan
    feat["std_pairwise_dist"] = float(np.std(pair)) if len(pair) else np.nan
    feat["nearest_neighbor_mean"] = float(np.mean(nn)) if len(nn) else np.nan
    feat["nearest_neighbor_std"] = float(np.std(nn)) if len(nn) else np.nan

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
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi]) if len(angles) >= 2 else np.asarray([0.0], dtype=float)
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

PAIR_PLOTS = [
    ("min_pairwise_dist", "fraction_near_boundary_0p1L"),
    ("min_pairwise_dist", "anisotropy"),
    ("std_radius", "fraction_near_boundary_0p1L"),
]

NONLINEAR_FEATURES = [
    "min_pairwise_dist",
    "fraction_near_boundary_0p1L",
    "anisotropy",
    "std_radius",
]

TERRAIN_TARGETS = ["flat_J_t", "hill_J_t", "rough_J_t"]


# ============================================================
# Stats helpers
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
    for feat in feature_cols:
        if feat not in df.columns or target_col not in df.columns:
            continue
        stats = safe_corr(df[feat], df[target_col])
        boot = bootstrap_corr(
            df,
            feat,
            target_col,
            n_boot=n_boot,
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
        )
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



def available_target_cols(df: pd.DataFrame, preferred: Sequence[str]) -> List[str]:
    return [c for c in preferred if c in df.columns]


# ============================================================
# RF importance + binning
# ============================================================


def random_forest_importance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    seed: int,
    n_estimators: int,
    permutation_repeats: int,
) -> pd.DataFrame:
    cols = [c for c in feature_cols if c in df.columns]
    z = df[cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(z) < 10 or len(cols) < 2:
        return pd.DataFrame()

    X = z[cols]
    y = z[target_col].astype(float)

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        n_jobs=1,
        min_samples_leaf=3,
    )
    model.fit(X, y)

    mdi = np.asarray(model.feature_importances_, dtype=float)
    perm = permutation_importance(
        model,
        X,
        y,
        n_repeats=int(permutation_repeats),
        random_state=int(seed),
        n_jobs=1,
        scoring="neg_mean_squared_error",
    )

    out = pd.DataFrame(
        {
            "feature": cols,
            "mdi_importance": mdi,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )
    out["mdi_rank"] = out["mdi_importance"].rank(ascending=False, method="dense").astype(int)
    out["perm_rank"] = out["permutation_importance_mean"].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(by="permutation_importance_mean", ascending=False).reset_index(drop=True)
    return out



def build_binned_mean_table(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_bins: int,
    bin_step: Optional[float],
    min_count: int,
) -> pd.DataFrame:
    z = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(z) < max(n_bins, 8):
        return pd.DataFrame()

    x = z[feature].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return pd.DataFrame()

    if bin_step is not None and bin_step > 0:
        left = math.floor(xmin / bin_step) * bin_step
        right = math.ceil(xmax / bin_step) * bin_step
        edges = np.arange(left, right + 0.5 * bin_step, bin_step)
        if len(edges) < 3:
            edges = np.linspace(xmin, xmax, int(n_bins) + 1)
    else:
        edges = np.linspace(xmin, xmax, int(n_bins) + 1)

    if len(np.unique(edges)) < 3:
        return pd.DataFrame()

    z["bin"] = pd.cut(z[feature], bins=edges, include_lowest=True, duplicates="drop")
    grp = z.groupby("bin", observed=False)[target].agg(["count", "mean", "std"]).reset_index()
    if len(grp) == 0:
        return pd.DataFrame()

    grp = grp[grp["count"] >= int(min_count)].copy()
    if len(grp) == 0:
        return pd.DataFrame()

    grp["se"] = grp["std"] / np.sqrt(grp["count"])
    grp["bin_left"] = grp["bin"].apply(lambda x: float(x.left)).astype(float)
    grp["bin_right"] = grp["bin"].apply(lambda x: float(x.right)).astype(float)
    grp["bin_center"] = 0.5 * (grp["bin_left"].astype(float) + grp["bin_right"].astype(float))
    grp["feature"] = feature
    grp["target"] = target
    return grp[["feature", "target", "bin", "bin_left", "bin_right", "bin_center", "count", "mean", "std", "se"]]


# ============================================================
# IO
# ============================================================


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



def build_feature_table(df_raw: pd.DataFrame, L: float) -> pd.DataFrame:
    rows = []
    for _, row in df_raw.iterrows():
        rec = dict(row)
        xy = load_layout_xy_from_row(row)
        rec.update(extract_layout_features(xy, L=L))
        rows.append(rec)
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================


def plot_bar_correlation(df_corr: pd.DataFrame, value_col: str, title: str, output_path: Path):
    if df_corr.empty or value_col not in df_corr.columns:
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
    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    sc = ax.scatter(z[x_col], z[y_col], c=z[color_col], s=28)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def plot_rf_importance(df_imp: pd.DataFrame, title: str, output_path: Path, value_col: str = "permutation_importance_mean"):
    if df_imp.empty or value_col not in df_imp.columns:
        return
    plot_df = df_imp.sort_values(by=value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], plot_df[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def plot_binned_mean_curve(df_bin: pd.DataFrame, feature: str, target: str, title: str, output_path: Path):
    if df_bin.empty:
        return
    d = df_bin.sort_values(by="bin_center").copy()
    fig, ax1 = plt.subplots(figsize=(8.0, 5.5))
    ax1.errorbar(d["bin_center"], d["mean"], yerr=d["se"], marker="o", capsize=3)
    ax1.set_xlabel(feature)
    ax1.set_ylabel(f"mean({target})")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    width = np.maximum((d["bin_right"] - d["bin_left"]).to_numpy(dtype=float), 1e-6)
    ax2.bar(d["bin_center"], d["count"], width=0.85 * width, alpha=0.18)
    ax2.set_ylabel("count")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def plot_layout_gallery(df_sorted: pd.DataFrame, target_col: str, output_path: Path, L: float, top_k: int, title: str):
    if df_sorted.empty or target_col not in df_sorted.columns:
        return

    top_df = df_sorted.nsmallest(top_k, target_col).reset_index(drop=True)
    top_k = len(top_df)
    if top_k == 0:
        return

    ncols = 5 if top_k >= 5 else top_k
    nrows = int(math.ceil(top_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, (_, row) in enumerate(top_df.iterrows()):
        ax = axes[i]
        xy = load_layout_xy_from_row(row)
        ax.scatter(xy[:, 0], xy[:, 1], s=45)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, linewidth=1.2))
        ax.set_title(
            f"rank {i+1} | {target_col}={row[target_col]:.4f}\n"
            f"min_d={row.get('min_pairwise_dist', np.nan):.3f}, nearB={row.get('fraction_near_boundary_0p1L', np.nan):.2f}\n"
            f"std_r={row.get('std_radius', np.nan):.3f}, anis={row.get('anisotropy', np.nan):.2f}",
            fontsize=9,
        )
    for ax in axes[top_k:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def plot_topk_geometry_profiles(df_sorted: pd.DataFrame, target_col: str, output_path: Path, top_k: int, title: str):
    if df_sorted.empty or target_col not in df_sorted.columns:
        return
    top_df = df_sorted.nsmallest(top_k, target_col).reset_index(drop=True)
    if len(top_df) == 0:
        return

    ranks = np.arange(1, len(top_df) + 1)
    radius_data: List[np.ndarray] = []
    gap_data: List[np.ndarray] = []
    min_d = []
    near_b = []

    for _, row in top_df.iterrows():
        prof = compute_layout_profiles(load_layout_xy_from_row(row))
        radius_data.append(prof["radii"])
        gap_data.append(prof["angular_gaps"])
        min_d.append(float(row.get("min_pairwise_dist", np.nan)))
        near_b.append(float(row.get("fraction_near_boundary_0p1L", np.nan)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(ranks, min_d)
    ax.set_title("min_pairwise_dist")
    ax.set_xlabel("top rank")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(ranks, near_b)
    ax.set_title("fraction_near_boundary_0p1L")
    ax.set_xlabel("top rank")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.boxplot(radius_data, tick_labels=[str(r) for r in ranks], showfliers=False)
    ax.set_title("radius distribution")
    ax.set_xlabel("top rank")
    ax.set_ylabel("radius")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.boxplot(gap_data, tick_labels=[str(r) for r in ranks], showfliers=False)
    ax.set_title("angular gap distribution")
    ax.set_xlabel("top rank")
    ax.set_ylabel("gap (rad)")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ============================================================
# Summary helpers
# ============================================================


def top_feature_lines(df_corr: pd.DataFrame, value_col: str, k: int = 6) -> List[str]:
    if df_corr.empty or value_col not in df_corr.columns:
        return ["  (none)"]
    d = df_corr.copy().sort_values(by=value_col, key=lambda s: np.abs(s), ascending=False).head(k)
    return [f"  - {r.feature}: {value_col}={getattr(r, value_col):.4f}" for r in d.itertuples(index=False)]



def top_rf_lines(df_imp: pd.DataFrame, k: int = 6) -> List[str]:
    if df_imp.empty:
        return ["  (none)"]
    d = df_imp.head(k)
    return [
        f"  - {r.feature}: perm={r.permutation_importance_mean:.6f}, mdi={r.mdi_importance:.6f}"
        for r in d.itertuples(index=False)
    ]



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
# Main analysis blocks
# ============================================================


def save_pair_plots(df: pd.DataFrame, target_col: str, tag: str, outdir: Path):
    for xcol, ycol in PAIR_PLOTS:
        plot_pair_colored(
            df,
            xcol,
            ycol,
            target_col,
            f"{tag}: {xcol} vs {ycol}, colored by {target_col}",
            outdir / f"pair_{tag}_{xcol}_vs_{ycol}.png",
        )



def save_nonlinear_analysis(
    df: pd.DataFrame,
    target_col: str,
    tag: str,
    outdir: Path,
    n_bins: int,
    bin_step: Optional[float],
    min_bin_count: int,
) -> List[str]:
    saved = []
    for feat in NONLINEAR_FEATURES:
        if feat not in df.columns or target_col not in df.columns:
            continue
        btab = build_binned_mean_table(df, feat, target_col, n_bins=n_bins, bin_step=bin_step, min_count=min_bin_count)
        if btab.empty:
            continue
        csv_path = outdir / f"binned_{tag}_{feat}_vs_{target_col}.csv"
        png_path = outdir / f"binned_{tag}_{feat}_vs_{target_col}.png"
        btab.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_binned_mean_curve(
            btab,
            feat,
            target_col,
            f"{tag}: binned mean of {target_col} vs {feat}",
            png_path,
        )
        saved.append(feat)
    return saved



def analyze_target(
    df: pd.DataFrame,
    target_col: str,
    tag: str,
    outdir: Path,
    seed: int,
    bootstrap_n: int,
    rf_estimators: int,
    rf_perm_repeats: int,
    top_k: int,
    L: float,
    n_bins: int,
    bin_step: Optional[float],
    min_bin_count: int,
    do_bootstrap: bool,
    do_top_gallery: bool,
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    if target_col not in df.columns:
        return results

    corr = correlation_table_with_bootstrap(df, target_col, FEATURE_COLS, bootstrap_n, seed) if do_bootstrap else correlation_table(df, target_col, FEATURE_COLS)
    corr.to_csv(outdir / f"corr_{tag}_vs_{target_col}.csv", index=False, encoding="utf-8-sig")
    results["corr"] = corr

    rf = random_forest_importance(df, target_col, FEATURE_COLS, seed=seed, n_estimators=rf_estimators, permutation_repeats=rf_perm_repeats)
    rf.to_csv(outdir / f"rf_importance_{tag}_vs_{target_col}.csv", index=False, encoding="utf-8-sig")
    results["rf"] = rf

    tvr = top_vs_rest_table(df, target_col, FEATURE_COLS, top_k=top_k, ascending=True)
    tvr.to_csv(outdir / f"top_vs_rest_{tag}_vs_{target_col}.csv", index=False, encoding="utf-8-sig")
    results["top_vs_rest"] = tvr

    plot_bar_correlation(corr, "spearman_rho", f"{tag}: Spearman correlation with {target_col}", outdir / f"bar_corr_{tag}_vs_{target_col}.png")
    plot_rf_importance(rf, f"{tag}: RF permutation importance for {target_col}", outdir / f"bar_rf_{tag}_vs_{target_col}.png")

    plot_scatter_grid(
        df,
        ["min_pairwise_dist", "mean_dist_to_boundary", "convex_hull_area", "angular_gap_std", "anisotropy", "mean_radius"],
        target_col,
        f"{tag}: feature vs {target_col}",
        outdir / f"scatter_{tag}_vs_{target_col}.png",
    )
    save_pair_plots(df, target_col, f"{tag}_vs_{target_col}", outdir)
    save_nonlinear_analysis(df, target_col, f"{tag}_vs_{target_col}", outdir, n_bins=n_bins, bin_step=bin_step, min_bin_count=min_bin_count)

    if do_top_gallery:
        sorted_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).sort_values(by=target_col, ascending=True).reset_index(drop=True)
        plot_layout_gallery(sorted_df, target_col, outdir / f"top{top_k}_layouts_{tag}_vs_{target_col}.png", L=L, top_k=top_k, title=f"Top {top_k} layouts | {tag} | ranked by {target_col}")
        plot_topk_geometry_profiles(sorted_df, target_col, outdir / f"top{top_k}_geometry_profiles_{tag}_vs_{target_col}.png", top_k=top_k, title=f"Top {top_k} geometry profiles | {tag} | ranked by {target_col}")

        top_df = sorted_df.head(top_k).copy()
        if len(top_df) > 0:
            # 兼容原表里已经存在 rank 列（例如 reeval summary 常见）
            top_df["rank"] = np.arange(1, len(top_df) + 1)
            preferred_cols = ["rank", target_col, "min_pairwise_dist", "fraction_near_boundary_0p1L", "std_radius", "anisotropy", "mean_radius", "angular_gap_std", "max_angular_gap"]
            seen = set()
            cols = []
            for c in preferred_cols:
                if c in top_df.columns and c not in seen:
                    cols.append(c)
                    seen.add(c)
            top_df = top_df[cols + [c for c in top_df.columns if c not in seen]]
            top_df[cols].to_csv(outdir / f"top{top_k}_geometry_table_{tag}_vs_{target_col}.csv", index=False, encoding="utf-8-sig")
            results["top_table"] = top_df[cols]

    return results


# ============================================================
# CLI
# ============================================================


def build_parser():
    p = argparse.ArgumentParser(description="完整增强版 layout 几何特征分析")
    p.add_argument("--all-csv", type=str, default="random_layout_5000_all_results.csv")
    p.add_argument("--reeval-csv", type=str, default="reeval_top50_summary.csv")
    p.add_argument("--outdir", type=str, default="feature_analysis_complete")
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--coarse-filter-quantile", type=float, default=0.995)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--rf-estimators", type=int, default=600)
    p.add_argument("--rf-perm-repeats", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--bin-step", type=float, default=0.02, help="等宽分箱步长；设为 <=0 时改用自动均匀 n_bins")
    p.add_argument("--min-bin-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260318)
    return p


# ============================================================
# Main
# ============================================================


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
    if "valid" in df_reeval_raw.columns:
        df_reeval_raw = df_reeval_raw[df_reeval_raw["valid"].astype(bool)].copy()

    print(f"[INFO] coarse CSV: {all_csv}")
    print(f"[INFO] reeval CSV: {reeval_csv}")
    print(f"[INFO] coarse rows raw: {len(df_all_raw)}")
    print(f"[INFO] reeval rows raw: {len(df_reeval_raw)}")

    df_all_feat = build_feature_table(df_all_raw, L=float(args.L))
    df_reeval_feat = build_feature_table(df_reeval_raw, L=float(args.L))

    df_all_feat.to_csv(outdir / "all5000_features.csv", index=False, encoding="utf-8-sig")
    df_reeval_feat.to_csv(outdir / "reeval_top50_features.csv", index=False, encoding="utf-8-sig")

    df_all_filtered = filter_upper_quantile(df_all_feat, "J_universal", float(args.coarse_filter_quantile)) if "J_universal" in df_all_feat.columns else df_all_feat.copy()
    df_all_filtered.to_csv(outdir / "all5000_features_filtered.csv", index=False, encoding="utf-8-sig")

    # ========================
    # Main targets
    # ========================
    result_bank: Dict[str, Dict[str, pd.DataFrame]] = {}

    if "J_universal" in df_all_feat.columns:
        result_bank["all5000_raw"] = analyze_target(
            df=df_all_feat,
            target_col="J_universal",
            tag="all5000_raw",
            outdir=outdir,
            seed=int(args.seed),
            bootstrap_n=int(args.bootstrap),
            rf_estimators=int(args.rf_estimators),
            rf_perm_repeats=int(args.rf_perm_repeats),
            top_k=int(args.top_k),
            L=float(args.L),
            n_bins=int(args.n_bins),
            bin_step=(float(args.bin_step) if float(args.bin_step) > 0 else None),
            min_bin_count=int(args.min_bin_count),
            do_bootstrap=False,
            do_top_gallery=False,
        )

    if "J_universal" in df_all_filtered.columns:
        result_bank["all5000_filtered"] = analyze_target(
            df=df_all_filtered,
            target_col="J_universal",
            tag="all5000_filtered",
            outdir=outdir,
            seed=int(args.seed),
            bootstrap_n=int(args.bootstrap),
            rf_estimators=int(args.rf_estimators),
            rf_perm_repeats=int(args.rf_perm_repeats),
            top_k=int(args.top_k),
            L=float(args.L),
            n_bins=int(args.n_bins),
            bin_step=(float(args.bin_step) if float(args.bin_step) > 0 else None),
            min_bin_count=int(args.min_bin_count),
            do_bootstrap=False,
            do_top_gallery=True,
        )

    if "mean_J" in df_reeval_feat.columns:
        result_bank["reeval_top50"] = analyze_target(
            df=df_reeval_feat,
            target_col="mean_J",
            tag="reeval_top50",
            outdir=outdir,
            seed=int(args.seed),
            bootstrap_n=int(args.bootstrap),
            rf_estimators=int(args.rf_estimators),
            rf_perm_repeats=int(args.rf_perm_repeats),
            top_k=min(int(args.top_k), len(df_reeval_feat)),
            L=float(args.L),
            n_bins=int(args.n_bins),
            bin_step=(float(args.bin_step) if float(args.bin_step) > 0 else None),
            min_bin_count=int(args.min_bin_count),
            do_bootstrap=True,
            do_top_gallery=True,
        )

    # ========================
    # Terrain-specific
    # ========================
    terrain_written: List[str] = []
    for tag, df_cur in [("all5000_filtered", df_all_filtered), ("reeval_top50", df_reeval_feat)]:
        for target in available_target_cols(df_cur, TERRAIN_TARGETS):
            analyze_target(
                df=df_cur,
                target_col=target,
                tag=f"{tag}_terrain",
                outdir=outdir,
                seed=int(args.seed),
                bootstrap_n=int(args.bootstrap),
                rf_estimators=int(args.rf_estimators),
                rf_perm_repeats=int(args.rf_perm_repeats),
                top_k=min(int(args.top_k), len(df_cur)),
                L=float(args.L),
                n_bins=int(args.n_bins),
                bin_step=(float(args.bin_step) if float(args.bin_step) > 0 else None),
                min_bin_count=int(args.min_bin_count),
                do_bootstrap=(tag == "reeval_top50"),
                do_top_gallery=False,
            )
            terrain_written.append(f"{tag}:{target}")

    # ========================
    # Coarse vs reeval validation on overlap
    # ========================
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

    # ========================
    # Summary
    # ========================
    lines: List[str] = []
    lines.append("Complete layout feature analysis summary")
    lines.append("=" * 72)
    lines.append(f"coarse_csv: {all_csv}")
    lines.append(f"reeval_csv: {reeval_csv}")
    lines.append(f"coarse_rows_raw: {len(df_all_feat)}")
    lines.append(f"coarse_rows_filtered: {len(df_all_filtered)}")
    lines.append(f"reeval_rows: {len(df_reeval_feat)}")
    lines.append("")

    if "all5000_filtered" in result_bank:
        lines.append("Top features | all5000 filtered | Spearman")
        lines.extend(top_feature_lines(result_bank["all5000_filtered"].get("corr", pd.DataFrame()), "spearman_rho"))
        lines.append("")
        lines.append("Random forest priority | all5000 filtered")
        lines.extend(top_rf_lines(result_bank["all5000_filtered"].get("rf", pd.DataFrame())))
        lines.append("")

    if "reeval_top50" in result_bank:
        lines.append("Top features | reeval top50 | Spearman")
        lines.extend(top_feature_lines(result_bank["reeval_top50"].get("corr", pd.DataFrame()), "spearman_rho"))
        lines.append("")
        lines.append("Random forest priority | reeval top50")
        lines.extend(top_rf_lines(result_bank["reeval_top50"].get("rf", pd.DataFrame())))
        lines.append("")

    if terrain_written:
        lines.append("Terrain-specific analyses generated")
        for item in terrain_written:
            lines.append(f"  - {item}")
        lines.append("")

    lines.append("Important new outputs")
    lines.append("  - pair_*.png : 关键二维几何特征图")
    lines.append("  - rf_importance_*.csv / bar_rf_*.png : 随机森林特征优先级")
    lines.append("  - top10_layouts_*.png : top10 layout 真图")
    lines.append("  - top10_geometry_profiles_*.png : top10 的最小间距/靠边比例/半径/角间隔对比")
    lines.append("  - binned_*.csv / binned_*.png : 非线性分箱分析")
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

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] outputs saved to: {outdir}")
    print(f"[DONE] summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
