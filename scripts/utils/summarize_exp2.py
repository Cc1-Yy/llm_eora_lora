from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "cls" / "exp2_summary_all.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "cls" / "report_exp2"


# ============================================================
# basic io / style
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#222222",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
        "patch.antialiased": True,
        "lines.antialiased": True,
        "text.color": "#222222",
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    df = pd.read_csv(path)

    numeric_cols = [
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "val_acc",
        "val_ce_loss",
        "val_kl_to_teacher",
        "val_mse_logits_to_teacher",
        "test_acc",
        "test_ce_loss",
        "test_kl_to_teacher",
        "test_mse_logits_to_teacher",
        "teacher_val_acc",
        "teacher_val_loss",
        "teacher_test_acc",
        "teacher_test_loss",
        "teacher_minus_test",
        "test_minus_teacher",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def style_axes(ax, add_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    if add_grid:
        ax.grid(True, axis="y", alpha=0.55, zorder=0)
    else:
        ax.grid(False)


def save_fig(fig: plt.Figure, out_path: Path, use_tight_layout: bool = True) -> None:
    if use_tight_layout:
        fig.tight_layout(pad=0.7)

    fig.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
        edgecolor="none",
    )

    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
        edgecolor="none",
    )

    plt.close(fig)
    print(f"[saved] {out_path}")
    print(f"[saved] {pdf_path}")


def fmt(x, nd: int = 4) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(x) for x in xs if pd.notna(x)]
    if len(vals) == 0:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    return mu, sd


# ============================================================
# data helpers
# ============================================================

def get_main_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["source"] == "main_exp2") & (df["branch"].isin(["LoRA", "EoRA"]))].copy()
    return out.reset_index(drop=True)


def get_confirm_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["source"] == "confirm_multiseed_exp2") & (df["branch"].isin(["LoRA", "EoRA"]))].copy()
    return out.reset_index(drop=True)


def get_teacher_test_acc(df: pd.DataFrame) -> Optional[float]:
    if "teacher_test_acc" in df.columns and df["teacher_test_acc"].notna().any():
        return float(df["teacher_test_acc"].dropna().iloc[0])
    return None


def get_teacher_val_acc(df: pd.DataFrame) -> Optional[float]:
    if "teacher_val_acc" in df.columns and df["teacher_val_acc"].notna().any():
        return float(df["teacher_val_acc"].dropna().iloc[0])
    return None


def ar_marker(ar: float) -> str:
    if np.isclose(ar, 1.0):
        return "o"
    if np.isclose(ar, 1.25):
        return "s"
    return "^"


def branch_color(branch: str) -> str:
    return {
        "LoRA": "#e45238",
        "EoRA": "#7dacd1",
        "Teacher": "#E8C86E",
    }.get(branch, "#666666")


def branch_ar_color(branch: str, ar: float) -> str:
    # same branch family, different saturation
    if branch == "LoRA":
        if np.isclose(ar, 1.0):
            return "#e45238"
        if np.isclose(ar, 1.25):
            return "#c43c25"
        return "#f07b65"
    if branch == "EoRA":
        if np.isclose(ar, 1.0):
            return "#7dacd1"
        if np.isclose(ar, 1.25):
            return "#5f97c4"
        return "#9fc2df"
    return "#666666"


def label_row(r: pd.Series) -> str:
    return f"{r['branch']}-r{int(r['rank'])}-ar{float(r['alpha_over_r']):g}"


# ============================================================
# tables
# ============================================================

def make_main_results_table(main_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "branch",
        "exp_name",
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "test_acc",
        "test_ce_loss",
        "test_kl_to_teacher",
        "test_mse_logits_to_teacher",
        "teacher_minus_test",
        "run_dir",
    ]
    keep = [c for c in cols if c in main_df.columns]
    out = main_df[keep].copy()
    out = out.sort_values(["branch", "rank", "alpha_over_r", "seed"], ascending=True)
    return out.reset_index(drop=True)


def make_confirm_summary_table(confirm_df: pd.DataFrame) -> pd.DataFrame:
    if len(confirm_df) == 0:
        return pd.DataFrame()

    rows = []
    grouped = confirm_df.groupby(["branch", "rank", "alpha_over_r"], dropna=False)
    for (branch, rank, ar), g in grouped:
        mu_acc, sd_acc = mean_std(g["test_acc"].tolist())
        mu_kl, sd_kl = mean_std(g["test_kl_to_teacher"].tolist())
        mu_mse, sd_mse = mean_std(g["test_mse_logits_to_teacher"].tolist())
        mu_gap, sd_gap = mean_std(g["teacher_minus_test"].tolist())
        rows.append({
            "branch": branch,
            "rank": rank,
            "alpha_over_r": ar,
            "n": len(g),
            "test_acc_mean": mu_acc,
            "test_acc_std": sd_acc,
            "test_kl_mean": mu_kl,
            "test_kl_std": sd_kl,
            "test_mse_mean": mu_mse,
            "test_mse_std": sd_mse,
            "teacher_minus_test_mean": mu_gap,
            "teacher_minus_test_std": sd_gap,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["branch", "rank", "alpha_over_r"]).reset_index(drop=True)
    return out


# ============================================================
# plots
# ============================================================

def plot_optimized_model_gap_vs_rank(main_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ars = sorted(main_df["alpha_over_r"].dropna().unique())

    for branch in ["LoRA", "EoRA"]:
        for ar in ars:
            sub = main_df[
                (main_df["branch"] == branch)
                & (np.isclose(main_df["alpha_over_r"], ar, equal_nan=False))
            ].sort_values("rank")
            if len(sub) == 0:
                continue
            ax.plot(
                sub["rank"].to_numpy(),
                sub["teacher_minus_test"].to_numpy(),
                marker=ar_marker(ar),
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.4,
                label=f"{branch} (α/r={ar:g})",
                color=branch_ar_color(branch, float(ar)),
            )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Optimized Model Accuracy - Test Accuracy")
    ax.set_title("Exp2-cls: Gap to Optimized Model vs Rank")
    ax.set_xticks(sorted(main_df["rank"].dropna().astype(int).unique()))
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir / "fig_exp2_cls_optimized_model_gap_vs_rank.png")


def plot_ranked_kl_bar(main_df: pd.DataFrame, out_dir: Path):
    df = main_df.copy()
    df["label"] = df.apply(label_row, axis=1)
    df = df.sort_values("test_kl_to_teacher", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    bars = ax.bar(
        np.arange(len(df)),
        df["test_kl_to_teacher"].to_numpy(),
        color=[branch_ar_color(r["branch"], float(r["alpha_over_r"])) for _, r in df.iterrows()],
        edgecolor="#333333",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"].tolist(), rotation=40, ha="right")
    ax.set_ylabel("Test KL to Optimized Model")
    ax.set_title("Exp2-cls: Ranked KL-to-Teacher Comparison")
    style_axes(ax, add_grid=True)

    for b, v in zip(bars, df["test_kl_to_teacher"].to_numpy()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(v) + 0.001,
            f"{float(v):.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    save_fig(fig, out_dir / "fig_exp2_cls_ranked_kl_bar.png")


def plot_acc_kl_tradeoff(main_df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for _, r in main_df.iterrows():
        ax.scatter(
            float(r["test_kl_to_teacher"]),
            float(r["test_acc"]),
            s=45 + 1.8 * float(r["rank"]),
            color=branch_ar_color(r["branch"], float(r["alpha_over_r"])),
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            marker=ar_marker(float(r["alpha_over_r"])),
            zorder=3,
        )
        ax.text(
            float(r["test_kl_to_teacher"]) + 0.0015,
            float(r["test_acc"]) + 0.0007,
            f"{r['branch'][0]}-r{int(r['rank'])}",
            fontsize=8.3,
        )

    if teacher_acc is not None:
        ax.axhline(
            y=float(teacher_acc),
            linestyle="--",
            linewidth=1.5,
            color="#444444",
            alpha=0.95,
            zorder=2,
            label=f"Optimized model acc = {teacher_acc:.3f}",
        )

    ax.set_xlabel("Test KL to Optimized Model")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp2-cls: Accuracy–KL Trade-off")
    style_axes(ax, add_grid=True)
    if teacher_acc is not None:
        ax.legend(frameon=False, loc="lower right")

    save_fig(fig, out_dir / "fig_exp2_cls_acc_kl_tradeoff.png")


def plot_main_rank_metrics_triptych(main_df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0))
    metrics = [
        ("test_acc", "Test Accuracy", teacher_acc),
        ("test_kl_to_teacher", "Test KL to Optimized Model", None),
        ("test_mse_logits_to_teacher", "Test Logits MSE to Optimized Model", None),
    ]

    ars = sorted(main_df["alpha_over_r"].dropna().unique())

    for ax, (metric, ylabel, teacher_line) in zip(axes, metrics):
        for branch in ["LoRA", "EoRA"]:
            for ar in ars:
                sub = main_df[
                    (main_df["branch"] == branch)
                    & (np.isclose(main_df["alpha_over_r"], ar, equal_nan=False))
                ].sort_values("rank")
                if len(sub) == 0:
                    continue
                ax.plot(
                    sub["rank"].to_numpy(),
                    sub[metric].to_numpy(),
                    marker=ar_marker(ar),
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    linewidth=2.4,
                    label=f"{branch} (α/r={ar:g})",
                    color=branch_ar_color(branch, float(ar)),
                )

        if teacher_line is not None:
            ax.axhline(
                y=float(teacher_line),
                linestyle="--",
                linewidth=1.5,
                color="#444444",
                alpha=0.95,
            )

        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(sorted(main_df["rank"].dropna().astype(int).unique()))
        style_axes(ax, add_grid=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Exp2-cls: Main Rank-wise Comparison", y=1.11, fontsize=15, fontweight="semibold")

    save_fig(fig, out_dir / "fig_exp2_cls_main_rank_metrics_triptych.png")


def plot_confirm_stability_triptych(confirm_df: pd.DataFrame, out_dir: Path):
    if len(confirm_df) == 0:
        return

    grouped = confirm_df.groupby(["branch", "rank", "alpha_over_r"], dropna=False)
    rows = []
    for (branch, rank, ar), g in grouped:
        mu_acc, sd_acc = mean_std(g["test_acc"].tolist())
        mu_kl, sd_kl = mean_std(g["test_kl_to_teacher"].tolist())
        mu_mse, sd_mse = mean_std(g["test_mse_logits_to_teacher"].tolist())
        rows.append({
            "label": f"{branch}-r{int(rank)}-ar{float(ar):g}",
            "branch": branch,
            "rank": rank,
            "alpha_over_r": ar,
            "acc_mean": mu_acc,
            "acc_std": sd_acc,
            "kl_mean": mu_kl,
            "kl_std": sd_kl,
            "mse_mean": mu_mse,
            "mse_std": sd_mse,
        })

    plot_df = pd.DataFrame(rows).sort_values(["branch", "rank", "alpha_over_r"]).reset_index(drop=True)
    xs = np.arange(len(plot_df))
    colors = [branch_ar_color(r["branch"], float(r["alpha_over_r"])) for _, r in plot_df.iterrows()]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))
    metric_specs = [
        ("acc_mean", "acc_std", "Mean Test Accuracy"),
        ("kl_mean", "kl_std", "Mean Test KL to Optimized Model"),
        ("mse_mean", "mse_std", "Mean Test Logits MSE"),
    ]

    for ax, (mcol, scol, title) in zip(axes, metric_specs):
        bars = ax.bar(
            xs,
            plot_df[mcol].to_numpy(),
            yerr=plot_df[scol].to_numpy(),
            capsize=4,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
            zorder=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(plot_df["label"].tolist(), rotation=24, ha="right")
        ax.set_title(title)
        style_axes(ax, add_grid=True)

        for b, v in zip(bars, plot_df[mcol].to_numpy()):
            ax.text(
                b.get_x() + b.get_width() / 2,
                float(v) + (0.001 if "acc" in mcol else 0.003),
                f"{float(v):.3f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
            )

    fig.suptitle("Exp2-cls: Confirmatory Stability Comparison", y=1.05, fontsize=15, fontweight="semibold")
    save_fig(fig, out_dir / "fig_exp2_cls_confirm_stability_triptych.png")


def _build_metric_grid(df: pd.DataFrame, branch: str, metric: str):
    sub = df[df["branch"] == branch].copy()
    ranks = sorted(sub["rank"].dropna().astype(int).unique())
    ars = sorted(sub["alpha_over_r"].dropna().unique())
    grid = np.full((len(ranks), len(ars)), np.nan)

    for i, r in enumerate(ranks):
        for j, ar in enumerate(ars):
            cell = sub[
                (sub["rank"] == r)
                & (np.isclose(sub["alpha_over_r"], ar, equal_nan=False))
            ]
            if len(cell) > 0:
                grid[i, j] = float(cell.iloc[0][metric])

    return ranks, ars, grid


def plot_combined_heatmaps(main_df: pd.DataFrame, out_dir: Path):
    branches = ["LoRA", "EoRA"]
    metrics = [
        ("test_acc", "Accuracy", "RdYlBu_r"),
        ("test_kl_to_teacher", "KL", "RdYlBu"),
        ("test_mse_logits_to_teacher", "MSE", "RdYlBu"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.0))
    for i, branch in enumerate(branches):
        for j, (metric, title, cmap_name) in enumerate(metrics):
            ax = axes[i, j]
            ranks, ars, grid = _build_metric_grid(main_df, branch, metric)

            if grid.size == 0:
                ax.set_visible(False)
                continue

            im = ax.imshow(grid, cmap=plt.get_cmap(cmap_name), aspect="auto")
            ax.set_xticks(np.arange(len(ars)))
            ax.set_xticklabels([f"{x:g}" for x in ars])
            ax.set_yticks(np.arange(len(ranks)))
            ax.set_yticklabels([str(x) for x in ranks])

            ax.set_xlabel("α/r")
            ax.set_ylabel("Rank")
            ax.set_title(f"{branch}: {title}")

            for r_idx in range(len(ranks)):
                for c_idx in range(len(ars)):
                    val = grid[r_idx, c_idx]
                    if np.isnan(val):
                        continue
                    ax.text(
                        c_idx,
                        r_idx,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        color="#222222",
                    )

            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8.5)

    fig.suptitle("Exp2-cls: Combined Heatmaps", y=1.02, fontsize=15, fontweight="semibold")
    save_fig(fig, out_dir / "fig_exp2_cls_combined_heatmaps.png", use_tight_layout=True)


# ============================================================
# main
# ============================================================

def main():
    set_plot_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    summary_csv = Path(args.summary_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    df = load_summary_csv(summary_csv)
    teacher_acc = get_teacher_test_acc(df)

    main_df = get_main_rows(df)
    confirm_df = get_confirm_rows(df)

    save_df(make_main_results_table(main_df), out_dir / "table_exp2_cls_main_results.csv")
    save_df(make_confirm_summary_table(confirm_df), out_dir / "table_exp2_cls_confirm_summary.csv")

    plot_optimized_model_gap_vs_rank(main_df, out_dir)
    plot_ranked_kl_bar(main_df, out_dir)
    plot_acc_kl_tradeoff(main_df, teacher_acc, out_dir)
    plot_main_rank_metrics_triptych(main_df, teacher_acc, out_dir)
    plot_confirm_stability_triptych(confirm_df, out_dir)
    plot_combined_heatmaps(main_df, out_dir)

    print("\nDone. Figures and tables saved under:")
    print(out_dir)


if __name__ == "__main__":
    main()