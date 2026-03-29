from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "cls" / "exp1_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "cls" / "report_exp1_figs"


# ============================================================
# global style (aligned with exp2 style)
# ============================================================

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


# ============================================================
# helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    df = pd.read_csv(path)

    num_cols = [
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "val_acc",
        "val_loss",
        "test_acc",
        "test_loss",
        "teacher_test_acc",
        "test_minus_teacher",
        "teacher_minus_test",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def get_teacher_acc(df: pd.DataFrame) -> Optional[float]:
    if "branch" in df.columns:
        tdf = df[df["branch"] == "Teacher"]
        if len(tdf) > 0 and tdf["test_acc"].notna().any():
            return float(tdf["test_acc"].dropna().iloc[0])

    if "teacher_test_acc" in df.columns and df["teacher_test_acc"].notna().any():
        return float(df["teacher_test_acc"].dropna().iloc[0])

    return None


def get_exp_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["branch"].isin(["LoRA", "EoRA"])].copy()


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def save_fig(fig: plt.Figure, out_path: Path, use_tight_layout: bool = True) -> None:
    if use_tight_layout:
        fig.tight_layout(pad=0.7)

    fig.savefig(
        out_path,
        dpi=900,
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


def style_axes(ax, add_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    if add_grid:
        ax.grid(True, axis="y", alpha=0.55)
    else:
        ax.grid(False)


def cmap_rdylbu():
    return plt.get_cmap("RdYlBu")


def single_method_colors():
    # aligned with exp2
    return {
        "Teacher": "#fde395",
        "LoRA": "#e45238",
        "EoRA": "#7dacd1",
    }


def matched_setting_colors():
    cmap = cmap_rdylbu()
    return {
        ("LoRA", 1.0): cmap(0.15),
        ("LoRA", 4.0): cmap(0.30),
        ("EoRA", 1.0): cmap(0.75),
        ("EoRA", 4.0): cmap(0.90),
    }


def eora_rank_colors():
    cmap = cmap_rdylbu()
    return {
        4: cmap(0.15),
        8: cmap(0.28),
        16: cmap(0.45),
        32: cmap(0.72),
        64: cmap(0.90),
    }


def eora_ar_colors():
    cmap = cmap_rdylbu()
    return {
        1.0: cmap(0.20),
        1.25: cmap(0.55),
        1.5: cmap(0.85),
    }


def add_branch_gap_positions(branches: List[str], gap: float = 0.6) -> List[float]:
    xs = []
    x = 0.0
    prev_branch = None
    for b in branches:
        if prev_branch is not None and b != prev_branch:
            x += gap
        xs.append(x)
        x += 1.0
        prev_branch = b
    return xs


# ============================================================
# tables
# ============================================================

def make_main_summary_table(df: pd.DataFrame, teacher_acc: Optional[float]) -> pd.DataFrame:
    exp_df = get_exp_rows(df)

    def best_of(branch: str):
        sub = exp_df[(exp_df["branch"] == branch) & exp_df["test_acc"].notna()].copy()
        if len(sub) == 0:
            return None
        return sub.sort_values("test_acc", ascending=False).iloc[0]

    rows = []

    if teacher_acc is not None:
        rows.append({
            "method": "Optimized model",
            "branch": "Teacher",
            "rank": None,
            "alpha": None,
            "alpha_over_r": None,
            "val_acc": None,
            "test_acc": teacher_acc,
            "teacher_minus_test": 0.0,
            "exp_name": "optimized_model",
        })

    best_lora = best_of("LoRA")
    if best_lora is not None:
        rows.append({
            "method": "Best LoRA",
            "branch": "LoRA",
            "rank": best_lora["rank"],
            "alpha": best_lora["alpha"],
            "alpha_over_r": best_lora["alpha_over_r"],
            "val_acc": best_lora["val_acc"],
            "test_acc": best_lora["test_acc"],
            "teacher_minus_test": best_lora["teacher_minus_test"],
            "exp_name": best_lora["exp_name"],
        })

    best_eora = best_of("EoRA")
    if best_eora is not None:
        rows.append({
            "method": "Best EoRA",
            "branch": "EoRA",
            "rank": best_eora["rank"],
            "alpha": best_eora["alpha"],
            "alpha_over_r": best_eora["alpha_over_r"],
            "val_acc": best_eora["val_acc"],
            "test_acc": best_eora["test_acc"],
            "teacher_minus_test": best_eora["teacher_minus_test"],
            "exp_name": best_eora["exp_name"],
        })

    reps = [
        ("LoRA", 4, 1.0),
        ("LoRA", 8, 1.0),
        ("LoRA", 16, 4.0),
        ("EoRA", 16, 1.25),
        ("EoRA", 32, 1.25),
        ("EoRA", 64, 1.25),
    ]
    for branch, rank, ar in reps:
        sub = exp_df[
            (exp_df["branch"] == branch)
            & (exp_df["rank"] == rank)
            & (exp_df["alpha_over_r"] == ar)
        ]
        if len(sub) == 0:
            continue
        r = sub.iloc[0]
        rows.append({
            "method": f"{branch} r{int(rank)} ar{ar:g}",
            "branch": branch,
            "rank": r["rank"],
            "alpha": r["alpha"],
            "alpha_over_r": r["alpha_over_r"],
            "val_acc": r["val_acc"],
            "test_acc": r["test_acc"],
            "teacher_minus_test": r["teacher_minus_test"],
            "exp_name": r["exp_name"],
        })

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["exp_name"], keep="first")
    return out


def make_full_results_table(df: pd.DataFrame) -> pd.DataFrame:
    exp_df = get_exp_rows(df).copy()
    cols = [
        "branch",
        "exp_name",
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "val_loss",
        "val_acc",
        "test_loss",
        "test_acc",
        "teacher_minus_test",
        "run_dir",
    ]
    cols = [c for c in cols if c in exp_df.columns]
    exp_df = exp_df[cols].sort_values(
        by=["branch", "rank", "alpha_over_r"],
        ascending=[True, True, True],
    )
    return exp_df


# ============================================================
# plots
# ============================================================

def plot_overall_bar(df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path):
    exp_df = get_exp_rows(df)

    rows = []

    if teacher_acc is not None:
        rows.append({
            "label": "Optimized",
            "value": teacher_acc,
            "branch": "Teacher",
        })

    reps = [
        ("LoRA", 4, 1.0),
        ("LoRA", 8, 1.0),
        ("LoRA", 16, 4.0),
        ("EoRA", 16, 1.25),
        ("EoRA", 32, 1.25),
        ("EoRA", 64, 1.25),
    ]
    for branch, rank, ar in reps:
        sub = exp_df[
            (exp_df["branch"] == branch)
            & (exp_df["rank"] == rank)
            & (exp_df["alpha_over_r"] == ar)
        ]
        if len(sub) == 0:
            continue
        rows.append({
            "label": f"{branch}\nr{rank}, ar={ar:g}",
            "value": float(sub.iloc[0]["test_acc"]),
            "branch": branch,
        })

    colors = single_method_colors()
    labels = [r["label"] for r in rows]
    vals = [r["value"] for r in rows]
    branches = [r["branch"] for r in rows]
    xs = add_branch_gap_positions(branches)

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bars = ax.bar(
        xs,
        vals,
        width=0.72,
        color=[colors[b] for b in branches],
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: Overall Comparison")
    ax.set_ylim(min(vals) - 0.03, max(vals) + 0.012)

    if teacher_acc is not None:
        ax.axhline(
            teacher_acc,
            linestyle="--",
            linewidth=1.5,
            color="#444444",
        )

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.0015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )

    style_axes(ax, add_grid=True)
    save_fig(fig, out_dir / "fig_exp1_cls_overall_bar.png")


def plot_matched_rank(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df)
    rank_list = [4, 8, 16]
    colors = matched_setting_colors()

    settings = [
        ("LoRA", 1.0, "LoRA (α/r=1)", colors[("LoRA", 1.0)]),
        ("EoRA", 1.0, "EoRA (α/r=1)", colors[("EoRA", 1.0)]),
        ("LoRA", 4.0, "LoRA (α/r=4)", colors[("LoRA", 4.0)]),
        ("EoRA", 4.0, "EoRA (α/r=4)", colors[("EoRA", 4.0)]),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    plotted_any = False

    for branch, ar, label, color in settings:
        xs, ys = [], []
        for r in rank_list:
            sub = exp_df[
                (exp_df["branch"] == branch)
                & (exp_df["rank"] == r)
                & (np.isclose(exp_df["alpha_over_r"], ar, equal_nan=False))
            ]
            if len(sub) == 0:
                continue
            xs.append(r)
            ys.append(float(sub.iloc[0]["test_acc"]))
        if len(xs) > 0:
            ax.plot(
                xs,
                ys,
                marker="o",
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.4,
                label=label,
                color=color,
            )
            plotted_any = True

    ax.set_xlabel("Rank")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: Matched-setting Comparison")
    ax.set_xticks(rank_list)
    style_axes(ax, add_grid=True)
    if plotted_any:
        ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir / "fig_exp1_cls_matched_rank.png")


def plot_eora_scaling(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df)
    eora = exp_df[exp_df["branch"] == "EoRA"].copy()
    rank_colors = eora_rank_colors()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    valid_ranks: List[int] = []
    for r in sorted(eora["rank"].dropna().unique()):
        sub = eora[eora["rank"] == r]
        if sub["alpha_over_r"].notna().sum() >= 2:
            valid_ranks.append(int(r))

    for r in valid_ranks:
        sub = eora[eora["rank"] == r].sort_values("alpha_over_r")
        ax.plot(
            sub["alpha_over_r"].to_numpy(),
            sub["test_acc"].to_numpy(),
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.4,
            label=f"r={r}",
            color=rank_colors.get(int(r), "#7dacd1"),
        )

    ax.set_xlabel("α/r")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: EoRA Scaling Sensitivity")
    style_axes(ax, add_grid=True)
    if len(valid_ranks) > 0:
        ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir / "fig_exp1_cls_eora_scaling.png")


def plot_eora_rank_effect(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df)
    eora = exp_df[exp_df["branch"] == "EoRA"].copy()
    ar_colors = eora_ar_colors()

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    target_ars = [1.0, 1.25, 1.5]
    plotted_any = False
    for ar in target_ars:
        sub = eora[np.isclose(eora["alpha_over_r"], ar, equal_nan=False)].sort_values("rank")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["rank"].to_numpy(),
            sub["test_acc"].to_numpy(),
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.4,
            label=f"α/r={ar:g}",
            color=ar_colors.get(ar, "#7dacd1"),
        )
        plotted_any = True

    ax.set_xlabel("Rank")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: EoRA Rank Effect")
    ax.set_xticks(sorted(eora["rank"].dropna().unique()))
    style_axes(ax, add_grid=True)
    if plotted_any:
        ax.legend(frameon=False)

    save_fig(fig, out_dir / "fig_exp1_cls_eora_rank_effect.png")


def plot_gap_to_teacher(df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    if teacher_acc is None:
        return

    if "teacher_minus_test" not in exp_df.columns or exp_df["teacher_minus_test"].isna().all():
        exp_df["teacher_minus_test"] = teacher_acc - exp_df["test_acc"]

    exp_df = exp_df.sort_values(["branch", "teacher_minus_test"], ascending=[True, True]).copy()
    exp_df["label"] = exp_df.apply(
        lambda r: f"{r['branch']}\nr{int(r['rank'])}, ar={r['alpha_over_r']:g}",
        axis=1,
    )

    colors = single_method_colors()
    branches = exp_df["branch"].tolist()
    xs = add_branch_gap_positions(branches)

    fig, ax = plt.subplots(figsize=(12.0, 5.7))
    bars = ax.bar(
        xs,
        exp_df["teacher_minus_test"].to_numpy(),
        width=0.72,
        color=[colors.get(b, "#7dacd1") for b in branches],
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(exp_df["label"].tolist(), rotation=70, ha="right")
    ax.set_ylabel("Optimized Model Accuracy - Test Accuracy")
    ax.set_title("Exp1-cls: Gap to Optimized Model")

    for b, v in zip(bars, exp_df["teacher_minus_test"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(v) + 0.0015,
            f"{float(v):.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )

    style_axes(ax, add_grid=True)
    save_fig(fig, out_dir / "fig_exp1_cls_gap_to_teacher.png")


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

    df = load_summary(summary_csv)
    teacher_acc = get_teacher_acc(df)

    main_table = make_main_summary_table(df, teacher_acc)
    full_table = make_full_results_table(df)
    save_df(main_table, out_dir / "table_exp1_cls_main_summary.csv")
    save_df(full_table, out_dir / "table_exp1_cls_full_results.csv")

    plot_overall_bar(df, teacher_acc, out_dir)
    plot_matched_rank(df, out_dir)
    plot_eora_scaling(df, out_dir)
    plot_eora_rank_effect(df, out_dir)
    plot_gap_to_teacher(df, teacher_acc, out_dir)

    print(f"Saved report figures and tables to: {out_dir}")


if __name__ == "__main__":
    main()