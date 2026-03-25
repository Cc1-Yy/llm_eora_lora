from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "cls" / "exp1_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "cls" / "report_exp1"


# ============================================================
# basic io / style
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def load_summary_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def get_teacher_acc(df: pd.DataFrame) -> Optional[float]:
    if "branch" not in df.columns:
        return None
    sub = df[df["branch"] == "Teacher"].copy()
    if len(sub) == 0:
        return None
    if "test_acc" not in sub.columns:
        return None
    vals = sub["test_acc"].dropna().tolist()
    if len(vals) == 0:
        return None
    return float(vals[0])


def get_exp_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["branch"].isin(["LoRA", "EoRA"])].copy()


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def style_axes(ax, add_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    if add_grid:
        ax.grid(True, axis="y", alpha=0.55)
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


def _same_float(a, b, tol: float = 1e-9) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


# ============================================================
# color helpers (aligned with exp2)
# ============================================================

def cmap_rdylbu():
    return plt.get_cmap("RdYlBu")


def method_colors():
    """
    Match Exp2 logic:
      - LoRA uses warmer tones
      - EoRA uses cooler tones
    """
    cmap = cmap_rdylbu()
    return {
        ("LoRA", 1.0): cmap(0.15),
        ("LoRA", 4.0): cmap(0.30),
        ("EoRA", 1.0): cmap(0.75),
        ("EoRA", 4.0): cmap(0.90),
    }


def single_method_colors():
    return {
        "LoRA": "#e45238",
        "EoRA": "#7dacd1",
        "Teacher": "#666666",
    }


def extra_colors():
    return {
        "teal": "#4BA3A6",
        "orange": "#F39C6B",
        "purple": "#9C89D9",
        "gold": "#C9A227",
        "gray": "#666666",
    }


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

    for branch in ["LoRA", "EoRA"]:
        r = best_of(branch)
        if r is None:
            continue
        rows.append({
            "method": f"Best {branch}",
            "branch": branch,
            "rank": int(r["rank"]) if pd.notna(r["rank"]) else None,
            "alpha": float(r["alpha"]) if pd.notna(r["alpha"]) else None,
            "alpha_over_r": float(r["alpha_over_r"]) if pd.notna(r["alpha_over_r"]) else None,
            "val_acc": float(r["val_acc"]) if pd.notna(r["val_acc"]) else None,
            "test_acc": float(r["test_acc"]) if pd.notna(r["test_acc"]) else None,
            "teacher_minus_test": float(r["teacher_minus_test"]) if pd.notna(r["teacher_minus_test"]) else None,
            "exp_name": r["exp_name"],
        })

    return pd.DataFrame(rows)


def make_full_results_table(df: pd.DataFrame) -> pd.DataFrame:
    exp_df = get_exp_rows(df).copy()
    cols = [
        "branch",
        "exp_name",
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "val_acc",
        "test_acc",
        "teacher_minus_test",
    ]
    keep = [c for c in cols if c in exp_df.columns]
    out = exp_df[keep].copy()
    out = out.sort_values(["branch", "rank", "alpha_over_r", "seed"], ascending=True)
    return out.reset_index(drop=True)


# ============================================================
# plots
# ============================================================

def plot_overall_bar(df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path) -> None:
    exp_df = get_exp_rows(df).copy()

    rows = []
    for _, r in exp_df.iterrows():
        if pd.isna(r.get("test_acc", np.nan)):
            continue
        label = f"{r['branch']}-r{int(r['rank'])}-ar{float(r['alpha_over_r']):g}"
        rows.append({
            "key": (r["branch"], int(r["rank"]), float(r["alpha_over_r"])),
            "label": label,
            "value": float(r["test_acc"]),
            "branch": r["branch"],
            "rank": int(r["rank"]),
            "alpha_over_r": float(r["alpha_over_r"]),
            "is_best": False,
        })

    # mark best per branch
    for branch in ["LoRA", "EoRA"]:
        sub = [x for x in rows if x["branch"] == branch]
        if len(sub) == 0:
            continue
        best = max(sub, key=lambda x: x["value"])
        best["is_best"] = True

    dedup = {}
    for r in rows:
        if r["key"] not in dedup:
            dedup[r["key"]] = r
        else:
            if r["is_best"] and not dedup[r["key"]]["is_best"]:
                dedup[r["key"]] = r

    clean_rows = list(dedup.values())

    branch_order = {"Teacher": 0, "LoRA": 1, "EoRA": 2}
    clean_rows = sorted(
        clean_rows,
        key=lambda r: (
            branch_order.get(r["branch"], 99),
            0 if r["is_best"] else 1,
            r["rank"] if r["rank"] is not None else -1,
            r["alpha_over_r"] if r["alpha_over_r"] is not None else -1,
        ),
    )

    color_map = single_method_colors()
    colors = [color_map[r["branch"]] for r in clean_rows]

    xs = []
    x = 0.0
    prev_branch = None
    for r in clean_rows:
        if prev_branch is not None and r["branch"] != prev_branch:
            x += 0.6
        xs.append(x)
        x += 1.0
        prev_branch = r["branch"]

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    bars = ax.bar(
        xs,
        [r["value"] for r in clean_rows],
        width=0.72,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
        zorder=3,
    )

    if teacher_acc is not None:
        ax.axhline(
            y=float(teacher_acc),
            linestyle="--",
            linewidth=1.5,
            color="#444444",
            alpha=0.95,
            zorder=2,
            label=f"Optimized model = {teacher_acc:.3f}",
        )

    vals = [r["value"] for r in clean_rows]
    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in clean_rows], rotation=42, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: Optimized Model vs LoRA vs EoRA")
    upper = max(vals) if len(vals) > 0 else 0.0
    if teacher_acc is not None:
        upper = max(upper, float(teacher_acc))
    ax.set_ylim(0.75, upper + 0.035)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.0035,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    style_axes(ax, add_grid=True)
    if teacher_acc is not None:
        ax.legend(frameon=False, loc="lower left")

    save_fig(fig, out_dir / "fig_exp1_cls_overall_bar.png")


def plot_matched_rank(df: pd.DataFrame, out_dir: Path) -> None:
    exp_df = get_exp_rows(df)
    rank_list = [4, 8, 16]

    colors = method_colors()

    settings = [
        ("LoRA", 1.0, "LoRA (α/r=1)"),
        ("EoRA", 1.0, "EoRA (α/r=1)"),
        ("LoRA", 4.0, "LoRA (α/r=4)"),
        ("EoRA", 4.0, "EoRA (α/r=4)"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for branch, ar, label in settings:
        xs, ys = [], []
        for r in rank_list:
            sub = exp_df[
                (exp_df["branch"] == branch)
                & (exp_df["rank"] == r)
                & (exp_df["alpha_over_r"].apply(lambda x: _same_float(x, ar)))
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
                color=colors[(branch, ar)],
            )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: Matched-Setting Comparison")
    ax.set_xticks(rank_list)
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir / "fig_exp1_cls_matched_rank.png")


def plot_eora_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    exp_df = get_exp_rows(df)
    eora = exp_df[exp_df["branch"] == "EoRA"].copy()

    rank_color_map = {
        4: "#F39C6B",
        8: "#D97B66",
        16: "#9C89D9",
        32: "#7dacd1",
        64: "#4BA3A6",
    }

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    valid_ranks: List[int] = []
    for r in sorted(eora["rank"].dropna().unique()):
        sub = eora[eora["rank"] == r]
        if sub["alpha_over_r"].notna().sum() >= 2:
            valid_ranks.append(int(r))

    for r in valid_ranks:
        sub = eora[eora["rank"] == r].sort_values("alpha_over_r")
        ax.plot(
            sub["alpha_over_r"],
            sub["test_acc"],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.4,
            label=f"r={r}",
            color=rank_color_map.get(int(r), "#7dacd1"),
        )

    ax.set_xlabel("Alpha / Rank")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: EoRA Scaling Sensitivity")
    style_axes(ax, add_grid=True)
    if len(valid_ranks) > 0:
        ax.legend(frameon=False)

    save_fig(fig, out_dir / "fig_exp1_cls_eora_scaling.png")


def plot_eora_rank_effect(df: pd.DataFrame, out_dir: Path) -> None:
    exp_df = get_exp_rows(df)
    eora = exp_df[exp_df["branch"] == "EoRA"].copy()

    target_ars = [0.75, 1.0, 1.25, 1.5]
    colors = {
        0.75: "#F39C6B",
        1.0: "#9C89D9",
        1.25: "#7dacd1",
        1.5: "#4BA3A6",
    }

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for ar in target_ars:
        sub = eora[eora["alpha_over_r"].apply(lambda x: _same_float(x, ar))].sort_values("rank")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["rank"],
            sub["test_acc"],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.4,
            label=f"α/r={ar:g}",
            color=colors[ar],
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Exp1-cls: EoRA Rank Effect")
    ax.set_xticks(sorted(eora["rank"].dropna().astype(int).unique()))
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir / "fig_exp1_cls_eora_rank_effect.png")


def plot_gap_to_teacher(df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path) -> None:
    if teacher_acc is None:
        return

    exp_df = get_exp_rows(df).copy()
    exp_df["teacher_gap"] = exp_df["teacher_minus_test"]

    exp_df = exp_df.sort_values(["branch", "rank", "alpha_over_r"]).reset_index(drop=True)

    labels = [
        f"{r['branch']}-r{int(r['rank'])}-ar{float(r['alpha_over_r']):g}"
        for _, r in exp_df.iterrows()
    ]
    colors = [single_method_colors().get(b, "#666666") for b in exp_df["branch"]]

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    bars = ax.bar(
        np.arange(len(exp_df)),
        exp_df["teacher_gap"].to_numpy(),
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax.set_xticks(np.arange(len(exp_df)))
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("Optimized - Test Accuracy")
    ax.set_title("Exp1-cls: Accuracy Gap to Optimized Model")
    style_axes(ax, add_grid=True)

    for b, v in zip(bars, exp_df["teacher_gap"].to_numpy()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.0007,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    save_fig(fig, out_dir / "fig_exp1_cls_gap_to_teacher.png")


def plot_triptych(df: pd.DataFrame, teacher_acc: Optional[float], out_dir: Path) -> None:
    exp_df = get_exp_rows(df).copy()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # panel 1: matched comparison
    colors = method_colors()
    rank_list = [4, 8, 16]
    settings = [
        ("LoRA", 1.0, "LoRA (α/r=1)"),
        ("EoRA", 1.0, "EoRA (α/r=1)"),
        ("LoRA", 4.0, "LoRA (α/r=4)"),
        ("EoRA", 4.0, "EoRA (α/r=4)"),
    ]
    for branch, ar, label in settings:
        xs, ys = [], []
        for r in rank_list:
            sub = exp_df[
                (exp_df["branch"] == branch)
                & (exp_df["rank"] == r)
                & (exp_df["alpha_over_r"].apply(lambda x: _same_float(x, ar)))
            ]
            if len(sub) == 0:
                continue
            xs.append(r)
            ys.append(float(sub.iloc[0]["test_acc"]))
        if len(xs) > 0:
            axes[0].plot(
                xs,
                ys,
                marker="o",
                markeredgecolor="white",
                markeredgewidth=0.8,
                linewidth=2.2,
                label=label,
                color=colors[(branch, ar)],
            )
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Matched Comparison")
    axes[0].set_xticks(rank_list)
    style_axes(axes[0], add_grid=True)

    # panel 2: EoRA scaling
    eora = exp_df[exp_df["branch"] == "EoRA"].copy()
    rank_color_map = {
        4: "#F39C6B",
        8: "#D97B66",
        16: "#9C89D9",
        32: "#7dacd1",
        64: "#4BA3A6",
    }
    valid_ranks: List[int] = []
    for r in sorted(eora["rank"].dropna().unique()):
        sub = eora[eora["rank"] == r]
        if sub["alpha_over_r"].notna().sum() >= 2:
            valid_ranks.append(int(r))
    for r in valid_ranks:
        sub = eora[eora["rank"] == r].sort_values("alpha_over_r")
        axes[1].plot(
            sub["alpha_over_r"],
            sub["test_acc"],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=2.2,
            label=f"r={r}",
            color=rank_color_map.get(int(r), "#7dacd1"),
        )
    axes[1].set_xlabel("Alpha / Rank")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("EoRA Scaling")
    style_axes(axes[1], add_grid=True)

    # panel 3: gap to teacher
    if teacher_acc is not None:
        sub = exp_df.sort_values(["branch", "rank", "alpha_over_r"]).copy()
        labels = [
            f"{r['branch']}-r{int(r['rank'])}-ar{float(r['alpha_over_r']):g}"
            for _, r in sub.iterrows()
        ]
        colors_bar = [single_method_colors().get(b, "#666666") for b in sub["branch"]]
        axes[2].bar(
            np.arange(len(sub)),
            sub["teacher_minus_test"].to_numpy(),
            color=colors_bar,
            edgecolor="#333333",
            linewidth=0.8,
        )
        axes[2].set_xticks(np.arange(len(sub)))
        axes[2].set_xticklabels(labels, rotation=40, ha="right")
        axes[2].set_title("Gap to Optimized")
        axes[2].set_ylabel("Optimized - Test Accuracy")
        style_axes(axes[2], add_grid=True)
    else:
        axes[2].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Exp1-cls: Main Result Overview", y=1.10, fontsize=15, fontweight="semibold")

    save_fig(fig, out_dir / "fig_exp1_cls_main_triptych.png")


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
    plot_triptych(df, teacher_acc, out_dir)

    print("\nDone. Figures and tables saved under:")
    print(out_dir)


if __name__ == "__main__":
    main()