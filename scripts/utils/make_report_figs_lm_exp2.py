# scripts/utils/make_report_figs_lm_exp2.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "lm" / "exp2_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lm" / "report_exp2"


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

    numeric_cols = [
        "rank", "alpha", "alpha_over_r", "seed",
        "val_loss", "val_ppl", "val_kl_to_teacher", "val_mse_logits_to_teacher",
        "test_loss", "test_ppl", "test_kl_to_teacher", "test_mse_logits_to_teacher",
        "teacher_test_loss", "teacher_test_ppl",
        "test_loss_minus_teacher", "test_ppl_minus_teacher",
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


def _same_float(a, b, tol: float = 1e-9) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) < tol


# ============================================================
# colors
# ============================================================

def single_method_colors():
    return {
        "LoRA-KD": "#e45238",
        "EoRA": "#7dacd1",
        "Teacher": "#FDE395",   # <- 按你的要求改成这个
        "TeacherLine": "#D9B84F",
    }


def extra_colors():
    return {
        "teal": "#4BA3A6",
        "orange": "#F39C6B",
    }


# ============================================================
# data helpers
# ============================================================

def get_teacher_test_loss(df: pd.DataFrame) -> Optional[float]:
    if "teacher_test_loss" not in df.columns:
        return None
    vals = df["teacher_test_loss"].dropna().tolist()
    if len(vals) == 0:
        return None
    return float(vals[0])


def get_teacher_test_ppl(df: pd.DataFrame) -> Optional[float]:
    if "teacher_test_ppl" not in df.columns:
        return None
    vals = df["teacher_test_ppl"].dropna().tolist()
    if len(vals) == 0:
        return None
    return float(vals[0])


def get_exp_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["branch"].isin(["LoRA-KD", "EoRA"])].copy()


def get_apfh_rows(df: pd.DataFrame) -> pd.DataFrame:
    exp_df = get_exp_rows(df)
    if "target_modules" not in exp_df.columns:
        return exp_df.copy()
    return exp_df[exp_df["target_modules"] == "c_attn,c_proj,c_fc,lm_head"].copy()


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["branch", "target_modules", "rank", "alpha_over_r"]

    metric_cols = [
        "test_loss",
        "test_ppl",
        "test_kl_to_teacher",
        "test_mse_logits_to_teacher",
        "test_loss_minus_teacher",
        "val_loss",
        "val_ppl",
        "val_kl_to_teacher",
        "val_mse_logits_to_teacher",
    ]

    keep_cols = [c for c in metric_cols if c in df.columns]

    g = (
        df.groupby(group_cols, dropna=False)[keep_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    flat_cols = []
    for col in g.columns:
        if isinstance(col, tuple):
            a, b = col
            if b == "":
                flat_cols.append(a)
            else:
                flat_cols.append(f"{a}_{b}")
        else:
            flat_cols.append(col)
    g.columns = flat_cols
    return g


def fmt_cfg(rank: int, ar: float) -> str:
    if abs(ar - round(ar)) < 1e-9:
        ar_txt = f"{int(round(ar))}"
    else:
        ar_txt = f"{ar:g}"
    return f"r{int(rank)}\nar={ar_txt}"


def lookup_metric(
    agg_df: pd.DataFrame,
    branch: str,
    rank: int,
    ar: float,
    metric: str,
) -> Tuple[Optional[float], float, int]:
    sub = agg_df[
        (agg_df["branch"] == branch)
        & (agg_df["rank"] == rank)
        & (agg_df["alpha_over_r"].apply(lambda x: _same_float(x, ar)))
    ]
    if len(sub) == 0:
        return None, 0.0, 0

    row = sub.iloc[0]
    m = row.get(f"{metric}_mean", np.nan)
    s = row.get(f"{metric}_std", np.nan)
    c = row.get(f"{metric}_count", 0)

    mean_val = None if pd.isna(m) else float(m)
    std_val = 0.0 if pd.isna(s) else float(s)
    cnt_val = 0 if pd.isna(c) else int(c)
    return mean_val, std_val, cnt_val


def apply_scientific_y(ax) -> None:
    fmt_y = ScalarFormatter(useMathText=True)
    fmt_y.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(fmt_y)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))


def set_consistent_headroom(
    ax,
    values,
    pad_ratio: float = 0.15,
    floor_to_zero: bool = False,
) -> None:
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return

    ymin = min(vals)
    ymax = max(vals)
    span = max(ymax - ymin, 1e-8)

    lower = ymin - pad_ratio * span
    upper = ymax + pad_ratio * span

    if floor_to_zero:
        lower = 0.0

    ax.set_ylim(lower, upper)


# ============================================================
# tables
# ============================================================

def make_main_summary_table(df: pd.DataFrame, teacher_loss: Optional[float], teacher_ppl: Optional[float]) -> pd.DataFrame:
    exp_df = get_apfh_rows(df).copy()

    def best_of(branch: str):
        sub = exp_df[(exp_df["branch"] == branch) & exp_df["test_kl_to_teacher"].notna()].copy()
        if len(sub) == 0:
            return None
        return sub.sort_values("test_kl_to_teacher", ascending=True).iloc[0]

    rows = []

    if teacher_loss is not None:
        rows.append({
            "method": "Optimized model",
            "branch": "Teacher",
            "rank": None,
            "alpha": None,
            "alpha_over_r": None,
            "test_loss": teacher_loss,
            "test_ppl": teacher_ppl,
            "test_kl_to_teacher": None,
            "test_mse_logits_to_teacher": None,
            "loss_gap_to_teacher": 0.0,
            "exp_name": "optimized_model",
        })

    for branch in ["LoRA-KD", "EoRA"]:
        r = best_of(branch)
        if r is None:
            continue
        rows.append({
            "method": f"Best {branch}",
            "branch": branch,
            "rank": int(r["rank"]) if pd.notna(r["rank"]) else None,
            "alpha": float(r["alpha"]) if "alpha" in r and pd.notna(r["alpha"]) else None,
            "alpha_over_r": float(r["alpha_over_r"]) if pd.notna(r["alpha_over_r"]) else None,
            "test_loss": float(r["test_loss"]) if pd.notna(r["test_loss"]) else None,
            "test_ppl": float(r["test_ppl"]) if pd.notna(r["test_ppl"]) else None,
            "test_kl_to_teacher": float(r["test_kl_to_teacher"]) if pd.notna(r["test_kl_to_teacher"]) else None,
            "test_mse_logits_to_teacher": float(r["test_mse_logits_to_teacher"]) if pd.notna(r["test_mse_logits_to_teacher"]) else None,
            "loss_gap_to_teacher": float(r["test_loss_minus_teacher"]) if pd.notna(r["test_loss_minus_teacher"]) else None,
            "exp_name": r["exp_name"],
        })

    return pd.DataFrame(rows)


def make_full_results_table(df: pd.DataFrame) -> pd.DataFrame:
    exp_df = get_exp_rows(df).copy()
    cols = [
        "branch",
        "exp_name",
        "run_name",
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "target_modules",
        "test_loss",
        "test_ppl",
        "test_kl_to_teacher",
        "test_mse_logits_to_teacher",
        "test_loss_minus_teacher",
        "kd_T",
        "kd_lambda",
        "kd_sup_lambda",
        "kd_loss_type",
    ]
    keep = [c for c in cols if c in exp_df.columns]
    out = exp_df[keep].copy()
    out = out.sort_values(["branch", "rank", "alpha_over_r", "seed"], ascending=True)
    return out.reset_index(drop=True)


# ============================================================
# plots
# ============================================================

def plot_test_loss_main_with_teacher_bar(
    agg_df: pd.DataFrame,
    teacher_loss: Optional[float],
    out_path: Path,
) -> None:
    colors = single_method_colors()

    points = [
        (32, 1.0),
        (64, 1.0),
        (128, 1.0),
        (64, 2.0),
    ]
    labels = ["Optimized\nmodel"] + [fmt_cfg(r, ar) for r, ar in points]

    xs = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(8.9, 5.9))  # <- 画高一点

    # teacher bar on far left
    if teacher_loss is not None:
        ax.bar(
            xs[0],
            float(teacher_loss),
            width=0.42,
            color=colors["Teacher"],
            edgecolor="#333333",
            linewidth=0.8,
            label="Optimized model",
            zorder=3,
        )

    lora_vals, lora_err = [], []
    eora_vals, eora_err = [], []

    for r, ar in points:
        lv, ls, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, "test_loss")
        ev, es, _ = lookup_metric(agg_df, "EoRA", r, ar, "test_loss")
        lora_vals.append(np.nan if lv is None else lv)
        lora_err.append(0.0 if lv is None else ls)
        eora_vals.append(np.nan if ev is None else ev)
        eora_err.append(0.0 if ev is None else es)

    # bars start from x=1
    bar_x = xs[1:]

    mask_l = ~np.isnan(np.array(lora_vals))
    ax.bar(
        bar_x[mask_l] - width / 2,
        np.array(lora_vals)[mask_l],
        width=width,
        yerr=np.array(lora_err)[mask_l],
        color=colors["LoRA-KD"],
        edgecolor="#333333",
        linewidth=0.8,
        label="LoRA-KD",
        zorder=3,
    )

    mask_e = ~np.isnan(np.array(eora_vals))
    ax.bar(
        bar_x[mask_e] + width / 2,
        np.array(eora_vals)[mask_e],
        width=width,
        yerr=np.array(eora_err)[mask_e],
        color=colors["EoRA"],
        edgecolor="#333333",
        linewidth=0.8,
        label="EoRA",
        zorder=3,
    )

    # teacher reference line across non-teacher bars
    if teacher_loss is not None:
        ax.hlines(
            y=float(teacher_loss),
            xmin=0.7,
            xmax=len(labels) - 0.3,
            colors=colors["TeacherLine"],
            linestyles="--",
            linewidth=1.6,
            zorder=4,
            label="Optimized reference",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Loss")
    ax.set_title("Exp2-lm: Shared-target output matching (test loss)")

    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, loc="upper right")  # <- 放右上角

    # y-lim 稍微高一点，避免图例压柱
    ymax = []
    if teacher_loss is not None:
        ymax.append(float(teacher_loss))
    ymax.extend([v for v in lora_vals if not np.isnan(v)])
    ymax.extend([v for v in eora_vals if not np.isnan(v)])
    if len(ymax) > 0:
        ymin = min(ymax) - 0.03
        ymax2 = max(ymax) + 0.08
        ax.set_ylim(ymin, ymax2)

    save_fig(fig, out_path)


def plot_grouped_bar_main(
    agg_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    teacher_line: Optional[float] = None,
    use_scientific: bool = False,
    is_gap_plot: bool = False,
) -> None:
    colors = single_method_colors()

    points = [
        (32, 1.0),
        (64, 1.0),
        (128, 1.0),
        (64, 2.0),
    ]
    labels = [fmt_cfg(r, ar) for r, ar in points]

    xs = np.arange(len(points))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    lora_vals, eora_vals = [], []
    lora_err, eora_err = [], []

    for r, ar in points:
        lv, ls, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, metric)
        ev, es, _ = lookup_metric(agg_df, "EoRA", r, ar, metric)

        lora_vals.append(np.nan if lv is None else lv)
        eora_vals.append(np.nan if ev is None else ev)
        lora_err.append(0.0 if lv is None else ls)
        eora_err.append(0.0 if ev is None else es)

    mask_l = ~np.isnan(np.array(lora_vals))
    ax.bar(
        xs[mask_l] - width / 2,
        np.array(lora_vals)[mask_l],
        width=width,
        yerr=np.array(lora_err)[mask_l],
        color=colors["LoRA-KD"],
        edgecolor="#333333",
        linewidth=0.8,
        label="LoRA-KD",
        zorder=3,
    )

    mask_e = ~np.isnan(np.array(eora_vals))
    ax.bar(
        xs[mask_e] + width / 2,
        np.array(eora_vals)[mask_e],
        width=width,
        yerr=np.array(eora_err)[mask_e],
        color=colors["EoRA"],
        edgecolor="#333333",
        linewidth=0.8,
        label="EoRA",
        zorder=3,
    )

    if teacher_line is not None:
        ax.axhline(
            y=float(teacher_line),
            linestyle="--",
            linewidth=1.6,
            color=colors["TeacherLine"],
            alpha=0.98,
            zorder=5,
            label="Optimized reference",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if use_scientific:
        apply_scientific_y(ax)

    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, loc="upper right")

    # 专门修复 gap 图里 y=0 线看不见的问题
    if is_gap_plot:
        vals = []
        vals.extend([v for v in lora_vals if not np.isnan(v)])
        vals.extend([v for v in eora_vals if not np.isnan(v)])
        if len(vals) > 0:
            ymin = min(vals)
            ymax = max(vals)
            span = max(ymax - ymin, 1e-6)
            lower = min(-0.01, ymin - 0.15 * span)   # <- 留一点负空间，让 0 线浮起来
            upper = ymax + 0.15 * span
            ax.set_ylim(lower, upper)

    save_fig(fig, out_path)


def plot_lora_kd_sensitivity(agg_df: pd.DataFrame, out_path: Path) -> None:
    extra = extra_colors()

    points = [
        (32, 1.0),
        (64, 1.0),
        (64, 2.0),
        (128, 1.0),
    ]
    labels = [fmt_cfg(r, ar).replace("\n", " / ") for r, ar in points]
    xs = np.arange(len(points))

    test_loss = []
    test_kl = []

    for r, ar in points:
        lv, _, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, "test_loss")
        kv, _, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, "test_kl_to_teacher")
        test_loss.append(np.nan if lv is None else lv)
        test_kl.append(np.nan if kv is None else kv)

    fig, ax1 = plt.subplots(figsize=(8.2, 5.0))

    ax1.plot(
        xs,
        test_loss,
        marker="o",
        markersize=6,
        linewidth=2.4,
        color=extra["orange"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Test loss",
    )
    ax1.set_xlabel("LoRA-KD configuration")
    ax1.set_ylabel("Test loss")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    style_axes(ax1, add_grid=True)

    ax2 = ax1.twinx()
    ax2.plot(
        xs,
        test_kl,
        marker="s",
        markersize=5.5,
        linewidth=2.2,
        color=extra["teal"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Test KL to teacher",
    )
    ax2.set_ylabel("Test KL to teacher")
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_color("#444444")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    ax1.set_title("Exp2-lm: LoRA-KD sensitivity under tm-apfh")

    save_fig(fig, out_path)


def plot_main_triptych(
    agg_df: pd.DataFrame,
    teacher_loss: Optional[float],
    out_path: Path,
) -> None:
    colors = single_method_colors()

    points = [
        (32, 1.0),
        (64, 1.0),
        (128, 1.0),
        (64, 2.0),
    ]
    labels_main = ["Optimized\nmodel"] + [fmt_cfg(r, ar) for r, ar in points]
    labels_other = [fmt_cfg(r, ar) for r, ar in points]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0))
    width = 0.28
    legend_loc = "upper right"

    # ---------- panel 1: test loss ----------
    ax = axes[0]
    xs = np.arange(len(labels_main))

    if teacher_loss is not None:
        ax.bar(
            xs[0],
            float(teacher_loss),
            width=0.42,
            color=colors["Teacher"],
            edgecolor="#333333",
            linewidth=0.8,
            label="Optimized model",
            zorder=3,
        )

    lora_vals, lora_err = [], []
    eora_vals, eora_err = [], []

    for r, ar in points:
        lv, ls, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, "test_loss")
        ev, es, _ = lookup_metric(agg_df, "EoRA", r, ar, "test_loss")
        lora_vals.append(np.nan if lv is None else lv)
        lora_err.append(0.0 if lv is None else ls)
        eora_vals.append(np.nan if ev is None else ev)
        eora_err.append(0.0 if ev is None else es)

    bar_x = xs[1:]

    mask_l = ~np.isnan(np.array(lora_vals))
    ax.bar(
        bar_x[mask_l] - width / 2,
        np.array(lora_vals)[mask_l],
        width=width,
        yerr=np.array(lora_err)[mask_l],
        color=colors["LoRA-KD"],
        edgecolor="#333333",
        linewidth=0.8,
        label="LoRA-KD",
        zorder=3,
    )

    mask_e = ~np.isnan(np.array(eora_vals))
    ax.bar(
        bar_x[mask_e] + width / 2,
        np.array(eora_vals)[mask_e],
        width=width,
        yerr=np.array(eora_err)[mask_e],
        color=colors["EoRA"],
        edgecolor="#333333",
        linewidth=0.8,
        label="EoRA",
        zorder=3,
    )

    if teacher_loss is not None:
        ax.hlines(
            y=float(teacher_loss),
            xmin=0.7,
            xmax=len(labels_main) - 0.3,
            colors=colors["TeacherLine"],
            linestyles="--",
            linewidth=1.6,
            zorder=4,
            label="Optimized reference",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels_main)
    ax.set_ylabel("Test Loss")
    ax.set_title("(a) Test loss")
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, loc=legend_loc)

    vals_a = []
    if teacher_loss is not None:
        vals_a.append(float(teacher_loss))
    vals_a.extend([v for v in lora_vals if not np.isnan(v)])
    vals_a.extend([v for v in eora_vals if not np.isnan(v)])
    set_consistent_headroom(ax, vals_a, pad_ratio=0.15, floor_to_zero=False)

    # ---------- panel 2: test KL ----------
    ax = axes[1]
    xs2 = np.arange(len(labels_other))

    lora_vals, lora_err = [], []
    eora_vals, eora_err = [], []
    for r, ar in points:
        lv, ls, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, "test_kl_to_teacher")
        ev, es, _ = lookup_metric(agg_df, "EoRA", r, ar, "test_kl_to_teacher")
        lora_vals.append(np.nan if lv is None else lv)
        lora_err.append(0.0 if lv is None else ls)
        eora_vals.append(np.nan if ev is None else ev)
        eora_err.append(0.0 if ev is None else es)

    mask_l = ~np.isnan(np.array(lora_vals))
    ax.bar(
        xs2[mask_l] - width / 2,
        np.array(lora_vals)[mask_l],
        width=width,
        yerr=np.array(lora_err)[mask_l],
        color=colors["LoRA-KD"],
        edgecolor="#333333",
        linewidth=0.8,
        label="LoRA-KD",
        zorder=3,
    )
    mask_e = ~np.isnan(np.array(eora_vals))
    ax.bar(
        xs2[mask_e] + width / 2,
        np.array(eora_vals)[mask_e],
        width=width,
        yerr=np.array(eora_err)[mask_e],
        color=colors["EoRA"],
        edgecolor="#333333",
        linewidth=0.8,
        label="EoRA",
        zorder=3,
    )

    ax.set_xticks(xs2)
    ax.set_xticklabels(labels_other)
    ax.set_ylabel("Test KL to Teacher")
    ax.set_title("(b) Teacher matching quality")
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, loc=legend_loc)

    vals_b = []
    vals_b.extend([v for v in lora_vals if not np.isnan(v)])
    vals_b.extend([v for v in eora_vals if not np.isnan(v)])
    set_consistent_headroom(ax, vals_b, pad_ratio=0.15, floor_to_zero=False)

    # ---------- panel 3: loss gap ----------
    ax = axes[2]
    xs3 = np.arange(len(labels_other))

    lora_vals, lora_err = [], []
    eora_vals, eora_err = [], []
    for r, ar in points:
        lv, ls, _ = lookup_metric(agg_df, "LoRA-KD", r, ar, "test_loss_minus_teacher")
        ev, es, _ = lookup_metric(agg_df, "EoRA", r, ar, "test_loss_minus_teacher")
        lora_vals.append(np.nan if lv is None else lv)
        lora_err.append(0.0 if lv is None else ls)
        eora_vals.append(np.nan if ev is None else ev)
        eora_err.append(0.0 if ev is None else es)

    mask_l = ~np.isnan(np.array(lora_vals))
    ax.bar(
        xs3[mask_l] - width / 2,
        np.array(lora_vals)[mask_l],
        width=width,
        yerr=np.array(lora_err)[mask_l],
        color=colors["LoRA-KD"],
        edgecolor="#333333",
        linewidth=0.8,
        label="LoRA-KD",
        zorder=3,
    )
    mask_e = ~np.isnan(np.array(eora_vals))
    ax.bar(
        xs3[mask_e] + width / 2,
        np.array(eora_vals)[mask_e],
        width=width,
        yerr=np.array(eora_err)[mask_e],
        color=colors["EoRA"],
        edgecolor="#333333",
        linewidth=0.8,
        label="EoRA",
        zorder=3,
    )

    ax.set_xticks(xs3)
    ax.set_xticklabels(labels_other)
    ax.set_ylabel("Test Loss Gap to Teacher")
    ax.set_title("(c) Distance to optimized teacher")
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False, loc=legend_loc)

    vals_c = []
    vals_c.extend([v for v in lora_vals if not np.isnan(v)])
    vals_c.extend([v for v in eora_vals if not np.isnan(v)])
    set_consistent_headroom(ax, vals_c, pad_ratio=0.15, floor_to_zero=True)

    save_fig(fig, out_path)


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
    teacher_loss = get_teacher_test_loss(df)
    teacher_ppl = get_teacher_test_ppl(df)

    apfh_df = get_apfh_rows(df)
    agg_df = aggregate_runs(apfh_df)

    main_table = make_main_summary_table(apfh_df, teacher_loss, teacher_ppl)
    full_table = make_full_results_table(df)

    save_df(main_table, out_dir / "table_exp2_lm_main_summary.csv")
    save_df(full_table, out_dir / "table_exp2_lm_full_results.csv")

    plot_main_triptych(
        agg_df=agg_df,
        teacher_loss=teacher_loss,
        out_path=out_dir / "fig_exp2_lm_main_triptych.png",
    )

    # 4) MSE
    plot_grouped_bar_main(
        agg_df=agg_df,
        metric="test_mse_logits_to_teacher",
        ylabel="Test Logit MSE to Teacher",
        title="Exp2-lm: Output-space matching quality (logit MSE)",
        out_path=out_dir / "fig_exp2_lm_test_mse_main.png",
        teacher_line=None,
        use_scientific=True,
        is_gap_plot=False,
    )

    # 5) LoRA-KD sensitivity
    plot_lora_kd_sensitivity(
        agg_df=agg_df,
        out_path=out_dir / "fig_exp2_lm_lora_kd_sensitivity.png",
    )

    print("\nDone. Figures and tables saved under:")
    print(out_dir)


if __name__ == "__main__":
    main()