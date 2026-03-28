from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "lm" / "exp1_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lm" / "report_exp1"


# ============================================================
# global style (aligned with cls exp1 style)
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


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[saved] {path}")


def _format_df_for_tex(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "--" if pd.isna(x) else f"{float(x):.4f}")
        elif pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].map(lambda x: "--" if pd.isna(x) else str(int(x)))
        else:
            out[col] = out[col].fillna("--").astype(str)

    return out


def save_tex_tabular(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Export tabular-only LaTeX for use with \\input{...} inside a table environment.
    """
    tex_df = _format_df_for_tex(df)
    latex = tex_df.to_latex(index=index, escape=True)
    path.write_text(latex, encoding="utf-8")
    print(f"[saved] {path}")


def save_tex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    index: bool = False,
) -> None:
    """
    Export a standalone LaTeX table environment.
    """
    tex_df = _format_df_for_tex(df)
    latex = tex_df.to_latex(
        index=index,
        escape=True,
        caption=caption,
        label=label,
    )
    path.write_text(latex, encoding="utf-8")
    print(f"[saved] {path}")


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


def style_axes(ax, add_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    if add_grid:
        ax.grid(True, axis="y", alpha=0.55)
    else:
        ax.grid(False)


def set_tight_ylim(ax, values, pad_ratio: float = 0.10, min_pad: float = 0.015):
    vals = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if len(vals) == 0:
        return

    vmin = float(vals.min())
    vmax = float(vals.max())
    span = vmax - vmin
    pad = max(min_pad, span * pad_ratio)

    ax.set_ylim(vmin - pad, vmax + pad)


def single_method_colors():
    return {
        "Teacher": "#fde395",
        "LoRA": "#e45238",
        "EoRA": "#7dacd1",
    }


def target_labels():
    return {
        "tm-ap": "tm-ap",
        "tm-apf": "tm-apf",
        "tm-apfh": "tm-apfh",
    }


def rank_palette_6():
    # fixed order requested by user
    return ["#feeaae", "#e2f2f0", "#f79c67", "#98bcd8", "#d73322", "#4573b4"]


def build_rank_style_map(ranks):
    ranks = sorted(int(r) for r in ranks if pd.notna(r))
    palette = rank_palette_6()

    if len(ranks) <= 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.035, 0.035, len(ranks))

    style_map = {}
    for i, r in enumerate(ranks):
        style_map[r] = {
            "color": palette[i % len(palette)],
            "offset": float(offsets[i]),
            "zorder": 10 + i,
        }
    return style_map


def _norm_col(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace("/", "_").replace("-", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _pretty_ar(x: float) -> str:
    if pd.isna(x):
        return "NA"
    if abs(float(x) - round(float(x))) < 1e-10:
        return f"{int(round(float(x)))}"
    return f"{float(x):.2f}".rstrip("0").rstrip(".")


def _infer_target_modules(raw: str) -> str:
    s = str(raw)
    parts = sorted({p.strip() for p in s.split(",") if p.strip()})
    if parts == ["c_attn", "c_proj"]:
        return "tm-ap"
    if parts == ["c_attn", "c_fc", "c_proj"]:
        return "tm-apf"
    if parts == ["c_attn", "c_fc", "c_proj", "lm_head"]:
        return "tm-apfh"

    sl = s.lower()
    if "tm-apfh" in sl:
        return "tm-apfh"
    if "tm-apf" in sl:
        return "tm-apf"
    if "tm-ap" in sl:
        return "tm-ap"
    return "unknown"


# ============================================================
# load / normalize
# ============================================================

def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [_norm_col(c) for c in df.columns]

    num_cols = [
        "seed",
        "rank",
        "alpha",
        "ar",
        "alpha_over_r",
        "val_loss",
        "val_ppl",
        "test_loss",
        "test_ppl",
        "loss_gap_to_teacher",
        "teacher_test_loss",
        "teacher_test_ppl",
        "trainable_params",
        "num_trainable_params",
        "trainable_param_count",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "branch" in df.columns:
        raw_branch = df["branch"].astype(str).str.strip().str.lower()
    elif "method" in df.columns:
        raw_branch = df["method"].astype(str).str.strip().str.lower()
    elif "type" in df.columns:
        raw_branch = df["type"].astype(str).str.strip().str.lower()
    else:
        raw_branch = pd.Series([""] * len(df), index=df.index)

    df["branch_std"] = raw_branch.replace({
        "lora": "LoRA",
        "eora": "EoRA",
        "teacher": "Teacher",
        "optimized": "Teacher",
    })

    if "target_modules" in df.columns:
        df["target_group"] = df["target_modules"].apply(_infer_target_modules)
    else:
        src = ""
        if "config_name" in df.columns:
            src = src + " " + df["config_name"].astype(str)
        if "path" in df.columns:
            src = src + " " + df["path"].astype(str)
        df["target_group"] = src.apply(_infer_target_modules)

    if "alpha_over_r" not in df.columns:
        df["alpha_over_r"] = np.nan

    if "ar" in df.columns:
        df["alpha_over_r"] = df["alpha_over_r"].fillna(df["ar"])

    if {"alpha", "rank"}.issubset(df.columns):
        df["alpha_over_r"] = df["alpha_over_r"].fillna(df["alpha"] / df["rank"])

    teacher_loss = get_teacher_loss(df)
    if "loss_gap_to_teacher" not in df.columns or df["loss_gap_to_teacher"].isna().all():
        if "test_loss" in df.columns and teacher_loss is not None:
            df["loss_gap_to_teacher"] = df["test_loss"] - teacher_loss

    if "trainable_params" not in df.columns:
        for alt in ["num_trainable_params", "trainable_param_count"]:
            if alt in df.columns:
                df["trainable_params"] = df[alt]
                break
        if "trainable_params" not in df.columns:
            df["trainable_params"] = np.nan

    return df


def get_teacher_loss(df: pd.DataFrame) -> Optional[float]:
    if "branch_std" in df.columns:
        tdf = df[df["branch_std"] == "Teacher"]
        if len(tdf) > 0 and "test_loss" in tdf.columns and tdf["test_loss"].notna().any():
            return float(tdf["test_loss"].dropna().iloc[0])

    if "teacher_test_loss" in df.columns and df["teacher_test_loss"].notna().any():
        return float(df["teacher_test_loss"].dropna().iloc[0])

    return None


def get_teacher_ppl(df: pd.DataFrame) -> Optional[float]:
    if "branch_std" in df.columns:
        tdf = df[df["branch_std"] == "Teacher"]
        if len(tdf) > 0 and "test_ppl" in tdf.columns and tdf["test_ppl"].notna().any():
            return float(tdf["test_ppl"].dropna().iloc[0])

    if "teacher_test_ppl" in df.columns and df["teacher_test_ppl"].notna().any():
        return float(df["teacher_test_ppl"].dropna().iloc[0])

    return None


def get_exp_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["branch_std"].isin(["LoRA", "EoRA"])].copy()


# ============================================================
# tables
# ============================================================

def make_main_summary_table(df: pd.DataFrame, teacher_loss: Optional[float], teacher_ppl: Optional[float]) -> pd.DataFrame:
    exp_df = get_exp_rows(df).copy()
    rows = []

    if teacher_loss is not None:
        rows.append({
            "method": "Optimized teacher",
            "branch": "Teacher",
            "target_group": "-",
            "rank": None,
            "alpha": None,
            "alpha_over_r": None,
            "test_loss": teacher_loss,
            "test_ppl": teacher_ppl,
            "loss_gap_to_teacher": 0.0,
            "config_name": "optimized_teacher",
        })

    for target in ["tm-ap", "tm-apf", "tm-apfh"]:
        for branch in ["LoRA", "EoRA"]:
            sub = exp_df[
                (exp_df["branch_std"] == branch) &
                (exp_df["target_group"] == target) &
                exp_df["test_loss"].notna()
            ].copy()
            if len(sub) == 0:
                continue
            best = sub.sort_values("test_loss", ascending=True).iloc[0]
            rows.append({
                "method": f"Best {branch} ({target})",
                "branch": branch,
                "target_group": target,
                "rank": best.get("rank", np.nan),
                "alpha": best.get("alpha", np.nan),
                "alpha_over_r": best.get("alpha_over_r", np.nan),
                "test_loss": best.get("test_loss", np.nan),
                "test_ppl": best.get("test_ppl", np.nan),
                "loss_gap_to_teacher": best.get("loss_gap_to_teacher", np.nan),
                "config_name": best.get("config_name", ""),
            })

    return pd.DataFrame(rows)


def make_full_results_table(df: pd.DataFrame) -> pd.DataFrame:
    exp_df = get_exp_rows(df).copy()
    cols = [
        "branch_std",
        "target_group",
        "config_name",
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "trainable_params",
        "val_loss",
        "val_ppl",
        "test_loss",
        "test_ppl",
        "loss_gap_to_teacher",
        "path",
    ]
    cols = [c for c in cols if c in exp_df.columns]
    exp_df = exp_df[cols].sort_values(
        by=["target_group", "branch_std", "rank", "alpha_over_r", "seed"],
        ascending=[True, True, True, True, True],
    )
    return exp_df


def make_grouped_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    exp_df = get_exp_rows(df).copy()

    grp_cols = ["branch_std", "target_group", "rank", "alpha", "alpha_over_r"]
    grp_cols = [c for c in grp_cols if c in exp_df.columns]

    grouped = (
        exp_df.groupby(grp_cols, dropna=False, as_index=False)
        .agg(
            n_runs=("test_loss", "size"),
            trainable_params=("trainable_params", "first"),
            val_loss_mean=("val_loss", "mean"),
            test_loss_mean=("test_loss", "mean"),
            test_loss_std=("test_loss", "std"),
            val_ppl_mean=("val_ppl", "mean"),
            test_ppl_mean=("test_ppl", "mean"),
            loss_gap_to_teacher_mean=("loss_gap_to_teacher", "mean"),
        )
    )

    if "test_loss_std" in grouped.columns:
        grouped["test_loss_std"] = grouped["test_loss_std"].fillna(0.0)

    grouped = grouped.sort_values(
        by=["target_group", "branch_std", "rank", "alpha_over_r"],
        ascending=[True, True, True, True],
    )

    return grouped


# ============================================================
# plots for main text
# ============================================================

def plot_best_by_target(df: pd.DataFrame, teacher_loss: Optional[float], out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    colors = single_method_colors()
    tlabels = target_labels()

    teacher_bar_color = "#fde395"

    rows = []

    # 先加 optimized model 柱子，放在最左边
    if teacher_loss is not None:
        rows.append({
            "branch": "Optimized model",
            "target_group": "teacher",
            "label": "Optimized\nmodel",
            "value": float(teacher_loss),
            "color": teacher_bar_color,
        })

    # 再加各 target group 的 best LoRA / EoRA
    for target in ["tm-ap", "tm-apf", "tm-apfh"]:
        for branch in ["LoRA", "EoRA"]:
            sub = exp_df[
                (exp_df["branch_std"] == branch) &
                (exp_df["target_group"] == target) &
                exp_df["test_loss"].notna()
            ].copy()
            if len(sub) == 0:
                continue
            best = sub.sort_values("test_loss", ascending=True).iloc[0]
            rows.append({
                "branch": branch,
                "target_group": target,
                "label": f"{branch}\n{tlabels[target]}",
                "value": float(best["test_loss"]),
                "color": colors.get(branch, "#7dacd1"),
            })

    if len(rows) == 0:
        return

    xs = np.arange(len(rows))
    vals = [r["value"] for r in rows]
    labels = [r["label"] for r in rows]
    bar_colors = [r["color"] for r in rows]

    fig, ax = plt.subplots(figsize=(9.4, 6.4))

    bars = ax.bar(
        xs,
        vals,
        width=0.72,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Loss")
    ax.set_title("Exp1-lm: Best Results by Target-Module Group", pad=12)

    # 顶部留白，避免数值和图例重叠
    top_ref = max(vals)
    bottom = min(vals)
    yrange = max(top_ref - bottom, 0.08)
    ax.set_ylim(bottom - 0.03, top_ref + 0.42 * yrange)

    # 图例改名
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=teacher_bar_color, edgecolor="#333333", label="Optimized model"),
        Patch(facecolor=colors["LoRA"], edgecolor="#333333", label="LoRA"),
        Patch(facecolor=colors["EoRA"], edgecolor="#333333", label="EoRA"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        borderaxespad=0.0,
    )

    # 标注具体数值，4位小数更清楚
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.012,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )

    style_axes(ax, add_grid=True)
    save_fig(fig, out_dir / "fig_exp1_lm_best_by_target.png", use_tight_layout=False)

def plot_tm_ap_matched(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    sub = exp_df[exp_df["target_group"] == "tm-ap"].copy()
    colors = single_method_colors()

    pairs = [
        (8, 1.0),
        (8, 2.0),
        (16, 1.0),
        (32, 1.0),
        (32, 2.0),
    ]

    xs = np.arange(len(pairs))
    lora_vals = []
    eora_vals = []
    xlabels = []

    for rank, ar in pairs:
        xlabels.append(f"r{rank}\nar={_pretty_ar(ar)}")

        ls = sub[
            (sub["branch_std"] == "LoRA") &
            (sub["rank"] == rank) &
            (np.isclose(sub["alpha_over_r"], ar, equal_nan=False))
        ]
        es = sub[
            (sub["branch_std"] == "EoRA") &
            (sub["rank"] == rank) &
            (np.isclose(sub["alpha_over_r"], ar, equal_nan=False))
        ]

        lora_vals.append(float(ls["test_loss"].mean()) if len(ls) else np.nan)
        eora_vals.append(float(es["test_loss"].mean()) if len(es) else np.nan)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot(
        xs,
        lora_vals,
        marker="o",
        linewidth=1.9,
        markersize=4.6,
        label="LoRA",
        color=colors["LoRA"],
    )
    ax.plot(
        xs,
        eora_vals,
        marker="o",
        linewidth=1.9,
        markersize=4.6,
        label="EoRA",
        color=colors["EoRA"],
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Matched configuration")
    ax.set_ylabel("Test Loss")
    ax.set_title("Exp1-lm: Matched Comparison under tm-ap")
    set_tight_ylim(ax, lora_vals + eora_vals, pad_ratio=0.12, min_pad=0.02)
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False)

    save_fig(fig, out_dir / "fig_exp1_lm_tm_ap_matched.png")


def plot_tm_apfh_matched(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    sub = exp_df[exp_df["target_group"] == "tm-apfh"].copy()
    colors = single_method_colors()

    pairs = [
        (32, 1.0),
        (64, 1.0),
        (128, 1.0),
    ]

    xs = np.arange(len(pairs))
    lora_vals = []
    eora_vals = []
    xlabels = []

    for rank, ar in pairs:
        xlabels.append(f"r{rank}\nar={_pretty_ar(ar)}")

        ls = sub[
            (sub["branch_std"] == "LoRA") &
            (sub["rank"] == rank) &
            (np.isclose(sub["alpha_over_r"], ar, equal_nan=False))
        ].copy()
        es = sub[
            (sub["branch_std"] == "EoRA") &
            (sub["rank"] == rank) &
            (np.isclose(sub["alpha_over_r"], ar, equal_nan=False))
        ].copy()

        lora_vals.append(float(ls["test_loss"].mean()) if len(ls) else np.nan)
        eora_vals.append(float(es["test_loss"].mean()) if len(es) else np.nan)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.plot(
        xs,
        lora_vals,
        marker="o",
        linewidth=1.9,
        markersize=4.6,
        label="LoRA",
        color=colors["LoRA"],
    )
    ax.plot(
        xs,
        eora_vals,
        marker="o",
        linewidth=1.9,
        markersize=4.6,
        label="EoRA",
        color=colors["EoRA"],
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Matched configuration")
    ax.set_ylabel("Test Loss")
    ax.set_title("Exp1-lm: Matched Comparison under tm-apfh (ar=1)")
    set_tight_ylim(ax, lora_vals + eora_vals, pad_ratio=0.12, min_pad=0.02)
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False)

    save_fig(fig, out_dir / "fig_exp1_lm_tm_apfh_matched.png")


def plot_eora_scaling_aligned_r64(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    eora = exp_df[
        (exp_df["branch_std"] == "EoRA") &
        (exp_df["rank"] == 64)
    ].copy()

    if len(eora) == 0:
        return

    palette = {
        "tm-ap": "#d73322",
        "tm-apf": "#98bcd8",
        "tm-apfh": "#4573b4",
    }

    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    for target in ["tm-ap", "tm-apf", "tm-apfh"]:
        sub = eora[eora["target_group"] == target].copy()
        if len(sub) == 0:
            continue

        tmp = sub.groupby("alpha_over_r", as_index=False)["test_loss"].mean()
        tmp = tmp.sort_values("alpha_over_r")

        ax.plot(
            tmp["alpha_over_r"].to_numpy(),
            tmp["test_loss"].to_numpy(),
            marker="o",
            linestyle="-",
            linewidth=1.9,
            markersize=4.6,
            label=target,
            color=palette[target],
        )

    xticks = sorted(eora["alpha_over_r"].dropna().unique())
    ax.set_xticks(xticks)
    ax.set_xticklabels([_pretty_ar(x) for x in xticks])

    ax.set_xlabel("α/r")
    ax.set_ylabel("Test Loss")
    ax.set_title("Exp1-lm: EoRA Scaling at r=64 across Target-Module Groups")
    set_tight_ylim(ax, eora["test_loss"].dropna().tolist(), pad_ratio=0.12, min_pad=0.02)
    style_axes(ax, add_grid=True)
    ax.legend(frameon=False)

    save_fig(fig, out_dir / "fig_exp1_lm_eora_scaling.png")


def plot_gap_to_teacher(df: pd.DataFrame, teacher_loss: Optional[float], out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    if teacher_loss is None:
        return

    if "loss_gap_to_teacher" not in exp_df.columns or exp_df["loss_gap_to_teacher"].isna().all():
        exp_df["loss_gap_to_teacher"] = exp_df["test_loss"] - teacher_loss

    rows = []
    for target in ["tm-ap", "tm-apf", "tm-apfh"]:
        for branch in ["LoRA", "EoRA"]:
            sub = exp_df[
                (exp_df["branch_std"] == branch) &
                (exp_df["target_group"] == target)
            ].copy()
            if len(sub) == 0:
                continue
            best = sub.sort_values("test_loss", ascending=True).iloc[0]
            rows.append({
                "label": f"{branch}\n{target}",
                "value": float(best["loss_gap_to_teacher"]),
                "branch": branch,
            })

    colors = single_method_colors()
    xs = np.arange(len(rows))
    vals = [r["value"] for r in rows]
    labels = [r["label"] for r in rows]
    branches = [r["branch"] for r in rows]

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bars = ax.bar(
        xs,
        vals,
        width=0.72,
        color=[colors.get(b, "#7dacd1") for b in branches],
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Loss Gap to Optimized Teacher")
    ax.set_title("Exp1-lm: Gap to Optimized Teacher")

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(v) + 0.002,
            f"{float(v):.3f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
        )

    style_axes(ax, add_grid=True)
    save_fig(fig, out_dir / "fig_exp1_lm_gap_to_teacher.png")


def plot_multiseed_confirmation(df: pd.DataFrame, out_dir: Path):
    exp_df = get_exp_rows(df).copy()
    colors = single_method_colors()

    rows = []

    candidates = [
        ("LoRA", "tm-apfh", 64, 2.0, "LoRA\nr64, ar=2"),
        ("LoRA", "tm-apfh", 128, 1.0, "LoRA\nr128, ar=1"),
        ("EoRA", "tm-apfh", 128, 1.0, "EoRA\nr128, ar=1"),
    ]

    for branch, target, rank, ar, label in candidates:
        sub = exp_df[
            (exp_df["branch_std"] == branch) &
            (exp_df["target_group"] == target) &
            (exp_df["rank"] == rank) &
            (np.isclose(exp_df["alpha_over_r"], ar, equal_nan=False))
        ].copy()

        if len(sub) == 0:
            continue

        rows.append({
            "label": label,
            "branch": branch,
            "mean": float(sub["test_loss"].mean()),
            "std": float(sub["test_loss"].std(ddof=1)) if len(sub) > 1 else 0.0,
        })

    if len(rows) == 0:
        return

    xs = np.arange(len(rows))
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    labels = [r["label"] for r in rows]
    bar_colors = [colors.get(r["branch"], "#7dacd1") for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bars = ax.bar(
        xs,
        means,
        yerr=stds,
        capsize=4,
        width=0.72,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.8,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test Loss")
    ax.set_title("Exp1-lm: Multi-seed Confirmation under tm-apfh")

    for b, v in zip(bars, means):
        ax.text(
            b.get_x() + b.get_width() / 2,
            float(v) + 0.01,
            f"{float(v):.4f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
        )

    ylim_vals = []
    for m, s in zip(means, stds):
        ylim_vals.extend([m - s, m + s])

    set_tight_ylim(ax, ylim_vals, pad_ratio=0.18, min_pad=0.02)
    style_axes(ax, add_grid=True)
    save_fig(fig, out_dir / "fig_exp1_lm_multiseed_confirm.png")


# ============================================================
# appendix-style plots
# ============================================================

def plot_completed_runs_appendix(df, branch: str, out_path: Path, title: str):
    exp_df = get_exp_rows(df).copy()
    sub_all = exp_df[exp_df["branch_std"] == branch].copy()
    tlabels = target_labels()

    all_ranks = sorted([int(r) for r in sub_all["rank"].dropna().unique()])
    style_map = build_rank_style_map(all_ranks)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.6), sharey=True)

    for ax, target in zip(axes, ["tm-ap", "tm-apf", "tm-apfh"]):
        sub = sub_all[sub_all["target_group"] == target].copy()
        if len(sub) == 0:
            ax.set_title(tlabels[target])
            style_axes(ax, add_grid=True)
            continue

        for r in sorted(sub["rank"].dropna().unique()):
            tmp = sub[sub["rank"] == r].copy()
            tmp = tmp.groupby("alpha_over_r", as_index=False)["test_loss"].mean()
            tmp = tmp.sort_values("alpha_over_r")

            st = style_map[int(r)]
            x = tmp["alpha_over_r"].to_numpy(dtype=float) + st["offset"]
            y = tmp["test_loss"].to_numpy(dtype=float)

            if len(tmp) >= 2:
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle="-",
                    linewidth=1.9,
                    markersize=4.6,
                    color=st["color"],
                    label=f"r={int(r)}",
                    zorder=st["zorder"],
                )
            else:
                ax.scatter(
                    x,
                    y,
                    s=28,
                    marker="o",
                    color=st["color"],
                    label=f"r={int(r)}",
                    zorder=st["zorder"],
                )

        xticks = sorted(sub["alpha_over_r"].dropna().unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels([_pretty_ar(x) for x in xticks])
        ax.set_xlabel("α/r")
        ax.set_title(tlabels[target])
        style_axes(ax, add_grid=True)

    # IMPORTANT: shared y-axis -> set once globally after all panels are drawn
    set_tight_ylim(axes[0], sub_all["test_loss"].dropna().tolist(), pad_ratio=0.12, min_pad=0.02)

    axes[0].set_ylabel("Test Loss")
    all_handles = []
    all_labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    if all_handles:
        uniq = {}
        for h, l in zip(all_handles, all_labels):
            if l not in uniq:
                uniq[l] = h
        axes[0].legend(uniq.values(), uniq.keys(), frameon=False, ncol=1)

    fig.suptitle(title, y=1.03)
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

    df = load_summary(summary_csv)
    teacher_loss = get_teacher_loss(df)
    teacher_ppl = get_teacher_ppl(df)

    main_table = make_main_summary_table(df, teacher_loss, teacher_ppl)
    full_table = make_full_results_table(df)
    grouped_table = make_grouped_summary_table(df)

    # CSV outputs
    save_df(main_table, out_dir / "table_exp1_lm_main_summary.csv")
    save_df(full_table, out_dir / "table_exp1_lm_full_results.csv")
    save_df(grouped_table, out_dir / "table_exp1_lm_grouped_summary.csv")

    # TEX outputs
    # 1) standalone table for direct use if needed
    save_tex_table(
        main_table,
        out_dir / "tab_exp1_lm_main_summary.tex",
        caption="Main language-modelling results for Experiment~1.",
        label="tab:exp1_lm_main_summary",
    )

    # 2) tabular-only files for appendix \input{...}
    save_tex_tabular(
        full_table,
        out_dir / "tab_exp1_lm_full_results.tex",
    )
    save_tex_tabular(
        grouped_table,
        out_dir / "tab_exp1_lm_grouped_summary.tex",
    )

    # main text
    plot_best_by_target(df, teacher_loss, out_dir)
    plot_tm_ap_matched(df, out_dir)
    plot_tm_apfh_matched(df, out_dir)
    plot_eora_scaling_aligned_r64(df, out_dir)
    plot_multiseed_confirmation(df, out_dir)

    # appendix / supplementary
    plot_gap_to_teacher(df, teacher_loss, out_dir)
    plot_completed_runs_appendix(
        df,
        branch="LoRA",
        out_path=out_dir / "fig_exp1_lm_lora_scaling_appendix.png",
        title="Exp1-lm Appendix: LoRA Completed Runs across Target-Module Groups",
    )
    plot_completed_runs_appendix(
        df,
        branch="EoRA",
        out_path=out_dir / "fig_exp1_lm_eora_completed_runs_appendix.png",
        title="Exp1-lm Appendix: EoRA Completed Runs across Target-Module Groups",
    )

    print(f"Saved report figures and tables to: {out_dir}")


if __name__ == "__main__":
    main()