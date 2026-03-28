# scripts/utils/make_report_figs_lm_exp3.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "lm" / "exp3_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lm" / "report_exp3"


# ============================================================
# global style
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


def _find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# color system (same style family as lm-exp2)
# ============================================================

def method_colors():
    return {
        "Teacher": "#fde395",
        "Quantized": "#e2f2f0",
        "LoRA": "#e45238",
        "EoRA": "#7dacd1",
        "LoRA_r16": "#f2b29f",
        "LoRA_r32": "#e45238",
        "EoRA_r16": "#bcd3e6",
        "EoRA_r32": "#7dacd1",
    }


# ============================================================
# load / normalize
# ============================================================

def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")

    raw = pd.read_csv(path)
    raw.columns = [_norm_col(c) for c in raw.columns]
    df = raw.copy()

    # unify key metric names robustly
    rename_map = {}

    test_kl_col = _find_first_existing(df, [
        "test_kl",
        "kl_to_teacher",
        "test_kl_to_teacher",
        "kl",
    ])
    if test_kl_col is not None and test_kl_col != "test_kl":
        rename_map[test_kl_col] = "test_kl"

    test_mse_col = _find_first_existing(df, [
        "test_mse",
        "mse_logits_to_teacher",
        "test_mse_logits_to_teacher",
        "mse",
    ])
    if test_mse_col is not None and test_mse_col != "test_mse":
        rename_map[test_mse_col] = "test_mse"

    ar_col = _find_first_existing(df, ["alpha_over_r", "ar"])
    if ar_col is not None and ar_col != "alpha_over_r":
        rename_map[ar_col] = "alpha_over_r"

    gain_loss_col = _find_first_existing(df, ["loss_gain_vs_quant", "gain_vs_quant_loss"])
    if gain_loss_col is not None and gain_loss_col != "loss_gain_vs_quant":
        rename_map[gain_loss_col] = "loss_gain_vs_quant"

    gain_kl_col = _find_first_existing(df, ["kl_gain_vs_quant", "gain_vs_quant_kl"])
    if gain_kl_col is not None and gain_kl_col != "kl_gain_vs_quant":
        rename_map[gain_kl_col] = "kl_gain_vs_quant"

    df = df.rename(columns=rename_map)

    for c in [
        "bit", "rank", "alpha_over_r",
        "test_loss", "test_ppl", "test_kl", "test_mse",
        "loss_gain_vs_quant", "kl_gain_vs_quant",
        "teacher_test_loss", "teacher_test_ppl"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "branch" in df.columns:
        raw_branch = df["branch"].astype(str).str.strip().str.lower()
        df["branch_std"] = raw_branch.replace({
            "quantized": "Quantized",
            "lora": "LoRA",
            "eora": "EoRA",
            "teacher": "Teacher",
            "optimized": "Teacher",
        })
    else:
        raise ValueError("Summary CSV must contain a 'branch' column.")

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


def get_quantized_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["branch_std"] == "Quantized"].copy()


def get_recovered_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["branch_std"].isin(["LoRA", "EoRA"])].copy()


# ============================================================
# tables
# ============================================================

def make_main_summary_table(
    df: pd.DataFrame,
    teacher_loss: Optional[float],
    teacher_ppl: Optional[float],
) -> pd.DataFrame:
    quant_df = get_quantized_rows(df)
    rec_df = get_recovered_rows(df)

    rows = []

    if teacher_loss is not None:
        rows.append({
            "method": "Optimized teacher",
            "branch": "Teacher",
            "bit": None,
            "rank": None,
            "alpha_over_r": None,
            "test_loss": teacher_loss,
            "test_ppl": teacher_ppl,
            "test_kl": None,
            "test_mse": None,
            "gain_vs_quant_loss": None,
            "gain_vs_quant_kl": None,
        })

    for bit in sorted(quant_df["bit"].dropna().unique()):
        qsub = quant_df[quant_df["bit"] == bit].copy()
        if len(qsub) > 0:
            q = qsub.iloc[0]
            rows.append({
                "method": f"Quantized baseline ({int(bit)}-bit)",
                "branch": "Quantized",
                "bit": int(bit),
                "rank": None,
                "alpha_over_r": None,
                "test_loss": q.get("test_loss", np.nan),
                "test_ppl": q.get("test_ppl", np.nan),
                "test_kl": q.get("test_kl", np.nan),
                "test_mse": q.get("test_mse", np.nan),
                "gain_vs_quant_loss": 0.0,
                "gain_vs_quant_kl": 0.0,
            })

        for branch in ["EoRA", "LoRA"]:
            sub = rec_df[(rec_df["bit"] == bit) & (rec_df["branch_std"] == branch)].copy()
            if len(sub) == 0:
                continue

            best_task = sub.sort_values("test_loss", ascending=True).iloc[0]
            rows.append({
                "method": f"Best {branch} ({int(bit)}-bit)",
                "branch": branch,
                "bit": int(bit),
                "rank": best_task.get("rank", np.nan),
                "alpha_over_r": best_task.get("alpha_over_r", np.nan),
                "test_loss": best_task.get("test_loss", np.nan),
                "test_ppl": best_task.get("test_ppl", np.nan),
                "test_kl": best_task.get("test_kl", np.nan),
                "test_mse": best_task.get("test_mse", np.nan),
                "gain_vs_quant_loss": best_task.get("loss_gain_vs_quant", np.nan),
                "gain_vs_quant_kl": best_task.get("kl_gain_vs_quant", np.nan),
            })

    return pd.DataFrame(rows)


def make_full_results_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "branch_std", "bit", "rank", "alpha_over_r",
        "test_loss", "test_ppl", "test_kl", "test_mse",
        "loss_gain_vs_quant", "kl_gain_vs_quant",
        "exp_name", "run_name"
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out = out.sort_values(
        by=[c for c in ["bit", "branch_std", "rank", "alpha_over_r"] if c in out.columns],
        ascending=True,
    )
    return out


# ============================================================
# plotting helpers
# ============================================================

def _get_value(df: pd.DataFrame, bit: int, branch: str, rank: Optional[int], metric: str) -> float:
    sub = df[(df["bit"] == bit) & (df["branch_std"] == branch)].copy()
    if rank is not None:
        sub = sub[sub["rank"] == rank]
    if len(sub) == 0 or metric not in sub.columns or sub[metric].isna().all():
        return np.nan
    return float(sub[metric].mean())


def _build_method_metric_rows(
    df: pd.DataFrame,
    metric: str,
    teacher_value: Optional[float] = None,
) -> Dict[int, List[dict]]:
    bits = sorted(int(b) for b in df["bit"].dropna().unique())
    rows = {}

    for bit in bits:
        r = []
        if teacher_value is not None:
            r.append({"label": "Optimized", "value": teacher_value, "color": method_colors()["Teacher"]})

        r.append({"label": "Quantized", "value": _get_value(df, bit, "Quantized", None, metric), "color": method_colors()["Quantized"]})
        r.append({"label": "EoRA-r16", "value": _get_value(df, bit, "EoRA", 16, metric), "color": method_colors()["EoRA_r16"]})
        r.append({"label": "EoRA-r32", "value": _get_value(df, bit, "EoRA", 32, metric), "color": method_colors()["EoRA_r32"]})
        r.append({"label": "LoRA-r16", "value": _get_value(df, bit, "LoRA", 16, metric), "color": method_colors()["LoRA_r16"]})
        r.append({"label": "LoRA-r32", "value": _get_value(df, bit, "LoRA", 32, metric), "color": method_colors()["LoRA_r32"]})
        rows[bit] = r

    return rows


# ============================================================
# main plots
# ============================================================

def plot_overall_metric_bar(
    df: pd.DataFrame,
    teacher_value: Optional[float],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    value_rotation: int = 90,
):
    rows_by_bit = _build_method_metric_rows(df, metric=metric, teacher_value=teacher_value)
    bits = sorted(rows_by_bit.keys())

    fig, axes = plt.subplots(1, len(bits), figsize=(11.8, 5.4), sharey=True)
    if len(bits) == 1:
        axes = [axes]

    all_vals = []
    legend_handles = None
    legend_labels = None

    for ax, bit in zip(axes, bits):
        rows = rows_by_bit[bit]
        xs = np.arange(len(rows))
        vals = [r["value"] for r in rows]
        labels = [r["label"] for r in rows]
        colors = [r["color"] for r in rows]

        bars = ax.bar(
            xs,
            vals,
            width=0.72,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
        )

        finite_vals = [v for v in vals if pd.notna(v)]
        offset = max(0.002, (max(finite_vals) - min(finite_vals)) * 0.015) if finite_vals else 0.002

        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    float(v) + offset,
                    f"{float(v):.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    rotation=0,
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(f"{bit}-bit")
        style_axes(ax, add_grid=True)

        all_vals.extend([v for v in vals if pd.notna(v)])

        if legend_handles is None:
            legend_handles = bars
            legend_labels = labels

    axes[0].set_ylabel(ylabel)
    set_tight_ylim(axes[0], all_vals, pad_ratio=0.10, min_pad=0.02)
    fig.suptitle(title, y=1.02)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            frameon=False,
            ncol=1,
        )

    save_fig(fig, out_path)


def plot_gain_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    bits = sorted(int(b) for b in df["bit"].dropna().unique())
    fig, axes = plt.subplots(1, len(bits), figsize=(10.6, 5.2), sharey=True)
    if len(bits) == 1:
        axes = [axes]

    all_vals = []

    for ax, bit in zip(axes, bits):
        sub = df[(df["bit"] == bit) & (df["branch_std"].isin(["LoRA", "EoRA"]))].copy()
        rows = []

        for branch, rank, label, color_key in [
            ("EoRA", 16, "EoRA-r16", "EoRA_r16"),
            ("EoRA", 32, "EoRA-r32", "EoRA_r32"),
            ("LoRA", 16, "LoRA-r16", "LoRA_r16"),
            ("LoRA", 32, "LoRA-r32", "LoRA_r32"),
        ]:
            ss = sub[(sub["branch_std"] == branch) & (sub["rank"] == rank)]
            val = float(ss[metric].mean()) if len(ss) and metric in ss.columns and ss[metric].notna().any() else np.nan
            rows.append({
                "label": label,
                "value": val,
                "color": method_colors()[color_key],
            })

        xs = np.arange(len(rows))
        vals = [r["value"] for r in rows]
        labels = [r["label"] for r in rows]
        colors = [r["color"] for r in rows]

        bars = ax.bar(
            xs,
            vals,
            width=0.72,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
        )

        finite_vals = [v for v in vals if pd.notna(v)]
        offset = max(0.001, (max(finite_vals) - min(finite_vals)) * 0.03) if finite_vals else 0.001

        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    float(v) + offset,
                    f"{float(v):.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    rotation=0,
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(f"{bit}-bit")
        style_axes(ax, add_grid=True)
        all_vals.extend([v for v in vals if pd.notna(v)])

    axes[0].set_ylabel(ylabel)
    set_tight_ylim(axes[0], all_vals, pad_ratio=0.15, min_pad=0.01)
    fig.suptitle(title, y=1.02)

    legend_handles = [
        Patch(facecolor=method_colors()["EoRA_r16"], edgecolor="#333333", label="EoRA-r16"),
        Patch(facecolor=method_colors()["EoRA_r32"], edgecolor="#333333", label="EoRA-r32"),
        Patch(facecolor=method_colors()["LoRA_r16"], edgecolor="#333333", label="LoRA-r16"),
        Patch(facecolor=method_colors()["LoRA_r32"], edgecolor="#333333", label="LoRA-r32"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        ncol=1,
    )

    save_fig(fig, out_path)


def plot_rank_sensitivity(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    bits = sorted(int(b) for b in df["bit"].dropna().unique())
    fig, axes = plt.subplots(1, len(bits), figsize=(10.8, 5.0), sharey=True)
    if len(bits) == 1:
        axes = [axes]

    all_vals = []

    for ax, bit in zip(axes, bits):
        sub = df[(df["bit"] == bit) & (df["branch_std"].isin(["LoRA", "EoRA"]))].copy()

        rows = []
        for branch, rank, label, color_key in [
            ("EoRA", 16, "EoRA-r16", "EoRA_r16"),
            ("EoRA", 32, "EoRA-r32", "EoRA_r32"),
            ("LoRA", 16, "LoRA-r16", "LoRA_r16"),
            ("LoRA", 32, "LoRA-r32", "LoRA_r32"),
        ]:
            ss = sub[(sub["branch_std"] == branch) & (sub["rank"] == rank)]
            val = float(ss[metric].mean()) if len(ss) and metric in ss.columns and ss[metric].notna().any() else np.nan
            rows.append({
                "label": label,
                "value": val,
                "color": method_colors()[color_key],
            })

        xs = np.arange(len(rows))
        vals = [r["value"] for r in rows]
        labels = [r["label"] for r in rows]
        colors = [r["color"] for r in rows]

        bars = ax.bar(
            xs,
            vals,
            width=0.72,
            color=colors,
            edgecolor="#333333",
            linewidth=0.8,
        )

        finite_vals = [v for v in vals if pd.notna(v)]
        offset = max(0.001, (max(finite_vals) - min(finite_vals)) * 0.03) if finite_vals else 0.001

        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    float(v) + offset,
                    f"{float(v):.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.3,
                    rotation=0,
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(f"{bit}-bit")
        style_axes(ax, add_grid=True)

        all_vals.extend([v for v in vals if pd.notna(v)])

    axes[0].set_ylabel(ylabel)
    set_tight_ylim(axes[0], all_vals, pad_ratio=0.12, min_pad=0.015)
    fig.suptitle(title, y=1.02)

    legend_handles = [
        Patch(facecolor=method_colors()["EoRA_r16"], edgecolor="#333333", label="EoRA-r16"),
        Patch(facecolor=method_colors()["EoRA_r32"], edgecolor="#333333", label="EoRA-r32"),
        Patch(facecolor=method_colors()["LoRA_r16"], edgecolor="#333333", label="LoRA-r16"),
        Patch(facecolor=method_colors()["LoRA_r32"], edgecolor="#333333", label="LoRA-r32"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        ncol=1,
    )

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

    save_df(main_table, out_dir / "table_exp3_lm_main_summary.csv")
    save_df(full_table, out_dir / "table_exp3_lm_full_results.csv")

    # -------------------------
    # main figures
    # -------------------------
    plot_overall_metric_bar(
        df,
        teacher_value=teacher_loss,
        metric="test_loss",
        ylabel="Test Loss",
        title="Exp3-lm: Quantization Recovery under 4-bit and 3-bit GPTQ",
        out_path=out_dir / "fig_exp3_lm_overall_loss_bar.png",
    )

    if "test_kl" in df.columns and df["test_kl"].notna().any():
        plot_overall_metric_bar(
            df,
            teacher_value=None,
            metric="test_kl",
            ylabel="Test KL to Optimized Teacher",
            title="Exp3-lm: Teacher-Alignment Recovery under 4-bit and 3-bit GPTQ",
            out_path=out_dir / "fig_exp3_lm_overall_kl_bar.png",
            value_rotation=0,
        )
    else:
        print("[skip] fig_exp3_lm_overall_kl_bar: no KL column/data")

    if "loss_gain_vs_quant" in df.columns and df["loss_gain_vs_quant"].notna().any():
        plot_gain_bar(
            df,
            metric="loss_gain_vs_quant",
            ylabel="Test Loss Gain vs Quantized Baseline",
            title="Exp3-lm: Task Recovery Gain relative to Quantized Baseline",
            out_path=out_dir / "fig_exp3_lm_loss_gain_bar.png",
        )
    else:
        print("[skip] fig_exp3_lm_loss_gain_bar: no loss_gain_vs_quant column/data")

    if "kl_gain_vs_quant" in df.columns and df["kl_gain_vs_quant"].notna().any():
        plot_gain_bar(
            df,
            metric="kl_gain_vs_quant",
            ylabel="Test KL Gain vs Quantized Baseline",
            title="Exp3-lm: Alignment Gain relative to Quantized Baseline",
            out_path=out_dir / "fig_exp3_lm_kl_gain_bar.png",
        )
    else:
        print("[skip] fig_exp3_lm_kl_gain_bar: no kl_gain_vs_quant column/data")

    plot_rank_sensitivity(
        df,
        metric="test_loss",
        ylabel="Test Loss",
        title="Exp3-lm: Rank Sensitivity under Different Quantization Levels",
        out_path=out_dir / "fig_exp3_lm_rank_sensitivity_loss.png",
    )

    if "test_kl" in df.columns and df["test_kl"].notna().any():
        plot_rank_sensitivity(
            df,
            metric="test_kl",
            ylabel="Test KL to Optimized Teacher",
            title="Exp3-lm: Rank Sensitivity of Teacher Alignment",
            out_path=out_dir / "fig_exp3_lm_rank_sensitivity_kl.png",
        )
    else:
        print("[skip] fig_exp3_lm_rank_sensitivity_kl: no KL column/data")


    print(f"Saved report figures and tables to: {out_dir}")


if __name__ == "__main__":
    main()