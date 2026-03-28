# scripts/utils/make_report_figs_lm_exp3_lora_init_compare.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "outputs" / "lm" / "exp3_lora_init_compare_summary"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lm" / "report_figs" / "exp3_lora_init_compare"

# ===== fixed colors =====
COLOR_RANDOM = "#e45238"
COLOR_EORAINIT = "#7dacd1"


# ============================================================
# IO helpers
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def must_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fmt_float(x, nd=4) -> str:
    if pd.isna(x):
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def latex_escape_text(s: str) -> str:
    if s is None:
        return "--"
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


# ============================================================
# Selection helpers
# ============================================================

def select_pair(
    pair_df: pd.DataFrame,
    bit: Optional[int],
    rank: Optional[int],
    seed: Optional[int],
    alpha_over_r: Optional[float],
) -> pd.Series:
    df = pair_df.copy()

    if "bit" in df.columns:
        df["bit_num"] = to_num(df["bit"])
    if "rank" in df.columns:
        df["rank_num"] = to_num(df["rank"])
    if "seed" in df.columns:
        df["seed_num"] = to_num(df["seed"])
    if "alpha_over_r" in df.columns:
        df["alpha_over_r_num"] = to_num(df["alpha_over_r"])

    if bit is not None and "bit_num" in df.columns:
        df = df[df["bit_num"] == bit]
    if rank is not None and "rank_num" in df.columns:
        df = df[df["rank_num"] == rank]
    if seed is not None and "seed_num" in df.columns:
        df = df[df["seed_num"] == seed]
    if alpha_over_r is not None and "alpha_over_r_num" in df.columns:
        df = df[np.isclose(df["alpha_over_r_num"], alpha_over_r)]

    if len(df) == 0:
        raise ValueError(
            f"No matched pair found for bit={bit}, rank={rank}, seed={seed}, alpha_over_r={alpha_over_r}"
        )

    if len(df) > 1:
        print("[Warn] Multiple matched pairs found. Using the first one.")
        print(df[["bit", "seed", "rank", "alpha_over_r", "target_modules"]].to_string(index=False))

    return df.iloc[0]


def get_run_rows(run_df: pd.DataFrame, pair_row: pd.Series) -> Tuple[pd.Series, pd.Series]:
    random_run_dir = str(pair_row["random_run_dir"])
    eorainit_run_dir = str(pair_row["eorainit_run_dir"])

    rand = run_df[run_df["run_dir"] == random_run_dir]
    eora = run_df[run_df["run_dir"] == eorainit_run_dir]

    if len(rand) != 1 or len(eora) != 1:
        raise ValueError("Could not uniquely find random/eorainit rows in run_summary.csv")

    return rand.iloc[0], eora.iloc[0]


def get_history_for_run(history_df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    df = history_df[history_df["run_dir"] == run_dir].copy()
    if len(df) == 0:
        raise ValueError(f"No history rows found for run_dir={run_dir}")

    if "global_step" in df.columns:
        df["global_step"] = to_num(df["global_step"])
    if "epoch" in df.columns:
        df["epoch"] = to_num(df["epoch"])

    tag = df["tag"].astype(str)
    keep = (tag == "Init") | tag.str.startswith("Eval@step")
    df = df[keep].copy()
    df = df.sort_values(["global_step", "epoch"], kind="mergesort")
    df = df.drop_duplicates(subset=["global_step"], keep="first")

    return df.reset_index(drop=True)


def method_label(init_mode: str) -> str:
    mode = str(init_mode).lower()
    if mode == "random":
        return "Random Initialization"
    if mode == "eora_adapter":
        return "EoRA Initialization"
    return str(init_mode)


def tag_suffix(pair_row: pd.Series) -> str:
    bit = int(float(pair_row["bit"])) if not pd.isna(pair_row["bit"]) else -1
    rank = int(float(pair_row["rank"])) if not pd.isna(pair_row["rank"]) else -1
    seed = pair_row["seed"]
    suffix = f"q{bit}_r{rank}"
    if not pd.isna(seed):
        suffix += f"_seed{int(float(seed))}"
    return suffix


# ============================================================
# Academic-style plotting helpers
# ============================================================

def setup_publication_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.2,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,
        "savefig.dpi": 400,
    })


def style_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, pad=8, weight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_alpha(0.9)
        spine.set_linewidth(0.9)


def add_two_method_lines(
    ax,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    label1: str,
    label2: str,
):
    ax.plot(
        x1, y1,
        marker="o",
        color=COLOR_RANDOM,
        label=label1,
        linewidth=2.1,
        markersize=4.0,
        markeredgewidth=0.6,
        markeredgecolor="white",
    )
    ax.plot(
        x2, y2,
        marker="o",
        color=COLOR_EORAINIT,
        label=label2,
        linewidth=2.1,
        markersize=4.0,
        markeredgewidth=0.6,
        markeredgecolor="white",
    )


# ============================================================
# Table builders
# ============================================================

def build_main_metrics_table(rand: pd.Series, eora: pd.Series) -> pd.DataFrame:
    rows = []
    for r in [rand, eora]:
        rows.append({
            "Method": method_label(r["init_mode"]),
            "Init Val Loss": r["init_val_loss"],
            "Init Test Loss": r["init_test_loss"],
            "Init Val KL": r["init_val_kl"],
            "Init Test KL": r["init_test_kl"],
            "Final Val Loss": r["final_val_loss"],
            "Final Test Loss": r["final_test_loss"],
            "Final Val KL": r["final_val_kl"],
            "Final Test KL": r["final_test_kl"],
            "Δ Val Loss": r["gain_val_loss_init_to_final"],
            "Δ Test Loss": r["gain_test_loss_init_to_final"],
            "Δ Val KL": r["gain_val_kl_init_to_final"],
            "Δ Test KL": r["gain_test_kl_init_to_final"],
        })
    return pd.DataFrame(rows)


def build_speed_table(rand: pd.Series, eora: pd.Series) -> pd.DataFrame:
    rows = []
    for r in [rand, eora]:
        rows.append({
            "Method": method_label(r["init_mode"]),
            "Step@50% Val-Loss Gain": r["step_val_loss_50pct_gain"],
            "Step@90% Val-Loss Gain": r["step_val_loss_90pct_gain"],
            "Step@50% Val-KL Gain": r["step_val_kl_50pct_gain"],
            "Step@90% Val-KL Gain": r["step_val_kl_90pct_gain"],
            "Best Hist Val Loss": r["best_hist_val_loss"],
            "Best Hist Val-Loss Step": r["best_hist_val_loss_step"],
            "Best Hist Val KL": r["best_hist_val_kl"],
            "Best Hist Val-KL Step": r["best_hist_val_kl_step"],
        })
    return pd.DataFrame(rows)


def build_key_checkpoint_table(
    rand_hist: pd.DataFrame,
    eora_hist: pd.DataFrame,
    rand: pd.Series,
    eora: pd.Series,
) -> pd.DataFrame:
    wanted_steps = [0, 100, 200, 400, 800, 1000, 1500, 2000, 2500, 3000]

    rand_map = {int(s): row for _, row in rand_hist.iterrows() for s in [int(row["global_step"])]}
    eora_map = {int(s): row for _, row in eora_hist.iterrows() for s in [int(row["global_step"])]}

    steps = [s for s in wanted_steps if s in rand_map and s in eora_map]

    rows = []
    for s in steps:
        rr = rand_map[s]
        ee = eora_map[s]
        rows.append({
            "Checkpoint": "Init" if s == 0 else f"Step {s}",
            "Random Val Loss": rr.get("val_loss"),
            "EoRAInit Val Loss": ee.get("val_loss"),
            "Random Val KL": rr.get("val_kl_to_teacher"),
            "EoRAInit Val KL": ee.get("val_kl_to_teacher"),
            "Random Test Loss": rr.get("test_loss"),
            "EoRAInit Test Loss": ee.get("test_loss"),
            "Random Test KL": rr.get("test_kl_to_teacher"),
            "EoRAInit Test KL": ee.get("test_kl_to_teacher"),
        })

    rows.append({
        "Checkpoint": "Final",
        "Random Val Loss": rand["final_val_loss"],
        "EoRAInit Val Loss": eora["final_val_loss"],
        "Random Val KL": rand["final_val_kl"],
        "EoRAInit Val KL": eora["final_val_kl"],
        "Random Test Loss": rand["final_test_loss"],
        "EoRAInit Test Loss": eora["final_test_loss"],
        "Random Test KL": rand["final_test_kl"],
        "EoRAInit Test KL": eora["final_test_kl"],
    })

    return pd.DataFrame(rows)


def build_settings_table(rand: pd.Series, eora: pd.Series, pair_row: pd.Series) -> pd.DataFrame:
    fields = [
        ("Bit", rand.get("bit"), eora.get("bit")),
        ("Rank", rand.get("rank"), eora.get("rank")),
        ("Alpha", rand.get("alpha"), eora.get("alpha")),
        ("Alpha / Rank", rand.get("alpha_over_r"), eora.get("alpha_over_r")),
        ("Target Modules", rand.get("target_modules"), eora.get("target_modules")),
        ("Quantized Backbone", rand.get("quantized_model_dir"), eora.get("quantized_model_dir")),
        ("Teacher / Optimized Model", rand.get("optimized_model_dir"), eora.get("optimized_model_dir")),
        ("Learning Rate", rand.get("lr"), eora.get("lr")),
        ("Grad Accum", rand.get("grad_accum_steps"), eora.get("grad_accum_steps")),
        ("Max Train Steps", rand.get("max_train_steps"), eora.get("max_train_steps")),
        ("Eval Every Steps", rand.get("eval_every_steps"), eora.get("eval_every_steps")),
        ("Initialization", "random", "eora_adapter"),
        ("EoRA Adapter Dir", "--", eora.get("adapter_dir")),
    ]

    rows = [{"Field": f, "RandomInit": a, "EoRAInit": b} for f, a, b in fields]
    return pd.DataFrame(rows)


def build_pairwise_table(pair_row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "Bit": pair_row.get("bit"),
        "Rank": pair_row.get("rank"),
        "Alpha / Rank": pair_row.get("alpha_over_r"),
        "Target Modules": pair_row.get("target_modules"),
        "Δ Init Val Loss (EoRAInit - RandomInit)": pair_row.get("delta_init_val_loss"),
        "Δ Init Test Loss (EoRAInit - RandomInit)": pair_row.get("delta_init_test_loss"),
        "Δ Init Val KL (EoRAInit - RandomInit)": pair_row.get("delta_init_val_kl"),
        "Δ Init Test KL (EoRAInit - RandomInit)": pair_row.get("delta_init_test_kl"),
        "Δ Final Val Loss (EoRAInit - RandomInit)": pair_row.get("delta_final_val_loss"),
        "Δ Final Test Loss (EoRAInit - RandomInit)": pair_row.get("delta_final_test_loss"),
        "Δ Final Val KL (EoRAInit - RandomInit)": pair_row.get("delta_final_val_kl"),
        "Δ Final Test KL (EoRAInit - RandomInit)": pair_row.get("delta_final_test_kl"),
        "Mean Δ Val Loss": pair_row.get("mean_delta_val_loss"),
        "Mean Δ Test Loss": pair_row.get("mean_delta_test_loss"),
        "Mean Δ Val KL": pair_row.get("mean_delta_val_kl"),
        "Mean Δ Test KL": pair_row.get("mean_delta_test_kl"),
        "EoRAInit Better Fraction (Val Loss)": pair_row.get("eorainit_better_fraction_val_loss"),
        "EoRAInit Better Fraction (Val KL)": pair_row.get("eorainit_better_fraction_val_kl"),
    }])


# ============================================================
# LaTeX export
# ============================================================

def df_to_latex_table(
    df: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
    float_cols: Optional[List[str]] = None,
):
    ensure_dir(out_path.parent)
    work = df.copy()

    if float_cols is None:
        float_cols = []
        for c in work.columns:
            if pd.api.types.is_numeric_dtype(work[c]):
                float_cols.append(c)

    for c in work.columns:
        if c in float_cols:
            work[c] = work[c].map(lambda x: fmt_float(x, 4))
        else:
            work[c] = work[c].map(lambda x: "--" if pd.isna(x) else latex_escape_text(x))

    latex_str = work.to_latex(index=False, escape=False)
    latex_str = latex_str.replace("\\begin{table}", "\\begin{table}[H]")
    if "\\caption{" not in latex_str:
        latex_str = latex_str.replace(
            "\\begin{table}[H]",
            f"\\begin{{table}}[H]\n\\caption{{{caption}}}\n\\label{{{label}}}",
            1,
        )

    with out_path.open("w", encoding="utf-8") as f:
        f.write(latex_str)


# ============================================================
# Big combined figure
# ============================================================

def make_combined_2x3_figure(
    rand_hist: pd.DataFrame,
    eora_hist: pd.DataFrame,
    out_path: Path,
):
    setup_publication_style()

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), dpi=500)
    axes = axes.flatten()

    x_rand = rand_hist["global_step"].to_numpy()
    x_eora = eora_hist["global_step"].to_numpy()

    panels = [
        ("Validation Loss", "Validation loss", "val_loss"),
        ("Validation KL", "Validation KL to teacher", "val_kl_to_teacher"),
        ("Validation Logits MSE", "Validation logits MSE", "val_mse_logits_to_teacher"),
        ("Test Loss", "Test loss", "test_loss"),
        ("Test KL", "Test KL to teacher", "test_kl_to_teacher"),
        ("Test Logits MSE", "Test logits MSE", "test_mse_logits_to_teacher"),
    ]

    for ax, (title, ylabel, key) in zip(axes, panels):
        y_rand = to_num(rand_hist[key]).to_numpy()
        y_eora = to_num(eora_hist[key]).to_numpy()

        add_two_method_lines(
            ax,
            x_rand, y_rand,
            x_eora, y_eora,
            "Random Initialization",
            "EoRA Initialization",
        )
        style_axis(ax, title=title, xlabel="Global step", ylabel=ylabel)

    # shared legend at top center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.92),
        fontsize=10,
    )

    fig.suptitle(
        "LoRA Initialization Comparison on the Same Quantized Backbone",
        y=0.965,
        fontsize=18,
        weight="semibold"
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])
    fig.savefig(out_path, dpi=900, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", type=str, default=str(DEFAULT_SUMMARY_DIR))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--bit", type=int, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--alpha_over_r", type=float, default=None)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    setup_publication_style()

    run_summary_csv = summary_dir / "run_summary.csv"
    history_csv = summary_dir / "history_wide.csv"
    pairwise_summary_csv = summary_dir / "pairwise_summary.csv"

    run_df = must_read_csv(run_summary_csv)
    history_df = must_read_csv(history_csv)
    pair_df = must_read_csv(pairwise_summary_csv)

    pair_row = select_pair(
        pair_df=pair_df,
        bit=args.bit,
        rank=args.rank,
        seed=args.seed,
        alpha_over_r=args.alpha_over_r,
    )

    rand_row, eora_row = get_run_rows(run_df, pair_row)
    rand_hist = get_history_for_run(history_df, rand_row["run_dir"])
    eora_hist = get_history_for_run(history_df, eora_row["run_dir"])

    suffix = tag_suffix(pair_row)

    # --------------------------------------------------------
    # Combined big figure
    # --------------------------------------------------------
    combined_fig = out_dir / f"fig_exp3_initcmp_{suffix}_combined_2x3.png"
    make_combined_2x3_figure(rand_hist, eora_hist, combined_fig)

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------
    df_main_metrics = build_main_metrics_table(rand_row, eora_row)
    df_speed = build_speed_table(rand_row, eora_row)
    df_key_ckpt = build_key_checkpoint_table(rand_hist, eora_hist, rand_row, eora_row)
    df_settings = build_settings_table(rand_row, eora_row, pair_row)
    df_pairwise = build_pairwise_table(pair_row)

    table_main_metrics_csv = out_dir / f"table_main_exp3_initcmp_{suffix}_metrics.csv"
    table_main_metrics_tex = out_dir / f"table_main_exp3_initcmp_{suffix}_metrics.tex"

    table_main_speed_csv = out_dir / f"table_main_exp3_initcmp_{suffix}_speed.csv"
    table_main_speed_tex = out_dir / f"table_main_exp3_initcmp_{suffix}_speed.tex"

    table_app_ckpt_csv = out_dir / f"table_app_exp3_initcmp_{suffix}_keycheckpoints.csv"
    table_app_ckpt_tex = out_dir / f"table_app_exp3_initcmp_{suffix}_keycheckpoints.tex"

    table_app_settings_csv = out_dir / f"table_app_exp3_initcmp_{suffix}_settings.csv"
    table_app_settings_tex = out_dir / f"table_app_exp3_initcmp_{suffix}_settings.tex"

    table_app_pairwise_csv = out_dir / f"table_app_exp3_initcmp_{suffix}_pairwise.csv"
    table_app_pairwise_tex = out_dir / f"table_app_exp3_initcmp_{suffix}_pairwise.tex"

    df_main_metrics.to_csv(table_main_metrics_csv, index=False)
    df_speed.to_csv(table_main_speed_csv, index=False)
    df_key_ckpt.to_csv(table_app_ckpt_csv, index=False)
    df_settings.to_csv(table_app_settings_csv, index=False)
    df_pairwise.to_csv(table_app_pairwise_csv, index=False)

    df_to_latex_table(
        df_main_metrics,
        table_main_metrics_tex,
        caption="Main results for the LoRA initialization comparison experiment, including initialization metrics, final metrics, and the improvement from initialization to the final restored model.",
        label=f"tab:exp3_initcmp_{suffix}_main_metrics",
    )

    df_to_latex_table(
        df_speed,
        table_main_speed_tex,
        caption="Convergence-speed summary for the LoRA initialization comparison experiment. Step@50\\% and Step@90\\% denote the first training step at which 50\\% and 90\\% of the total validation improvement have been achieved.",
        label=f"tab:exp3_initcmp_{suffix}_speed",
    )

    df_to_latex_table(
        df_key_ckpt,
        table_app_ckpt_tex,
        caption="Key checkpoints from the initialization-comparison experiment, showing representative validation and test metrics at selected training steps.",
        label=f"tab:exp3_initcmp_{suffix}_checkpoints",
    )

    df_to_latex_table(
        df_settings,
        table_app_settings_tex,
        caption="Controlled experimental settings for the initialization-comparison experiment. Except for the initialization strategy, the two runs use the same backbone, LoRA setup, and training budget.",
        label=f"tab:exp3_initcmp_{suffix}_settings",
    )

    df_to_latex_table(
        df_pairwise,
        table_app_pairwise_tex,
        caption="Pairwise summary of EoRA-initialized LoRA minus randomly initialized LoRA. Negative values for loss/KL deltas indicate better performance for EoRA initialization.",
        label=f"tab:exp3_initcmp_{suffix}_pairwise",
    )

    print("\n=== Generated report figures/tables for LoRA init comparison ===\n")
    print(f"Selected pair: bit={pair_row['bit']}, rank={pair_row['rank']}, seed={pair_row['seed']}")
    print(f"Random run   : {rand_row['run_dir']}")
    print(f"EoRA-init run: {eora_row['run_dir']}\n")
    print("[Main combined figure]")
    print(combined_fig)
    print("\n[Tables]")
    print(table_main_metrics_tex)
    print(table_main_speed_tex)
    print(table_app_ckpt_tex)
    print(table_app_settings_tex)
    print(table_app_pairwise_tex)
    print("\nDone.")


if __name__ == "__main__":
    main()