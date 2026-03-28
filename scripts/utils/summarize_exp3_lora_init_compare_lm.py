# scripts/utils/summarize_exp3_lora_init_compare_lm.py
from __future__ import annotations

import json
import re
import csv
import math
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP3_ROOT = PROJECT_ROOT / "outputs" / "lm" / "exp3"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lm" / "exp3_lora_init_compare_summary"
DEFAULT_PREFIX = "33_lora_init_compare"


# ============================================================
# Basic IO
# ============================================================

def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: List[Dict[str, Any]], out_path: Path, fieldnames: Optional[List[str]] = None):
    ensure_dir(out_path.parent)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if fieldnames:
                w.writeheader()
        return

    if fieldnames is None:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        fieldnames = list(sorted(keys))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# ============================================================
# Small helpers
# ============================================================

def fmt(x, nd=4):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def as_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def as_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def infer_bit_from_text(*items: Any) -> Optional[int]:
    pats = [
        re.compile(r"(?:^|[_\-/])q([234])(?:$|[_\-/])", re.IGNORECASE),
        re.compile(r"([234])bit", re.IGNORECASE),
        re.compile(r"gptq([234])", re.IGNORECASE),
    ]
    for item in items:
        if item is None:
            continue
        s = str(item).lower()
        for pat in pats:
            m = pat.search(s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    return None


def list_to_string(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, list):
        return ",".join(str(v) for v in x)
    return str(x)


def parse_rank_alpha_from_cfg(cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    lora_cfg = cfg.get("lora", {}) if isinstance(cfg.get("lora", {}), dict) else {}
    rank = as_int(lora_cfg.get("rank"))
    alpha = as_float(lora_cfg.get("alpha"))
    alpha_over_r = None
    if rank not in (None, 0) and alpha is not None:
        alpha_over_r = alpha / rank
    return rank, alpha, alpha_over_r


def parse_init_mode(cfg: Dict[str, Any], meta: Dict[str, Any]) -> Optional[str]:
    for src in [cfg, meta]:
        d = src.get("lora_init", {})
        if isinstance(d, dict):
            mode = d.get("mode")
            if mode is not None:
                return str(mode)
    return None


def parse_target_modules(cfg: Dict[str, Any], meta: Dict[str, Any]) -> Optional[str]:
    for src in [cfg, meta]:
        d = src.get("lora", {})
        if isinstance(d, dict) and "target_modules" in d:
            return list_to_string(d.get("target_modules"))
    return None


def parse_train_field(cfg: Dict[str, Any], meta: Dict[str, Any], key: str):
    for src in [cfg, meta]:
        d = src.get("train", {})
        if isinstance(d, dict) and key in d:
            return d.get(key)
    return None


def event_order(tag: str) -> int:
    if tag == "Init":
        return 0
    if str(tag).startswith("Eval@step"):
        return 1
    if str(tag).startswith("EpochEnd"):
        return 2
    if tag == "FinalBestRestored":
        return 3
    return 9


# ============================================================
# History processing
# ============================================================

def collapse_history(history_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge val/test rows that share the same (tag, epoch, global_step).
    """
    grouped: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    for r in history_rows:
        tag = str(r.get("tag"))
        epoch = as_int(r.get("epoch")) or 0
        step = as_int(r.get("global_step")) or 0
        split = str(r.get("split", ""))

        key = (tag, epoch, step)
        if key not in grouped:
            grouped[key] = {
                "tag": tag,
                "epoch": epoch,
                "global_step": step,
            }

        out = grouped[key]
        for k, v in r.items():
            if k in {"tag", "epoch", "global_step", "split"}:
                continue
            if split:
                out[f"{split}_{k}"] = v
            else:
                out[k] = v

    rows = list(grouped.values())
    rows.sort(key=lambda x: (as_int(x.get("global_step")) or 0, event_order(str(x.get("tag")))))
    return rows


def best_row_by_metric(rows: List[Dict[str, Any]], metric_key: str) -> Optional[Dict[str, Any]]:
    cand = [r for r in rows if as_float(r.get(metric_key)) is not None]
    if not cand:
        return None
    return min(cand, key=lambda r: float(r[metric_key]))


def first_step_reaching_progress(
    rows: List[Dict[str, Any]],
    metric_key: str,
    init_value: Optional[float],
    best_value: Optional[float],
    frac: float,
) -> Optional[int]:
    """
    Lower is better.
    Find the first global_step where the run has achieved at least frac of total improvement:
      progress = (init - current) / (init - best)
    """
    init_value = as_float(init_value)
    best_value = as_float(best_value)
    if init_value is None or best_value is None:
        return None

    total_gain = init_value - best_value
    if total_gain <= 0:
        return None

    threshold = init_value - frac * total_gain
    for r in rows:
        cur = as_float(r.get(metric_key))
        if cur is None:
            continue
        if cur <= threshold:
            return as_int(r.get("global_step")) or 0
    return None


# ============================================================
# Run-level extraction
# ============================================================

def find_init_compare_run_dirs(exp3_root: Path, prefix: str) -> List[Path]:
    run_dirs: List[Path] = []
    if not exp3_root.exists():
        return run_dirs

    for exp_dir in exp3_root.iterdir():
        if not exp_dir.is_dir():
            continue
        if not exp_dir.name.startswith(prefix):
            continue

        runs_dir = exp_dir / "runs"
        if not runs_dir.exists():
            continue

        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                run_dirs.append(run_dir)

    run_dirs.sort()
    return run_dirs


def extract_run_summary(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics = safe_read_json(run_dir / "metrics.json") or {}
    cfg = safe_read_json(run_dir / "config_used.json") or {}
    meta = safe_read_json(run_dir / "meta.json") or {}
    init_metrics_file = safe_read_json(run_dir / "init_metrics.json") or {}
    history_raw = safe_read_jsonl(run_dir / "eval_history.jsonl")

    if not metrics and not history_raw:
        return None, [], []

    exp_dir = run_dir.parent.parent
    exp_name = exp_dir.name
    run_name = run_dir.name

    rank, alpha, alpha_over_r = parse_rank_alpha_from_cfg(cfg)
    init_mode = parse_init_mode(cfg, meta)
    target_modules = parse_target_modules(cfg, meta)

    quantized_model_dir = cfg.get("quantized_model_dir") or meta.get("quantized_model_dir")
    optimized_model_dir = cfg.get("optimized_model_dir") or meta.get("optimized_model_dir")
    model_name = cfg.get("model_name") or meta.get("model_name")
    tokenizer_name = cfg.get("tokenizer_name") or meta.get("tokenizer_name")

    bit = infer_bit_from_text(exp_name, run_name, quantized_model_dir, model_name)
    seed = cfg.get("seed", meta.get("seed"))
    max_train_steps = parse_train_field(cfg, meta, "max_train_steps")
    eval_every_steps = parse_train_field(cfg, meta, "eval_every_steps")
    lr = parse_train_field(cfg, meta, "lr")
    grad_accum_steps = parse_train_field(cfg, meta, "grad_accum_steps")

    lora_init_cfg = cfg.get("lora_init", {}) if isinstance(cfg.get("lora_init", {}), dict) else {}
    adapter_dir = lora_init_cfg.get("adapter_dir", None)

    init_part = metrics.get("init") or init_metrics_file or {}
    init_val = init_part.get("val", {}) or {}
    init_test = init_part.get("test", {}) or {}

    final_val = metrics.get("val", {}) or {}
    final_test = metrics.get("test", {}) or {}

    history_wide = collapse_history(history_raw)

    # decorate history with run metadata
    history_rows_out: List[Dict[str, Any]] = []
    for r in history_wide:
        out = {
            "exp_name": exp_name,
            "run_name": run_name,
            "init_mode": init_mode,
            "bit": bit,
            "rank": rank,
            "alpha": alpha,
            "alpha_over_r": alpha_over_r,
            "target_modules": target_modules,
            "global_step": r.get("global_step"),
            "epoch": r.get("epoch"),
            "tag": r.get("tag"),
            "run_dir": safe_relpath(run_dir, PROJECT_ROOT),
        }
        out.update(r)
        history_rows_out.append(out)

    best_hist_val_loss_row = best_row_by_metric(history_wide, "val_loss")
    best_hist_val_kl_row = best_row_by_metric(history_wide, "val_kl_to_teacher")

    init_val_loss = as_float(init_val.get("loss"))
    init_test_loss = as_float(init_test.get("loss"))
    init_val_kl = as_float(init_val.get("kl_to_teacher"))
    init_test_kl = as_float(init_test.get("kl_to_teacher"))

    # final metrics in metrics.json are already after best-state restore
    final_val_loss = as_float(final_val.get("loss"))
    final_test_loss = as_float(final_test.get("loss"))
    final_val_kl = as_float(final_val.get("kl_to_teacher"))
    final_test_kl = as_float(final_test.get("kl_to_teacher"))

    row = {
        "exp_name": exp_name,
        "run_name": run_name,
        "init_mode": init_mode,
        "bit": bit,
        "seed": seed,
        "rank": rank,
        "alpha": alpha,
        "alpha_over_r": alpha_over_r,
        "target_modules": target_modules,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "quantized_model_dir": quantized_model_dir,
        "optimized_model_dir": optimized_model_dir,
        "adapter_dir": adapter_dir,
        "max_train_steps": max_train_steps,
        "eval_every_steps": eval_every_steps,
        "lr": lr,
        "grad_accum_steps": grad_accum_steps,
        "run_dir": safe_relpath(run_dir, PROJECT_ROOT),
        "history_path": safe_relpath(run_dir / "eval_history.jsonl", PROJECT_ROOT),
        "metrics_path": safe_relpath(run_dir / "metrics.json", PROJECT_ROOT),
        "config_path": safe_relpath(run_dir / "config_used.json", PROJECT_ROOT),

        "init_val_loss": init_val_loss,
        "init_val_ppl": as_float(init_val.get("ppl")),
        "init_val_kl": init_val_kl,
        "init_val_mse": as_float(init_val.get("mse_logits_to_teacher")),
        "init_test_loss": init_test_loss,
        "init_test_ppl": as_float(init_test.get("ppl")),
        "init_test_kl": init_test_kl,
        "init_test_mse": as_float(init_test.get("mse_logits_to_teacher")),

        "final_val_loss": final_val_loss,
        "final_val_ppl": as_float(final_val.get("ppl")),
        "final_val_kl": final_val_kl,
        "final_val_mse": as_float(final_val.get("mse_logits_to_teacher")),
        "final_test_loss": final_test_loss,
        "final_test_ppl": as_float(final_test.get("ppl")),
        "final_test_kl": final_test_kl,
        "final_test_mse": as_float(final_test.get("mse_logits_to_teacher")),

        "gain_val_loss_init_to_final": (init_val_loss - final_val_loss) if (init_val_loss is not None and final_val_loss is not None) else None,
        "gain_test_loss_init_to_final": (init_test_loss - final_test_loss) if (init_test_loss is not None and final_test_loss is not None) else None,
        "gain_val_kl_init_to_final": (init_val_kl - final_val_kl) if (init_val_kl is not None and final_val_kl is not None) else None,
        "gain_test_kl_init_to_final": (init_test_kl - final_test_kl) if (init_test_kl is not None and final_test_kl is not None) else None,

        "best_hist_val_loss": as_float(best_hist_val_loss_row.get("val_loss")) if best_hist_val_loss_row else None,
        "best_hist_val_loss_step": as_int(best_hist_val_loss_row.get("global_step")) if best_hist_val_loss_row else None,
        "best_hist_val_loss_test_loss": as_float(best_hist_val_loss_row.get("test_loss")) if best_hist_val_loss_row else None,
        "best_hist_val_loss_val_kl": as_float(best_hist_val_loss_row.get("val_kl_to_teacher")) if best_hist_val_loss_row else None,
        "best_hist_val_loss_test_kl": as_float(best_hist_val_loss_row.get("test_kl_to_teacher")) if best_hist_val_loss_row else None,

        "best_hist_val_kl": as_float(best_hist_val_kl_row.get("val_kl_to_teacher")) if best_hist_val_kl_row else None,
        "best_hist_val_kl_step": as_int(best_hist_val_kl_row.get("global_step")) if best_hist_val_kl_row else None,
        "best_hist_val_kl_test_loss": as_float(best_hist_val_kl_row.get("test_loss")) if best_hist_val_kl_row else None,
        "best_hist_val_kl_test_kl": as_float(best_hist_val_kl_row.get("test_kl_to_teacher")) if best_hist_val_kl_row else None,

        "step_val_loss_50pct_gain": first_step_reaching_progress(history_wide, "val_loss", init_val_loss, final_val_loss, 0.50),
        "step_val_loss_90pct_gain": first_step_reaching_progress(history_wide, "val_loss", init_val_loss, final_val_loss, 0.90),
        "step_val_kl_50pct_gain": first_step_reaching_progress(history_wide, "val_kl_to_teacher", init_val_kl, final_val_kl, 0.50),
        "step_val_kl_90pct_gain": first_step_reaching_progress(history_wide, "val_kl_to_teacher", init_val_kl, final_val_kl, 0.90),

        "num_history_events": len(history_wide),
    }

    return row, history_rows_out, history_wide


# ============================================================
# Pairwise comparison
# ============================================================

def pair_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("bit"),
        row.get("seed"),
        row.get("rank"),
        row.get("alpha_over_r"),
        row.get("target_modules"),
        row.get("max_train_steps"),
    )


def mean_or_none(xs: List[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_pairwise_outputs(
    run_summaries: List[Dict[str, Any]],
    history_by_run: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for r in run_summaries:
        grouped[pair_key(r)].append(r)

    pairwise_summary_rows: List[Dict[str, Any]] = []
    pairwise_step_rows: List[Dict[str, Any]] = []

    for k, rs in grouped.items():
        random_rows = [r for r in rs if str(r.get("init_mode")).lower() == "random"]
        eora_rows = [r for r in rs if str(r.get("init_mode")).lower() == "eora_adapter"]

        if len(random_rows) != 1 or len(eora_rows) != 1:
            continue

        rrand = random_rows[0]
        reora = eora_rows[0]

        rand_hist = history_by_run.get(rrand["run_dir"], [])
        eora_hist = history_by_run.get(reora["run_dir"], [])

        rand_by_step = {as_int(x.get("global_step")): x for x in rand_hist if as_int(x.get("global_step")) is not None}
        eora_by_step = {as_int(x.get("global_step")): x for x in eora_hist if as_int(x.get("global_step")) is not None}

        matched_steps = sorted(set(rand_by_step.keys()) & set(eora_by_step.keys()))

        step_deltas_val_loss = []
        step_deltas_test_loss = []
        step_deltas_val_kl = []
        step_deltas_test_kl = []

        better_val_loss_count = 0
        better_val_kl_count = 0

        for step in matched_steps:
            rr = rand_by_step[step]
            ee = eora_by_step[step]

            row = {
                "bit": rrand.get("bit"),
                "seed": rrand.get("seed"),
                "rank": rrand.get("rank"),
                "alpha_over_r": rrand.get("alpha_over_r"),
                "target_modules": rrand.get("target_modules"),
                "max_train_steps": rrand.get("max_train_steps"),
                "global_step": step,

                "random_tag": rr.get("tag"),
                "random_val_loss": rr.get("val_loss"),
                "random_test_loss": rr.get("test_loss"),
                "random_val_kl": rr.get("val_kl_to_teacher"),
                "random_test_kl": rr.get("test_kl_to_teacher"),

                "eorainit_tag": ee.get("tag"),
                "eorainit_val_loss": ee.get("val_loss"),
                "eorainit_test_loss": ee.get("test_loss"),
                "eorainit_val_kl": ee.get("val_kl_to_teacher"),
                "eorainit_test_kl": ee.get("test_kl_to_teacher"),
            }

            d_val_loss = None
            d_test_loss = None
            d_val_kl = None
            d_test_kl = None

            if as_float(ee.get("val_loss")) is not None and as_float(rr.get("val_loss")) is not None:
                d_val_loss = float(ee["val_loss"]) - float(rr["val_loss"])
                step_deltas_val_loss.append(d_val_loss)
                if d_val_loss < 0:
                    better_val_loss_count += 1

            if as_float(ee.get("test_loss")) is not None and as_float(rr.get("test_loss")) is not None:
                d_test_loss = float(ee["test_loss"]) - float(rr["test_loss"])
                step_deltas_test_loss.append(d_test_loss)

            if as_float(ee.get("val_kl_to_teacher")) is not None and as_float(rr.get("val_kl_to_teacher")) is not None:
                d_val_kl = float(ee["val_kl_to_teacher"]) - float(rr["val_kl_to_teacher"])
                step_deltas_val_kl.append(d_val_kl)
                if d_val_kl < 0:
                    better_val_kl_count += 1

            if as_float(ee.get("test_kl_to_teacher")) is not None and as_float(rr.get("test_kl_to_teacher")) is not None:
                d_test_kl = float(ee["test_kl_to_teacher"]) - float(rr["test_kl_to_teacher"])
                step_deltas_test_kl.append(d_test_kl)

            row["delta_eorainit_minus_random_val_loss"] = d_val_loss
            row["delta_eorainit_minus_random_test_loss"] = d_test_loss
            row["delta_eorainit_minus_random_val_kl"] = d_val_kl
            row["delta_eorainit_minus_random_test_kl"] = d_test_kl
            pairwise_step_rows.append(row)

        pair_row = {
            "bit": rrand.get("bit"),
            "seed": rrand.get("seed"),
            "rank": rrand.get("rank"),
            "alpha_over_r": rrand.get("alpha_over_r"),
            "target_modules": rrand.get("target_modules"),
            "max_train_steps": rrand.get("max_train_steps"),

            "random_run_dir": rrand.get("run_dir"),
            "eorainit_run_dir": reora.get("run_dir"),

            "random_init_val_loss": rrand.get("init_val_loss"),
            "eorainit_init_val_loss": reora.get("init_val_loss"),
            "delta_init_val_loss": (
                as_float(reora.get("init_val_loss")) - as_float(rrand.get("init_val_loss"))
                if as_float(reora.get("init_val_loss")) is not None and as_float(rrand.get("init_val_loss")) is not None
                else None
            ),

            "random_init_test_loss": rrand.get("init_test_loss"),
            "eorainit_init_test_loss": reora.get("init_test_loss"),
            "delta_init_test_loss": (
                as_float(reora.get("init_test_loss")) - as_float(rrand.get("init_test_loss"))
                if as_float(reora.get("init_test_loss")) is not None and as_float(rrand.get("init_test_loss")) is not None
                else None
            ),

            "random_init_val_kl": rrand.get("init_val_kl"),
            "eorainit_init_val_kl": reora.get("init_val_kl"),
            "delta_init_val_kl": (
                as_float(reora.get("init_val_kl")) - as_float(rrand.get("init_val_kl"))
                if as_float(reora.get("init_val_kl")) is not None and as_float(rrand.get("init_val_kl")) is not None
                else None
            ),

            "random_init_test_kl": rrand.get("init_test_kl"),
            "eorainit_init_test_kl": reora.get("init_test_kl"),
            "delta_init_test_kl": (
                as_float(reora.get("init_test_kl")) - as_float(rrand.get("init_test_kl"))
                if as_float(reora.get("init_test_kl")) is not None and as_float(rrand.get("init_test_kl")) is not None
                else None
            ),

            "random_final_val_loss": rrand.get("final_val_loss"),
            "eorainit_final_val_loss": reora.get("final_val_loss"),
            "delta_final_val_loss": (
                as_float(reora.get("final_val_loss")) - as_float(rrand.get("final_val_loss"))
                if as_float(reora.get("final_val_loss")) is not None and as_float(rrand.get("final_val_loss")) is not None
                else None
            ),

            "random_final_test_loss": rrand.get("final_test_loss"),
            "eorainit_final_test_loss": reora.get("final_test_loss"),
            "delta_final_test_loss": (
                as_float(reora.get("final_test_loss")) - as_float(rrand.get("final_test_loss"))
                if as_float(reora.get("final_test_loss")) is not None and as_float(rrand.get("final_test_loss")) is not None
                else None
            ),

            "random_final_val_kl": rrand.get("final_val_kl"),
            "eorainit_final_val_kl": reora.get("final_val_kl"),
            "delta_final_val_kl": (
                as_float(reora.get("final_val_kl")) - as_float(rrand.get("final_val_kl"))
                if as_float(reora.get("final_val_kl")) is not None and as_float(rrand.get("final_val_kl")) is not None
                else None
            ),

            "random_final_test_kl": rrand.get("final_test_kl"),
            "eorainit_final_test_kl": reora.get("final_test_kl"),
            "delta_final_test_kl": (
                as_float(reora.get("final_test_kl")) - as_float(rrand.get("final_test_kl"))
                if as_float(reora.get("final_test_kl")) is not None and as_float(rrand.get("final_test_kl")) is not None
                else None
            ),

            "random_step_val_loss_50pct_gain": rrand.get("step_val_loss_50pct_gain"),
            "eorainit_step_val_loss_50pct_gain": reora.get("step_val_loss_50pct_gain"),
            "random_step_val_loss_90pct_gain": rrand.get("step_val_loss_90pct_gain"),
            "eorainit_step_val_loss_90pct_gain": reora.get("step_val_loss_90pct_gain"),

            "random_step_val_kl_50pct_gain": rrand.get("step_val_kl_50pct_gain"),
            "eorainit_step_val_kl_50pct_gain": reora.get("step_val_kl_50pct_gain"),
            "random_step_val_kl_90pct_gain": rrand.get("step_val_kl_90pct_gain"),
            "eorainit_step_val_kl_90pct_gain": reora.get("step_val_kl_90pct_gain"),

            "matched_eval_points": len(matched_steps),
            "mean_delta_val_loss": mean_or_none(step_deltas_val_loss),
            "mean_delta_test_loss": mean_or_none(step_deltas_test_loss),
            "mean_delta_val_kl": mean_or_none(step_deltas_val_kl),
            "mean_delta_test_kl": mean_or_none(step_deltas_test_kl),
            "eorainit_better_fraction_val_loss": (
                better_val_loss_count / len(matched_steps) if matched_steps else None
            ),
            "eorainit_better_fraction_val_kl": (
                better_val_kl_count / len(matched_steps) if matched_steps else None
            ),
        }

        pairwise_summary_rows.append(pair_row)

    return pairwise_summary_rows, pairwise_step_rows


# ============================================================
# Console summary
# ============================================================

def print_console_summary(
    run_summaries: List[Dict[str, Any]],
    pairwise_summaries: List[Dict[str, Any]],
    out_dir: Path,
):
    print("\n================ LORA INIT COMPARE SUMMARY ================\n")
    print(f"Runs found: {len(run_summaries)}")
    print(f"Output dir: {out_dir}\n")

    print("--- Run summary ---")
    for r in sorted(run_summaries, key=lambda x: (x.get("bit"), x.get("rank"), x.get("init_mode") or "")):
        print(
            f"bit={r.get('bit')}  r={r.get('rank')}  init={r.get('init_mode')}  "
            f"init_val_loss={fmt(r.get('init_val_loss'))}  "
            f"final_val_loss={fmt(r.get('final_val_loss'))}  "
            f"init_val_kl={fmt(r.get('init_val_kl'))}  "
            f"final_val_kl={fmt(r.get('final_val_kl'))}  "
            f"step50(loss)={r.get('step_val_loss_50pct_gain')}  "
            f"step90(loss)={r.get('step_val_loss_90pct_gain')}"
        )

    print("\n--- Pairwise summary (EoRAInit - RandomInit) ---")
    for r in sorted(pairwise_summaries, key=lambda x: (x.get("bit"), x.get("rank"))):
        print(
            f"bit={r.get('bit')}  r={r.get('rank')}  "
            f"Δinit_val_loss={fmt(r.get('delta_init_val_loss'))}  "
            f"Δfinal_val_loss={fmt(r.get('delta_final_val_loss'))}  "
            f"Δinit_val_kl={fmt(r.get('delta_init_val_kl'))}  "
            f"Δfinal_val_kl={fmt(r.get('delta_final_val_kl'))}  "
            f"mean_Δval_loss={fmt(r.get('mean_delta_val_loss'))}  "
            f"mean_Δval_kl={fmt(r.get('mean_delta_val_kl'))}  "
            f"better_frac_loss={fmt(r.get('eorainit_better_fraction_val_loss'))}  "
            f"better_frac_kl={fmt(r.get('eorainit_better_fraction_val_kl'))}"
        )

    print("\n==========================================================\n")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP3_ROOT))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    prefix = str(args.prefix)

    ensure_dir(out_dir)

    run_dirs = find_init_compare_run_dirs(root, prefix)

    run_summaries: List[Dict[str, Any]] = []
    history_rows_all: List[Dict[str, Any]] = []
    history_by_run: Dict[str, List[Dict[str, Any]]] = {}

    for run_dir in run_dirs:
        run_summary, history_rows, history_wide = extract_run_summary(run_dir)
        if run_summary is None:
            continue

        run_summaries.append(run_summary)
        history_rows_all.extend(history_rows)
        history_by_run[run_summary["run_dir"]] = history_wide

    pairwise_summaries, pairwise_step_rows = build_pairwise_outputs(run_summaries, history_by_run)

    run_summary_csv = out_dir / "run_summary.csv"
    history_csv = out_dir / "history_wide.csv"
    pairwise_summary_csv = out_dir / "pairwise_summary.csv"
    pairwise_step_csv = out_dir / "pairwise_step_compare.csv"

    write_csv(run_summaries, run_summary_csv)
    write_csv(history_rows_all, history_csv)
    write_csv(pairwise_summaries, pairwise_summary_csv)
    write_csv(pairwise_step_rows, pairwise_step_csv)

    print(f"Saved: {run_summary_csv}")
    print(f"Saved: {history_csv}")
    print(f"Saved: {pairwise_summary_csv}")
    print(f"Saved: {pairwise_step_csv}")

    print_console_summary(run_summaries, pairwise_summaries, out_dir)


if __name__ == "__main__":
    main()