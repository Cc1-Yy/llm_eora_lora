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

DEFAULT_EXP2_ROOT = PROJECT_ROOT / "outputs" / "cls" / "exp2"
DEFAULT_CONFIRM_ROOT = PROJECT_ROOT / "outputs" / "cls" / "confirm_multiseed" / "exp2"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "cls" / "exp2_summary_all.csv"
DEFAULT_TEACHER_METRICS = PROJECT_ROOT / "outputs" / "cls" / "exp0" / "optimized_sst2" / "metrics.json"


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_branch(exp_name: str) -> Optional[str]:
    name = exp_name.lower()
    if "lora" in name:
        return "LoRA"
    if "eora" in name:
        return "EoRA"
    return None


def parse_r_ar(name: str) -> Tuple[Optional[int], Optional[float]]:
    m = re.search(r"_r(\d+)_ar(\d+(\.\d+)?)", name)
    if not m:
        m = re.search(r"r(\d+)_ar(\d+(\.\d+)?)", name)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def parse_seed(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def load_teacher_metrics(path: Path) -> Dict[str, Any]:
    js = safe_read_json(path)
    if not js:
        return {}

    test = js.get("test", {}) or js.get("test.py", {}) or {}
    val = js.get("val", {}) or {}

    return {
        "teacher_val_loss": val.get("loss"),
        "teacher_val_acc": val.get("accuracy"),
        "teacher_test_loss": test.get("loss"),
        "teacher_test_acc": test.get("accuracy"),
        "teacher_seed": js.get("seed"),
        "teacher_model_name": js.get("model_name"),
        "teacher_task_type": js.get("task_type"),
    }


def maybe_extract_alpha(
    metrics_json: Dict[str, Any],
    rank: Optional[int],
    alpha_over_r: Optional[float],
    branch: Optional[str],
) -> Optional[float]:
    cfg_key = None
    if branch == "LoRA":
        cfg_key = "lora"
    elif branch == "EoRA":
        cfg_key = "eora"

    if cfg_key is not None:
        cfg = metrics_json.get(cfg_key, {}) or {}
        alpha = cfg.get("alpha")
        if alpha is not None:
            try:
                return float(alpha)
            except Exception:
                pass

    if rank is not None and alpha_over_r is not None:
        try:
            return float(rank) * float(alpha_over_r)
        except Exception:
            return None

    return None


def extract_metrics(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", {}) or metrics_json.get("test.py", {}) or {}
    return {
        "val_acc": val.get("accuracy"),
        "val_ce_loss": val.get("ce_loss"),
        "val_kl_to_teacher": val.get("kl_to_teacher"),
        "val_mse_logits_to_teacher": val.get("mse_logits_to_teacher"),
        "test_acc": test.get("accuracy"),
        "test_ce_loss": test.get("ce_loss"),
        "test_kl_to_teacher": test.get("kl_to_teacher"),
        "test_mse_logits_to_teacher": test.get("mse_logits_to_teacher"),
    }


def scan_exp2(
    root: Path,
    source_name: str,
    teacher_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows

    teacher_val_acc = teacher_metrics.get("teacher_val_acc")
    teacher_val_loss = teacher_metrics.get("teacher_val_loss")
    teacher_test_acc = teacher_metrics.get("teacher_test_acc")
    teacher_test_loss = teacher_metrics.get("teacher_test_loss")

    for metrics_path in root.rglob("metrics.json"):
        metrics_json = safe_read_json(metrics_path)
        if not metrics_json:
            continue

        rel = metrics_path.relative_to(root)
        exp_name = rel.parts[0]
        run_dir = metrics_path.parent
        run_name = run_dir.name

        branch = infer_branch(exp_name)
        if branch is None:
            continue

        r, ar = parse_r_ar(exp_name)
        if r is None or ar is None:
            r, ar = parse_r_ar(run_name)

        seed = metrics_json.get("seed")
        if seed is None:
            seed = parse_seed(exp_name)
        if seed is None:
            seed = parse_seed(run_name)
        if seed is None:
            meta_json = safe_read_json(run_dir / "meta.json")
            if meta_json:
                seed = meta_json.get("seed")

        m = extract_metrics(metrics_json)
        alpha = maybe_extract_alpha(metrics_json, r, ar, branch)

        teacher_minus_test = None
        test_minus_teacher = None
        if teacher_test_acc is not None and m["test_acc"] is not None:
            teacher_minus_test = float(teacher_test_acc) - float(m["test_acc"])
            test_minus_teacher = float(m["test_acc"]) - float(teacher_test_acc)

        row = {
            "source": source_name,
            "branch": branch,
            "exp_name": exp_name,
            "run_name": run_name,
            "seed": seed,
            "rank": r,
            "alpha": alpha,
            "alpha_over_r": ar,

            "val_acc": m["val_acc"],
            "val_ce_loss": m["val_ce_loss"],
            "val_kl_to_teacher": m["val_kl_to_teacher"],
            "val_mse_logits_to_teacher": m["val_mse_logits_to_teacher"],

            "test_acc": m["test_acc"],
            "test_ce_loss": m["test_ce_loss"],
            "test_kl_to_teacher": m["test_kl_to_teacher"],
            "test_mse_logits_to_teacher": m["test_mse_logits_to_teacher"],

            "teacher_val_acc": teacher_val_acc,
            "teacher_val_loss": teacher_val_loss,
            "teacher_test_acc": teacher_test_acc,
            "teacher_test_loss": teacher_test_loss,

            "teacher_minus_test": teacher_minus_test,
            "test_minus_teacher": test_minus_teacher,

            "task_type": metrics_json.get("task_type"),
            "model_name": metrics_json.get("model_name"),
            "optimized_model_dir": metrics_json.get("optimized_model_dir") or metrics_json.get("teacher_model_dir"),
            "run_dir": safe_relpath(run_dir, PROJECT_ROOT),
        }
        rows.append(row)

    return rows


def fmt(x, nd=4):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


def sort_key(r: Dict[str, Any]):
    source_order = {"main_exp2": 0, "confirm_multiseed_exp2": 1}
    branch_order = {"LoRA": 0, "EoRA": 1}
    return (
        source_order.get(r["source"], 99),
        branch_order.get(r["branch"], 99),
        r["rank"] if r["rank"] is not None else 10**9,
        r["alpha_over_r"] if r["alpha_over_r"] is not None else 10**9,
        r["seed"] if r["seed"] is not None else -1,
        r["exp_name"],
        r["run_name"],
    )


def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "branch",
        "exp_name",
        "run_name",
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

        "task_type",
        "model_name",
        "optimized_model_dir",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_all_rows(rows: List[Dict[str, Any]]):
    print("\n================ ALL EXP2 ROWS ================\n")
    for i, r in enumerate(rows, 1):
        print(
            f"[{i:02d}] "
            f"source={r['source']} | "
            f"branch={r['branch']} | "
            f"exp_name={r['exp_name']} | "
            f"seed={r['seed']} | "
            f"rank={r['rank']} | "
            f"alpha={fmt(r['alpha'])} | "
            f"ar={r['alpha_over_r']} | "
            f"val_acc={fmt(r['val_acc'])} | "
            f"val_ce={fmt(r['val_ce_loss'])} | "
            f"val_kl={fmt(r['val_kl_to_teacher'])} | "
            f"val_mse={fmt(r['val_mse_logits_to_teacher'])} | "
            f"test_acc={fmt(r['test_acc'])} | "
            f"test_ce={fmt(r['test_ce_loss'])} | "
            f"test_kl={fmt(r['test_kl_to_teacher'])} | "
            f"test_mse={fmt(r['test_mse_logits_to_teacher'])} | "
            f"teacher_val_acc={fmt(r['teacher_val_acc'])} | "
            f"teacher_test_acc={fmt(r['teacher_test_acc'])} | "
            f"teacher_minus_test={fmt(r['teacher_minus_test'])} | "
            f"run_dir={r['run_dir']}"
        )
    print("\n==============================================\n")


def print_grouped_summary(rows: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["source"], r["branch"], r["rank"], r["alpha_over_r"])].append(r)

    print("\n============= GROUPED EXP2 SUMMARY =============\n")
    keys = sorted(
        grouped.keys(),
        key=lambda x: (
            x[0],
            x[1],
            x[2] if x[2] is not None else 10**9,
            x[3] if x[3] is not None else 10**9,
        ),
    )

    for k in keys:
        rs = grouped[k]
        mu_acc, sd_acc = mean_std([r["test_acc"] for r in rs])
        mu_kl, sd_kl = mean_std([r["test_kl_to_teacher"] for r in rs])
        mu_mse, sd_mse = mean_std([r["test_mse_logits_to_teacher"] for r in rs])
        mu_gap, sd_gap = mean_std([r["teacher_minus_test"] for r in rs])

        print(
            f"source={k[0]} | branch={k[1]} | r={k[2]} | ar={k[3]} | n={len(rs)} | "
            f"test_acc={fmt(mu_acc)}±{fmt(sd_acc)} | "
            f"test_kl={fmt(mu_kl)}±{fmt(sd_kl)} | "
            f"test_mse={fmt(mu_mse)}±{fmt(sd_mse)} | "
            f"teacher_minus_test={fmt(mu_gap)}±{fmt(sd_gap)}"
        )
    print("\n===============================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP2_ROOT))
    ap.add_argument("--confirm_root", type=str, default=str(DEFAULT_CONFIRM_ROOT))
    ap.add_argument("--include_confirm", action="store_true")
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    ap.add_argument("--teacher_metrics", type=str, default=str(DEFAULT_TEACHER_METRICS))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    confirm_root = Path(args.confirm_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    teacher_metrics_path = Path(args.teacher_metrics).resolve()

    teacher_metrics = load_teacher_metrics(teacher_metrics_path)

    rows = []
    rows.extend(scan_exp2(root, "main_exp2", teacher_metrics))

    if args.include_confirm:
        rows.extend(scan_exp2(confirm_root, "confirm_multiseed_exp2", teacher_metrics))

    rows = sorted(rows, key=sort_key)
    write_csv(rows, out_csv)

    print(f"Saved CSV: {out_csv}")
    print(f"Teacher val_acc:  {fmt(teacher_metrics.get('teacher_val_acc'))}")
    print(f"Teacher val_loss: {fmt(teacher_metrics.get('teacher_val_loss'))}")
    print(f"Teacher test_acc: {fmt(teacher_metrics.get('teacher_test_acc'))}")
    print(f"Teacher test_loss:{fmt(teacher_metrics.get('teacher_test_loss'))}")

    print_all_rows(rows)
    print_grouped_summary(rows)


if __name__ == "__main__":
    main()