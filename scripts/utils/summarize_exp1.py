import json
import re
import csv
import math
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP1_ROOT = PROJECT_ROOT / "outputs" / "cls" / "exp1"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "cls" / "exp1_summary.csv"
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


def get_exp_name_from_metrics_path(metrics_path: Path, exp_root: Path) -> str:
    rel = metrics_path.relative_to(exp_root)
    return rel.parts[0]


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


def extract_metrics(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", {}) or metrics_json.get("test.py", {}) or {}
    return {
        "val_loss": val.get("loss"),
        "val_acc": val.get("accuracy"),
        "test_loss": test.get("loss"),
        "test_acc": test.get("accuracy"),
    }


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


def maybe_extract_alpha(metrics_json: Dict[str, Any], rank: Optional[int], alpha_over_r: Optional[float], branch: Optional[str]) -> Optional[float]:
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


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def scan_exp1(root: Path, teacher_test_acc: Optional[float]) -> List[Dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows

    for metrics_path in root.rglob("metrics.json"):
        metrics_json = safe_read_json(metrics_path)
        if not metrics_json:
            continue

        exp_name = get_exp_name_from_metrics_path(metrics_path, root)
        branch = infer_branch(exp_name)
        if branch is None:
            continue

        run_dir = metrics_path.parent
        run_name = run_dir.name

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

        test_minus_teacher = None
        teacher_minus_test = None
        if teacher_test_acc is not None and m["test_acc"] is not None:
            test_minus_teacher = float(m["test_acc"]) - float(teacher_test_acc)
            teacher_minus_test = float(teacher_test_acc) - float(m["test_acc"])

        row = {
            "branch": branch,
            "exp_name": exp_name,
            "run_name": run_name,
            "seed": seed,
            "rank": r,
            "alpha": alpha,
            "alpha_over_r": ar,
            "val_acc": m["val_acc"],
            "val_loss": m["val_loss"],
            "test_acc": m["test_acc"],
            "test_loss": m["test_loss"],
            "teacher_test_acc": teacher_test_acc,
            "test_minus_teacher": test_minus_teacher,
            "teacher_minus_test": teacher_minus_test,
            "task_type": metrics_json.get("task_type"),
            "num_labels": metrics_json.get("num_labels"),
            "model_name": metrics_json.get("model_name"),
            "optimized_model_dir": metrics_json.get("optimized_model_dir"),
            "run_dir": safe_relpath(run_dir, PROJECT_ROOT),
        }
        rows.append(row)

    return rows


def build_teacher_row(teacher_metrics: Dict[str, Any], teacher_metrics_path: Path) -> Optional[Dict[str, Any]]:
    if not teacher_metrics:
        return None

    if teacher_metrics.get("teacher_test_acc") is None:
        return None

    return {
        "branch": "Teacher",
        "exp_name": "optimized_model",
        "run_name": "optimized_model",
        "seed": teacher_metrics.get("teacher_seed"),
        "rank": None,
        "alpha": None,
        "alpha_over_r": None,
        "val_acc": teacher_metrics.get("teacher_val_acc"),
        "val_loss": teacher_metrics.get("teacher_val_loss"),
        "test_acc": teacher_metrics.get("teacher_test_acc"),
        "test_loss": teacher_metrics.get("teacher_test_loss"),
        "teacher_test_acc": teacher_metrics.get("teacher_test_acc"),
        "test_minus_teacher": 0.0,
        "teacher_minus_test": 0.0,
        "task_type": teacher_metrics.get("teacher_task_type"),
        "num_labels": None,
        "model_name": teacher_metrics.get("teacher_model_name"),
        "optimized_model_dir": None,
        "run_dir": safe_relpath(teacher_metrics_path.parent, PROJECT_ROOT),
    }


def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "branch",
        "exp_name",
        "run_name",
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
        "task_type",
        "num_labels",
        "model_name",
        "optimized_model_dir",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


def print_console_summary(rows: List[Dict[str, Any]], teacher_test_acc: Optional[float], root: Path):
    exp_rows = [r for r in rows if r["branch"] in {"LoRA", "EoRA"}]
    lora = [r for r in exp_rows if r["branch"] == "LoRA"]
    eora = [r for r in exp_rows if r["branch"] == "EoRA"]

    def best_of(branch_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ok = [r for r in branch_rows if r["test_acc"] is not None]
        if not ok:
            return None
        return sorted(ok, key=lambda x: x["test_acc"], reverse=True)[0]

    print("\n================ EXP1 SUMMARY ================\n")
    print(f"Scan dir: {root}")
    print(f"Found runs: LoRA={len(lora)}, EoRA={len(eora)}")
    print(f"Teacher test_acc: {fmt(teacher_test_acc)}\n")

    bL = best_of(lora)
    bE = best_of(eora)

    if bL:
        print(
            f"[BEST LoRA]  {bL['exp_name']}  seed={bL['seed']}  rank={bL['rank']}  "
            f"alpha={fmt(bL['alpha'])}  ar={bL['alpha_over_r']}  "
            f"test_acc={fmt(bL['test_acc'])}  teacher_minus_test={fmt(bL['teacher_minus_test'])}"
        )
    else:
        print("[BEST LoRA]  NA")

    if bE:
        print(
            f"[BEST EoRA]  {bE['exp_name']}  seed={bE['seed']}  rank={bE['rank']}  "
            f"alpha={fmt(bE['alpha'])}  ar={bE['alpha_over_r']}  "
            f"test_acc={fmt(bE['test_acc'])}  teacher_minus_test={fmt(bE['teacher_minus_test'])}"
        )
    else:
        print("[BEST EoRA]  NA")

    grouped = defaultdict(list)
    for r in exp_rows:
        grouped[(r["branch"], r["rank"], r["alpha_over_r"])].append(r)

    print("\n--- Aggregated by (branch, rank, alpha/r) ---")
    keys = sorted(
        grouped.keys(),
        key=lambda x: (
            x[0],
            x[1] if x[1] is not None else 10**9,
            x[2] if x[2] is not None else 10**9,
        ),
    )
    for k in keys:
        rs = grouped[k]
        mu_acc, sd_acc = mean_std([r["test_acc"] for r in rs])
        mu_gap, sd_gap = mean_std([r["teacher_minus_test"] for r in rs])
        print(
            f"{k[0]:4s}  r={k[1]}  ar={k[2]}  "
            f"n={len(rs)}  test_acc={fmt(mu_acc)}±{fmt(sd_acc)}  "
            f"teacher_minus_test={fmt(mu_gap)}±{fmt(sd_gap)}"
        )

    print("\n=============================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP1_ROOT))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    ap.add_argument("--teacher_metrics", type=str, default=str(DEFAULT_TEACHER_METRICS))
    ap.add_argument("--include_teacher_row", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    teacher_metrics_path = Path(args.teacher_metrics).resolve()

    teacher_metrics = load_teacher_metrics(teacher_metrics_path)
    teacher_test_acc = teacher_metrics.get("teacher_test_acc")

    rows = scan_exp1(root, teacher_test_acc)

    if args.include_teacher_row:
        teacher_row = build_teacher_row(teacher_metrics, teacher_metrics_path)
        if teacher_row is not None:
            rows.append(teacher_row)

    def sort_key(r):
        branch_order = {"Teacher": 0, "LoRA": 1, "EoRA": 2}
        rk = r["rank"] if r["rank"] is not None else -1
        ar = r["alpha_over_r"] if r["alpha_over_r"] is not None else -1
        sd = r["seed"] if r["seed"] is not None else -1
        return (
            branch_order.get(r["branch"], 99),
            rk,
            ar,
            sd,
            r["exp_name"],
            r["run_name"],
        )

    rows = sorted(rows, key=sort_key)
    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")
    print_console_summary(rows, teacher_test_acc, root)


if __name__ == "__main__":
    main()