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


def load_teacher_test_acc(path: Path) -> Optional[float]:
    js = safe_read_json(path)
    if not js:
        return None
    test = js.get("test", {}) or js.get("test.py", {}) or {}
    acc = test.get("accuracy")
    return float(acc) if acc is not None else None


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

        seed = parse_seed(exp_name)
        if seed is None:
            seed = parse_seed(run_name)

        m = extract_metrics(metrics_json)
        gap = None
        if teacher_test_acc is not None and m["test_acc"] is not None:
            gap = float(m["test_acc"]) - float(teacher_test_acc)

        row = {
            "branch": branch,
            "exp_name": exp_name,
            "run_name": run_name,
            "seed": seed,
            "rank": r,
            "alpha_over_r": ar,
            "val_acc": m["val_acc"],
            "val_loss": m["val_loss"],
            "test_acc": m["test_acc"],
            "test_loss": m["test_loss"],
            "test_gap_to_teacher": gap,
            "run_dir": str(run_dir.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/"),
        }
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "branch", "exp_name", "run_name", "seed", "rank", "alpha_over_r",
        "val_acc", "val_loss", "test_acc", "test_loss", "test_gap_to_teacher",
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
    lora = [r for r in rows if r["branch"] == "LoRA"]
    eora = [r for r in rows if r["branch"] == "EoRA"]

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
            f"[BEST LoRA]  {bL['exp_name']}  seed={bL['seed']}  rank={bL['rank']}  ar={bL['alpha_over_r']}"
            f"  test_acc={fmt(bL['test_acc'])}  gap={fmt(bL['test_gap_to_teacher'])}"
        )
    else:
        print("[BEST LoRA]  NA")

    if bE:
        print(
            f"[BEST EoRA]  {bE['exp_name']}  seed={bE['seed']}  rank={bE['rank']}  ar={bE['alpha_over_r']}"
            f"  test_acc={fmt(bE['test_acc'])}  gap={fmt(bE['test_gap_to_teacher'])}"
        )
    else:
        print("[BEST EoRA]  NA")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["branch"], r["rank"], r["alpha_over_r"])].append(r)

    print("\n--- Aggregated by (branch, rank, alpha/r) ---")
    keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1] if x[1] is not None else 10**9, x[2] if x[2] is not None else 10**9))
    for k in keys:
        rs = grouped[k]
        mu_acc, sd_acc = mean_std([r["test_acc"] for r in rs])
        mu_gap, sd_gap = mean_std([r["test_gap_to_teacher"] for r in rs])
        print(
            f"{k[0]:4s}  r={k[1]}  ar={k[2]}  "
            f"n={len(rs)}  test_acc={fmt(mu_acc)}±{fmt(sd_acc)}  "
            f"gap={fmt(mu_gap)}±{fmt(sd_gap)}"
        )

    print("\n=============================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP1_ROOT))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    ap.add_argument("--teacher_metrics", type=str, default=str(DEFAULT_TEACHER_METRICS))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    teacher_metrics = Path(args.teacher_metrics).resolve()

    teacher_test_acc = load_teacher_test_acc(teacher_metrics)

    rows = scan_exp1(root, teacher_test_acc)

    def sort_key(r):
        rk = r["rank"] if r["rank"] is not None else 10**9
        ar = r["alpha_over_r"] if r["alpha_over_r"] is not None else 10**9
        sd = r["seed"] if r["seed"] is not None else -1
        return (r["branch"], rk, ar, sd, r["exp_name"], r["run_name"])

    rows = sorted(rows, key=sort_key)
    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")
    print_console_summary(rows, teacher_test_acc, root)


if __name__ == "__main__":
    main()