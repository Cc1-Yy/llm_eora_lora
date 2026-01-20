import os
import json
import re
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/utils -> project root
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

LORA_ROOT = OUTPUTS_DIR / "exp1_lora"
EORA_ROOT = OUTPUTS_DIR / "exp1_eora"

SUMMARY_CSV = OUTPUTS_DIR / "exp1_summary.csv"


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_r_ar_from_path(path: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Expect folder name like: r8_ar4
    Return (r, ar)
    """
    name = path.parent.name  # metrics.json -> run_dir
    m = re.search(r"r(\d+)_ar(\d+(\.\d+)?)", name)
    if not m:
        return None, None
    r = int(m.group(1))
    ar = float(m.group(2))
    return r, ar


def extract_metrics(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports your metrics format:
      {"val": {"loss":..., "accuracy":...}, "test": {...}}
    """
    out = {}
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", {}) or {}

    out["val_loss"] = val.get("loss")
    out["val_acc"] = val.get("accuracy")
    out["test_loss"] = test.get("loss")
    out["test_acc"] = test.get("accuracy")
    return out


def scan_branch(branch: str, root: Path) -> List[Dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows

    for metrics_path in root.rglob("metrics.json"):
        metrics_json = safe_read_json(metrics_path)
        if not metrics_json:
            continue

        run_dir = metrics_path.parent
        r, ar = parse_r_ar_from_path(metrics_path)
        m = extract_metrics(metrics_json)

        row = {
            "branch": branch,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "run_name": run_dir.name,
            "rank": r,
            "alpha_over_r": ar,
            "val_acc": m["val_acc"],
            "val_loss": m["val_loss"],
            "test_acc": m["test_acc"],
            "test_loss": m["test_loss"],
        }
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "branch", "run_name", "rank", "alpha_over_r",
        "val_acc", "val_loss", "test_acc", "test_loss",
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


def print_console_summary(rows: List[Dict[str, Any]]):
    # split
    lora = [r for r in rows if r["branch"] == "LoRA"]
    eora = [r for r in rows if r["branch"] == "EoRA"]

    def best_of(branch_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ok = [r for r in branch_rows if r["test_acc"] is not None]
        if not ok:
            return None
        return sorted(ok, key=lambda x: x["test_acc"], reverse=True)[0]

    print("\n================ EXP1 SUMMARY (copy this block to send) ================\n")
    print(f"Found runs: LoRA={len(lora)}, EoRA={len(eora)}")
    print(f"Scan dirs:\n  {LORA_ROOT}\n  {EORA_ROOT}\n")

    bL = best_of(lora)
    bE = best_of(eora)
    if bL:
        print(f"[BEST LoRA]  {bL['run_name']}  rank={bL['rank']}  ar={bL['alpha_over_r']}"
              f"  val_acc={fmt(bL['val_acc'])}  test_acc={fmt(bL['test_acc'])}")
    else:
        print("[BEST LoRA]  NA (no runs found)")

    if bE:
        print(f"[BEST EoRA]  {bE['run_name']}  rank={bE['rank']}  ar={bE['alpha_over_r']}"
              f"  val_acc={fmt(bE['val_acc'])}  test_acc={fmt(bE['test_acc'])}")
    else:
        print("[BEST EoRA]  NA (no runs found)")

    # Pairwise compare by (rank, ar)
    def keypair(r):
        return (r.get("rank"), r.get("alpha_over_r"))

    lora_map = {(r["rank"], r["alpha_over_r"]): r for r in lora}
    eora_map = {(r["rank"], r["alpha_over_r"]): r for r in eora}

    pairs = sorted(set(lora_map.keys()) | set(eora_map.keys()), key=lambda x: (x[0] if x[0] else 1e9, x[1] if x[1] else 1e9))

    print("\n--- Pairwise LoRA vs EoRA (same rank & alpha/r) ---")
    for (rk, ar) in pairs:
        L = lora_map.get((rk, ar))
        E = eora_map.get((rk, ar))

        Ls = f"LoRA: val={fmt(L['val_acc'])} test={fmt(L['test_acc'])}" if L else "LoRA: NA"
        Es = f"EoRA: val={fmt(E['val_acc'])} test={fmt(E['test_acc'])}" if E else "EoRA: NA"

        # gap (LoRA - EoRA) on test
        gap = None
        if L and E and L["test_acc"] is not None and E["test_acc"] is not None:
            gap = float(L["test_acc"]) - float(E["test_acc"])
        gap_s = f" gap(L-E)={fmt(gap)}" if gap is not None else ""
        print(f"  (r={rk}, ar={ar})  {Ls}   |   {Es}{gap_s}")

    # Quick suggestion heuristic
    print("\n--- Quick diagnosis hints ---")
    if bL and bE:
        diff = float(bL["test_acc"]) - float(bE["test_acc"])
        print(f"Best gap (LoRA - EoRA) on test: {fmt(diff)}")
        if diff < 0.005:
            print("=> EoRA is very close to LoRA. Next: sweep rank upward or try more target modules.")
        elif diff < 0.02:
            print("=> EoRA somewhat behind LoRA. Next: try larger rank or try alpha/r sweep for EoRA (0.5,1,2,4).")
        else:
            print("=> EoRA significantly behind LoRA. Next: check head-copy logic and ensure base/optimized match; then sweep rank.")
    else:
        print("Not enough runs to diagnose. Run sweep configs first.")

    print("\n=======================================================================\n")


def main():
    rows = []
    rows += scan_branch("LoRA", LORA_ROOT)
    rows += scan_branch("EoRA", EORA_ROOT)

    # sort stable
    def sort_key(r):
        rk = r["rank"] if r["rank"] is not None else 10**9
        ar = r["alpha_over_r"] if r["alpha_over_r"] is not None else 10**9
        return (r["branch"], rk, ar, r["run_name"])

    rows = sorted(rows, key=sort_key)

    write_csv(rows, SUMMARY_CSV)
    print(f"Saved CSV: {SUMMARY_CSV}")

    print_console_summary(rows)


if __name__ == "__main__":
    main()
