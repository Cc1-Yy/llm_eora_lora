import re
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUT_CSV = OUTPUTS_DIR / "eora_support_summary.csv"

# outputs/eora_support_M2_attnproj_fc_r64/r64_ar1/metrics.json
TOP_RE = re.compile(r"^eora_support_(?P<set>.+)_r(?P<rank>\d+)$", re.IGNORECASE)
RUN_RE = re.compile(r"r(?P<rank>\d+)_ar(?P<ar>\d+(\.\d+)?)", re.IGNORECASE)

def safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def fmt(x, nd=4):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def main():
    rows: List[Dict[str, Any]] = []

    for top in OUTPUTS_DIR.iterdir():
        if not top.is_dir():
            continue
        m = TOP_RE.match(top.name)
        if not m:
            continue

        set_name = m.group("set")
        rank_top = int(m.group("rank"))

        for metrics_path in top.rglob("metrics.json"):
            metrics = safe_read_json(metrics_path)
            if not metrics:
                continue

            run_dir = metrics_path.parent
            run_name = run_dir.name
            mrun = RUN_RE.search(run_name)
            ar = float(mrun.group("ar")) if mrun else None

            meta = safe_read_json(run_dir / "meta.json") or {}
            eora_cfg = meta.get("eora", {}) if isinstance(meta.get("eora", {}), dict) else {}
            teacher_dir = meta.get("optimized_model_dir", "")

            val = metrics.get("val", {}) or {}
            test = metrics.get("test", {}) or {}

            rows.append({
                "set": set_name,
                "rank(top)": rank_top,
                "run_name": run_name,
                "ar(name)": ar,
                "target_modules": "|".join(eora_cfg.get("target_modules", [])) if eora_cfg else "",
                "teacher_dir": teacher_dir,
                "val_loss": val.get("loss"),
                "val_ppl": val.get("ppl"),
                "test_loss": test.get("loss"),
                "test_ppl": test.get("ppl"),
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            })

    rows.sort(key=lambda r: (r["set"], r["rank(top)"], (r["ar(name)"] if r["ar(name)"] is not None else 1e9)))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = ["set","rank(top)","run_name","ar(name)","target_modules","val_ppl","test_ppl","val_loss","test_loss","teacher_dir","run_dir"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved CSV: {OUT_CSV}")

    # console best per set+rank
    print("\n--- Best (lowest test_ppl) per set ---")
    by_set = {}
    for r in rows:
        tp = r.get("test_ppl")
        if tp is None:
            continue
        key = r["set"]
        if key not in by_set or float(tp) < float(by_set[key]["test_ppl"]):
            by_set[key] = r

    for k in sorted(by_set.keys()):
        b = by_set[k]
        print(f"{k:20s} best={b['run_name']:10s} rank={b['rank(top)']}  test_ppl={fmt(b['test_ppl'])}  val_ppl={fmt(b['val_ppl'])}  tm={b['target_modules']}")

if __name__ == "__main__":
    main()