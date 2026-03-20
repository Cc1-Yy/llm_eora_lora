import re
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"
OUT_CSV = OUTPUTS / "teacher_strength_summary.csv"

TEACHER_RE = re.compile(r"^teacher_steps(\d+)$", re.IGNORECASE)
EORA_RE = re.compile(r"^eora_vs_teacher_steps(\d+)$", re.IGNORECASE)

def safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_best_teacher_run(steps_dir: Path) -> Optional[Path]:
    # outputs/teacher_stepsXXXX/runs/**/metrics.json
    runs = steps_dir / "runs"
    if not runs.exists():
        return None
    best = None
    best_tp = None
    for mp in runs.rglob("metrics.json"):
        m = safe_read_json(mp)
        if not m:
            continue
        tp = (m.get("test", {}) or {}).get("ppl")
        if tp is None:
            continue
        if best is None or float(tp) < float(best_tp):
            best = mp
            best_tp = tp
    return best

def find_best_eora_run(steps_dir: Path) -> Optional[Path]:
    # outputs/eora_vs_teacher_stepsXXXX/**/metrics.json
    best = None
    best_tp = None
    for mp in steps_dir.rglob("metrics.json"):
        m = safe_read_json(mp)
        if not m:
            continue
        tp = (m.get("test", {}) or {}).get("ppl")
        if tp is None:
            continue
        if best is None or float(tp) < float(best_tp):
            best = mp
            best_tp = tp
    return best

def main():
    rows: List[Dict[str, Any]] = []

    # scan teacher dirs
    for p in OUTPUTS.iterdir():
        if not p.is_dir():
            continue
        mt = TEACHER_RE.match(p.name)
        me = EORA_RE.match(p.name)
        if mt:
            steps = int(mt.group(1))
            best_mp = find_best_teacher_run(p)
            if best_mp:
                mj = safe_read_json(best_mp) or {}
                rows.append({
                    "steps": steps,
                    "kind": "teacher",
                    "val_ppl": (mj.get("val", {}) or {}).get("ppl"),
                    "test_ppl": (mj.get("test", {}) or {}).get("ppl"),
                    "run_dir": str(best_mp.parent.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                })
        if me:
            steps = int(me.group(1))
            best_mp = find_best_eora_run(p)
            if best_mp:
                mj = safe_read_json(best_mp) or {}
                meta = safe_read_json(best_mp.parent / "meta.json") or {}
                rows.append({
                    "steps": steps,
                    "kind": "eora",
                    "val_ppl": (mj.get("val", {}) or {}).get("ppl"),
                    "test_ppl": (mj.get("test", {}) or {}).get("ppl"),
                    "teacher_dir": meta.get("optimized_model_dir", ""),
                    "run_dir": str(best_mp.parent.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                })

    rows.sort(key=lambda r: (r["steps"], 0 if r["kind"]=="teacher" else 1))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = ["steps","kind","val_ppl","test_ppl","teacher_dir","run_dir"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved CSV: {OUT_CSV}")

    # quick console view
    print("\n--- Summary (best per steps) ---")
    by_steps = {}
    for r in rows:
        by_steps.setdefault(r["steps"], {})[r["kind"]] = r
    for s in sorted(by_steps.keys()):
        t = by_steps[s].get("teacher")
        e = by_steps[s].get("eora")
        if t:
            print(f"steps={s:5d} teacher test_ppl={t['test_ppl']:.4f}")
        if e:
            print(f"          eora   test_ppl={e['test_ppl']:.4f}")

if __name__ == "__main__":
    main()