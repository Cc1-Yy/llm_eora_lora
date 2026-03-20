# scripts/utils/summarize_exp1_lm.py
from __future__ import annotations

import json
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, DefaultDict
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUTS_DIR / "exp1_lm_summary.csv"

_RUN_RE = re.compile(r"r(\d+)_ar(\d+(\.\d+)?)", re.IGNORECASE)


# -----------------------------
# IO helpers
# -----------------------------
def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# -----------------------------
# Parsing helpers
# -----------------------------
def parse_r_ar_from_run_name(run_name: str) -> Tuple[Optional[int], Optional[float]]:
    m = _RUN_RE.search(run_name or "")
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def infer_branch_from_metrics_path(metrics_path: Path, outputs_dir: Path) -> Optional[str]:
    """
    Infer branch ONLY from path relative to outputs_dir to avoid false positives
    from project folder names (e.g., 'llm_eora_lora').
    """
    try:
        rel = metrics_path.relative_to(outputs_dir)
    except Exception:
        return None
    if not rel.parts:
        return None
    top = rel.parts[0].lower()
    if "exp1_lora" in top:
        return "LoRA"
    if "exp1_eora" in top:
        return "EoRA"
    return None


def extract_metrics(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    LM: {"val":{"loss","ppl"}, "test":{"loss","ppl"}}
    CLS: {"val":{"loss","accuracy"}, "test":{"loss","accuracy"}}
    """
    out: Dict[str, Any] = {}
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", metrics_json.get("test.py", {})) or {}

    out["val_loss"] = val.get("loss")
    out["test_loss"] = test.get("loss")

    out["val_ppl"] = val.get("ppl")
    out["test_ppl"] = test.get("ppl")

    out["val_acc"] = val.get("accuracy")
    out["test_acc"] = test.get("accuracy")

    # optional extras (mostly LoRA training script)
    for k in [
        "best_val_loss", "global_step", "total_scheduler_steps",
        "warmup_steps", "grad_accum_steps", "stopped_by_max_train_steps"
    ]:
        out[k] = metrics_json.get(k)

    return out


def pick_cfg_or_meta(run_dir: Path) -> Dict[str, Any]:
    """
    Prefer:
      1) config_used.json (LoRA training)
      2) meta.json (EoRA closed-form)
    """
    cfg = safe_read_json(run_dir / "config_used.json")
    if cfg:
        return cfg
    meta = safe_read_json(run_dir / "meta.json")
    if meta:
        return meta
    return {}


def is_lm_run(metrics_json: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    # 1) config task_type
    t = (cfg.get("task_type") or "").lower()
    if t in ["lm", "causal_lm", "causallm", "causal-lm"]:
        return True
    # 2) ppl key in metrics
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", metrics_json.get("test.py", {})) or {}
    if "ppl" in val or "ppl" in test:
        return True
    return False


def join_list(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "|".join(str(i) for i in x)
    return str(x)


def detect_teacher_type(teacher_dir: Optional[str]) -> str:
    """
    Classify teacher type from path string.
      - optimized_full: your fully fine-tuned teacher (optimized_lm_*)
      - lora_merged: merged LoRA teacher you created (teacher_lora_*)
      - other: something else (custom)
      - none: no teacher specified (LoRA branch)
    """
    if not teacher_dir:
        return "none"
    s = str(teacher_dir).lower()
    if "optimized_lm" in s or "optimized_" in s:
        return "optimized_full"
    if "teacher_lora" in s or ("teacher" in s and "lora" in s):
        return "lora_merged"
    return "other"


# -----------------------------
# Scan
# -----------------------------
def scan_exp1_lm(outputs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not outputs_dir.exists():
        return rows

    # only scan exp1_lora* / exp1_eora* roots under outputs
    roots = []
    for p in outputs_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if ("exp1_lora" in name) or ("exp1_eora" in name):
            roots.append(p)

    for root in roots:
        for metrics_path in root.rglob("metrics.json"):
            metrics_json = safe_read_json(metrics_path)
            if not metrics_json:
                continue

            run_dir = metrics_path.parent
            cfg = pick_cfg_or_meta(run_dir)

            # keep LM only
            if not is_lm_run(metrics_json, cfg):
                continue

            branch = infer_branch_from_metrics_path(metrics_path, outputs_dir)
            if branch not in ("LoRA", "EoRA"):
                continue

            run_name = run_dir.name
            rank_name, ar_name = parse_r_ar_from_run_name(run_name)

            # adapter cfg
            adap_cfg = cfg.get("lora", {}) if branch == "LoRA" else cfg.get("eora", {})
            if not isinstance(adap_cfg, dict):
                adap_cfg = {}

            rank_cfg = adap_cfg.get("rank", None)
            alpha_cfg = adap_cfg.get("alpha", None)
            try:
                rank_cfg_i = int(rank_cfg) if rank_cfg is not None else None
            except Exception:
                rank_cfg_i = None
            try:
                alpha_cfg_f = float(alpha_cfg) if alpha_cfg is not None else None
            except Exception:
                alpha_cfg_f = None

            ar_calc = None
            if rank_cfg_i and alpha_cfg_f is not None and rank_cfg_i > 0:
                ar_calc = alpha_cfg_f / float(rank_cfg_i)

            data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
            train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}

            m = extract_metrics(metrics_json)

            # teacher info (only meaningful for EoRA; keep for LoRA too as "none")
            teacher_dir = cfg.get("optimized_model_dir") if isinstance(cfg, dict) else None
            teacher_type = detect_teacher_type(teacher_dir)

            row = {
                "branch": branch,
                "teacher_type": teacher_type,
                "teacher_dir": str(teacher_dir) if teacher_dir else "",
                "experiment_root": safe_rel(root, PROJECT_ROOT),
                "run_name": run_name,
                "run_dir": safe_rel(run_dir, PROJECT_ROOT),

                "rank(name)": rank_name,
                "ar(name)": ar_name,
                "rank(cfg)": rank_cfg_i,
                "alpha(cfg)": alpha_cfg_f,
                "ar(calc)": ar_calc,

                "dropout": adap_cfg.get("dropout"),
                "target_modules": join_list(adap_cfg.get("target_modules")),
                "svd_on_cpu": adap_cfg.get("svd_on_cpu") if branch == "EoRA" else "",

                "lr": train_cfg.get("lr"),
                "weight_decay": train_cfg.get("weight_decay"),
                "warmup_ratio": train_cfg.get("warmup_ratio"),
                "max_train_steps": train_cfg.get("max_train_steps"),
                "num_epochs": train_cfg.get("num_epochs"),
                "grad_accum_steps(cfg)": train_cfg.get("grad_accum_steps"),
                "use_amp": train_cfg.get("use_amp"),
                "scheduler": train_cfg.get("scheduler"),

                "max_length": data_cfg.get("max_length"),
                "batch_size": data_cfg.get("batch_size"),

                "val_loss": m.get("val_loss"),
                "val_ppl": m.get("val_ppl"),
                "test_loss": m.get("test_loss"),
                "test_ppl": m.get("test_ppl"),

                "best_val_loss": m.get("best_val_loss"),
                "global_step": m.get("global_step"),
            }
            rows.append(row)

    return rows


# -----------------------------
# Output
# -----------------------------
def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "branch", "teacher_type", "teacher_dir",
        "experiment_root", "run_name",
        "rank(name)", "ar(name)", "rank(cfg)", "alpha(cfg)", "ar(calc)",
        "dropout", "target_modules", "svd_on_cpu",
        "lr", "weight_decay", "warmup_ratio", "max_train_steps", "num_epochs",
        "grad_accum_steps(cfg)", "use_amp", "scheduler",
        "max_length", "batch_size",
        "val_loss", "val_ppl", "test_loss", "test_ppl",
        "best_val_loss", "global_step",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x, nd=4):
    if x is None or x == "":
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def best_by_test_ppl(rs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ok = [r for r in rs if r.get("test_ppl") is not None]
    if not ok:
        return None
    return sorted(ok, key=lambda x: float(x["test_ppl"]))[0]  # smaller is better


def keep_best_by_test_ppl(map_: Dict[Tuple, Dict[str, Any]], key: Tuple, row: Dict[str, Any]):
    """
    Keep best (lowest test_ppl); fallback to val_ppl if test missing.
    """
    def score(x):
        tp = x.get("test_ppl")
        vp = x.get("val_ppl")
        if tp is not None:
            return float(tp)
        if vp is not None:
            return float(vp) + 1e6  # always worse than having test
        return 1e18

    if key not in map_:
        map_[key] = row
        return
    if score(row) < score(map_[key]):
        map_[key] = row


def print_console_summary(rows: List[Dict[str, Any]]):
    lora = [r for r in rows if r["branch"] == "LoRA"]
    eora = [r for r in rows if r["branch"] == "EoRA"]

    print("\n================ EXP1 LM SUMMARY (copy this block to send) ================\n")
    print(f"Found runs: LoRA={len(lora)}, EoRA={len(eora)}")

    # Best LoRA overall
    bL = best_by_test_ppl(lora)
    if bL:
        print(
            f"[BEST LoRA] {bL['run_name']}  "
            f"tm={bL.get('target_modules','')}  "
            f"val_ppl={fmt(bL.get('val_ppl'))} test_ppl={fmt(bL.get('test_ppl'))}  "
            f"val_loss={fmt(bL.get('val_loss'))} test_loss={fmt(bL.get('test_loss'))}"
        )
    else:
        print("[BEST LoRA] NA")

    # EoRA: best overall and best per teacher_type
    bE = best_by_test_ppl(eora)
    if bE:
        print(
            f"[BEST EoRA overall] {bE['run_name']}  "
            f"teacher_type={bE.get('teacher_type')}  "
            f"tm={bE.get('target_modules','')}  "
            f"val_ppl={fmt(bE.get('val_ppl'))} test_ppl={fmt(bE.get('test_ppl'))}"
        )
    else:
        print("[BEST EoRA overall] NA")

    # Counts by teacher_type (EoRA only)
    eora_by_tt: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in eora:
        eora_by_tt[r.get("teacher_type", "other")].append(r)

    if eora_by_tt:
        print("\n--- EoRA runs by teacher_type ---")
        for tt in sorted(eora_by_tt.keys()):
            rs = eora_by_tt[tt]
            b = best_by_test_ppl(rs)
            if b:
                print(
                    f"  {tt}: n={len(rs)}  best={b['run_name']}  "
                    f"test_ppl={fmt(b.get('test_ppl'))} val_ppl={fmt(b.get('val_ppl'))}  "
                    f"teacher_dir={b.get('teacher_dir','')}"
                )
            else:
                print(f"  {tt}: n={len(rs)}  best=NA")

    # Pairwise compare:
    # Use key = (r, ar, target_modules, teacher_type)
    lora_best: Dict[Tuple, Dict[str, Any]] = {}
    eora_best: Dict[Tuple, Dict[str, Any]] = {}

    def make_key(row: Dict[str, Any]) -> Tuple:
        return (
            row.get("rank(name)"),
            row.get("ar(name)"),
            row.get("target_modules", ""),
            row.get("teacher_type", "none") if row.get("branch") == "EoRA" else "none",
        )

    for r in lora:
        keep_best_by_test_ppl(lora_best, make_key(r), r)
    for r in eora:
        keep_best_by_test_ppl(eora_best, make_key(r), r)

    keys_all = sorted(
        set(lora_best.keys()) | set(eora_best.keys()),
        key=lambda x: (
            x[3],                              # teacher_type first (EoRA groups)
            x[0] if x[0] is not None else 10**9,
            x[1] if x[1] is not None else 10**9,
            x[2],
        ),
    )

    print("\n--- Pairwise LoRA vs EoRA (same r/ar/target_modules, grouped by teacher_type) ---")
    for (rk, ar, tm, tt) in keys_all:
        L = lora_best.get((rk, ar, tm, "none"))
        E = eora_best.get((rk, ar, tm, tt)) if tt != "none" else None

        Ls = f"LoRA: test_ppl={fmt(L.get('test_ppl'))} val_ppl={fmt(L.get('val_ppl'))}" if L else "LoRA: NA"
        Es = f"EoRA({tt}): test_ppl={fmt(E.get('test_ppl'))} val_ppl={fmt(E.get('val_ppl'))}" if E else f"EoRA({tt}): NA"

        gap = None
        if L and E and L.get("test_ppl") is not None and E.get("test_ppl") is not None:
            gap = float(E["test_ppl"]) - float(L["test_ppl"])  # >0 => EoRA worse
        gap_s = f" gap(E-L)={fmt(gap)}" if gap is not None else ""
        print(f"  (tt={tt}, r={rk}, ar={ar}, tm={tm})  {Ls}   |   {Es}{gap_s}")

    print("\n==========================================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", type=str, default=str(DEFAULT_OUTPUTS_DIR))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_csv = Path(args.out_csv)

    rows = scan_exp1_lm(outputs_dir)

    def sort_key(r: Dict[str, Any]):
        rk = r.get("rank(name)")
        ar = r.get("ar(name)")
        rk = rk if rk is not None else 10**9
        ar = ar if ar is not None else 10**9
        return (r.get("branch", ""), r.get("teacher_type", ""), rk, ar, r.get("target_modules", ""), r.get("run_dir", ""))

    rows = sorted(rows, key=sort_key)

    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")

    print_console_summary(rows)


if __name__ == "__main__":
    main()