# scripts/utils/summarize_exp2.py
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Helpers
# -------------------------

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _get_nested(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _format(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, (int, float)) and (math.isfinite(float(x))):
        return f"{float(x):.{nd}f}"
    return str(x)

def _short(p: Path) -> str:
    # friendlier printing
    return str(p).replace("\\", "/")


# -------------------------
# Run structure
# -------------------------

@dataclass
class Run:
    branch: str            # "LoRA" or "EoRA"
    run_id: str            # folder name
    path: Path

    rank: Optional[int]
    alpha: Optional[float]
    ar: Optional[float]
    temperature: Optional[float]

    val_acc: Optional[float]
    test_acc: Optional[float]

    val_kl: Optional[float]
    test_kl: Optional[float]

    val_mse: Optional[float]
    test_mse: Optional[float]

    val_ce: Optional[float]
    test_ce: Optional[float]

    # raw dicts (for debugging if needed)
    metrics: Dict[str, Any]
    meta: Dict[str, Any]


def _infer_rank_ar_T_from_run_id(run_id: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Accept names like:
      r16_ar1_T1
      r64_ar1.25_T1
      r32_ar0.75_T2
    """
    m_r = re.search(r"r(\d+)", run_id)
    m_ar = re.search(r"ar([0-9]+(?:\.[0-9]+)?)", run_id)
    m_T = re.search(r"T([0-9]+(?:\.[0-9]+)?)", run_id)
    rank = int(m_r.group(1)) if m_r else None
    ar = float(m_ar.group(1)) if m_ar else None
    T = float(m_T.group(1)) if m_T else None
    return rank, ar, T


def _collect_one_run(branch: str, run_dir: Path) -> Optional[Run]:
    metrics_path = run_dir / "metrics.json"
    meta_path = run_dir / "meta.json"

    metrics = _read_json(metrics_path) or {}
    meta = _read_json(meta_path) or {}

    # Many of your scripts write:
    # metrics.json: {"val": {...}, "test.py": {...}}
    val = metrics.get("val", {}) if isinstance(metrics.get("val"), dict) else {}
    test = metrics.get("test.py", {}) if isinstance(metrics.get("test.py"), dict) else {}

    # pull common metrics (Exp2)
    val_acc = _safe_float(val.get("accuracy"))
    test_acc = _safe_float(test.get("accuracy"))

    val_kl = _safe_float(val.get("kl_to_teacher"))
    test_kl = _safe_float(test.get("kl_to_teacher"))

    val_mse = _safe_float(val.get("mse_logits_to_teacher"))
    test_mse = _safe_float(test.get("mse_logits_to_teacher"))

    val_ce = _safe_float(val.get("ce_loss"))
    test_ce = _safe_float(test.get("ce_loss"))

    # rank/alpha/ar/T:
    # Prefer meta/config fields if present; else infer from folder name.
    run_id = run_dir.name
    r_id, ar_id, T_id = _infer_rank_ar_T_from_run_id(run_id)

    # try from meta/config snapshot if present
    # meta may contain "lora" or "eora" dict, and "distill.temperature"
    lora_meta = meta.get("lora", {}) if isinstance(meta.get("lora"), dict) else {}
    eora_meta = meta.get("eora", {}) if isinstance(meta.get("eora"), dict) else {}
    distill_meta = meta.get("distill", {}) if isinstance(meta.get("distill"), dict) else {}

    rank = None
    alpha = None
    temp = None

    if branch == "LoRA":
        rank = lora_meta.get("rank", None)
        alpha = lora_meta.get("alpha", None)
    else:
        rank = eora_meta.get("rank", None)
        alpha = eora_meta.get("alpha", None)

    temp = distill_meta.get("temperature", None)

    # fallback to folder parse
    if rank is None:
        rank = r_id
    if temp is None:
        temp = T_id

    rank = int(rank) if rank is not None else None
    alpha = _safe_float(alpha)

    ar = None
    if alpha is not None and rank is not None and rank > 0:
        ar = alpha / rank
    elif ar_id is not None:
        ar = ar_id

    # If metrics.json doesn't have expected fields, still keep run if has any test.py dict
    if not metrics:
        return None

    return Run(
        branch=branch,
        run_id=run_id,
        path=run_dir,
        rank=rank,
        alpha=alpha,
        ar=ar,
        temperature=_safe_float(temp),
        val_acc=val_acc,
        test_acc=test_acc,
        val_kl=val_kl,
        test_kl=test_kl,
        val_mse=val_mse,
        test_mse=test_mse,
        val_ce=val_ce,
        test_ce=test_ce,
        metrics=metrics,
        meta=meta,
    )


def _scan_runs(root: Path, branch: str) -> List[Run]:
    runs: List[Run] = []
    if not root.exists():
        return runs
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "metrics.json").exists():
            continue
        r = _collect_one_run(branch, child)
        if r is not None:
            runs.append(r)
    return runs


# -------------------------
# Ranking / best selection
# -------------------------

def _score_key_for_best(r: Run) -> Tuple[float, float, float]:
    """
    Smaller is better.
    Primary: test_kl
    Secondary: test_mse
    Tertiary: -test_acc (i.e. larger acc better)
    """
    kl = r.test_kl if r.test_kl is not None else float("inf")
    mse = r.test_mse if r.test_mse is not None else float("inf")
    acc = r.test_acc if r.test_acc is not None else -1.0
    return (kl, mse, -acc)

def _pick_best(runs: List[Run]) -> Optional[Run]:
    if not runs:
        return None
    return sorted(runs, key=_score_key_for_best)[0]


# -------------------------
# Pairwise table
# -------------------------

def _pair_key(r: Run) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    # pair by (rank, ar, temperature)
    def _round_ar(x):
        if x is None:
            return None
        return round(float(x), 4)
    def _round_T(x):
        if x is None:
            return None
        return round(float(x), 4)
    return (r.rank, _round_ar(r.ar), _round_T(r.temperature))


def main():
    project_root = Path(__file__).resolve().parents[2]  # scripts/utils -> project root
    out_root = project_root / "outputs"

    lora_root = out_root / "exp2_lora"
    eora_root = out_root / "exp2_eora"

    lora_runs = _scan_runs(lora_root, "LoRA")
    eora_runs = _scan_runs(eora_root, "EoRA")

    best_lora = _pick_best(lora_runs)
    best_eora = _pick_best(eora_runs)

    print("\n================ EXP2 SUMMARY (copy this block to send) ================\n")
    print(f"Found runs: LoRA={len(lora_runs)}, EoRA={len(eora_runs)}")
    print("Scan dirs:")
    print(f"  {_short(lora_root)}")
    print(f"  {_short(eora_root)}")
    print("")

    if best_lora:
        print(f"[BEST LoRA]  {best_lora.run_id}  rank={best_lora.rank}  ar={_format(best_lora.ar,3)}  "
              f"T={_format(best_lora.temperature,2)}  "
              f"test_acc={_format(best_lora.test_acc)}  test_kl={_format(best_lora.test_kl)}  test_mse={_format(best_lora.test_mse)}")
    else:
        print("[BEST LoRA]  NA")

    if best_eora:
        print(f"[BEST EoRA]  {best_eora.run_id}  rank={best_eora.rank}  ar={_format(best_eora.ar,3)}  "
              f"T={_format(best_eora.temperature,2)}  "
              f"test_acc={_format(best_eora.test_acc)}  test_kl={_format(best_eora.test_kl)}  test_mse={_format(best_eora.test_mse)}")
    else:
        print("[BEST EoRA]  NA")

    print("\n--- Pairwise LoRA vs EoRA (same rank & alpha/r & T) ---")

    lora_map: Dict[Tuple[Optional[int], Optional[float], Optional[float]], Run] = { _pair_key(r): r for r in lora_runs }
    eora_map: Dict[Tuple[Optional[int], Optional[float], Optional[float]], Run] = { _pair_key(r): r for r in eora_runs }

    all_keys = sorted(set(lora_map.keys()) | set(eora_map.keys()), key=lambda x: (x[0] or -1, x[1] or -1, x[2] or -1))

    def _line_for(r: Optional[Run]) -> str:
        if r is None:
            return "NA"
        return f"acc={_format(r.test_acc)} kl={_format(r.test_kl)} mse={_format(r.test_mse)}"

    # show pairwise comparison focusing on teacher distance (KL then MSE)
    best_gap = None
    best_gap_key = None

    for k in all_keys:
        lr = lora_map.get(k, None)
        er = eora_map.get(k, None)

        rnk, ar, T = k
        tag = f"(r={rnk}, ar={_format(ar,3)}, T={_format(T,2)})"

        if lr and er and lr.test_kl is not None and er.test_kl is not None:
            gap = float(lr.test_kl) - float(er.test_kl)  # positive => LoRA worse than EoRA in KL
            if best_gap is None or abs(gap) > abs(best_gap):
                best_gap = gap
                best_gap_key = k
            print(f"  {tag}  LoRA[{lr.run_id}]: {_line_for(lr)}   |   EoRA[{er.run_id}]: {_line_for(er)}   "
                  f"gap_KL(L-E)={_format(gap,4)}")
        else:
            # still print if either exists (for visibility)
            print(f"  {tag}  LoRA: {_line_for(lr)}   |   EoRA: {_line_for(er)}")

    print("\n--- Quick diagnosis hints ---")

    # summarize best KL per rank for each method
    def _best_by_rank(runs: List[Run]) -> Dict[int, Run]:
        out = {}
        for r in runs:
            if r.rank is None or r.test_kl is None:
                continue
            if r.rank not in out or _score_key_for_best(r) < _score_key_for_best(out[r.rank]):
                out[r.rank] = r
        return out

    l_best = _best_by_rank(lora_runs)
    e_best = _best_by_rank(eora_runs)

    if l_best:
        print("Best LoRA (by lowest test.py KL) per rank:")
        for rk in sorted(l_best):
            r = l_best[rk]
            print(f"  r={rk}: {r.run_id}  ar={_format(r.ar,3)} T={_format(r.temperature,2)}  "
                  f"test_acc={_format(r.test_acc)} test_kl={_format(r.test_kl)} test_mse={_format(r.test_mse)}")
    else:
        print("Best LoRA per rank: NA")

    if e_best:
        print("Best EoRA (by lowest test.py KL) per rank:")
        for rk in sorted(e_best):
            r = e_best[rk]
            print(f"  r={rk}: {r.run_id}  ar={_format(r.ar,3)} T={_format(r.temperature,2)}  "
                  f"test_acc={_format(r.test_acc)} test_kl={_format(r.test_kl)} test_mse={_format(r.test_mse)}")
    else:
        print("Best EoRA per rank: NA")

    # Suggest next moves
    # 1) If pairwise keys missing (only LoRA or only EoRA), suggest aligning sweeps
    missing_pairs = [(k, lora_map.get(k), eora_map.get(k)) for k in all_keys if (lora_map.get(k) is None or eora_map.get(k) is None)]
    if missing_pairs:
        print(f"\nYou have {len(missing_pairs)} unmatched settings (only one side has run).")
        print("=> If you want clean pairwise plots, run the missing counterpart configs for those (rank, ar, T).")

    # 2) If LoRA degrades at ar=1.25 for same rank/T, hint about lr/epochs/T
    # quick heuristic check: for each rank,T compare ar~1.0 vs ar~1.25
    def _find(run_list: List[Run], rank: int, ar: float, T: float) -> Optional[Run]:
        for r in run_list:
            if r.rank == rank and r.temperature is not None and abs(r.temperature - T) < 1e-6 and r.ar is not None and abs(r.ar - ar) < 1e-3:
                return r
        return None

    ranks_to_check = sorted({r.rank for r in lora_runs if r.rank is not None})
    for rk in ranks_to_check:
        # only T=1.0 in your current design, but keep generic
        Ts = sorted({r.temperature for r in lora_runs if r.rank == rk and r.temperature is not None})
        for T in Ts:
            r1 = _find(lora_runs, rk, 1.0, float(T))
            r125 = _find(lora_runs, rk, 1.25, float(T))
            if r1 and r125 and r1.test_kl is not None and r125.test_kl is not None:
                if r125.test_kl > r1.test_kl * 1.15:
                    print(f"\nLoRA rank {rk} at T={_format(T,2)}: ar=1.25 KL is worse than ar=1.0.")
                    print("=> Consider: lower lr (e.g. 5e-5), fewer epochs (2), or try temperature T=2/4 to stabilize KL training.")

    print("\n=======================================================================\n")


if __name__ == "__main__":
    main()
