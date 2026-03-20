# scripts/utils/summarize_exp2.py
from __future__ import annotations

import json
import math
import re
import csv
import argparse
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_EXP2_ROOT = OUTPUTS_DIR / "cls" / "exp2"
DEFAULT_SUMMARY_CSV = OUTPUTS_DIR / "cls" / "exp2_summary.csv"


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


def _format(x, nd=4):
    if x is None:
        return "NA"
    try:
        xf = float(x)
        if math.isfinite(xf):
            return f"{xf:.{nd}f}"
    except Exception:
        pass
    return str(x)


def _short(p: Path) -> str:
    return str(p).replace("\\", "/")


def _infer_branch(exp_name: str) -> Optional[str]:
    name = exp_name.lower()
    if "lora" in name:
        return "LoRA"
    if "eora" in name:
        return "EoRA"
    return None


def _get_exp_name_from_metrics_path(metrics_path: Path, exp_root: Path) -> str:
    rel = metrics_path.relative_to(exp_root)
    return rel.parts[0]


def _parse_seed(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def _mean_std(xs: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return mu, math.sqrt(var)


# -------------------------
# Run structure
# -------------------------

@dataclass
class Run:
    branch: str
    exp_name: str
    run_id: str
    path: Path

    seed: Optional[int]

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

    metrics: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class Agg:
    branch: str
    rank: Optional[int]
    ar: Optional[float]
    temperature: Optional[float]
    n: int

    test_acc_mean: Optional[float]
    test_acc_std: Optional[float]

    test_kl_mean: Optional[float]
    test_kl_std: Optional[float]

    test_mse_mean: Optional[float]
    test_mse_std: Optional[float]

    test_ce_mean: Optional[float]
    test_ce_std: Optional[float]


def _infer_rank_ar_T_from_name(name: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    m_r = re.search(r"r(\d+)", name)
    m_ar = re.search(r"ar([0-9]+(?:\.[0-9]+)?)", name)
    m_T = re.search(r"T([0-9]+(?:\.[0-9]+)?)", name)

    rank = int(m_r.group(1)) if m_r else None
    ar = float(m_ar.group(1)) if m_ar else None
    T = float(m_T.group(1)) if m_T else None
    return rank, ar, T


def _collect_one_run(branch: str, exp_name: str, run_dir: Path) -> Optional[Run]:
    metrics_path = run_dir / "metrics.json"
    meta_path = run_dir / "meta.json"

    metrics = _read_json(metrics_path) or {}
    meta = _read_json(meta_path) or {}

    if not metrics:
        return None

    val = metrics.get("val", {}) if isinstance(metrics.get("val"), dict) else {}
    test = (
        metrics.get("test", {}) if isinstance(metrics.get("test"), dict)
        else metrics.get("test.py", {}) if isinstance(metrics.get("test.py"), dict)
        else {}
    )

    val_acc = _safe_float(val.get("accuracy"))
    test_acc = _safe_float(test.get("accuracy"))

    val_kl = _safe_float(val.get("kl_to_teacher"))
    test_kl = _safe_float(test.get("kl_to_teacher"))

    val_mse = _safe_float(val.get("mse_logits_to_teacher"))
    test_mse = _safe_float(test.get("mse_logits_to_teacher"))

    val_ce = _safe_float(val.get("ce_loss"))
    test_ce = _safe_float(test.get("ce_loss"))

    run_id = run_dir.name

    r_exp, ar_exp, T_exp = _infer_rank_ar_T_from_name(exp_name)
    r_id, ar_id, T_id = _infer_rank_ar_T_from_name(run_id)

    lora_meta = meta.get("lora", {}) if isinstance(meta.get("lora"), dict) else {}
    eora_meta = meta.get("eora", {}) if isinstance(meta.get("eora"), dict) else {}
    distill_meta = meta.get("distill", {}) if isinstance(meta.get("distill"), dict) else {}
    kd_meta = meta.get("kd", {}) if isinstance(meta.get("kd"), dict) else {}

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
    if temp is None:
        temp = kd_meta.get("T", None)

    if rank is None:
        rank = r_exp if r_exp is not None else r_id
    if temp is None:
        temp = T_exp if T_exp is not None else T_id

    rank = int(rank) if rank is not None else None
    alpha = _safe_float(alpha)

    ar = None
    if alpha is not None and rank is not None and rank > 0:
        ar = alpha / rank
    else:
        ar = ar_exp if ar_exp is not None else ar_id

    seed = _parse_seed(exp_name)
    if seed is None:
        seed = _parse_seed(run_id)

    return Run(
        branch=branch,
        exp_name=exp_name,
        run_id=run_id,
        path=run_dir,
        seed=seed,
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


def _scan_runs(root: Path) -> List[Run]:
    runs: List[Run] = []
    if not root.exists():
        return runs

    for metrics_path in sorted(root.rglob("metrics.json")):
        exp_name = _get_exp_name_from_metrics_path(metrics_path, root)
        branch = _infer_branch(exp_name)
        if branch is None:
            continue

        run_dir = metrics_path.parent
        r = _collect_one_run(branch, exp_name, run_dir)
        if r is not None:
            runs.append(r)
    return runs


# -------------------------
# Ranking / grouping
# -------------------------

def _score_key_for_best(r: Run) -> Tuple[float, float, float]:
    kl = r.test_kl if r.test_kl is not None else float("inf")
    mse = r.test_mse if r.test_mse is not None else float("inf")
    acc = r.test_acc if r.test_acc is not None else -1.0
    return (kl, mse, -acc)


def _pick_best(runs: List[Run]) -> Optional[Run]:
    if not runs:
        return None
    return sorted(runs, key=_score_key_for_best)[0]


def _group_key(r: Run) -> Tuple[Optional[int], Optional[float], Optional[float], str]:
    def _round_or_none(x):
        return None if x is None else round(float(x), 6)
    return (r.rank, _round_or_none(r.ar), _round_or_none(r.temperature), r.branch)


def _pair_key_from_group(g: Agg) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    return (g.rank, g.ar, g.temperature)


def _aggregate_runs(runs: List[Run]) -> List[Agg]:
    groups: Dict[Tuple[Optional[int], Optional[float], Optional[float], str], List[Run]] = defaultdict(list)
    for r in runs:
        groups[_group_key(r)].append(r)

    out: List[Agg] = []
    for (rank, ar, temp, branch), rs in groups.items():
        acc_mu, acc_sd = _mean_std([r.test_acc for r in rs])
        kl_mu, kl_sd = _mean_std([r.test_kl for r in rs])
        mse_mu, mse_sd = _mean_std([r.test_mse for r in rs])
        ce_mu, ce_sd = _mean_std([r.test_ce for r in rs])

        out.append(
            Agg(
                branch=branch,
                rank=rank,
                ar=ar,
                temperature=temp,
                n=len(rs),
                test_acc_mean=acc_mu,
                test_acc_std=acc_sd,
                test_kl_mean=kl_mu,
                test_kl_std=kl_sd,
                test_mse_mean=mse_mu,
                test_mse_std=mse_sd,
                test_ce_mean=ce_mu,
                test_ce_std=ce_sd,
            )
        )

    out = sorted(
        out,
        key=lambda g: (
            g.branch,
            g.rank if g.rank is not None else 10**9,
            g.ar if g.ar is not None else 10**9,
            g.temperature if g.temperature is not None else 10**9,
        ),
    )
    return out


# -------------------------
# CSV
# -------------------------

def _write_csv(rows: List[Run], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "branch", "exp_name", "run_id", "seed",
        "rank", "alpha", "ar", "temperature",
        "val_acc", "test_acc",
        "val_kl", "test_kl",
        "val_mse", "test_mse",
        "val_ce", "test_ce",
        "run_dir",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({
                "branch": r.branch,
                "exp_name": r.exp_name,
                "run_id": r.run_id,
                "seed": r.seed,
                "rank": r.rank,
                "alpha": r.alpha,
                "ar": r.ar,
                "temperature": r.temperature,
                "val_acc": r.val_acc,
                "test_acc": r.test_acc,
                "val_kl": r.val_kl,
                "test_kl": r.test_kl,
                "val_mse": r.val_mse,
                "test_mse": r.test_mse,
                "val_ce": r.val_ce,
                "test_ce": r.test_ce,
                "run_dir": _short(r.path.resolve().relative_to(PROJECT_ROOT.resolve())),
            })


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP2_ROOT))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()

    runs = _scan_runs(root)

    lora_runs = [r for r in runs if r.branch == "LoRA"]
    eora_runs = [r for r in runs if r.branch == "EoRA"]

    best_lora = _pick_best(lora_runs)
    best_eora = _pick_best(eora_runs)

    runs = sorted(
        runs,
        key=lambda r: (
            r.branch,
            r.rank if r.rank is not None else 10**9,
            r.ar if r.ar is not None else 10**9,
            r.temperature if r.temperature is not None else 10**9,
            r.seed if r.seed is not None else -1,
            r.exp_name,
            r.run_id,
        ),
    )

    _write_csv(runs, out_csv)
    print(f"Saved CSV: {_short(out_csv)}")

    aggs = _aggregate_runs(runs)
    lora_aggs = [g for g in aggs if g.branch == "LoRA"]
    eora_aggs = [g for g in aggs if g.branch == "EoRA"]

    print("\n================ EXP2 SUMMARY ================\n")
    print(f"Scan dir:\n  {_short(root)}\n")
    print(f"Found raw runs: LoRA={len(lora_runs)}, EoRA={len(eora_runs)}")
    print(f"Found grouped settings: LoRA={len(lora_aggs)}, EoRA={len(eora_aggs)}\n")

    if best_lora:
        print(
            f"[BEST LoRA RAW]  {best_lora.exp_name}  seed={best_lora.seed}  "
            f"rank={best_lora.rank}  ar={_format(best_lora.ar,3)}  T={_format(best_lora.temperature,2)}  "
            f"test_acc={_format(best_lora.test_acc)}  test_kl={_format(best_lora.test_kl)}  test_mse={_format(best_lora.test_mse)}"
        )
    else:
        print("[BEST LoRA RAW]  NA")

    if best_eora:
        print(
            f"[BEST EoRA RAW]  {best_eora.exp_name}  seed={best_eora.seed}  "
            f"rank={best_eora.rank}  ar={_format(best_eora.ar,3)}  T={_format(best_eora.temperature,2)}  "
            f"test_acc={_format(best_eora.test_acc)}  test_kl={_format(best_eora.test_kl)}  test_mse={_format(best_eora.test_mse)}"
        )
    else:
        print("[BEST EoRA RAW]  NA")

    print("\n--- Aggregated by (branch, rank, ar, T) ---")
    for g in aggs:
        print(
            f"{g.branch:4s}  r={g.rank}  ar={_format(g.ar,3)}  T={_format(g.temperature,2)}  n={g.n}  "
            f"acc={_format(g.test_acc_mean)}±{_format(g.test_acc_std)}  "
            f"kl={_format(g.test_kl_mean)}±{_format(g.test_kl_std)}  "
            f"mse={_format(g.test_mse_mean)}±{_format(g.test_mse_std)}"
        )

    print("\n--- Pairwise LoRA vs EoRA (same rank & ar & T, aggregated) ---")
    lora_map = {_pair_key_from_group(g): g for g in lora_aggs}
    eora_map = {_pair_key_from_group(g): g for g in eora_aggs}

    all_keys = sorted(
        set(lora_map.keys()) | set(eora_map.keys()),
        key=lambda x: (
            x[0] if x[0] is not None else 10**9,
            x[1] if x[1] is not None else 10**9,
            x[2] if x[2] is not None else 10**9,
        ),
    )

    for k in all_keys:
        lg = lora_map.get(k)
        eg = eora_map.get(k)
        tag = f"(r={k[0]}, ar={_format(k[1],3)}, T={_format(k[2],2)})"

        if lg is None:
            print(f"  {tag}  LoRA: NA   |   EoRA: acc={_format(eg.test_acc_mean)} kl={_format(eg.test_kl_mean)} mse={_format(eg.test_mse_mean)}")
            continue
        if eg is None:
            print(f"  {tag}  LoRA: acc={_format(lg.test_acc_mean)} kl={_format(lg.test_kl_mean)} mse={_format(lg.test_mse_mean)}   |   EoRA: NA")
            continue

        gap_kl = None
        if lg.test_kl_mean is not None and eg.test_kl_mean is not None:
            gap_kl = float(lg.test_kl_mean) - float(eg.test_kl_mean)  # positive => LoRA worse on KL

        print(
            f"  {tag}  "
            f"LoRA: acc={_format(lg.test_acc_mean)} kl={_format(lg.test_kl_mean)} mse={_format(lg.test_mse_mean)}   |   "
            f"EoRA: acc={_format(eg.test_acc_mean)} kl={_format(eg.test_kl_mean)} mse={_format(eg.test_mse_mean)}   "
            f"gap_KL(L-E)={_format(gap_kl,4)}"
        )

    print("\n=============================================\n")


if __name__ == "__main__":
    main()