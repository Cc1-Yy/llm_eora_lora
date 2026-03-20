# scripts/utils/inspect_lm_runs.py
"""
用法：
1. 看整个 LM 目录的结果总表
python scripts/utils/inspect_lm_runs.py --root outputs/lm
2. 只看 Exp1
python scripts/utils/inspect_lm_runs.py --root outputs/lm/exp1
3. 只看包含 tm-apf 的结果
python scripts/utils/inspect_lm_runs.py --root outputs/lm/exp1 --match tm-apf
4. 看某个具体 run 的详细信息
例如看 2_eora_optfull_r64_ar1_tm-apf：
python scripts/utils/inspect_lm_runs.py --root outputs/lm/exp1 --detail 2_eora_optfull_r64_ar1_tm-apf
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def safe_get(d: Optional[Dict[str, Any]], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_run_type(
    metrics: Optional[Dict[str, Any]],
    meta: Optional[Dict[str, Any]],
    run_info: Optional[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]],
) -> str:
    if run_info is not None:
        return "optimized"
    if meta is not None and ("eora" in meta or "optimized_model_dir" in meta):
        return "eora"
    if metrics is not None and ("lora" in metrics or "best_val_loss" in metrics):
        return "lora"
    if cfg is not None:
        if "eora" in cfg:
            return "eora"
        if "lora" in cfg:
            return "lora"
    return "unknown"


def extract_target_modules(kind: str, meta, metrics, cfg):
    if kind == "eora":
        return safe_get(meta, "eora", "target_modules") or safe_get(cfg, "eora", "target_modules")
    if kind == "lora":
        return safe_get(metrics, "lora", "target_modules") or safe_get(cfg, "lora", "target_modules")
    return None


def extract_rank_alpha(kind: str, meta, metrics, cfg):
    if kind == "eora":
        rank = safe_get(meta, "eora", "rank")
        alpha = safe_get(meta, "eora", "alpha")
        if rank is None:
            rank = safe_get(cfg, "eora", "rank")
        if alpha is None:
            alpha = safe_get(cfg, "eora", "alpha")
        return rank, alpha
    if kind == "lora":
        rank = safe_get(metrics, "lora", "rank")
        alpha = safe_get(metrics, "lora", "alpha")
        if rank is None:
            rank = safe_get(cfg, "lora", "rank")
        if alpha is None:
            alpha = safe_get(cfg, "lora", "alpha")
        return rank, alpha
    return None, None


def extract_train_cfg(metrics, run_info, cfg):
    train_cfg = {}
    if run_info is not None:
        train_cfg = safe_get(run_info, "train", default={}) or {}
    elif metrics is not None and "train" in metrics:
        train_cfg = metrics.get("train", {}) or {}
    elif cfg is not None:
        train_cfg = cfg.get("train", {}) or {}
    return train_cfg


def extract_seed(metrics, meta, run_info, cfg):
    for src in (metrics, meta, run_info, cfg):
        if isinstance(src, dict) and "seed" in src:
            return src["seed"]
    return None


def extract_model_name(metrics, meta, run_info, cfg):
    for src in (metrics, meta, run_info, cfg):
        if isinstance(src, dict) and "model_name" in src:
            return src["model_name"]
    return None


def extract_task_type(metrics, meta, run_info, cfg):
    for src in (metrics, meta, run_info, cfg):
        if isinstance(src, dict) and "task_type" in src:
            return src["task_type"]
    return None


def extract_teacher_path(meta, cfg):
    return (
        safe_get(meta, "optimized_model_dir")
        or safe_get(cfg, "optimized_model_dir")
        or safe_get(cfg, "teacher_model_dir")
    )


def fmt_float(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def fmt_modules(mods):
    if not mods:
        return "-"
    if isinstance(mods, list):
        return ",".join(str(x) for x in mods)
    return str(mods)


def collect_runs(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent

        metrics = load_json(metrics_path)
        meta = load_json(run_dir / "meta.json")
        run_info = load_json(run_dir / "run_info.json")

        cfg = None
        if (run_dir / "config_used.yaml").exists():
            cfg = load_yaml(run_dir / "config_used.yaml")
        elif (run_dir / "config_used.json").exists():
            cfg = load_json(run_dir / "config_used.json")

        kind = infer_run_type(metrics, meta, run_info, cfg)

        val = safe_get(metrics, "val", default={}) or {}
        test = safe_get(metrics, "test", default={}) or {}
        # 兼容你之前 exp3 里可能误写成 test.py 的情况
        if not test and metrics is not None and "test.py" in metrics:
            test = metrics.get("test.py", {}) or {}

        run_name = (
            safe_get(metrics, "run_name")
            or safe_get(run_info, "run_name")
            or safe_get(meta, "run_tag")
            or run_dir.name
        )

        rank, alpha = extract_rank_alpha(kind, meta, metrics, cfg)
        target_modules = extract_target_modules(kind, meta, metrics, cfg)
        train_cfg = extract_train_cfg(metrics, run_info, cfg)

        row = {
            "path": str(run_dir).replace("\\", "/"),
            "type": kind,
            "run_name": run_name,
            "model_name": extract_model_name(metrics, meta, run_info, cfg),
            "task_type": extract_task_type(metrics, meta, run_info, cfg),
            "seed": extract_seed(metrics, meta, run_info, cfg),
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "val_loss": safe_get(val, "loss"),
            "val_ppl": safe_get(val, "ppl"),
            "test_loss": safe_get(test, "loss"),
            "test_ppl": safe_get(test, "ppl"),
            "best_val_loss": safe_get(metrics, "best_val_loss"),
            "global_step": safe_get(metrics, "global_step"),
            "max_train_steps": train_cfg.get("max_train_steps"),
            "lr": train_cfg.get("lr"),
            "grad_accum_steps": train_cfg.get("grad_accum_steps"),
            "teacher_path": extract_teacher_path(meta, cfg),
        }
        rows.append(row)

    return rows


def print_table(rows: List[Dict[str, Any]], limit: Optional[int] = None):
    if not rows:
        print("No runs found.")
        return

    rows = sorted(
        rows,
        key=lambda r: (
            float("inf") if r["test_ppl"] is None else float(r["test_ppl"]),
            r["path"],
        ),
    )

    if limit is not None:
        rows = rows[:limit]

    header = (
        f"{'type':<10} {'run_name':<22} {'rank':>6} {'alpha':>6} "
        f"{'val_ppl':>10} {'test_ppl':>10} {'best_val':>10} {'gstep':>8} {'mods':<24}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{str(r['type']):<10} "
            f"{str(r['run_name'])[:22]:<22} "
            f"{str(r['rank'] if r['rank'] is not None else '-'):>6} "
            f"{str(r['alpha'] if r['alpha'] is not None else '-'):>6} "
            f"{fmt_float(r['val_ppl']):>10} "
            f"{fmt_float(r['test_ppl']):>10} "
            f"{fmt_float(r['best_val_loss']):>10} "
            f"{str(r['global_step'] if r['global_step'] is not None else '-'):>8} "
            f"{fmt_modules(r['target_modules'])[:24]:<24}"
        )


def print_detail(rows: List[Dict[str, Any]], keyword: str):
    matches = [r for r in rows if keyword in r["path"] or keyword in r["run_name"]]
    if not matches:
        print(f"No matched run for keyword: {keyword}")
        return

    if len(matches) > 1:
        print(f"Matched {len(matches)} runs. Showing all:\n")

    for r in matches:
        print("=" * 100)
        print(f"path           : {r['path']}")
        print(f"type           : {r['type']}")
        print(f"run_name       : {r['run_name']}")
        print(f"model_name     : {r['model_name']}")
        print(f"task_type      : {r['task_type']}")
        print(f"seed           : {r['seed']}")
        print(f"rank           : {r['rank']}")
        print(f"alpha          : {r['alpha']}")
        print(f"target_modules : {fmt_modules(r['target_modules'])}")
        print(f"teacher_path   : {r['teacher_path']}")
        print(f"val_loss       : {fmt_float(r['val_loss'], 6)}")
        print(f"val_ppl        : {fmt_float(r['val_ppl'], 6)}")
        print(f"test_loss      : {fmt_float(r['test_loss'], 6)}")
        print(f"test_ppl       : {fmt_float(r['test_ppl'], 6)}")
        print(f"best_val_loss  : {fmt_float(r['best_val_loss'], 6)}")
        print(f"global_step    : {r['global_step']}")
        print(f"max_train_steps: {r['max_train_steps']}")
        print(f"lr             : {r['lr']}")
        print(f"grad_accum     : {r['grad_accum_steps']}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs/lm", help="Root dir to scan")
    ap.add_argument("--match", type=str, default=None, help="Only keep runs whose path contains this keyword")
    ap.add_argument("--detail", type=str, default=None, help="Show detailed info for matched runs")
    ap.add_argument("--limit", type=int, default=None, help="Limit displayed rows")
    args = ap.parse_args()

    root = Path(args.root)
    rows = collect_runs(root)

    if args.match:
        rows = [r for r in rows if args.match in r["path"] or args.match in r["run_name"]]

    print_table(rows, limit=args.limit)

    if args.detail:
        print()
        print_detail(rows, args.detail)


if __name__ == "__main__":
    main()