# scripts/utils/summarize_exp2_lm.py
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
DEFAULT_EXP2_ROOT = PROJECT_ROOT / "outputs" / "lm" / "exp2"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "lm" / "exp2_summary.csv"

# 默认给目录，不强制给具体 metrics.json
# 脚本会自动在这个目录下递归寻找 metrics.json，并优先选 val_loss 最低的 teacher
DEFAULT_TEACHER_SOURCE = PROJECT_ROOT / "outputs" / "lm" / "exp0" / "optimized_lm_small"


# ============================================================
# Basic IO helpers
# ============================================================

def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# ============================================================
# Parsing helpers
# ============================================================

def infer_branch(exp_name: str) -> Optional[str]:
    name = exp_name.lower()
    if "lora" in name:
        return "LoRA-KD" if "kd" in name else "LoRA"
    if "eora" in name:
        return "EoRA"
    return None


def get_exp_name_from_metrics_path(metrics_path: Path, exp_root: Path) -> str:
    rel = metrics_path.relative_to(exp_root)
    return rel.parts[0]


def parse_phase(exp_name: str) -> Optional[str]:
    """
    Examples:
      0_baseline_xxx   -> 0
      1_xxx            -> 1
      1a_xxx           -> 1a
      1b_xxx           -> 1b
      2_xxx            -> 2
      2a_xxx           -> 2a
      2b_xxx           -> 2b
      3_xxx            -> 3
    """
    m = re.match(r"^(\d+[a-z]?)_", exp_name.lower())
    if not m:
        return None
    return m.group(1)


def parse_r_ar(name: str) -> Tuple[Optional[int], Optional[float]]:
    m = re.search(r"_r(\d+)_ar(\d+(\.\d+)?)", name)
    if not m:
        m = re.search(r"r(\d+)_ar(\d+(\.\d+)?)", name)
    if not m:
        return None, None
    return int(m.group(1)), float(m.group(2))


def parse_seed(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name.lower())
    if not m:
        return None
    return int(m.group(1))


def infer_target_tag_from_name(name: str) -> Optional[str]:
    m = re.search(r"(tm-[a-z0-9]+)", name.lower())
    if not m:
        return None
    return m.group(1)


# ============================================================
# LM metrics extraction
# ============================================================

def extract_metrics_lm(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", {}) or metrics_json.get("test.py", {}) or {}

    return {
        "val_loss": val.get("loss"),
        "val_ppl": val.get("ppl"),
        "val_kl_to_teacher": val.get("kl_to_teacher"),
        "val_mse_logits_to_teacher": val.get("mse_logits_to_teacher"),
        "val_valid_tokens": val.get("valid_tokens"),
        "test_loss": test.get("loss"),
        "test_ppl": test.get("ppl"),
        "test_kl_to_teacher": test.get("kl_to_teacher"),
        "test_mse_logits_to_teacher": test.get("mse_logits_to_teacher"),
        "test_valid_tokens": test.get("valid_tokens"),
    }


def resolve_teacher_metrics(source: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Accept either:
      - a metrics.json file
      - or a directory containing multiple metrics.json files

    If a directory is given, choose the candidate with the lowest val_loss.
    """
    if source.is_file():
        js = safe_read_json(source)
        return (js or {}), source if js else None

    if not source.exists():
        return {}, None

    candidates = list(source.rglob("metrics.json"))
    if not candidates:
        return {}, None

    scored: List[Tuple[float, float, Path, Dict[str, Any]]] = []
    for p in candidates:
        js = safe_read_json(p)
        if not js:
            continue
        val = js.get("val", {}) or {}
        val_loss = val.get("loss")
        try:
            score = float(val_loss)
        except Exception:
            score = float("inf")
        mtime = p.stat().st_mtime
        scored.append((score, -mtime, p, js))

    if not scored:
        return {}, None

    scored.sort(key=lambda x: (x[0], x[1]))
    _, _, best_path, best_json = scored[0]
    return best_json, best_path


def load_teacher_metrics(source: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    js, used_path = resolve_teacher_metrics(source)
    if not js:
        return {}, None

    val = js.get("val", {}) or {}
    test = js.get("test", {}) or js.get("test.py", {}) or {}

    out = {
        "teacher_val_loss": val.get("loss"),
        "teacher_val_ppl": val.get("ppl"),
        "teacher_test_loss": test.get("loss"),
        "teacher_test_ppl": test.get("ppl"),
        "teacher_seed": js.get("seed"),
        "teacher_model_name": js.get("model_name"),
        "teacher_task_type": js.get("task_type"),
    }
    return out, used_path


# ============================================================
# Metadata extraction
# ============================================================

def maybe_extract_cfg(
    metrics_json: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]],
    cfg_key: str,
) -> Dict[str, Any]:
    d = metrics_json.get(cfg_key)
    if isinstance(d, dict) and d:
        return d
    if meta_json:
        d = meta_json.get(cfg_key)
        if isinstance(d, dict) and d:
            return d
    return {}


def maybe_extract_alpha(
    metrics_json: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]],
    rank: Optional[int],
    alpha_over_r: Optional[float],
    branch: Optional[str],
) -> Optional[float]:
    cfg_key = None
    if branch in {"LoRA", "LoRA-KD"}:
        cfg_key = "lora"
    elif branch == "EoRA":
        cfg_key = "eora"

    if cfg_key is not None:
        cfg = maybe_extract_cfg(metrics_json, meta_json, cfg_key)
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


def maybe_extract_target_modules(
    metrics_json: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]],
    branch: Optional[str],
) -> Optional[str]:
    cfg_key = None
    if branch in {"LoRA", "LoRA-KD"}:
        cfg_key = "lora"
    elif branch == "EoRA":
        cfg_key = "eora"

    if cfg_key is None:
        return None

    cfg = maybe_extract_cfg(metrics_json, meta_json, cfg_key)
    mods = cfg.get("target_modules")

    if isinstance(mods, list):
        try:
            return ",".join(str(x) for x in mods)
        except Exception:
            pass
    if isinstance(mods, str):
        return mods

    return None


def maybe_extract_train_fields(
    metrics_json: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    train_cfg = metrics_json.get("train", {}) or {}
    if not train_cfg and meta_json:
        train_cfg = meta_json.get("train", {}) or {}

    return {
        "best_val_loss": metrics_json.get("best_val_loss"),
        "global_step": metrics_json.get("global_step"),
        "max_train_steps": metrics_json.get("max_train_steps", train_cfg.get("max_train_steps")),
        "lr": train_cfg.get("lr"),
        "grad_accum": metrics_json.get("grad_accum_steps", train_cfg.get("grad_accum_steps")),
    }


def maybe_extract_teacher_path(
    metrics_json: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]],
) -> Optional[str]:
    for k in ["teacher_model_dir", "optimized_model_dir"]:
        v = metrics_json.get(k)
        if v is not None:
            return v
        if meta_json:
            v = meta_json.get(k)
            if v is not None:
                return v
    return None


def maybe_extract_kd_fields(
    metrics_json: Dict[str, Any],
    meta_json: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    kd_cfg = metrics_json.get("kd", {}) or {}
    if not kd_cfg and meta_json:
        kd_cfg = meta_json.get("kd", {}) or {}

    return {
        "kd_T": kd_cfg.get("T"),
        "kd_lambda": kd_cfg.get("lambda"),
        "kd_sup_lambda": kd_cfg.get("sup_lambda"),
        "kd_loss_type": kd_cfg.get("loss"),
    }


# ============================================================
# Main scanning
# ============================================================

def scan_exp2_lm(
    root: Path,
    teacher_test_loss: Optional[float],
    teacher_test_ppl: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
        meta_json = safe_read_json(run_dir / "meta.json")

        r, ar = parse_r_ar(exp_name)
        if r is None or ar is None:
            r, ar = parse_r_ar(run_name)

        seed = metrics_json.get("seed")
        if seed is None and meta_json:
            seed = meta_json.get("seed")
        if seed is None:
            seed = parse_seed(exp_name)
        if seed is None:
            seed = parse_seed(run_name)

        m = extract_metrics_lm(metrics_json)
        alpha = maybe_extract_alpha(metrics_json, meta_json, r, ar, branch)
        target_modules = maybe_extract_target_modules(metrics_json, meta_json, branch)
        if target_modules is None:
            target_modules = infer_target_tag_from_name(exp_name)

        train_fields = maybe_extract_train_fields(metrics_json, meta_json)
        kd_fields = maybe_extract_kd_fields(metrics_json, meta_json)

        model_name = metrics_json.get("model_name")
        if model_name is None and meta_json:
            model_name = meta_json.get("model_name")

        task_type = metrics_json.get("task_type")
        if task_type is None and meta_json:
            task_type = meta_json.get("task_type")

        teacher_path = maybe_extract_teacher_path(metrics_json, meta_json)

        test_loss_minus_teacher = None
        test_ppl_minus_teacher = None
        if teacher_test_loss is not None and m["test_loss"] is not None:
            test_loss_minus_teacher = float(m["test_loss"]) - float(teacher_test_loss)
        if teacher_test_ppl is not None and m["test_ppl"] is not None:
            test_ppl_minus_teacher = float(m["test_ppl"]) - float(teacher_test_ppl)

        row = {
            "phase": parse_phase(exp_name),
            "branch": branch,
            "exp_name": exp_name,
            "run_name": run_name,
            "seed": seed,
            "rank": r,
            "alpha": alpha,
            "alpha_over_r": ar,
            "target_modules": target_modules,
            "val_loss": m["val_loss"],
            "val_ppl": m["val_ppl"],
            "val_kl_to_teacher": m["val_kl_to_teacher"],
            "val_mse_logits_to_teacher": m["val_mse_logits_to_teacher"],
            "val_valid_tokens": m["val_valid_tokens"],
            "test_loss": m["test_loss"],
            "test_ppl": m["test_ppl"],
            "test_kl_to_teacher": m["test_kl_to_teacher"],
            "test_mse_logits_to_teacher": m["test_mse_logits_to_teacher"],
            "test_valid_tokens": m["test_valid_tokens"],
            "teacher_test_loss": teacher_test_loss,
            "teacher_test_ppl": teacher_test_ppl,
            "test_loss_minus_teacher": test_loss_minus_teacher,
            "test_ppl_minus_teacher": test_ppl_minus_teacher,
            "best_val_loss": train_fields["best_val_loss"],
            "global_step": train_fields["global_step"],
            "max_train_steps": train_fields["max_train_steps"],
            "lr": train_fields["lr"],
            "grad_accum": train_fields["grad_accum"],
            "kd_T": kd_fields["kd_T"],
            "kd_lambda": kd_fields["kd_lambda"],
            "kd_sup_lambda": kd_fields["kd_sup_lambda"],
            "kd_loss_type": kd_fields["kd_loss_type"],
            "task_type": task_type,
            "model_name": model_name,
            "teacher_path": teacher_path,
            "run_dir": safe_relpath(run_dir, PROJECT_ROOT),
        }
        rows.append(row)

    return rows


def build_teacher_row(teacher_metrics: Dict[str, Any], teacher_metrics_path: Path) -> Optional[Dict[str, Any]]:
    if not teacher_metrics:
        return None
    if teacher_metrics.get("teacher_test_loss") is None:
        return None

    return {
        "phase": "teacher",
        "branch": "Teacher",
        "exp_name": "optimized_model",
        "run_name": "optimized_model",
        "seed": teacher_metrics.get("teacher_seed"),
        "rank": None,
        "alpha": None,
        "alpha_over_r": None,
        "target_modules": None,
        "val_loss": teacher_metrics.get("teacher_val_loss"),
        "val_ppl": teacher_metrics.get("teacher_val_ppl"),
        "val_kl_to_teacher": None,
        "val_mse_logits_to_teacher": None,
        "val_valid_tokens": None,
        "test_loss": teacher_metrics.get("teacher_test_loss"),
        "test_ppl": teacher_metrics.get("teacher_test_ppl"),
        "test_kl_to_teacher": None,
        "test_mse_logits_to_teacher": None,
        "test_valid_tokens": None,
        "teacher_test_loss": teacher_metrics.get("teacher_test_loss"),
        "teacher_test_ppl": teacher_metrics.get("teacher_test_ppl"),
        "test_loss_minus_teacher": 0.0,
        "test_ppl_minus_teacher": 0.0,
        "best_val_loss": None,
        "global_step": None,
        "max_train_steps": None,
        "lr": None,
        "grad_accum": None,
        "kd_T": None,
        "kd_lambda": None,
        "kd_sup_lambda": None,
        "kd_loss_type": None,
        "task_type": teacher_metrics.get("teacher_task_type"),
        "model_name": teacher_metrics.get("teacher_model_name"),
        "teacher_path": None,
        "run_dir": safe_relpath(teacher_metrics_path.parent, PROJECT_ROOT),
    }


# ============================================================
# CSV writing
# ============================================================

def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "phase",
        "branch",
        "exp_name",
        "run_name",
        "seed",
        "rank",
        "alpha",
        "alpha_over_r",
        "target_modules",
        "val_loss",
        "val_ppl",
        "val_kl_to_teacher",
        "val_mse_logits_to_teacher",
        "val_valid_tokens",
        "test_loss",
        "test_ppl",
        "test_kl_to_teacher",
        "test_mse_logits_to_teacher",
        "test_valid_tokens",
        "teacher_test_loss",
        "teacher_test_ppl",
        "test_loss_minus_teacher",
        "test_ppl_minus_teacher",
        "best_val_loss",
        "global_step",
        "max_train_steps",
        "lr",
        "grad_accum",
        "kd_T",
        "kd_lambda",
        "kd_sup_lambda",
        "kd_loss_type",
        "task_type",
        "model_name",
        "teacher_path",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Console summary
# ============================================================

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


def print_console_summary(
    rows: List[Dict[str, Any]],
    teacher_test_loss: Optional[float],
    teacher_test_ppl: Optional[float],
    root: Path,
    teacher_metrics_used: Optional[Path],
):
    exp_rows = [r for r in rows if r["branch"] in {"LoRA", "LoRA-KD", "EoRA"}]
    lora = [r for r in exp_rows if r["branch"] in {"LoRA", "LoRA-KD"}]
    eora = [r for r in exp_rows if r["branch"] == "EoRA"]

    def best_of(branch_rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ok = [r for r in branch_rows if r["test_kl_to_teacher"] is not None]
        if ok:
            return sorted(ok, key=lambda x: x["test_kl_to_teacher"])[0]
        ok = [r for r in branch_rows if r["test_loss"] is not None]
        if not ok:
            return None
        return sorted(ok, key=lambda x: x["test_loss"])[0]

    print("\n================ EXP2 LM SUMMARY ================\n")
    print(f"Scan dir: {root}")
    print(f"Found runs: LoRA-family={len(lora)}, EoRA={len(eora)}")
    print(f"Teacher metrics source: {teacher_metrics_used if teacher_metrics_used else 'NA'}")
    print(f"Teacher test_loss: {fmt(teacher_test_loss)}")
    print(f"Teacher test_ppl : {fmt(teacher_test_ppl)}\n")

    bL = best_of(lora)
    bE = best_of(eora)

    if bL:
        print(
            f"[BEST LoRA/KD]  {bL['exp_name']}  seed={bL['seed']}  rank={bL['rank']}  "
            f"alpha={fmt(bL['alpha'])}  ar={bL['alpha_over_r']}  "
            f"target={bL['target_modules']}  "
            f"test_loss={fmt(bL['test_loss'])}  test_ppl={fmt(bL['test_ppl'])}  "
            f"test_kl_to_teacher={fmt(bL['test_kl_to_teacher'])}  "
            f"test_mse_logits_to_teacher={fmt(bL['test_mse_logits_to_teacher'])}"
        )
    else:
        print("[BEST LoRA/KD]  NA")

    if bE:
        print(
            f"[BEST EoRA]      {bE['exp_name']}  seed={bE['seed']}  rank={bE['rank']}  "
            f"alpha={fmt(bE['alpha'])}  ar={bE['alpha_over_r']}  "
            f"target={bE['target_modules']}  "
            f"test_loss={fmt(bE['test_loss'])}  test_ppl={fmt(bE['test_ppl'])}  "
            f"test_kl_to_teacher={fmt(bE['test_kl_to_teacher'])}  "
            f"test_mse_logits_to_teacher={fmt(bE['test_mse_logits_to_teacher'])}"
        )
    else:
        print("[BEST EoRA]      NA")

    print("\n--- Best by target_modules ---")
    by_target = defaultdict(list)
    for r in exp_rows:
        by_target[(r["branch"], r["target_modules"])].append(r)

    for (branch, target), rs in sorted(by_target.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        ok = [r for r in rs if r["test_kl_to_teacher"] is not None]
        if ok:
            best = sorted(ok, key=lambda x: x["test_kl_to_teacher"])[0]
        else:
            ok = [r for r in rs if r["test_loss"] is not None]
            if not ok:
                continue
            best = sorted(ok, key=lambda x: x["test_loss"])[0]

        print(
            f"{branch:7s}  target={target}  "
            f"best={best['exp_name']}  seed={best['seed']}  "
            f"test_loss={fmt(best['test_loss'])}  "
            f"test_kl_to_teacher={fmt(best['test_kl_to_teacher'])}"
        )

    grouped = defaultdict(list)
    for r in exp_rows:
        grouped[(r["branch"], r["target_modules"], r["rank"], r["alpha_over_r"])].append(r)

    print("\n--- Aggregated by (branch, target_modules, rank, alpha/r) ---")
    keys = sorted(
        grouped.keys(),
        key=lambda x: (
            x[0],
            str(x[1]),
            x[2] if x[2] is not None else 10**9,
            x[3] if x[3] is not None else 10**9,
        ),
    )

    for k in keys:
        rs = grouped[k]
        mu_loss, sd_loss = mean_std([r["test_loss"] for r in rs])
        mu_ppl, sd_ppl = mean_std([r["test_ppl"] for r in rs])
        mu_kl, sd_kl = mean_std([r["test_kl_to_teacher"] for r in rs])
        mu_mse, sd_mse = mean_std([r["test_mse_logits_to_teacher"] for r in rs])
        mu_gap, sd_gap = mean_std([r["test_loss_minus_teacher"] for r in rs])

        print(
            f"{k[0]:7s}  target={k[1]}  r={k[2]}  ar={k[3]}  "
            f"n={len(rs)}  "
            f"test_loss={fmt(mu_loss)}±{fmt(sd_loss)}  "
            f"test_ppl={fmt(mu_ppl)}±{fmt(sd_ppl)}  "
            f"test_kl={fmt(mu_kl)}±{fmt(sd_kl)}  "
            f"test_mse={fmt(mu_mse)}±{fmt(sd_mse)}  "
            f"loss_gap_to_teacher={fmt(mu_gap)}±{fmt(sd_gap)}"
        )

    print("\n================================================\n")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP2_ROOT))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    ap.add_argument(
        "--teacher_metrics_source",
        type=str,
        default=str(DEFAULT_TEACHER_SOURCE),
        help="Either a metrics.json file or a directory to recursively search for metrics.json",
    )
    ap.add_argument("--include_teacher_row", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()
    teacher_source = Path(args.teacher_metrics_source).resolve()

    teacher_metrics, teacher_metrics_used = load_teacher_metrics(teacher_source)
    teacher_test_loss = teacher_metrics.get("teacher_test_loss")
    teacher_test_ppl = teacher_metrics.get("teacher_test_ppl")

    rows = scan_exp2_lm(root, teacher_test_loss, teacher_test_ppl)

    if args.include_teacher_row and teacher_metrics_used is not None:
        teacher_row = build_teacher_row(teacher_metrics, teacher_metrics_used)
        if teacher_row is not None:
            rows.append(teacher_row)

    def sort_key(r):
        branch_order = {"Teacher": 0, "LoRA": 1, "LoRA-KD": 2, "EoRA": 3}
        phase = r["phase"] if r["phase"] is not None else ""
        rk = r["rank"] if r["rank"] is not None else -1
        ar = r["alpha_over_r"] if r["alpha_over_r"] is not None else -1
        sd = r["seed"] if r["seed"] is not None else -1
        target = r["target_modules"] if r["target_modules"] is not None else ""
        return (
            branch_order.get(r["branch"], 99),
            phase,
            target,
            rk,
            ar,
            sd,
            r["exp_name"],
            r["run_name"],
        )

    rows = sorted(rows, key=sort_key)
    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")
    print_console_summary(rows, teacher_test_loss, teacher_test_ppl, root, teacher_metrics_used)


if __name__ == "__main__":
    main()