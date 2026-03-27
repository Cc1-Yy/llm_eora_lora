import json
import re
import csv
import math
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EXP0_ROOT = PROJECT_ROOT / "outputs" / "lm" / "exp0"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "outputs" / "lm" / "exp0_summary_lm.csv"


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


def fmt(x, nd: int = 4) -> str:
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


def parse_seed(name: str) -> Optional[int]:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def parse_steps(name: str) -> Optional[int]:
    m = re.search(r"steps(\d+)", name)
    return int(m.group(1)) if m else None


def parse_bits(name: str) -> Optional[int]:
    m = re.search(r"(^|[_\-])(\d+)bit($|[_\-])", name.lower())
    if m:
        return int(m.group(2))

    # fallback: if path/config name says 3bit / 4bit without separators
    m = re.search(r"(\d+)bit", name.lower())
    if m:
        return int(m.group(1))
    return None


def extract_lm_metrics(metrics_json: Dict[str, Any]) -> Dict[str, Any]:
    val = metrics_json.get("val", {}) or {}
    test = metrics_json.get("test", {}) or metrics_json.get("test.py", {}) or {}
    return {
        "val_loss": val.get("loss"),
        "val_ppl": val.get("ppl"),
        "test_loss": test.get("loss"),
        "test_ppl": test.get("ppl"),
    }


def build_row(
    *,
    kind: str,
    exp_name: str,
    run_name: str,
    seed: Optional[int],
    steps: Optional[int],
    bits: Optional[int],
    task_type: Optional[str],
    model_name: Optional[str],
    val_loss: Optional[float],
    val_ppl: Optional[float],
    test_loss: Optional[float],
    test_ppl: Optional[float],
    run_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "kind": kind,
        "exp_name": exp_name,
        "run_name": run_name,
        "seed": seed,
        "steps": steps,
        "bits": bits,
        "task_type": task_type,
        "model_name": model_name,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "run_dir": safe_relpath(run_dir, PROJECT_ROOT),
    }
    if extra:
        row.update(extra)
    return row


def is_under_runs(path: Path) -> bool:
    return "runs" in path.parts


def infer_exp_name_from_rooted_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 0 else path.name


def scan_optimized_runs(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows

    for metrics_path in root.rglob("metrics.json"):
        if not is_under_runs(metrics_path):
            continue

        metrics_json = safe_read_json(metrics_path)
        if not metrics_json:
            continue

        run_dir = metrics_path.parent
        run_name = run_dir.name
        exp_name = infer_exp_name_from_rooted_path(metrics_path, root)

        run_info = safe_read_json(run_dir / "run_info.json") or {}
        cfg_used = (
            safe_read_json(run_dir / "config_used.json")
            or safe_read_json(run_dir / "config_used.yaml")  # will fail safely, kept for symmetry
            or {}
        )
        # config_used.yaml is yaml, not json; safe_read_json will return None, which is fine.

        m = extract_lm_metrics(metrics_json)

        seed = metrics_json.get("seed")
        if seed is None:
            seed = run_info.get("seed")
        if seed is None:
            seed = parse_seed(run_name)
        if seed is None:
            seed = parse_seed(exp_name)

        steps = None
        train_cfg = run_info.get("train", {}) or {}
        if train_cfg.get("max_train_steps") is not None:
            try:
                steps = int(train_cfg.get("max_train_steps"))
            except Exception:
                steps = None
        if steps is None:
            steps = parse_steps(run_name)

        row = build_row(
            kind="Optimized",
            exp_name=exp_name,
            run_name=run_name,
            seed=seed,
            steps=steps,
            bits=None,
            task_type=metrics_json.get("task_type"),
            model_name=metrics_json.get("model_name"),
            val_loss=m["val_loss"],
            val_ppl=m["val_ppl"],
            test_loss=m["test_loss"],
            test_ppl=m["test_ppl"],
            run_dir=run_dir,
            extra={
                "source_metrics": safe_relpath(metrics_path, PROJECT_ROOT),
            },
        )
        rows.append(row)

    return rows


def looks_like_quant_dir(path: Path) -> bool:
    name = str(path).lower().replace("\\", "/")
    return (
        "quant" in name
        or "gptq" in name
        or "3bit" in name
        or "4bit" in name
        or "8bit" in name
    )


def scan_quantized_artifacts(root: Path) -> List[Dict[str, Any]]:
    """
    Heuristic scan:
    - skip anything under runs/
    - look for directories that appear quant-related
    - optionally read metrics.json if present
    """
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows

    seen = set()

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if is_under_runs(path):
            continue
        if not looks_like_quant_dir(path):
            continue

        # avoid duplicate nested reporting:
        # only keep dirs that look like actual artifacts
        has_modelish_file = any(
            (path / fn).exists()
            for fn in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "quantize_config.json",
                "generation_config.json",
            ]
        )
        has_metrics = (path / "metrics.json").exists()
        if not has_modelish_file and not has_metrics:
            continue

        norm = str(path.resolve()).replace("\\", "/")
        if norm in seen:
            continue
        seen.add(norm)

        metrics_json = safe_read_json(path / "metrics.json") or {}
        meta_json = safe_read_json(path / "meta.json") or {}

        m = extract_lm_metrics(metrics_json) if metrics_json else {
            "val_loss": None,
            "val_ppl": None,
            "test_loss": None,
            "test_ppl": None,
        }

        exp_name = infer_exp_name_from_rooted_path(path, root)
        run_name = path.name
        bits = parse_bits(run_name)
        if bits is None:
            bits = parse_bits(exp_name)
        if bits is None:
            bits = parse_bits(norm)

        row = build_row(
            kind="Quantized",
            exp_name=exp_name,
            run_name=run_name,
            seed=meta_json.get("seed") or metrics_json.get("seed"),
            steps=None,
            bits=bits,
            task_type=meta_json.get("task_type") or metrics_json.get("task_type"),
            model_name=meta_json.get("model_name") or metrics_json.get("model_name"),
            val_loss=m["val_loss"],
            val_ppl=m["val_ppl"],
            test_loss=m["test_loss"],
            test_ppl=m["test_ppl"],
            run_dir=path,
            extra={
                "source_metrics": safe_relpath(path / "metrics.json", PROJECT_ROOT) if has_metrics else None,
                "optimized_model_dir": meta_json.get("optimized_model_dir") or metrics_json.get("optimized_model_dir"),
                "quantized_model_dir": meta_json.get("quantized_model_dir"),
            },
        )
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "kind",
        "exp_name",
        "run_name",
        "seed",
        "steps",
        "bits",
        "task_type",
        "model_name",
        "val_loss",
        "val_ppl",
        "test_loss",
        "test_ppl",
        "optimized_model_dir",
        "quantized_model_dir",
        "source_metrics",
        "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_console_summary(rows: List[Dict[str, Any]], root: Path):
    optimized = [r for r in rows if r["kind"] == "Optimized"]
    quantized = [r for r in rows if r["kind"] == "Quantized"]

    def best_teacher(rs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ok = [r for r in rs if r["test_ppl"] is not None]
        if not ok:
            return None
        return sorted(ok, key=lambda x: x["test_ppl"])[0]

    print("\n================ EXP0 LM SUMMARY ================\n")
    print(f"Scan dir: {root}")
    print(f"Found entries: Optimized={len(optimized)}, Quantized={len(quantized)}")

    bt = best_teacher(optimized)
    if bt:
        print(
            f"[BEST Optimized]  {bt['exp_name']}  run={bt['run_name']}  "
            f"seed={bt['seed']}  steps={bt['steps']}  "
            f"val_loss={fmt(bt['val_loss'])}  val_ppl={fmt(bt['val_ppl'])}  "
            f"test_loss={fmt(bt['test_loss'])}  test_ppl={fmt(bt['test_ppl'])}"
        )
    else:
        print("[BEST Optimized]  NA")

    print("\n--- Aggregated Optimized by exp_name ---")
    grouped = defaultdict(list)
    for r in optimized:
        grouped[r["exp_name"]].append(r)

    keys = sorted(grouped.keys())
    if not keys:
        print("NA")
    else:
        for k in keys:
            rs = grouped[k]
            mu_test_ppl, sd_test_ppl = mean_std([r["test_ppl"] for r in rs])
            mu_test_loss, sd_test_loss = mean_std([r["test_loss"] for r in rs])
            print(
                f"{k:24s}  n={len(rs)}  "
                f"test_loss={fmt(mu_test_loss)}±{fmt(sd_test_loss)}  "
                f"test_ppl={fmt(mu_test_ppl)}±{fmt(sd_test_ppl)}"
            )

    print("\n--- Quantized artifacts ---")
    if not quantized:
        print("NA")
    else:
        quantized_sorted = sorted(
            quantized,
            key=lambda x: (
                x["bits"] if x["bits"] is not None else 10**9,
                x["exp_name"],
                x["run_name"],
            ),
        )
        for r in quantized_sorted:
            print(
                f"{r['exp_name']:24s}  run={r['run_name']}  bits={r['bits']}  "
                f"test_loss={fmt(r['test_loss'])}  test_ppl={fmt(r['test_ppl'])}  "
                f"path={r['run_dir']}"
            )

    print("\n=================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(DEFAULT_EXP0_ROOT))
    ap.add_argument("--out_csv", type=str, default=str(DEFAULT_SUMMARY_CSV))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_csv = Path(args.out_csv).resolve()

    rows = []
    rows.extend(scan_optimized_runs(root))
    rows.extend(scan_quantized_artifacts(root))

    def sort_key(r: Dict[str, Any]):
        kind_order = {"Optimized": 0, "Quantized": 1}
        return (
            kind_order.get(r["kind"], 99),
            r["exp_name"],
            r["steps"] if r["steps"] is not None else 10**9,
            r["bits"] if r["bits"] is not None else 10**9,
            r["seed"] if r["seed"] is not None else 10**9,
            r["run_name"],
        )

    rows = sorted(rows, key=sort_key)
    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")
    print_console_summary(rows, root)


if __name__ == "__main__":
    main()