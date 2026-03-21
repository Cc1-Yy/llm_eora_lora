# scripts/core/02_train_quantized_lm.py
from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional

import yaml
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from gptqmodel import GPTQModel, QuantizeConfig


# ------------------------------------------------------------
# Robust project-root import behavior
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_task_type(config: Dict[str, Any]) -> None:
    if str(config.get("task_type", "")).lower() == "lm":
        config["task_type"] = "causal_lm"


def _get_split_from_loaded(ds_obj: Any, split: str):
    """
    Works for:
      - DatasetDict loaded from disk / HF
      - plain Dataset
    """
    if isinstance(ds_obj, DatasetDict):
        if split not in ds_obj:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(ds_obj.keys())}")
        return ds_obj[split]
    if isinstance(ds_obj, Dataset):
        return ds_obj
    raise TypeError(f"Unsupported dataset object type: {type(ds_obj)}")


def _load_text_dataset_from_cfg(corpus_cfg: Dict[str, Any]):
    """
    Supports two styles:

    1) Local HF dataset saved by save_to_disk:
       local_disk_path: data_cache/wikitext2_raw
       split: train
       text_key: text

    2) HF dataset:
       dataset_name: wikitext
       dataset_config_name: wikitext-2-raw-v1
       split: train
       text_key: text

       or shorthand:
       dataset_name: wikitext/wikitext-2-raw-v1
    """
    split = corpus_cfg.get("split", "train")
    text_key = corpus_cfg.get("text_key", "text")

    local_disk_path = corpus_cfg.get("local_disk_path")
    if local_disk_path:
        ds_obj = load_from_disk(local_disk_path)
        ds = _get_split_from_loaded(ds_obj, split)
        return ds, text_key

    dataset_name = corpus_cfg.get("dataset_name")
    if not dataset_name:
        raise ValueError(
            "Calibration corpus config must provide either "
            "`local_disk_path` or `dataset_name`."
        )

    dataset_config_name = corpus_cfg.get("dataset_config_name", None)

    # Support shorthand "name/config"
    if dataset_config_name is None and "/" in dataset_name:
        name0, name1 = dataset_name.split("/", 1)
        dataset_name, dataset_config_name = name0, name1

    if dataset_config_name is not None:
        ds = load_dataset(dataset_name, dataset_config_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    return ds, text_key


def _collect_calibration_texts(
    ds,
    text_key: str,
    max_samples: int,
    seed: int = 42,
    min_chars: int = 2,
) -> List[str]:
    """
    Prepare plain text samples for GPTQ calibration.
    Filters empty / whitespace-only strings and truncates to max_samples.
    """
    if text_key not in ds.column_names:
        raise ValueError(
            f"text_key='{text_key}' not found in dataset columns: {ds.column_names}"
        )

    # Shuffle first so first N are not biased
    try:
        ds = ds.shuffle(seed=seed)
    except Exception:
        pass

    texts: List[str] = []
    upper = min(len(ds), max_samples * 4)  # over-scan a bit in case many empties

    for i in range(upper):
        x = ds[i][text_key]
        if x is None:
            continue
        if not isinstance(x, str):
            x = str(x)
        x = x.strip()
        if len(x) < min_chars:
            continue
        texts.append(x)
        if len(texts) >= max_samples:
            break

    if len(texts) == 0:
        raise ValueError("No valid calibration texts were collected.")

    return texts


def _build_quant_config(config: Dict[str, Any]) -> QuantizeConfig:
    """
    Accept either top-level keys or nested `quantization:` block.

    Example:
      bits: 4
      group_size: 128

    or
      quantization:
        bits: 4
        group_size: 128
        desc_act: false
        damp_percent: 0.1
    """
    qcfg = config.get("quantization", {}) if isinstance(config.get("quantization", {}), dict) else {}

    bits = int(qcfg.get("bits", config.get("bits", 4)))
    group_size = int(qcfg.get("group_size", config.get("group_size", 128)))
    desc_act = bool(qcfg.get("desc_act", config.get("desc_act", False)))

    # Optional GPTQ knobs
    kwargs = {
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
    }

    if "damp_percent" in qcfg or "damp_percent" in config:
        kwargs["damp_percent"] = float(qcfg.get("damp_percent", config.get("damp_percent")))

    if "sym" in qcfg or "sym" in config:
        kwargs["sym"] = bool(qcfg.get("sym", config.get("sym")))

    if "true_sequential" in qcfg or "true_sequential" in config:
        kwargs["true_sequential"] = bool(
            qcfg.get("true_sequential", config.get("true_sequential"))
        )

    return QuantizeConfig(**kwargs)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    _normalize_task_type(config)

    if config.get("task_type") != "causal_lm":
        raise ValueError(
            f"[02_train_quantized_lm] This script is for causal LM only. "
            f"Got task_type={config.get('task_type')}"
        )

    model_id_or_path = config.get("optimized_model_dir", None)
    if not model_id_or_path:
        raise ValueError("Config must provide `optimized_model_dir`.")

    out_dir = config.get("quant_output_dir", None)
    if not out_dir:
        raise ValueError("Config must provide `quant_output_dir`.")

    ensure_dir(out_dir)

    data_cfg = config.get("data", {})
    train_corpus_cfg = data_cfg.get("train_corpus", None)
    if not isinstance(train_corpus_cfg, dict):
        raise ValueError(
            "LM quantization expects `data.train_corpus` in config."
        )

    calibration_num_samples = int(config.get("calibration_num_samples", 512))
    calibration_batch_size = int(config.get("calibration_batch_size", 1))
    calibration_seed = int(config.get("seed", 42))

    # 1) Load calibration corpus
    ds, text_key = _load_text_dataset_from_cfg(train_corpus_cfg)
    calibration_texts = _collect_calibration_texts(
        ds=ds,
        text_key=text_key,
        max_samples=calibration_num_samples,
        seed=calibration_seed,
    )

    print("=== LM Quantization ===")
    print("optimized_model_dir:", model_id_or_path)
    print("quant_output_dir   :", out_dir)
    print("calib split        :", train_corpus_cfg.get("split", "train"))
    print("text_key           :", text_key)
    print("num calib texts    :", len(calibration_texts))
    print("example calib text :", repr(calibration_texts[0][:120]))

    # 2) Build quant config
    quant_config = _build_quant_config(config)
    print("quant_config       :", quant_config)

    # 3) Load and quantize
    model = GPTQModel.load(model_id_or_path, quant_config)
    model.quantize(calibration_texts, batch_size=calibration_batch_size)
    model.save(out_dir)

    # 4) Save meta for bookkeeping
    meta = {
        "task_type": config.get("task_type"),
        "optimized_model_dir": model_id_or_path,
        "quant_output_dir": out_dir,
        "calibration_num_samples": calibration_num_samples,
        "calibration_batch_size": calibration_batch_size,
        "seed": calibration_seed,
        "text_key": text_key,
        "train_corpus": train_corpus_cfg,
        "quantization": config.get("quantization", {}),
    }
    save_json(meta, os.path.join(out_dir, "quant_meta.json"))

    print("Quantized LM model saved to:", out_dir)
    print("Saved meta to:", os.path.join(out_dir, "quant_meta.json"))


if __name__ == "__main__":
    main()