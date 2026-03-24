# scripts/core/32_exp3_eora_lm.py
from __future__ import annotations

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# Robust project-root import behavior
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import get_dataloaders


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_task_type(config: Dict[str, Any]) -> None:
    if str(config.get("task_type", "")).lower() == "lm":
        config["task_type"] = "causal_lm"


def maybe_set_speed_knobs():
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_set_tokenizer_pad(tokenizer):
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _resolve_quantized_model_dir(cfg: Dict[str, Any]) -> str:
    """
    Priority:
      1) quantized_model_dir
      2) model_name  (if user sets model_name to quantized path, like baseline eval config)
    """
    qdir = cfg.get("quantized_model_dir", None)
    if qdir:
        return str(qdir)

    model_name = cfg.get("model_name", None)
    if model_name:
        return str(model_name)

    raise ValueError(
        "[32_exp3_eora_lm] Config must provide either `quantized_model_dir` "
        "or set `model_name` to the quantized GPTQ directory."
    )


def _run_tag_from_cfg(cfg: Dict[str, Any]) -> str:
    if cfg.get("run_tag"):
        return str(cfg["run_tag"])
    if cfg.get("run_name"):
        return str(cfg["run_name"])

    eora_cfg = cfg.get("eora", {})
    rank = _safe_int(eora_cfg.get("rank", 32), 32)
    alpha = float(eora_cfg.get("alpha", rank))
    ar = alpha / max(rank, 1)
    return f"r{rank}_ar{ar:g}"


# ============================================================
# Calibration text helpers
# ============================================================

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


def _load_text_dataset_from_spec(spec: Dict[str, Any]):
    """
    Supports:
      1) local_disk_path + split + text_key
      2) dataset_name + dataset_config_name(optional) + split + text_key
      3) shorthand dataset_name: "name/config"
    """
    split = spec.get("split", "train")
    text_key = spec.get("text_key", "text")

    local_disk_path = spec.get("local_disk_path")
    if local_disk_path:
        ds_obj = load_from_disk(local_disk_path)
        ds = _get_split_from_loaded(ds_obj, split)
        return ds, text_key

    dataset_name = spec.get("dataset_name") or spec.get("name")
    if not dataset_name:
        raise ValueError(
            "Calibration corpus spec must provide either `local_disk_path` or `dataset_name`/`name`."
        )

    dataset_config_name = spec.get("dataset_config_name", spec.get("config", None))

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
    Prepare plain text samples for EoRA/GPTQ calibration.

    Note:
      - This currently assumes a non-streaming / indexable dataset.
      - It uses column_names, len(ds), ds.shuffle(...), ds[i].
    """
    if text_key not in ds.column_names:
        raise ValueError(
            f"text_key='{text_key}' not found in dataset columns: {ds.column_names}"
        )

    try:
        ds = ds.shuffle(seed=seed)
    except Exception:
        pass

    texts: List[str] = []
    upper = min(len(ds), max_samples * 4)  # overscan for empties

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


def _build_calibration_texts_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """
    Priority:
      1) eora.calibration_local_txt
      2) data.train_corpus   (recommended; matches your LM pipeline)
      3) data.dataset_name   (legacy fallback)
    """
    eora_cfg = cfg.get("eora", {}) if isinstance(cfg.get("eora", {}), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}

    n = int(eora_cfg.get("calibration_num_samples", cfg.get("calibration_num_samples", 512)))
    seed = int(cfg.get("seed", 42))

    local_txt = eora_cfg.get("calibration_local_txt", None)
    if local_txt:
        p = Path(local_txt)
        if not p.exists():
            raise FileNotFoundError(f"eora.calibration_local_txt not found: {p}")
        texts: List[str] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    texts.append(s)
                if len(texts) >= n:
                    break
        if not texts:
            raise ValueError(f"eora.calibration_local_txt is empty: {p}")
        return texts

    train_spec = data_cfg.get("train_corpus", None)
    if isinstance(train_spec, dict):
        ds, text_key = _load_text_dataset_from_spec(train_spec)
        return _collect_calibration_texts(
            ds=ds,
            text_key=text_key,
            max_samples=n,
            seed=seed,
        )

    # legacy fallback
    dataset_name = data_cfg.get("dataset_name", None)
    if dataset_name:
        spec = {
            "dataset_name": dataset_name,
            "dataset_config_name": data_cfg.get("dataset_config_name", None),
            "split": data_cfg.get("split", "train"),
            "text_key": data_cfg.get("text_key", "text"),
        }
        ds, text_key = _load_text_dataset_from_spec(spec)
        return _collect_calibration_texts(
            ds=ds,
            text_key=text_key,
            max_samples=n,
            seed=seed,
        )

    raise ValueError(
        "[32_exp3_eora_lm] Cannot build calibration texts. "
        "Provide one of:\n"
        "  - eora.calibration_local_txt\n"
        "  - data.train_corpus\n"
        "  - legacy data.dataset_name"
    )


# ============================================================
# Forward / evaluation helpers
# ============================================================

def _unwrap_forward_model(model_or_wrapper):
    """
    GPTQModel.load(...) may return a wrapper object rather than a raw nn.Module.
    Prefer the underlying .model when available.
    """
    inner = getattr(model_or_wrapper, "model", None)
    if inner is not None:
        return inner
    return model_or_wrapper


def _set_eval_mode(model_or_wrapper):
    if hasattr(model_or_wrapper, "eval"):
        try:
            model_or_wrapper.eval()
        except Exception:
            pass

    inner = getattr(model_or_wrapper, "model", None)
    if inner is not None and hasattr(inner, "eval"):
        try:
            inner.eval()
        except Exception:
            pass


@torch.no_grad()
def evaluate_causal_lm(model_or_wrapper, dataloader, device: str) -> Dict[str, float]:
    """
    LM eval that does not rely on next(model.parameters()).
    Safer for GPTQModel wrappers.

    Important:
      - We compute token-weighted average loss, not example-weighted loss.
      - For causal LM, outputs.loss is already averaged over valid target tokens
        in the batch, so we reweight by the number of valid tokens.
    """
    model = _unwrap_forward_model(model_or_wrapper)
    _set_eval_mode(model_or_wrapper)

    total_loss = 0.0
    total_valid_tokens = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("[32_exp3_eora_lm] LM evaluation requires `labels` in the batch.")

        outputs = model(**batch)
        loss = outputs.loss

        # causal LM valid next-token positions
        valid = (labels[:, 1:] != -100).float()
        n_valid = float(valid.sum().item())

        if n_valid <= 0:
            continue

        total_loss += float(loss.item()) * n_valid
        total_valid_tokens += n_valid

    avg_loss = total_loss / max(total_valid_tokens, 1.0)
    ppl = float(math.exp(min(avg_loss, 20.0)))  # avoid overflow

    return {
        "loss": float(avg_loss),
        "ppl": ppl,
        "valid_tokens": float(total_valid_tokens),
    }


def _set_tensor_like_attr(obj, attr_name: str, new_tensor: torch.Tensor):
    old = getattr(obj, attr_name)
    if isinstance(old, torch.nn.Parameter):
        setattr(obj, attr_name, torch.nn.Parameter(new_tensor, requires_grad=False))
    else:
        setattr(obj, attr_name, new_tensor)


def repair_loaded_gptq_lora_orientations(student_or_wrapper) -> int:
    """
    Fix GPTQModel-loaded LoRA/EoRA adapter tensor orientations so that
    adapter.apply(x, out) can do:

        out + (x @ lora_A) @ lora_B

    which requires:
        lora_A: [in_features, rank]
        lora_B: [rank, out_features]

    This is especially needed for GPT-2 Conv1D-style modules where saved
    adapter tensors may be loaded in a transposed / swapped orientation.
    """
    model = _unwrap_forward_model(student_or_wrapper)

    fixed = 0
    seen = 0

    for name, mod in model.named_modules():
        adapter = getattr(mod, "adapter", None)
        if adapter is None:
            continue

        A = getattr(adapter, "lora_A", None)
        B = getattr(adapter, "lora_B", None)
        if A is None or B is None:
            continue
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            continue
        if A.ndim != 2 or B.ndim != 2:
            continue

        # Try common GPTQ quant-linear attrs first
        in_features = getattr(mod, "in_features", None)
        if in_features is None:
            in_features = getattr(mod, "infeatures", None)

        out_features = getattr(mod, "out_features", None)
        if out_features is None:
            out_features = getattr(mod, "outfeatures", None)

        # If still unavailable, infer from currently loaded adapter shapes.
        # We only act when a repair pattern is unambiguous.
        seen += 1

        A_dev, A_dtype = A.device, A.dtype
        B_dev, B_dtype = B.device, B.dtype

        # Already correct:
        #   A = [in, r], B = [r, out]
        if (
            in_features is not None
            and out_features is not None
            and A.shape[0] == in_features
            and B.shape[1] == out_features
            and A.shape[1] == B.shape[0]
        ):
            continue

        new_A = None
        new_B = None

        # Case 1:
        #   A loaded as [out, r]
        #   B loaded as [r, in]
        # Needed:
        #   new_A = B^T  -> [in, r]
        #   new_B = A^T  -> [r, out]
        if (
            in_features is not None
            and out_features is not None
            and A.shape[0] == out_features
            and B.shape[1] == in_features
            and A.shape[1] == B.shape[0]
        ):
            new_A = B.T.contiguous()
            new_B = A.T.contiguous()

        # Case 2:
        #   A loaded as [r, in]
        #   B loaded as [out, r]
        # Needed:
        #   transpose each
        elif (
            in_features is not None
            and out_features is not None
            and A.shape[1] == in_features
            and B.shape[0] == out_features
            and A.shape[0] == B.shape[1]
        ):
            new_A = A.T.contiguous()
            new_B = B.T.contiguous()

        # Case 3:
        #   A loaded as [r, out]
        #   B loaded as [in, r]
        # Needed:
        #   swap without transpose
        elif (
            in_features is not None
            and out_features is not None
            and A.shape[1] == out_features
            and B.shape[0] == in_features
            and A.shape[0] == B.shape[1]
        ):
            new_A = B.contiguous()
            new_B = A.contiguous()

        if new_A is None or new_B is None:
            print(
                f"[Adapter Repair][Skip] {name}: "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}, "
                f"in={in_features}, out={out_features}"
            )
            continue

        # Final sanity check
        if in_features is not None and out_features is not None:
            if not (
                new_A.shape[0] == in_features
                and new_B.shape[1] == out_features
                and new_A.shape[1] == new_B.shape[0]
            ):
                print(
                    f"[Adapter Repair][BadFix] {name}: "
                    f"new_A={tuple(new_A.shape)}, new_B={tuple(new_B.shape)}, "
                    f"in={in_features}, out={out_features}"
                )
                continue

        new_A = new_A.to(device=A_dev, dtype=A_dtype)
        new_B = new_B.to(device=B_dev, dtype=B_dtype)

        _set_tensor_like_attr(adapter, "lora_A", new_A)
        _set_tensor_like_attr(adapter, "lora_B", new_B)

        print(
            f"[Adapter Repair][Fixed] {name}: "
            f"A {tuple(A.shape)} -> {tuple(new_A.shape)}, "
            f"B {tuple(B.shape)} -> {tuple(new_B.shape)}"
        )
        fixed += 1

    print(f"[Adapter Repair] fixed {fixed} / {seen} adapter modules")
    return fixed


@torch.no_grad()
def eval_with_teacher_lm(
    student_or_wrapper,
    teacher: torch.nn.Module,
    dataloader,
    device: str,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    Teacher-alignment metrics for causal LM:
      - masked KL(teacher || student) on valid next-token positions
      - masked MSE(logits) on valid next-token positions
    """
    student = _unwrap_forward_model(student_or_wrapper)
    _set_eval_mode(student_or_wrapper)
    teacher.eval()

    T = max(float(temperature), 1e-6)

    total_valid_tokens = 0.0
    kl_sum = 0.0
    mse_sum = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError(
                "[32_exp3_eora_lm] batch missing labels; cannot compute teacher-alignment metrics."
            )

        model_inputs = {k: v for k, v in batch.items() if k != "labels"}

        out_s = student(**model_inputs)
        out_t = teacher(**model_inputs)

        logits_s = out_s.logits
        logits_t = out_t.logits

        s = logits_s[:, :-1, :]
        t = logits_t[:, :-1, :]
        y = labels[:, 1:]

        valid = (y != -100)
        valid_f = valid.float()
        n_valid = float(valid_f.sum().item())

        if n_valid <= 0:
            continue

        s_logp = torch.log_softmax(s / T, dim=-1)
        t_prob = torch.softmax(t / T, dim=-1)
        token_kl = torch.sum(
            t_prob * (torch.log(t_prob + 1e-12) - s_logp), dim=-1
        ) * (T * T)

        token_mse = torch.mean((s - t) ** 2, dim=-1)

        kl_sum += float((token_kl * valid_f).sum().item())
        mse_sum += float((token_mse * valid_f).sum().item())
        total_valid_tokens += n_valid

    denom = max(total_valid_tokens, 1.0)
    return {
        "kl_to_teacher": kl_sum / denom,
        "mse_logits_to_teacher": mse_sum / denom,
        "valid_tokens": total_valid_tokens,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    normalize_task_type(cfg)

    if cfg.get("task_type") != "causal_lm":
        raise ValueError(
            f"[32_exp3_eora_lm] This script is for causal LM only. "
            f"Got task_type={cfg.get('task_type')}"
        )

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    maybe_set_speed_knobs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    optimized_model_dir = cfg.get("optimized_model_dir", None)
    if not optimized_model_dir:
        raise ValueError("[32_exp3_eora_lm] Config must provide `optimized_model_dir`.")

    quantized_model_dir = _resolve_quantized_model_dir(cfg)
    tokenizer_name = cfg.get("tokenizer_name", optimized_model_dir)

    # final run directory
    # use cfg.output_dir directly, to stay consistent with your current project style
    output_dir = str(cfg.get("output_dir", os.path.join("outputs", "lm", "exp3", _run_tag_from_cfg(cfg))))
    ensure_dir(output_dir)

    adapter_dir = os.path.join(output_dir, "adapter")
    ensure_dir(adapter_dir)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    maybe_set_tokenizer_pad(tokenizer)

    # dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(cfg, tokenizer)
    try:
        print("train batches:", len(train_loader))
    except Exception:
        print("train batches: (iterable/unknown)")
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # teacher
    teacher = AutoModelForCausalLM.from_pretrained(optimized_model_dir)
    teacher.to(device)
    teacher.eval()
    if hasattr(teacher, "config") and getattr(teacher.config, "use_cache", None) is True:
        teacher.config.use_cache = False

    # EoRA / GPTQModel
    from gptqmodel import GPTQModel
    from gptqmodel.adapter.adapter import Lora

    eora_cfg = cfg.get("eora", {}) if isinstance(cfg.get("eora", {}), dict) else {}
    rank = int(eora_cfg.get("rank", 32))
    alpha = float(eora_cfg.get("alpha", rank))  # kept for bookkeeping / run naming
    concat_size = int(eora_cfg.get("calibration_dataset_concat_size", 0))
    repair_orientation = bool(eora_cfg.get("repair_adapter_orientation", True))
    target_modules = eora_cfg.get("target_modules", None)
    print("requested target_modules:", target_modules)
    calibration_texts = _build_calibration_texts_from_cfg(cfg)

    print("=== Exp3 EoRA-LM (quantized recovery) ===")
    print("optimized_model_dir:", optimized_model_dir)
    print("quantized_model_dir:", quantized_model_dir)
    print("tokenizer_name     :", tokenizer_name)
    print("output_dir         :", output_dir)
    print("adapter_dir        :", adapter_dir)
    print("eora rank          :", rank)
    print("eora alpha         :", alpha)
    print("repair orientation :", repair_orientation)
    print("num calib texts    :", len(calibration_texts))
    print("example calib text :", repr(calibration_texts[0][:120]))

    # 1) create adapter object
    # NOTE:
    # GPTQModel's EoRA flow uses rank directly for adapter generation/loading.
    eora = Lora(path=adapter_dir, rank=rank)

    # 2) generate EoRA adapter from (optimized fp model, quantized model, calibration texts)
    GPTQModel.adapter.generate(
        adapter=eora,
        model_id_or_path=optimized_model_dir,
        quantized_model_id_or_path=quantized_model_dir,
        calibration_dataset=calibration_texts,
        calibration_dataset_concat_size=concat_size,
    )

    # 3) load quantized model + generated EoRA adapter
    student = GPTQModel.load(
        model_id_or_path=quantized_model_dir,
        adapter=eora,
    )
    _set_eval_mode(student)

    inner_student = _unwrap_forward_model(student)
    if hasattr(inner_student, "config") and getattr(inner_student.config, "use_cache", None) is True:
        inner_student.config.use_cache = False

    # ---- repair loaded adapter orientation if requested ----
    repaired_adapter_modules = 0
    if repair_orientation:
        repaired_adapter_modules = repair_loaded_gptq_lora_orientations(student)
    else:
        print("[Adapter Repair] skipped by config (eora.repair_adapter_orientation=false)")

    # save tokenizer with adapter for convenience
    try:
        tokenizer.save_pretrained(adapter_dir)
    except Exception as e:
        print(f"[Warn] Failed to save tokenizer to adapter_dir: {e}")

    # 4) evaluate task metrics
    val_task = evaluate_causal_lm(student, val_loader, device=device)
    test_task = evaluate_causal_lm(student, test_loader, device=device)

    # 5) evaluate teacher-alignment metrics
    temperature = float(
        cfg.get("distill", {}).get(
            "temperature",
            cfg.get("kd", {}).get("T", 1.0)
        )
    )

    val_align = eval_with_teacher_lm(
        student, teacher, val_loader, device=device, temperature=temperature
    )
    test_align = eval_with_teacher_lm(
        student, teacher, test_loader, device=device, temperature=temperature
    )

    val_metrics = {**val_task, **val_align}
    test_metrics = {**test_task, **test_align}

    # 6) save meta + metrics
    meta = {
        "seed": seed,
        "model_name": cfg.get("model_name"),
        "tokenizer_name": tokenizer_name,
        "task_type": cfg.get("task_type"),
        "optimized_model_dir": optimized_model_dir,
        "quantized_model_dir": quantized_model_dir,
        "output_dir": output_dir,
        "adapter_dir": adapter_dir,
        "run_tag": _run_tag_from_cfg(cfg),
        "eora": cfg.get("eora", {}),
        "distill": cfg.get("distill", {}),
        "kd": cfg.get("kd", {}),
        "data": cfg.get("data", {}),
        "calibration_num_samples": len(calibration_texts),
        "calibration_dataset_concat_size": concat_size,
        "repair_adapter_orientation": repair_orientation,
        "repaired_adapter_modules": repaired_adapter_modules,
        "notes": [
            "This run uses GPTQModel EoRA generation on a pre-quantized GPTQ model.",
            "Task metrics are token-weighted loss/ppl; alignment metrics are KL and logits-MSE to the optimized teacher.",
            "alpha is recorded for experiment bookkeeping, but GPTQModel EoRA generation primarily uses rank.",
            "Calibration text collection currently assumes a non-streaming / indexable dataset.",
        ],
    }
    save_json(meta, os.path.join(output_dir, "meta.json"))

    metrics = {
        "seed": seed,
        "model_name": cfg.get("model_name"),
        "tokenizer_name": tokenizer_name,
        "task_type": cfg.get("task_type"),
        "optimized_model_dir": optimized_model_dir,
        "quantized_model_dir": quantized_model_dir,
        "eora": cfg.get("eora", {}),
        "distill": cfg.get("distill", {}),
        "kd": cfg.get("kd", {}),
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(metrics, os.path.join(output_dir, "metrics.json"))

    print("=== Exp3 EoRA-LM done ===")
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("Saved to:", output_dir)
    print("Adapter saved to:", adapter_dir)


if __name__ == "__main__":
    main()