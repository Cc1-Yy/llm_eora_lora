# scripts/core/33_exp3_lora_init_compare_lm.py
from __future__ import annotations

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import load_file as safetensors_load_file
except Exception:
    safetensors_load_file = None

# ---- Make imports robust ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_utils import load_base_model_and_tokenizer
from src.data_utils import get_dataloaders
from src.lora_utils import add_lora_to_model, print_trainable_params


# ============================================================
# Basic utils
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Dict[str, Any], path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def _pretty_ar(alpha: float, rank: int) -> str:
    ar = float(alpha) / max(int(rank), 1)
    if abs(ar - round(ar)) < 1e-9:
        return f"{ar:.0f}"
    return f"{ar:.2f}".rstrip("0").rstrip(".")


def _get_run_name(config: Dict[str, Any]) -> str:
    if config.get("run_name"):
        return str(config["run_name"])

    lcfg = config.get("lora", {})
    r = lcfg.get("rank", None)
    a = lcfg.get("alpha", None)

    init_cfg = config.get("lora_init", {}) if isinstance(config.get("lora_init", {}), dict) else {}
    init_mode = str(init_cfg.get("mode", "random")).lower()
    init_tag = "rand" if init_mode == "random" else "eorainit"

    if r is None or a is None:
        return init_tag

    try:
        r = int(r)
        a = float(a)
        ar_str = _pretty_ar(a, r)
        return f"r{r}_ar{ar_str}_{init_tag}"
    except Exception:
        return init_tag


def _make_amp_helpers(device: torch.device, use_amp: bool):
    use_amp = bool(use_amp)
    if device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def _autocast():
            return torch.amp.autocast("cuda", enabled=use_amp)

        return scaler, _autocast

    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def _autocast():
        return torch.autocast("cpu", enabled=False)

    return scaler, _autocast


def _estimate_tokens_per_step(config: Dict[str, Any]) -> int:
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    bs = int(data_cfg.get("batch_size", 1))
    max_len = int(data_cfg.get("max_length", 128))
    acc = int(train_cfg.get("grad_accum_steps", 1))
    acc = max(1, acc)
    return bs * max_len * acc


def _get_train_knobs(config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = config.get("train", {})
    out = dict(train_cfg) if isinstance(train_cfg, dict) else {}

    out["log_every_steps"] = int(train_cfg.get("log_every_steps", train_cfg.get("log_every", 50)))
    out["eval_every_steps"] = int(train_cfg.get("eval_every_steps", train_cfg.get("eval_every", 1000)))

    out["grad_accum_steps"] = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    out["num_epochs"] = max(1, int(train_cfg.get("num_epochs", 1)))

    mts = train_cfg.get("max_train_steps", None)
    out["max_train_steps"] = int(mts) if mts is not None else None

    out["warmup_ratio"] = float(train_cfg.get("warmup_ratio", 0.0))
    out["use_amp"] = bool(train_cfg.get("use_amp", True))
    out["lr"] = float(train_cfg.get("lr", 2e-5))
    out["weight_decay"] = float(train_cfg.get("weight_decay", 0.0))
    out["grad_clip"] = float(train_cfg.get("grad_clip", 1.0))
    out["scheduler"] = str(train_cfg.get("scheduler", "linear")).lower()

    return out


# ============================================================
# Quantized / wrapper helpers
# ============================================================

def _is_gptq_quantized_dir(path: str) -> bool:
    if not isinstance(path, str):
        return False
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "quantize_config.json")):
        return True
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        qcfg = obj.get("quantization_config", {})
        return isinstance(qcfg, dict) and qcfg.get("quant_method") == "gptq"
    except Exception:
        return False


def _resolve_quantized_model_dir(config: Dict[str, Any]) -> Optional[str]:
    qdir = config.get("quantized_model_dir", None)
    if qdir:
        return str(qdir)
    model_name = config.get("model_name", None)
    if model_name and _is_gptq_quantized_dir(model_name):
        return str(model_name)
    return None


def load_quantized_backbone_for_compare(config: Dict[str, Any], device: torch.device):
    """
    For the init-comparison experiment, bypass the transformers+optimum GPTQ
    loading path when model_name points to a local GPTQ checkpoint, because that
    path can fail in some environments (e.g. QuantizeConfig NameError).

    Instead, load the quantized model with GPTQModel directly and load tokenizer
    from tokenizer_name (or model_name as fallback).
    """
    model_name = config.get("model_name") or config.get("base_model_ckpt")
    if not model_name:
        raise ValueError("config['model_name'] (or base_model_ckpt) is required.")

    tokenizer_name = config.get("tokenizer_name", model_name)

    if _is_gptq_quantized_dir(model_name):
        from gptqmodel import GPTQModel

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        dev = "cuda:0" if device.type == "cuda" else "cpu"
        model = GPTQModel.from_quantized(model_name, device=dev)

        # Set pad token id on inner config when possible
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "config"):
            if getattr(inner.config, "pad_token_id", None) is None:
                inner.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    # non-quantized fallback
    return load_base_model_and_tokenizer(config)


def _unwrap_trainable_backbone(model_or_wrapper):
    if isinstance(model_or_wrapper, torch.nn.Module):
        inner = getattr(model_or_wrapper, "model", None)
        if isinstance(inner, torch.nn.Module):
            print("[Load] Using inner `.model` as LoRA trainable backbone.")
            return inner
        return model_or_wrapper

    inner = getattr(model_or_wrapper, "model", None)
    if isinstance(inner, torch.nn.Module):
        print("[Load] Using wrapper `.model` as LoRA trainable backbone.")
        return inner

    raise TypeError(
        f"[33_exp3_lora_init_compare_lm] Loaded model is not a trainable nn.Module. "
        f"type={type(model_or_wrapper)}"
    )


def _disable_use_cache(model):
    for obj in [model, getattr(model, "model", None)]:
        if obj is None:
            continue
        if hasattr(obj, "config") and getattr(obj.config, "use_cache", None) is True:
            obj.config.use_cache = False


def disable_triton_dequant_if_present(model):
    n = 0
    for _, m in model.named_modules():
        if hasattr(m, "_triton_dequant_enabled"):
            try:
                m._triton_dequant_enabled = False
                n += 1
            except Exception:
                pass
    print(f"[Patch] Disabled Triton dequant on {n} modules.")


# ============================================================
# Eval helpers
# ============================================================

@torch.no_grad()
def evaluate_causal_lm(model: torch.nn.Module, dataloader, device: torch.device) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_valid_tokens = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError("[33_exp3_lora_init_compare_lm] LM eval requires labels in the batch.")

        outputs = model(**batch)
        loss = outputs.loss

        valid = (labels[:, 1:] != -100).float()
        n_valid = float(valid.sum().item())
        if n_valid <= 0:
            continue

        total_loss += float(loss.item()) * n_valid
        total_valid_tokens += n_valid

    avg_loss = total_loss / max(total_valid_tokens, 1.0)
    ppl = float(math.exp(min(avg_loss, 20.0)))

    return {
        "loss": float(avg_loss),
        "ppl": ppl,
        "valid_tokens": float(total_valid_tokens),
    }


@torch.no_grad()
def eval_with_teacher_lm(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    dataloader,
    device: torch.device,
    temperature: float = 1.0,
) -> Dict[str, float]:
    student.eval()
    teacher.eval()

    T = max(float(temperature), 1e-6)

    total_valid_tokens = 0.0
    kl_sum = 0.0
    mse_sum = 0.0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.get("labels", None)
        if labels is None:
            raise ValueError(
                "[33_exp3_lora_init_compare_lm] batch missing labels; cannot compute teacher-alignment metrics."
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
# EoRA -> LoRA init helpers
# ============================================================

def _normalize_adapter_key(k: str) -> str:
    k = k.replace(".lora_A.default.weight", ".lora_A.weight")
    k = k.replace(".lora_B.default.weight", ".lora_B.weight")
    return k


def _candidate_module_names(module_name: str) -> List[str]:
    names = [module_name]
    prefixes = ["base_model.model.", "base_model.", "model."]

    for p in prefixes:
        if module_name.startswith(p):
            names.append(module_name[len(p):])

    for p in prefixes:
        names.append(p + module_name)

    out = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _load_adapter_state(adapter_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load adapter weights from an adapter directory.

    Supported cases:
      1) Standard PEFT names:
         - adapter_model.safetensors
         - adapter_model.bin
      2) GPTQModel / custom names:
         - any *.safetensors / *.bin / *.pt under adapter_dir (recursive),
           excluding obvious config/metadata files.

    Preference order:
      - exact standard filenames
      - files with names containing adapter / lora / eora
      - then any remaining candidate
    """
    adapter_dir = str(adapter_dir)
    root = Path(adapter_dir)

    if not root.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    # ---- 1) exact standard filenames first ----
    exact_sf = root / "adapter_model.safetensors"
    exact_bin = root / "adapter_model.bin"

    if exact_sf.exists():
        if safetensors_load_file is None:
            raise ImportError(
                "safetensors is required to load adapter_model.safetensors. "
                "Please ensure safetensors is installed."
            )
        print(f"[LoRA EoRA-init] loading adapter weights from: {exact_sf}")
        return safetensors_load_file(str(exact_sf))

    if exact_bin.exists():
        print(f"[LoRA EoRA-init] loading adapter weights from: {exact_bin}")
        obj = torch.load(str(exact_bin), map_location="cpu")
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Unexpected adapter_model.bin content type: {type(obj)}")

    # ---- 2) recursive fuzzy search ----
    ignore_names = {
        "adapter_config.json",
        "config.json",
        "generation_config.json",
        "quantize_config.json",
        "quant_meta.json",
        "meta.json",
        "metrics.json",
        "run_info.json",
        "config_used.json",
    }

    candidates: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        name = p.name.lower()
        suffix = p.suffix.lower()

        if name in ignore_names:
            continue

        if suffix in {".safetensors", ".bin", ".pt"}:
            candidates.append(p)

    if not candidates:
        # surface directory contents to make debugging easier
        sample = []
        try:
            for p in root.rglob("*"):
                sample.append(str(p.relative_to(root)).replace("\\", "/"))
                if len(sample) >= 30:
                    break
        except Exception:
            pass

        raise FileNotFoundError(
            f"No adapter weight file found under {adapter_dir}. "
            f"Searched for *.safetensors / *.bin / *.pt recursively. "
            f"Sample contents: {sample}"
        )

    def score(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        # lower score = higher priority
        pri = 10
        if name == "adapter_model.safetensors":
            pri = 0
        elif name == "adapter_model.bin":
            pri = 1
        elif "adapter" in name:
            pri = 2
        elif "eora" in name:
            pri = 3
        elif "lora" in name:
            pri = 4
        return (pri, len(name), name)

    candidates = sorted(candidates, key=score)
    chosen = candidates[0]
    print(f"[LoRA EoRA-init] loading adapter weights from discovered file: {chosen}")

    suffix = chosen.suffix.lower()
    if suffix == ".safetensors":
        if safetensors_load_file is None:
            raise ImportError(
                "safetensors is required to load adapter weights from a .safetensors file. "
                "Please ensure safetensors is installed."
            )
        return safetensors_load_file(str(chosen))

    obj = torch.load(str(chosen), map_location="cpu")
    if isinstance(obj, dict):
        return obj

    raise ValueError(f"Unexpected adapter state content type in {chosen}: {type(obj)}")


def initialize_lora_from_eora_adapter(
    model: torch.nn.Module,
    adapter_dir: str,
    strict: bool = True,
) -> Dict[str, Any]:
    src_state = _load_adapter_state(adapter_dir)
    src_norm = {_normalize_adapter_key(k): v for k, v in src_state.items()}

    copied = 0
    skipped = []
    shape_mismatch = []

    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        if "default" not in module.lora_A or "default" not in module.lora_B:
            skipped.append((module_name, "missing default adapter slot"))
            continue

        tgt_A = module.lora_A["default"].weight
        tgt_B = module.lora_B["default"].weight

        src_A = None
        src_B = None
        src_A_key = None
        src_B_key = None

        for cand in _candidate_module_names(module_name):
            kA = f"{cand}.lora_A.weight"
            kB = f"{cand}.lora_B.weight"
            if kA in src_norm and kB in src_norm:
                src_A = src_norm[kA]
                src_B = src_norm[kB]
                src_A_key = kA
                src_B_key = kB
                break

        if src_A is None or src_B is None:
            skipped.append((module_name, "no matching source key"))
            continue

        if tuple(src_A.shape) != tuple(tgt_A.shape) or tuple(src_B.shape) != tuple(tgt_B.shape):
            shape_mismatch.append({
                "module": module_name,
                "src_A_key": src_A_key,
                "src_B_key": src_B_key,
                "src_A_shape": tuple(src_A.shape),
                "src_B_shape": tuple(src_B.shape),
                "tgt_A_shape": tuple(tgt_A.shape),
                "tgt_B_shape": tuple(tgt_B.shape),
            })
            continue

        tgt_A.data.copy_(src_A.to(device=tgt_A.device, dtype=tgt_A.dtype))
        tgt_B.data.copy_(src_B.to(device=tgt_B.device, dtype=tgt_B.dtype))
        copied += 1

    if strict and (len(skipped) > 0 or len(shape_mismatch) > 0 or copied == 0):
        msg = {
            "copied": copied,
            "skipped": skipped[:10],
            "shape_mismatch": shape_mismatch[:10],
        }
        raise RuntimeError(f"[LoRA EoRA-init] strict copy failed: {json.dumps(msg, ensure_ascii=False, indent=2)}")

    print(f"[LoRA EoRA-init] copied {copied} modules from {adapter_dir}")
    if skipped:
        print(f"[LoRA EoRA-init] skipped {len(skipped)} modules (showing up to 10): {skipped[:10]}")
    if shape_mismatch:
        print(
            f"[LoRA EoRA-init] shape mismatch on {len(shape_mismatch)} modules "
            f"(showing up to 5): {shape_mismatch[:5]}"
        )

    return {
        "copied_modules": copied,
        "num_skipped": len(skipped),
        "num_shape_mismatch": len(shape_mismatch),
        "skipped_preview": skipped[:10],
        "shape_mismatch_preview": shape_mismatch[:5],
    }


# ============================================================
# History logging helpers
# ============================================================

def build_eval_record(
    tag: str,
    epoch: int,
    global_step: int,
    split_name: str,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    rec = {
        "tag": tag,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "split": split_name,
    }
    for k, v in metrics.items():
        rec[k] = v
    return rec


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    normalize_task_type(config)

    if config.get("task_type") != "causal_lm":
        raise ValueError(
            f"[33_exp3_lora_init_compare_lm] This script is for causal LM only. "
            f"Got: {config.get('task_type')}"
        )

    seed = int(config.get("seed", 42))
    set_seed(seed)
    maybe_set_speed_knobs()

    quantized_model_dir = _resolve_quantized_model_dir(config)
    optimized_model_dir = config.get("optimized_model_dir") or config.get("teacher_model_dir")
    if not optimized_model_dir:
        raise ValueError("[33_exp3_lora_init_compare_lm] Config must provide optimized_model_dir (teacher).")

    init_cfg = config.get("lora_init", {}) if isinstance(config.get("lora_init", {}), dict) else {}
    init_mode = str(init_cfg.get("mode", "random")).lower()
    eora_adapter_dir = init_cfg.get("adapter_dir", None)
    init_strict = bool(init_cfg.get("strict", True))
    eval_before_train = bool(init_cfg.get("evaluate_before_train", True))

    if init_mode not in {"random", "eora_adapter"}:
        raise ValueError(f"Unsupported lora_init.mode: {init_mode}")

    if init_mode == "eora_adapter" and not eora_adapter_dir:
        raise ValueError("lora_init.mode='eora_adapter' requires lora_init.adapter_dir")

    base_output_dir = str(config.get("output_dir", "outputs/lm/exp3/lora_init_compare"))
    ensure_dir(base_output_dir)

    runs_dir = os.path.join(base_output_dir, "runs")
    ensure_dir(runs_dir)

    run_name = _get_run_name(config)
    run_dir = os.path.join(runs_dir, run_name)
    ensure_dir(run_dir)

    history_path = os.path.join(run_dir, "eval_history.jsonl")
    if os.path.exists(history_path):
        os.remove(history_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)
    print(f"[Run] base_output_dir={base_output_dir}")
    print(f"[Run] run_name={run_name}")
    print(f"[Run] run_dir={run_dir}")
    print(f"[Run] tokens/step(est)={_estimate_tokens_per_step(config)}")
    print(f"[Run] quantized_model_dir={quantized_model_dir}")
    print(f"[Run] optimized_model_dir={optimized_model_dir}")
    print(f"[Init] mode={init_mode}")
    if eora_adapter_dir:
        print(f"[Init] eora_adapter_dir={eora_adapter_dir}")

    # --------------------------------------------------------
    # 1) Load quantized backbone + tokenizer
    # --------------------------------------------------------
    loaded_model, tokenizer = load_quantized_backbone_for_compare(config, device)
    maybe_set_tokenizer_pad(tokenizer)

    base_model = _unwrap_trainable_backbone(loaded_model)
    _disable_use_cache(base_model)
    base_model.to(device)

    # --------------------------------------------------------
    # 2) Dataloaders
    # --------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    try:
        print("train batches:", len(train_loader))
    except Exception:
        print("train batches: (iterable/unknown)")
    try:
        print("val batches:", len(val_loader))
        print("test batches:", len(test_loader))
    except Exception:
        pass

    # --------------------------------------------------------
    # 3) Add LoRA on top of quantized backbone
    # --------------------------------------------------------
    model = add_lora_to_model(base_model, config)
    _disable_use_cache(model)
    model.to(device)
    disable_triton_dequant_if_present(model)

    print_trainable_params(model)

    init_report = {
        "mode": init_mode,
        "adapter_dir": eora_adapter_dir,
        "strict": init_strict,
    }

    if init_mode == "eora_adapter":
        copy_report = initialize_lora_from_eora_adapter(
            model=model,
            adapter_dir=eora_adapter_dir,
            strict=init_strict,
        )
        init_report.update(copy_report)

    # --------------------------------------------------------
    # 4) Teacher for alignment eval
    # --------------------------------------------------------
    teacher = AutoModelForCausalLM.from_pretrained(optimized_model_dir)
    teacher.to(device)
    teacher.eval()
    _disable_use_cache(teacher)

    # --------------------------------------------------------
    # 5) Train config
    # --------------------------------------------------------
    train_cfg = _get_train_knobs(config)

    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    num_epochs = int(train_cfg["num_epochs"])
    grad_clip = float(train_cfg["grad_clip"])

    use_amp = bool(train_cfg["use_amp"])
    grad_accum_steps = int(train_cfg["grad_accum_steps"])

    max_train_steps = train_cfg["max_train_steps"]
    warmup_ratio = float(train_cfg["warmup_ratio"])
    scheduler_type = str(train_cfg["scheduler"]).lower()

    log_every = int(train_cfg["log_every_steps"])
    eval_every = int(train_cfg["eval_every_steps"])

    temperature = float(
        config.get("distill", {}).get(
            "temperature",
            config.get("kd", {}).get("T", 1.0)
        )
    )
    select_best_by = str(config.get("select_best_by", "loss")).lower()

    optim_params = [p for p in model.parameters() if p.requires_grad]
    if len(optim_params) == 0:
        raise ValueError("[33_exp3_lora_init_compare_lm] No trainable parameters found after adding LoRA.")
    optimizer = AdamW(optim_params, lr=lr, weight_decay=weight_decay)

    if max_train_steps is not None:
        total_steps = int(max_train_steps)
        stop_mode = f"step-based stop at max_train_steps={max_train_steps}"
    else:
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            raise ValueError(
                "[33_exp3_lora_init_compare_lm] train_loader has no len (likely streaming). "
                "Please set train.max_train_steps in config."
            )
        opt_steps_per_epoch = (steps_per_epoch + grad_accum_steps - 1) // grad_accum_steps
        total_steps = int(opt_steps_per_epoch * max(num_epochs, 1))
        stop_mode = f"epoch-based stop (total_steps~{total_steps})"

    total_steps = max(1, int(total_steps))
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    if scheduler_type not in ["linear", ""]:
        print(f"[Warn] scheduler='{scheduler_type}' not implemented; using linear warmup-decay.")
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler, autocast = _make_amp_helpers(device, use_amp)

    print(
        f"[Train] task=causal_lm {stop_mode} "
        f"total_steps={total_steps} warmup_steps={warmup_steps} "
        f"grad_accum={grad_accum_steps} lr={lr:g} wd={weight_decay:g}"
    )
    print(f"[SelectBest] metric={select_best_by} temperature={temperature:g}")

    # --------------------------------------------------------
    # 6) Optional step-0 eval
    # --------------------------------------------------------
    init_metrics = None
    if eval_before_train:
        init_val_task = evaluate_causal_lm(model, val_loader, device=device)
        init_test_task = evaluate_causal_lm(model, test_loader, device=device)
        init_val_align = eval_with_teacher_lm(model, teacher, val_loader, device=device, temperature=temperature)
        init_test_align = eval_with_teacher_lm(model, teacher, test_loader, device=device, temperature=temperature)

        init_metrics = {
            "val": {**init_val_task, **init_val_align},
            "test": {**init_test_task, **init_test_align},
        }
        save_json(init_metrics, os.path.join(run_dir, "init_metrics.json"))

        append_jsonl(build_eval_record("Init", 0, 0, "val", init_metrics["val"]), history_path)
        append_jsonl(build_eval_record("Init", 0, 0, "test", init_metrics["test"]), history_path)

        print("[InitEval] val:", init_metrics["val"])
        print("[InitEval] test:", init_metrics["test"])

    # --------------------------------------------------------
    # 7) Train loop
    # --------------------------------------------------------
    best_score = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    global_step = 0
    micro_step = 0
    stopped_by_steps = False

    adapter_dir = os.path.join(run_dir, "adapter")
    adapter_best_dir = os.path.join(run_dir, "adapter_best")
    ensure_dir(adapter_dir)
    ensure_dir(adapter_best_dir)

    optimizer.zero_grad(set_to_none=True)

    def _eval_and_maybe_best(tag: str, epoch: int):
        nonlocal best_score, best_state

        val_task = evaluate_causal_lm(model, val_loader, device=device)
        val_align = eval_with_teacher_lm(
            model, teacher, val_loader, device=device, temperature=temperature
        )
        val_metrics = {**val_task, **val_align}

        test_task = evaluate_causal_lm(model, test_loader, device=device)
        test_align = eval_with_teacher_lm(
            model, teacher, test_loader, device=device, temperature=temperature
        )
        test_metrics = {**test_task, **test_align}

        append_jsonl(build_eval_record(tag, epoch, global_step, "val", val_metrics), history_path)
        append_jsonl(build_eval_record(tag, epoch, global_step, "test", test_metrics), history_path)

        if select_best_by == "kl_to_teacher":
            score = float(val_metrics.get("kl_to_teacher", 1e9))
        else:
            score = float(val_metrics.get("loss", 1e9))

        print(f"\n[{tag}] val={val_metrics}")
        print(f"[{tag}] test={test_metrics}")

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            try:
                model.save_pretrained(adapter_best_dir)
                tokenizer.save_pretrained(adapter_best_dir)
                print(f"[Best] adapter_best saved. best_{select_best_by}={best_score:.6f}")
            except Exception as e:
                print(f"[Best] save adapter_best failed: {e}")

        return val_metrics, test_metrics

    epoch = 0
    while True:
        epoch += 1
        model.train()

        if (max_train_steps is None) and (epoch > num_epochs):
            break

        loop = tqdm(train_loader, desc=f"[LoRA InitCmp-LM] Epoch {epoch}", leave=True)

        ep_loss_sum = 0.0
        ep_count = 0

        for batch in loop:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
                loss_to_backward = loss / float(grad_accum_steps)

            scaler.scale(loss_to_backward).backward()
            micro_step += 1

            bs = int(batch["input_ids"].size(0))
            ep_loss_sum += float(loss.item()) * bs
            ep_count += bs

            if micro_step % grad_accum_steps == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

                if log_every > 0 and (global_step % log_every == 0):
                    lr_now = float(scheduler.get_last_lr()[0])
                    loop.set_postfix(loss=float(loss.item()), lr=lr_now)

                if eval_every > 0 and (global_step % eval_every == 0):
                    _eval_and_maybe_best(f"Eval@step {global_step}", epoch)

                if (max_train_steps is not None) and (global_step >= max_train_steps):
                    stopped_by_steps = True
                    break

        ep_train_loss = ep_loss_sum / max(ep_count, 1)
        print(f"Epoch {epoch}: train_loss={ep_train_loss:.4f} (global_step={global_step})")

        _eval_and_maybe_best(f"EpochEnd {epoch}", epoch)

        if stopped_by_steps:
            break

        if (max_train_steps is None) and (epoch >= num_epochs):
            break

        if ep_count == 0:
            print("[Warn] Empty epoch (no batches). Stopping.")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------------------------------------------------
    # 8) Final eval
    # --------------------------------------------------------
    val_task = evaluate_causal_lm(model, val_loader, device=device)
    test_task = evaluate_causal_lm(model, test_loader, device=device)

    val_align = eval_with_teacher_lm(
        model, teacher, val_loader, device=device, temperature=temperature
    )
    test_align = eval_with_teacher_lm(
        model, teacher, test_loader, device=device, temperature=temperature
    )

    val_metrics = {**val_task, **val_align}
    test_metrics = {**test_task, **test_align}

    append_jsonl(build_eval_record("FinalBestRestored", epoch, global_step, "val", val_metrics), history_path)
    append_jsonl(build_eval_record("FinalBestRestored", epoch, global_step, "test", test_metrics), history_path)

    # save final adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    meta = {
        "seed": seed,
        "task_type": config.get("task_type"),
        "model_name": config.get("model_name"),
        "tokenizer_name": config.get("tokenizer_name"),
        "optimized_model_dir": optimized_model_dir,
        "quantized_model_dir": quantized_model_dir,
        "lora": config.get("lora", {}),
        "lora_init": init_cfg,
        "init_report": init_report,
        "train": config.get("train", {}),
        "data": config.get("data", {}),
        "distill": config.get("distill", {}),
        "select_best_by": select_best_by,
        "best_score": best_score,
        "best_metric": select_best_by,
        "global_step": global_step,
        "total_scheduler_steps": total_steps,
        "warmup_steps": warmup_steps,
        "grad_accum_steps": grad_accum_steps,
        "stopped_by_max_train_steps": bool(stopped_by_steps),
        "run_name": run_name,
        "run_dir": run_dir,
        "adapter_dir": adapter_dir,
        "adapter_best_dir": adapter_best_dir,
        "eval_history_path": history_path,
        "notes": [
            "This run compares LoRA initialization strategies on a quantized GPTQ backbone.",
            "Supported init modes: random / eora_adapter.",
            "When using eora_adapter, only overlapping LoRA modules are copied.",
            "Task metrics are token-weighted loss/ppl; alignment metrics are KL and logits-MSE to the optimized teacher.",
            "Triton dequant is disabled after loading the LoRA model when available, to avoid unstable 3-bit eval paths.",
            "Each evaluation event is appended to eval_history.jsonl for convergence comparison.",
        ],
    }
    save_json(meta, os.path.join(run_dir, "meta.json"))

    metrics = {
        "init": init_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    save_json(config, os.path.join(run_dir, "config_used.json"))

    print("=== LoRA Init Comparison LM done ===")
    if init_metrics is not None:
        print("Init val metrics:", init_metrics["val"])
        print("Init test metrics:", init_metrics["test"])
    print("Final val metrics:", val_metrics)
    print("Final test metrics:", test_metrics)
    print("Saved to:", run_dir)
    print("Adapter saved to:", adapter_dir)
    print("Best adapter saved to:", adapter_best_dir)
    print("Eval history saved to:", history_path)


if __name__ == "__main__":
    main()