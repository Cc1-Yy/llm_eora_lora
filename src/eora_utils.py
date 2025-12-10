# src/eora_utils.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import os

from datasets import load_dataset

from gptqmodel import GPTQModel
from gptqmodel.adapter.adapter import Lora


def _build_calibration_texts(config: Dict[str, Any]):
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "glue/sst2")
    n = int(config.get("eora", {}).get("calibration_num_samples", 512))

    # 最小可行：先从任务数据里拿文本
    # 你后面可以换 C4/WikiText 做更规范的校准集
    if "/" in dataset_name:
        d0, d1 = dataset_name.split("/", 1)
        ds = load_dataset(d0, d1)
        # SST-2 的文本字段
        text_key = "sentence"
    else:
        ds = load_dataset(dataset_name)
        text_key = "text"

    texts = ds["train"].select(range(n))[text_key]
    return texts


def generate_eora_adapter(config: Dict[str, Any]) -> Lora:
    """
    按 GPTQModel 官方方式：
      - adapter = Lora(path=..., rank=...)
      - GPTQModel.adapter.generate(...)
    """
    eora_cfg = config.get("eora", {})
    rank = int(eora_cfg.get("rank", 8))

    optimized_model_dir = config.get("optimized_model_dir")
    quantized_model_dir = config.get("quantized_model_dir")

    if not optimized_model_dir or not quantized_model_dir:
        raise ValueError(
            "GPTQModel EoRA requires BOTH optimized_model_dir (full precision) "
            "and quantized_model_dir (GPTQ quantized)."
        )

    # 1) 校准文本
    calibration_texts = _build_calibration_texts(config)

    # 2) EoRA adapter 保存路径
    # 注意：官方就是用 Lora adapter 类来承载 EoRA
    save_path = os.path.join(quantized_model_dir, f"eora_rank{rank}")

    eora = Lora(
        path=save_path,
        rank=rank,
    )

    # 3) 生成 EoRA
    GPTQModel.adapter.generate(
        adapter=eora,
        model_id_or_path=optimized_model_dir,
        quantized_model_id_or_path=quantized_model_dir,
        calibration_dataset=calibration_texts,
        calibration_dataset_concat_size=0,
    )

    return eora


def load_quant_with_eora(config: Dict[str, Any], eora: Lora):
    quantized_model_dir = config["quantized_model_dir"]
    model = GPTQModel.load(
        model_id_or_path=quantized_model_dir,
        adapter=eora
    )
    return model
