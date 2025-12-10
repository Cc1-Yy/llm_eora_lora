# scripts/0_quantize_optimized.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    model_id_or_path = config["optimized_model_dir"]
    out_dir = config.get("quant_output_dir", "outputs/optimized_gptq_4bit")

    # 这份校准数据只需要是“文本列表”
    # 你现在用 SST-2 也能先跑通；后面建议换更标准的 LM 校准语料
    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "glue/sst2")
    max_calib = int(config.get("calibration_num_samples", 512))

    # 兼容 glue/sst2 写法
    if "/" in dataset_name:
        d0, d1 = dataset_name.split("/", 1)
        ds = load_dataset(d0, d1)
        text_key = "sentence"
    else:
        ds = load_dataset(dataset_name)
        text_key = "text"

    calibration_dataset = ds["train"].select(range(max_calib))[text_key]

    quant_config = QuantizeConfig(bits=4, group_size=128)

    model = GPTQModel.load(model_id_or_path, quant_config)
    model.quantize(calibration_dataset, batch_size=1)
    model.save(out_dir)

    print("Quantized model saved to:", out_dir)


if __name__ == "__main__":
    main()
