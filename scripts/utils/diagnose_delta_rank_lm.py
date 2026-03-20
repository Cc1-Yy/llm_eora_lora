from __future__ import annotations

import os
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_gpt2_conv1d(m: torch.nn.Module) -> bool:
    return m.__class__.__name__ == "Conv1D"


def _to_out_in(weight: torch.Tensor, fan_in_fan_out: bool) -> torch.Tensor:
    # Conv1D stores [in, out]; treat as [out, in] by transpose.
    return weight.T if fan_in_fan_out else weight


def _collect_target_layers(model: torch.nn.Module, target_modules: List[str]) -> List[Tuple[str, torch.nn.Module]]:
    """
    Collect modules whose last name equals one of target_modules and has .weight
    Example names:
      transformer.h.0.attn.c_attn
      transformer.h.0.attn.c_proj
      transformer.h.0.mlp.c_fc
      transformer.h.0.mlp.c_proj
      lm_head
    """
    tset = set(target_modules)
    out = []
    for name, mod in model.named_modules():
        last = name.split(".")[-1]
        if last in tset and hasattr(mod, "weight") and getattr(mod, "weight") is not None:
            out.append((name, mod))
    return out


@torch.no_grad()
def _svdvals(delta_oi: torch.Tensor) -> torch.Tensor:
    # CPU SVD is more stable for repeated calls
    x = delta_oi.detach().float().cpu()
    # singular values only
    s = torch.linalg.svdvals(x)
    return s


def _energy_fractions(s: torch.Tensor, ranks: List[int]) -> Dict[int, float]:
    # energy ~ sum(s^2)
    e = (s * s)
    tot = float(e.sum().item()) + 1e-12
    cs = torch.cumsum(e, dim=0)
    out = {}
    for r in ranks:
        rr = min(int(r), int(s.numel()))
        out[r] = float(cs[rr - 1].item() / tot) if rr > 0 else 0.0
    return out


def _residual_ratio(s: torch.Tensor, r: int) -> float:
    # Frobenius residual ratio of best rank-r approx:
    # ||ΔW - ΔW_r||_F / ||ΔW||_F = sqrt(sum_{i>r} s_i^2 / sum_i s_i^2)
    e = (s * s)
    tot = float(e.sum().item()) + 1e-12
    rr = min(int(r), int(s.numel()))
    tail = float(e[rr:].sum().item())
    return (tail / tot) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="gpt2")
    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="outputs/diagnostics/delta_rank_diag.csv")
    ap.add_argument("--target_modules", type=str, default="c_attn,c_proj,c_fc", help="comma-separated")
    ap.add_argument("--ranks", type=str, default="8,16,32,64,128")
    ap.add_argument("--max_layers", type=int, default=-1, help="optional cap for speed; -1 = all")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]

    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_dir)

    base.eval()
    teacher.eval()

    base_layers = dict(base.named_modules())
    t_layers = _collect_target_layers(teacher, target_modules)

    if args.max_layers > 0:
        t_layers = t_layers[: args.max_layers]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # global aggregation (energy-weighted)
    global_energy = torch.zeros(len(ranks), dtype=torch.float64)
    global_total = 0.0

    for name, tmod in t_layers:
        if name not in base_layers:
            continue
        bmod = base_layers[name]
        if not hasattr(bmod, "weight") or bmod.weight is None:
            continue

        # decide fan_in_fan_out
        fan = _is_gpt2_conv1d(tmod) or _is_gpt2_conv1d(bmod)

        Wt = _to_out_in(tmod.weight.data, fan)
        Wb = _to_out_in(bmod.weight.data, fan)
        dW = (Wt - Wb).contiguous()

        s = _svdvals(dW)
        ef = _energy_fractions(s, ranks)
        rr32 = _residual_ratio(s, 32) if 32 in ranks else _residual_ratio(s, ranks[min(len(ranks)-1, 2)])
        # accumulate global energy (sum s^2 across layers)
        e = (s * s).double()
        global_total += float(e.sum().item())
        cs = torch.cumsum(e, dim=0)
        for i, r in enumerate(ranks):
            rr = min(r, int(s.numel()))
            global_energy[i] += cs[rr - 1] if rr > 0 else 0.0

        row = {
            "layer": name,
            "shape_out_in": f"{Wt.shape[0]}x{Wt.shape[1]}",
            "fan_in_fan_out": int(fan),
            "sv_rank": int(s.numel()),
            "residual_ratio_r32": float(rr32),
        }
        for r in ranks:
            row[f"energy_frac_r{r}"] = ef[r]
            row[f"residual_ratio_r{r}"] = _residual_ratio(s, r)
        rows.append(row)

    # write csv
    keys = ["layer", "shape_out_in", "fan_in_fan_out", "sv_rank"] + \
           [f"energy_frac_r{r}" for r in ranks] + \
           [f"residual_ratio_r{r}" for r in ranks] + \
           ["residual_ratio_r32"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # print summary
    print(f"Saved CSV: {out_path}")
    print(f"Teacher: {args.teacher_dir}")
    print(f"Target modules: {target_modules}")
    print(f"Layers analyzed: {len(rows)}")

    if global_total > 0:
        print("\n--- Global (energy-weighted across layers) energy coverage ---")
        for i, r in enumerate(ranks):
            frac = float(global_energy[i].item() / global_total)
            print(f"  r={r}: energy_frac={frac:.4f}")


if __name__ == "__main__":
    main()