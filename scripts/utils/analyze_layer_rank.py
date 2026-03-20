import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM

BASE_MODEL = "gpt2"
TEACHER = "outputs/optimized_lm_small/model"

TARGET = ["c_attn","c_proj","c_fc"]

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER)

rows = []

for (n1,p1),(n2,p2) in zip(base.named_parameters(),
                           teacher.named_parameters()):

    if not any(t in n1 for t in TARGET):
        continue

    delta = (p2 - p1).detach().cpu()

    if delta.ndim != 2:
        continue

    s = torch.linalg.svdvals(delta)

    energy = (s**2)

    total = energy.sum()

    for r in [8,16,32,64,128]:

        frac = energy[:r].sum()/total

        rows.append({
            "layer": n1,
            "rank": r,
            "energy_frac": float(frac)
        })

df = pd.DataFrame(rows)

df.to_csv("outputs/layer_rank_energy.csv",index=False)

print("Saved outputs/layer_rank_energy.csv")