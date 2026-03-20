import os
import json
import csv

BASE = "outputs"

ranks = [32,64,128,256]

rows = []

for r in ranks:

    path = f"{BASE}/eora_rank{r}/r{r}_ar1/metrics.json"

    if not os.path.exists(path):
        continue

    with open(path) as f:
        m = json.load(f)

    rows.append({
        "rank": r,
        "val_ppl": m["val"]["ppl"],
        "test_ppl": m["test"]["ppl"]
    })

with open("outputs/rank_scaling_summary.csv","w",newline="") as f:

    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Saved outputs/rank_scaling_summary.csv")