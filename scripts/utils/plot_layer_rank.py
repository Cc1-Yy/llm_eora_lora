import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/layer_rank_energy.csv")

ranks = [8,16,32,64,128]

for r in ranks:

    sub = df[df.rank==r]

    plt.plot(sub.energy_frac.values,label=f"r={r}")

plt.legend()
plt.xlabel("layer")
plt.ylabel("energy coverage")

plt.savefig("outputs/layer_rank_energy.png")

print("Saved figure")