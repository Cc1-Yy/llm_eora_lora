import numpy as np
import matplotlib.pyplot as plt

cmaps = ['RdYlBu', 'RdYlBu_r']

gradient = np.linspace(0, 1, 256)
gradient = np.vstack([gradient, gradient])

fig, axes = plt.subplots(len(cmaps), 1, figsize=(8, 2.5))

for ax, cmap in zip(axes, cmaps):
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    ax.text(-0.02, 0.5, cmap, va='center', ha='right', transform=ax.transAxes)

plt.tight_layout()
plt.show()