import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

scores = pd.read_csv("composite_scores.csv")
scores = scores.sort_values("total", ascending=False).reset_index(drop=True)

n = len(scores)
score_cols = ["tb", "ce", "drv", "mg"]
max_vals = [40, 25, 25, 10]
labels = ["Tumor Burden\n(0–40)", "Clonal Expansion\n(0–25)", "Driver Score\n(0–25)", "Mutation Gain\n(0–10)"]

fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(16, 10),
                                       gridspec_kw={"width_ratios": [3, 0.5]})

data = scores[score_cols].values.astype(float)

norm_data = data / np.array(max_vals)[None, :]
cmap = plt.cm.RdYlGn
cmap.set_bad("#f0f0f0")

im = ax_heat.imshow(norm_data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

# Cell annotations
for i in range(n):
    for j in range(4):
        val = data[i, j]
        color = "white" if norm_data[i, j] > 0.65 else "black"
        ax_heat.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                     color=color, fontweight="bold")

# Row labels (patient IDs)
ax_heat.set_yticks(range(n))
ax_heat.set_yticklabels(scores["patient"].values, fontsize=7, fontfamily="monospace")

# Column labels
ax_heat.set_xticks(range(4))
ax_heat.set_xticklabels(labels, fontsize=9, fontweight="bold")

ax_heat.set_xlabel("")
ax_heat.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax_heat.spines[:].set_visible(False)

# Horizontal separator lines
for i in range(n):
    ax_heat.axhline(i, color="white", linewidth=0.3)

# Bar for total score
totals = scores["total"].values
bar_colors = plt.cm.RdYlGn(np.clip(totals / 100, 0, 1))
ax_bar.barh(range(n), totals, color=bar_colors, edgecolor="white", linewidth=0.3)

for i, t in enumerate(totals):
    color = "white" if t > 60 else "black"
    ax_bar.text(t + 1, i, f"{t:.0f}", va="center", fontsize=7,
                color="#333", fontweight="bold")

ax_bar.set_xlim(0, 115)
ax_bar.set_yticks(range(n))
ax_bar.set_yticklabels([""] * n)
ax_bar.invert_yaxis()
ax_bar.set_xlabel("Total (0–100)", fontsize=10, fontweight="bold")
ax_bar.spines[:].set_visible(False)
ax_bar.tick_params(left=False)

# Colorbar
cbar_ax = fig.add_axes([0.02, 0.415, 0.01, 0.17])
cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
cbar.ax.tick_params(labelsize=7)
cbar.set_label("% of max", fontsize=8)

# Tertile brackets
n_per_tile = n // 3
for idx, (label, y_start) in enumerate([("High", 0), ("Mid", n_per_tile), ("Low", 2 * n_per_tile)]):
    y_center = y_start + n_per_tile / 2 - 0.5 if idx < 2 else y_start + (n - 2 * n_per_tile) / 2 - 0.5
    ax_heat.text(-0.55, y_center, label, ha="center", va="center", fontsize=8,
                 fontweight="bold", rotation=90,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#eee", edgecolor="#ccc"))

fig.suptitle("Component Scores by Patient (sorted by Total Score)", fontsize=14, fontweight="bold", y=0.97)
fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.02)
fig.savefig("component_heatmap.png", dpi=200, bbox_inches="tight")
print(f"Saved component_heatmap.png  |  {n} patients, range {totals.min():.0f}–{totals.max():.0f}")
