import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cs = pd.read_csv("composite_scores.csv").sort_values("total", ascending=False)
fig, ax = plt.subplots(figsize=(14, 18))
colors = {"tb": "#1f77b4", "ce": "#ff7f0e", "drv": "#2ca02c", "mg": "#d62728"}
y = np.arange(len(cs))
for i, comp in enumerate(["tb", "ce", "drv", "mg"]):
    left = cs.iloc[:, 1:1+i].sum(axis=1) if i > 0 else np.zeros(len(cs))
    ax.barh(y, cs[comp], 0.6, left=left, color=colors[comp], label=comp)
ax.set_yticks(y)
ax.set_yticklabels(cs["patient"])
ax.set_xlabel("Score")
ax.set_title("Composite Score Components (final calibrated)")
ax.legend(loc="lower right")
ax.invert_yaxis()
for idx, row in cs.iterrows():
    pos = list(cs.index).index(idx)
    ax.text(row["total"] + 1, pos, f"{row['total']:.0f}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("component_heatmap.png", dpi=150)
plt.close()
print("Saved component_heatmap.png")
