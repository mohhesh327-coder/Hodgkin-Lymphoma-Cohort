import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cs = pd.read_csv("composite_scores.csv")

COMPONENTS = ["tb", "ce", "drv", "mg"]
MAXES = {"tb": 40, "ce": 25, "drv": 25, "mg": 10}
TIER_BOUNDS = [(95, "High-confidence"), (55, "Strong recurrence"),
               (30, "Possible"), (10, "Weak"), (-np.inf, "None")]
TIER_ORDER = ["High-confidence", "Strong recurrence", "Possible", "Weak", "None"]

def tier(total):
    for bound, name in TIER_BOUNDS:
        if total >= bound:
            return name
    return "None"

ref_tier = cs["total"].apply(tier)

# --- Figure 1: Sensitivity Analysis ---
perturbations = {
    "Remove": lambda s, mx: s * 0,
    "Halve":  lambda s, mx: (s * 0.5).clip(upper=mx),
    "Double": lambda s, mx: (s * 2.0).clip(upper=mx),
    "Max out":lambda s, mx: pd.Series(mx, index=s.index),
}

results = {p: [] for p in perturbations}
for comp in COMPONENTS:
    mx = MAXES[comp]
    for pname, pfunc in perturbations.items():
        perturbed = cs.copy()
        perturbed[comp] = pfunc(perturbed[comp], mx).round(2)
        new_total = perturbed[COMPONENTS].sum(axis=1).round(2)
        new_tier = new_total.apply(tier)
        n_changed = (new_tier != ref_tier).sum()
        results[pname].append(n_changed)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
x = np.arange(len(COMPONENTS))
width = 0.18
colors = {"Remove": "#e74c3c", "Halve": "#f39c12", "Double": "#3498db", "Max out": "#2ecc71"}
for i, (pname, vals) in enumerate(results.items()):
    bars = ax.bar(x + i * width - 1.5 * width, vals, width, label=pname, color=colors[pname], edgecolor="gray", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(v),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([c.upper() for c in COMPONENTS], fontsize=12)
ax.set_ylabel("Patients changing tier", fontsize=12)
ax.set_title("Sensitivity: Tier Changes Under Component Perturbations", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, max(max(v) for v in results.values()) + 5)

# --- Figure 2: Component-Contribution Chart ---
ax = axes[1]
cs_sorted = cs.sort_values("total", ascending=True).reset_index(drop=True)
y = np.arange(len(cs_sorted))
comp_colors = {"tb": "#1f77b4", "ce": "#ff7f0e", "drv": "#2ca02c", "mg": "#d62728"}
cumul = np.zeros(len(cs_sorted))
for comp in COMPONENTS:
    vals = cs_sorted[comp].values
    ax.barh(y, vals, 0.7, left=cumul, color=comp_colors[comp], label=comp.upper(), edgecolor="white", linewidth=0.3)
    cumul += vals
ax.set_yticks(y)
ax.set_yticklabels(cs_sorted["patient"], fontsize=8)
ax.set_xlabel("Score", fontsize=12)
ax.set_title("Component Contribution by Patient", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.invert_yaxis()
# tier separator lines and labels
tier_colors = {"High-confidence": "#27ae60", "Strong recurrence": "#2980b9",
               "Possible": "#f39c12", "Weak": "#e67e22", "None": "#e74c3c"}
tier_bg = {"High-confidence": "#d5f5e3", "Strong recurrence": "#d6eaf8",
           "Possible": "#fef9e7", "Weak": "#fdebd0", "None": "#fadbd8"}
for bound, tname in TIER_BOUNDS:
    if tname == "None":
        continue
    # find the first patient (from top, which is bottom of chart) whose total >= bound
    idx = cs_sorted["total"].searchsorted(bound)
    if idx < len(cs_sorted):
        ax.axhline(y=idx - 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.text(cs_sorted["total"].max() * 0.75, idx - 0.5, f"  {tname} (≥{bound})",
                fontsize=7, color="gray", va="bottom")
# total labels
for idx, row in cs_sorted.iterrows():
    ax.text(row["total"] + 0.5, idx, f"{row['total']:.0f}", va="center", fontsize=6, color="gray")

plt.tight_layout()
plt.savefig("figure_sensitivity_contribution.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure_sensitivity_contribution.png")
