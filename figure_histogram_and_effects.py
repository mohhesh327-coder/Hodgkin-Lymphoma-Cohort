import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cs = pd.read_csv("composite_scores.csv")
tv = pd.read_csv("trusted_variants.csv")

# ---------- Histogram of scores ----------
scores = cs["total"]

TIER_BOUNDS = [
    (95, "High-confidence", "#27ae60"),
    (55, "Strong recurrence", "#2980b9"),
    (30, "Possible", "#f39c12"),
    (10, "Weak", "#e67e22"),
    (0, "None", "#e74c3c"),
]

# ---------- Variant effect breakdown ----------
def parse_effect(e):
    if pd.isna(e):
        return "Unknown"
    e = str(e).strip()
    primary = e.split(",")[0].strip().lower()
    if primary in ("", "unknown"):
        return "Unknown"
    if primary == "missense":
        return "Missense"
    if primary == "nonsense":
        return "Nonsense"
    if primary.startswith("frameshift"):
        return "Frameshift"
    if primary.startswith("nonframeshift"):
        return "Non-frameshift indel"
    if primary == "synonymous":
        return "Synonymous"
    if primary == "stoploss":
        return "Stoploss"
    return primary.capitalize()

tv["effect_group"] = tv["Variant Effect"].apply(parse_effect)
effect_counts = tv["effect_group"].value_counts()
# Order by frequency descending
effect_order = effect_counts.index.tolist()

# ---------- Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("GSE244989 Cohort: Score Distribution & Variant Effect Catalog", fontsize=15, fontweight="bold", y=0.97)

# --- Panel 1: Histogram with tier boundaries ---
ax = axes[0]

# Bin edges at 10-unit intervals
bins = np.arange(0, 111, 5)
n, bin_edges, patches = ax.hist(scores, bins=bins, edgecolor="white", linewidth=0.8, zorder=3)

# Color each bar by its bin's tier
for i, patch in enumerate(patches):
    bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
    for bound, name, color in TIER_BOUNDS:
        if bin_center >= bound:
            patch.set_facecolor(color)
            break

# Tier boundary lines
for bound, name, color in TIER_BOUNDS:
    ax.axvline(x=bound, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(bound, ax.get_ylim()[1]*0.95, f"  {name}\n  ({bound})",
            fontsize=7, color=color, ha="left", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor=color, linewidth=0.5))

# Annotate peaks
for i, (score, count) in enumerate(zip(*np.unique(scores.round(0).astype(int), return_counts=True))):
    if count >= 2:
        ax.annotate(f"n={count}", xy=(score, count), xytext=(score, count+0.4),
                    fontsize=6, ha="center", va="bottom", color="gray")

ax.set_xlabel("Recurrence Score", fontsize=11)
ax.set_ylabel("Patients", fontsize=11)
ax.set_title("Recurrence Score Distribution", fontsize=12, fontweight="bold")
ax.set_xlim(-3, 108)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

# --- Panel 2: Variant effect types ---
ax = axes[1]

effect_colors = {
    "Missense": "#3498db",
    "Unknown": "#95a5a6",
    "Nonsense": "#e74c3c",
    "Frameshift": "#e67e22",
    "Non-frameshift indel": "#2ecc71",
    "Synonymous": "#f1c40f",
    "Stoploss": "#9b59b6",
}

colors_plot = [effect_colors.get(e, "#bdc3c7") for e in effect_order]
values = effect_counts.values
total = values.sum()

# Horizontal bar chart
bars = ax.barh(effect_order, values, color=colors_plot, edgecolor="white", linewidth=0.8, height=0.6)
for bar, v in zip(bars, values):
    pct = v / total * 100
    ax.text(bar.get_width() + total*0.005, bar.get_y() + bar.get_height()/2,
            f"{v:,} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Variants", fontsize=11)
ax.set_title(f"Variant Effect Types (n={total:,} trusted variants)", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.set_xlim(0, values.max() * 1.35)
ax.invert_yaxis()

# Inset pie for proportions
from matplotlib.patches import Wedge
inset_ax = ax.inset_axes([0.65, 0.35, 0.33, 0.6])
wedges, texts, autotexts = inset_ax.pie(
    values, labels=None, autopct="%1.0f%%",
    colors=colors_plot, startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 0.8},
    textprops={"fontsize": 5.5},
    pctdistance=0.7,
)
# Only label the top 3 + group rest
for t in autotexts:
    t.set_fontsize(5.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figure_histogram_and_effects.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure_histogram_and_effects.png")
