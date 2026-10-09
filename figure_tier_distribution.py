import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cs = pd.read_csv("composite_scores.csv")

TIERS = [
    (95, "High-confidence", "#27ae60"),
    (55, "Strong recurrence", "#2980b9"),
    (30, "Possible", "#f39c12"),
    (10, "Weak", "#e67e22"),
    (0, "None", "#e74c3c"),
]

def get_tier(score):
    for bound, name, _ in TIERS:
        if score >= bound:
            return name
    return "None"

def get_color(score):
    for bound, name, color in TIERS:
        if score >= bound:
            return color
    return "#e74c3c"

cs["tier"] = cs["total"].apply(get_tier)
cs["color"] = cs["total"].apply(get_color)

tier_order = ["High-confidence", "Strong recurrence", "Possible", "Weak", "None"]
tier_counts = cs["tier"].value_counts()
tier_counts = tier_counts.reindex(tier_order, fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), gridspec_kw={"width_ratios": [1.8, 1, 1.2]})
fig.suptitle("GSE244989 Cohort: Recurrence Score Tier Distribution", fontsize=16, fontweight="bold", y=0.98)

# --- Panel 1: Waterfall / lollipop of individual scores ---
ax = axes[0]
cs_sorted = cs.sort_values("total", ascending=True).reset_index(drop=True)
y = np.arange(len(cs_sorted))
ax.scatter(cs_sorted["total"], y, c=cs_sorted["color"], s=40, edgecolors="gray", linewidth=0.5, zorder=3)
# Vertical lines from 0 to each point
for idx, row in cs_sorted.iterrows():
    ax.plot([0, row["total"]], [idx, idx], color=row["color"], linewidth=1.2, alpha=0.6, zorder=2)
# Tier boundaries
for bound, name, color in TIERS:
    ax.axvline(x=bound, color=color, linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(bound, -0.5, f"  {name} \u2265{bound}", fontsize=7, color=color, ha="left", va="bottom",
            rotation=0, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(cs_sorted["patient"], fontsize=7)
ax.set_xlabel("Recurrence Score", fontsize=11)
ax.set_title("Per-Patient Scores with Tier Boundaries", fontsize=11, fontweight="bold")
ax.set_xlim(-5, 110)
ax.invert_yaxis()
# Mark the 9 at 100
for idx, row in cs_sorted.iterrows():
    if row["total"] >= 100:
        ax.annotate("", xy=(row["total"], idx), xytext=(105, idx),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.5), fontsize=6, color="gray")
ax.text(106, -0.5, "9 at 100", fontsize=8, fontweight="bold", color="#27ae60", ha="left")

# --- Panel 2: Tier count bar chart ---
ax = axes[1]
colors_list = [c for _, _, c in TIERS]
bars = ax.barh(tier_order, tier_counts.values, color=colors_list, edgecolor="gray", linewidth=0.8, height=0.6)
for bar, count in zip(bars, tier_counts.values):
    pct = count / 46 * 100
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct:.0f}%)", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, tier_counts.max() + 8)
ax.set_xlabel("Patients", fontsize=11)
ax.set_title("Tier Breakdown", fontsize=11, fontweight="bold")
ax.tick_params(axis="y", labelsize=9)
ax.invert_yaxis()
# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel 3: Donut chart ---
ax = axes[2]
wedges, texts, autotexts = ax.pie(
    tier_counts.values, labels=None, autopct="%1.0f%%",
    colors=colors_list, startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 9, "fontweight": "bold"},
    pctdistance=0.75,
)
centre = plt.Circle((0, 0), 0.55, fc="white", linewidth=0)
ax.add_artist(centre)
ax.text(0, 0, f"n={len(cs)}", ha="center", va="center", fontsize=13, fontweight="bold")
ax.set_title("Proportion", fontsize=11, fontweight="bold")
# Legend
legend_patches = [mpatches.Patch(color=c, label=f"{n} ({tier_counts[n]})") for n, (_, _, c) in zip(tier_order, TIERS)]
ax.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, title="Tier", title_fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figure_tier_distribution.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure_tier_distribution.png")
