"""
Exercises 1-4: Correlation matrix, Waterfall, Plotly interactive, Manuscript panel
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import os

cs = pd.read_csv("composite_scores.csv")
DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

# =============================================================================
# EXERCISE 1 — Correlation matrix
# =============================================================================
COMPONENTS = ["tb", "ce", "drv", "mg"]
corr = cs[COMPONENTS + ["total"]].corr(method="pearson")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1,
            center=0, square=True, linewidths=0.8, cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax)
ax.set_title("Exercise 1: Pearson Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("exercise1_correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print("E1: Saved exercise1_correlation_matrix.png")

# Find most correlated pair (excluding self and total)
best_pair, best_r = None, -1
for i, c1 in enumerate(COMPONENTS):
    for c2 in COMPONENTS[i+1:]:
        r = abs(corr.loc[c1, c2])
        if r > best_r:
            best_r = r
            best_pair = (c1, c2)
print(f"E1: Most correlated pair: {best_pair} (r={corr.loc[best_pair[0], best_pair[1]]:.3f})")

# =============================================================================
# EXERCISE 2 — Waterfall chart for one patient
# =============================================================================
# Pick patient LH210010 (tb=40, ce=25, drv=25, mg=10, total=100)
pid = "LH210010"
row = cs[cs["patient"] == pid].iloc[0]
vals = [row["tb"], row["ce"], row["drv"], row["mg"]]
labels = ["TB (0-40)\nVAF rise sum", "CE (0-25)\nInc count", "DRV (0-25)\nDriver inc", "MG (0-10)\nGain count"]
colors_wf = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(8, 5))
cumulative = 0
for i, (v, lbl, c) in enumerate(zip(vals, labels, colors_wf)):
    bottom = cumulative
    ax.bar(i, v, 0.5, bottom=bottom, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    # Annotate the increment
    ax.text(i, bottom + v/2, f"+{v:.0f}", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    cumulative += v

# Total bar
ax.bar(4, cumulative, 0.5, bottom=0, color="#34495e", edgecolor="white", linewidth=1.5, zorder=3)
ax.text(4, cumulative/2, f"={cumulative:.0f}", ha="center", va="center", fontsize=13, fontweight="bold", color="white")

# Connecting lines (waterfall effect)
cumulative = 0
for i, v in enumerate(vals):
    cum_next = cumulative + v
    if i < 3:
        ax.plot([i+0.25, i+0.75], [cum_next, cum_next], color="gray", linewidth=0.8, linestyle="--", zorder=2)
    cumulative = cum_next

ax.set_xticks(range(5))
ax.set_xticklabels(labels + ["Total"], fontsize=8)
ax.set_ylabel("Score Contribution", fontsize=11)
ax.set_title(f"Exercise 2: Waterfall Chart — Patient {pid} (Total={cumulative:.0f})", fontsize=13, fontweight="bold")
ax.set_ylim(0, cumulative * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("exercise2_waterfall.png", dpi=200, bbox_inches="tight")
plt.close()
print("E2: Saved exercise2_waterfall.png")

# =============================================================================
# EXERCISE 3 — Interactive Plotly heatmap
# =============================================================================
import plotly.graph_objects as go

cs_sorted = cs.sort_values("total", ascending=False).reset_index(drop=True)

# Gene hit counts per patient
tv = pd.read_csv("trusted_variants.csv")
def first_gene(g):
    return str(g).split(",")[0].strip()
tv["gene"] = tv["variant_id"].apply(lambda v: str(v).split("|")[1].strip() if "|" in str(v) else "").apply(first_gene)
drv = tv[tv["gene"].isin(DRIVER_GENES)]
gene_hits = drv.groupby("patient_id")["gene"].nunique().to_dict()

hover_texts = []
for _, row in cs_sorted.iterrows():
    pid = row["patient"]
    n_genes = gene_hits.get(pid, 0)
    ht = f"Patient: {pid}<br>TB: {row['tb']:.1f}<br>CE: {row['ce']:.0f}<br>DRV: {row['drv']:.0f}<br>MG: {row['mg']:.0f}<br>Total: {row['total']:.0f}<br>Driver genes hit: {n_genes}"
    hover_texts.append(ht)

z_data = cs_sorted[["tb","ce","drv","mg","total"]].values.T  # shape (5, 46)
hover_text_matrix = [[ht for ht in hover_texts] for _ in range(5)]  # shape (5, 46)
fig_plotly = go.Figure(data=go.Heatmap(
    z=z_data,
    x=cs_sorted["patient"].tolist(),
    y=["TB", "CE", "DRV", "MG", "Total"],
    text=hover_text_matrix,
    hoverinfo="text",
    colorscale="RdBu_r",
    zmin=0,
    zmax=100,
    showscale=True,
    colorbar=dict(title=dict(text="Score", side="right")),
))
fig_plotly.update_layout(
    title="Exercise 3: Interactive Component Scores (hover for details)",
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=11)),
    height=500,
    width=1000,
)
fig_plotly.write_html("exercise3_interactive_heatmap.html")
print("E3: Saved exercise3_interactive_heatmap.html")

# =============================================================================
# EXERCISE 4 — Manuscript Figure Panel (2x2, A-D, 300 dpi)
# =============================================================================
TIER_BOUNDS = [
    (95, "High-confidence", "#27ae60"),
    (55, "Strong recurrence", "#2980b9"),
    (30, "Possible", "#f39c12"),
    (10, "Weak", "#e67e22"),
    (0, "None", "#e74c3c"),
]

# Build sub-panels directly for a clean 2×2 manuscript figure

# --- Panel A: Component heatmap (simplified) ---
fig_panel = plt.figure(figsize=(20, 18))
gs = fig_panel.add_gridspec(2, 2, hspace=0.08, wspace=0.06)

# A: Component stacked bar
ax_a = fig_panel.add_subplot(gs[0, 0])
cs_sorted_p = cs.sort_values("total", ascending=True).reset_index(drop=True)
y_a = np.arange(len(cs_sorted_p))
cumul_a = np.zeros(len(cs_sorted_p))
comp_colors = {"tb": "#1f77b4", "ce": "#ff7f0e", "drv": "#2ca02c", "mg": "#d62728"}
for comp in ["tb","ce","drv","mg"]:
    vals = cs_sorted_p[comp].values
    ax_a.barh(y_a, vals, 0.7, left=cumul_a, color=comp_colors[comp], label=comp.upper(), edgecolor="white", linewidth=0.3)
    cumul_a += vals
ax_a.set_yticks(y_a)
ax_a.set_yticklabels(cs_sorted_p["patient"], fontsize=6)
ax_a.set_xlabel("Score", fontsize=10)
ax_a.set_title("A  Component Contribution", fontsize=12, fontweight="bold", loc="left")
ax_a.legend(fontsize=7, loc="lower right")
ax_a.invert_yaxis()
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# B: Driver gene landscape (simplified VAF heatmap)
ax_b = fig_panel.add_subplot(gs[0, 1])
tv_b = pd.read_csv("trusted_variants.csv")
tv_b["gene"] = tv_b["variant_id"].apply(lambda v: str(v).split("|")[1].strip() if "|" in str(v) else "").apply(first_gene)
drv_b = tv_b[tv_b["gene"].isin(DRIVER_GENES)]
mv_b = drv_b.groupby(["patient_id","gene"])["VAF"].max().reset_index()
pivot_b = mv_b.pivot_table(index="gene", columns="patient_id", values="VAF")
pivot_b = pivot_b.reindex(columns=cs.sort_values("total",ascending=False)["patient"].tolist())
for g in DRIVER_GENES:
    if g not in pivot_b.index:
        pivot_b.loc[g] = np.nan
gene_freq_b = pivot_b.notna().sum(axis=1)
gene_order_b = gene_freq_b.sort_values(ascending=False).index.tolist()
pivot_b = pivot_b.loc[gene_order_b]
sns.heatmap(pivot_b, ax=ax_b, cmap="rocket_r", vmin=0, vmax=1, cbar_kws={"label":"Max VAF","shrink":0.4},
            xticklabels=[p if i%max(1,len(pivot_b.columns)//12)==0 else "" for i,p in enumerate(pivot_b.columns)],
            yticklabels=True, linewidths=0.3, linecolor="#eeeeee")
ax_b.set_title("B  Driver Gene Landscape", fontsize=12, fontweight="bold", loc="left")
ax_b.set_xlabel("Patient", fontsize=10)
ax_b.set_ylabel("")
ax_b.tick_params(axis="x", labelsize=5, rotation=45)
ax_b.tick_params(axis="y", labelsize=7)

# C: Driver frequency bar chart
ax_c = fig_panel.add_subplot(gs[1, 0])
freq_c = pivot_b.notna().sum(axis=1).sort_values(ascending=True)
colors_c = ["#e74c3c" if g=="KRAS" else "#3498db" for g in freq_c.index]
bars_c = ax_c.barh(freq_c.index, freq_c.values, color=colors_c, edgecolor="white", linewidth=0.6, height=0.6)
for bar, v in zip(bars_c, freq_c.values):
    ax_c.text(v+0.3, bar.get_y()+bar.get_height()/2, f"{v}", va="center", fontsize=8, fontweight="bold")
ax_c.set_xlabel("Patients", fontsize=10)
ax_c.set_title("C  Driver Gene Frequency", fontsize=12, fontweight="bold", loc="left")
ax_c.set_xlim(0, 50)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.tick_params(labelsize=8)

# D: Tier distribution donut
ax_d = fig_panel.add_subplot(gs[1, 1])
tier_order_d = ["High-confidence","Strong recurrence","Possible","Weak","None"]
tier_counts_d = cs["total"].apply(lambda s: next((n for b,n,c in TIER_BOUNDS if s>=b), "None"))
counts_d = tier_counts_d.value_counts().reindex(tier_order_d, fill_value=0)
colors_d = [c for _,_,c in TIER_BOUNDS]
wedges_d, texts_d, autotexts_d = ax_d.pie(
    counts_d.values, labels=[f"{n}\n({c})" for n,c in zip(tier_order_d, counts_d.values)],
    colors=colors_d, startangle=90, wedgeprops={"edgecolor":"white","linewidth":1.5},
    textprops={"fontsize":8}, autopct="%1.0f%%", pctdistance=0.7)
centre_d = plt.Circle((0,0), 0.5, fc="white", linewidth=0)
ax_d.add_artist(centre_d)
ax_d.text(0, 0, f"n={len(cs)}", ha="center", va="center", fontsize=14, fontweight="bold")
ax_d.set_title("D  Tier Distribution", fontsize=12, fontweight="bold", loc="left")

# Caption
fig_panel.text(0.5, 0.005,
    "Figure 1. Recurrence scoring system for the GSE244989 Hodgkin lymphoma cohort (n=46). "
    "(A) Component contribution stacked bars for all patients, sorted by total recurrence score. "
    "(B) Driver gene VAF heatmap showing max VAF per gene per patient. "
    "(C) Frequency of driver genes across the cohort. "
    "(D) Tier distribution with counts and proportions.",
    ha="center", va="bottom", fontsize=9, style="italic", wrap=True)

plt.savefig("exercise4_manuscript_panel.png", dpi=300, bbox_inches="tight")
plt.close()
print("E4: Saved exercise4_manuscript_panel.png (300 dpi)")
print("\nAll exercises complete.")
