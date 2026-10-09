import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

tv = pd.read_csv("trusted_variants.csv")
cs = pd.read_csv("composite_scores.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

def first_gene(g):
    return str(g).split(",")[0].strip()

tv["first_gene"] = tv["Genes"].apply(first_gene)
tv["variant_gene"] = tv["variant_id"].apply(lambda v: str(v).split("|")[1].strip() if "|" in str(v) else "")
tv["first_gene_from_vid"] = tv["variant_gene"].apply(first_gene)

# Use gene from variant_id (more precise for multi-gene rows)
use_gene = tv["first_gene_from_vid"]
tv["gene"] = use_gene

drv = tv[tv["gene"].isin(DRIVER_GENES)]

# Trajectory lookup: (patient, variant) -> trajectory
traj_lookup = traj.set_index(["patient_id","variant_id"])["trajectory"].to_dict()

# Per patient-gene: max VAF, trajectory info
def agg_gene(grp):
    max_vaf = grp["VAF"].max()
    # Check if any variant in this gene-grp has an increasing trajectory
    has_inc = any(traj_lookup.get((grp.name[0], vid), "") == "increasing" for vid in grp["variant_id"])
    has_dec = any(traj_lookup.get((grp.name[0], vid), "") == "decreasing" for vid in grp["variant_id"])
    # Which variant has the max VAF?
    max_row = grp.loc[grp["VAF"].idxmax()]
    max_vid = max_row["variant_id"]
    max_vaf_traj = traj_lookup.get((grp.name[0], max_vid), "stable")
    return pd.Series({"max_vaf": max_vaf, "trajectory": max_vaf_traj, "has_inc": has_inc})

gene_summary = drv.groupby(["patient_id","gene"]).apply(agg_gene).reset_index()

# Sort patients by total score descending
patient_order = cs.sort_values("total", ascending=False)["patient"].tolist()

# Gene frequency (patients per gene)
gene_freq = gene_summary.groupby("gene")["patient_id"].nunique()
gene_order = gene_freq.sort_values(ascending=False).index.tolist()
# Move KRAS to bottom (0 patients)
if "KRAS" in gene_order:
    gene_order.remove("KRAS")
    gene_order.append("KRAS")

# Build matrix: patients × genes, value = max VAF (or NaN if absent)
matrix = gene_summary.pivot_table(index="gene", columns="patient_id", values="max_vaf", aggfunc="max")
# Ensure all 12 genes present
for g in DRIVER_GENES:
    if g not in matrix.index:
        matrix.loc[g] = np.nan
matrix = matrix.reindex(index=gene_order, columns=patient_order)

# Also build trajectory matrix for point overlays
traj_matrix = gene_summary.pivot_table(index="gene", columns="patient_id", values="trajectory", aggfunc="first")
for g in DRIVER_GENES:
    if g not in traj_matrix.index:
        traj_matrix.loc[g] = ""
traj_matrix = traj_matrix.reindex(index=gene_order, columns=patient_order)

# --- Figure ---
fig = plt.figure(figsize=(22, 10))
gs = fig.add_gridspec(3, 4, height_ratios=[0.6, 6, 1.2], width_ratios=[15, 1.5, 0.8, 0.3],
                       hspace=0.05, wspace=0.05)

# Top panel: recurrence score bar
ax_top = fig.add_subplot(gs[0, 0])
scores = cs.set_index("patient").loc[patient_order, "total"]
colors_top = ["#27ae60" if s>=95 else "#2980b9" if s>=55 else "#f39c12" if s>=30 else "#e67e22" if s>=10 else "#e74c3c" for s in scores]
ax_top.bar(range(len(scores)), scores.values, color=colors_top, width=0.9, edgecolor="white", linewidth=0.3)
ax_top.set_ylabel("Score", fontsize=9)
ax_top.set_ylim(0, 105)
ax_top.set_xlim(-0.5, len(scores)-0.5)
ax_top.tick_params(axis="x", labelbottom=False, length=0)
ax_top.tick_params(axis="y", labelsize=7)
ax_top.set_title("Recurrence Score", fontsize=10, fontweight="bold", pad=3)
ax_top.spines["bottom"].set_visible(False)
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)

# Spacer above colorbar
ax_spacer = fig.add_subplot(gs[0, 1])
ax_spacer.axis("off")

# Main heatmap
ax = fig.add_subplot(gs[1, 0])
vmin, vmax = 0, 1
norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.15, vmax=1)
cmap = sns.color_palette("rocket_r", as_cmap=True)

n_genes = len(gene_order)
n_patients = len(patient_order)

# Draw heatmap manually for fine control
im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

# Overlay trajectory markers
for i, gene in enumerate(gene_order):
    for j, patient in enumerate(patient_order):
        val = matrix.loc[gene, patient]
        traj_val = traj_matrix.loc[gene, patient]
        if pd.isna(val):
            # Draw X for absent
            ax.plot(j, i, "x", color="#cccccc", markersize=4, markeredgewidth=0.8)
        else:
            # Trajectory marker in top-left corner
            if traj_val == "increasing":
                ax.plot(j-0.25, i-0.25, "^", color="white", markersize=4, markeredgewidth=0.3)
            elif traj_val == "decreasing":
                ax.plot(j-0.25, i-0.25, "v", color="white", markersize=4, markeredgewidth=0.3)

# Gene labels
ax.set_yticks(range(n_genes))
ax.set_yticklabels(gene_order, fontsize=9, fontweight="bold")
# Patient labels (show every Nth to avoid crowding)
step = max(1, n_patients // 15)
ax.set_xticks(range(n_patients))
ax.set_xticklabels([patient_order[i] if i%step==0 else "" for i in range(n_patients)], fontsize=5.5, rotation=45, ha="right")
ax.set_xlim(-0.5, n_patients-0.5)
ax.set_ylim(n_genes-0.5, -0.5)

# Gene frequency side bar
ax_freq = fig.add_subplot(gs[1, 1])
freq_vals = gene_freq.reindex(gene_order, fill_value=0).values
ax_freq.barh(range(n_genes), freq_vals, color="#34495e", height=0.7, edgecolor="white", linewidth=0.3)
ax_freq.set_xlim(0, n_patients+2)
ax_freq.set_yticks(range(n_genes))
ax_freq.set_yticklabels([])
ax_freq.set_xlabel("Patients", fontsize=8)
ax_freq.tick_params(axis="x", labelsize=6)
ax_freq.invert_yaxis()
ax_freq.spines["top"].set_visible(False)
ax_freq.spines["right"].set_visible(False)
for i, v in enumerate(freq_vals):
    ax_freq.text(v+0.3, i, str(v), va="center", fontsize=7, fontweight="bold")

# Colorbar
ax_cbar = fig.add_subplot(gs[1, 2])
cbar = fig.colorbar(im, cax=ax_cbar, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_label("Max VAF", fontsize=9)
cbar.ax.tick_params(labelsize=7)

# Legend for trajectory markers
ax_legend = fig.add_subplot(gs[2, 0])
ax_legend.axis("off")
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#b0b0b0", markersize=8, label="Missense / other"),
    Line2D([0],[0], marker="^", color="w", markerfacecolor="white", markersize=8, label="Increasing"),
    Line2D([0],[0], marker="v", color="w", markerfacecolor="white", markersize=8, label="Decreasing"),
    Line2D([0],[0], marker="x", color="#cccccc", markersize=6, label="Not detected", linestyle="None"),
]
ax_legend.legend(handles=legend_elements, loc="center", ncol=6, fontsize=8, frameon=False)

# Title
fig.suptitle("Driver Gene Landscape: Max VAF per Gene per Patient", fontsize=15, fontweight="bold", y=0.98)

plt.savefig("figure_driver_landscape_vaf.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure_driver_landscape_vaf.png")
