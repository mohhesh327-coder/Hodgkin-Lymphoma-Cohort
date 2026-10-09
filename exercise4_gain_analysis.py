import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

df = pd.read_csv("filtered_variants.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

# Build gene -> LoD map
lod_map = df.groupby("Genes")["LoD"].first().to_dict()

def get_gene(gene_str):
    return str(gene_str).split(",")[0].strip()

def any_driver(gene_str):
    return get_gene(gene_str) in DRIVER_GENES

# Detect gains per patient
all_gains = []
for pid in df["patient_id"].unique():
    p = df[df["patient_id"] == pid]
    all_tps = sorted(p["tp_label"].unique())
    if len(all_tps) < 2:
        continue

    for vid, grp in p.groupby("variant_id"):
        grp = grp.sort_values("tp_order")
        tp_vafs = grp.groupby("tp_label")["VAF"].mean()
        gene = get_gene(grp["Genes"].iloc[0] if len(grp) > 0 else "UNKNOWN")
        lod = lod_map.get(gene, 0.01)

        # Looking for: absent at T0 (VAF=0 or NaN), present at ANY later TP
        first_tp = all_tps[0]
        later_tps = all_tps[1:]

        vaf_at_t0 = tp_vafs.get(first_tp, 0)
        if vaf_at_t0 > 0.001:
            continue  # already present at baseline

        # Check later timepoints
        gained_tps = {}
        for tp in later_tps:
            v = tp_vafs.get(tp, 0)
            if v > lod:
                gained_tps[tp] = v

        if gained_tps:
            last_tp = list(gained_tps.keys())[-1]
            last_vaf = gained_tps[last_tp]
            all_gains.append({
                "patient": pid, "variant": vid, "gene": gene,
                "first_appears": list(gained_tps.keys())[0],
                "peak_tp": last_tp, "peak_VAF": last_vaf,
                "n_timepoints": len(all_tps),
                "n_gained_tps": len(gained_tps),
                "is_driver": any_driver(gene)
            })

gains_df = pd.DataFrame(all_gains)
print(f"Total gains across cohort: {len(gains_df)}")
print(f"Driver gains: {gains_df['is_driver'].sum()} / {len(gains_df)}")

# Per patient
pp = gains_df.groupby("patient").size().sort_values(ascending=False)
print(f"\nPer-patient gains:")
print(f"  Mean: {pp.mean():.1f}, Median: {pp.median():.0f}, Range: {pp.min()}-{pp.max()}")

# Gains by gene
gene_counts = gains_df["gene"].value_counts()
print(f"\nTop genes gained:")
for g, c in gene_counts.head(15).items():
    d = "D" if g in DRIVER_GENES else "."
    print(f"  {g:12s} [{d}] : {c}")

# === FIGURE: 2 panels ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Distribution of gains per patient
ax1 = axes[0]
bins = np.linspace(0, pp.max() + 5, 25)
ax1.hist(pp, bins=bins, color="#d73027", edgecolor="white", linewidth=0.5, alpha=0.8)
ax1.axvline(pp.mean(), color="#1a9850", linestyle="--", linewidth=2, label=f"Mean = {pp.mean():.0f}")
ax1.axvline(pp.median(), color="#4575b4", linestyle=":", linewidth=2, label=f"Median = {pp.median():.0f}")
ax1.set_xlabel("Gained mutations per patient", fontsize=12)
ax1.set_ylabel("Number of patients", fontsize=12)
ax1.set_title(f"Gained mutations across the cohort (n={len(gains_df)} total)", fontsize=13, fontweight="bold", loc="left")
ax1.legend(fontsize=10)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.text(0.98, 0.98, f"Range: {pp.min()}-{pp.max()}\n{gains_df['is_driver'].sum()} driver ({gains_df['is_driver'].mean():.0%})",
         transform=ax1.transAxes, fontsize=9, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

# Panel 2: Gains by gene (top 15)
ax2 = axes[1]
top_genes = gene_counts.head(15)
colors = ["#d73027" if g in DRIVER_GENES else "#999999" for g in top_genes.index]
bars = ax2.barh(range(len(top_genes)), top_genes.values, color=colors, edgecolor="white", height=0.65)
for i, (g, c) in enumerate(top_genes.items()):
    ax2.text(c + 1, i, str(c), va="center", fontsize=10, fontweight="bold",
             color="#d73027" if g in DRIVER_GENES else "#666")
ax2.set_yticks(range(len(top_genes)))
ax2.set_yticklabels(top_genes.index, fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel("Gains (patient\u2013variant)", fontsize=12)
ax2.set_title("Genes with most gained mutations", fontsize=13, fontweight="bold", loc="left")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Legend
from matplotlib.lines import Line2D
legend = [Line2D([0],[0],color="#d73027",lw=4,label="Driver gene"),
          Line2D([0],[0],color="#999999",lw=4,label="Other")]
ax2.legend(handles=legend, fontsize=9, loc="lower right")

fig.suptitle("Gained mutations (absent at baseline, appear on treatment)", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("exercise4_gained_mutations.png", dpi=200, bbox_inches="tight")
print(f"\nSaved exercise4_gained_mutations.png")
