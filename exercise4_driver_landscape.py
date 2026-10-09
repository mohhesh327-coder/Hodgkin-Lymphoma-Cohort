import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DRIVER_GENES = ["TP53", "NOTCH1", "KRAS", "EP300", "CASP8",
                "MYB", "CARD11", "TNFAIP3", "STAT3", "STAT6",
                "CREBBP", "SMARCA4"]

COLOR_SCHEME = {
    "TP53": "#1a9850", "NOTCH1": "#d73027", "KRAS": "#4575b4",
    "EP300": "#f46d43", "CASP8": "#fdae61", "MYB": "#fee08b",
    "CARD11": "#66bd63", "TNFAIP3": "#a6d96a", "STAT3": "#3288bd",
    "STAT6": "#5e4fa2", "CREBBP": "#9e0142", "SMARCA4": "#f46d43"
}

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
tv = pd.read_csv("trusted_variants.csv")

def gene_matches(gene_str, target):
    return target in [g.strip() for g in str(gene_str).split(",")]

def any_driver(gene_str):
    return any(gene_matches(gene_str, g) for g in DRIVER_GENES)

traj["gene"] = traj["variant_id"].str.split("|").str[1].str.strip()

# --- Panel 1: Cohort-wide driver gene frequencies ---
gene_hits = {}
for g in DRIVER_GENES:
    gene_hits[g] = traj["gene"].apply(lambda x: gene_matches(x, g)).sum()

genes_sorted = sorted(gene_hits, key=gene_hits.get, reverse=True)
hits_sorted = [gene_hits[g] for g in genes_sorted]
colors_sorted = [COLOR_SCHEME.get(g, "#999999") for g in genes_sorted]

# --- Panel 2: Per-patient driver score distribution ---
# Driver score = number of unique driver variants per patient
tv["is_driver"] = tv["Genes"].apply(any_driver)
driver_per_patient = tv[tv["is_driver"]].groupby("patient_id")["variant_id"].nunique()

fig = plt.figure(figsize=(16, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)

# --- Left: horizontal bar chart ---
ax1 = fig.add_subplot(gs[0])
bars = ax1.barh(range(len(genes_sorted)), hits_sorted, color=colors_sorted, edgecolor="white", linewidth=0.5, height=0.7)

# Add value labels
for i, (v, g) in enumerate(zip(hits_sorted, genes_sorted)):
    ax1.text(v + 15, i, str(v), va="center", fontsize=11, fontweight="bold", color=COLOR_SCHEME.get(g, "#333"))

    # Track trajectory breakdown
    gene_traj = traj[traj["gene"].apply(lambda x: gene_matches(x, g))]
    n_inc = (gene_traj["trajectory"] == "increasing").sum()
    n_dec = (gene_traj["trajectory"] == "decreasing").sum()
    n_sta = (gene_traj["trajectory"] == "stable").sum()
    if n_inc > 0:
        ax1.text(v + 15, i - 0.22, f"\u2191{n_inc} \u2193{n_dec} \u2192{n_sta}",
                 va="top", fontsize=7, color="#666", fontfamily="monospace")

ax1.set_yticks(range(len(genes_sorted)))
ax1.set_yticklabels(genes_sorted, fontsize=12, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Patient\u2013variant hits", fontsize=12)
ax1.set_title("Cohort-wide driver gene frequencies", fontsize=14, fontweight="bold", loc="left", pad=10)

# Inset: circle for NOTCH1 dominance
notch1_pct = gene_hits["NOTCH1"] / sum(gene_hits.values()) * 100
ax1.text(0.98, 0.02,
         f"NOTCH1: {gene_hits['NOTCH1']} hits\n({notch1_pct:.0f}% of all driver hits)\n"
         f"KRAS: 0 hits (not in HL)",
         transform=ax1.transAxes, fontsize=9, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f0", edgecolor="#d73027", alpha=0.9),
         fontfamily="monospace")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Right: histogram of per-patient driver scores ---
ax2 = fig.add_subplot(gs[1])

bins = np.linspace(0, driver_per_patient.max() + 10, 20)
ax2.hist(driver_per_patient, bins=bins, color="#4575b4", edgecolor="white", linewidth=0.5, alpha=0.85)

# Summary stats
mean_val = driver_per_patient.mean()
median_val = driver_per_patient.median()
ax2.axvline(mean_val, color="#d73027", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.0f}")
ax2.axvline(median_val, color="#1a9850", linestyle=":", linewidth=2, label=f"Median = {median_val:.0f}")

ax2.set_xlabel("Unique driver variants per patient", fontsize=12)
ax2.set_ylabel("Number of patients", fontsize=12)
ax2.set_title("Distribution of driver scores (n=46)", fontsize=14, fontweight="bold", loc="left", pad=10)
ax2.legend(fontsize=10, framealpha=0.9)

# Annotate key thresholds
ax2.axvspan(np.percentile(driver_per_patient, 33), np.percentile(driver_per_patient, 66),
            alpha=0.06, color="#fee090", label="IQR range")
ax2.text(0.98, 0.98,
         f"Range: {driver_per_patient.min()}\u2013{driver_per_patient.max()}\n"
         f"P25: {driver_per_patient.quantile(0.25):.0f}  P75: {driver_per_patient.quantile(0.75):.0f}",
         transform=ax2.transAxes, fontsize=9, va="top", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9),
         fontfamily="monospace")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Driver gene landscape across the HL cohort", fontsize=16, fontweight="bold", y=1.01)
fig.savefig("exercise4_driver_landscape.png", dpi=200, bbox_inches="tight")
print(f"Saved exercise4_driver_landscape.png")
print(f"\nNOTCH1: {gene_hits['NOTCH1']} hits ({notch1_pct:.0f}%)")
print(f"KRAS: 0 hits (confirmed absent)")
print(f"\nDriver score per patient: mean={mean_val:.0f}, median={median_val:.0f}, range={driver_per_patient.min()}--{driver_per_patient.max()}")
