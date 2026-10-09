import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

# ─── Load data ───
somatic = pd.read_csv("somatic_variants_only.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

# Load master with germline flag
import os
if os.path.exists("master_with_germline_flag.csv"):
    master = pd.read_csv("master_with_germline_flag.csv")
else:
    master = pd.read_csv("master_chronological.csv")
    master["germline"] = False

print("="*70)
print("1. HOW MANY MUTATIONS DOES EACH PATIENT CARRY?")
print("="*70)

# Count somatic mutations per patient
mut_count = somatic.groupby("patient_id").agg(
    total_somatic_calls=("variant_id", "count"),
    unique_variant_loci=("variant_id", "nunique"),
    unique_genes=("Genes", "nunique"),
).sort_values("total_somatic_calls", ascending=False)

print(mut_count.to_string())
print(f"\nMean: {mut_count['total_somatic_calls'].mean():.0f}")
print(f"Median: {mut_count['total_somatic_calls'].median():.0f}")
print(f"Range: {mut_count['total_somatic_calls'].min()} - {mut_count['total_somatic_calls'].max()}")

# ─── Also count by group ───
print("\nBy group:")
somatic["group"] = somatic["patient_id"].apply(
    lambda x: "BR" if x.startswith("BR") else ("LH" if x.startswith("LH") else ("C" if x.startswith("C") else "Other"))
)
print(somatic.groupby("group").agg(
    patients=("patient_id", "nunique"),
    mean_calls=("variant_id", "count"),
    mean_unique_variants=("variant_id", "nunique")
))

# ─── Restrict to patients with >=3 timepoints for trajectory analysis ───
def patient_group(pid):
    if pid.startswith("BR"): return "BR"
    if pid.startswith("LH"): return "LH"
    if pid.startswith("C"):  return "C"
    return "Other"

master["group"] = master["patient_id"].apply(patient_group)
longitudinal_patients = master[master["group"].isin(["LH", "C"])].groupby("patient_id")["tp_label"].nunique()
longitudinal_patients = longitudinal_patients[longitudinal_patients >= 3].index.tolist()
print(f"\nLongitudinal patients (>=3 timepoints): {longitudinal_patients}")

# Filter trajectory data
traj_long = traj[traj["patient_id"].isin(longitudinal_patients)].copy()
traj_long["group"] = traj_long["patient_id"].apply(
    lambda x: "LH" if x.startswith("LH") else "C"
)

print("\n" + "="*70)
print("2. INCREASING MUTATIONS (potential resistance/relapse)")
print("="*70)

increasing = traj_long[traj_long["trajectory"] == "increasing"].copy()
if len(increasing) == 0:
    increasing = traj[traj["trajectory"] == "increasing"].copy()

increasing = increasing.sort_values("last_VAF", ascending=False)
print(f"\nTotal increasing trajectory cases: {len(increasing)}")
print(f"Across {increasing['patient_id'].nunique()} patients")
print(f"\nAll increasing mutations:")
print(increasing.to_string(index=False))

# Parse gene names
increasing["gene"] = increasing["variant_id"].apply(
    lambda x: x.split(" | ")[1] if " | " in str(x) else x
)
print(f"\nGenes with increasing mutations:")
print(increasing["gene"].value_counts().to_string())

print("\n" + "="*70)
print("3. VISUALIZE ONE PATIENT'S TRAJECTORIES")
print("="*70)

# Pick the most informative patient: LH200004 (most timepoints, many variants)
target_patient = "LH200004"
print(f"Selected patient: {target_patient}")

pat_data = master[(master["patient_id"] == target_patient) & ~master["germline"]].copy()
pat_data = pat_data.sort_values("tp_order")

n_tp = pat_data["tp_order"].nunique()
n_muts = pat_data["variant_id"].nunique()
print(f"  Timepoints: {n_tp}")
print(f"  Somatic variants: {n_muts}")
print(f"  Timepoints: {sorted(pat_data['tp_label'].unique())}")

# Get trajectory data for this patient
pat_traj = traj[traj["patient_id"] == target_patient]
print(f"  Trajectory breakdown:")
print(pat_traj["trajectory"].value_counts().to_string())

# Plot: VAF trajectories for top variants
# Pick the 15 variants with highest mean VAF that are detected in >= 2 timepoints
variant_means = pat_data.groupby("variant_id")["VAF"].mean().sort_values(ascending=False)
top_vars = []
for v in variant_means.index:
    n_tp_detected = (pat_data[pat_data["variant_id"] == v]["VAF"] > 0.02).sum()
    if n_tp_detected >= 2:
        top_vars.append(v)
    if len(top_vars) >= 15:
        break

# Build a pivot for the heatmap/trajectory plot
fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})

# (A) VAF trajectories
for var_id in top_vars:
    sub = pat_data[pat_data["variant_id"] == var_id].sort_values("tp_order")
    label = var_id.split(" | ")[1] if " | " in str(var_id) else var_id
    # Get trajectory classification
    traj_class = pat_traj[pat_traj["variant_id"] == var_id]["trajectory"].values
    traj_label = traj_class[0] if len(traj_class) > 0 else ""
    ls = "-" if traj_label == "stable" else ("--" if traj_label == "increasing" else ":")
    lw = 2.5 if traj_label == "increasing" else 1.5
    color = "red" if traj_label == "increasing" else ("green" if traj_label == "stable" else "gray")
    axes[0].plot(sub["tp_order"], sub["VAF"], marker="o", label=f"{label} ({traj_label})",
                 linestyle=ls, linewidth=lw, color=color, markersize=6, alpha=0.8)

axes[0].set_xlabel("Timepoint order")
axes[0].set_ylabel("VAF")
axes[0].set_title(f"{target_patient}: Somatic mutation VAF trajectories (top {len(top_vars)} variants)")
axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
axes[0].set_xticks(range(n_tp))
axes[0].set_xticklabels(sorted(pat_data["tp_label"].unique()), rotation=45)

# (B) Total ctDNA burden over time
tp_burden = pat_data.groupby("tp_order")["VAF"].sum()
axes[1].bar(tp_burden.index, tp_burden.values, color="steelblue", alpha=0.8, edgecolor="white", width=0.6)
axes[1].set_xlabel("Timepoint order")
axes[1].set_ylabel("Sum of VAFs")
axes[1].set_title("Total ctDNA variant burden over time")
axes[1].set_xticks(range(n_tp))
axes[1].set_xticklabels(sorted(pat_data["tp_label"].unique()), rotation=45)

# Annotate with number of detected variants
for tp in sorted(pat_data["tp_order"].unique()):
    n_vars = (pat_data[pat_data["tp_order"] == tp]["VAF"] > 0.02).sum()
    axes[1].annotate(f"n={n_vars}", (tp, tp_burden[tp]), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("patient_LH200004_trajectories.png", dpi=150)
print(f"\nSaved patient_LH200004_trajectories.png")

# ─── Also generate a multi-panel overview for top 3 most dynamic patients ───
print("\nGenerating trajectory panels for top dynamic patients...")
dynamic_patients = traj_long.groupby("patient_id").apply(
    lambda g: (g["trajectory"] == "increasing").sum() + (g["trajectory"] == "decreasing").sum()
).sort_values(ascending=False).head(3).index.tolist()

fig, axes = plt.subplots(len(dynamic_patients), 2, figsize=(16, 5 * len(dynamic_patients)))
if len(dynamic_patients) == 1:
    axes = axes.reshape(1, -1)

for idx, pid in enumerate(dynamic_patients):
    sub = master[(master["patient_id"] == pid) & ~master["germline"]].sort_values("tp_order")
    n_tp = sub["tp_order"].nunique()
    
    # VAF trajectories (top 10 variants)
    variant_means = sub.groupby("variant_id")["VAF"].mean().sort_values(ascending=False)
    top_vars_pid = variant_means.head(10).index.tolist()
    
    for var_id in top_vars_pid:
        vsub = sub[sub["variant_id"] == var_id].sort_values("tp_order")
        label = var_id.split(" | ")[1] if " | " in str(var_id) else var_id
        axes[idx, 0].plot(vsub["tp_order"], vsub["VAF"], marker="o", label=label, alpha=0.7, markersize=5)
    axes[idx, 0].set_title(f"{pid}: VAF trajectories")
    axes[idx, 0].set_xlabel("Timepoint")
    axes[idx, 0].set_ylabel("VAF")
    axes[idx, 0].set_xticks(range(n_tp))
    axes[idx, 0].set_xticklabels(sorted(sub["tp_label"].unique()), rotation=45)
    axes[idx, 0].legend(fontsize=7)
    
    # Burden
    burden = sub.groupby("tp_order")["VAF"].sum()
    axes[idx, 1].bar(burden.index, burden.values, color="coral", alpha=0.7, edgecolor="white")
    axes[idx, 1].set_title(f"{pid}: Total ctDNA burden")
    axes[idx, 1].set_xlabel("Timepoint")
    axes[idx, 1].set_ylabel("Sum VAF")
    axes[idx, 1].set_xticks(range(n_tp))
    axes[idx, 1].set_xticklabels(sorted(sub["tp_label"].unique()), rotation=45)

plt.tight_layout()
plt.savefig("top3_dynamic_patients.png", dpi=150)
print("Saved top3_dynamic_patients.png")

print("\n" + "="*70)
print("4. PER-PATIENT TREND SUMMARY")
print("="*70)

# Build a comprehensive per-patient table
trend_summary = traj_long.groupby("patient_id").agg(
    total_detected=("trajectory", "count"),
    n_increasing=("trajectory", lambda x: (x == "increasing").sum()),
    n_decreasing=("trajectory", lambda x: (x == "decreasing").sum()),
    n_stable=("trajectory", lambda x: (x == "stable").sum()),
).reset_index()

trend_summary["pct_increasing"] = (trend_summary["n_increasing"] / trend_summary["total_detected"] * 100).round(1)
trend_summary["pct_decreasing"] = (trend_summary["n_decreasing"] / trend_summary["total_detected"] * 100).round(1)
trend_summary["pct_stable"] = (trend_summary["n_stable"] / trend_summary["total_detected"] * 100).round(1)
trend_summary["dominant_trend"] = trend_summary[["pct_decreasing", "pct_stable", "pct_increasing"]].idxmax(axis=1)
trend_summary["dominant_trend"] = trend_summary["dominant_trend"].str.replace("pct_", "")
trend_summary["group"] = trend_summary["patient_id"].apply(
    lambda x: "BR" if x.startswith("BR") else ("LH" if x.startswith("LH") else ("C" if x.startswith("C") else "Other"))
)

trend_summary = trend_summary.sort_values("total_detected", ascending=False)
print(trend_summary.to_string(index=False))

print("\nSummary by dominant trend:")
print(trend_summary["dominant_trend"].value_counts().to_string())

# Summary stats
print(f"\nAcross {len(trend_summary)} patients:")
print(f"  Mean % decreasing: {trend_summary['pct_decreasing'].mean():.1f}%")
print(f"  Mean % stable:     {trend_summary['pct_stable'].mean():.1f}%")
print(f"  Mean % increasing: {trend_summary['pct_increasing'].mean():.1f}%")

# ─── Summary plot ───
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Stacked bar chart
trend_plot = trend_summary.sort_values("pct_decreasing")
axes[0].barh(trend_plot["patient_id"], trend_plot["pct_decreasing"], label="Decreasing", color="skyblue")
axes[0].barh(trend_plot["patient_id"], trend_plot["pct_stable"], left=trend_plot["pct_decreasing"],
             label="Stable", color="gray")
axes[0].barh(trend_plot["patient_id"], trend_plot["pct_increasing"],
             left=trend_plot["pct_decreasing"] + trend_plot["pct_stable"],
             label="Increasing", color="red")
axes[0].set_xlabel("Percentage of variants")
axes[0].set_title("Per-patient trajectory composition")
axes[0].legend(loc="lower right")

# Total detected variants per patient
trend_summary_sorted = trend_summary.sort_values("total_detected", ascending=False)
axes[1].bar(range(len(trend_summary_sorted)), trend_summary_sorted["total_detected"], color="steelblue", edgecolor="white")
axes[1].set_xticks(range(len(trend_summary_sorted)))
axes[1].set_xticklabels(trend_summary_sorted["patient_id"], rotation=90, fontsize=8)
axes[1].set_ylabel("Total detected variant-patient pairs")
axes[1].set_title("Per-patient detected mutation count")

plt.tight_layout()
plt.savefig("per_patient_trend_summary.png", dpi=150)
print("\nSaved per_patient_trend_summary.png")

# ─── Save ───
increasing.to_csv("increasing_mutations_detail.csv", index=False)
trend_summary.to_csv("per_patient_trend_summary.csv", index=False)
print("\nSaved:")
print("  increasing_mutations_detail.csv")
print("  per_patient_trend_summary.csv")
