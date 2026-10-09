import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)

df = pd.read_csv("master_chronological.csv")
print(f"Loaded {len(df):,} rows, {df['patient_id'].nunique()} patients\n")

# ─── Patient group mapping ───
def patient_group(pid):
    if pid.startswith("BR"): return "BR"
    if pid.startswith("LH"): return "LH"
    if pid.startswith("C"):  return "C"
    return "Other"

df["group"] = df["patient_id"].apply(patient_group)

# Known Hodgkin lymphoma driver genes (from the AmpliSeq HL panel literature)
HL_DRIVERS = [
    "STAT6", "GNA13", "SOCS1", "TNFAIP3", "B2M", "NFKBIA", "NFKBIE",
    "CREBBP", "EP300", "CARD11", "ITPKB", "PTPN1", "KLHL6", "MYB",
    "BCL10", "REL", "BCL6", "MYC", "TP53", "FAS", "TNFRSF14",
    "CIITA", "CD58", "CD83", "NFATC1", "IRF4", "IKZF3", "ETV6",
    "TRAF3", "MAP3K14", "XPO1", "KMT2D", "ARID1A", "CYLD",
    "SGK1", "HIST1H1E", "HIST1H2BC", "CASP8", "NOTCH1", "ABL1",
    "CSF1R", "CSF2RB", "SMARCA4", "NUMA1", "PIK3CD", "RET",
    "STAT3", "IL18BP", "SH3BP5", "LOC100128494", "HMGXB3"
]

# ═══════════════════════════════════════════════════════════════
# 1. GERMLINE FILTERING
# ═══════════════════════════════════════════════════════════════
print("="*70)
print("1. GERMLINE FILTERING")
print("="*70)

# Strategy: A variant is likely germline if:
# (a) It appears in a DNA NORMAL sample, OR
# (b) Its VAF is consistently near 0.5 or 1.0 across all timepoints (for patients with >=2 tp)

# (a) Tag variants found in DNA NORMAL samples
normal_timepoints = df[df["timepoint"].str.lower().str.contains("dna normal", na=False)]["timepoint"].unique()
print(f"DNA NORMAL timepoints found: {len(normal_timepoints)}")

# Get all (patient, variant_id) pairs that appear in NORMAL samples
normal_variants = df[df["timepoint"].isin(normal_timepoints)][["patient_id", "variant_id"]].drop_duplicates()
normal_variants["in_normal"] = True
print(f"Unique variant-patient pairs in NORMAL: {len(normal_variants)}")

# (b) Germline by VAF consistency: VAF consistently ~50% or ~100% across timepoints
def is_germline_by_vaf(grp):
    """Flag variants where VAF is consistently near 0.5 or 1.0 across timepoints."""
    vafs = grp["VAF"].dropna().values
    if len(vafs) < 2:
        return False
    # Check if all non-zero VAFs are in germline ranges
    non_zero = vafs[vafs > 0.05]
    if len(non_zero) == 0:
        return False
    # Germline heterozygous: 0.4-0.6 range
    het = (non_zero >= 0.35) & (non_zero <= 0.65)
    # Germline homozygous: 0.85-1.0 range
    hom = (non_zero >= 0.85) & (non_zero <= 1.1)
    # All non-zero VAFs must be in germline ranges
    return (het.all() or hom.all())

# (c) Use ClinVar annotation for known benign
known_benign = df["ClinVar"].str.lower().str.contains("benign|polimorfismo|polymorphism", na=False)

# Compute germline flags
germline_by_vaf = df.groupby(["patient_id", "variant_id"], group_keys=False).apply(
    lambda g: pd.Series({"germline_vaf": is_germline_by_vaf(g)})
).reset_index()

df = df.merge(germline_by_vaf, on=["patient_id", "variant_id"], how="left")
df = df.merge(normal_variants, on=["patient_id", "variant_id"], how="left")
df["germline"] = df["in_normal"].fillna(False) | df["germline_vaf"].fillna(False)

print(f"Variants flagged as germline: {df['germline'].sum():,} / {len(df):,} ({df['germline'].mean()*100:.1f}%)")

# Separate somatic vs germline
somatic = df[~df["germline"]].copy()
print(f"Somatic variants: {len(somatic):,}")
print(f"Germline variants: {df['germline'].sum():,}")

# Show germline examples
germline_examples = df[df["germline"]].groupby("variant_id").agg(
    n_patients=("patient_id", "nunique"),
    mean_VAF=("VAF", "mean")
).sort_values("n_patients", ascending=False).head(10)
print("\nTop germline variants (by patient count):")
print(germline_examples.to_string())

# ═══════════════════════════════════════════════════════════════
# 2. STRATIFY BY PATIENT GROUP
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. STRATIFY BY PATIENT GROUP")
print("="*70)

# Per-group variant burden
group_stats = somatic.groupby("group").agg(
    n_patients=("patient_id", "nunique"),
    total_variants=("variant_id", "count"),
    unique_variants=("variant_id", "nunique"),
    mean_VAF=("VAF", "mean"),
    median_VAF=("VAF", "median"),
    mean_coverage=("Coverage", "mean")
)
print("\nPer-group somatic variant statistics:")
print(group_stats.to_string())

# Per-group trajectory patterns (load existing trajectory data)
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
traj["group"] = traj["patient_id"].apply(patient_group)

group_traj = traj.groupby(["group", "trajectory"]).size().unstack(fill_value=0)
print("\nPer-group trajectory counts:")
print(group_traj.to_string())

group_traj_pct = group_traj.div(group_traj.sum(axis=1), axis=0) * 100
print("\nPer-group trajectory percentages:")
print(group_traj_pct.round(1).astype(str) + "%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
group_stats["total_variants"].plot(kind="bar", ax=axes[0], color=["steelblue", "coral", "seagreen", "orange"], edgecolor="white")
axes[0].set_ylabel("Total somatic variant calls")
axes[0].set_title("Somatic variant burden by group")

group_traj_pct.plot(kind="bar", ax=axes[1], color={"increasing": "red", "decreasing": "skyblue", "stable": "gray"}, edgecolor="white")
axes[1].set_ylabel("Percentage")
axes[1].set_title("Trajectory distribution by group")
axes[1].legend(title="Trajectory")
plt.tight_layout()
plt.savefig("germline_filtered_group_stratification.png", dpi=150)
print("\nSaved germline_filtered_group_stratification.png")

# ═══════════════════════════════════════════════════════════════
# 3. FOCUS ON DRIVER MUTATIONS (somatic only)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. FOCUS ON DRIVER MUTATIONS")
print("="*70)

# Parse individual genes from the Genes column (handles multi-gene entries like "CSF1R,HMGXB3")
def parse_genes(gene_str):
    genes = str(gene_str).split(",")
    return [g.strip() for g in genes]

somatic["gene_list"] = somatic["Genes"].apply(parse_genes)

# Flag driver mutations
def is_driver(gene_list):
    return any(g in HL_DRIVERS for g in gene_list)

somatic["is_driver"] = somatic["gene_list"].apply(is_driver)
driver_df = somatic[somatic["is_driver"]].copy()
print(f"Driver mutation calls: {len(driver_df):,} / {len(somatic):,} ({len(driver_df)/len(somatic)*100:.1f}%)")
print(f"Unique driver genes involved: {driver_df['Genes'].nunique()}")

# Driver gene prevalence (patients affected)
driver_patient_count = driver_df.groupby("Genes")["patient_id"].nunique().sort_values(ascending=False)
print("\nDriver gene prevalence (patients affected, top 20):")
print(driver_patient_count.head(20).to_string())

# Driver mutation trajectory patterns
# Load the trajectory data and filter for driver variants
driver_traj = traj[traj["variant_id"].str.contains("|".join(HL_DRIVERS), na=False)]
print(f"\nDriver mutations with trajectory data: {len(driver_traj)}")

# VAF dynamics of key drivers
print("\nVAF dynamics for top driver genes:")
for gene in ["STAT6", "GNA13", "SOCS1", "TNFAIP3", "B2M", "CREBBP", "EP300", "CARD11", "TP53", "NOTCH1", "MYB"]:
    gene_variants = driver_df[driver_df["Genes"].str.contains(gene, na=False)]
    if len(gene_variants) == 0:
        continue
    gene_traj = traj[traj["variant_id"].str.contains(gene, na=False)]
    if len(gene_traj) == 0:
        continue
    traj_counts = gene_traj["trajectory"].value_counts()
    print(f"  {gene}: {len(gene_traj)} trajectories — "
          f"inc={traj_counts.get('increasing',0)}, dec={traj_counts.get('decreasing',0)}, "
          f"stable={traj_counts.get('stable',0)}")

# Driver gene heatmap
driver_pivot = driver_df.pivot_table(
    index="patient_id",
    columns="Genes",
    values="VAF",
    aggfunc="max"
).fillna(0)

# Filter to top genes
top_genes = driver_patient_count.head(15).index.tolist()
driver_pivot_top = driver_pivot[[c for c in top_genes if c in driver_pivot.columns]]

# Sort patients
driver_pivot_top["total"] = driver_pivot_top.sum(axis=1)
driver_pivot_top = driver_pivot_top.sort_values("total", ascending=False)
driver_pivot_top = driver_pivot_top.drop(columns=["total"])

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(driver_pivot_top, cmap="Reds", vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Max VAF"}, linewidths=0.5, linecolor="lightgray")
ax.set_title("Driver gene mutation landscape (max VAF per patient)", fontsize=14)
ax.set_ylabel("Patient")
ax.set_xlabel("Driver Gene")
plt.tight_layout()
plt.savefig("driver_mutation_heatmap.png", dpi=150)
print("\nSaved driver_mutation_heatmap.png")

# ═══════════════════════════════════════════════════════════════
# 4. CORRELATE WITH CLINICAL OUTCOME (inferred from ctDNA dynamics)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. CLINICAL OUTCOME CORRELATION")
print("="*70)

# Infer clinical outcome from ctDNA trajectory patterns
# Only for LH & C patients with longitudinal timepoints
longitudinal = somatic[somatic["group"].isin(["LH", "C"])].copy()
long_patients = longitudinal.groupby("patient_id")["tp_label"].nunique()
long_patients = long_patients[long_patients >= 3].index  # at least 3 timepoints

print(f"Longitudinal patients (>=3 timepoints): {len(long_patients)}")

# For each patient, compute:
# - Trajectory of total ctDNA burden over time
# - Clearance rate (fraction of variants that become undetectable)
# - Whether any variants emerge at late timepoints

outcome_records = []
for pid in long_patients:
    sub = longitudinal[longitudinal["patient_id"] == pid].copy()
    if sub.empty:
        continue
    
    # Total VAF burden per timepoint (sum of VAFs of all somatic variants)
    tp_burden = sub.groupby("tp_order")["VAF"].sum()
    
    # Trajectory per variant in this patient
    pid_traj = traj[traj["patient_id"] == pid]
    
    # Fraction decreasing
    n_total = len(pid_traj)
    n_decreasing = (pid_traj["trajectory"] == "decreasing").sum()
    n_increasing = (pid_traj["trajectory"] == "increasing").sum()
    n_stable = (pid_traj["trajectory"] == "stable").sum()
    
    decrease_frac = n_decreasing / n_total if n_total > 0 else 0
    increase_frac = n_increasing / n_total if n_total > 0 else 0
    
    # Clearance: fraction of initial mutations that drop to VAF < 0.02 at last tp
    # Get mutations present at first timepoint
    first_tp = sub["tp_order"].min()
    last_tp = sub["tp_order"].max()
    
    first_muts = sub[sub["tp_order"] == first_tp]["variant_id"].unique()
    last_muts = sub[sub["tp_order"] == last_tp]
    last_muts_dict = dict(zip(last_muts["variant_id"], last_muts["VAF"]))
    
    cleared = sum(1 for m in first_muts if m in last_muts_dict and last_muts_dict[m] < 0.02)
    persistent = sum(1 for m in first_muts if m in last_muts_dict and last_muts_dict[m] >= 0.02)
    clearance_rate = cleared / len(first_muts) if len(first_muts) > 0 else 0
    
    # Emergent mutations: present at last tp but not at first tp
    emergent = sum(1 for m in last_muts["variant_id"].unique() if m not in first_muts)
    
    # Mean trajectory
    first_burden = tp_burden.iloc[0] if len(tp_burden) > 0 else 0
    last_burden = tp_burden.iloc[-1] if len(tp_burden) > 0 else 0
    
    # Infer outcome
    if clearance_rate > 0.7 and emergent == 0:
        inferred_outcome = "Complete responder"
    elif clearance_rate > 0.3 and emergent < 3:
        inferred_outcome = "Partial responder"
    elif emergent >= 3 or increase_frac > 0.1:
        inferred_outcome = "Progression/Relapse"
    else:
        inferred_outcome = "Stable disease"
    
    outcome_records.append({
        "patient_id": pid,
        "group": patient_group(pid),
        "n_timepoints": sub["tp_order"].nunique(),
        "total_variants": n_total,
        "decrease_frac": decrease_frac,
        "increase_frac": increase_frac,
        "stable_frac": n_stable / n_total if n_total > 0 else 0,
        "clearance_rate": clearance_rate,
        "emergent_mutations": emergent,
        "first_burden": first_burden,
        "last_burden": last_burden,
        "burden_change_pct": ((last_burden - first_burden) / first_burden * 100) if first_burden > 0 else 0,
        "inferred_outcome": inferred_outcome
    })

outcome_df = pd.DataFrame(outcome_records)
print(f"\nInferred clinical outcomes ({len(outcome_df)} patients):")
print(outcome_df["inferred_outcome"].value_counts().to_string())
print(f"\nPer-patient details:")
print(outcome_df.to_string(index=False))

# Summary plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
outcome_counts = outcome_df["inferred_outcome"].value_counts()
outcome_counts.plot(kind="bar", ax=axes[0], color=["green", "orange", "red", "gray"], edgecolor="white")
axes[0].set_ylabel("Patients")
axes[0].set_title("Inferred clinical outcomes")
axes[0].set_xticklabels(outcome_counts.index, rotation=30, ha="right")

# Clearance rate vs emergent mutations
scatter = axes[1].scatter(
    outcome_df["clearance_rate"],
    outcome_df["emergent_mutations"],
    c=outcome_df["increase_frac"],
    cmap="RdYlGn_r",
    s=100,
    alpha=0.8,
    edgecolors="black"
)
for _, r in outcome_df.iterrows():
    axes[1].annotate(r["patient_id"], (r["clearance_rate"], r["emergent_mutations"]),
                     fontsize=8, alpha=0.7)
axes[1].set_xlabel("Clearance rate (fraction of mutations cleared)")
axes[1].set_ylabel("Emergent mutations (new at last TP)")
axes[1].set_title("Clearance vs Emergence (color = fraction increasing)")
plt.colorbar(scatter, ax=axes[1], label="Fraction increasing")
plt.tight_layout()
plt.savefig("clinical_outcome_inference.png", dpi=150)
print("Saved clinical_outcome_inference.png")

# ═══════════════════════════════════════════════════════════════
# 5. RESISTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. RESISTANCE ANALYSIS")
print("="*70)

# Focus on LH and C patients with >= 3 timepoints
res_df = longitudinal[longitudinal["patient_id"].isin(long_patients)].copy()
print(f"Resistance analysis on {len(res_df):,} variant calls from {len(long_patients)} patients")

# Strategy: find mutations that:
# (a) Are absent at first timepoint but appear later (emergent)
# (b) Show increasing VAF trend across timepoints

emergent_records = []
for pid in long_patients:
    sub = res_df[res_df["patient_id"] == pid]
    if sub.empty:
        continue
    first_tp = sub["tp_order"].min()
    last_tp = sub["tp_order"].max()
    
    first_variants = set(sub[sub["tp_order"] == first_tp]["variant_id"].unique())
    all_variants = set(sub["variant_id"].unique())
    
    # Emergent: not in first TP
    emergent = all_variants - first_variants
    for var_id in emergent:
        var_data = sub[sub["variant_id"] == var_id].sort_values("tp_order")
        if var_data.empty:
            continue
        first_appearance = var_data["tp_order"].iloc[0]
        last_vaf = var_data["VAF"].iloc[-1]
        max_vaf = var_data["VAF"].max()
        genes = var_data["Genes"].iloc[0]
        effect = var_data["Variant Effect"].iloc[0] if "Variant Effect" in var_data.columns else ""
        emergent_records.append({
            "patient_id": pid,
            "variant_id": var_id,
            "genes": genes,
            "first_tp_appeared": first_appearance,
            "last_tp": last_tp,
            "VAF_at_emergence": var_data["VAF"].iloc[0],
            "VAF_at_last_tp": last_vaf,
            "max_VAF": max_vaf,
            "variant_effect": effect,
            "n_timepoints_detected": len(var_data["tp_order"].unique())
        })

emergent_df = pd.DataFrame(emergent_records)
if len(emergent_df) > 0:
    # Sort by max VAF (higher = more likely clinically relevant)
    emergent_df = emergent_df.sort_values("max_VAF", ascending=False)
    print(f"\nEmergent mutations detected: {len(emergent_df)}")
    print("\nTop emergent mutations (highest VAF):")
    print(emergent_df.head(20).to_string(index=False))
    
    # Per-patient emergent mutation count
    per_patient_emergent = emergent_df.groupby("patient_id").agg(
        n_emergent=("variant_id", "count"),
        mean_VAF=("max_VAF", "mean"),
        top_genes=("genes", lambda x: ", ".join(x.head(5)))
    ).sort_values("n_emergent", ascending=False)
    print("\nPer-patient emergent mutation summary:")
    print(per_patient_emergent.to_string())
    
    # Focus on known resistance mechanisms
    # In HL, resistance often involves:
    # - NF-kB pathway (TNFAIP3, NFKBIA, NFKBIE, CYLD, REL)
    # - JAK-STAT pathway (STAT6, SOCS1)
    # - Immune evasion (B2M, CD58, CIITA)
    # - Epigenetic modifiers (CREBBP, EP300, KMT2D)
    resistance_genes = ["TNFAIP3", "NFKBIA", "NFKBIE", "CYLD", "REL",
                        "STAT6", "SOCS1", "JAK1", "JAK2",
                        "B2M", "CD58", "CIITA",
                        "CREBBP", "EP300", "KMT2D", "ARID1A",
                        "TP53", "MYC", "BCL6", "CARD11"]
    
    resistance_emergent = emergent_df[
        emergent_df["genes"].apply(lambda g: any(rg in str(g).upper().split(",") for rg in resistance_genes))
    ]
    print(f"\nEmergent mutations in known resistance genes: {len(resistance_emergent)}")
    if len(resistance_emergent) > 0:
        print(resistance_emergent.to_string(index=False))
else:
    print("No emergent mutations found.")

# Also check for increasing trajectory mutations in specific genes
print("\nIncreasing trajectory mutations (potential resistance):")
inc_traj = traj[(traj["trajectory"] == "increasing") & 
                (traj["patient_id"].isin(long_patients))]
if len(inc_traj) > 0:
    inc_traj = inc_traj.sort_values("last_VAF", ascending=False)
    # Annotate with gene names
    inc_traj["gene"] = inc_traj["variant_id"].apply(lambda x: x.split(" | ")[1] if " | " in str(x) else x)
    print(inc_traj.head(20).to_string(index=False))
    
    # Check if any are in resistance genes
    inc_resistance = inc_traj[inc_traj["gene"].apply(lambda g: any(rg in str(g).upper() for rg in resistance_genes))]
    if len(inc_resistance) > 0:
        print(f"\nIncreasing mutations in known resistance genes:")
        print(inc_resistance.to_string(index=False))

# Summary plot
if len(emergent_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-patient emergent count
    em_count = emergent_df["patient_id"].value_counts()
    em_count.plot(kind="bar", ax=axes[0], color="crimson", edgecolor="white")
    axes[0].set_ylabel("Number of emergent mutations")
    axes[0].set_title("Emergent mutations per patient")
    axes[0].tick_params(axis="x", rotation=45)
    
    # Emergent mutation VAF distribution
    axes[1].hist(emergent_df["max_VAF"], bins=30, color="darkred", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Max VAF of emergent mutation")
    axes[1].set_ylabel("Count")
    axes[1].set_title("VAF distribution of emergent mutations")
    plt.tight_layout()
    plt.savefig("resistance_analysis.png", dpi=150)
    print("Saved resistance_analysis.png")

# ─── Save all output data ───
df.to_csv("master_with_germline_flag.csv", index=False)
somatic.to_csv("somatic_variants_only.csv", index=False)
outcome_df.to_csv("inferred_clinical_outcomes.csv", index=False)
if len(emergent_df) > 0:
    emergent_df.to_csv("emergent_mutations.csv", index=False)

print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print("="*70)
print("\nSaved files:")
print("  master_with_germline_flag.csv — all variants with germline flag")
print("  somatic_variants_only.csv — somatic (non-germline) variants")
print("  inferred_clinical_outcomes.csv — per-patient outcome inference")
print("  emergent_mutations.csv — resistance candidate mutations")
print("  germline_filtered_group_stratification.png")
print("  driver_mutation_heatmap.png")
print("  clinical_outcome_inference.png")
print("  resistance_analysis.png")
