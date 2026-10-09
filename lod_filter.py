import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)

df = pd.read_csv("master_with_germline_flag.csv")
df["group"] = df["patient_id"].apply(
    lambda x: "BR" if x.startswith("BR") else ("LH" if x.startswith("LH") else ("C" if x.startswith("C") else "Other"))
)

# Normal samples
normal = df[df["timepoint"].str.lower().str.contains("dna normal", na=False)].copy()
print(f"DNA NORMAL rows: {len(normal)}")
print(f"Unique patients with normals: {normal['patient_id'].nunique()}")
print(f"Normal timepoints: {normal['timepoint'].unique()}")
print(f"\nVAF in normals: mean={normal['VAF'].mean():.4f}, median={normal['VAF'].median():.4f}")

# ─── Step 1: Compute per-gene background from normals ───
print("\n" + "="*70)
print("STEP 1: PER-GENE BACKGROUND FROM NORMALS")
print("="*70)

# For each gene, compute background statistics from all normal samples
# Parse individual genes from multi-gene entries
def split_genes(gene_str):
    return [g.strip() for g in str(gene_str).split(",")]

# Explode multi-gene rows so each gene gets its own row
normal_exploded = normal.copy()
normal_exploded["gene_list"] = normal_exploded["Genes"].apply(split_genes)
normal_exploded = normal_exploded.explode("gene_list")

# Compute per-gene background
gene_background = normal_exploded.groupby("gene_list").agg(
    n_variants=("variant_id", "count"),
    n_unique_loci=("variant_id", "nunique"),
    mean_VAF=("VAF", "mean"),
    median_VAF=("VAF", "median"),
    std_VAF=("VAF", "std"),
    max_VAF=("VAF", "max"),
    p95_VAF=("VAF", lambda x: x.quantile(0.95)),
    p99_VAF=("VAF", lambda x: x.quantile(0.99)),
).reset_index()

gene_background = gene_background.sort_values("n_variants", ascending=False)
print(f"Background computed for {len(gene_background)} genes")
print("Top 20 genes by variant count in normals:")
print(gene_background.head(20).to_string(index=False))

# Also compute a global background for genes with too few observations
# (some genes may not appear in normals)
global_mean = normal["VAF"].mean()
global_std = normal["VAF"].std()
global_p99 = normal["VAF"].quantile(0.99)
print(f"\nGlobal background (fallback): mean={global_mean:.4f}, std={global_std:.4f}, p99={global_p99:.4f}")

# Fill missing genes (those never seen in normals) with global values
all_genes_in_panel = df["Genes"].apply(split_genes).explode().unique()
missing_genes = set(all_genes_in_panel) - set(gene_background["gene_list"])
print(f"Genes with no normal data (will use global): {len(missing_genes)}")

for mg in missing_genes:
    gene_background = pd.concat([
        gene_background,
        pd.DataFrame([{
            "gene_list": mg,
            "n_variants": 0,
            "n_unique_loci": 0,
            "mean_VAF": global_mean,
            "median_VAF": global_mean,
            "std_VAF": global_std,
            "max_VAF": 0,
            "p95_VAF": global_mean + 1.645 * global_std,
            "p99_VAF": global_mean + 2.326 * global_std
        }])
    ], ignore_index=True)

gene_background = gene_background.set_index("gene_list")

# ─── Step 2: Derive a dynamic LoD from background ───
print("\n" + "="*70)
print("STEP 2: DYNAMIC LoD FROM BACKGROUND")
print("="*70)

# LoD = mean + 3*SD per gene (99.7% confidence), with a floor of 0.01 (1% VAF)
# Also compute a variant-level LoD that accounts for how many times we've seen this locus
gene_background["LoD_mean_3sd"] = gene_background.apply(
    lambda r: max(r["mean_VAF"] + 3 * r["std_VAF"], 0.01) if r["n_variants"] >= 2 else global_p99,
    axis=1
)

# For frequently observed genes, also compute a variant-specific background
# (if a specific locus appears in normals, its individual background is more reliable)
variant_background = normal.groupby("variant_id").agg(
    n_normals=("patient_id", "nunique"),
    mean_VAF=("VAF", "mean"),
    max_VAF=("VAF", "max"),
    std_VAF=("VAF", "std"),
).reset_index()

variant_background["LoD_variant"] = variant_background.apply(
    lambda r: max(r["mean_VAF"] + 3 * r["std_VAF"], 0.01) if r["n_normals"] >= 2 else None,
    axis=1
)
variant_background = variant_background.set_index("variant_id")

print(f"Gene-level LoDs computed for {len(gene_background)} genes")
print(f"Variant-level LoDs computed for {variant_background['LoD_variant'].notna().sum()} variants")

# Show some LoDs
print("\nSample LoD values by gene (top 15 by frequency):")
lod_show = gene_background[gene_background["n_variants"] >= 3].head(15)[["n_variants", "mean_VAF", "std_VAF", "LoD_mean_3sd"]]
print(lod_show.to_string())

# ─── Step 3: Apply the filter ───
print("\n" + "="*70)
print("STEP 3: APPLY THE FILTER")
print("="*70)

# For each row in the full dataset, determine the appropriate LoD
def get_lod(row):
    gene = row["Genes"]
    variant = row["variant_id"]
    
    # Prefer variant-level LoD if available
    if variant in variant_background.index:
        v_lod = variant_background.loc[variant, "LoD_variant"]
        if pd.notna(v_lod):
            return v_lod
    
    # Fall back to gene-level LoD
    if gene in gene_background.index:
        return gene_background.loc[gene, "LoD_mean_3sd"]
    
    # Global fallback
    return max(global_mean + 3 * global_std, 0.01)

df["LoD"] = df.apply(get_lod, axis=1)
df["above_LoD"] = df["VAF"] > df["LoD"]
df["filter_pass"] = df["above_LoD"]

print(f"Variants above LoD: {df['above_LoD'].sum():,} / {len(df):,} ({df['above_LoD'].mean()*100:.1f}%)")
print(f"LoD stats: mean={df['LoD'].mean():.4f}, median={df['LoD'].median():.4f}, "
      f"p25={df['LoD'].quantile(0.25):.4f}, p75={df['LoD'].quantile(0.75):.4f}")

# How does this compare with the germline flag?
print(f"\nComparison with germline flag:")
print(pd.crosstab(df["germline"], df["above_LoD"], margins=True))

# ─── Step 4: The allow list — rescuing known drivers ───
print("\n" + "="*70)
print("STEP 4: ALLOW LIST — RESCUING KNOWN DRIVERS")
print("="*70)

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

def is_driver(gene_str):
    genes = [g.strip() for g in str(gene_str).split(",")]
    return any(g in HL_DRIVERS for g in genes)

df["is_driver"] = df["Genes"].apply(is_driver)

# Rescue: if a variant is in a known driver gene AND has VAF > 0.01 (1%), keep it
# even if it's below the per-gene LoD
driver_rescue = df["is_driver"] & (df["VAF"] > 0.01) & ~df["above_LoD"]
n_rescued = driver_rescue.sum()
print(f"Variants rescued by driver allow list: {n_rescued}")

# Apply rescue
df["filter_pass"] = df["filter_pass"] | driver_rescue

# Also rescue any variant in a driver gene from a non-normal sample with VAF > 0.02
# in a patient that has >= 2 timepoints (likely real somatic)
non_normal = df[~df["timepoint"].str.lower().str.contains("dna normal", na=False)]
patient_tp_counts = non_normal.groupby("patient_id")["tp_label"].nunique()
longitudinal_pids = patient_tp_counts[patient_tp_counts >= 2].index
df["is_longitudinal"] = df["patient_id"].isin(longitudinal_pids)

driver_longitudinal_rescue = (df["is_driver"] & df["is_longitudinal"] & 
                               (df["VAF"] > 0.015) & ~df["filter_pass"])
n_rescued2 = driver_longitudinal_rescue.sum()
print(f"Additional rescues (driver + longitudinal + VAF>0.015): {n_rescued2}")

df["filter_pass"] = df["filter_pass"] | driver_longitudinal_rescue

print(f"Total rescued: {n_rescued + n_rescued2}")
print(f"Final passed filter: {df['filter_pass'].sum():,} / {len(df):,} ({df['filter_pass'].mean()*100:.1f}%)")

# ─── Step 5: The block list — killing recurrent artifacts ───
print("\n" + "="*70)
print("STEP 5: BLOCK LIST — KILLING RECURRENT ARTIFACTS")
print("="*70)

# Strategy: if a variant appears in >= 3 normal samples at VAF > 0.02, block it
# everywhere (it's almost certainly a recurrent artifact/germline)

# Also: if a variant appears at high VAF (>0.5) in normal and has mean VAF > 0.1,
# it's likely germline — block it
block_candidates = normal.groupby("variant_id").agg(
    n_normals_present=("patient_id", "nunique"),
    mean_VAF_in_normals=("VAF", "mean"),
    max_VAF_in_normals=("VAF", "max")
).reset_index()

# Rule 1: Present in >= 3 normals at VAF > 0.02
block_rule1 = block_candidates[
    (block_candidates["n_normals_present"] >= 3) & 
    (block_candidates["mean_VAF_in_normals"] > 0.02)
]

# Rule 2: Present in normal at VAF > 0.5 (likely germline) 
block_rule2 = block_candidates[
    block_candidates["max_VAF_in_normals"] > 0.5
]

# Rule 3: Present in any normal AND seen at very high frequency (>50% of normal samples)
total_normals = normal["patient_id"].nunique()
block_rule3 = block_candidates[
    (block_candidates["n_normals_present"] >= total_normals * 0.5) &
    (block_candidates["mean_VAF_in_normals"] > 0.05)
]

block_list = pd.concat([
    block_rule1["variant_id"],
    block_rule2["variant_id"],
    block_rule3["variant_id"]
]).unique()

print(f"Block list size: {len(block_list)} variant IDs")
print(f"  Rule 1 (>=3 normals, mean VAF>0.02): {len(block_rule1)}")
print(f"  Rule 2 (max VAF>0.5 in normal): {len(block_rule2)}")
print(f"  Rule 3 (>50% of normals, mean VAF>0.05): {len(block_rule3)}")

# Show the top blocked variants
blocked_variants_detail = block_candidates[
    block_candidates["variant_id"].isin(block_list)
].sort_values("n_normals_present", ascending=False)
print("\nTop blocked variants:")
print(blocked_variants_detail.head(20).to_string(index=False))

# Apply block list (but NOT to driver genes — allow list takes priority)
df["is_blocked"] = df["variant_id"].isin(block_list)
# Only block if NOT a driver
df["filter_pass"] = df["filter_pass"] & ~(df["is_blocked"] & ~df["is_driver"])

n_blocked = (df["is_blocked"] & ~df["is_driver"]).sum()
print(f"\nVariants blocked (non-driver): {n_blocked}")
print(f"Final passed filter after block list: {df['filter_pass'].sum():,} / {len(df):,} ({df['filter_pass'].mean()*100:.1f}%)")

# ─── Summary table ───
print("\n" + "="*70)
print("FILTERING SUMMARY")
print("="*70)
print(f"{'Filter step':40s} {'Remaining':>10s} {'% of total':>10s}")
print("-"*60)

n_total = len(df)
print(f"{'Total raw variants':40s} {n_total:>10,} {'100.0':>10s}")
print(f"{'Above LoD (gene-level dynamic threshold)':40s} {df['above_LoD'].sum():>10,} {df['above_LoD'].mean()*100:>9.1f}%")
print(f"{'After driver allow list rescue':40s} {(df['filter_pass']).sum():>10,} {(df['filter_pass']).mean()*100:>9.1f}%")
print(f"{'After block list removal':40s} {(df['filter_pass']).sum():>10,} {(df['filter_pass']).mean()*100:>9.1f}%")

# Save filtered dataset
df["final_label"] = df["filter_pass"].map({True: "pass", False: "fail"})
# Annotate reason
def annotate_reason(row):
    if row["final_label"] == "pass":
        if row["is_driver"] and not row["above_LoD"]:
            return "driver_rescue"
        return "above_LoD"
    if row["is_blocked"] and not row["is_driver"]:
        return "blocked_artifact"
    if row["germline"]:
        return "germline"
    return "below_LoD"

df["filter_reason"] = df.apply(annotate_reason, axis=1)

print("\nFilter outcome breakdown:")
print(df["filter_reason"].value_counts().to_string())

# Save
df.to_csv("filtered_variants.csv", index=False)
gene_background.to_csv("gene_background_stats.csv")
blocked_variants_detail.to_csv("block_list.csv", index=False)
print("\nSaved:")
print("  filtered_variants.csv  — Full dataset with per-variant filter decisions")
print("  gene_background_stats.csv  — Per-gene background & LoD")
print("  block_list.csv  — Recurrent artifact variants")

# ─── Visualization ───
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. LoD distribution by gene
lod_genes = gene_background[gene_background["n_variants"] >= 2].sort_values("LoD_mean_3sd", ascending=False)
top_lod = lod_genes.head(20)
axes[0, 0].barh(range(len(top_lod)), top_lod["LoD_mean_3sd"].values, color="crimson", edgecolor="white")
axes[0, 0].set_yticks(range(len(top_lod)))
axes[0, 0].set_yticklabels(top_lod.index, fontsize=8)
axes[0, 0].set_xlabel("LoD (VAF threshold)")
axes[0, 0].set_title("Top 20 genes by LoD (mean + 3 SD in normals)")

# 2. Filter outcome pie
outcome_counts = df["filter_reason"].value_counts()
colors = {"above_LoD": "green", "driver_rescue": "orange", "below_LoD": "gray", 
           "germline": "blue", "blocked_artifact": "red"}
axes[0, 1].pie(outcome_counts.values, labels=outcome_counts.index, autopct="%1.1f%%",
               colors=[colors.get(k, "lightgray") for k in outcome_counts.index],
               startangle=90)
axes[0, 1].set_title("Filter outcome distribution")

# 3. VAF distribution before/after filtering
axes[1, 0].hist(df[df["filter_pass"]]["VAF"], bins=80, alpha=0.7, color="green", 
                 label="Passed filter", edgecolor="white")
axes[1, 0].hist(df[~df["filter_pass"]]["VAF"], bins=80, alpha=0.5, color="red",
                 label="Failed filter", edgecolor="white")
axes[1, 0].set_xlabel("VAF")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("VAF distribution: passed vs failed")
axes[1, 0].legend()

# 4. Per-patient variant count before/after
before = df.groupby("patient_id").size().rename("before")
after = df[df["filter_pass"]].groupby("patient_id").size().rename("after")
compare = pd.DataFrame({"before": before, "after": after}).fillna(0)
compare = compare.sort_values("before", ascending=False)
compare.head(20).plot(kind="bar", ax=axes[1, 1], color=["lightcoral", "seagreen"], edgecolor="white")
axes[1, 1].set_ylabel("Variants")
axes[1, 1].set_title("Top 20 patients: variants before vs after filtering")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("dynamic_LoD_filtering.png", dpi=150)
print("\nSaved dynamic_LoD_filtering.png")
