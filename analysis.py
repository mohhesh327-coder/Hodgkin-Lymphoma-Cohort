import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re, warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

data_dir = os.path.dirname(os.path.abspath(__file__))
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".xlsx")])
print(f"Found {len(all_files)} xlsx files\n")

# ─── Parse all xlsx files into a unified table ───
all_rows = []

for f in all_files:
    fpath = os.path.join(data_dir, f)
    patient_id = re.sub(r"_OK|\.xlsx", "", f)
    xls = pd.ExcelFile(fpath)

    for sheet in xls.sheet_names:
        sheet_clean = sheet.lower().replace("\ufffd", "u").replace(" ", "").strip()
        if sheet_clean in ("hoja1", "summary", ""):
            continue

        df_raw = pd.read_excel(fpath, sheet_name=sheet, header=None)
        if len(df_raw) < 2:
            continue

        # Find the header row: look for a row containing "Locus" and "Genes"
        header_row_idx = None
        for idx in range(min(25, len(df_raw))):
            row_vals = [str(v).strip() for v in df_raw.iloc[idx].tolist() if pd.notna(v)]
            if "Locus" in row_vals:
                header_row_idx = idx
                break

        if header_row_idx is None:
            continue

        # If this is a COMUN / common variant sheet with no data rows, skip
        if header_row_idx + 1 >= len(df_raw):
            continue

        header = [str(h) if pd.notna(h) else f"col_{j}" for j, h in enumerate(df_raw.iloc[header_row_idx].tolist())]
        data_df = df_raw.iloc[header_row_idx + 1:].copy()
        data_df.columns = header
        data_df = data_df.reset_index(drop=True)

        # Get sample name from metadata (row 5 or nearby)
        sample_name = sheet
        for meta_row in range(min(15, len(df_raw))):
            meta_vals = [str(v) for v in df_raw.iloc[meta_row].tolist() if pd.notna(v)]
            for mv in meta_vals:
                if mv.startswith("##sampleNames="):
                    sample_name = mv.replace("##sampleNames=", "")
                    break
            if sample_name != sheet:
                break

        # Determine which frequency column to use
        possible_freq = ["% Frequency", "Allele Frequency %", "Variant Frequency", "Allele Frequency", "AF", "Frequency"]
        freq_col = None
        for pf in possible_freq:
            if pf in data_df.columns:
                freq_col = pf
                break

        # Determine which columns to keep
        keep_cols = {}
        if "Genes" in data_df.columns:
            keep_cols["Genes"] = "Genes"
        if freq_col:
            keep_cols["% Frequency"] = freq_col
        if "Coverage" in data_df.columns:
            keep_cols["Coverage"] = "Coverage"
        if "Locus" in data_df.columns:
            keep_cols["Locus"] = "Locus"
        if "Genotype" in data_df.columns:
            keep_cols["Genotype"] = "Genotype"
        if "Type" in data_df.columns:
            keep_cols["Type"] = "Type"
        if "Location" in data_df.columns:
            keep_cols["Location"] = "Location"
        if "Variant Effect" in data_df.columns:
            keep_cols["Variant Effect"] = "Variant Effect"
        if "ClinVar" in data_df.columns:
            keep_cols["ClinVar"] = "ClinVar"
        if "Ref" in data_df.columns:
            keep_cols["Ref"] = "Ref"
        if "Observed Allele" in data_df.columns:
            keep_cols["Observed Allele"] = "Observed Allele"
        if "Filter" in data_df.columns:
            keep_cols["Filter"] = "Filter"

        if not keep_cols:
            continue

        # Handle duplicate column names by taking first occurrence
        col_counts = data_df.columns.value_counts()
        dup_cols = col_counts[col_counts > 1].index.tolist()

        # Build subset
        sub = pd.DataFrame()
        for new_name, old_name in keep_cols.items():
            if old_name in data_df.columns:
                if old_name in dup_cols:
                    # Take the first occurrence
                    cols = data_df.columns.tolist()
                    idx = cols.index(old_name)
                    sub[new_name] = data_df.iloc[:, idx]
                else:
                    sub[new_name] = data_df[old_name]
        sub["patient_id"] = patient_id
        sub["timepoint"] = sample_name
        sub["sheet_name"] = sheet

        # Convert frequency to numeric
        if "% Frequency" in sub.columns:
            sub["% Frequency"] = pd.to_numeric(sub["% Frequency"], errors="coerce").fillna(0)
        else:
            sub["% Frequency"] = 0.0

        if "Coverage" in sub.columns:
            sub["Coverage"] = pd.to_numeric(sub["Coverage"], errors="coerce").fillna(0)

        all_rows.append(sub)

master = pd.concat(all_rows, ignore_index=True, sort=False)
master["VAF"] = master["% Frequency"] / 100.0

print(f"Total variant rows loaded: {len(master):,}")
print(f"Number of patients: {master['patient_id'].nunique()}")
print(f"Number of timepoints/samples: {master['timepoint'].nunique()}")
print(f"Columns: {master.columns.tolist()}")
print(master.head(10).to_string())

master.to_csv("cleaned_variants.csv", index=False)
print("\nSaved cleaned_variants.csv")

# ═══════════════════════════════════════════════════════
# Step 2 — Inspect cohort structure
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: COHORT STRUCTURE")
print("="*70)

# Patient prefix groups
def patient_group(pid):
    if pid.startswith("BR"): return "BR (biopsy/reference)"
    if pid.startswith("LH"): return "LH (Hodgkin lymphoma)"
    if pid.startswith("C"):  return "C (other)"
    return pid

master["cohort_group"] = master["patient_id"].apply(patient_group)
print(f"\nPatients per group:")
print(master.groupby("cohort_group")["patient_id"].nunique())

print(f"\nSample types per patient (~first sheet name per patient):")
print(master.groupby("patient_id")["sheet_name"].apply(lambda x: list(set(x))).head(10))

print(f"\nTotal unique genes detected: {master['Genes'].nunique()}")
print(f"Total unique loci: {master['Locus'].nunique()}")
print(f"Variant types: {master['Type'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════
# Step 3 — Variants per patient
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: VARIANTS PER PATIENT")
print("="*70)

vpp = master.groupby("patient_id").agg(
    total_variants=("Locus", "count"),
    unique_loci=("Locus", "nunique"),
    unique_genes=("Genes", "nunique"),
    mean_VAF=("VAF", "mean"),
    timepoints=("timepoint", "nunique")
).sort_values("total_variants", ascending=False)

print(vpp.to_string())
print(f"\nSummary: median variants/patient = {vpp['total_variants'].median():.0f}, "
      f"range = {vpp['total_variants'].min()} - {vpp['total_variants'].max()}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vpp["total_variants"].plot(kind="hist", bins=30, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_xlabel("Total variants per patient")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of total variants per patient")

vpp["unique_genes"].plot(kind="hist", bins=30, ax=axes[1], color="coral", edgecolor="white")
axes[1].set_xlabel("Unique genes per patient")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of unique mutated genes per patient")
plt.tight_layout()
plt.savefig("step3_variants_per_patient.png", dpi=150)
print("\nSaved step3_variants_per_patient.png")

# ═══════════════════════════════════════════════════════
# Step 4 — VAF distribution
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: VAF DISTRIBUTION")
print("="*70)

vaf = master["VAF"].dropna()
print(f"VAF stats (n={len(vaf):,}):")
print(f"  Mean: {vaf.mean():.4f}")
print(f"  Median: {vaf.median():.4f}")
print(f"  Q1: {vaf.quantile(0.25):.4f}, Q3: {vaf.quantile(0.75):.4f}")
print(f"  % VAF > 0.05: {(vaf > 0.05).mean()*100:.1f}%")
print(f"  % VAF > 0.30: {(vaf > 0.30).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(vaf, bins=80, color="darkgreen", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("VAF")
axes[0].set_ylabel("Count")
axes[0].set_title("VAF distribution (all variants)")
axes[0].axvline(0.05, color="red", ls="--", alpha=0.5, label="VAF=0.05")
axes[0].axvline(0.30, color="orange", ls="--", alpha=0.5, label="VAF=0.30")
axes[0].legend()

# Log-scale view
axes[1].hist(vaf, bins=80, color="darkgreen", edgecolor="white", alpha=0.8)
axes[1].set_yscale("log")
axes[1].set_xlabel("VAF")
axes[1].set_ylabel("Count (log)")
axes[1].set_title("VAF distribution (log scale)")
axes[1].axvline(0.05, color="red", ls="--", alpha=0.5)
axes[1].axvline(0.30, color="orange", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("step4_vaf_distribution.png", dpi=150)
print("Saved step4_vaf_distribution.png")

# ═══════════════════════════════════════════════════════
# Step 5 — Top mutated genes
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: TOP MUTATED GENES")
print("="*70)

# Count how many patients have at least one variant in each gene
gene_patient_count = master.groupby("Genes")["patient_id"].nunique().sort_values(ascending=False)
gene_variant_count = master.groupby("Genes")["Locus"].count().sort_values(ascending=False)

top_n = 20
top_genes = pd.DataFrame({
    "n_patients": gene_patient_count,
    "n_variants": gene_variant_count
}).head(top_n)

print(f"Top {top_n} most frequently mutated genes (by patients affected):")
print(top_genes.to_string())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = plt.cm.YlOrRd_r(np.linspace(0.3, 0.9, top_n))

top_genes["n_patients"].sort_values().plot(kind="barh", ax=axes[0], color=colors[::-1], edgecolor="white")
axes[0].set_xlabel("Number of patients with variant")
axes[0].set_title(f"Top {top_n} genes by affected patients")

top_genes["n_variants"].sort_values().plot(kind="barh", ax=axes[1], color=colors[::-1], edgecolor="white")
axes[1].set_xlabel("Number of variant occurrences")
axes[1].set_title(f"Top {top_n} genes by total variant count")
plt.tight_layout()
plt.savefig("step5_top_mutated_genes.png", dpi=150)
print("Saved step5_top_mutated_genes.png")

# ═══════════════════════════════════════════════════════
# Step 6 — Timepoints per patient
# ═══════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6: TIMEPOINTS PER PATIENT")
print("="*70)

tp_count = master.groupby("patient_id")["timepoint"].nunique().sort_values(ascending=False)
print(f"Timepoints per patient:")
print(tp_count.to_string())
print(f"\nSummary: median timepoints/patient = {tp_count.median():.0f}, "
      f"range = {tp_count.min()} - {tp_count.max()}")

# Detailed timepoint info for patients with many timepoints
print("\nDetailed timepoints (patients with 5+ timepoints):")
for pid in tp_count[tp_count >= 5].index:
    tps = master[master["patient_id"] == pid]["timepoint"].unique()
    print(f"  {pid}: {sorted(tps)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tp_count.plot(kind="hist", bins=20, ax=axes[0], color="purple", edgecolor="white")
axes[0].set_xlabel("Number of timepoints")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of timepoints per patient")

# Bar plot
tp_count.sort_values().plot(kind="barh", ax=axes[1], color="mediumpurple", edgecolor="white", figsize=(10, 12))
axes[1].set_xlabel("Number of timepoints")
axes[1].set_ylabel("Patient")
axes[1].set_title("Timepoints per patient")
plt.tight_layout()
plt.savefig("step6_timepoints_per_patient.png", dpi=150)
print("Saved step6_timepoints_per_patient.png")

print("\n" + "="*70)
print("ALL STEPS COMPLETED")
print("="*70)
