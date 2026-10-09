import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("filtered_variants.csv")
print(f"Total rows: {len(df):,}")

# Keep only trusted (pass) rows
trusted = df[df["filter_pass"]].copy()
print(f"Trusted (filter_pass=True): {len(trusted):,} ({len(trusted)/len(df)*100:.1f}%)")

# Report what this removed
removed = df[~df["filter_pass"]]
print(f"\nRemoved: {len(removed)} rows")
print("Removed by reason:")
print(removed["filter_reason"].value_counts().to_string())

# ─── Rebuild the variant_id column ───
trusted["variant_id"] = trusted["Locus"] + " | " + trusted["Genes"]

# ─── Pivot: VAF matrix ───
pivot_vaf = trusted.pivot_table(
    index="variant_id",
    columns=["patient_id", "tp_label"],
    values="VAF",
    aggfunc="max"
).fillna(0)

print(f"\nVAF pivot: {pivot_vaf.shape[0]} trusted variants x {pivot_vaf.shape[1]} (patient,tp) columns")

# ─── Binary matrix (VAF > 0.05) ───
pivot_binary = (pivot_vaf > 0.05).astype(int)
print(f"Binary pivot: {pivot_binary.shape[0]} x {pivot_binary.shape[1]}")

# ─── Summary stats ───
print(f"\nTrusted variants per patient:")
vp = trusted.groupby("patient_id").agg(
    n_variants=("variant_id", "count"),
    n_unique=("variant_id", "nunique"),
    n_genes=("Genes", "nunique")
).sort_values("n_variants", ascending=False)
print(vp.to_string())

print(f"\nTotal patients: {trusted['patient_id'].nunique()}")
print(f"Total unique variant loci: {trusted['variant_id'].nunique()}")
print(f"Total unique genes: {trusted['Genes'].nunique()}")

# ─── Save ───
pivot_vaf.to_csv("trusted_pivot_vaf.csv")
pivot_binary.to_csv("trusted_pivot_binary.csv")
trusted.to_csv("trusted_variants.csv", index=False)

print("\nSaved:")
print("  trusted_variants.csv        — filtered variant table (pass only)")
print("  trusted_pivot_vaf.csv       — VAF pivot (trusted variants only)")
print("  trusted_pivot_binary.csv    — binary pivot (VAF>0.05)")
