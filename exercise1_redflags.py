import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
trusted = pd.read_csv("trusted_variants.csv")

inc = traj[traj["trajectory"] == "increasing"].copy()
inc["gene"] = inc["variant_id"].str.split(" | ", n=1, regex=False).str[-1]

print("=" * 65)
print("HUNTING THE RED FLAGS — INCREASING MUTATIONS")
print("=" * 65)

print(f"\nTotal increasing trajectories: {len(inc)}")
print(f"Unique increasing variant loci: {inc['variant_id'].nunique()}")
print(f"Unique increasing genes: {inc['gene'].nunique()}")
print(f"Patients with increasing clones: {inc['patient_id'].nunique()}")

inc["fold"] = inc["last_VAF"] / inc["first_VAF"].clip(1e-6)

print("\n--- ALL INCREASING (sorted by fold change) ---")
inc_sorted = inc.sort_values("fold", ascending=False)
print(inc_sorted[["patient_id", "variant_id", "first_VAF", "last_VAF", "fold"]].to_string(index=False))
# Note: inc_sorted does NOT have is_driver yet — will be added below

# Cross-reference with driver gene list
print("\n" + "=" * 65)
print("CROSS-REFERENCE WITH DRIVER GENES")
print("=" * 65)

# Load the driver gene list if available, or infer from data
driver_genes = ["STAT3", "STAT6", "STAT5B", "GNA13", "GNAI2", "SOCS1", "TNFAIP3",
                "BCL2", "BCL6", "MYC", "MYB", "REL", "NFKBIA", "NFKBIE", "IKBKB",
                "CARD11", "CD58", "B2M", "CIITA", "CD274", "PDCD1LG2", "JAK2",
                "JAK3", "IL4R", "IL13", "NOTCH1", "CREBBP", "EP300", "TP53",
                "FAS", "PIM1", "RHOA", "ITPKB", "XPO1", "MEF2B", "EZH2",
                "TNFRSF14", "IRF4", "IRF8", "CSF1R", "CASP8", "NFKBIZ",
                "ACTB", "ACTG1", "CALR", "BRAF", "KRAS", "NRAS", "MAP2K1",
                "PIK3CD", "PIK3CA", "PIK3R1", "PTEN", "MTOR", "TSC1", "TSC2",
                "KMT2D", "HIST1H1E", "ARID1A", "SMARCA4", "BCOR", "BCORL1",
                "ZFP36L1", "ZFP36L2", "DDX3X", "FOXO1", "KLF2", "PRDM1",
                "RBM15", "SPEN", "ID3", "TCF3", "CCND3", "CDKN2A", "CDKN2B",
                "RB1", "ATM", "CHEK2", "BRCA2", "FAT1", "NCOR2", "TBL1XR1",
                "NFKBIE", "NFKB1", "MAP3K14", "TRAF3", "BIRC3", "LTBR",
                "CYLD", "CARD11", "MALT1", "BCL10", "PRKCB", "PRKCD",
                "PTPN1", "PTPN2", "PTPN6", "SHP1", "SHP2", "DUSP2",
                "JUNB", "JUND", "FOS", "FOSB", "ATF3", "BATF", "BATF3",
                "IRF1", "IRF2", "IRF3", "IRF5", "IRF7", "STAT1", "STAT2",
                "STAT4", "STAT5A", "STAT5B", "STAT6"]

known_drivers = [g.upper() for g in driver_genes]
inc["is_driver"] = inc["gene"].str.upper().isin(known_drivers)

print(f"\nIncreasing in driver genes: {inc['is_driver'].sum()} / {len(inc)}")
print(f"Non-driver increasing: {(~inc['is_driver']).sum()} / {len(inc)}")

print("\n--- DRIVER-GENE INCREASING ---")
driver_inc = inc[inc["is_driver"]].sort_values("fold", ascending=False)
print(driver_inc[["patient_id", "gene", "first_VAF", "last_VAF", "fold"]].to_string(index=False))

print("\n--- NON-DRIVER INCREASING ---")
nondriver_inc = inc[~inc["is_driver"]].sort_values("fold", ascending=False)
print(nondriver_inc[["patient_id", "gene", "first_VAF", "last_VAF", "fold"]].to_string(index=False))

# Summary
print("\n" + "=" * 65)
print("ANSWER: TWO RED FLAGS")
print("=" * 65)
print("After filtering, there are NOT exactly 2 — there are 80 increasing trajectories.")
print("But if the question means '2 most extreme':")
inc_sorted = inc.sort_values("fold", ascending=False)
top2 = inc_sorted.head(2)
for _, r in top2.iterrows():
    print(f"\n  RED FLAG: {r['gene']}")
    print(f"    Patient: {r['patient_id']}")
    print(f"    VAF: {r['first_VAF']:.4f} -> {r['last_VAF']:.4f} ({r['fold']:.1f}x)")
    print(f"    Variant: {r['variant_id']}")
    print(f"    Driver gene: {'YES' if r['is_driver'] else 'NO'}")
