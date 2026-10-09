# ── Cell 12: Per-(Patient, Timepoint) max VAF table ──

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("trusted_variants.csv")

# One row per (patient, tp_label): max VAF at that timepoint
# Group by patient + tp_label, keep tp_order for sorting
tumor_burden = (
    df.groupby(["patient_id", "tp_label", "tp_order"], as_index=False)
    .agg(max_vaf=("VAF", "max"),
         n_variants=("VAF", "count"),
         sum_vaf=("VAF", "sum"),
         genes_detected=("Genes", lambda x: ",".join(sorted(set(x)))))
    .sort_values(["patient_id", "tp_order"])
    .reset_index(drop=True)
)

# Show
print(f"Shape: {tumor_burden.shape}  |  Patients: {tumor_burden['patient_id'].nunique()}")
print(f"\nSample rows:")
print(tumor_burden.head(10).to_string(index=False))

# Quick look: does max_vaf tend to decrease over time?
print("\n--- Max VAF by TP order (all patients) ---")
print(tumor_burden.groupby("tp_label")["max_vaf"].agg(["mean", "median", "count"]).to_string())

tumor_burden.to_csv("tumor_burden_per_tp.csv", index=False)
print("\nSaved tumor_burden_per_tp.csv")
