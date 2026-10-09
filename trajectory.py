import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─── Load pivot VAF matrix ───
df = pd.read_csv("pivot_vaf.csv", header=[0, 1], index_col=0)
print(f"Loaded pivot: {df.shape[0]} variants x {df.shape[1]} (patient,tp) columns")

# Drop artifact rows
artifact_patterns = ["ClinVar |", "Locus |", "Unnamed:"]
artifacts = df.index.to_series().apply(lambda x: any(p in str(x) for p in artifact_patterns))
df_clean = df[~artifacts].copy()
print(f"Clean matrix: {df_clean.shape[0]} variants x {df_clean.shape[1]} columns\n")

# ─── Classify per-patient per-variant ───
def classify_trajectory(vals):
    """Classify a single mutation's VAF trajectory across timepoints.
    
    Rules:
      - undetected: all VAFs == 0
      - increasing: last > 1.5 * first_nonzero
      - decreasing: last < 0.5 * first_nonzero
      - stable: everything else
    """
    vals = np.array(vals, dtype=float)
    if (vals == 0).all():
        return "undetected"
    nonzero_idx = np.where(vals > 0)[0]
    first_nonzero = vals[nonzero_idx[0]]
    last = vals[-1]
    if last > 1.5 * first_nonzero:
        return "increasing"
    elif last < 0.5 * first_nonzero:
        return "decreasing"
    else:
        return "stable"

patient_ids = df_clean.columns.get_level_values(0).unique()
print(f"Processing {len(patient_ids)} patients...")

records = []
for pid in patient_ids:
    # Select columns for this patient
    pid_cols = df_clean.columns.get_level_values(0) == pid
    sub = df_clean.loc[:, pid_cols]
    sub.columns = sub.columns.droplevel(0)  # now just tp_label
    
    # Skip patients with < 2 timepoints
    if sub.shape[1] < 2:
        continue
    
    for variant_id in sub.index:
        vals = sub.loc[variant_id].values.astype(float)
        traj = classify_trajectory(vals)
        if traj != "undetected":
            # Get first non-zero and last values
            nonzero_idx = np.where(vals > 0)[0]
            first = vals[nonzero_idx[0]] if len(nonzero_idx) > 0 else 0
            last = vals[-1]
            records.append({
                "patient_id": pid,
                "variant_id": variant_id,
                "trajectory": traj,
                "first_VAF": first,
                "last_VAF": last,
                "n_timepoints": sub.shape[1],
                "first_tp": sub.columns[nonzero_idx[0]] if len(nonzero_idx) > 0 else "",
                "last_tp": sub.columns[-1]
            })

traj_df = pd.DataFrame(records)
print(f"Total variant-patient trajectories classified: {len(traj_df)}")

# ─── Summary ───
print("\n" + "="*70)
print("TRAJECTORY CLASSIFICATION (per patient per variant)")
print("="*70)
summary = traj_df["trajectory"].value_counts()
print(summary.to_string())
print(f"\nPercentages:")
print((summary / summary.sum() * 100).round(1).astype(str) + "%")

# ─── Top per category ───
print("\n" + "="*70)
print("EXAMPLES PER CATEGORY")
print("="*70)
for cat in ["increasing", "decreasing", "stable"]:
    sub = traj_df[traj_df["trajectory"] == cat]
    print(f"\n  {cat.upper()} ({len(sub)} cases):")
    display = sub.sort_values("first_VAF", ascending=False).head(8)
    for _, r in display.iterrows():
        print(f"    {r['patient_id']} | {r['variant_id']}")
        print(f"      T{r['first_tp']}: VAF={r['first_VAF']:.3f}  ->  T{r['last_tp']}: VAF={r['last_VAF']:.3f}")

# ─── Per-patient breakdown ───
print("\n" + "="*70)
print("PER-PATIENT TRAJECTORY BREAKDOWN")
print("="*70)
pp_summary = traj_df.groupby("patient_id")["trajectory"].value_counts().unstack(fill_value=0)
# Reorder columns
for c in ["increasing", "decreasing", "stable"]:
    if c not in pp_summary.columns:
        pp_summary[c] = 0
pp_summary = pp_summary[["increasing", "decreasing", "stable"]]
pp_summary["total_detected"] = pp_summary.sum(axis=1)
pp_summary = pp_summary.sort_values("total_detected", ascending=False)
print(pp_summary.to_string())

# ─── Summary across patients ───
print("\n" + "="*70)
print("AGGREGATE TRAJECTORY PROPORTIONS PER PATIENT")
print("="*70)
pct = pp_summary.div(pp_summary["total_detected"], axis=0) * 100
print(pct.round(1).astype(str) + "%")

# ─── Save ───
traj_df.to_csv("trajectory_per_patient_per_variant.csv", index=False)
pp_summary.to_csv("trajectory_patient_summary.csv")
print("\nSaved:")
print("  trajectory_per_patient_per_variant.csv")
print("  trajectory_patient_summary.csv")
