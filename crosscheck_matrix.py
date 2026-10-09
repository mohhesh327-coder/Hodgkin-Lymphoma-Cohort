import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

# ── Save Filtered_ctDNA_matrix.csv from trusted pivot ──
print("Saving Filtered_ctDNA_matrix.csv from trusted pivot...")
pivot = pd.read_csv("trusted_pivot_vaf.csv", header=[0, 1], index_col=0)
pivot.to_csv("Filtered_ctDNA_matrix.csv")
print(f"  Shape: {pivot.shape[0]} variants x {pivot.shape[1]} (patient,TP) columns")
print(f"  Saved.\n")

# ── Cross-check ──
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
trusted = pd.read_csv("trusted_variants.csv")
inc = traj[traj["trajectory"] == "increasing"].copy()

print("="*60)
print("CROSS-CHECK: Clonal expansion vs Filtered_ctDNA_matrix")
print("="*60)

# 1. Matrix coverage
inc["in_matrix"] = inc["variant_id"].isin(pivot.index)
print(f"\n1. MATRIX COVERAGE: {inc['in_matrix'].mean()*100:.1f}% of expanding clones present")
if (~inc["in_matrix"]).sum() > 0:
    print(f"   Missing: {inc[~inc['in_matrix']][['patient_id','variant_id']].to_string(index=False)}")

# 2. VAF consistency — sample 20
print(f"\n2. VAF CONSISTENCY (sample of 20)")
mismatches = 0
sample = inc.sample(min(20, len(inc)), random_state=42)
for _, row in sample.iterrows():
    pid, vid = row["patient_id"], row["variant_id"]
    pt_cols = pivot.columns[pivot.columns.get_level_values(0) == pid]
    if vid not in pivot.index or len(pt_cols)==0:
        continue
    vals = pivot.loc[vid, pt_cols].values
    nz = vals[vals > 0]
    m_first = nz[0] if len(nz)>0 else 0
    m_last = nz[-1] if len(nz)>0 else 0
    ok = abs(row["first_VAF"]-m_first)<0.005 and abs(row["last_VAF"]-m_last)<0.005
    if not ok: mismatches += 1
    print(f"  {'OK' if ok else 'MISMATCH':10s} {pid:10s} traj=({row['first_VAF']:.3f}->{row['last_VAF']:.3f}) mat=({m_first:.3f}->{m_last:.3f})")
print(f"  Mismatches: {mismatches}/20")

# 3. Per-patient count (vectorized)
print(f"\n3. PER-PATIENT EXPANSION COUNTS")
inc_pp_traj = inc.groupby("patient_id").size()

# From matrix: vectorized per patient
inc_pp_mat = {}
patient_ids = traj["patient_id"].unique()
for pid in patient_ids:
    pt_cols = pivot.columns[pivot.columns.get_level_values(0) == pid]
    if len(pt_cols) < 2:
        inc_pp_mat[pid] = 0; continue
    sub = pivot[pt_cols]
    first_nz = sub.values.argmax(axis=1)  # not quite right — but nonzero[0]
    # Use apply for the vector
    def is_inc(row):
        nz = row.values[row.values > 0]
        return len(nz) >= 2 and nz[-1] > 1.5 * nz[0]
    inc_pp_mat[pid] = sub.apply(is_inc, axis=1).sum()

mat_series = pd.Series(inc_pp_mat)
cmp = pd.DataFrame({"traj": inc_pp_traj, "matrix": mat_series}).fillna(0).astype(int)
cmp["diff"] = cmp["traj"] - cmp["matrix"]
print(f"  Total expanding: traj={cmp['traj'].sum()}, matrix={cmp['matrix'].sum()}")
print(f"  Correlation: r={cmp['traj'].corr(cmp['matrix']):.4f}")
disp = cmp[cmp["diff"] != 0]
if len(disp) > 0:
    print(f"  Discrepant patients ({len(disp)}):")
    print(disp.to_string())

# ── Summary ──
print(f"\n{'='*60}")
print(f"CROSS-CHECK SUMMARY")
print(f"{'='*60}")
print(f"  File saved: Filtered_ctDNA_matrix.csv ({pivot.shape[0]} x {pivot.shape[1]})")
print(f"  Matrix coverage: {inc['in_matrix'].mean()*100:.1f}%")
print(f"  VAF consistency: {mismatches}/20 mismatched")
print(f"  Per-patient count correlation: r={cmp['traj'].corr(cmp['matrix']):.4f}")
print(f"  Total expanding: traj={cmp['traj'].sum()} vs matrix={cmp['matrix'].sum()}")
print("Done.")
