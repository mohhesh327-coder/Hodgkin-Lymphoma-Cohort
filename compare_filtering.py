import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# ─── Load both pivots ───
orig_vaf = pd.read_csv("pivot_vaf.csv", header=[0, 1], index_col=0)
trusted_vaf = pd.read_csv("trusted_pivot_vaf.csv", header=[0, 1], index_col=0)

# Drop artifact rows from original
artifact_patterns = ["ClinVar |", "Locus |", "Unnamed:"]
orig_vaf = orig_vaf[~orig_vaf.index.to_series().apply(lambda x: any(p in str(x) for p in artifact_patterns))]

print("="*70)
print("PIVOT SHAPE COMPARISON")
print("="*70)
print(f"{'Metric':40s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
print("-"*70)
print(f"{'Variants (rows)':40s} {orig_vaf.shape[0]:>10,} {trusted_vaf.shape[0]:>10,} {trusted_vaf.shape[0] - orig_vaf.shape[0]:>+10,}")
print(f"{'(patient, tp) columns':40s} {orig_vaf.shape[1]:>10,} {trusted_vaf.shape[1]:>10,} {trusted_vaf.shape[1] - orig_vaf.shape[1]:>+10,}")
print(f"{'Non-zero entries':40s} {(orig_vaf > 0).sum().sum():>10,} {(trusted_vaf > 0).sum().sum():>10,} {(trusted_vaf > 0).sum().sum() - (orig_vaf > 0).sum().sum():>+10,}")
print(f"{'Sparsity (% zero)':40s} {(orig_vaf == 0).sum().sum() / orig_vaf.size * 100:>9.1f}% {(trusted_vaf == 0).sum().sum() / trusted_vaf.size * 100:>9.1f}% {(trusted_vaf == 0).sum().sum() / trusted_vaf.size * 100 - (orig_vaf == 0).sum().sum() / orig_vaf.size * 100:>+9.1f}%")
orig_vals = orig_vaf.values.flatten()
orig_vals = orig_vals[orig_vals > 0]
trusted_vals = trusted_vaf.values.flatten()
trusted_vals = trusted_vals[trusted_vals > 0]
orig_median = float(np.median(orig_vals))
trusted_median = float(np.median(trusted_vals))
orig_mean = float(np.mean(orig_vals))
trusted_mean = float(np.mean(trusted_vals))
print(f"{'Mean VAF (non-zero)':40s} {orig_mean:>10.4f} {trusted_mean:>10.4f} {trusted_mean - orig_mean:>+10.4f}")
print(f"{'Median VAF (non-zero)':40s} {orig_median:>10.4f} {trusted_median:>10.4f} {trusted_median - orig_median:>+10.4f}")

# ─── Which variants were removed? ───
removed_variants = set(orig_vaf.index) - set(trusted_vaf.index)
kept_variants = set(trusted_vaf.index)
extra_variants = set(trusted_vaf.index) - set(orig_vaf.index)
print(f"\nVariants removed by filter: {len(removed_variants):,}")
print(f"Variants unique to trusted: {len(extra_variants):,}")

# VAF distribution of removed variants
if removed_variants:
    removed_data = orig_vaf.loc[list(removed_variants)]
    removed_vals_flat = removed_data.values.flatten()
    removed_vals_flat = removed_vals_flat[removed_vals_flat > 0]
    if len(removed_vals_flat) > 0:
        print(f"\nVAF of removed variants (non-zero): mean={removed_vals_flat.mean():.4f}, median={float(np.median(removed_vals_flat)):.4f}")

# Per-patient impact
print("\n" + "="*70)
print("PER-PATIENT VARIANT COUNT: BEFORE vs AFTER")
print("="*70)

orig_counts = (orig_vaf > 0).sum(axis=0)
trusted_counts = (trusted_vaf > 0).sum(axis=0)

# Group by patient
orig_per_patient = orig_counts.groupby(level=0).sum()
trusted_per_patient = trusted_counts.groupby(level=0).sum()

compare = pd.DataFrame({"before": orig_per_patient, "after": trusted_per_patient}).fillna(0)
compare["removed"] = compare["before"] - compare["after"]
compare["pct_kept"] = (compare["after"] / compare["before"] * 100).round(1)
compare = compare.sort_values("before", ascending=False)
print(compare.to_string())

print(f"\nMean kept: {compare['pct_kept'].mean():.1f}%  (range: {compare['pct_kept'].min():.1f}% - {compare['pct_kept'].max():.1f}%)")

# ─── Reclassify trajectories on trusted matrix ───
print("\n" + "="*70)
print("TRAJECTORY CLASSIFICATION: BEFORE vs AFTER")
print("="*70)

def classify_trajectory(vals):
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

patient_ids = orig_vaf.columns.get_level_values(0).unique()

# Original trajectories
orig_records = []
for pid in patient_ids:
    sub = orig_vaf.loc[:, orig_vaf.columns.get_level_values(0) == pid]
    if sub.shape[1] < 2:
        continue
    sub.columns = sub.columns.droplevel(0)
    for var_id in sub.index:
        traj = classify_trajectory(sub.loc[var_id].values)
        if traj != "undetected":
            orig_records.append({"patient_id": pid, "variant_id": var_id, "trajectory": traj})

orig_traj = pd.DataFrame(orig_records)
orig_counts = orig_traj["trajectory"].value_counts()

# Trusted trajectories
trusted_records = []
for pid in patient_ids:
    sub = trusted_vaf.loc[:, trusted_vaf.columns.get_level_values(0) == pid]
    if sub.shape[1] < 2:
        continue
    sub.columns = sub.columns.droplevel(0)
    for var_id in sub.index:
        traj = classify_trajectory(sub.loc[var_id].values)
        if traj != "undetected":
            trusted_records.append({"patient_id": pid, "variant_id": var_id, "trajectory": traj})

trusted_traj = pd.DataFrame(trusted_records)
trusted_counts = trusted_traj["trajectory"].value_counts()

print(f"{'Trajectory':15s} {'Before':>10s} {'After':>10s} {'Change':>10s} {'% Before':>10s} {'% After':>10s}")
print("-"*65)
for cat in ["increasing", "decreasing", "stable"]:
    b = orig_counts.get(cat, 0)
    a = trusted_counts.get(cat, 0)
    bp = b / orig_counts.sum() * 100
    ap = a / trusted_counts.sum() * 100
    print(f"{cat:15s} {b:>10,} {a:>10,} {a-b:>+10,} {bp:>9.1f}% {ap:>9.1f}%")

# Per-patient trajectory comparison
print("\n" + "="*70)
print("PER-PATIENT TRAJECTORY SHIFT")
print("="*70)

orig_pp = orig_traj.groupby("patient_id")["trajectory"].value_counts().unstack(fill_value=0)
trusted_pp = trusted_traj.groupby("patient_id")["trajectory"].value_counts().unstack(fill_value=0)

for cat in ["increasing", "decreasing", "stable"]:
    if cat not in orig_pp.columns:
        orig_pp[cat] = 0
    if cat not in trusted_pp.columns:
        trusted_pp[cat] = 0

orig_pp["total"] = orig_pp.sum(axis=1)
trusted_pp["total"] = trusted_pp.sum(axis=1)

# Compare proportions
print(f"\n{'Patient':12s} {'Before_total':>13s} {'After_total':>12s} {'% kept':>7s} {'% dec before':>13s} {'% dec after':>12s} {'% inc before':>13s} {'% inc after':>12s}")
print("-"*90)
for pid in sorted(orig_pp.index):
    if pid not in trusted_pp.index:
        continue
    b_total = orig_pp.loc[pid, "total"]
    a_total = trusted_pp.loc[pid, "total"]
    pct_kept = a_total / b_total * 100 if b_total > 0 else 0
    b_dec = orig_pp.loc[pid, "decreasing"] / b_total * 100
    a_dec = trusted_pp.loc[pid, "decreasing"] / a_total * 100
    b_inc = orig_pp.loc[pid, "increasing"] / b_total * 100
    a_inc = trusted_pp.loc[pid, "increasing"] / a_total * 100
    print(f"{pid:12s} {b_total:>13,} {a_total:>12,} {pct_kept:>6.1f}% {b_dec:>12.1f}% {a_dec:>11.1f}% {b_inc:>12.1f}% {a_inc:>11.1f}%")

# ─── Plot ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Variant count per patient: before vs after
all_pids = sorted(set(list(orig_per_patient.index) + list(trusted_per_patient.index)))
x = range(len(all_pids))
b_vals = [orig_per_patient.get(pid, 0) for pid in all_pids]
a_vals = [trusted_per_patient.get(pid, 0) for pid in all_pids]
axes[0, 0].plot(x, b_vals, "o-", color="coral", label="Before", alpha=0.7, markersize=4)
axes[0, 0].plot(x, a_vals, "s-", color="seagreen", label="After", alpha=0.7, markersize=4)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(all_pids, rotation=90, fontsize=6)
axes[0, 0].set_ylabel("Non-zero variant counts")
axes[0, 0].set_title("Per-patient variant count: before vs after")
axes[0, 0].legend()

# 2. Histogram of kept percentage
axes[0, 1].hist(compare["pct_kept"], bins=20, color="steelblue", edgecolor="white")
axes[0, 1].set_xlabel("% of variants kept after filtering")
axes[0, 1].set_ylabel("Patients")
axes[0, 1].set_title("Distribution of retention rate per patient")

# 3. Trajectory composition before/after
cats = ["increasing", "decreasing", "stable"]
b_counts = [orig_counts.get(c, 0) for c in cats]
a_counts = [trusted_counts.get(c, 0) for c in cats]
x2 = np.arange(len(cats))
w = 0.35
bars1 = axes[0, 2].bar(x2 - w/2, b_counts, w, label="Before", color="coral", alpha=0.8)
bars2 = axes[0, 2].bar(x2 + w/2, a_counts, w, label="After", color="seagreen", alpha=0.8)
axes[0, 2].set_xticks(x2)
axes[0, 2].set_xticklabels(cats)
axes[0, 2].set_ylabel("Count")
axes[0, 2].set_title("Trajectory counts: before vs after")
axes[0, 2].legend()
# Add value labels
for bar in bars1:
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f"{bar.get_height():,}", ha="center", fontsize=8)
for bar in bars2:
    axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f"{bar.get_height():,}", ha="center", fontsize=8)

# 4. Trajectory proportions (stacked)
orig_pct = pd.DataFrame({c: orig_pp[c] / orig_pp["total"] * 100 for c in cats})
trusted_pct = pd.DataFrame({c: trusted_pp[c] / trusted_pp["total"] * 100 for c in cats})
orig_pct_sorted = orig_pct.sort_values("decreasing", ascending=True)
trusted_pct_sorted = trusted_pct.loc[orig_pct_sorted.index]

ax4 = axes[1, 0]
ax4.barh(range(len(orig_pct_sorted)), orig_pct_sorted["decreasing"], color="skyblue", label="Decreasing")
ax4.barh(range(len(orig_pct_sorted)), orig_pct_sorted["stable"], left=orig_pct_sorted["decreasing"], color="gray", label="Stable")
ax4.barh(range(len(orig_pct_sorted)), orig_pct_sorted["increasing"], 
         left=orig_pct_sorted["decreasing"] + orig_pct_sorted["stable"], color="red", label="Increasing")
ax4.set_yticks(range(len(orig_pct_sorted)))
ax4.set_yticklabels(orig_pct_sorted.index, fontsize=7)
ax4.set_xlabel("%")
ax4.set_title("Before: per-patient trajectory composition")
ax4.legend(loc="lower right", fontsize=7)

ax5 = axes[1, 1]
ax5.barh(range(len(trusted_pct_sorted)), trusted_pct_sorted["decreasing"], color="skyblue", label="Decreasing")
ax5.barh(range(len(trusted_pct_sorted)), trusted_pct_sorted["stable"], left=trusted_pct_sorted["decreasing"], color="gray", label="Stable")
ax5.barh(range(len(trusted_pct_sorted)), trusted_pct_sorted["increasing"],
         left=trusted_pct_sorted["decreasing"] + trusted_pct_sorted["stable"], color="red", label="Increasing")
ax5.set_yticks(range(len(trusted_pct_sorted)))
ax5.set_yticklabels(trusted_pct_sorted.index, fontsize=7)
ax5.set_xlabel("%")
ax5.set_title("After: per-patient trajectory composition")
ax5.legend(loc="lower right", fontsize=7)

# 6. VAF distribution (use already-flattened arrays)
orig_vaf_vals = orig_vals
trusted_vaf_vals = trusted_vals
axes[1, 2].hist(orig_vaf_vals, bins=80, alpha=0.5, color="coral", label=f"Before (n={len(orig_vaf_vals):,})", density=True, edgecolor="white")
axes[1, 2].hist(trusted_vaf_vals, bins=80, alpha=0.5, color="seagreen", label=f"After (n={len(trusted_vaf_vals):,})", density=True, edgecolor="white")
axes[1, 2].set_xlabel("VAF")
axes[1, 2].set_ylabel("Density")
axes[1, 2].set_title("VAF distribution (non-zero): before vs after")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig("filtering_before_after_comparison.png", dpi=150)
print("\nSaved filtering_before_after_comparison.png")

# ─── Save trajectory comparison ───
compare["patient_id"] = compare.index
compare.to_csv("filtering_comparison_per_patient.csv", index=False)
print("Saved filtering_comparison_per_patient.csv")
