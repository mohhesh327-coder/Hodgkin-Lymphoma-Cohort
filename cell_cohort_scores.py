# ── Cell: Run scores across cohort, cross-check, distribution plot ──

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

# ── Load inputs ──
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
precomputed = pd.read_csv("tumor_burden_scores.csv")   # cell 13 pre-computed scores
archetype_scores = pd.read_csv("tumor_burden_composite.csv")  # just ran

print("=" * 60)
print("1. RUN SCORES ACROSS THE FULL COHORT")
print("=" * 60)

def tumor_burden_score(traj_df, patient_id=None,
                       weight_inc=2.0, weight_dec=0.5, weight_stable=1.0, weight_undetected=0.0,
                       method="vaf_weighted"):
    df = traj_df.copy()
    if patient_id is not None:
        df = df[df["patient_id"] == patient_id]
    weights = {"increasing": weight_inc, "decreasing": weight_dec,
               "stable": weight_stable, "undetected": weight_undetected}
    results = []
    for pid, grp in df.groupby("patient_id"):
        for arch, w in weights.items():
            sub = grp[grp["trajectory"] == arch]
            n = len(sub)
            if n == 0:
                results.append({"patient_id": pid, "trajectory_archetype": arch,
                                "n_variants": 0, "contribution": 0.0})
                continue
            if method == "vaf_weighted":
                contrib = (sub["last_VAF"] * w).sum()
            elif method == "count":
                contrib = n * w
            elif method == "proportion":
                contrib = (sub["last_VAF"] * w).sum() / w
            else:
                raise ValueError(f"Unknown method: {method}")
            results.append({"patient_id": pid, "trajectory_archetype": arch,
                            "n_variants": n, "contribution": round(contrib, 4)})
    out = pd.DataFrame(results)
    comp = out.groupby("patient_id")["contribution"].sum().reset_index()
    comp.columns = ["patient_id", "composite_score"]
    out = out.merge(comp, on="patient_id")
    out = out.sort_values(["patient_id", "trajectory_archetype"]).reset_index(drop=True)
    return out, comp

# Run across all 46 patients
scores_df, composite = tumor_burden_score(traj)
n_patients = len(composite)
print(f"Scored {n_patients} patients")
print(f"\nComposite score summary:")
print(composite["composite_score"].describe().to_string())

# ── Per-archetype summary ──
arch_summary = scores_df.groupby("trajectory_archetype").agg(
    n_patients=("patient_id", "count"),
    total_contribution=("contribution", "sum"),
    mean_contribution=("contribution", "mean"),
    median_contribution=("contribution", "median"),
).round(4)
print(f"\nArchetype contributions across cohort:")
print(arch_summary.to_string())

# ── Cross-check ──
print("\n" + "=" * 60)
print("2. CROSS-CHECK AGAINST PRE-COMPUTED SCORES")
print("=" * 60)

# Method 1: composite from archetype model
arch_comp = composite.rename(columns={"composite_score": "archetype_composite"})

# Method 2: score_sum from tumor_burden_scores.csv (sum of VAFs at last TP)
sum_scores = precomputed[["patient_id", "score_sum"]].copy()

# Method 3: score_max from precomputed
max_scores = precomputed[["patient_id", "score_max"]].copy()

# Method 4: score_drivers from precomputed
driver_scores = precomputed[["patient_id", "score_drivers"]].copy()

# Method 5: earlier archetype composite from cell 13
old_arch = archetype_scores[["patient_id", "composite_score"]].rename(
    columns={"composite_score": "old_archetype_composite"})

# Merge all
check = arch_comp.merge(sum_scores, on="patient_id") \
                 .merge(max_scores, on="patient_id") \
                 .merge(driver_scores, on="patient_id") \
                 .merge(old_arch, on="patient_id")

# Compute correlations
corr_cols = ["archetype_composite", "score_sum", "score_max", "score_drivers", "old_archetype_composite"]
print(f"\nScore method correlations (n={len(check)} patients):")
print(check[corr_cols].corr().round(4).to_string())

print(f"\nFirst 5 patients:")
print(check.head().to_string(index=False))

# Identify discrepancies
check["sum_diff"] = (check["archetype_composite"] - check["score_sum"]).abs()
print(f"\nMax |archetype - score_sum| discrepancy: {check['sum_diff'].max():.4f}")
print(f"Mean |archetype - score_sum| discrepancy: {check['sum_diff'].mean():.4f}")
print(f"Patients with discrepancy > 0.5:")
disp = check[check["sum_diff"] > 0.5][["patient_id", "archetype_composite", "score_sum", "sum_diff"]]
print(disp.to_string(index=False) if len(disp) > 0 else "  (none)")

# ── Quick sanity: why would they differ? ──
# Archetype composite = sum(last_VAF * weight), score_sum = sum(last_VAF)
# They should match if all weights = 1 (stable). Let's verify with weight_stable=1.
print("\nSanity: running with all weights = 1 (should match score_sum)...")
_, comp_flat = tumor_burden_score(traj, weight_inc=1.0, weight_dec=1.0,
                                   weight_stable=1.0, weight_undetected=0.0)
comp_flat = comp_flat.rename(columns={"composite_score": "flat_composite"})
check2 = comp_flat.merge(sum_scores, on="patient_id")
check2["diff"] = (check2["flat_composite"] - check2["score_sum"]).abs()
print(f"Max diff with flat weights: {check2['diff'].max():.6f} — should be ~0")

# ── Distribution figure ──
print("\n" + "=" * 60)
print("3. DISTRIBUTION FIGURE")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 3a. Histogram of composite scores
axes[0, 0].hist(composite["composite_score"], bins=20, color="steelblue", edgecolor="white", density=True)
axes[0, 0].set_xlabel("Composite tumor burden score")
axes[0, 0].set_ylabel("Density")
axes[0, 0].set_title("A) Distribution of composite scores (all weights)")

# Overlay KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(composite["composite_score"])
x_kde = np.linspace(composite["composite_score"].min(), composite["composite_score"].max(), 200)
axes[0, 0].plot(x_kde, kde(x_kde), "r-", lw=2, alpha=0.7)
axes[0, 0].axvline(composite["composite_score"].median(), color="orange", ls="--", lw=2,
                   label=f"Median = {composite['composite_score'].median():.2f}")
axes[0, 0].axvline(composite["composite_score"].mean(), color="green", ls=":", lw=2,
                   label=f"Mean = {composite['composite_score'].mean():.2f}")
axes[0, 0].legend(fontsize=8)

# 3b. Histogram of log10 composite scores
axes[0, 1].hist(np.log10(composite["composite_score"].clip(lower=0.001)), bins=20,
                color="coral", edgecolor="white", density=True)
axes[0, 1].set_xlabel("log10(composite score)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_title("B) Log-transformed composite scores")

# 3c. Per-archetype contribution boxplot
arch_plot = scores_df[scores_df["contribution"] > 0].copy()
sns.boxplot(data=arch_plot, x="trajectory_archetype", y="contribution",
            order=["increasing", "decreasing", "stable"],
            palette="Set2", ax=axes[0, 2])
axes[0, 2].set_title("C) Per-archetype contribution distribution")
axes[0, 2].set_ylabel("Contribution to composite")
axes[0, 2].set_xlabel("")

# 3d. Stacked bar per patient (sorted by composite)
top_n = 20
comp_sorted = composite.sort_values("composite_score", ascending=False)
top_patients = comp_sorted.head(top_n)["patient_id"].tolist()

arch_pivot = scores_df.pivot_table(
    index="patient_id", columns="trajectory_archetype",
    values="contribution", aggfunc="sum"
).fillna(0)

arch_pivot_top = arch_pivot.loc[top_patients]
arch_pivot_top[["increasing", "decreasing", "stable"]].plot(
    kind="barh", stacked=True, ax=axes[1, 0],
    color={"increasing": "red", "decreasing": "skyblue", "stable": "gray"},
    width=0.8
)
axes[1, 0].set_title(f"D) Top {top_n} patients: archetype composition")
axes[1, 0].set_xlabel("Contribution to composite score")
axes[1, 0].invert_yaxis()

# 3e. Scatter: archetype composite vs score_sum
axes[1, 1].scatter(check["score_sum"], check["archetype_composite"],
                   c="steelblue", alpha=0.6, edgecolors="white", s=50)
# Identity line
lims = [0, max(check["score_sum"].max(), check["archetype_composite"].max()) * 1.05]
axes[1, 1].plot(lims, lims, "r--", alpha=0.5, label="Identity")
axes[1, 1].set_xlabel("score_sum (sum of last_VAF)")
axes[1, 1].set_ylabel("archetype_composite")
axes[1, 1].set_title("E) Archetype composite vs sum-of-VAFs")
axes[1, 1].legend()
r2 = check["archetype_composite"].corr(check["score_sum"]) ** 2
axes[1, 1].text(0.05, 0.95, f"R² = {r2:.4f}", transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment="top")

# 3f. Increasing contribution vs composite (highlight patients with high increasing)
axes[1, 2].scatter(composite["composite_score"], arch_pivot["increasing"],
                   c="red", alpha=0.6, edgecolors="white", s=50)
# Annotate outliers
high_inc = arch_pivot["increasing"] > 1.0
for pid in arch_pivot.index[high_inc]:
    axes[1, 2].annotate(pid,
                        (composite.loc[composite["patient_id"] == pid, "composite_score"].values[0],
                         arch_pivot.loc[pid, "increasing"]),
                        fontsize=6, alpha=0.7)
axes[1, 2].set_xlabel("Composite score")
axes[1, 2].set_ylabel("Increasing contribution")
axes[1, 2].set_title("F) Increasing clones vs total burden")
axes[1, 2].axhline(1.0, color="gray", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("tumor_burden_distribution.png", dpi=150)
print("Saved tumor_burden_distribution.png")

# ── Save cross-check ──
check.to_csv("tumor_burden_crosscheck.csv", index=False)
print("Saved tumor_burden_crosscheck.csv")

print("\nDone.")
