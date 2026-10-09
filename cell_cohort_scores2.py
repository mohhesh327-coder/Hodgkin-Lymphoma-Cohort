import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings("ignore"); sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

def tumor_burden_score(traj_df, w_inc=2.0, w_dec=0.5, w_stable=1.0, w_und=0.0, method="vaf_weighted"):
    weights = {"increasing": w_inc, "decreasing": w_dec, "stable": w_stable, "undetected": w_und}
    out = []
    for pid, grp in traj_df.groupby("patient_id"):
        tot = 0
        for arch, w in weights.items():
            sub = grp[grp["trajectory"] == arch]
            n = len(sub)
            if method == "vaf_weighted":
                c = (sub["last_VAF"] * w).sum()
            elif method == "count":
                c = n * w
            elif method == "proportion":
                c = (sub["last_VAF"] * w).sum() / max(w, 0.001)
            else:
                raise ValueError(f"Unknown method: {method}")
            tot += c
            out.append({"patient_id": pid, "trajectory_archetype": arch,
                        "n_variants": n, "contribution": round(c, 4)})
        for row in out[-4:]:
            row["composite_score"] = round(tot, 4)
    return pd.DataFrame(out)

scores_df = tumor_burden_score(traj)
composite = scores_df.groupby("patient_id")["composite_score"].first().reset_index()

print("=" * 60)
print("COHORT TUMOR BURDEN SCORES (N=%d patients)" % len(composite))
print("=" * 60)
print(composite["composite_score"].describe().round(3).to_string())

# ── Cross-check vs pre-computed ──
pre = pd.read_csv("tumor_burden_scores.csv")
# Take max TP per patient (last timepoint for each)
pre_last = pre.loc[pre.groupby("patient_id")["tp_order"].idxmax()][["patient_id","score_sum","score_max","score_drivers"]]

check = composite.merge(pre_last, on="patient_id", how="left")
print("\nCross-check vs pre-computed (last TP only):")
for c in ["score_sum","score_max","score_drivers"]:
    r = check["composite_score"].corr(check[c])
    print(f"  r(archetype, {c}) = {r:.4f}")

# With flat weights should ~match score_sum at last TP
scores_flat = tumor_burden_score(traj, w_inc=1, w_dec=1, w_stable=1, w_und=0)
comp_flat = scores_flat.groupby("patient_id")["composite_score"].first()
check2 = comp_flat.to_frame("flat_composite").merge(pre_last[["patient_id","score_sum"]], on="patient_id")
check2["diff"] = (check2["flat_composite"] - check2["score_sum"]).abs()
print(f"\nFlat-weight vs score_sum: max diff = {check2['diff'].max():.4f} (should be 0)")
print("(If non-zero: score_sum sums VAF at LAST TP; archetype uses last_VAF from trajectory)")

# ── Figure ──
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

axes[0,0].hist(composite["composite_score"], bins=20, color="steelblue", edgecolor="white", density=True)
axes[0,0].set_xlabel("Composite tumor burden score"); axes[0,0].set_ylabel("Density")
axes[0,0].set_title("A) Composite score distribution")
axes[0,0].axvline(composite["composite_score"].median(), color="orange", ls="--",
                  label=f"Median={composite['composite_score'].median():.2f}")
axes[0,0].axvline(composite["composite_score"].mean(), color="green", ls=":",
                  label=f"Mean={composite['composite_score'].mean():.2f}")
axes[0,0].legend(fontsize=8)

axes[0,1].hist(np.log10(composite["composite_score"].clip(0.001)), bins=20, color="coral", edgecolor="white")
axes[0,1].set_xlabel("log10(score)"); axes[0,1].set_ylabel("Density")
axes[0,1].set_title("B) Log-transformed scores")

arch_p = scores_df[scores_df["contribution"]>0]
sns.boxplot(data=arch_p, x="trajectory_archetype", y="contribution",
            order=["increasing","decreasing","stable"], palette="Set2", ax=axes[0,2])
axes[0,2].set_title("C) Archetype contributions")

comp_sorted = composite.sort_values("composite_score", ascending=False)
top20 = comp_sorted.head(20)["patient_id"].tolist()
arch_piv = scores_df.pivot_table(index="patient_id", columns="trajectory_archetype",
                                  values="contribution", aggfunc="sum").fillna(0)
arch_piv.loc[top20][["increasing","decreasing","stable"]].plot(
    kind="barh", stacked=True, ax=axes[1,0],
    color={"increasing":"red","decreasing":"skyblue","stable":"gray"}, width=0.8)
axes[1,0].set_title("D) Top 20: archetype composition"); axes[1,0].set_xlabel("Contribution"); axes[1,0].invert_yaxis()

axes[1,1].scatter(check["score_sum"], check["composite_score"], c="steelblue", alpha=0.6, edgecolors="white", s=50)
lims = [0, max(check["score_sum"].max(), check["composite_score"].max())*1.05]
axes[1,1].plot(lims, lims, "r--", alpha=0.5, label="Identity")
axes[1,1].set_xlabel("score_sum"); axes[1,1].set_ylabel("archetype_composite")
axes[1,1].set_title(f"E) Archetype vs sum-of-VAFs (r={check['composite_score'].corr(check['score_sum']):.3f})")

axes[1,2].scatter(composite["composite_score"], arch_piv["increasing"], c="red", alpha=0.6, edgecolors="white", s=50)
for pid in arch_piv.index[arch_piv["increasing"] > 1.0]:
    axes[1,2].annotate(pid, (composite.set_index("patient_id").loc[pid,"composite_score"],
                               arch_piv.loc[pid,"increasing"]), fontsize=6, alpha=0.7)
axes[1,2].set_xlabel("Composite score"); axes[1,2].set_ylabel("Increasing contribution")
axes[1,2].set_title("F) Increasing clones vs total burden"); axes[1,2].axhline(1, color="gray", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("tumor_burden_distribution.png", dpi=150)
print("\nSaved tumor_burden_distribution.png")

scores_df.to_csv("tumor_burden_cohort_scores.csv", index=False)
composite.to_csv("tumor_burden_cohort_composite.csv", index=False)
print("Saved tumor_burden_cohort_scores.csv, tumor_burden_cohort_composite.csv")
print("Done.")
