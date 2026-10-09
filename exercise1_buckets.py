import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings("ignore"); sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
trusted = pd.read_csv("trusted_variants.csv")
burden = pd.read_csv("tumor_burden_cohort_composite.csv")

# ── Bucket by composite score ──
burden["bucket"] = pd.qcut(burden["composite_score"], q=3, labels=["low", "medium", "high"])
print(burden.groupby("bucket")["composite_score"].describe().round(2).to_string())

# Pick one patient per bucket (closest to median of that bucket)
picks = {}
for b in ["low", "medium", "high"]:
    sub = burden[burden["bucket"] == b].copy()
    sub["dist"] = (sub["composite_score"] - sub["composite_score"].median()).abs()
    picks[b] = sub.sort_values("dist").iloc[0]["patient_id"]
    print(f"\n{b.upper()} bucket pick: {picks[b]} (score={sub.sort_values('dist').iloc[0]['composite_score']:.2f})")

# ── Inspect each ──
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
for i, (bucket, pid) in enumerate(picks.items()):
    print("\n" + "=" * 70)
    print(f"BUCKET: {bucket.upper()}  |  Patient: {pid}")
    print("=" * 70)

    pt_traj = traj[traj["patient_id"] == pid].sort_values("last_VAF", ascending=False)
    print(f"  Total variant-patient trajectories: {len(pt_traj)}")

    # Archetype breakdown
    arch = pt_traj.groupby("trajectory").agg(
        n=("variant_id", "count"),
        total_last_VAF=("last_VAF", "sum"),
        mean_last_VAF=("last_VAF", "mean")
    ).round(4)
    print(f"\n  Archetype breakdown:")
    print(arch.to_string())

    # Timepoints
    pt_trusted = trusted[trusted["patient_id"] == pid]
    tps = sorted(pt_trusted["tp_label"].unique())
    print(f"\n  Timepoints: {tps}")

    # Top 5 variants by last_VAF
    top5 = pt_traj.head(5)
    print(f"\n  Top 5 variants (by last_VAF):")
    for _, r in top5.iterrows():
        print(f"    {r['variant_id']:50s} traj={r['trajectory']:12s} first_VAF={r['first_VAF']:.4f} last_VAF={r['last_VAF']:.4f}")

    # VAF trajectory plot for top 5 variants
    ax = axes[i, 0]
    for _, r in top5.iterrows():
        var_data = pt_trusted[pt_trusted["variant_id"] == r["variant_id"]]
        var_tp = var_data.groupby("tp_order")["VAF"].max().sort_index()
        if len(var_tp) > 0:
            label = r["variant_id"].split(" | ")[-1][:25]  # gene name truncated
            ax.plot(var_tp.index, var_tp.values, "o-", ms=5, label=label)
    ax.set_title(f"{pid} [{bucket.upper()}] — Top 5 VAF trajectories")
    ax.set_xlabel("TP order"); ax.set_ylabel("VAF")
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(-0.02, 1.05)

    # Archetype composition pie
    ax2 = axes[i, 1]
    arch_counts = pt_traj["trajectory"].value_counts()
    colors_arch = {"increasing": "red", "decreasing": "skyblue", "stable": "gray", "undetected": "white"}
    ax2.pie(arch_counts.values, labels=[f"{k}\n({v})" for k,v in arch_counts.items()],
            colors=[colors_arch.get(k, "lightgray") for k in arch_counts.index],
            autopct="%1.0f%%", startangle=90)
    ax2.set_title(f"{pid} — Archetype composition")

    # VAF distribution by archetype
    ax3 = axes[i, 2]
    for arch_type in ["increasing", "decreasing", "stable"]:
        subset = pt_traj[pt_traj["trajectory"] == arch_type]
        if len(subset) > 0:
            ax3.hist(subset["last_VAF"], bins=20, alpha=0.5, label=f"{arch_type} (n={len(subset)})", density=True)
    ax3.set_xlabel("last_VAF"); ax3.set_ylabel("Density")
    ax3.set_title(f"{pid} — VAF distribution by archetype"); ax3.legend(fontsize=8)

    # Also print key driver variants of interest
    drivers = pt_traj[pt_traj["variant_id"].str.contains("STAT|NFKB|REL|MYC|BCL|TP53|CREBBP|EP300|NOTCH", case=False)]
    if len(drivers) > 0:
        print(f"\n  Key driver variants:")
        for _, r in drivers.sort_values("last_VAF", ascending=False).head(8).iterrows():
            print(f"    {r['variant_id']:55s} traj={r['trajectory']:12s} last_VAF={r['last_VAF']:.4f}")

plt.tight_layout()
plt.savefig("bucket_inspection.png", dpi=150)
print("\nSaved bucket_inspection.png")
print("\nDone.")
