import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, warnings
warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
trusted = pd.read_csv("trusted_variants.csv")
burden = pd.read_csv("tumor_burden_cohort_composite.csv")

inc = traj[traj["trajectory"] == "increasing"].copy()
inc["gene"] = inc["variant_id"].str.split(" | ", n=1, regex=False).str[-1]
inc["fold"] = inc["last_VAF"] / inc["first_VAF"].clip(1e-6)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)

# ── A. Per-patient count of expanding clones ──
ax = fig.add_subplot(gs[0, 0])
inc_per_patient = inc.groupby("patient_id").size()
all_patients = traj["patient_id"].unique()
count_dist = pd.Series(0, index=all_patients)
count_dist[inc_per_patient.index] = inc_per_patient.values
count_dist = count_dist.sort_values(ascending=False)

colors = ["#e41a1c" if v > 0 else "#cccccc" for v in count_dist.values]
ax.bar(range(len(count_dist)), count_dist.values, color=colors, edgecolor="white", width=0.8)
ax.set_xticks(range(len(count_dist))); ax.set_xticklabels(count_dist.index, fontsize=5, rotation=90)
ax.set_ylabel("Expanding clones (n)");
ax.set_title("A) Per-patient expanding clone count\n(red=≥1, gray=0)")
ax.axhline(0, color="gray", lw=0.5)
n_with = (count_dist > 0).sum()
ax.text(0.95, 0.95, f"{n_with}/{len(count_dist)} patients\nhave expanding clones",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── B. Histogram of expanding clone counts ──
ax = fig.add_subplot(gs[0, 1])
counts_all = count_dist.values
ax.hist(counts_all, bins=range(0, counts_all.max()+2), color="#e41a1c", edgecolor="white", align="left")
ax.set_xlabel("Expanding clones per patient"); ax.set_ylabel("Patients");
ax.set_title("B) Distribution");
ax.set_xticks(range(0, int(counts_all.max())+1))
nz = counts_all[counts_all > 0]
ax.text(0.95, 0.95, f"Median={np.median(nz):.0f}\nMax={counts_all.max()}\nPatients with ≥1: {len(nz)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── C. Cumulative expansion magnitude per patient ──
ax = fig.add_subplot(gs[0, 2])
mag = inc.groupby("patient_id")["last_VAF"].sum()
mag_all = pd.Series(0.0, index=all_patients)
mag_all[mag.index] = mag.values
mag_all = mag_all.sort_values(ascending=False)

colors_c = ["#e41a1c" if v > 0 else "#cccccc" for v in mag_all.values]
ax.bar(range(len(mag_all)), mag_all.values, color=colors_c, edgecolor="white", width=0.8)
ax.set_xticks(range(len(mag_all))); ax.set_xticklabels(mag_all.index, fontsize=5, rotation=90)
ax.set_ylabel("Sum of last_VAF (expanding)");
ax.set_title("C) Cumulative expansion magnitude")

# ── D. Genes most frequently expanding ──
ax = fig.add_subplot(gs[0, 3])
gene_counts = inc["gene"].value_counts().head(15)
ax.barh(range(len(gene_counts)), gene_counts.values, color="#e41a1c", edgecolor="white")
ax.set_yticks(range(len(gene_counts))); ax.set_yticklabels(gene_counts.index, fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("Patients with this expanding clone");
ax.set_title("D) Genes with expanding clones")

# ── E. Fold-change distribution of expanding clones ──
ax = fig.add_subplot(gs[1, 0])
ax.hist(inc["fold"], bins=30, color="#e41a1c", edgecolor="white", alpha=0.8)
ax.set_xlabel("Fold change (last_VAF / first_VAF)"); ax.set_ylabel("Count");
ax.set_title("E) Fold-change of expanding clones")
ax.axvline(1.5, color="darkred", ls="--", lw=1)
ax.text(0.95, 0.95, f"n={len(inc)}\nMin={inc['fold'].min():.1f}x\nMax={inc['fold'].max():.1f}x\nMedian={inc['fold'].median():.1f}x",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── F. first_VAF vs last_VAF scatter ──
ax = fig.add_subplot(gs[1, 1])
ax.scatter(inc["first_VAF"], inc["last_VAF"], c="#e41a1c", alpha=0.6, s=30, edgecolors="white", linewidth=0.5)
lims = [0, inc[["first_VAF", "last_VAF"]].max().max() * 1.1]
ax.plot(lims, lims, "gray", ls="--", alpha=0.4, label="no change")
ax.plot(lims, [1.5*x for x in lims], "r:", alpha=0.4, label="1.5× threshold")
ax.set_xlabel("First nonzero VAF"); ax.set_ylabel("Last VAF");
ax.set_title("F) First vs last VAF (expanding)")
ax.legend(fontsize=7); ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.7)

# ── G. Timepoints of emergence ──
ax = fig.add_subplot(gs[1, 2])
inc["first_tp_num"] = inc["first_tp"].str.extract(r"(\d+)").astype(int)
tp_counts = inc["first_tp_num"].value_counts().sort_index()
ax.bar(tp_counts.index, tp_counts.values, color="#e41a1c", edgecolor="white", width=0.6)
ax.set_xlabel("First timepoint number"); ax.set_ylabel("Expanding clones");
ax.set_title("G) When do expanding clones appear?")
late = tp_counts[tp_counts.index >= 3].sum() if any(tp_counts.index >= 3) else 0
ax.text(0.95, 0.95, f"Late emergence (T3+): {late}/{len(inc)}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── H. Patients with most expanding clones: detail ──
ax = fig.add_subplot(gs[1, 3])
top5_patients = count_dist.sort_values(ascending=False).head(5).index
top5 = inc[inc["patient_id"].isin(top5_patients)]
# Group by patient and gene
gene_patient = top5.groupby(["patient_id", "gene"]).agg(
    n=("variant_id", "count"),
    total_fold=("fold", "sum"),
    last_VAF=("last_VAF", "mean")
).round(3).reset_index()

# Create a pivot-ish display: draw as table
display_data = []
for pid in top5_patients:
    sub = top5[top5["patient_id"] == pid].sort_values("fold", ascending=False).head(4)
    for _, r in sub.iterrows():
        display_data.append([pid, r["gene"], f"{r['first_VAF']:.3f}→{r['last_VAF']:.3f}",
                             f"{r['fold']:.1f}x", r["variant_id"][:30]])
table_data = pd.DataFrame(display_data, columns=["Patient", "Gene", "VAF trajectory", "Fold", "Locus"])
ax.axis("off")
table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                 cellLoc="left", loc="center", fontsize=7)
table.auto_set_column_width([0, 1, 2, 3, 4])
table.auto_set_font_size(False); table.set_fontsize(7)
for (j, i), cell in table.get_celld().items():
    if j == 0: cell.set_facecolor("#e41a1c"); cell.set_text_props(color="white", fontweight="bold")
ax.set_title("H) Top 5 patients: expanding clones detail", fontsize=9, y=0.98)

# ── I. Expanding clones: burden vs emergence ──
ax = fig.add_subplot(gs[2, 0:2])
merged = burden.merge(
    inc.groupby("patient_id").agg(n_expanding=("variant_id", "count"),
                                   mag_expanding=("last_VAF", "sum")).reset_index(),
    on="patient_id", how="left").fillna(0)
ax.scatter(merged["composite_score"], merged["n_expanding"], s=merged["mag_expanding"]*50+20,
           c=merged["mag_expanding"], cmap="YlOrRd", alpha=0.7, edgecolors="white", linewidth=0.5)
for _, r in merged[merged["n_expanding"] >= 3].iterrows():
    ax.annotate(r["patient_id"], (r["composite_score"], r["n_expanding"]), fontsize=7, alpha=0.8)
ax.set_xlabel("Composite tumor burden score"); ax.set_ylabel("Expanding clones (n)")
ax.set_title("I) Burden vs expanding clone count\n(size = expansion magnitude)")
cb = plt.colorbar(ax.collections[0], ax=ax, shrink=0.6)
cb.set_label("Expansion magnitude (sum last_VAF)", fontsize=8)

# ── J. Longitudinal trajectories of expanding clones ──
ax = fig.add_subplot(gs[2, 2:])
# Plot VAF timecourse for all expanding clones with 3+ TPs
inc_3tp = inc[inc["n_timepoints"] >= 3]
for _, row in inc_3tp.iterrows():
    pid, vid = row["patient_id"], row["variant_id"]
    pt = trusted[(trusted["patient_id"] == pid) & (trusted["variant_id"] == vid)]
    vafs = pt.groupby("tp_order")["VAF"].max().sort_index()
    gene = vid.split(" | ")[-1][:15]
    ax.plot(vafs.index, vafs.values, "o-", lw=1.2, alpha=0.6, ms=4, label=f"{gene} ({pid})" if pid != "LH210011" else "")
    if pid == "LH210011":
        ax.plot(vafs.index, vafs.values, "o-", lw=2.5, alpha=1.0, ms=6, color="red", label=f"** {gene} ({pid})")

# Highlight the FAS trajectory (biggest inc)
fas = inc_3tp[inc_3tp["variant_id"].str.contains("FAS", na=False)]
if len(fas) > 0:
    row = fas.iloc[0]
    pid, vid = row["patient_id"], row["variant_id"]
    pt = trusted[(trusted["patient_id"] == pid) & (trusted["variant_id"] == vid)]
    vafs = pt.groupby("tp_order")["VAF"].max().sort_index()
    ax.plot(vafs.index, vafs.values, "s--", lw=3, color="darkred", ms=8, label=f"FAS ({pid}) ★")

ax.set_xlabel("Timepoint order"); ax.set_ylabel("VAF");
ax.set_title(f"J) Longitudinal trajectories of expanding clones\n(n={len(inc_3tp)} with ≥3 TPs, LH210011 FAS highlighted in red)")
ax.legend(fontsize=7, loc="best", ncol=2)

plt.savefig("GSE244989_clonal_expansion.png", dpi=200, bbox_inches="tight")
print("Saved GSE244989_clonal_expansion.png")
print("Done.")
