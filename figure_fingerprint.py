import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, warnings
warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
trusted = pd.read_csv("trusted_variants.csv")

col_palette = {"increasing": "#e41a1c", "decreasing": "#377eb8", "stable": "#4daf4a", "undetected": "#cccccc"}

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.30)

# ── A. Global pie ──
ax = fig.add_subplot(gs[0, 0])
counts = traj["trajectory"].value_counts()
wedges, texts, autotexts = ax.pie(
    counts.values, labels=counts.index, autopct="%1.1f%%",
    colors=[col_palette[k] for k in counts.index],
    startangle=90, explode=(0, 0.05, 0.05, 0.05)[:len(counts)],
    textprops={"fontsize": 9})
for at in autotexts: at.set_fontsize(10); at.set_fontweight("bold")
ax.set_title("A) Global trajectory distribution\n(N = {:,} variant–patient pairs)".format(len(traj)), fontsize=10)

# ── B. Per-patient stacked bars (sorted by %decreasing) ──
ax = fig.add_subplot(gs[0, 1:])
pp = traj.groupby("patient_id")["trajectory"].value_counts().unstack(fill_value=0)
pp_pct = pp.div(pp.sum(axis=1), axis=0) * 100
pp_pct = pp_pct.sort_values("decreasing", ascending=False)

colors = [col_palette[c] for c in pp_pct.columns]
pp_pct.plot(kind="barh", stacked=True, ax=ax, color=colors, width=0.85, legend=False)
ax.set_xlabel("% of variants"); ax.set_ylabel("Patient")
ax.set_title("B) Per-patient trajectory fingerprint (sorted by % decreasing)")
ax.axvline(50, color="gray", ls="--", lw=0.5, alpha=0.4)
ax.set_xlim(0, 100)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=col_palette[k], label=k) for k in pp_pct.columns]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, title="Archetype")

# ── C. Summary bar (mean ± SD across patients) ──
ax = fig.add_subplot(gs[1, 0])
means = pp_pct.mean()
sds = pp_pct.std()
x_pos = range(len(means))
bars = ax.bar(x_pos, means.values, yerr=sds.values, capsize=4,
              color=[col_palette[c] for c in means.index], edgecolor="white", width=0.6)
ax.set_xticks(x_pos); ax.set_xticklabels(means.index)
ax.set_ylabel("Mean % per patient"); ax.set_title("C) Cohort average composition")
for bar, val in zip(bars, means.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")

# ── D. Group-stratified (BR / LH / C / X) ──
ax = fig.add_subplot(gs[1, 1])
trusted_small = trusted[["patient_id", "group"]].drop_duplicates()
pp_g = pp.copy()
pp_g.index.name = "patient_id"
pp_g = pp_g.reset_index().merge(trusted_small, on="patient_id", how="left")
pp_g["group"] = pp_g["group"].fillna("X")

group_order = pp_g.groupby("group")["patient_id"].count().sort_values(ascending=False).index
x_pos = np.arange(len(group_order))
width = 0.25
for i, arch in enumerate(["increasing", "decreasing", "stable"]):
    means = pp_g.groupby("group")[arch].mean()
    ax.bar(x_pos + (i-1)*width, means.values, width, color=col_palette[arch], label=arch, alpha=0.85, edgecolor="white")
ax.set_xticks(x_pos); ax.set_xticklabels(group_order)
ax.set_ylabel("Mean variant count"); ax.set_title("D) By clinical group")
ax.legend(fontsize=7)

# ── E. Trajectory count vs composite score ──
ax = fig.add_subplot(gs[1, 2])
burden = pd.read_csv("tumor_burden_cohort_composite.csv")
pp_counts = pp.copy()
pp_counts.index.name = "patient_id"
merged = pp_counts.reset_index().merge(burden, on="patient_id")
for arch in ["increasing", "decreasing", "stable"]:
    ax.scatter(merged["composite_score"], merged[arch], 
               c=col_palette[arch], alpha=0.5, s=20, label=arch, edgecolors="white", linewidth=0.3)
ax.set_xlabel("Composite tumor burden score"); ax.set_ylabel("Variant count")
ax.set_title("E) Burden vs trajectory count")
ax.legend(fontsize=7)
ax.set_xscale("symlog")

# ── F. Fold-change distribution ──
ax = fig.add_subplot(gs[1, 3])
traj["fold"] = traj["last_VAF"] / traj["first_VAF"].clip(1e-6)
for arch in ["increasing", "decreasing", "stable"]:
    subset = traj[traj["trajectory"] == arch]["fold"]
    ax.hist(np.log2(subset.clip(1e-6)), bins=60, alpha=0.4, color=col_palette[arch],
            label=f"{arch} (n={len(subset)})", density=True)
ax.axvline(np.log2(1.5), color="red", ls="--", lw=0.8)
ax.axvline(np.log2(0.5), color="blue", ls="--", lw=0.8)
ax.set_xlabel("log2(last_VAF / first_VAF)"); ax.set_ylabel("Density")
ax.set_title("F) Fold-change distribution"); ax.legend(fontsize=7)

# ── G. Number of trajectory classes per patient ──
ax = fig.add_subplot(gs[2, 0])
n_classes = (pp > 0).sum(axis=1)
n_classes.value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white", width=0.7)
ax.set_xlabel("Number of trajectory classes present"); ax.set_ylabel("Patients")
ax.set_title("G) Diversity of archetypes per patient")

# ── H. Trajectory Sankey-style: T0 presence → trajectory ──
ax = fig.add_subplot(gs[2, 1:3])
# Presence at first TP vs trajectory outcome
pres_at_t0 = traj["first_VAF"] > 0.01  # detected at first TP
traj["first_detected"] = pres_at_t0
ct = traj.groupby(["first_detected", "trajectory"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

x = [0, 1]
for i, arch in enumerate(["increasing", "decreasing", "stable"]):
    vals = ct_pct[arch].values if arch in ct_pct else [0, 0]
    ax.plot(x, vals, "o-", color=col_palette[arch], lw=2.5, label=arch, ms=8)
    for xi, yi, vi in zip(x, vals, ct[arch].values if arch in ct else [0, 0]):
        ax.text(xi, yi+1, f"{yi:.1f}%\n(n={vi})", ha="center", fontsize=7, color=col_palette[arch])
ax.set_xticks(x); ax.set_xticklabels(["First TP VAF ≤ 0.01", "First TP VAF > 0.01"])
ax.set_ylabel("% of trajectories")
ax.set_title("H) First-TP detection → trajectory outcome")
ax.legend(fontsize=7); ax.set_ylim(0, 105)

# ── I. 3D-like heatmap: patient x trajectory type ──
ax = fig.add_subplot(gs[2, 3])
pp_log = np.log10(pp + 1)
im = ax.imshow(pp_log.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_yticks(range(len(pp_log))); ax.set_yticklabels(pp_log.index, fontsize=5)
ax.set_xticks(range(len(pp_log.columns))); ax.set_xticklabels(pp_log.columns)
ax.set_title("I) log10(variant count + 1)")
plt.colorbar(im, ax=ax, shrink=0.6)

# ── J. Summary text panel ──
ax = fig.add_subplot(gs[3, :])
ax.axis("off")
n_patients = traj["patient_id"].nunique()
n_variants = traj["variant_id"].nunique()
summary = (
    f"GSE244989 Treated-Cohort Fingerprint\n"
    f"{'─' * 50}\n"
    f"Patients: {n_patients}  |  Variant–patient pairs: {len(traj):,}  |  Unique variants: {n_variants:,}\n"
    f"\n"
    f"Decreasing:  {counts.get('decreasing',0):,} ({counts.get('decreasing',0)/len(traj)*100:.1f}%)\n"
    f"  →  66.6% cleared to exactly 0 — treatment response signature\n"
    f"\n"
    f"Stable:      {counts.get('stable',0):,} ({counts.get('stable',0)/len(traj)*100:.1f}%)\n"
    f"  →  Persistent clones — resistance/minimal residual disease\n"
    f"\n"
    f"Increasing:  {counts.get('increasing',0):,} ({counts.get('increasing',0)/len(traj)*100:.1f}%)\n"
    f"  →  Emerging clones — potential progression markers\n"
    f"\n"
    f"Signature: A classical treated-Hodgkin lymphoma ctDNA landscape —\n"
    f"           most mutations clear with therapy (↓ ~67%),\n"
    f"           a minority persist (∼ 32%), and rare clones emerge (∼ 1%)."
)
ax.text(0.5, 0.5, summary, transform=ax.transAxes, ha="center", va="center",
        fontsize=11, fontfamily="monospace", linespacing=1.5)

plt.savefig("GSE244989_trend_fingerprint.png", dpi=200, bbox_inches="tight")
print("Saved GSE244989_trend_fingerprint.png")
print("Done.")
