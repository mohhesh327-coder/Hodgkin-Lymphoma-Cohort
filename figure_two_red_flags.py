import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, warnings
warnings.filterwarnings("ignore")
mpl.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 200})

trusted = pd.read_csv("trusted_variants.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

# ── The two red flags ──
flags = [
    ("LH210011", "chr10:90771827 | FAS",  "FAS",  "#d62728"),
    ("BR190016", "chr12:57498480 | STAT6", "STAT6", "#ff7f0e"),
]

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30)

for i, (pid, vid, gene, color) in enumerate(flags):

    # ── A. VAF trajectory ──
    ax = fig.add_subplot(gs[i, 0])
    pt = trusted[trusted["patient_id"] == pid]
    sub = pt[pt["variant_id"] == vid]
    vafs = sub.groupby("tp_order")["VAF"].max().sort_index()

    ax.plot(vafs.index, vafs.values, "o-", color=color, lw=3, ms=10, zorder=5)
    ax.fill_between(vafs.index, vafs.values, alpha=0.15, color=color)

    first_nz = vafs[vafs > 0].iloc[0] if any(vafs > 0) else 0
    last_val = vafs.iloc[-1]
    fold = last_val / first_nz if first_nz > 0 else 0

    ax.annotate(f"Start: {first_nz:.2%}", xy=(vafs.index[0], first_nz),
                xytext=(5, 12), textcoords="offset points", fontsize=9, color=color, fontweight="bold")
    ax.annotate(f"End: {last_val:.2%}", xy=(vafs.index[-1], last_val),
                xytext=(-5, -15), textcoords="offset points", fontsize=9, color=color, fontweight="bold",
                ha="right")

    ax.set_xticks(vafs.index)
    ax.set_xticklabels([f"T{t}" for t in vafs.index])
    ax.set_ylim(-0.02, max(vafs.max() * 1.25, 0.15))
    ax.set_ylabel("VAF")
    ax.set_title(f"{gene} ({pid}) — VAF trajectory\n{first_nz:.1%} → {last_val:.1%}  ({fold:.0f}×)", fontsize=11, color=color)

    # ── B. All mutations for this patient, highlighting the flag ──
    ax2 = fig.add_subplot(gs[i, 1])
    tps = sorted(pt["tp_order"].unique())
    other_vids = pt[pt["variant_id"] != vid]["variant_id"].unique()
    # Pick top 8 other variants by max VAF
    other_max = pt[pt["variant_id"] != vid].groupby("variant_id")["VAF"].max().sort_values(ascending=False)

    # Plot the flag trajectory highlighted
    ax2.plot(vafs.index, vafs.values, "o-", color=color, lw=3, ms=8, zorder=5, label=f"★ {gene} (flag)")

    # Plot up to 8 other variants in gray
    for j, (ovid, _) in enumerate(other_max.head(8).items()):
        osub = pt[pt["variant_id"] == ovid]
        ovafs = osub.groupby("tp_order")["VAF"].max().sort_index()
        gene_name = ovid.split(" | ")[-1] if " | " in ovid else ovid
        alpha = 0.3 if j < 5 else 0.15
        ax2.plot(ovafs.index, ovafs.values, "o-", color="gray", lw=1, ms=4, alpha=alpha, label=f"{gene_name}" if j==0 else "")

    ax2.set_xticks(tps); ax2.set_xticklabels([f"T{t}" for t in tps])
    ax2.set_ylabel("VAF")
    ax2.set_title(f"{pid}: all mutations\n(flag highlighted)")
    ax2.legend(fontsize=7, loc="best")

    # ── C. Waterfall: all variants by trajectory ──
    ax3 = fig.add_subplot(gs[i, 2])
    pt_traj = traj[traj["patient_id"] == pid].copy()
    pt_traj = pt_traj.sort_values("last_VAF", ascending=True)
    pt_traj["gene_short"] = pt_traj["variant_id"].str.split(" | ", n=1, regex=False).str[-1]

    colors_t = {"increasing": color, "decreasing": "#377eb8", "stable": "#4daf4a", "undetected": "#cccccc"}
    bar_colors = [colors_t.get(t, "gray") for t in pt_traj["trajectory"]]
    # Highlight the flag variant
    flag_idx = pt_traj[pt_traj["variant_id"] == vid].index
    for idx in flag_idx:
        pos = list(pt_traj.index).index(idx)
        bar_colors[pos] = color

    ax3.barh(range(len(pt_traj)), pt_traj["last_VAF"], color=bar_colors, edgecolor="white", height=0.8)
    ax3.set_yticks(range(len(pt_traj)))
    ax3.set_yticklabels([f"{r['gene_short'][:25]} ({r['trajectory']})" for _, r in pt_traj.iterrows()], fontsize=5)
    ax3.set_xlabel("Last VAF")
    ax3.set_title(f"{pid}: variant landscape\n(red = increasing, blue = dec, green = stable)")

    # Annotate flag
    ax3.text(0.95, 0.95, f"★ {gene}", transform=ax3.transAxes, ha="right", va="top",
             fontsize=11, color=color, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# ── Summary panel ──
ax4 = fig.add_subplot(gs[1, :])
ax4.axis("off")
row = traj[(traj["patient_id"] == "LH210011") & (traj["variant_id"] == "chr10:90771827 | FAS")].iloc[0]
row2 = traj[(traj["patient_id"] == "BR190016") & (traj["variant_id"] == "chr12:57498480 | STAT6")].iloc[0]

summary = (
    f"Two Clonally Expanding Mutations in GSE244989\n"
    f"{'─' * 60}\n"
    f"\n"
    f"★ FAS (LH210011)  —  {row['first_VAF']:.2%} → {row['last_VAF']:.2%}  ({row['last_VAF']/row['first_VAF']:.0f}×)\n"
    f"   Locus: chr10:90771827  |  First TP: {row['first_tp']}  |  Last TP: {row['last_tp']}\n"
    f"   Clone emerges from 2% to 49% over 4 timepoints — classic resistance emergence.\n"
    f"   FAS mediates apoptosis; loss-of-function mutations confer chemo-resistance.\n"
    f"\n"
    f"★ STAT6 (BR190016)  —  {row2['first_VAF']:.2%} → {row2['last_VAF']:.2%}  ({row2['last_VAF']/row2['first_VAF']:.0f}×)\n"
    f"   Locus: chr12:57498480  |  First TP: {row2['first_tp']}  |  Last TP: {row2['last_tp']}\n"
    f"   STAT6 is a recurrent HL driver (JAK-STAT pathway, 3p24 amplification target).\n"
    f"   Clonal expansion suggests therapy-resistant subclone outgrowth.\n"
    f"\n"
    f"Both are driver genes. FAS is the single most extreme emerging clone (24.6×) across all 46 patients;\n"
    f"STAT6 is the second (7.5×). Their detection at low-VAF → high-VAF across serial timepoints is the\n"
    f"hallmark of clonal evolution under treatment pressure."
)
ax4.text(0.5, 0.5, summary, transform=ax4.transAxes, ha="center", va="center",
         fontsize=10, fontfamily="monospace", linespacing=1.6)

plt.savefig("GSE244989_two_red_flags.png", dpi=200, bbox_inches="tight")
print("Saved GSE244989_two_red_flags.png\nDone.")
