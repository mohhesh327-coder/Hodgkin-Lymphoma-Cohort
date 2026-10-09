import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PATIENT = "LH210011"
COLORS = {"increasing": "#d73027", "decreasing": "#4575b4", "stable": "#fee090", "undetected": "#d3d3d3"}
N_COLS = 3

tv = pd.read_csv("trusted_variants.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

# Filter to patient
p_tv = tv[tv["patient_id"] == PATIENT].copy()
p_traj = traj[traj["patient_id"] == PATIENT].copy()
tp_labels = sorted(p_tv["tp_label"].unique())
tp_order = [p_tv[p_tv["tp_label"] == t]["tp_order"].iloc[0] for t in tp_labels]

# Build per-variant VAF trajectory
variant_rows = []
for vid, grp in p_tv.groupby("variant_id"):
    grp = grp.sort_values("tp_order")
    vafs = [grp[grp["tp_label"] == t]["VAF"].values[0] if t in grp["tp_label"].values else 0.0 for t in tp_labels]
    gene = grp["Genes"].iloc[0]
    loc = grp["Locus"].iloc[0]
    traj_row = p_traj[p_traj["variant_id"] == vid]
    trend = traj_row["trajectory"].values[0] if len(traj_row) > 0 else "unknown"
    variant_rows.append({"variant_id": vid, "gene": gene, "locus": loc, "trend": trend, "vafs": vafs})

variants = pd.DataFrame(variant_rows)
variants = variants.sort_values("trend", key=lambda x: x.map({"increasing": 0, "stable": 1, "decreasing": 2, "undetected": 3}))

# Set up subplots: first is the main overview, then individual panels for increasing variants
inc_variants = variants[variants["trend"] == "increasing"]
n_inc = len(inc_variants)
n_total = len(variants)

rows = 1 + (n_inc + N_COLS - 1) // N_COLS
fig = plt.figure(figsize=(18, 4 * rows))
gs = fig.add_gridspec(rows, N_COLS, hspace=0.35, wspace=0.3)

colors_trend = [COLORS.get(t, "#999999") for t in variants["trend"]]
n_tp = len(tp_labels)

# --- Main panel: all trajectories ---
ax0 = fig.add_subplot(gs[0, :])
for idx, row in variants.iterrows():
    vafs = row["vafs"]
    trend = row["trend"]
    c = COLORS.get(trend, "#999999")
    alpha = 0.9 if trend == "increasing" else 0.25
    lw = 3.0 if trend == "increasing" else 0.8
    ax0.plot(range(n_tp), vafs, color=c, alpha=alpha, linewidth=lw)
    if trend == "increasing":
        last_vaf = vafs[-1]
        ax0.annotate(row["gene"], xy=(n_tp - 1, last_vaf), xytext=(n_tp - 1 + 0.15, last_vaf),
                     fontsize=9, fontweight="bold", color=c, va="center",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.5))

ax0.set_xticks(range(n_tp))
ax0.set_xticklabels(tp_labels)
ax0.set_xlabel("Timepoint")
ax0.set_ylabel("VAF")
ax0.set_title(f"Patient {PATIENT} — All {n_total} Mutation Trajectories", fontsize=14, fontweight="bold")
ax0.set_ylim(-0.03, 1.03)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=COLORS["increasing"], lw=3, label=f"Increasing ({n_inc})"),
    Line2D([0], [0], color=COLORS["decreasing"], lw=2, alpha=0.5, label=f"Decreasing ({len(variants[variants['trend']=='decreasing'])})"),
    Line2D([0], [0], color=COLORS["stable"], lw=2, alpha=0.5, label=f"Stable ({len(variants[variants['trend']=='stable'])})"),
]
ax0.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

ax0.text(0.02, 0.96,
         "Thick decreasing bundle: tumor-derived ctDNA\nclearing with treatment\n\nUpward outlier(s): expanding subclones\nwith treatment resistance potential",
         transform=ax0.transAxes, fontsize=10, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.85), fontfamily="monospace")

# --- Individual panels for each increasing variant ---
for i, (_, row) in enumerate(inc_variants.iterrows()):
    r = 1 + i // N_COLS
    c = i % N_COLS
    ax = fig.add_subplot(gs[r, c])

    vafs = row["vafs"]
    ax.plot(range(n_tp), vafs, color=COLORS["increasing"], linewidth=3.5, marker="o", markersize=8, zorder=5)

    # Fill under curve
    ax.fill_between(range(n_tp), vafs, alpha=0.15, color=COLORS["increasing"])

    # Annotate VAF values
    for j, v in enumerate(vafs):
        ax.annotate(f"{v:.1%}", xy=(j, v), xytext=(j, v + 0.04),
                    fontsize=8, ha="center", va="bottom", fontweight="bold",
                    color=COLORS["increasing"])

    ax.set_xticks(range(n_tp))
    ax.set_xticklabels(tp_labels)
    ax.set_ylim(-0.02, min(max(vafs) * 1.4, 1.05))
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("VAF")
    first_vaf = next((v for v in vafs if v > 0), 0)
    last_vaf = vafs[-1]
    fold = last_vaf / first_vaf if first_vaf > 0 else float("inf")
    ax.set_title(f"{row['gene']}  ({row['variant_id'].split('|')[0].strip()})\n{first_vaf:.1%} -> {last_vaf:.1%}  ({fold:.1f}x)", fontsize=10)

# If subplots are unused, hide them
total_subplots = 1 + n_inc
for i in range(total_subplots, rows * N_COLS):
    r = 1 + i // N_COLS
    c = i % N_COLS
    if r < rows:
        fig.add_subplot(gs[r, c]).set_visible(False)

fig.savefig("exercise4_subclone_spotlight.png", dpi=200, bbox_inches="tight")
print(f"Saved exercise4_subclone_spotlight.png  |  Patient {PATIENT}: {n_total} variants ({n_inc} increasing)")

# Also print a quick summary
print(f"\n{'Variant':<50} {'Trend':<12} {'First VAF':<10} {'Last VAF':<10} {'Fold':<8}")
print("-" * 90)
for _, row in variants.iterrows():
    vafs = row["vafs"]
    first_vaf = next((v for v in vafs if v > 0), 0)
    last_vaf = vafs[-1]
    fold = last_vaf / first_vaf if first_vaf > 0 else 0
    label = f"{row['gene']} ({row['variant_id'].split('|')[0].strip()})"
    print(f"{label:<50} {row['trend']:<12} {first_vaf:<10.1%} {last_vaf:<10.1%} {fold:<8.1f}x")
