import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import ceil

# Load data
tv = pd.read_csv("trusted_variants.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
cs = pd.read_csv("composite_scores.csv").sort_values("total", ascending=False)
patients = cs["patient"].tolist()
N = len(patients)
print(f"Total patients: {N}")

# Build lookup: (patient_id, variant_id) -> trajectory
traj_lookup = traj.set_index(["patient_id", "variant_id"])["trajectory"].to_dict()

# Determine grid
ncols = 6
nrows = ceil(N / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), sharex=False, sharey=False)
fig.suptitle("VAF Trajectories by Patient (sorted by recurrence score)", fontsize=16, fontweight="bold", y=0.98)

# Colors for trajectory types
tcolor = {"increasing": "#e74c3c", "decreasing": "#2980b9", "stable": "#bdc3c7"}
tlabel = {"increasing": "Inc", "decreasing": "Dec", "stable": "Sta"}

for idx, pid in enumerate(patients):
    r, c = divmod(idx, ncols)
    ax = axes[r, c]

    pdata = tv[tv["patient_id"] == pid].copy()
    if pdata.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{pid}", fontsize=7)
        continue

    # Get sorted unique timepoints for this patient
    tps = sorted(pdata["tp_order"].unique())
    tp_labels = {o: l for o, l in zip(pdata["tp_order"], pdata["tp_label"])}
    tp_labels = [tp_labels[t] for t in tps]
    x_pos = np.arange(len(tps))

    variants = pdata["variant_id"].unique()
    n_var = len(variants)
    alpha_val = max(0.08, min(0.5, 15.0 / n_var))
    lw_val = max(0.3, min(1.0, 0.8 - n_var / 2000))
    for vid in variants:
        vdf = pdata[pdata["variant_id"] == vid].sort_values("tp_order")
        x_vals = [list(tps).index(o) for o in vdf["tp_order"].values if o in tps]
        y_vals = vdf["VAF"].values
        if len(x_vals) != len(y_vals):
            continue
        ttype = traj_lookup.get((pid, vid), "stable")
        ax.plot(x_vals, y_vals, color=tcolor.get(ttype, "#bdc3c7"),
                linewidth=lw_val, alpha=alpha_val)

    # Styling
    score = cs[cs["patient"] == pid]["total"].values[0]
    ax.set_title(f"{pid} ({score:.0f})", fontsize=6.5, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.3, len(tps) - 0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tp_labels, fontsize=4, rotation=45)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_ylabel("VAF", fontsize=5)

    # Add count statistics
    ptraj = traj[traj["patient_id"] == pid]
    inc_c = (ptraj["trajectory"] == "increasing").sum()
    dec_c = (ptraj["trajectory"] == "decreasing").sum()
    sta_c = (ptraj["trajectory"] == "stable").sum()
    ax.text(0.02, 0.98, f"\u2191{inc_c} \u2193{dec_c} \u2192{sta_c}",
            transform=ax.transAxes, fontsize=5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

# Hide unused subplots
for idx in range(N, nrows * ncols):
    r, c = divmod(idx, ncols)
    axes[r, c].set_visible(False)

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], color=c, lw=2, label=l) for c, l in
                   zip(["#e74c3c","#2980b9","#bdc3c7"], ["Increasing","Decreasing","Stable"])]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("figure_trajectory_grid.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figure_trajectory_grid.png")
