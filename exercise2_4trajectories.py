import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, warnings
warnings.filterwarnings("ignore"); sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
trusted = pd.read_csv("trusted_variants.csv")

# ── Find good illustrative examples for each archetype ──
# increasing: last_VAF > 1.5 * first_VAF, ideally with 3+ timepoints
inc = traj[(traj["trajectory"] == "increasing") & (traj["n_timepoints"] >= 3)].copy()
inc["fold_change"] = inc["last_VAF"] / inc["first_VAF"].clip(lower=0.001)
inc = inc.sort_values("fold_change", ascending=False)
print(f"Best increasing candidates ({len(inc)} with >=3 TPs):")
print(inc.head(5)[["patient_id","variant_id","first_VAF","last_VAF","fold_change"]].to_string(index=False))

# decreasing: last_VAF < 0.5 * first_VAF, started high, ended low
dec = traj[(traj["trajectory"] == "decreasing") & (traj["n_timepoints"] >= 3) & (traj["first_VAF"] > 0.1)].copy()
dec = dec.sort_values("first_VAF", ascending=False)
print(f"\nBest decreasing candidates ({len(dec)} with >=3 TPs, first_VAF>0.1):")
print(dec.head(5)[["patient_id","variant_id","first_VAF","last_VAF"]].to_string(index=False))

# stable: VAF stable across 3+ TPs, moderate VAF
stab = traj[(traj["trajectory"] == "stable") & (traj["n_timepoints"] >= 3) & (traj["first_VAF"] > 0.05) & (traj["first_VAF"] < 0.5)].copy()
stab["vaf_range"] = (stab["first_VAF"] - stab["last_VAF"]).abs()
stab = stab.sort_values("vaf_range")
print(f"\nBest stable candidates ({len(stab)} with >=3 TPs, 0.05<VAF<0.5):")
print(stab.head(5)[["patient_id","variant_id","first_VAF","last_VAF"]].to_string(index=False))

# undetected: all zeros, with multiple TPs — just pick a patient with 3+ TPs and show a flat 0 line
und = traj[(traj["trajectory"] == "undetected") & (traj["n_timepoints"] >= 3)]
print(f"\nUndetected candidates: {len(und)}")

# ── Pick one of each ──
picks = {
    "increasing": inc.iloc[0],
    "decreasing": dec.iloc[0],
    "stable": stab.iloc[0],
}

# For undetected, we need to show a variant never detected — find patient with 3+ TPs
# and pick a variant absent in all of them
long_patients = trusted.groupby("patient_id")["tp_label"].nunique()
long_patients = long_patients[long_patients >= 3].index
# Show all-zero row for a patient — use the first patient that has "T0" and "T1" and "T2"
# with a variant not in trusted rows
# Actually, we'll just show an example: pick a patient with 3+ TPs and show empty trajectory
sample_pid = und["patient_id"].iloc[0] if len(und) > 0 else long_patients[0]

print("\n" + "=" * 60)
print("SELECTED TRAJECTORIES")
print("=" * 60)

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (arch, row) in zip(axes.flat, picks.items()):
    pid = row["patient_id"]
    vid = row["variant_id"]

    pt_trusted = trusted[(trusted["patient_id"] == pid) & (trusted["variant_id"] == vid)]
    # Group by tp_order, take max VAF
    tp_data = pt_trusted.groupby("tp_order")["VAF"].max().sort_index()
    tp_orders = tp_data.index.tolist()
    tp_vafs = tp_data.values.tolist()

    # Fill in TPs where variant wasn't detected (gap) — note which TPs exist
    all_tps = sorted(trusted[trusted["patient_id"] == pid]["tp_order"].unique())

    # Build full timeline
    full_vafs = []
    for t in all_tps:
        if t in tp_data.index:
            full_vafs.append(tp_data[t])
        else:
            full_vafs.append(0.0)

    color = {"increasing": "red", "decreasing": "royalblue", "stable": "seagreen"}.get(arch, "gray")
    marker = {"increasing": "^", "decreasing": "v", "stable": "s"}.get(arch, "o")

    ax.plot(all_tps, full_vafs, f"-{marker}", color=color, ms=8, lw=2.5, label=f"{pid}")
    ax.fill_between(all_tps, full_vafs, alpha=0.1, color=color)

    # Highlight first and last VAF
    first_nz = row["first_VAF"]
    last_val = row["last_VAF"]
    ax.axhline(first_nz, color=color, ls=":", alpha=0.4, xmin=0, xmax=0.4)
    ax.axhline(last_val, color=color, ls="--", alpha=0.4, xmin=0.6, xmax=1.0)

    # Annotations
    ax.annotate(f"first={first_nz:.3f}", xy=(all_tps[0], full_vafs[0]),
                xytext=(5, 10), textcoords="offset points", fontsize=9, color=color)
    if full_vafs[-1] > 0:
        ax.annotate(f"last={last_val:.3f}", xy=(all_tps[-1], full_vafs[-1]),
                    xytext=(-30, -15), textcoords="offset points", fontsize=9, color=color)

    gene = vid.split(" | ")[-1] if " | " in vid else vid
    ax.set_title(f"{arch.upper()} — {gene} ({pid})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Timepoint order", fontsize=10)
    ax.set_ylabel("VAF", fontsize=10)
    ax.set_xticks(all_tps)
    ax.set_xticklabels([f"T{t}" if t == 0 else f"T{t}" for t in all_tps])
    ax.set_ylim(-0.02, max(max(full_vafs) * 1.2, 0.1))
    ax.legend(fontsize=9)

# ── Undetected: show a patient with 3+ TPs and a variant never seen ──
ax = axes[1, 1]
# Pick a patient with 3+ TPs and show flat zero
# Actually, find a variant that was truly never detected for a patient with 3+ TPs
# Better: just show a concept example from a real patient timepoint set
pid_ex = long_patients[0]
all_tps = sorted(trusted[trusted["patient_id"] == pid_ex]["tp_order"].unique())
zero_vafs = [0.0] * len(all_tps)
ax.plot(all_tps, zero_vafs, "-o", color="gray", ms=8, lw=2.5, alpha=0.6)
ax.axhline(0.0, color="gray", ls="--", alpha=0.3)
ax.set_title(f"UNDETECTED — never detected ({pid_ex})", fontsize=12, fontweight="bold")
ax.set_xlabel("Timepoint order", fontsize=10)
ax.set_ylabel("VAF", fontsize=10)
ax.set_xticks(all_tps)
ax.set_xticklabels([f"T{t}" for t in all_tps])
ax.set_ylim(-0.02, 0.15)
ax.text(0.5, 0.5, "VAF = 0 at all timepoints\n(clone absent)", 
        transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray", alpha=0.7)

plt.tight_layout()
plt.savefig("four_archetypes_trajectories.png", dpi=150)
print("\nSaved four_archetypes_trajectories.png")

# Print details
print("\n--- Selection details ---")
for arch, row in picks.items():
    pid, vid = row["patient_id"], row["variant_id"]
    gene = vid.split(" | ")[-1] if " | " in vid else vid
    print(f"  {arch:12s}: {gene:30s} {pid:10s} first_VAF={row['first_VAF']:.4f} last_VAF={row['last_VAF']:.4f} n_TPs={row['n_timepoints']}")
print(f"  undetected  : {'(example)':30s} {pid_ex:10s} (no data — all zeros)")

print("\nDone.")
