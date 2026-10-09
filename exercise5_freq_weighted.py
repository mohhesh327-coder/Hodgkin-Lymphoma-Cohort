import pandas as pd
import numpy as np

DRIVER_GENES = ["TP53", "NOTCH1", "KRAS", "EP300", "CASP8",
                "MYB", "CARD11", "TNFAIP3", "STAT3", "STAT6",
                "CREBBP", "SMARCA4"]

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
tv = pd.read_csv("trusted_variants.csv")

# --- Helper ---
def gene_matches(gene_str, target):
    return target in [g.strip() for g in str(gene_str).split(",")]

def any_driver(gene_str):
    return any(gene_matches(gene_str, g) for g in DRIVER_GENES)

def get_first_gene(gene_str):
    return str(gene_str).split(",")[0].strip()

traj["gene"] = traj["variant_id"].str.split("|").str[1].str.strip()
traj["first_gene"] = traj["gene"].apply(get_first_gene)

# === Cohort frequency per driver gene ===
gene_hits = {}
for g in DRIVER_GENES:
    gene_hits[g] = traj["gene"].apply(lambda x: gene_matches(x, g)).sum()

total_driver_hits = sum(gene_hits.values())
print(f"Total driver hits: {total_driver_hits}")
for g in DRIVER_GENES:
    pct = gene_hits[g] / total_driver_hits * 100
    w = total_driver_hits / gene_hits[g] if gene_hits[g] > 0 else 0
    print(f"  {g:10s}: {gene_hits[g]:4d} hits ({pct:4.1f}%)  inverse-freq weight = {w:6.2f}")

# ================================================================
# EXERCISE 1: Frequency-weighted Driver Score
# ================================================================
print("\n" + "=" * 70)
print("EXERCISE 1: FREQUENCY-WEIGHTED DRIVER SCORE")
print("=" * 70)

# Build per-patient summary: all variants + trajectory
patient_scores = []
for pid in sorted(traj["patient_id"].unique()):
    p_traj = traj[traj["patient_id"] == pid]
    p_tv = tv[tv["patient_id"] == pid]

    n_inc = (p_traj["trajectory"] == "increasing").sum()
    n_dec = (p_traj["trajectory"] == "decreasing").sum()
    n_sta = (p_traj["trajectory"] == "stable").sum()

    # --- Original flat-point score from Ex3 ---
    inc_pts_flat = {0: 0, 1: 12}.get(n_inc, 25)
    dec_pts = 5 if n_dec > 0 else 0
    sta_pts = min(n_sta * 10, 30)
    old_score = inc_pts_flat + dec_pts + sta_pts

    # --- Frequency-weighted score ---
    inc_drivers = p_traj[(p_traj["trajectory"] == "increasing") & (p_traj["gene"].apply(any_driver))]
    inc_weight_sum = 0
    for _, row in inc_drivers.iterrows():
        g = row["first_gene"]
        if g in DRIVER_GENES and gene_hits[g] > 0:
            w = total_driver_hits / gene_hits[g]  # inverse frequency
        else:
            w = 0
        inc_weight_sum += w

    weighted_score = inc_weight_sum + dec_pts + sta_pts

    patient_scores.append({
        "patient": pid,
        "n_inc": n_inc, "n_dec": n_dec, "n_sta": n_sta,
        "n_driver_inc": len(inc_drivers),
        "inc_genes": [row["first_gene"] for _, row in inc_drivers.iterrows()],
        "old_score": old_score,
        "weighted_score": round(weighted_score, 2)
    })

scores = pd.DataFrame(patient_scores)

# Rankings
scores["old_rank"] = scores["old_score"].rank(ascending=False)
scores["weighted_rank"] = scores["weighted_score"].rank(ascending=False)
scores["rank_change"] = scores["old_rank"] - scores["weighted_rank"]

print(f"\n{'Patient':<12} {'Old':>8} {'Weighted':>10} {'OldR':>5} {'NewR':>5} {'Chg':>5} {'Driver inc genes'}")
print("-" * 80)
for _, row in scores.sort_values("weighted_rank").iterrows():
    genes_str = ",".join(row["inc_genes"]) if row["inc_genes"] else "-"
    print(f"{row['patient']:<12} {row['old_score']:>8.0f} {row['weighted_score']:>10.2f} "
          f"{int(row['old_rank']):>5d} {int(row['weighted_rank']):>5d} {row['rank_change']:>+5.0f}  {genes_str}")

# Spearman correlation
from scipy.stats import spearmanr
rho, pval = spearmanr(scores["old_score"], scores["weighted_score"])
print(f"\nSpearman r = {rho:.4f} (p = {pval:.2e})")

# Which gene has the biggest effect?
print("\nWhich gene has the biggest effect on scoring?")
gene_impact = {}
for g in DRIVER_GENES:
    has_gene = scores["inc_genes"].apply(lambda genes: g in genes)
    if has_gene.any():
        avg_weight = total_driver_hits / gene_hits[g]
        gene_impact[g] = {"patients": has_gene.sum(), "weight": avg_weight,
                          "total_contribution": avg_weight * has_gene.sum()}
    else:
        gene_impact[g] = {"patients": 0, "weight": 0, "total_contribution": 0}

impact_df = pd.DataFrame(gene_impact).T.sort_values("total_contribution", ascending=False)
print(f"{'Gene':10s} {'Patients':>9} {'Weight':>8} {'Total Contrib':>14}")
for g, row in impact_df.iterrows():
    print(f"{g:10s} {int(row['patients']):>9d} {row['weight']:>8.2f} {row['total_contribution']:>14.2f}")

top_gene = impact_df.index[0]
print(f"\n*** {top_gene} has the biggest effect on scoring "
      f"(appears in {int(impact_df.loc[top_gene,'patients'])} patients, "
      f"weight={impact_df.loc[top_gene,'weight']:.1f}, "
      f"total contribution={impact_df.loc[top_gene,'total_contribution']:.0f})")

# Save
scores.to_csv("scoring_freq_weighted.csv", index=False)
print("\nSaved scoring_freq_weighted.csv")

# ================================================================
# EXERCISE 2: Remove TP53 from high-weight tier
# ================================================================
print("\n" + "=" * 70)
print("EXERCISE 2: REMOVE TP53 FROM HIGH-WEIGHT TIER")
print("=" * 70)

# High-weight tier = NOTCH1 (25 pts), TP53 (25 pts)
# Mid-weight tier = other drivers (12 pts each)
# Non-driver inc = 0 pts

HIGH_TIER_25 = {"NOTCH1", "TP53"}
HIGH_TIER_NO_TP53 = {"NOTCH1"}

def score_tiered(patient_id, high_tier):
    p_traj = traj[traj["patient_id"] == patient_id]
    inc = p_traj[p_traj["trajectory"] == "increasing"].copy()
    inc["first_gene"] = inc["gene"].apply(get_first_gene)

    inc_pts = 0
    for _, row in inc.iterrows():
        g = row["first_gene"]
        if g in high_tier:
            inc_pts += 25
        elif any_driver(row["gene"]):
            inc_pts += 12

    n_dec = (p_traj["trajectory"] == "decreasing").sum()
    n_sta = (p_traj["trajectory"] == "stable").sum()
    dec_pts = 5 if n_dec > 0 else 0
    sta_pts = min(n_sta * 10, 30)
    return inc_pts + dec_pts + sta_pts

# Score both versions
for label, high_tier in [("With TP53 in high tier", HIGH_TIER_25),
                          ("TP53 removed from high tier", HIGH_TIER_NO_TP53)]:
    tier_scores = []
    for pid in sorted(traj["patient_id"].unique()):
        s = score_tiered(pid, high_tier)
        tier_scores.append({"patient": pid, "score": s})
    ts = pd.DataFrame(tier_scores)
    ts["rank"] = ts["score"].rank(ascending=False)
    print(f"\n{label}:")
    print(f"  Mean score: {ts['score'].mean():.1f}, Median: {ts['score'].median():.0f}")
    ts_sorted = ts.sort_values("score", ascending=False)
    for _, row in ts_sorted.iterrows():
        bar = "#" * max(1, int(row["score"] / 5))
        print(f"  {row['patient']:<12} {row['score']:>5.0f} {bar}")

# Compare
scores_with_tp53 = pd.DataFrame([{"patient": pid, "score_with_tp53": score_tiered(pid, HIGH_TIER_25)}
                                  for pid in sorted(traj["patient_id"].unique())])
scores_no_tp53 = pd.DataFrame([{"patient": pid, "score_no_tp53": score_tiered(pid, HIGH_TIER_NO_TP53)}
                                for pid in sorted(traj["patient_id"].unique())])

compare = scores_with_tp53.merge(scores_no_tp53, on="patient")
compare["diff"] = compare["score_with_tp53"] - compare["score_no_tp53"]
dropped = compare[compare["diff"] > 0]
print(f"\n\nPatients with TP53 inc mutation who lose 25 pts:")
print(f"{'Patient':<12} {'Score with TP53':>16} {'Score without':>13}")
for _, row in dropped.iterrows():
    print(f"{row['patient']:<12} {row['score_with_tp53']:>16.0f} {row['score_no_tp53']:>13.0f}")

# Which patients had TP53 increasing?
tp53_inc = traj[(traj["trajectory"] == "increasing") & (traj["gene"].apply(lambda x: gene_matches(x, "TP53")))]
print(f"\nPatients with TP53-increasing mutations: {len(tp53_inc)}")
for _, row in tp53_inc.iterrows():
    print(f"  {row['patient_id']}  {row['variant_id']}")

# === FIGURE: comparison of old vs freq-weighted rankings ===
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Gene weights
ax1 = axes[0]
impact_sorted = impact_df[impact_df["weight"] > 0].sort_values("weight")
genes_plot = impact_sorted.index.tolist()
weights = impact_sorted["weight"].values
colors1 = ["#d73027"] * len(genes_plot)
bars1 = ax1.barh(range(len(genes_plot)), weights, color=colors1, height=0.6, edgecolor="white")
for i, (g, w) in enumerate(zip(genes_plot, weights)):
    ax1.text(w + 0.5, i, f"{w:.1f}  ({int(impact_sorted.loc[g,'patients'])} pts)", va="center", fontsize=9)
ax1.set_yticks(range(len(genes_plot)))
ax1.set_yticklabels(genes_plot, fontsize=11, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Inverse-frequency weight", fontsize=12)
ax1.set_title("Driver gene weights\n( = total_hits / gene_hits )", fontsize=13, fontweight="bold", loc="left")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Panel 2: Ranking comparison (old flat vs freq-weighted)
ax2 = axes[1]
scores_plot = scores.sort_values("weighted_score", ascending=False).head(30)
x = range(len(scores_plot))
ax2.plot(x, scores_plot["old_rank"].values, "o-", color="#4575b4", linewidth=1.5, markersize=6, label="Flat-point rank")
ax2.plot(x, scores_plot["weighted_rank"].values, "s-", color="#d73027", linewidth=1.5, markersize=6, label="Freq-weighted rank")
for i, (_, row) in enumerate(scores_plot.iterrows()):
    if abs(row["rank_change"]) >= 5:
        ax2.annotate(row["patient"], xy=(i, row["weighted_rank"]), fontsize=7, color="#d73027", fontweight="bold",
                     ha="center", va="bottom" if row["rank_change"] > 0 else "top")
ax2.set_xlabel("Patients (sorted by weighted rank)", fontsize=12)
ax2.set_ylabel("Rank (lower = better)", fontsize=12)
ax2.set_title("Ranking comparison: flat-point vs freq-weighted", fontsize=13, fontweight="bold", loc="left")
ax2.legend(fontsize=10)
ax2.invert_yaxis()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.text(0.98, 0.98, f"Spearman r = {rho:.4f}", transform=ax2.transAxes, fontsize=10,
         va="top", ha="right", bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ccc"))

fig.suptitle("Exercise 1: Frequency-weighted Driver Scoring", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("exercise5_freq_weighted.png", dpi=200, bbox_inches="tight")
print("\nSaved exercise5_freq_weighted.png")
