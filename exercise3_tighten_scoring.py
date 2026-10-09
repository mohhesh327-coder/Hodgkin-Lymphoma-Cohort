import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")

# ── Old scoring (VAF-weighted, inc weight=2.0, dec weight=0.5, stable=1.0) ──
def score_old(traj_df):
    inc_w, dec_w, sta_w = 2.0, 0.5, 1.0
    out = {}
    for pid, grp in traj_df.groupby("patient_id"):
        s = 0
        for _, r in grp.iterrows():
            if r["trajectory"] == "increasing":   s += r["last_VAF"] * inc_w
            elif r["trajectory"] == "decreasing": s += r["last_VAF"] * dec_w
            elif r["trajectory"] == "stable":     s += r["last_VAF"] * sta_w
        out[pid] = round(s, 4)
    return pd.Series(out)

# ── New scoring (flat points: inc: 0/12/25, dec#0, stable) ──
def score_new(traj_df):
    out = {}
    for pid, grp in traj_df.groupby("patient_id"):
        n_inc = (grp["trajectory"] == "increasing").sum()
        n_dec = (grp["trajectory"] == "decreasing").sum()
        n_sta = (grp["trajectory"] == "stable").sum()

        # Increasing: tiered points
        if n_inc == 0:    inc_pts = 0
        elif n_inc == 1:  inc_pts = 12
        else:             inc_pts = 25  # 2 or more

        # Decreasing: +5 if any clearing
        dec_pts = 5 if n_dec > 0 else 0

        # Stable: +10 per variant (capped at 30)
        sta_pts = min(n_sta * 10, 30)

        out[pid] = inc_pts + dec_pts + sta_pts
    return pd.Series(out)

old_scores = score_old(traj)
new_scores = score_new(traj)

compare = pd.DataFrame({"old_vaf_weighted": old_scores, "new_flat_points": new_scores})
compare["new_flat_points"] = compare["new_flat_points"].fillna(0)
compare["diff"] = compare["new_flat_points"] - compare["old_vaf_weighted"]

print("="*65)
print("SCORING COMPARISON: VAF-weighted vs Flat-points")
print("="*65)

# Print all patients sorted by new score descending
compare_sorted = compare.sort_values("new_flat_points", ascending=False)
compare_sorted["old_rank"] = compare_sorted["old_vaf_weighted"].rank(ascending=False).astype(int)
compare_sorted["new_rank"] = compare_sorted["new_flat_points"].rank(ascending=False).astype(int)
compare_sorted["rank_change"] = compare_sorted["old_rank"] - compare_sorted["new_rank"]

print(f"\n{'Patient':12s} {'Old (VAF-w)':>12s} {'New (points)':>13s} {'Diff':>8s} {'Old->New':>10s}")
print("-"*55)
for pid, row in compare_sorted.iterrows():
    arrow = "->".join([str(row["old_rank"]), str(row["new_rank"])])
    print(f"{pid:12s} {row['old_vaf_weighted']:>12.2f} {int(row['new_flat_points']):>13d} {row['diff']:>+8.2f} {arrow:>10s}")

# ── Which patients' scores change meaningfully? ──
print("\n" + "="*65)
print("PATIENTS WHOSE SCORE CHANGES (>5 points or rank shift >5)")
print("="*65)

changed = compare_sorted[compare_sorted["diff"].abs() > 5]
if len(changed) > 0:
    ch = changed[["old_vaf_weighted", "new_flat_points", "diff"]].copy()
    ch["new_flat_points"] = ch["new_flat_points"].astype(int)
    print(ch.to_string())

big_rank = compare_sorted[compare_sorted["rank_change"].abs() >= 5]
if len(big_rank) > 0:
    print(f"\n  {len(big_rank)} patients with rank shift >= 5:")
    for pid, row in big_rank.iterrows():
        print(f"    {pid:12s}  rank {row['old_rank']} -> {row['new_rank']}  (D{int(row['rank_change']):+d})")

# ── Explain why ──
print("\n" + "="*65)
print("EXPLANATION")
print("="*65)

inc_counts = traj[traj["trajectory"] == "increasing"].groupby("patient_id").size()
dec_counts = traj[traj["trajectory"] == "decreasing"].groupby("patient_id").size()
sta_counts = traj[traj["trajectory"] == "stable"].groupby("patient_id").size()

for pid in compare_sorted.index:
    nic = inc_counts.get(pid, 0)
    ndc = dec_counts.get(pid, 0)
    nsc = sta_counts.get(pid, 0)
    old = compare.loc[pid, "old_vaf_weighted"]
    new = compare.loc[pid, "new_flat_points"]
    inc_pts = 0 if nic == 0 else (12 if nic == 1 else 25)
    dec_pts = 5 if ndc > 0 else 0
    sta_pts = min(nsc * 10, 30)
    if abs(compare.loc[pid, "diff"]) > 5 or abs(compare_sorted.loc[pid, "rank_change"]) >= 5:
        print(f"\n  {pid:12s}  inc={nic} dec={ndc} stable={nsc}  ->  "
              f"inc_pts={inc_pts}+dec_pts={dec_pts}+sta_pts={sta_pts} = {new}")
        print(f"              Old VAF-weighted: {old:.2f}  |  "
              f"Reason: {'inc clone bonus' if nic > 0 else 'many stable variants with high VAF'}")

# ── Save ──
compare_sorted.to_csv("scoring_comparison.csv")
print(f"\nSaved scoring_comparison.csv")
print("Done.")
