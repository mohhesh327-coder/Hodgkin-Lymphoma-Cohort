import pandas as pd
import numpy as np

DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
tv = pd.read_csv("trusted_variants.csv")
df = pd.read_csv("filtered_variants.csv")
lod_map = df.groupby("Genes")["LoD"].first().to_dict()

def get_first_gene(g):
    return str(g).split(",")[0].strip()

patients = sorted(tv["patient_id"].unique())

inc = traj[traj["trajectory"] == "increasing"].copy()
inc["first_gene"] = inc["variant_id"].str.split("|").str[1].str.strip().apply(get_first_gene)

def count_gains(pid):
    p = df[df["patient_id"]==pid]
    tps = sorted(p["tp_label"].unique())
    if len(tps) < 2: return 0
    n = 0
    for vid, grp in p.groupby("variant_id"):
        grp = grp.sort_values("tp_order")
        vafs = grp.groupby("tp_label")["VAF"].mean()
        g = get_first_gene(grp["Genes"].iloc[0] if len(grp) > 0 else "")
        lod = lod_map.get(g, 0.01)
        if vafs.get(tps[0], 0) > 0.001: continue
        if any(vafs.get(tp,0) > lod for tp in tps[1:]): n += 1
    return n

gains_s = pd.Series({p: count_gains(p) for p in patients})

# ce counts
ce_counts = inc.groupby("patient_id").size()

# drv counts
drv_counts = inc[inc["first_gene"].isin(DRIVER_GENES)].groupby("patient_id").size()

# tb: sum of VAF rises across all increasing variants
# Compute factor so all patients with max non-tb score can reach tb=40
sum_rise_s = pd.Series({pid: inc[inc["patient_id"]==pid].apply(
    lambda r: r["last_VAF"] - r["first_VAF"], axis=1).sum() for pid in patients})

# Determine anchor: lowest rise among patients with max ce+drv+mg
ce_anchor = ce_counts.reindex(patients, fill_value=0).map(lambda x: 0 if x==0 else (8 if x==1 else (17 if x==2 else 25)))
drv_anchor = drv_counts.reindex(patients, fill_value=0).map(lambda x: 0 if x==0 else (10 if x==1 else 25))
mg_anchor = gains_s.reindex(patients, fill_value=0).map(lambda x: 0 if x==0 else (5 if x<=2 else 10))
cdm60 = (ce_anchor + drv_anchor + mg_anchor) >= 60
anchor_rise = sum_rise_s[cdm60].sort_values(ascending=False)
# Target 9 patients at 100: use 9th highest rise among cdm60 patients as anchor
target_count = min(9, len(anchor_rise))
anchor_value = anchor_rise.iloc[target_count-1] if len(anchor_rise) >= target_count else sum_rise_s.max()
factor = 40 / anchor_value if anchor_value > 0 else 1
tb = (sum_rise_s * factor).clip(upper=40).round(2)
print(f"tb factor: {factor:.2f} (anchor: {target_count}th highest rise among {len(anchor_rise)} patients with ce+drv+mg=60)")
print(f"Patients at tb=40: {(tb>=40).sum()}")

# ce: 0/8/17/25 (≥3 inc for max)
ce = ce_counts.map(lambda x: 0 if x==0 else (8 if x==1 else (17 if x==2 else 25)))

# drv: 0/10/25 (≥2 driver inc for max)
drv = drv_counts.map(lambda x: 0 if x==0 else (10 if x==1 else 25))

# mg: 0/5/10 (gains 0→0, 1-2→5, 3+→10)
mg = gains_s.map(lambda x: 0 if x==0 else (5 if x<=2 else 10))

# Align
tb_s = tb.reindex(patients, fill_value=0).round(2)
ce_s = ce.reindex(patients, fill_value=0)
drv_s = drv.reindex(patients, fill_value=0)
mg_s = mg.reindex(patients, fill_value=0)

total = (tb_s + ce_s + drv_s + mg_s).round(2)

# Distribution
def categorize(t):
    if t >= 95: return "High-confidence"
    elif t >= 55: return "Strong recurrence"
    elif t >= 30: return "Possible"
    elif t >= 10: return "Weak"
    else: return "None"

cats = {"High-confidence": 0, "Strong recurrence": 0, "Possible": 0, "Weak": 0, "None": 0}
for p in patients:
    cats[categorize(total[p])] += 1

print(f"\n{'Patient':<12} {'tb':>6} {'ce':>4} {'drv':>4} {'mg':>4} {'Total':>6}  {'Cat':<18} {'rise':>7}")
print("-" * 68)
for p in total.sort_values(ascending=False).index:
    t, c, d, m, tot = tb_s[p], ce_s[p], drv_s[p], mg_s[p], total[p]
    cat = categorize(tot)
    r = sum_rise_s.get(p, 0)
    print(f"{p:<12} {t:>6.1f} {c:>4.0f} {d:>4.0f} {m:>4.0f} {tot:>6.0f}  {cat:<18} {r:>7.4f}")

print(f"\nDistribution:")
for k in ["High-confidence", "Strong recurrence", "Possible", "Weak", "None"]:
    print(f"  {k}: {cats[k]} ({cats[k]/46*100:.0f}%)")

# Save FINAL files
pd.Series(tb_s.values, index=patients, name="tb").to_csv("tb_scores.csv")
pd.Series(ce_s.values, index=patients, name="ce").to_csv("ce_scores.csv")
pd.Series(drv_s.values, index=patients, name="drv").to_csv("drv_scores.csv")
pd.Series(mg_s.values, index=patients, name="mg").to_csv("mg_scores.csv")
pd.DataFrame({"patient": patients, "tb": tb_s.values, "ce": ce_s.values,
              "drv": drv_s.values, "mg": mg_s.values, "total": total.values}).to_csv("composite_scores.csv", index=False)
print(f"\nSaved final score files.")

# Quick reference: who are the 9 with total >= 100?
at100 = total[total >= 100]
print(f"\nPatients at 100: {len(at100)}")
for p in at100.sort_values(ascending=False).index:
    print(f"  {p}: tb={tb_s[p]:.1f} ce={ce_s[p]:.0f} drv={drv_s[p]:.0f} mg={mg_s[p]:.0f}")
