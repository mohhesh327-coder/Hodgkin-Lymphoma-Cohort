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

# tb = sum of (last_VAF - first_VAF) for all increasing variants
# This captures the MAGNITUDE of clonal expansion (how much VAF actually rose)
tb_raw = {}
for pid in patients:
    p_inc = inc[inc["patient_id"] == pid]
    total_rise = 0
    for _, row in p_inc.iterrows():
        first_vaf = row["first_VAF"]
        last_vaf = row["last_VAF"]
        total_rise += last_vaf - first_vaf
    tb_raw[pid] = total_rise

tb_series = pd.Series(tb_raw)
max_rise = tb_series.max()
tb = (tb_series / max_rise * 40).round(2)

# ce (unchanged)
ce_counts = inc.groupby("patient_id").size()
ce = ce_counts.map(lambda x: 0 if x==0 else (8 if x==1 else (17 if x==2 else 25)))

# drv (unchanged)
drv_counts = inc[inc["first_gene"].isin(DRIVER_GENES)].groupby("patient_id").size()
drv = drv_counts.map(lambda x: 0 if x==0 else (10 if x==1 else (20 if x==2 else 25)))

# mg (unchanged)
mg = gains_s.map(lambda x: 0 if x==0 else (5 if x<=3 else 10))

# Align
all_idx = patients
tb_s = tb.reindex(all_idx, fill_value=0)
ce_s = ce.reindex(all_idx, fill_value=0)
drv_s = drv.reindex(all_idx, fill_value=0)
mg_s = mg.reindex(all_idx, fill_value=0)

total = tb_s + ce_s + drv_s + mg_s

print(f"{'Patient':<12} {'tb':>6} {'ce':>4} {'drv':>4} {'mg':>4} {'Total':>6} {'inc':>4} {'rise':>7}")
print("-" * 55)
for p in total.sort_values(ascending=False).index:
    t = tb_s[p]; c = ce_s[p]; d = drv_s[p]; m = mg_s[p]; tot = t + c + d + m
    ni = ce_counts.get(p, 0)
    r = tb_raw.get(p, 0)
    star = " *" if tot >= 100 else ""
    print(f"{p:<12} {t:>6.2f} {c:>4.0f} {d:>4.0f} {m:>4.0f} {tot:>6.0f}{star} {ni:>4d} {r:>7.4f}")

n100 = (total >= 100).sum()
print(f"\nPerfect (≥100): {n100}")

# Check the "60 club" pattern: tb≈0, ce=25, drv=25, mg=10
pattern60 = (ce_s == 25) & (drv_s == 25) & (mg_s == 10) & (tb_s < 5)
print(f"Pattern 'tb≈0, ce=25, drv=25, mg=10' (score ~60): {pattern60.sum()}")
for p in total[pattern60].sort_values(ascending=False).index:
    print(f"  {p}: tb={tb_s[p]:.2f} ce={ce_s[p]:.0f} drv={drv_s[p]:.0f} mg={mg_s[p]:.0f} total={total[p]:.0f}")

# Full distribution
print(f"\nDistribution by total:")
cats = {"high_conf (≥95)": 0, "strong (55-65)": 0, "possible (35-54)": 0, "weak (15-34)": 0, "none (<15)": 0}
for p in patients:
    t = total[p]
    if t >= 95: cats["high_conf (≥95)"] += 1
    elif t >= 55: cats["strong (55-65)"] += 1
    elif t >= 35: cats["possible (35-54)"] += 1
    elif t >= 15: cats["weak (15-34)"] += 1
    else: cats["none (<15)"] += 1
for k, v in cats.items():
    print(f"  {k}: {v} ({v/46*100:.0f}%)")
