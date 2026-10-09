import pandas as pd
import numpy as np

DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
tv = pd.read_csv("trusted_variants.csv")
df = pd.read_csv("filtered_variants.csv")
lod_map = df.groupby("Genes")["LoD"].first().to_dict()

def get_first_gene(g):
    return str(g).split(",")[0].strip()
def any_driver(g):
    return any(d in [x.strip() for x in str(g).split(",")] for d in DRIVER_GENES)

patients = sorted(tv["patient_id"].unique())

# tb = mean VAF across ALL observations * 400 cap 40 (captures total ctDNA level)
# Higher mean VAF = more ctDNA circulating
mean_vaf = tv.groupby("patient_id")["VAF"].mean()
tb2 = (mean_vaf * 400).clip(upper=40).round(2)

# Alternative: tb = baseline (T0) VAF sum, scaled
t0_vaf = tv[tv["tp_label"] == "T0"].groupby("patient_id")["VAF"].sum()
# But not all patients have T0 data in trusted variants (below LoD at T0)
# Use filtered_variants instead
t0_vaf_all = df[df["tp_label"] == "T0"].groupby("patient_id")["VAF"].sum()
# Scale to 0-40: cap at 1.0 (100% total VAF = 40)
tb_t0 = (t0_vaf_all * 40).clip(upper=40).round(2)

# tb = max VAF at any timepoint (single highest variant VAF) * 40
max_vaf = tv.groupby("patient_id")["VAF"].max()
tb_max = (max_vaf * 40).round(2)

# tb = sqrt(mean VAF sum per timepoint) * 15
tps = tv.groupby(["patient_id", "tp_label"])["VAF"].sum().groupby("patient_id").mean()
tb_sqrt = np.sqrt(tps) * 15

print("Comparing tb definitions:")
for name, s in [("mean_vaf*400", tb2), ("t0_sum*40", tb_t0), ("max_vaf*40", tb_max), ("sqrt(mean)*15", tb_sqrt)]:
    n0 = (s == 0).sum()
    ncap = (s >= 40).sum()
    print(f"{name:>20s}: range {s.min():.2f}-{s.max():.2f}, zeros={n0}, capped={ncap}")

# Build full dataset with trajectory counts
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

# Build with tb = baseline T0 VAF sum scaled
tb_use = tb_t0  # baseline burden
ce = inc.groupby("patient_id").size().map(lambda x: 0 if x==0 else (8 if x==1 else (17 if x==2 else 25)))
drv = inc[inc["first_gene"].isin(DRIVER_GENES)].groupby("patient_id").size().map(lambda x: 0 if x==0 else (10 if x==1 else (20 if x==2 else 25)))
mg = gains_s.map(lambda x: 0 if x==0 else (5 if x<=3 else 10))

# Align all to same index
all_idx = patients
tb_s = tb_use.reindex(all_idx, fill_value=0)
ce_s = ce.reindex(all_idx, fill_value=0)
drv_s = drv.reindex(all_idx, fill_value=0)
mg_s = mg.reindex(all_idx, fill_value=0)

# Check
print(f"\nWith tb = T0_sum*40:")
for name, s in [("tb", tb_s), ("ce", ce_s), ("drv", drv_s), ("mg", mg_s)]:
    print(f"  {name}: range {s.min():.0f}-{s.max():.0f}")

total = tb_s + ce_s + drv_s + mg_s
n100 = (total >= 100).sum()
n60 = ((total >= 55) & (total <= 65)).sum()
print(f"\nAt 100: {n100}")
print(f"At ~60: {n60}")

# Also try tb = sqrt(mean)*15
tb_s2 = tb_sqrt.reindex(all_idx, fill_value=0).clip(upper=40)
total2 = tb_s2 + ce_s + drv_s + mg_s
n100_2 = (total2 >= 100).sum()
n60_2 = ((total2 >= 55) & (total2 <= 65)).sum()
print(f"\nWith tb = sqrt(mean)*15 cap 40:")
print(f"  At 100: {n100_2}")
print(f"  At ~60: {n60_2}")

# Show the full table with sqrt(mean)*15 cap 40
print(f"\n{'Patient':<12} {'tb':>5} {'ce':>4} {'drv':>4} {'mg':>4} {'Total':>6}")
print("-" * 40)
for p in sorted(total2.sort_values(ascending=False).index, key=lambda x: -total2[x]):
    star = " *" if total2[p] >= 100 else ""
    print(f"{p:<12} {tb_s2[p]:>5.0f} {ce_s[p]:>4.0f} {drv_s[p]:>4.0f} {mg_s[p]:>4.0f} {total2[p]:>6.0f}{star}")

# For "score ~60 with tb=0, ce=25, drv=25, mg=10" pattern = 0+25+25+10=60
# Find patients where ce=25, drv=25, mg=10 but tb < 5
pattern60 = (ce_s == 25) & (drv_s == 25) & (mg_s == 10) & (tb_s2 < 5)
print(f"\nPattern '60 w/ tb=0, ce=25, drv=25, mg=10': {pattern60.sum()} patients")
for p in total2[pattern60].sort_values(ascending=False).index:
    print(f"  {p}: tb={tb_s2[p]:.0f} ce={ce_s[p]:.0f} drv={drv_s[p]:.0f} mg={mg_s[p]:.0f} total={total2[p]:.0f}")
