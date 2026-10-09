import pandas as pd
import numpy as np

DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
tv = pd.read_csv("trusted_variants.csv")
tbs = pd.read_csv("tumor_burden_scores.csv", index_col=0)
df = pd.read_csv("filtered_variants.csv")
lod_map = df.groupby("Genes")["LoD"].first().to_dict()

def get_first_gene(gene_str):
    return str(gene_str).split(",")[0].strip()
def any_driver(g):
    return any(d in [x.strip() for x in str(g).split(",")] for d in DRIVER_GENES)

patients = sorted(tv["patient_id"].unique())

# Build per-patient dataset
rows = []
for pid in patients:
    p_traj = traj[traj["patient_id"] == pid]
    p_tv = tv[tv["patient_id"] == pid]
    p_tv["is_driver"] = p_tv["Genes"].apply(any_driver)

    n_inc = (p_traj["trajectory"] == "increasing").sum()
    n_dec = (p_traj["trajectory"] == "decreasing").sum()
    n_sta = (p_traj["trajectory"] == "stable").sum()

    inc = p_traj[p_traj["trajectory"] == "increasing"].copy()
    inc["first_gene"] = inc["variant_id"].str.split("|").str[1].str.strip().apply(get_first_gene)
    n_driver_inc = inc["first_gene"].isin(DRIVER_GENES).sum()

    driver_genes_affected = set()
    for _, r in p_tv[p_tv["is_driver"]].iterrows():
        for g in str(r["Genes"]).split(","):
            g = g.strip()
            if g in DRIVER_GENES:
                driver_genes_affected.add(g)

    n_driver_uv = p_tv[p_tv["is_driver"]]["variant_id"].nunique()

    # Gains
    p_df = df[df["patient_id"] == pid]
    all_tps = sorted(p_df["tp_label"].unique())
    n_gains = 0
    if len(all_tps) >= 2:
        for vid, grp in p_df.groupby("variant_id"):
            grp = grp.sort_values("tp_order")
            vafs = grp.groupby("tp_label")["VAF"].mean()
            g = get_first_gene(grp["Genes"].iloc[0] if len(grp) > 0 else "")
            lod = lod_map.get(g, 0.01)
            if vafs.get(all_tps[0], 0) > 0.001: continue
            if any(vafs.get(tp, 0) > lod for tp in all_tps[1:]): n_gains += 1

    mean_sum = tbs.loc[pid, "score_sum"] if pid in tbs.index else 0

    rows.append({
        "patient": pid, "n_inc": n_inc, "n_driver_inc": n_driver_inc,
        "n_driver_genes": len(driver_genes_affected), "n_driver_uv": n_driver_uv,
        "n_gains": n_gains, "mean_sum": round(mean_sum, 2)
    })

res = pd.DataFrame(rows)

# ====== FINAL SUB-SCORE DEFINITIONS ======
# tb (0-40): driver genes affected * 5, capped at 40
#   Captures breadth of driver gene involvement = tumor mutational burden
tb = (res["n_driver_genes"] * 5).clip(upper=40).values

# ce (0-25): tiered based on increasing mutation count
#   0→0, 1→8, 2→17, 3+→25 (requires >=3 for max)
ce = res["n_inc"].map(lambda x: 0 if x == 0 else (8 if x == 1 else (17 if x == 2 else 25))).values

# drv (0-25): tiered based on driver increasing mutation count
#   0→0, 1→10, 2→20, 3+→25 (requires >=3 for max)
drv = res["n_driver_inc"].map(lambda x: 0 if x == 0 else (10 if x == 1 else (20 if x == 2 else 25))).values

# mg (0-10): tiered based on gained mutation count
#   0→0, 1-3→5, 4+→10
mg = res["n_gains"].map(lambda x: 0 if x == 0 else (5 if x <= 3 else 10)).values

total = tb + ce + drv + mg

# ====== OUTPUT ======
print(f"{'Patient':<12} {'tb':>4} {'ce':>4} {'drv':>4} {'mg':>4} {'Total':>6}  {'inc':>3} {'drv_inc':>8} {'gains':>6} {'drv_genes':>10}")
print("-" * 75)
for i in np.argsort(-total):
    p = res.iloc[i]["patient"]
    star = " *" if total[i] >= 100 else ""
    print(f"{p:<12} {tb[i]:>4.0f} {ce[i]:>4.0f} {drv[i]:>4.0f} {mg[i]:>4.0f} {total[i]:>6.0f}{star}"
          f"  {res.iloc[i]['n_inc']:>3d} {res.iloc[i]['n_driver_inc']:>8d} {res.iloc[i]['n_gains']:>6d} {res.iloc[i]['n_driver_genes']:>10d}")

n100 = (total >= 100).sum()
print(f"\nPerfect scorers (>= 100): {n100}")
print(f"Range: {total.min():.0f}–{total.max():.0f}")

# Save final DataFrames
pd.Series(tb, index=res["patient"], name="tb").to_csv("tb_scores.csv")
pd.Series(ce, index=res["patient"], name="ce").to_csv("ce_scores.csv")
pd.Series(drv, index=res["patient"], name="drv").to_csv("drv_scores.csv")
pd.Series(mg, index=res["patient"], name="mg").to_csv("mg_scores.csv")
pd.DataFrame({"patient": res["patient"], "tb": tb, "ce": ce, "drv": drv, "mg": mg, "total": total}).to_csv("composite_scores.csv", index=False)
print("\nSaved tb_scores.csv, ce_scores.csv, drv_scores.csv, mg_scores.csv, composite_scores.csv")
