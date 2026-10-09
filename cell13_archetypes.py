# ── Cell 13: Tumor burden score with 4 trajectory archetypes ──

import pandas as pd
import numpy as np

def tumor_burden_score(traj_df, patient_id=None,
                       weight_inc=2.0, weight_dec=0.5, weight_stable=1.0, weight_undetected=0.0,
                       method="vaf_weighted"):
    """
    Compute a per-patient composite tumor burden score from trajectory archetypes.

    Four archetypes are recognised:
      increasing  — last VAF > 1.5× first VAF  →  emerging clone, high concern
      decreasing  — last VAF < 0.5× first VAF  →  clearing clone, low concern
      stable      — otherwise                   →  persistent clone, moderate
      undetected  — never detected              →  no contribution

    Parameters
    ----------
    traj_df : pd.DataFrame
        Must have columns: patient_id, trajectory, first_VAF, last_VAF.
        (Same format as trajectory_per_patient_per_variant.csv)
    patient_id : str, optional
        Filter to one patient. If None, compute for all patients.
    weight_inc, weight_dec, weight_stable, weight_undetected : float
        Archetype weights applied to per-variant contributions.
    method : str
        'vaf_weighted'   = sum(last_VAF * weight)       — VAF-magnitude aware
        'count'          = n_variants * weight           — simple variant count
        'proportion'     = sum(last_VAF * weight) / sum(weight) — normalised

    Returns
    -------
    pd.DataFrame with columns: patient_id, trajectory_archetype, n_variants,
    contribution, composite_score
    """
    df = traj_df.copy()
    if patient_id is not None:
        df = df[df["patient_id"] == patient_id]

    # Archetype weights map
    weights = {
        "increasing":  weight_inc,
        "decreasing":  weight_dec,
        "stable":      weight_stable,
        "undetected":  weight_undetected,
    }

    results = []

    for pid, grp in df.groupby("patient_id"):
        for arch, w in weights.items():
            sub = grp[grp["trajectory"] == arch]
            n = len(sub)
            if n == 0:
                results.append({
                    "patient_id": pid,
                    "trajectory_archetype": arch,
                    "n_variants": 0,
                    "contribution": 0.0,
                })
                continue

            if method == "vaf_weighted":
                contrib = (sub["last_VAF"] * w).sum()
            elif method == "count":
                contrib = n * w
            elif method == "proportion":
                contrib = (sub["last_VAF"] * w).sum() / w
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append({
                "patient_id": pid,
                "trajectory_archetype": arch,
                "n_variants": n,
                "contribution": round(contrib, 4),
            })

    out = pd.DataFrame(results)

    # Per-patient composite score
    composite = out.groupby("patient_id")["contribution"].sum().reset_index()
    composite.columns = ["patient_id", "composite_score"]

    out = out.merge(composite, on="patient_id")
    out = out.sort_values(["patient_id", "trajectory_archetype"]).reset_index(drop=True)

    return out


# ── Apply ──

traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
print(f"Loaded {len(traj):,} variant-patient trajectories")

# Score all patients (default: vaf_weighted with inc=2, dec=0.5, stable=1, undet=0)
scores = tumor_burden_score(traj)

print(f"\nPer-archetype contributions and composite score:")
print(scores.to_string(index=False))

# Pivot for a cleaner patient view
pivot = scores.pivot_table(
    index="patient_id",
    columns="trajectory_archetype",
    values="contribution",
    aggfunc="sum",
).fillna(0)

pivot.columns = [f"contrib_{c}" for c in pivot.columns]
pivot["composite_score"] = pivot.sum(axis=1)
pivot = pivot.reset_index()

print(f"\n--- Patient summary (method=vaf_weighted) ---")
print(pivot.round(4).to_string(index=False))

# Compare methods
print("\n--- Composite score: vaf_weighted vs count vs proportion ---")
for method in ["vaf_weighted", "count", "proportion"]:
    tmp = tumor_burden_score(traj, method=method)
    comp = tmp.groupby("patient_id")["contribution"].sum().rename(f"score_{method}")
    if method == "vaf_weighted":
        all_scores = comp.to_frame()
    else:
        all_scores = all_scores.join(comp)

print(all_scores.round(4).to_string())
print("\nCorrelation between methods:")
print(all_scores.corr().round(3))

# Save
scores.to_csv("tumor_burden_archetypes.csv", index=False)
pivot.round(4).to_csv("tumor_burden_composite.csv", index=False)
print("\nSaved tumor_burden_archetypes.csv and tumor_burden_composite.csv")
