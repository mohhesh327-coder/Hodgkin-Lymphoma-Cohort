# ── Cell 13: tumor_burden_score() function ──

import pandas as pd
import numpy as np

def tumor_burden_score(row, method="max", weights=None):
    """
    Compute a tumor burden score for a single (patient, timepoint).

    Parameters
    ----------
    row : pd.Series
        A row from the trusted_variants DataFrame subset for one (patient, tp).
    method : str
        'max'      -> max VAF (simple)
        'sum'      -> sum of all VAFs
        'sqrt_sum' -> sqrt(sum of VAF²)  — L2 norm
        'drivers'  -> sum of VAFs in driver genes only (is_driver=True)
        'weighted' -> sum of VAF * weight per gene (weights dict required)
    weights : dict or None
        Gene -> weight multiplier, used when method='weighted'.

    Returns
    -------
    float
    """
    if method == "max":
        return row["VAF"].max()
    elif method == "sum":
        return row["VAF"].sum()
    elif method == "sqrt_sum":
        return np.sqrt((row["VAF"] ** 2).sum())
    elif method == "drivers":
        return row.loc[row["is_driver"], "VAF"].sum()
    elif method == "weighted":
        if weights is None:
            raise ValueError("weights dict required for method='weighted'")
        return sum(row["VAF"].iloc[i] * weights.get(row["Genes"].iloc[i], 1)
                   for i in range(len(row)))
    else:
        raise ValueError(f"Unknown method: {method}")


# ── Quick test: compute scores for all patient-TPs ──

df = pd.read_csv("trusted_variants.csv")

results = []
for (pid, tp), grp in df.groupby(["patient_id", "tp_label"]):
    tp_order = grp["tp_order"].iloc[0]
    results.append({
        "patient_id": pid,
        "tp_label": tp,
        "tp_order": tp_order,
        "score_max": tumor_burden_score(grp, "max"),
        "score_sum": tumor_burden_score(grp, "sum"),
        "score_sqrt": tumor_burden_score(grp, "sqrt_sum"),
        "score_drivers": tumor_burden_score(grp, "drivers"),
        "n_variants": len(grp),
    })

scores = pd.DataFrame(results).sort_values(["patient_id", "tp_order"]).reset_index(drop=True)

print(f"Shape: {scores.shape}")
print(f"\nFirst 10 rows:")
print(scores.head(10).to_string(index=False))

# Compare methods
print("\n--- Score method correlations ---")
print(scores[["score_max", "score_sum", "score_sqrt", "score_drivers"]].corr().round(3))

scores.to_csv("tumor_burden_scores.csv", index=False)
print("\nSaved tumor_burden_scores.csv")
