# ── Cell 14: classify_mutation_trend — per-mutation three-way helper ──

import pandas as pd
import numpy as np

def classify_mutation_trend(vafs, inc_threshold=1.5, dec_threshold=0.5):
    """
    Classify a single mutation's trajectory across timepoints.

    Parameters
    ----------
    vafs : pd.Series
        VAF values indexed by tp_order.  Holes (NaN or 0) represent
        timepoints where the mutation was not detected.
    inc_threshold : float
        Fold-increase threshold (default 1.5× first nonzero VAF).
    dec_threshold : float
        Fold-decrease threshold (default 0.5× first nonzero VAF).

    Returns
    -------
    label : str
        'increasing' | 'decreasing' | 'stable'
    detail : dict
        Metadata for diagnostics.
    """
    vals = vafs.dropna().values.copy()
    vals = vals[vals > 0]                     # nonzero detection events

    if len(vals) == 0:
        return "stable", {"n_detected": 0, "first_VAF": 0.0, "last_VAF": 0.0}

    first = vals[0]
    last  = vals[-1]

    if last > inc_threshold * first:
        label = "increasing"
    elif last < dec_threshold * first:
        label = "decreasing"
    else:
        label = "stable"

    detail = {
        "n_detected": len(vals),
        "first_VAF": round(first, 4),
        "last_VAF":  round(last, 4),
        "fold_change": round(last / max(first, 1e-6), 3),
    }
    return label, detail


# ── Quick demo on real examples ──

trusted = pd.read_csv("trusted_variants.csv")

# Pick one patient with multiple TPs and show a few mutations
pid = "LH210011"
pt = trusted[trusted["patient_id"] == pid]

# Get all unique variants in this patient
variants = pt["variant_id"].unique()

print(f"Patient {pid} — mutation trend classification demo\n")
print(f"{'Variant':55s} {'TPs':>4s}  {'Trajectory':12s}  {'first_VAF':>9s}  {'last_VAF':>9s}  {'fold':>6s}")
print("-" * 100)

for vid in variants[:10]:   # first 10
    sub = pt[pt["variant_id"] == vid]
    vaf_series = sub.groupby("tp_order")["VAF"].max()
    label, detail = classify_mutation_trend(vaf_series)
    tps = ",".join(str(t) for t in sorted(vaf_series.index))
    print(f"{vid:55s} {len(vaf_series):>4d}  {label:12s}  {detail['first_VAF']:>9.4f}  {detail['last_VAF']:>9.4f}  {detail['fold_change']:>6.2f}")
