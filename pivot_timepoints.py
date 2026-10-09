import pandas as pd
import numpy as np
import re, warnings
warnings.filterwarnings("ignore")

master = pd.read_csv("cleaned_variants.csv")
print(f"Loaded {len(master):,} rows, {master['patient_id'].nunique()} patients\n")

# ═══════════════════════════════════════════════════════════════
# Parse chronological order from timepoint names
# ═══════════════════════════════════════════════════════════════
def parse_timepoint_sort_key(tp_name):
    """Return a tuple (sort_priority, sort_date, sort_seq) for ordering timepoints."""
    name = str(tp_name).strip()

    # COMUN / COMMON sheets → skip (place at end)
    clean = name.lower().replace(" ", "").replace("\ufffd", "u").replace("�", "u")
    if "comun" in clean or "com" == clean[:3] or "sangre-com" in clean:
        return (99, "9999-99-99", 0)

    # DNA NORMAL → first priority
    if "dna normal" in name.lower() or "dna_normal" in name.lower():
        return (0, "0000-00-00", -1)

    # Try to extract date (ddmmyy or ddmmyyyy pattern like 040919, 161019, 240620, 110820)
    date_match = re.search(r'_(\d{6})_?', name)
    if date_match:
        d, m, y = date_match.group(1)[0:2], date_match.group(1)[2:4], date_match.group(1)[4:6]
        y_full = "20" + y
        date_str = f"{y_full}-{m}-{d}"
        # Extract sequence info from the rest
        seq = 0
        seq_match = re.search(r'S[._](\d+(?:[._]\d+)?)', name)
        if seq_match:
            parts = seq_match.group(1).replace("_", ".").split(".")
            seq = float(parts[0]) * 100 + (float(parts[1]) if len(parts) > 1 else 0)
        return (1, date_str, seq)

    # Try _ddmmyyyy pattern (e.g., BRCA...)
    date_match2 = re.search(r'_(\d{2})(\d{2})(\d{4})_', name)
    if date_match2:
        d, m, y = date_match2.group(1), date_match2.group(2), date_match2.group(3)
        date_str = f"{y}-{m}-{d}"
        return (1, date_str, 0)

    # Try dates in parentheses like "(S_1.5)"
    date_match3 = re.search(r'\(S[._](\d+(?:[._]\d+)?)\)', name)
    if date_match3:
        parts = date_match3.group(1).replace("_", ".").split(".")
        seq = float(parts[0]) * 100 + (float(parts[1]) if len(parts) > 1 else 0)
        return (2, "0000-00-00", seq)

    # S_X.X pattern (e.g., LH_808_S.0, LH_808_S1.1, LH_808_S1.2)
    seq_match = re.search(r'S[._](\d+(?:[._]\d+)?)', name)
    if seq_match:
        parts = seq_match.group(1).replace("_", ".").split(".")
        seq = float(parts[0]) * 100 + (float(parts[1]) if len(parts) > 1 else 0)
        return (2, "0000-00-00", seq)

    # Sample X_v1 pattern (BR samples)
    sample_match = re.search(r'Sample\s*(\d+)', name)
    if sample_match:
        return (3, "0000-00-00", int(sample_match.group(1)))

    # BR/REP pattern with trailing numbers (e.g., BR190010_2_02_1_v1, BR190010_2_02_2_v1)
    rep_match = re.search(r'_(\d+)_v?\d*$', name)
    if rep_match:
        return (4, "0000-00-00", int(rep_match.group(1)))

    # BIOPSIA / biopsy samples — check if there's a number
    biop_match = re.search(r'BIOPSIA\s*(\d+)', name, re.IGNORECASE)
    if biop_match:
        return (2, "0000-00-00", int(biop_match.group(1)) * 1000)

    # SANGRE (blood) samples
    sang_match = re.search(r'SANGRE\s*(\d+)', name, re.IGNORECASE)
    if sang_match:
        return (2, "0000-00-00", 1000 + int(sang_match.group(1)))

    # Fallback: by the name itself
    return (5, "0000-00-00", hash(name) % 10000)


# Apply ordering per patient
master["sort_tuple"] = master["timepoint"].apply(parse_timepoint_sort_key)
master["sort_priority"] = master["sort_tuple"].apply(lambda x: x[0])
master["sort_date"] = master["sort_tuple"].apply(lambda x: x[1])
master["sort_seq"] = master["sort_tuple"].apply(lambda x: x[2])

# Filter out COMUN/COMMON sheets (priority 99) — these are cross-sample summaries
master_clean = master[master["sort_priority"] < 99].copy()

# Within each patient, assign a sequential timepoint index
def assign_timepoint_order(grp):
    grp = grp.sort_values(["sort_priority", "sort_date", "sort_seq"])
    tp_order_map = {tp: i for i, tp in enumerate(grp["timepoint"].unique())}
    grp["tp_order"] = grp["timepoint"].map(tp_order_map)
    grp["tp_label"] = grp["timepoint"].map(tp_order_map).apply(lambda x: f"T{x}")
    return grp

print(f"master_clean columns before groupby: {master_clean.columns.tolist()}")
master_clean = (master_clean
    .groupby("patient_id", group_keys=False)
    .apply(assign_timepoint_order)
)

# group_keys=False strips the group key; re-add it from the original
orig_pid = master.loc[master_clean.index, "patient_id"] if "patient_id" not in master_clean.columns else None
if "patient_id" not in master_clean.columns:
    master_clean["patient_id"] = orig_pid.values if orig_pid is not None else None
master_clean = master_clean.reset_index(drop=True)

print("Chronological timepoint ordering applied.")
print(f"Filtered out COMUN/common sheets. Remaining: {len(master_clean):,} rows")
print(f"master_clean columns: {master_clean.columns.tolist()}")
print(f"Timepoints per patient:")
tp_count_clean = master_clean.groupby("patient_id")["tp_label"].nunique()
print(tp_count_clean.sort_values(ascending=False).to_string())

# Show the mapping for a few patients
print("\nSample timepoint mappings:")
for pid in ["LH200004", "C191819", "LH190001", "BR190003"]:
    sub = master_clean[master_clean["patient_id"] == pid][["timepoint", "tp_order", "tp_label"]].drop_duplicates().sort_values("tp_order")
    print(f"\n{pid}:")
    for _, r in sub.iterrows():
        print(f"  T{r['tp_order']} ({r['tp_label']}) -> {r['timepoint']}")

# ═══════════════════════════════════════════════════════════════
# Step 2 — Pivot the table
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: PIVOT TABLE")
print("="*70)

# Create a locus+gene identifier
master_clean["variant_id"] = master_clean["Locus"] + " | " + master_clean["Genes"]

# Pivot 1: VAF matrix — variant_id x (patient, timepoint) with VAF as values
pivot_vaf = master_clean.pivot_table(
    index="variant_id",
    columns=["patient_id", "tp_label"],
    values="VAF",
    aggfunc="max"  # if multiple entries per variant+timepoint, take max VAF
).fillna(0)

print(f"\nPivot (VAF) shape: {pivot_vaf.shape[0]} variants × {pivot_vaf.shape[1]} (patient,tp) columns")
print(pivot_vaf.iloc[:8, :10].to_string())

# Pivot 2: Presence/absence matrix (binary)
pivot_binary = (pivot_vaf > 0.05).astype(int)
print(f"\nBinary pivot (VAF>0.05): shape {pivot_binary.shape[0]} × {pivot_binary.shape[1]}")
print(pivot_binary.iloc[:8, :10].to_string())

# Pivot 3: By patient only — how many patients carry each variant
variant_patient_count = master_clean.groupby("variant_id")["patient_id"].nunique().sort_values(ascending=False)
print(f"\nTop 20 most prevalent variants (by patients affected):")
print(variant_patient_count.head(20).to_string())

# Save
pivot_vaf.to_csv("pivot_vaf.csv")
pivot_binary.to_csv("pivot_binary.csv")
master_clean.to_csv("master_chronological.csv", index=False)
print("\nSaved:")
print("  master_chronological.csv — full table with tp_order & tp_label")
print("  pivot_vaf.csv — VAF values (variant_id × [patient_tp])")
print("  pivot_binary.csv — binary presence (VAF>0.05)")
