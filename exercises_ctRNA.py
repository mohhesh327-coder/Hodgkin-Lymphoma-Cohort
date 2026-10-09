"""
Exercises 1-4: ctRNA panel design, sensitivity analysis, fusion simulation, CHIP
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cs = pd.read_csv("composite_scores.csv")

# =============================================================================
# EXERCISE 1 — 20-gene ctRNA target panel for HL monitoring
# =============================================================================
"""
Design rationale: ctRNA detects (a) transcript expression levels and
(b) fusion transcripts. Unlike ctDNA, it captures gene expression dysregulation
even in the absence of somatic mutations.

HL hallmark biology:
  - Constitutive NF-kB activation (REL amplification, TNFAIP3 inactivation)
  - JAK-STAT pathway activation (STAT3/6, SOCS1)
  - Immune evasion via 9p24.1 amplification (PD-L1/PD-L2)
  - CD30 overexpression (hallmark HRS cell marker)
  - Apoptosis resistance (CASP8, TP53, BCL2 family)

PANEL (20 genes):
"""
panel_data = {
    "Gene": [
        # --- Expression monitoring (12 genes) ---
        "CD274", "PDCD1LG2", "REL", "STAT3", "TNFRSF8", "MKI67",
        "BCL2", "MYC", "SOCS1", "TNFAIP3", "CASP8", "JAK2",
        # --- Fusion detection (5 genes) ---
        "CIITA", "NOTCH1", "BCL6", "ALK", "ROS1",
        # --- Both expression + fusion (3 genes) ---
        "STAT6", "CD30", "CREBBP",
    ],
    "Target": [
        "Expression", "Expression", "Expression", "Expression", "Expression", "Expression",
        "Expression", "Expression", "Expression", "Expression", "Expression", "Expression",
        "Fusion", "Fusion", "Fusion", "Fusion", "Fusion",
        "Both", "Both", "Both",
    ],
    "Rationale": [
        "PD-L1; 9p24.1 amp hallmark; immune checkpoint; expressed on HRS",
        "PD-L2; co-amplified with CD274 at 9p24.1; immune evasion",
        "NF-kB subunit; 2p16.1 amp in HL; constitutive pathway activation",
        "Master TF; constitutively phospho-activated in HL; JAK-STAT hub",
        "CD30; defining HRS surface marker; therapeutic target (brentuximab)",
        "Ki-67; proliferation index; correlates with aggressive disease",
        "Anti-apoptotic; overexpressed in HL; therapy resistance marker",
        "Master regulator; MYC rearrangements + overexpression in aggressive HL",
        "JAK-STAT negative regulator; LOF by mutation or silencing in HL",
        "NF-kB negative regulator; frequently inactivated in HL",
        "Caspase-8; apoptosis initiation; frequently mutated in HL",
        "JAK2; 9p24.1 amp target; activates STAT3/6; therapeutic target (ruxolitinib)",
        "MHC2 TF; recurrent fusions with CD274/PD-L2 in HL",
        "NOTCH pathway; fusions with MIR4673 observed in cohort",
        "B-cell TF; rearrangements with 3q27 partners in lymphoma",
        "Kinase; ALK+ ALCL fusions (NPM1-ALK); therapeutic target",
        "Kinase; ROS1 rearrangements in lymphoma; targetable",
        "JAK-STAT TF; IL-13/IL-4 signaling; fusions detected in lymphoma",
        ("CD30 ligand; both expression (HRS marker) and fusions "
         "(TNFRSF8 rearrangements in lymphoma)"),
        "Histone acetyltransferase; fusions with EP300 in lymphoma; expression in HL",
    ],
}

panel_df = pd.DataFrame(panel_data)
print("=" * 80)
print("EXERCISE 1: 20-gene ctRNA Panel for Hodgkin Lymphoma Monitoring")
print("=" * 80)
for _, row in panel_df.iterrows():
    print(f"  {row['Gene']:<8} [{row['Target']:<12}] {row['Rationale']}")

# Expression-specific features (what ctRNA captures that ctDNA misses)
print("""
Key ctRNA advantages over ctDNA for HL:
  1. 9p24.1 amplification (CD274/PD-L1): ctRNA detects overexpression even
     without focal amp detection (CNV is noisy in targeted ctDNA panels)
  2. CD30 expression: hallmark HRS marker; no DNA-level equivalent
  3. Proliferation index (MKI67): ctRNA captures real-time tumor activity
  4. Fusion transcripts: ctRNA detects chimeric RNA directly; ctDNA needs
     structural variant breakpoints which are hard in targeted panels
  5. NF-kB / JAK-STAT pathway activation: transcriptional signatures of
     pathway activation are richer than individual mutation patterns
""")

# =============================================================================
# EXERCISE 2 — Sensitivity rescue analysis for 9 low-score patients
# =============================================================================
print("=" * 80)
print("EXERCISE 2: ctRNA Rescue Potential for Low-Scoring Patients")
print("=" * 80)

rescue_table = []
# None patients (score < 10)
none_patients = [
    ("BR190003", 5, "Has 25 driver vars but 0 inc. Gains=7 suggest late-emerging clones. ctRNA could detect REL overexpression (NF-kB) and CD274/PD-L1 expression from 9p24.1 amp, revealing active disease that ctDNA mutations alone miss."),
    ("BR190004", 0, "24 driver vars, 6 gains, 0 inc. Pure stable/dec disease. ctRNA JAK-STAT activation signature (STAT3, SOCS1) could flag pathway dysregulation. PD-L1 expression monitoring would detect immune checkpoint signaling."),
    ("BR190007", 5, "22 driver vars, 6 gains, 0 inc. Similar profile to BR190004. ctRNA MKI67 proliferation index would capture active tumor growth dynamics. CD30 expression monitoring confirms HRS cell presence."),
    ("BR190015", 5, "24 driver vars, 8 gains, 0 inc. Gains suggest subclonal sweeps. ctRNA fusion scanning for CIITA-CD274/PD-L2 fusions (recurrent in HL) would add independent signal. NF-kB target gene signature."),
    ("BR190017", 5, "38 driver vars, 14 gains, 2 TPs. Highest driver count among None patients. ctRNA expression signature of NF-kB (REL targets) and JAK-STAT (STAT6 targets) would confirm tumor activity despite absence of inc mutations."),
    ("BR190020", 0, "20 driver vars, 4 gains, 0 inc. Low overall signal. ctRNA could rescue via CD274/PD-L2 fusion detection + MKI67 expression. If no ctRNA signal either, likely true negative (no active disease)."),
    ("BR200026", 0, "24 driver vars, 7 gains, 0 inc. ctRNA pathway activation signature for NF-kB and JAK-STAT. CIITA fusion scanning. Proliferation index via MKI67."),
    ("BR200030", 0, "21 driver vars, 4 gains, 0 inc. BCL10 multi-gene reads suggest possible NF-kB fusion. ctRNA targeted fusion panel for BCL10-IGH (known in MALT lymphoma)."),
    ("BR200032", 5, "19 driver vars, 6 gains, 0 inc. ctRNA CD30 and PD-L1 expression levels. SOCS1 silencing detection (JAK-STAT activation proxy)."),
    ("BR210038", 5, "21 driver vars, 6 gains, 0 inc. MIR4673,NOTCH1 multi-gene read present — possible NOTCH1 fusion candidate. ctRNA NOTCH1 fusion transcript detection would directly validate."),
    ("LH230118", 5, "37 driver vars, 19 gains, 3 TPs. Highest gains among all None. ctRNA could detect which gained genes are transcriptionally active. Pathway activation signature (NF-kB, JAK-STAT) from expressed mutant alleles."),
]
# Weak patients (score 10-29)
weak_patients = [
    ("BR190010", 25, "1 inc (ce=8) giving weak signal. 31 driver vars suggest tumor present but clonal expansion below detection. ctRNA NF-kB target gene signature + JAK-STAT activation would amplify the weak DNA signal."),
    ("BR190011", 28, "1 inc variant. Has EP300 multi-gene reads — possible EP300-CREBBP fusion. ctRNA fusion detection could independently confirm tumor activity. PD-L1 expression monitoring."),
    ("BR190014", 10, "0 inc but mg=10 (gains). ctRNA expression signature is the main rescue path. CD30 + MKI67 expression could transform from Weak to Strong recurrence tier."),
    ("BR200021", 24, "144 driver vars, 311 total vars — massive tumor burden but only 1 inc. ctRNA NF-kB pathway activation signature (REL targets TNF, NFKBIA, etc.) would strongly elevate score. MIR4673,NOTCH1 reads suggest NOTCH1 fusion."),
    ("BR200023", 11, "21 driver vars, 1 inc, 0 mg. Low overall. ctRNA CD274 expression + JAK-STAT signature could provide signal. Without either, likely quiescent disease."),
    ("BR200027", 19, "20 driver vars, 1 inc, mg=5. ctRNA apoptosis signature (CASP8, BCL2 expression) could add dimension. NF-kB target gene panel."),
    ("C222201", 10, "84 driver vars, 505 total, 9 TPs — extensive monitoring but 0 inc. BCL10 and REL multi-gene reads suggest NF-kB fusions. ctRNA would be transformative here: fusion detection (BCL10, REL, CIITA) + expression profiling across all 9 TPs."),
    ("LH210012", 10, "42 driver vars, 5 TPs, 0 inc, mg=10. Extensive gains. ctRNA can determine which gained genes are expressed. JAK-STAT and NF-kB pathway activation profiles."),
]

all_low = none_patients + weak_patients
print(f"\n{'Patient':<10} {'Score':>5} {'Category':<6} {'ctRNA Rescue Hypothesis':<70}")
print("-" * 95)
for pid, score, hyp in all_low:
    cat = "None" if score < 10 else "Weak"
    print(f"{pid:<10} {score:>5} {cat:<6} {hyp:<70}")

# =============================================================================
# EXERCISE 3 — Simulated fusion data + integration
# =============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Simulated Fusion Dataset + Scoring Integration")
print("=" * 80)

np.random.seed(42)

# Pick 10 patients: mix of high and low scorers
selected = ["LH200003", "LH210010", "BR200029", "C212118", "BR190005",
            "BR200021", "BR190010", "C222201", "BR190004", "LH230118"]
print(f"Selected patients: {selected}")

# Simulated fusions based on HL biology + multi-gene read evidence
fusion_catalog = {
    "CIITA-CD274":  {"genes": ("CIITA", "CD274"),  "relevance": "HL hallmark fusion (9p24.1 / MHC2 TF)", "max_score": 15},
    "NPM1-ALK":     {"genes": ("NPM1", "ALK"),     "relevance": "ALCL fusion, ALK inhibitor target", "max_score": 15},
    "BCL6-IGH":     {"genes": ("BCL6", "IGH"),     "relevance": "Lymphoma B-cell TF rearrangement", "max_score": 12},
    "NOTCH1-MIR4673":{"genes": ("NOTCH1", "MIR4673"),"relevance": "Observed in cohort multi-gene reads", "max_score": 10},
    "REL-LINC01185": {"genes": ("REL", "LINC01185"),"relevance": "Observed in cohort (NF-kB activation)", "max_score": 10},
    "EP300-CREBBP":  {"genes": ("EP300", "CREBBP"), "relevance": "Histone acetyltransferase fusion", "max_score": 8},
    "BCL10-IGH":     {"genes": ("BCL10", "IGH"),    "relevance": "NF-kB activating fusion", "max_score": 12},
    "FAS-FAS-AS1":   {"genes": ("FAS", "FAS-AS1"),  "relevance": "Observed in cohort (apoptosis)", "max_score": 6},
    "CSF1R-HMGXB3":  {"genes": ("CSF1R", "HMGXB3"), "relevance": "Observed in cohort (likely artifact)", "max_score": 3},
    "NUMA1-IL18BP":  {"genes": ("NUMA1", "IL18BP"), "relevance": "Observed in cohort", "max_score": 5},
}

# Patient-fusion assignment with realistic junction/spanning reads
fusion_data = []
fusion_assignments = {
    "LH200003": [("CIITA-CD274", 45, 120), ("NPM1-ALK", 12, 35)],
    "LH210010": [("NOTCH1-MIR4673", 28, 80), ("REL-LINC01185", 15, 42)],
    "BR200029": [("CIITA-CD274", 32, 95)],
    "C212118":  [("BCL6-IGH", 18, 55), ("EP300-CREBBP", 8, 22)],
    "BR190005": [("FAS-FAS-AS1", 22, 68)],
    "BR200021": [("NOTCH1-MIR4673", 35, 102), ("BCL10-IGH", 14, 38)],
    "BR190010": [("CIITA-CD274", 6, 18)],
    "C222201":  [("BCL10-IGH", 25, 72), ("REL-LINC01185", 20, 58), ("EP300-CREBBP", 10, 30)],
    "BR190004": [],
    "LH230118": [("NUMA1-IL18BP", 8, 25)],
}

rows = []
for pid, fusions in fusion_assignments.items():
    for fname, junc, span in fusions:
        rows.append({"Patient": pid, "Fusion_Name": fname,
                      "Junction_Reads": junc, "Spanning_Frags": span,
                      "Genes_5p": fusion_catalog[fname]["genes"][0],
                      "Genes_3p": fusion_catalog[fname]["genes"][1],
                      "Relevance": fusion_catalog[fname]["relevance"]})

fusion_df = pd.DataFrame(rows)
fusion_df.to_csv("simulated_fusions.csv", index=False)
print(f"\nSimulated fusion data: {len(fusion_df)} rows across {fusion_df['Patient'].nunique()} patients")
print(fusion_df.to_string(index=False))


def fusion_score(pid, fusion_data, catalog):
    """Score a patient based on detected fusions (range 0-15)."""
    patient_fusions = fusion_data[fusion_data["Patient"] == pid]
    if len(patient_fusions) == 0:
        return 0.0

    total = 0.0
    for _, f in patient_fusions.iterrows():
        fname = f["Fusion_Name"]
        junc = f["Junction_Reads"]
        span = f["Spanning_Frags"]
        max_s = catalog[fname]["max_score"]

        # Score based on evidence strength
        evidence = min(1.0, (junc * 2 + span) / 100)
        # Weight by biological relevance
        bio_weight = max_s / 15.0
        score = evidence * max_s

        # Bonus for high confidence
        if junc >= 20 and span >= 50:
            score = min(score * 1.3, max_s)
        elif junc >= 5 and span >= 15:
            score = min(score * 1.1, max_s)

        total += score

    # Cap at 15 and combine: multiple fusions add less than linearly
    if len(patient_fusions) >= 3:
        total = min(total * 0.7, 15)
    elif len(patient_fusions) == 2:
        total = min(total * 0.85, 15)
    else:
        total = min(total, 15)

    return round(total, 1)


# Compute fusion scores
fusion_scores = {p: fusion_score(p, fusion_df, fusion_catalog) for p in selected}
for p in selected:
    print(f"  {p}: fusion_score = {fusion_scores[p]}")

# Integrate into composite pipeline
cs_subset = cs[cs["patient"].isin(selected)].copy()
cs_subset["fu"] = cs_subset["patient"].map(fusion_scores).fillna(0)
cs_subset["new_total"] = cs_subset["total"] + cs_subset["fu"]

print(f"\n{'Patient':<10} {'Old':>5} {'fu':>5} {'New':>5} {'Rank':>5}")
print("-" * 35)
cs_sorted_old = cs_subset.sort_values("total", ascending=False)
cs_sorted_new = cs_subset.sort_values("new_total", ascending=False)
for _, row in cs_sorted_new.iterrows():
    old_rank = list(cs_sorted_old["patient"]).index(row["patient"]) + 1
    new_rank = list(cs_sorted_new["patient"]).index(row["patient"]) + 1
    delta = "⬆" if new_rank < old_rank else ("⬇" if new_rank > old_rank else "=")
    print(f"{row['patient']:<10} {row['total']:>5.0f} {row['fu']:>5.1f} {row['new_total']:>5.1f} {new_rank:>3}/{old_rank} {delta}")

# Ranking correlation
from scipy.stats import spearmanr
rho, pval = spearmanr(cs_subset["total"], cs_subset["new_total"])
print(f"\nSpearman rho (old vs new): {rho:.3f} (p={pval:.4f})")

# =============================================================================
# EXERCISE 4 — CHIP vs fusion specificity
# =============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Why Fusion Transcripts Are Immune to CHIP")
print("=" * 80)

print("""
CHIP Mutations:
  - Clonal Hematopoiesis of Indeterminate Potential
  - Somatic mutations in hematopoietic stem cells (DNMT3A, TET2, ASXL1, TP53)
  - These mutations appear in ctDNA as cell-free DNA from blood cells
  - They create FALSE POSITIVES in ctDNA tumor mutation detection
  - CHIP mutations are at the DNA level in the blood cell genome

Why Fusion TRANSCRIPTS are CHIP-Immune:
  - CHIP mutations exist in the DNA of blood cells but do NOT generate
    fusion transcripts (they are point mutations/indels, not rearrangements)
  - Fusion transcripts arise from chromosomal rearrangements that fuse two
    genes into a chimeric transcription unit
  - Hematopoietic stem cells with CHIP mutations do NOT harbor the lymphoma-
    specific chromosomal rearrangements (e.g., CIITA-CD274, REL rearrangements)
  - ctRNA specifically captures RNA transcripts; fusion transcripts detected
    in plasma must originate from tumor cells expressing the chimeric RNA

Thus:
    ctRNA fusion detection = HIGH SPECIFICITY for tumor
    ctDNA mutation detection = LOWER SPECIFICITY (CHIP contamination)

One Scenario Where a Fusion Transcript Produces a False Positive:

    Scenario: Clonal hematopoiesis with a rare but recurrent chromosomal
              rearrangement in hematopoietic cells.

    Example: A patient has a t(14;18)(q32;q21) IGH-BCL2 rearrangement in
             a small population of peripheral blood B-cells (benign
             B-cell clonality or "in situ lymphoma" — not yet malignant).

    Why this is a false positive:
      - The IGH-BCL2 fusion transcript can be detected in ctRNA
      - This rearrangement is a hallmark of follicular lymphoma but can
        also be present at low levels in healthy individuals' B-cells
        (circulating t(14;18)+ cells increase with age)
      - The ctRNA assay cannot distinguish between:
          a) Malignant lymphoma cells shedding the fusion transcript
          b) Benign clonal B-cells with the same rearrangement
      - Result: False positive fusion detection leading to overestimation
        of tumor burden or incorrect diagnosis of lymphoma recurrence

    Mitigation:
      - Require VAF/fragment evidence: fusion junction reads should be
        present at RNA level with >10 reads and correlate with ctDNA
        mutation VAFs
      - Background correction: sequence age-matched healthy controls to
        establish population baseline for common IGH rearrangements
      - Longitudinal dynamics: true tumor fusions increase with progression;
        benign B-cell clones remain stable or fluctuate slowly

Other edge cases for fusion false positives:
  1. Trans-splicing artifacts: RNA polymerase can join exons from different
     genes without genomic rearrangement (rare but documented)
  2. In vitro fusion during library prep: PCR-mediated recombination of
     homologous sequences creates artificial chimeric reads
  3. Pseudo-gene reads: reads mapping to processed pseudogenes can appear
     as intergenic fusions
  4. Normal tissue-specific fusions: Some fusions are present in normal
     tissues (e.g., BCR-ABL1-like fusions in normal lymphocytes)
""")

print("=" * 80)
print("Exercises 1-4 complete. See output above.")
print("=" * 80)
