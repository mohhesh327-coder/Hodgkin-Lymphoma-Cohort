"""
Exercises 1-4: Clinical reports (DOCX), SOP, patient summary, validation study.
"""
import pandas as pd, numpy as np, os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

IMG_DIR = r"D:\GSE244989 Hodgkin Lymphoma Cohort"

# =============================================================================
# EXERCISE 1 — Three clinical reports as Word documents
# =============================================================================
print("=== Exercise 1: Generating clinical reports (DOCX) ===")

def make_clinical_report(pid, tier_name, score, components, output_path):
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title
    h = doc.add_heading("Clinical ctDNA Report — Research Use Only", level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    # Patient header table
    tbl = doc.add_table(rows=2, cols=4)
    tbl.style = "Light Grid Accent 1"
    cells = [
        ("Patient ID", pid),
        ("Tier", tier_name),
        ("Total Score", f"{score}/100"),
        ("Date", "2026-07-29"),
    ]
    for i, (k, v) in enumerate(cells):
        tbl.rows[0].cells[i].text = k
        tbl.rows[1].cells[i].text = v
        for p in tbl.rows[1].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True

    doc.add_heading("Score Breakdown", level=2)
    p = doc.add_paragraph()
    comp_names = {"tb": "Tumor Burden (0-40)", "ce": "Clonal Expansion (0-25)",
                  "drv": "Driver Inc. (0-25)", "mg": "Mutation Gain (0-10)"}
    for k, label in comp_names.items():
        p.add_run(f"{label}: {components[k]:.1f}  ").bold = True
    p.add_run(f"\nTotal: {score}/100")

    # Component interpretation
    doc.add_heading("Component Interpretation", level=2)
    interps = {
        "tb": ["Low", "Moderate", "High"][
            0 if components["tb"] < 10 else (1 if components["tb"] < 25 else 2)],
        "ce": ["None", "Low", "Moderate", "High"][
            0 if components["ce"] == 0 else (1 if components["ce"] <= 8 else (2 if components["ce"] <= 17 else 3))],
        "drv": ["None", "Low", "Moderate", "High"][
            0 if components["drv"] == 0 else (1 if components["drv"] <= 10 else (2 if components["drv"] <= 20 else 3))],
        "mg": ["None", "Low", "High"][
            0 if components["mg"] == 0 else (1 if components["mg"] <= 5 else 2)],
    }
    for k, label in comp_names.items():
        doc.add_paragraph(f"{label}: {interps[k]}", style="List Bullet")

    # Embed PNG report figure
    png_path = os.path.join(IMG_DIR, f"report_{pid}.png")
    if os.path.exists(png_path):
        doc.add_heading("Visual Summary", level=2)
        doc.add_picture(png_path, width=Inches(6.5))

    # Top 5 mutations
    doc.add_heading("Top 5 Variants by VAF (Latest Timepoint)", level=2)
    # (data embedded as text since PNG already has the table)
    doc.add_paragraph("See visual summary above for variant details.")

    # Limitations
    doc.add_heading("Limitations", level=2)
    doc.add_paragraph(
        "1. CHIP (Clonal Hematopoiesis): Mutations in DNMT3A, TET2, ASXL1, TP53 "
        "from blood cells can produce false-positive ctDNA signals.\n"
        "2. Panel coverage: 46-gene targeted AmpliSeq panel; novel drivers outside "
        "these regions are missed.\n"
        "3. Sensitivity: Gene-specific LoD filters may discard true subclonal mutations.\n"
        "4. Research use only: This score is not clinically validated.\n"
        f"5. Timepoints: {components.get('_ntp', 2)} available; trajectory confidence "
        "improves with more timepoints."
    )

    # Recommendations
    doc.add_heading("Recommendations", level=2)
    if score >= 80:
        rec = (
            "Strong molecular evidence of active disease. Recommended: "
            "(a) PET-CT correlation; (b) Consider therapy adjustment if clinically indicated; "
            "(c) Repeat ctDNA at next cycle; (d) Biopsy confirmation of actionable mutations "
            "before targeted therapy."
        )
    elif score >= 40:
        rec = (
            "Ambiguous molecular signal. Recommended: "
            "(a) Short-interval repeat ctDNA (4-6 weeks); (b) Clinical correlation; "
            "(c) Consider ctRNA complementary assay; (d) Review decreasing variant clearance patterns."
        )
    else:
        rec = (
            "No significant ctDNA evidence of active disease. IMPORTANT: Does NOT rule out "
            "residual disease. May reflect: (a) Disease below detection threshold; "
            "(b) Low-shedding tumor; (c) Effective treatment response. "
            "Recommended: (a) Maintain clinical surveillance; (b) Continue ctDNA monitoring; "
            "(c) Consider ctRNA assay if high clinical suspicion."
        )
    doc.add_paragraph(rec)

    doc.save(output_path)
    print(f"  Saved {output_path}")

# Generate reports for all 3 patients
cs = pd.read_csv(os.path.join(IMG_DIR, "composite_scores.csv"))
patients_report = [
    ("LH210010", "High-Confidence Recurrence"),
    ("LH200002", "Possible Recurrence"),
    ("BR200023", "No Recurrence Signal"),
]
for pid, tier in patients_report:
    row = cs[cs["patient"]==pid].iloc[0]
    comps = {"tb": row["tb"], "ce": row["ce"], "drv": row["drv"], "mg": row["mg"], "_ntp": 3}
    make_clinical_report(pid, tier, row["total"], comps,
                         os.path.join(IMG_DIR, f"report_{pid}.docx"))

# =============================================================================
# EXERCISE 2 — Incidental-Finding SOP
# =============================================================================
print("\n=== Exercise 2: Incidental-Finding SOP ===")

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

h = doc.add_heading("Standard Operating Procedure", level=1)
h.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("SOP ID: CTDNA-IF-001 | Version 1.0 | Effective: 2026-07-29", level=3)
doc.add_heading("Title: Management of Incidental Findings in ctDNA Analysis Pipeline", level=2)

doc.add_heading("1. Purpose", level=2)
doc.add_paragraph(
    "This SOP defines the procedure for identifying, reviewing, confirming, and "
    "communicating incidental findings discovered during the research ctDNA analysis "
    "pipeline. Incidental findings are clinically actionable variants detected in "
    "germline genes that are unrelated to the primary oncologic indication."
)

doc.add_heading("2. Scope", level=2)
doc.add_paragraph(
    "Applies to all ctDNA samples processed through the GSE244989 research pipeline. "
    "Covers variants identified in the 46-gene AmpliSeq HL panel that meet "
    "incidental-finding criteria. Does NOT cover known AML/MDS-associated clonal "
    "hematopoiesis variants (DNMT3A, TET2, ASXL1) which are expected in this population."
)

doc.add_heading("3. Genes Triggering the Protocol (ACMG v3.0 List)", level=2)
doc.add_paragraph(
    "The following genes in the panel overlap with ACMG-recommended secondary "
    "findings genes. Any pathogenic or likely pathogenic variant in these genes "
    "triggers the protocol:"
)
acmg_genes = [
    ("TP53", "Li-Fraumeni syndrome", "Tumor suppressor; any exonic pathogenic variant"),
    ("MLH1", "Lynch syndrome", "MMR gene; frameshift/nonsense/splice"),
    ("MSH2", "Lynch syndrome", "MMR gene; frameshift/nonsense/splice"),
    ("MSH6", "Lynch syndrome", "MMR gene; frameshift/nonsense/splice"),
    ("PMS2", "Lynch syndrome", "MMR gene; frameshift/nonsense/splice"),
    ("BRCA1", "HBOC syndrome", "Frameshift/nonsense/splice"),
    ("BRCA2", "HBOC syndrome", "Frameshift/nonsense/splice"),
    ("RET", "MEN2", "Missense (specific hotspots: M918T, C634R)"),
    ("VHL", "von Hippel-Lindau", "Any pathogenic missense/truncating"),
]
tbl = doc.add_table(rows=1+len(acmg_genes), cols=3)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, txt in enumerate(["Gene", "Associated Syndrome", "Trigger Criteria"]):
    hdr[i].text = txt
    for p in hdr[i].paragraphs:
        for r in p.runs:
            r.bold = True
for j, (gene, syn, crit) in enumerate(acmg_genes):
    tbl.rows[j+1].cells[0].text = gene
    tbl.rows[j+1].cells[1].text = syn
    tbl.rows[j+1].cells[2].text = crit

doc.add_heading("4. Notification Workflow", level=2)
doc.add_paragraph(
    "Step 1 — Detection: Pipeline flags any variant meeting trigger criteria with "
    "VAF > 0.05 and PASS filter. Variants are reviewed by the bioinformatics lead.\n"
    "Step 2 — Review: Two independent molecular pathologists review the flagged "
    "variant. Discordant calls are adjudicated by a third reviewer.\n"
    "Step 3 — Confirmation: Orthogonal testing via Sanger sequencing or orthogonal "
    "NGS assay on a fresh peripheral blood sample (not ctDNA). Germline origin is "
    "confirmed by testing a buccal swab or skin biopsy.\n"
    "Step 4 — Notification: Confirmed pathogenic germline variants are communicated "
    "to the patient's primary oncologist within 5 business days. The report includes "
    "the gene, variant, associated syndrome, and recommendation for genetic counseling.\n"
    "Step 5 — Documentation: All findings are recorded in the incidental-finding "
    "registry with IRB acknowledgment. The patient's study consent form must address "
    "incidental findings disclosure preferences."
)

doc.add_heading("5. Confirmatory Testing Requirements", level=2)
doc.add_paragraph(
    "1. Fresh sample collection: 5 mL peripheral blood in EDTA tube.\n"
    "2. DNA extraction from buffy coat (not plasma).\n"
    "3. Orthogonal sequencing: Sanger sequencing of the variant locus, or "
    "targeted NGS using a CLIA-certified germline panel.\n"
    "4. Variant classification per ACMG/AMP 2015 guidelines (PVS1-PVS4, PS1-PS4, "
    "PM1-PM6, PP1-PP5).\n"
    "5. Only variants classified as Pathogenic (Class 5) or Likely Pathogenic (Class 4) "
    "after orthogonal confirmation are reported.\n"
    "6. Turnaround time: 21 business days from initial detection to final report."
)

doc.add_heading("6. Communication to Patient", level=2)
doc.add_paragraph(
    "1. Initial notification: The primary oncologist informs the patient in person "
    "or via telehealth. A genetic counselor is present or scheduled for follow-up.\n"
    "2. Language: The patient-facing summary uses plain language (grade 6 reading level) "
    "and avoids variant-level technical details. Focus is on the syndrome, "
    "recurrence risks, and available preventive measures.\n"
    "3. Opt-out: Patients who previously elected not to receive incidental findings "
    "are excluded from notification. Their data is anonymized in the registry.\n"
    "4. Family implications: The patient is informed about cascade testing for "
    "first-degree relatives. Family member testing is arranged through clinical genetics."
)

doc.add_heading("7. Quality Control", level=2)
doc.add_paragraph(
    "1. Annual audit of all incidental-finding cases.\n"
    "2. Turnaround time metrics tracked quarterly.\n"
    "3. False-positive rate monitored — target <5%.\n"
    "4. SOP reviewed and updated annually or upon addition of new genes to the panel."
)

doc.add_heading("8. References", level=2)
doc.add_paragraph(
    "1. ACMG v3.0: Miller et al. (2023) Genet Med.\n"
    "2. Richards et al. (2015) ACMG/AMP variant classification guidelines.\n"
    "3. GSE244989 study protocol IRB-approved consent form."
)

doc.save(os.path.join(IMG_DIR, "sop_incidental_findings.docx"))
print("  Saved sop_incidental_findings.docx")

# =============================================================================
# EXERCISE 3 — Patient-facing summary (BR190005_OK style)
# =============================================================================
print("\n=== Exercise 3: Patient-facing summary ===")

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(12)

# Use LH210010 data as a high-scorer
cs_data = pd.read_csv(os.path.join(IMG_DIR, "composite_scores.csv"))
pid_high = "LH210010"
row_high = cs_data[cs_data["patient"]==pid_high].iloc[0]

h = doc.add_heading("Your Liquid Biopsy Results — A Summary for You", level=1)
h.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in h.runs:
    run.font.color.rgb = RGBColor(0x27, 0x60, 0x60)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Patient ID: ---  |  Date: July 29, 2026")
run.font.size = Pt(9)
run.font.color.rgb = RGBColor(0x7F, 0x8C, 0x8D)

doc.add_paragraph(
    "Dear Patient,\n\n"
    "We recently analyzed your blood sample using a test that looks for tiny "
    "fragments of DNA released by cancer cells into your bloodstream. This is "
    "called a liquid biopsy. Here is what we found, explained in plain language."
)

doc.add_heading("What the test looks for", level=2)
doc.add_paragraph(
    "Cancer cells contain changes in their DNA that make them different from "
    "healthy cells. Our test detects these changes in your blood. Think of it "
    "like finding footprints in the sand — the footprints tell us that someone "
    "has been there, even if we cannot see them directly."
)

doc.add_heading("What we found", level=2)
doc.add_paragraph(
    "In your sample, we detected a strong signal from the cancer. This means "
    "the test found many DNA footprints from the cancer cells in your blood. "
    "We also noticed that some of these footprints are becoming more numerous "
    "over time, which means the cancer is likely still active.\n\n"
    "We identified specific changes in several genes that are known to play a "
    "role in Hodgkin lymphoma. These findings help us understand your cancer's "
    "unique fingerprint. Some of these changes may also point toward treatment "
    "options that could work well for you."
)

doc.add_heading("What this means for you", level=2)
doc.add_paragraph(
    "A strong signal on this test does not always mean the cancer is growing "
    "rapidly, but it does suggest the cancer is present and active. Your doctor "
    "will use this information together with your scans, symptoms, and other "
    "tests to decide on the best next steps.\n\n"
    "Your doctor may recommend:\n"
    "• A PET-CT scan to see where the cancer is in your body\n"
    "• A follow-up liquid biopsy in a few weeks or months to track changes\n"
    "• A discussion about treatment options based on the gene changes we found\n\n"
    "This test is still considered 'research use only,' which means it is not "
    "yet approved by the FDA for routine care. Your doctor will explain what "
    "the results mean and help you make informed decisions."
)

doc.add_heading("Important things to know", level=2)
doc.add_paragraph(
    "1. This test is very sensitive — it can detect tiny amounts of cancer DNA, "
    "but it can also sometimes pick up DNA from healthy blood cells. This is "
    "called CHIP (Clonal Hematopoiesis). Your doctor knows about this and will "
    "interpret the results carefully.\n\n"
    "2. A negative result (no signal) does not mean you are cancer-free. "
    "Some cancers do not release much DNA into the blood. This is why your "
    "doctor will use multiple tests to track your health.\n\n"
    "3. If you have any questions about these results, please ask your doctor "
    "or a genetic counselor. We are here to help you understand."
)

doc.add_paragraph("\n— The GSE244989 Research Team")
doc.add_paragraph("If you have questions, please contact your study coordinator.")

doc.save(os.path.join(IMG_DIR, "patient_summary_BR190005_OK.docx"))
print("  Saved patient_summary_BR190005_OK.docx")

# =============================================================================
# EXERCISE 4 — Validation Study Design
# =============================================================================
print("\n=== Exercise 4: Validation study synopsis ===")

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

h = doc.add_heading("Clinical Validation Study Synopsis", level=1)
h.alignment = WD_ALIGN_PARAGRAPH.CENTER
for run in h.runs:
    run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

doc.add_heading("Title", level=2)
doc.add_paragraph(
    "Validation of a 100-Point Recurrence Score for Circulating Tumor DNA "
    "Monitoring in Classical Hodgkin Lymphoma: A Multicenter Retrospective Study"
)

doc.add_heading("Study Identifier", level=2)
doc.add_paragraph("CTDNA-HL-VALID-001 | Version 1.0 | 2026-07-29")

doc.add_heading("Background and Rationale", level=2)
doc.add_paragraph(
    "The GSE244989 discovery cohort (n=46) established a composable recurrence "
    "scoring system (TB/CE/DRV/MG; range 0-100) for ctDNA-based monitoring of "
    "classical Hodgkin lymphoma. Before clinical adoption, the score requires "
    "independent validation for its ability to predict progression-free survival "
    "(PFS) and discriminate between recurrence risk tiers."
)

doc.add_heading("Study Design", level=2)
doc.add_paragraph(
    "Multicenter, retrospective, blinded validation study. Three independent sites "
    "will contribute archived ctDNA samples and linked clinical outcome data. "
    "The central lab will compute recurrence scores blinded to clinical outcomes. "
    "Statistical analysis will be performed by an independent biostatistics core."
)

doc.add_heading("Sample Size and Power Calculation", level=2)
doc.add_paragraph(
    "Primary endpoint: 2-year PFS. Expected event rate: 30% (based on historical "
    "HL data showing ~70% 2-year PFS in relapsed/refractory populations).\n\n"
    "Assumptions:\n"
    "  - Cox proportional hazards model with recurrence score as continuous predictor\n"
    "  - Expected hazard ratio per 10-point increase: 1.5 (based on discovery cohort "
    "effect size)\n"
    "  - Type I error: alpha=0.05 (two-sided)\n"
    "  - Power: 80%\n"
    "  - Correlation between score and PFS: estimated rho=0.4\n\n"
    "Using the Hsieh & Lavori (1998) formula for survival analysis:\n"
    "  n = (Z_alpha/2 + Z_beta)^2 / (rho^2 * log(HR)^2 * (1 - event_rate))\n"
    "  n = (1.96 + 0.84)^2 / (0.16 * 0.405^2 * 0.7)\n"
    "  n ≈ 215 patients\n\n"
    "Adding 15% for technical failures (insufficient ctDNA, failed QC):\n"
    "  Final target: 250 patients"
)

doc.add_heading("Cohort Selection Criteria", level=2)
doc.add_paragraph(
    "Inclusion:\n"
    "  - Age ≥18 years\n"
    "  - Histologically confirmed classical Hodgkin lymphoma (all subtypes)\n"
    "  - At least two plasma samples collected ≥4 weeks apart during routine care\n"
    "  - Minimum 12 months of clinical follow-up from first ctDNA sample\n"
    "  - Available clinical outcome data (PFS, OS, treatment received)\n\n"
    "Exclusion:\n"
    "  - Concurrent malignancy (except non-melanoma skin cancer)\n"
    "  - Known HIV infection (affects ctDNA interpretation)\n"
    "  - Allogeneic stem cell transplant within 6 months of first ctDNA sample\n"
    "  - Insufficient plasma (< 2 mL per timepoint)"
)

doc.add_heading("Primary Endpoint", level=2)
doc.add_paragraph(
    "2-year progression-free survival. Progression defined by Lugano 2014 criteria "
    "(PET-CT or CT-based). Death from any cause is an event."
)

doc.add_heading("Secondary Endpoints", level=2)
doc.add_paragraph(
    "1. Overall survival (OS) at 2 and 5 years\n"
    "2. Time to next treatment (TTNT)\n"
    "3. Sensitivity and specificity of score ≥80 vs <80 for 1-year PFS\n"
    "4. Area under the ROC curve (AUC) for recurrence score vs clinical stage\n"
    "5. Positive predictive value (PPV) and negative predictive value (NPV) at "
    "optimal cutpoint determined by Youden index"
)

doc.add_heading("Statistical Analysis Plan", level=2)
doc.add_paragraph(
    "1. ROC Analysis: Compute AUC for recurrence score as predictor of 2-year PFS. "
    "Compare to AUC for clinical stage (Ann Arbor) using DeLong test.\n\n"
    "2. Cutpoint Optimization: Determine optimal recurrence score threshold using "
    "Youden index (maximizing sensitivity + specificity - 1) on the ROC curve. "
    "Validate threshold via 10-fold cross-validation.\n\n"
    "3. Kaplan-Meier Analysis: Stratify patients into three risk groups (Low: <40, "
    "Intermediate: 40-79, High: ≥80). Compare PFS curves using log-rank test. "
    "Report median PFS and 95% CI per group.\n\n"
    "4. Cox Proportional Hazards: Univariate and multivariate analysis with "
    "recurrence score as continuous variable. Covariates: age, stage, "
    "performance status, prior therapy lines. Report hazard ratios and 95% CI.\n\n"
    "5. Subgroup Analysis: Consistency of score performance across: (a) nodular "
    "sclerosis vs mixed cellularity subtypes; (b) first-line vs relapsed/refractory; "
    "(c) PET-negative vs PET-positive at time of ctDNA collection.\n\n"
    "6. Longitudinal Assessment: For patients with ≥3 timepoints, compute "
    "score trajectory (slope) and test its association with PFS in a "
    "time-dependent Cox model."
)

doc.add_heading("Data-Sharing Plan", level=2)
doc.add_paragraph(
    "1. De-identified individual patient data will be deposited in a public "
    "repository (e.g., GEO or dbGaP) within 12 months of study completion.\n"
    "2. Data elements: patient demographics, ctDNA recurrence scores per timepoint, "
    "clinical outcomes (PFS, OS), variant-level VAF matrices.\n"
    "3. Analysis code (Python/R scripts) will be published on GitHub under MIT license.\n"
    "4. Summary statistics and ROC curves will be reported per STARD 2015 guidelines.\n"
    "5. No protected health information will be shared. All dates shifted by a "
    "random offset per patient to preserve temporal relationships."
)

doc.add_heading("Study Sites and Timeline", level=2)
tbl = doc.add_table(rows=5, cols=3)
tbl.style = "Light Grid Accent 1"
tbl.rows[0].cells[0].text = "Milestone"
tbl.rows[0].cells[1].text = "Target Date"
tbl.rows[0].cells[2].text = "Deliverable"
milestones = [
    ("Site recruitment and IRB approval", "Q1 2027", "3 sites approved"),
    ("Sample collection and ctDNA extraction", "Q2 2027", "250 samples"),
    ("Sequencing and score computation", "Q3 2027", "Score dataset"),
    ("Statistical analysis and manuscript", "Q4 2027", "Results + manuscript"),
]
for j, (ms, dt, dl) in enumerate(milestones):
    tbl.rows[j+1].cells[0].text = ms
    tbl.rows[j+1].cells[1].text = dt
    tbl.rows[j+1].cells[2].text = dl

doc.add_paragraph("\nEnd of synopsis. Version 1.0 — For review by scientific advisory board.")

doc.save(os.path.join(IMG_DIR, "validation_study_synopsis.docx"))
print("  Saved validation_study_synopsis.docx")

print("\nAll exercises complete.")
