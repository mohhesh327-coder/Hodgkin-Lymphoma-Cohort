"""
Generate three one-page clinical reports for patients from each tier.
Uses a multi-panel PDF-style figure per patient.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from math import ceil

# === Load data ===
cs = pd.read_csv("composite_scores.csv")
tv = pd.read_csv("trusted_variants.csv")
traj = pd.read_csv("trajectory_per_patient_per_variant.csv")
df = pd.read_csv("filtered_variants.csv")

DRIVER_GENES = ["TP53","NOTCH1","KRAS","EP300","CASP8","MYB","CARD11","TNFAIP3","STAT3","STAT6","CREBBP","SMARCA4"]

def first_gene(g):
    return str(g).split(",")[0].strip()

# === Patient definitions ===
PATIENTS = {
    "LH210010": {
        "tier": "High-Confidence Recurrence",
        "desc": "Perfect scorer (100/100) — clear ctDNA evidence of active disease."
    },
    "LH200002": {
        "tier": "Possible Recurrence",
        "desc": "Ambiguous zone (51/100) — weak clonal expansion but detectable tumor DNA."
    },
    "BR200023": {
        "tier": "No Recurrence Signal",
        "desc": "Minimal signal (11/100) — below actionable threshold."
    }
}

TCOLORS = {"High-Confidence Recurrence": "#27ae60",
           "Possible Recurrence": "#f39c12",
           "No Recurrence Signal": "#e74c3c"}

def parse_clinvar(cv):
    if pd.isna(cv) or cv.strip() == "":
        return "Not annotated"
    c = str(cv).lower()
    if "patog" in c or "pathogenic" in c:
        return "Pathogenic"
    if "benign" in c and "uncertain" not in c:
        return "Benign"
    if "uncertain" in c:
        return "Uncertain significance"
    if "likely benign" in c:
        return "Likely benign"
    if "likely pathogenic" in c:
        return "Likely pathogenic"
    if "tolerado" in c or "tolerated" in c:
        return "Likely benign (in silico)"
    if "utr" in c or "polimorfismo" in c:
        return "Benign (UTR/polymorphism)"
    return "Not annotated"

def build_report(pid):
    row = cs[cs["patient"]==pid].iloc[0]
    tier_name = PATIENTS[pid]["tier"]
    tier_desc = PATIENTS[pid]["desc"]
    tier_color = TCOLORS[tier_name]

    pm = tv[tv["patient_id"]==pid].copy()
    pt = traj[traj["patient_id"]==pid].copy()
    tps_ordered = sorted(pm["tp_order"].unique())
    tp_labels = [pm[pm["tp_order"]==t]["tp_label"].iloc[0] for t in tps_ordered]
    n_tp = len(tps_ordered)

    # Trajectory lookup
    traj_lookup = pt.set_index("variant_id")["trajectory"].to_dict()

    # --- Create figure (A3-ish at 200 dpi, multi-panel) ---
    fig = plt.figure(figsize=(16, 22))
    gs = fig.add_gridspec(5, 3, height_ratios=[0.6, 1.2, 0.8, 1.2, 0.5],
                           hspace=0.3, wspace=0.25)

    # === HEADER: patient info ===
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis("off")
    score_color = "#27ae60" if row["total"] >= 80 else ("#f39c12" if row["total"] >= 40 else "#e74c3c")
    ax_h.text(0.02, 0.7, f"Patient: {pid}", fontsize=18, fontweight="bold", va="center")
    ax_h.text(0.02, 0.3, f"Recurrence Score: {row['total']:.0f}/100   |   Tier: {tier_name}",
              fontsize=13, color=score_color, fontweight="bold", va="center")
    ax_h.text(0.55, 0.5, tier_desc, fontsize=11, color="gray", va="center",
              style="italic", wrap=True)

    # === PANEL 1: Score breakdown ===
    ax1 = fig.add_subplot(gs[1, 0])
    comps = ["TB", "CE", "DRV", "MG"]
    vals = [row["tb"], row["ce"], row["drv"], row["mg"]]
    colors_comp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax1.barh(comps, vals, color=colors_comp, edgecolor="white", linewidth=1.2, height=0.6)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f"{v:.1f}",
                 va="center", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 45)
    ax1.set_xlabel("Score")
    ax1.set_title("Score Breakdown", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # === PANEL 2: Peak VAF trajectory ===
    ax2 = fig.add_subplot(gs[1, 1:])
    # Find the variant with highest latest-VAF among increasing variants
    inc_variants = pt[pt["trajectory"]=="increasing"]["variant_id"].tolist()
    if inc_variants:
        # Pick the one with highest last_VAF
        top_inc = pt[pt["variant_id"].isin(inc_variants)].sort_values("last_VAF", ascending=False)
        peak_vid = top_inc.iloc[0]["variant_id"]
        peak_gene = top_inc.iloc[0]["variant_id"].split("|")[1].strip() if "|" in str(top_inc.iloc[0]["variant_id"]) else "?"
        vdf = pm[pm["variant_id"]==peak_vid].sort_values("tp_order")
        x = range(len(vdf))
        ax2.plot(x, vdf["VAF"].values, "-o", color="#e74c3c", linewidth=2.5, markersize=8, zorder=3)
        ax2.set_xticks(range(len(tps_ordered)))
        ax2.set_xticklabels(tp_labels, fontsize=10)
        ax2.set_ylim(-0.02, max(vdf["VAF"].max()*1.2, 0.1))
        ax2.set_ylabel("VAF", fontsize=11)
        ax2.set_title(f"Peak Increasing Variant: {peak_gene} ({peak_vid[:30]}...)",
                      fontsize=11, fontweight="bold")
    else:
        # No inc variant — show the variant with highest overall VAF change
        vdf = pm.groupby("variant_id").filter(lambda g: len(g)>=2)
        if len(vdf)>0:
            vdf = vdf.groupby("variant_id").apply(lambda g: g.sort_values("tp_order"))
            vdf = vdf.reset_index(drop=True)
            # Pick variant with max VAF at latest TP
            latest_tp = max(tps_ordered)
            top_at_latest = pm[pm["tp_order"]==latest_tp].sort_values("VAF", ascending=False)
            if len(top_at_latest)>0:
                peak_vid = top_at_latest.iloc[0]["variant_id"]
                peak_gene = top_at_latest.iloc[0]["variant_id"].split("|")[1].strip() if "|" in str(top_at_latest.iloc[0]["variant_id"]) else "?"
                vdf = pm[pm["variant_id"]==peak_vid].sort_values("tp_order")
                x = range(len(vdf))
                ax2.plot(x, vdf["VAF"].values, "-o", color="#2980b9", linewidth=2.5, markersize=8, zorder=3)
                ax2.set_xticks(range(len(tps_ordered)))
                ax2.set_xticklabels(tp_labels, fontsize=10)
                ax2.set_ylim(-0.02, max(vdf["VAF"].max()*1.2, 0.1))
                ax2.set_ylabel("VAF", fontsize=11)
                ax2.set_title(f"Highest-VAF Variant: {peak_gene} ({peak_vid[:30]}...)",
                              fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # === PANEL 3: Per-mutation trend bundle ===
    ax3 = fig.add_subplot(gs[2, :])
    # Show all longitudinal variants as small lines
    pm_long = pm.groupby("variant_id").filter(lambda g: len(g)>=2)
    if len(pm_long) > 0:
        for vid, grp in pm_long.groupby("variant_id"):
            grp = grp.sort_values("tp_order")
            x = [list(tps_ordered).index(t) for t in grp["tp_order"].values]
            ttype = traj_lookup.get(vid, "stable")
            c = {"increasing":"#e74c3c","decreasing":"#2980b9","stable":"#bdc3c7"}.get(ttype, "#bdc3c7")
            alpha = 0.7 if ttype != "stable" else 0.2
            lw = 1.2 if ttype != "stable" else 0.4
            ax3.plot(x, grp["VAF"].values, color=c, alpha=alpha, linewidth=lw)
    ax3.set_xticks(range(len(tps_ordered)))
    ax3.set_xticklabels(tp_labels, fontsize=10)
    ax3.set_ylim(-0.02, 1.02)
    ax3.set_ylabel("VAF", fontsize=11)
    ax3.set_xlabel("Timepoint", fontsize=11)
    ax3.set_title("All Longitudinal Variants (red=inc, blue=dec, gray=stable)", fontsize=11, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    legend_elements = [Line2D([0],[0],color=c, lw=2, label=l) for c,l in
                       zip(["#e74c3c","#2980b9","#bdc3c7"], ["Increasing","Decreasing","Stable"])]
    ax3.legend(handles=legend_elements, fontsize=8, loc="upper right")

    # === PANEL 4: Top 5 mutations by latest VAF ===
    ax4 = fig.add_subplot(gs[3, :])
    ax4.axis("off")
    latest_tp = max(tps_ordered)
    top5 = pm[pm["tp_order"]==latest_tp].sort_values("VAF", ascending=False).head(5)

    col_labels = ["Gene", "Variant", "VAF", "Effect", "ClinVar", "Driver"]
    table_data = []
    for _, r in top5.iterrows():
        vid = r["variant_id"]
        var_short = vid.split("|")[0].split(":")[-1][:15] if "|" in str(vid) else str(vid)[:15]
        gene = first_gene(r["Genes"])
        cv_parsed = parse_clinvar(r["ClinVar"])
        is_driver = "✓" if gene in DRIVER_GENES else ""
        table_data.append([
            gene,
            var_short,
            f"{r['VAF']:.4f}",
            str(r["Variant Effect"])[:12],
            cv_parsed[:18],
            is_driver,
        ])

    table = ax4.table(cellText=table_data, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       colWidths=[0.10, 0.20, 0.08, 0.15, 0.20, 0.06])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Style header
    for j, col in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    # Color driver rows
    for i, row_data in enumerate(table_data):
        if row_data[4] == "✓" or row_data[0] in DRIVER_GENES:
            table[(i+1), 0].set_facecolor("#d5f5e3")
    ax4.set_title(f"Top 5 Mutations at Latest Timepoint ({tp_labels[-1]})",
                  fontsize=12, fontweight="bold", pad=20)

    # === PANEL 5: Limitations and Recommendations ===
    ax5 = fig.add_subplot(gs[4, :])
    ax5.axis("off")

    # Counts for context
    n_inc = (pt["trajectory"]=="increasing").sum()
    n_dec = (pt["trajectory"]=="decreasing").sum()
    n_sta = (pt["trajectory"]=="stable").sum()
    n_gains = len(df[(df["patient_id"]==pid) & (df["filter_pass"])])
    # Driver gene hits
    pm["gene"] = pm["Genes"].apply(first_gene)
    n_drv_genes = pm[pm["gene"].isin(DRIVER_GENES)]["gene"].nunique()

    limitations = (
        f"LIMITATIONS — This assessment is based on a research-use-only targeted ctDNA panel "
        f"(GSE244989, AmpliSeq HL panel). Key limitations include: (1) CHIP (Clonal Hematopoiesis "
        f"of Indeterminate Potential) — mutations in DNMT3A, TET2, ASXL1, and TP53 in blood cells "
        f"can produce false-positive ctDNA signals indistinguishable from tumor-derived DNA; "
        f"(2) Panel coverage — 46 genes targeting known HL hotspots; novel or non-canonical "
        f"driver events outside these regions are missed; (3) Limited sensitivity for low-VAF "
        f"variants — the LoD filter removes variants below gene-specific thresholds, which may "
        f"discard true subclonal mutations; (4) Recurrence score is a composite research metric "
        f"and has not been clinically validated; treatment decisions should not be based on this "
        f"score alone; (5) {n_tp} timepoints available — additional timepoints would improve "
        f"trajectory classification confidence."
    )

    recommendations = (
        f"RECOMMENDATIONS — Based on a score of {row['total']:.0f}/100 ({tier_name}): "
    )
    if row["total"] >= 80:
        recommendations += (
            f"Strong molecular evidence of active disease ({n_inc} increasing variants, "
            f"{n_drv_genes} driver genes affected). Recommended actions: (a) Correlate with "
            f"imaging (PET-CT) to confirm metabolic activity; (b) Consider ctDNA-guided therapy "
            f"adjustment if clinical progression is concordant; (c) Repeat ctDNA assessment at "
            f"next treatment cycle to monitor trajectory; (d) Biopsy confirmation of any new "
            f"actionable mutations (e.g., NOTCH1, REL) before targeted therapy initiation."
        )
    elif row["total"] >= 40:
        recommendations += (
            f"Ambiguous molecular signal ({n_inc} increasing, {n_drv_genes} driver genes). "
            f"Recommended actions: (a) Short-interval repeat ctDNA (4-6 weeks) to determine "
            f"if the borderline signal is trending up or down; (b) Correlate with clinical "
            f"status — if asymptomatic, consider watchful waiting; (c) Add parallel ctRNA "
            f"assessment for fusion transcripts (e.g., CIITA-CD274) which may provide "
            f"complementary evidence; (d) Review all decreasing variants — their clearance "
            f"patterns may indicate treatment response."
        )
    else:
        recommendations += (
            f"No significant ctDNA evidence of active disease ({n_inc} increasing variants, "
            f"{n_drv_genes} driver genes with stable/decaying signal). IMPORTANT: A 'no signal' "
            f"result does NOT rule out residual disease — it may reflect: (a) Disease below "
            f"ctDNA detection threshold; (b) Tumor with low shedding rate (e.g., fibrotic or "
            f"necrotic lesions); (c) Effective treatment response with minimal residual disease. "
            f"Recommended actions: (a) Maintain clinical surveillance with imaging; "
            f"(b) Continue ctDNA monitoring at next scheduled visit; (c) Consider ctRNA-based "
            f"complementary assay for expression/fusion markers if high clinical suspicion persists."
        )

    ax5.text(0.02, 0.85, limitations, fontsize=8.5, color="gray", va="top", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9", edgecolor="#dddddd"))
    ax5.text(0.02, 0.22, recommendations, fontsize=8.5, color="#2c3e50", va="top", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#ebf5fb", edgecolor="#3498db"))

    plt.savefig(f"report_{pid}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved report_{pid}.png")

for pid in PATIENTS:
    build_report(pid)
print("\nAll 3 reports generated.")
