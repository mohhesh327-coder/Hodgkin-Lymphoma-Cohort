# Hodgkin Lymphoma Cohort — ctDNA Recurrence Scoring (GSE244989)

Longitudinal circulating tumor DNA (ctDNA) analysis and a 100-point recurrence
scoring system for the GSE244989 classical Hodgkin lymphoma cohort (n = 46 patients).

## Overview

This project takes raw per-patient AmpliSeq variant tables from GEO accession
**GSE244989** and builds a complete ctDNA monitoring pipeline:

1. **Parsing** — 46 patient xlsx files → unified variant table (13,837 raw rows)
2. **Chronological ordering** — ddMMyy dates embedded in sample names → timepoint labels (T0, T1, …)
3. **Germline / artifact filtering** — per-gene limit of detection (mean + 3×SD from 11 normals), driver-gene rescue at VAF > 0.01, block list for variants in ≥3 normals → **11,423 trusted variants** (82.6% pass rate)
4. **Trajectory classification** — `increasing` / `decreasing` / `stable` per variant per patient (~8,800 classifications)
5. **Mutation gain detection** — variants absent at baseline, present at any later timepoint
6. **Composable scoring** — four additive components, 0–100 total
7. **Figures, reports, and study design** — 30+ publication figures, clinical reports, SOPs

## The Recurrence Score (0–100)

| Component | Range | Definition |
|-----------|-------|------------|
| **TB** (Tumor Burden) | 0–40 | Sum of VAF rises across increasing variants, scaled ×123.08 |
| **CE** (Clonal Expansion) | 0 / 8 / 17 / 25 | Count of increasing mutations (0 / 1 / 2 / ≥3) |
| **DRV** (Driver Inc.) | 0 / 10 / 25 | Increasing mutations in 12 HL driver genes (0 / 1 / ≥2) |
| **MG** (Mutation Gain) | 0 / 5 / 10 | Newly gained mutations (0 / 1–2 / ≥3) |

### Tier boundaries

| Tier | Score | Count | % |
|------|-------|-------|---|
| High-confidence | ≥ 95 | 11 | 24% |
| Strong recurrence | 55–94 | 9 | 20% |
| Possible | 30–54 | 7 | 15% |
| Weak | 10–29 | 8 | 17% |
| None | < 10 | 11 | 24% |

**9 patients score a perfect 100/100.**

## HL Driver Genes (12)

TP53 · NOTCH1 · KRAS · EP300 · CASP8 · MYB · CARD11 · TNFAIP3 · STAT3 · STAT6 · CREBBP · SMARCA4

Cohort highlights: NOTCH1 dominates (940 patient-variant hits, 21%); KRAS has 0 hits;
64 of 80 increasing trajectories are in driver genes.

## Repository Contents

### Scripts (`*.py`) — 43 files
| Script | Purpose |
|--------|---------|
| `analysis.py`, `advanced_analysis.py` | Main pipeline entry points |
| `lod_filter.py` | Per-gene LoD germline/artifact filter |
| `trajectory.py` | Trajectory classification |
| `final_calibrated.py` | Final tb/ce/drv/mg scoring calibration |
| `exercises_all.py` | Correlation, waterfall, plotly, manuscript panel |
| `exercises_ctRNA.py` | ctRNA panel design, fusion simulation, CHIP analysis |
| `generate_clinical_reports.py` | 3 patient reports (PNG) |
| `generate_exercises.py` | DOCX reports, SOP, patient summary, validation study |
| `figure_*.py` | All standalone figure generators |
| `exercise*.py` | Per-exercise analysis scripts |

### Data (`*.csv`) — 37 files
| File | Description |
|------|-------------|
| `filtered_variants.csv` | Full dataset with per-variant filter decisions |
| `trusted_variants.csv` | Pass-only variants (11,423 rows) |
| `trajectory_per_patient_per_variant.csv` | ~8,800 trajectory classifications |
| `composite_scores.csv` | Final tb/ce/drv/mg/total scores for 46 patients |
| `trusted_pivot_vaf.csv` / `trusted_pivot_binary.csv` | Trusted VAF / binary matrices |
| `simulated_fusions.csv` | Synthetic fusion data (10 patients) |
| `scoring_freq_weighted.csv` | Frequency-weighted driver scoring |

### Figures (`*.png`) — 37 files
- `figure_trajectory_grid.png` — all 46 patients' trajectories sorted by score
- `figure_driver_landscape_vaf.png` — max VAF per gene per patient heatmap
- `figure_sensitivity_contribution.png` — sensitivity analysis + stacked contribution
- `figure_tier_distribution.png` — tier lollipop / bar / donut (3-panel)
- `figure_histogram_and_effects.png` — score histogram + variant effect catalog
- `component_heatmap.png` — final component heatmap
- `report_*.png` — 3 one-page clinical reports
- `exercise*_*.png` — per-exercise figures
- Plus fingerprint, clonal expansion, red flags, stress test, etc.

### Documents (`*.docx`) — 6 files
- `report_LH210010.docx`, `report_LH200002.docx`, `report_BR200023.docx` — clinical reports (High / Possible / None tiers)
- `sop_incidental_findings.docx` — incidental-finding SOP (ACMG genes, notification workflow)
- `patient_summary_BR190005_OK.docx` — plain-language patient summary (~200 words)
- `validation_study_synopsis.docx` — 250-patient validation study design (power calc, ROC/KM plan)

### Raw data (`*.xlsx` + `*.tar.gz`) — 47 files
- 46 per-patient xlsx variant tables (original GEO processed data)
- `GSE244989_Processed_data.tar.gz` (5.75 MB) — original GEO archive

### Other
- `exercise3_interactive_heatmap.html` — plotly interactive heatmap

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
python-docx>=1.0
Pillow
scipy
openpyxl
```

Install: `pip install -r requirements.txt`

## How to Run

```bash
# 1. Main pipeline
python analysis.py            # parse + order + filter
python lod_filter.py          # germline / LoD filter
python trajectory.py          # trajectory classification

# 2. Final scoring
python final_calibrated.py    # tb / ce / drv / mg / total

# 3. Figures
python figure_trajectory_grid.py
python figure_driver_landscape_vaf.py
python figure_tier_distribution.py

# 4. Reports
python generate_clinical_reports.py
python generate_exercises.py
```

## Reproducibility Notes

- **Chronological ordering**: ddMMyy dates parsed from sample names; `tp_order` from date+sequence
- **Trajectory rule**: increasing if last_VAF > 1.5× first; decreasing if < 0.5×; else stable
- **LoD**: per-gene max(mean + 3×SD in 11 normals, 0.01)
- **Random seeds**: fixed (42) in all simulation scripts
- **Final calibration**: tb factor = 40 / 9th-highest rise among patients with ce+drv+mg=60

## Limitations (Research Use Only)

1. **CHIP** — clonal hematopoiesis variants (DNMT3A, TET2, ASXL1, TP53) in blood cells can produce false-positive ctDNA signals
2. **Panel coverage** — 46-gene targeted AmpliSeq panel; novel drivers outside these regions are missed
3. **LoD sensitivity** — gene-specific thresholds may discard true subclonal mutations
4. **Not clinically validated** — the 100-point score is a research metric; clinical adoption requires the validation study in `validation_study_synopsis.docx`

## Data Source

- GEO accession: [GSE244989](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244989)
- Study: longitudinal ctDNA in classical Hodgkin lymphoma (targeted AmpliSeq HL panel)
- Cohort: 46 patients, 168 sample columns, 2–11 timepoints per patient

## License

Research use only. Original data subject to GEO terms.
