"""
Pseudocode for two additional scoring features:
  1. Fusion feature — detect gene fusions from paired-end sequencing data
  2. Pathway activation feature — score pathway-level signaling activation
"""

# =============================================================================
# FUSION FEATURE  (scoring range: 0–15 points)
# =============================================================================
"""
PSEUDOCODE: fusion_score(patient_id) -> int (0–15)

# Step 1: Identify candidate fusions from BAM alignment
# Use discordant read pairs and split-read evidence

FUNCTION detect_fusions(patient_id):
    candidates = []
    
    FOR each BAM file for patient_id:
        # 1a. Discordant read pairs (mates map to different genes)
        discordant = QUERY_BAM(
            mate_mapping = DIFFERENT_GENE,
            min_mapq = 30,
            proper_pair = FALSE
        )
        
        # 1b. Split reads (reads spanning fusion junctions)
        split = QUERY_BAM(
            supplementary_alignment = TRUE,
            min_mapq = 30,
            min_split_bases = 20
        )
        
        # 1c. Cluster by breakpoint coordinates (±500 bp)
        clusters = CLUSTER_BY_BREAKPOINT(
            discordant ∪ split,
            window = 500,
            min_reads_per_cluster = 3
        )
        
        FOR each cluster in clusters:
            gene_pair = ANNOTATE_GENES(
                cluster.breakpoint_5p,
                cluster.breakpoint_3p
            )
            IF gene_pair.genes ARE VALID:
                candidates.APPEND(FusionCandidate(
                    gene_5p = gene_pair.gene_5p,
                    gene_3p = gene_pair.gene_3p,
                    breakpoint_5p = cluster.breakpoint_5p,
                    breakpoint_3p = cluster.breakpoint_3p,
                    supporting_reads = len(cluster),
                    split_reads = COUNT(cluster, type=SPLIT),
                    discordant_pairs = COUNT(cluster, type=DISCORDANT)
                ))

    # Step 2: Filter candidates with confidence criteria
    high_confidence = []
    FOR each candidate in candidates:
        score = 0
        IF candidate.split_reads >= 2:               score += 3  # strong junction evidence
        IF candidate.discordant_pairs >= 5:           score += 2  # structural support
        IF candidate.genes ARE IN ONCOGENE_LIST:      score += 2  # biological relevance
        IF candidate.genes ARE IN HL_DRIVER_LIST:     score += 3  # HL-specific relevance
        IF candidate.fusion IS IN-FRAME:              score += 2  # preserves reading frame
        IF candidate.fusion PRESERVES KINASE DOMAIN:  score += 3  # functional domain
        
        IF score >= 5:  # confidence threshold
            high_confidence.APPEND(candidate)

    RETURN high_confidence


# Step 3: Score the patient based on detected fusions

FUNCTION fusion_score(patient_id):
    fusions = detect_fusions(patient_id)
    
    IF len(fusions) == 0:
        RETURN 0
    
    base_score = 0
    
    # Score based on number of distinct fusions
    IF len(fusions) >= 3:  base_score = 15  # multiple drivers
    ELIF len(fusions) == 2: base_score = 12  # two independent events
    ELIF len(fusions) == 1:
        fusion = fusions[0]
        # Single fusion scored by biological impact
        IF fusion.genes ARE IN HL_DRIVER_LIST:
            IF fusion IS RECURRENT IN COHORT:  base_score = 12
            ELSE:                                base_score = 10
        ELIF fusion IS IN-FRAME AND PRESERVES DOMAIN:
            base_score = 8
        ELSE:
            base_score = 5
    
    # Bonus for clonal prevalence (VAF of fusion reads)
    max_vaf = MAX(fusion.vaf FOR fusion in fusions)
    IF max_vaf > 0.30:           base_score += 3   # clonal/early
    ELIF max_vaf > 0.10:         base_score += 1   # subclonal
    
    RETURN MIN(base_score, 15)


# Notes:
# - In targeted AmpliSeq panels, fusion detection is limited by amplicon design.
#   Multi-gene amplicons (e.g., "CSF1R,HMGXB3") may reflect panel design artifacts
#   rather than true fusions. WGS/WCS data is preferred.
# - The pseudocode above assumes WGS or hybrid-capture WES with real paired-end reads.
# - For this cohort, multi-gene variant_id entries in trusted_variants.csv provide
#   a proxy: count split-alignment reads per patient as a fusion suspicion score.
"""


# =============================================================================
# PATHWAY ACTIVATION FEATURE  (scoring range: 0–20 points)
# =============================================================================
"""
PSEUDOCODE: pathway_activation_score(patient_id) -> int (0–20)

# Step 1: Define HL-relevant pathways with gene members and weights

PATHWAYS = {
    "NF-kB": {
        "genes":       ["CARD11", "TNFAIP3", "REL", "BCL10", "NFKB1", "NFKB2", "IKBKB"],
        "weight":       5,      # max points for this pathway
        "activating":   ["CARD11", "REL"],   # GOF mutations increase signaling
        "inactivating": ["TNFAIP3"],          # LOF mutations increase signaling (tumor suppressor)
        "sign":         +1,     # positive = activation
    },
    "JAK-STAT": {
        "genes":       ["STAT3", "STAT6", "JAK1", "JAK2", "JAK3", "SOCS1"],
        "weight":       5,
        "activating":   ["STAT3", "STAT6", "JAK1", "JAK2"],
        "inactivating": ["SOCS1"],            # negative regulator LOF = activation
        "sign":         +1,
    },
    "NOTCH": {
        "genes":       ["NOTCH1", "NOTCH2", "FBXW7"],
        "weight":       3,
        "activating":   ["NOTCH1", "NOTCH2"],
        "inactivating": ["FBXW7"],
        "sign":         +1,
    },
    "TP53 / Apoptosis": {
        "genes":       ["TP53", "CASP8", "FAS", "EP300", "CREBBP", "BCL2", "BAX"],
        "weight":       4,
        "activating":   [],
        "inactivating": ["TP53", "CASP8", "FAS"],  # LOF = loss of tumor suppression
        "sign":         -1,     # negative = pathway disabled (pro-cancer)
    },
    "MAPK / PI3K": {
        "genes":       ["KRAS", "NRAS", "BRAF", "PIK3CD", "PIK3CA", "PTEN", "MTOR"],
        "weight":       3,
        "activating":   ["KRAS", "NRAS", "BRAF", "PIK3CD", "PIK3CA"],
        "inactivating": ["PTEN"],               # LOF = activation of PI3K signaling
        "sign":         +1,
    },
}


# Step 2: Per-patient pathway scoring

FUNCTION pathway_activation_score(patient_id):
    mutations = GET_MUTATIONS(patient_id)   # list of (gene, vaf, effect_type, is_driver)
    
    IF len(mutations) == 0:
        RETURN 0
    
    pathway_scores = {}
    
    FOR each pathway_name, config in PATHWAYS:
        pathway_mutations = [m for m in mutations if m.gene in config["genes"]]
        
        IF len(pathway_mutations) == 0:
            pathway_scores[pathway_name] = 0
            CONTINUE
        
        # Calculate pathway perturbation score
        activation_score = 0
        inactivation_score = 0
        
        FOR each mutation in pathway_mutations:
            weight = 1.0
            
            # Weight by variant effect severity
            IF mutation.effect_type == "frameshift" OR mutation.effect_type == "nonsense":
                weight = 1.5       # truncating = strong impact
            ELIF mutation.effect_type == "missense":
                weight = 1.0
            ELIF mutation.effect_type == "splice_site" OR mutation.effect_type == "nonframeshift_deletion":
                weight = 1.2
            ELSE:
                weight = 0.5       # unknown / synonymous (lower confidence)
            
            # Weight by VAF (clonality)
            vaf_weight = 1.0
            IF mutation.vaf > 0.30:    vaf_weight = 1.5   # clonal
            ELIF mutation.vaf > 0.10:  vaf_weight = 1.0   # subclonal
            ELSE:                       vaf_weight = 0.5   # low VAF, uncertain
            
            composite_weight = weight * vaf_weight
            
            # Assign to activation or inactivation
            IF mutation.gene IN config["activating"]:
                activation_score += composite_weight
            IF mutation.gene IN config["inactivating"]:
                inactivation_score += composite_weight
        
        # Combine: pathway net score
        IF config["sign"] == +1:
            net = activation_score + inactivation_score  # both contribute to activation
        ELSE:  # sign == -1 (tumor suppressor pathway)
            net = -(activation_score + inactivation_score)  # negative = disabled
        
        # Normalize to pathway weight cap
        net = CLAMP(net, -config["weight"], config["weight"])
        
        # Apply non-linearity: one strong hit can be as bad as many weak ones
        IF net > 0:
            net = MIN(net, config["weight"])  # cap positive
            pathway_scores[pathway_name] = net
        ELSE:
            pathway_scores[pathway_name] = net  # negative preserved for tumor suppressors
    
    
    # Step 3: Aggregate into single 0–20 score
    
    total_activation = 0
    total_suppression = 0
    
    FOR each pathway_name, score in pathway_scores:
        IF score > 0:
            total_activation += score
        ELSE:
            total_suppression += abs(score)
    
    # Activation score (0–20): sum of activated pathway scores capped at 20
    raw = total_activation + total_suppression  # both pro-tumor
    final_score = CLAMP(raw, 0, 20)
    
    RETURN final_score


# Step 4: Alternative — binary pathway fingerprint

FUNCTION pathway_fingerprint(patient_id) -> dict:
    """
    Returns which pathways are "activated" (score >= threshold)
    Useful for clustering patients by pathway profile.
    """
    mutations = GET_MUTATIONS(patient_id)
    fingerprint = {}
    
    FOR name, config in PATHWAYS:
        hits = [m for m in mutations if m.gene in config["genes"]]
        IF config["sign"] == +1:
            activating_hits = [m for m in hits if m.gene in config["activating"]]
            inactivating_hits = [m for m in hits if m.gene in config["inactivating"]]
            fingerprint[name] = (len(activating_hits) + len(inactivating_hits)) >= 1
        ELSE:
            inactivating_hits = [m for m in hits if m.gene in config["inactivating"]]
            fingerprint[name] = len(inactivating_hits) >= 1  # TS pathway disabled
    
    RETURN fingerprint


# Example usage for patient LH210010:
#   mutations = [
#       ("NOTCH1",  0.45, "missense",  True),   # NF-kB activating via non-canonical
#       ("NOTCH1",  0.12, "missense",  True),   # also NOTCH pathway
#       ("STAT6",   0.08, "missense",  True),   # JAK-STAT activating
#       ("CARD11",  0.35, "missense",  True),   # NF-kB activating
#       ("CASP8",   0.22, "missense",  True),   # TP53/Apoptosis inactivating
#       ("EP300",   0.18, "missense",  True),   # TP53/Apoptosis inactivating
#   ]
#   score = pathway_activation_score("LH210010")
#   # Expected: NF-kB=strong, JAK-STAT=moderate, NOTCH=moderate, TP53=moderate suppression
#   # Total: ~12–14 / 20
"""

# =============================================================================
# INTEGRATION NOTE
# =============================================================================
"""
These two features (Fusion 0-15, Pathway 0-20) can be added to the existing
composable scoring system:
  - tb:  0-40  (tumor burden)
  - ce:  0-25  (clonal expansion)
  - drv: 0-25  (driver increasing)
  - mg:  0-10  (gained mutations)
  - fu:  0-15  (fusion burden)      ← NEW
  - pa:  0-20  (pathway activation)  ← NEW
  ————————————————————
  Total: 0-135

This expands the scoring range while maintaining the same additive architecture.
The tier boundaries would need recalibration with the new max of 135.
"""
print("Pseudocode printed to stdout. See _pseudocode_features.py for reference.")
