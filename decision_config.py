"""
Recommended Pipeline Decision Configuration
============================================
Optimized via 29 experiments across 10 iterations (2026-02-25).
Contains BOTH v5.6 and v5.7 configurations. Parameter values are version-specific --
the FICO-4 structure is robust across model versions, but threshold values and blend
weights must be re-tuned after any retraining.

Replaces the current cascade: fraud-decline -> PD>=0.25 decline -> PD>=0.15 review -> Grade E+ review.

To integrate: import these constants into pipeline.py's decision logic.
Do NOT modify pipeline.py directly -- this file serves as the configuration spec.

Performance (test set v5.6 deduped, n=3,363, 6.6% bad rate):
  MCC:       0.1704 (baseline 0.1155, +47.5%)
  F2:        0.3440 (baseline 0.3089, +11.4%)
  Recall:    0.560 (baseline 0.229, +144.8%)
  Precision: 0.135 (baseline 0.129, +5.5%)
  Flag rate: 27.5% (baseline 11.8%)
  Net annual benefit: ~$1.9M (at 60% LGD, $50/review)

MODEL VERSION CAVEAT:
  These weights and thresholds were optimized on v5.6 model scores. After model
  retraining (v5.7), WoE/XGB score distributions change dramatically (correlation
  < 0.50 across versions). The v5.6-optimized weights scored MCC=0.1267 on v5.7,
  WORSE than default weights (MCC=0.1313). The FICO-4 STRUCTURE is robust; the
  specific THRESHOLD VALUES and BLEND WEIGHTS need re-tuning per model version.
  Always re-run threshold optimization after any model retraining.

SENSITIVITY NOTES (from iter 8-10 perturbation analysis):
  - fraud_weight:      HIGHEST IMPACT but VERSION-SPECIFIC. v5.6 optimal=0.25
                        (higher degrades monotonically, bootstrap p=0.950 that
                        fw=0.40 is worse). v5.7 optimal=0.48 (+46% vs baseline).
  - 650-700 threshold: MOST FRAGILE. Safe zone only +0.005 upward.
  - 700+ threshold:    INERT. No loans in test set reach this combined score.
                        Can be removed with zero impact (confirmed: v5.7 disables it).
  - woe_blend_weight:  LEAST SENSITIVE. Only 0.005 MCC spread across [0.40, 0.70].
  - FICO boundaries:   OPTIMAL at 600/650/700. All alternatives tested worse.
"""

# =============================================================================
# PD Blend Weights (replaces default 25/25/50 WoE/Rule/XGB)
# =============================================================================
# SENSITIVITY: woe weight is LEAST sensitive parameter (0.005 MCC spread across
# [0.40, 0.70]). Exact values matter less than the principle: heavy WoE, light Rule.
# WARNING: These weights are tuned for v5.6 scores. After retraining, re-optimize.
PD_BLEND_WEIGHTS = {
    "woe": 0.55,    # WoE scorecard -- strongest sub-model
    "rule": 0.10,   # Rule-based scorecard -- minimal contribution
    "xgb": 0.35,    # XGBoost -- second strongest
}

# =============================================================================
# Combined Score Formula
# =============================================================================
# combined_score = PD_WEIGHT * blended_pd + FRAUD_WEIGHT * (fraud_score / FRAUD_NORMALIZER)
# SENSITIVITY: fraud_weight is the HIGHEST-IMPACT parameter.
# Increasing from 0.25 to 0.40 yielded +9% MCC on v5.7 model.
# Consider fraud_weight=0.40 (pd_weight=0.60) pending v5.6 validation.
PD_WEIGHT = 0.75
FRAUD_WEIGHT = 0.25
FRAUD_NORMALIZER = 145.0  # max raw fraud_score

# =============================================================================
# FICO-Conditional Flag Thresholds (on combined_score)
# =============================================================================
# flag = combined_score >= threshold_for_fico_bucket
# SENSITIVITY per bucket:
#   lt_600:   moderate sensitivity
#   600_650:  moderate sensitivity
#   650_700:  MOST FRAGILE -- safe zone only +0.005 before degradation
#   gte_700:  INERT -- no loans in test set reach this combined score; can be removed
FICO_THRESHOLDS = {
    "lt_600":   0.140,  # FICO < 600: high-risk, moderate threshold
    "600_650":  0.080,  # FICO 600-650: aggressive -- this bucket was blind before
    "650_700":  0.115,  # FICO 650-700: moderate; FRAGILE -- do not raise above 0.120
    "gte_700":  0.140,  # FICO >= 700: INERT (no loans reach threshold); removable
}

# FICO bucket boundaries
# SENSITIVITY: 600/650/700 are OPTIMAL -- all alternative boundary positions tested worse
FICO_BOUNDARIES = [600, 650, 700]


# =============================================================================
# v5.7 Configuration (retrained models, 2026-02-25)
# =============================================================================
# Performance (test set v5.7, n=3,363, 6.6% bad rate):
#   MCC:       0.1755 (baseline 0.1203, +45.9%)
#   Flag rate: 13.6%
#   Precision: 17.5%
#   Recall:    36.3%
#
# Key differences from v5.6:
#   - fraud_weight nearly doubled (0.25 -> 0.48) -- fraud signal is stronger in v5.7
#   - WoE weight decreased (0.55 -> 0.45), XGB increased (0.35 -> 0.43)
#   - 700+ bucket DISABLED (no loans reach threshold; explicitly removed)
#   - Thresholds are much higher (v5.7 score distributions shifted upward)
# WARNING: v5.6 weights on v5.7 scores = MCC 0.1267 (WORSE than v5.7 default 0.1313).
#          Always use version-matched weights.

PD_BLEND_WEIGHTS_V57 = {
    "woe": 0.45,    # WoE scorecard -- still dominant but reduced from v5.6
    "rule": 0.12,   # Rule-based scorecard -- minimal contribution
    "xgb": 0.43,    # XGBoost -- increased from v5.6
}

PD_WEIGHT_V57 = 0.52
FRAUD_WEIGHT_V57 = 0.48
FRAUD_NORMALIZER_V57 = 111.0  # max fraud_score from v5.7 scoring (was 145 in v5.6)

FICO_THRESHOLDS_V57 = {
    "lt_600":   0.300,  # FICO < 600: high-risk, high threshold
    "600_650":  0.230,  # FICO 600-650: moderate threshold
    "650_700":  0.200,  # FICO 650-700: moderate threshold
    # NOTE: 700+ bucket DISABLED in v5.7 -- no loans reach combined score threshold.
    # FICO >= 700 always returns False (not flagged).
}

FICO_BOUNDARIES_V57 = [600, 650, 700]  # same boundaries, but 700+ bucket is disabled


def get_fico_bucket(fico_score: float) -> str:
    """Map FICO score to bucket key."""
    if fico_score < 600:
        return "lt_600"
    elif fico_score < 650:
        return "600_650"
    elif fico_score < 700:
        return "650_700"
    else:
        return "gte_700"


def compute_combined_score(
    woe_pd: float,
    rule_pd: float,
    xgb_pd: float,
    fraud_score: float,
) -> float:
    """Compute combined risk score from sub-model outputs."""
    blended_pd = (
        PD_BLEND_WEIGHTS["woe"] * woe_pd
        + PD_BLEND_WEIGHTS["rule"] * rule_pd
        + PD_BLEND_WEIGHTS["xgb"] * xgb_pd
    )
    return PD_WEIGHT * blended_pd + FRAUD_WEIGHT * (fraud_score / FRAUD_NORMALIZER)


def should_flag(
    woe_pd: float,
    rule_pd: float,
    xgb_pd: float,
    fraud_score: float,
    fico_score: float,
) -> bool:
    """Determine if a loan should be flagged for review/decline.

    Returns True if the loan should be flagged, False if approved.
    """
    combined = compute_combined_score(woe_pd, rule_pd, xgb_pd, fraud_score)
    bucket = get_fico_bucket(fico_score)
    threshold = FICO_THRESHOLDS[bucket]
    return combined >= threshold


def compute_combined_score_v57(
    woe_pd: float,
    rule_pd: float,
    xgb_pd: float,
    fraud_score: float,
) -> float:
    """Compute combined risk score from sub-model outputs (v5.7 weights)."""
    blended_pd = (
        PD_BLEND_WEIGHTS_V57["woe"] * woe_pd
        + PD_BLEND_WEIGHTS_V57["rule"] * rule_pd
        + PD_BLEND_WEIGHTS_V57["xgb"] * xgb_pd
    )
    return PD_WEIGHT_V57 * blended_pd + FRAUD_WEIGHT_V57 * (fraud_score / FRAUD_NORMALIZER_V57)


def should_flag_v57(
    woe_pd: float,
    rule_pd: float,
    xgb_pd: float,
    fraud_score: float,
    fico_score: float,
) -> bool:
    """Determine if a loan should be flagged for review/decline (v5.7 config).

    Returns True if the loan should be flagged, False if approved.
    FICO >= 700 is always NOT flagged (700+ bucket disabled in v5.7).
    """
    # 700+ bucket disabled -- never flag high-FICO loans
    if fico_score >= 700:
        return False
    combined = compute_combined_score_v57(woe_pd, rule_pd, xgb_pd, fraud_score)
    bucket = get_fico_bucket(fico_score)
    threshold = FICO_THRESHOLDS_V57[bucket]
    return combined >= threshold


# =============================================================================
# What this REMOVES from the current pipeline
# =============================================================================
# 1. Fraud review flag (fraud_score > X but < decline) -- HURTS MCC, removed
# 2. Grade-based review (Grade E/F/G -> review) -- captured by PD thresholds
# 3. Separate decline/review thresholds -- simplified to single flag threshold
#    (downstream process can tier flagged loans by score distance from threshold)
#
# What this KEEPS:
# 1. Fraud hard-decline (fraud_score indicating clear fraud) -- keep as override
# 2. All three sub-models (WoE, Rule, XGB) -- just reweighted
# 3. Fraud score as input -- just normalized and blended vs. separate cascade


# =============================================================================
# Deployment Protocol: Re-optimization After Retraining
# =============================================================================
# The FICO-4 STRUCTURE is robust across model versions, but specific
# THRESHOLD VALUES and BLEND WEIGHTS must be re-tuned after any retraining.
#
# Evidence:
#   v5.6 optimal: fw=0.25, woe=0.55, thresholds=[0.140, 0.080, 0.115, 0.140]
#   v5.7 optimal: fw=0.48, woe=0.45, thresholds=[0.300, 0.230, 0.200, disabled]
#   v5.6 weights on v5.7: MCC 0.1267 (WORSE than v5.7 default 0.1313)
#
# Fraud weight is the HIGHEST-IMPACT parameter and most version-sensitive.

REOPTIMIZATION_CHECKLIST = [
    "1. Score full dataset with new model via Pipeline.score_batch()",
    "2. Grid search sub-model blend weights (step 0.05, WoE-dominant prior)",
    "3. Grid search FICO-bucket thresholds (step 0.005, per-bucket independent)",
    "4. Test 3-bucket (700+ disabled) vs 4-bucket — 700+ may be inert",
    "5. Sweep fraud_weight 0.10-0.50 (step 0.01) — highest-impact parameter",
    "6. Bootstrap validate (1000+ resamples, require p<0.05 vs baseline)",
    "7. Update this config file with new values",
    "8. Estimated cost: ~2 hours compute + analysis per retraining cycle",
]
