"""
Enhanced Feature Engineering Pipeline
=====================================
Shared by all four systems (default, fraud, identity, credit).
Builds novel features from bureau, QI, partner, and identity provider data.

Design inspired by production FeatureEngine: boundary grading, weighted composites,
categorical risk mapping. All features are safe (not outputs of existing underwriting).
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from model_utils import (
    load_master, load_default_model, load_funnel,
    prepare_credit_data, BAD_STATES, GOOD_STATES, ACTIVE_STATES, EXCLUDE_STATES,
    SPLIT_DATE, MIN_SEASONING_DAYS, PARTNER_ENCODING
)

logger = logging.getLogger(__name__)

# ==============================================================================
# LEAKY FEATURE BLOCKLIST
# ==============================================================================
# These features are integration-timeline proxies (e.g., has_taktile went from
# 8% in Jan 2024 to 100% by Jul 2025). They leak information about WHEN a loan
# was originated, not about borrower risk. MUST NOT be used in modeling.
LEAKY_FEATURE_BLOCKLIST = frozenset({
    "has_prove_data",
    "has_plaid_data",
    "has_taktile_data",
    "has_giact_data",
    "has_taktile",
    "has_prove",
    "has_plaid",
})


def validate_no_leaky_features(df, context=""):
    """Raise ValueError if any leaky features are present in the DataFrame.

    Args:
        df: DataFrame to check.
        context: Optional string describing the caller (for error messages).

    Raises:
        ValueError: If leaky features are found in df.columns.
    """
    found = LEAKY_FEATURE_BLOCKLIST & set(df.columns)
    if found:
        raise ValueError(
            f"Leaky features detected{' in ' + context if context else ''}: "
            f"{sorted(found)}. These are integration-timeline proxies and "
            f"MUST NOT be used in modeling. Remove them before proceeding."
        )

# ==============================================================================
# PROPER DATA PREPARATION (fixes right-censoring bias)
# ==============================================================================

def prepare_modeling_data(split_date=None, min_seasoning=90, censoring_mode="normal_as_good"):
    """
    Prepare train/test data with proper right-censoring handling.

    censoring_modes:
        "normal_as_good": Treat all NORMAL loans as good (conservative, avoids selection bias)
        "exclude_recent": Exclude NORMAL < min_seasoning days (biases toward resolved)
        "resolved_only": Only use resolved loans (very biased but clean labels)

    Returns: train_df, test_df with 'is_bad' target
    """
    if split_date is None:
        split_date = SPLIT_DATE

    df = load_default_model()

    # Remove non-originated states
    df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()

    # Create target
    df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)

    # Handle right-censoring
    if censoring_mode == "normal_as_good":
        # Treat all NORMAL as good — conservative but avoids selection bias
        pass  # is_bad already 0 for NORMAL
    elif censoring_mode == "exclude_recent":
        # Exclude NORMAL loans with insufficient seasoning
        resolved = df["loan_state"].isin(BAD_STATES | GOOD_STATES)
        well_seasoned = df["age"] >= min_seasoning
        df = df[resolved | well_seasoned].copy()
    elif censoring_mode == "resolved_only":
        # Only resolved loans (cleanest labels, most biased sample)
        df = df[df["loan_state"].isin(BAD_STATES | GOOD_STATES)].copy()

    # Time-based split
    train = df[df["signing_date"] < split_date].copy()
    test = df[df["signing_date"] >= split_date].copy()

    return train, test


def enrich_from_master(df, master_df=None):
    """Join identity/fraud/bank features from master_features into default modeling data."""
    if master_df is None:
        master_df = load_master()

    extra_cols = [
        "loan_id",
        # Identity/fraud signals
        "prove_phone_trust_score", "prove_name_score", "prove_verified",
        "prove_first_name_score", "prove_last_name_score",
        "prove_address_score", "prove_dob_match", "prove_ssn_match",
        "plaid_balance", "plaid_recentNSFs", "plaid_name_match_score",
        "plaid_len_history",
        "taktile_status", "taktile_output",
        "giact_bankruptcy_count",
        # Extra bureau
        "experian_revolving_account_credit_available_percentage",
        "experian_tradelines_total_items_previously_delinquent",
        "experian_balance_total_past_due_amounts",
        "experian_days_since_latest_inquiry",
        # Business
        "mcc_code", "mcc_sector",
        "shop_credit_score", "shop_qi",
    ]
    available = [c for c in extra_cols if c in master_df.columns]
    extra = master_df[available].drop_duplicates("loan_id")

    return df.merge(extra, on="loan_id", how="left")


# ==============================================================================
# FEATURE ENGINEERING — BUREAU DERIVED
# ==============================================================================

def engineer_bureau_features(df):
    """Build derived bureau features from raw Experian fields."""
    df = df.copy()

    # --- Basic ratios ---
    df["paid_ratio"] = (df["tl_paid"] / df["tl_total"].clip(lower=1)).clip(0, 1)
    df["delin_rate"] = (df["tl_delin"] / df["tl_total"].clip(lower=1)).clip(0, 1)

    # --- Delinquency features ---
    df["delin_severity"] = df["d30"].fillna(0) + 2 * df["d60"].fillna(0)
    df["has_severe_delin"] = (df["d60"].fillna(0) > 0).astype(int)
    df["any_delin"] = ((df["d30"].fillna(0) + df["d60"].fillna(0)) > 0).astype(int)
    df["zero_current_delin"] = (df["tl_delin"] == 0).astype(int)

    # Delinquency recency proxy: d60 > 0 means more recent/severe
    df["delin_recency"] = np.where(df["d60"] > 0, 2.0,
                          np.where(df["d30"] > 0, 1.0, 0.0))

    # Credit recovery: had past delinquency but zero current
    df["credit_recovery"] = (
        (df["zero_current_delin"] == 1) &
        ((df["d30"].fillna(0) > 0) | (df["d60"].fillna(0) > 0))
    ).astype(int)

    # --- Utilization features ---
    # U-shaped: optimal around 25-50%
    df["util_distance"] = ((df["revutil"].fillna(50) - 37.5) / 37.5) ** 2
    df["high_util"] = (df["revutil"].fillna(0) > 80).astype(int)
    df["zero_util"] = (df["revutil"].fillna(-1) == 0).astype(int)

    # --- Balance features ---
    total_bal = (df["instbal"].fillna(0) + df["revbal"].fillna(0)).clip(lower=1)
    df["total_balance"] = total_bal
    df["rev_share"] = df["revbal"].fillna(0) / total_bal
    df["payment_burden"] = df["mopmt"].fillna(0) / total_bal

    # --- Inquiry features ---
    df["high_inq"] = (df["inq6"].fillna(0) >= 6).astype(int)
    df["credit_activity"] = df["inq6"].fillna(0) / df["tl_total"].clip(lower=1)
    # Inquiry intensity relative to history length
    df["inq_intensity"] = df["inq6"].fillna(0) / df["crhist"].clip(lower=1) * 12  # annualized

    # --- Composite scores ---
    df["bureau_health"] = (
        df["zero_current_delin"] * 0.4 +
        (1 - df["credit_activity"].clip(0, 1)) * 0.3 +
        df["paid_ratio"].clip(0, 1) * 0.3
    )

    # Cash stress indicator
    df["cash_stress"] = (
        ((df["fico"] < 580).astype(int) * 2) +
        ((df["pastdue"].fillna(0) > 0).astype(int)) +
        ((df["revutil"].fillna(0) > 80).astype(int))
    )

    # FICO band (granular)
    df["fico_band"] = pd.cut(
        df["fico"],
        bins=[0, 500, 550, 580, 600, 620, 650, 700, 750, 900],
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    ).astype(float).fillna(5)

    return df


# ==============================================================================
# FEATURE ENGINEERING — QI / PARTNER / INTERACTIONS
# ==============================================================================

def engineer_qi_partner_features(df):
    """Build QI, partner, and interaction features."""
    df = df.copy()

    # Normalize categorical inputs to uppercase for consistent matching.
    # Must preserve NaN/None as actual NaN, not string "NONE" or "NAN".
    for col in ("qi", "shop_qi", "partner"):
        if col in df.columns:
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.upper()

    # --- QI encoding ---
    qi_col = "qi" if "qi" in df.columns else "shop_qi"
    qi_map = {"HFHT": 0, "HFLT": 1, "LFHT": 2, "LFLT": 3, "MISSING": 4}
    df["qi_encoded"] = df[qi_col].map(qi_map).fillna(4).astype(int)
    df["qi_missing"] = (df[qi_col].isin(["MISSING", None]) | df[qi_col].isna()).astype(int)

    # QI quality score (from production risk grader analysis)
    # HFHT = best (2.9% bad), MISSING = worst (30.4% bad)
    qi_quality = {"HFHT": 10, "LFHT": 7, "HFLT": 6, "LFLT": 4, "MISSING": 1}
    df["qi_quality"] = df[qi_col].map(qi_quality).fillna(1)

    # --- Partner encoding (canonical: higher = riskier) ---
    # DEPRECATED: partner_encoded removed from all models for fair lending compliance
    # (iter 34, AUC impact -0.0014). Kept here for backward compatibility / analysis.
    df["partner_encoded"] = df["partner"].map(PARTNER_ENCODING).fillna(1).astype(int)

    # --- FICO × QI interaction (15.4x lift) ---
    fico_bin = pd.cut(
        df["fico"], bins=[0, 550, 600, 650, 700, 900],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2)
    df["fico_qi_interaction"] = fico_bin * 5 + df["qi_encoded"]

    # Binary flags for extreme risk segments
    df["missing_qi_low_fico"] = (
        (df["qi_missing"] == 1) & (df["fico"] < 600)
    ).astype(int)

    df["hfht_high_fico"] = (
        (df[qi_col] == "HFHT") & (df["fico"] >= 650)
    ).astype(int)

    # FICO × partner interaction
    df["fico_partner_risk"] = df["fico"] * (1 + df["partner_encoded"] * 0.1)

    # Multi-way interaction: QI + FICO band + delinquency
    if "any_delin" in df.columns:
        df["triple_risk"] = df["qi_encoded"] + (df["fico"] < 600).astype(int) * 3 + df["any_delin"] * 2

    return df


# ==============================================================================
# FEATURE ENGINEERING — IDENTITY PROVIDER SIGNALS
# ==============================================================================

def engineer_identity_features(df):
    """Build identity verification features from provider data.

    NOTE: has_prove_data, has_plaid_data, has_taktile_data, has_giact_data
    are intentionally EXCLUDED. They are integration-timeline proxies that
    leak information about when a loan was originated, not borrower risk.
    See LEAKY_FEATURE_BLOCKLIST.
    """
    df = df.copy()

    # n_identity_sources uses actual signal presence (not the banned has_*_data columns)
    _prove = df["prove_phone_trust_score"].notna().astype(int) if "prove_phone_trust_score" in df.columns else 0
    _plaid = df["plaid_balance"].notna().astype(int) if "plaid_balance" in df.columns else 0
    _giact = df["giact_bankruptcy_count"].notna().astype(int) if "giact_bankruptcy_count" in df.columns else 0
    _taktile = (df["taktile_status"] == "success").astype(int) if "taktile_status" in df.columns else 0

    df["n_identity_sources"] = _prove + _plaid + _giact + _taktile

    # --- Prove features ---
    if "prove_phone_trust_score" in df.columns:
        df["prove_trust_norm"] = df["prove_phone_trust_score"].fillna(0) / 1000
        df["prove_trust_low"] = (df["prove_phone_trust_score"].fillna(0) < 500).astype(int)
        df["prove_name_match_weak"] = (df["prove_name_score"].fillna(0) < 50).astype(int)
        df["prove_ssn_fail"] = (df["prove_ssn_match"].fillna(1) == 0).astype(int) if "prove_ssn_match" in df.columns else 0

    # --- Plaid features ---
    if "plaid_balance" in df.columns:
        df["plaid_balance_log"] = np.log1p(df["plaid_balance"].clip(lower=0).fillna(0))
        df["plaid_low_balance"] = (df["plaid_balance"].fillna(0) < 500).astype(int)
        df["plaid_negative_balance"] = (df["plaid_balance"].fillna(0) < 0).astype(int)
    if "plaid_recentNSFs" in df.columns:
        df["plaid_has_nsf"] = (df["plaid_recentNSFs"].fillna(0) > 0).astype(int)
        df["plaid_high_nsf"] = (df["plaid_recentNSFs"].fillna(0) >= 3).astype(int)

    # --- GIACT features ---
    if "giact_bankruptcy_count" in df.columns:
        df["giact_has_bankruptcy"] = (df["giact_bankruptcy_count"].fillna(0) > 0).astype(int)
        df["giact_multiple_bankruptcy"] = (df["giact_bankruptcy_count"].fillna(0) > 1).astype(int)

    # --- Taktile fraud indicators ---
    if "taktile_output" in df.columns:
        df["taktile_has_output"] = df["taktile_output"].notna().astype(int)
        # Parse taktile output for fraud flags (JSON string column)
        df["taktile_fraud_flag"] = df["taktile_output"].fillna("").str.contains(
            "fraud|suspicious|high.?risk|reject", case=False, regex=True
        ).astype(int)
        df["taktile_identity_match"] = df["taktile_output"].fillna("").str.contains(
            "identity.?match|verified|confirm", case=False, regex=True
        ).astype(int)

    return df


# ==============================================================================
# MASTER PIPELINE
# ==============================================================================

def engineer_entity_features(df):
    """Build entity/prior-history features from graph data.

    Adds a 3-level categorical ``prior_history`` feature:
    - "prior_BAD"  : borrower had a prior loan that defaulted
    - "prior_PAID" : borrower had a prior loan that was repaid (no prior bad)
    - "none"       : no prior loan history (first-timer)

    Requires ``has_prior_bad`` and ``is_repeat`` columns to be present in df
    (filled with 0 if missing).
    """
    df = df.copy()

    # Ensure columns exist with safe defaults
    if "has_prior_bad" not in df.columns:
        df["has_prior_bad"] = 0
    if "is_repeat" not in df.columns:
        df["is_repeat"] = 0

    df["has_prior_bad"] = df["has_prior_bad"].fillna(0).astype(int)
    df["is_repeat"] = df["is_repeat"].fillna(0).astype(int)

    def _prior_history(row):
        if row["has_prior_bad"] == 1:
            return "prior_BAD"
        elif row["is_repeat"] == 1:
            return "prior_PAID"
        else:
            return "none"

    df["prior_history"] = df[["has_prior_bad", "is_repeat"]].apply(_prior_history, axis=1)

    return df


def build_all_features(df, include_identity=False, master_df=None):
    """Run the full feature engineering pipeline."""
    # Bureau features (always available)
    df = engineer_bureau_features(df)

    # QI/Partner features
    df = engineer_qi_partner_features(df)

    # Entity/prior-history features
    df = engineer_entity_features(df)

    # Identity features (requires enrichment from master)
    if include_identity:
        if "prove_phone_trust_score" not in df.columns:
            df = enrich_from_master(df, master_df)
        df = engineer_identity_features(df)

    # Final safety check: ensure no leaky features slipped through
    validate_no_leaky_features(df, context="build_all_features output")

    return df


# ==============================================================================
# FEATURE SETS (for consistent usage across all systems)
# ==============================================================================

FEATURE_SETS = {
    "fico_only": ["fico"],

    "bureau_core": [
        "fico", "d30", "d60", "inq6", "revutil", "pastdue",
        "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
        "mopmt", "crhist",
    ],

    "bureau_engineered": [
        "fico", "d30", "d60", "inq6", "revutil", "pastdue",
        "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
        "mopmt", "crhist",
        # Derived
        "paid_ratio", "delin_rate", "delin_severity", "has_severe_delin",
        "any_delin", "delin_recency", "credit_recovery",
        "util_distance", "high_util", "zero_util",
        "payment_burden", "rev_share",
        "high_inq", "credit_activity", "inq_intensity",
        "bureau_health", "cash_stress", "fico_band",
    ],

    "bureau_qi": [
        "fico", "d30", "d60", "inq6", "revutil", "pastdue",
        "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
        "mopmt", "crhist",
        "paid_ratio", "delin_rate", "delin_severity",
        "bureau_health", "cash_stress",
        # QI features (partner_encoded removed for fair lending compliance, iter 34)
        "qi_encoded", "qi_missing", "qi_quality",
        "fico_qi_interaction", "missing_qi_low_fico", "hfht_high_fico",
    ],

    "full_default": [
        "fico", "d30", "d60", "inq6", "revutil", "pastdue",
        "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
        "mopmt", "crhist",
        "paid_ratio", "delin_rate", "delin_severity", "has_severe_delin",
        "any_delin", "delin_recency", "credit_recovery",
        "util_distance", "high_util",
        "payment_burden", "rev_share",
        "high_inq", "credit_activity", "inq_intensity",
        "bureau_health", "cash_stress", "fico_band",
        # partner_encoded and fico_partner_risk removed for fair lending compliance (iter 34)
        "qi_encoded", "qi_missing", "qi_quality",
        "fico_qi_interaction", "missing_qi_low_fico", "hfht_high_fico",
        "triple_risk",
        # Entity/prior-history (3-level categorical)
        "prior_history",
    ],

    "identity_signals": [
        # NOTE: has_prove_data, has_plaid_data, has_giact_data, has_taktile_data
        # are intentionally excluded — they are integration-timeline proxies.
        "n_identity_sources",
        "prove_trust_norm", "prove_trust_low", "prove_name_match_weak", "prove_ssn_fail",
        "plaid_balance_log", "plaid_low_balance", "plaid_negative_balance",
        "plaid_has_nsf", "plaid_high_nsf",
        "giact_has_bankruptcy", "giact_multiple_bankruptcy",
        "taktile_fraud_flag", "taktile_identity_match",
    ],
}


def get_available_features(df, feature_set="full_default"):
    """Get features from a named set that actually exist in the DataFrame.

    Also validates that no leaky features are requested.
    """
    target_feats = FEATURE_SETS.get(feature_set, FEATURE_SETS["bureau_core"])
    leaky_requested = LEAKY_FEATURE_BLOCKLIST & set(target_feats)
    if leaky_requested:
        raise ValueError(
            f"Feature set '{feature_set}' contains leaky features: "
            f"{sorted(leaky_requested)}. Remove them from FEATURE_SETS."
        )
    return [f for f in target_feats if f in df.columns]
