"""Shared utilities for credit, fraud, and identity models."""
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Optional
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    precision_recall_curve, classification_report,
    confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
SPLIT_DATE = pd.Timestamp("2025-07-01")
MIN_SEASONING_DAYS = 180  # 6 months for right-censoring


# ---------------------------------------------------------------------------
# Shared value helpers (used by FraudGate, PaymentMonitor, etc.)
# ---------------------------------------------------------------------------

def _is_missing(val: Any) -> bool:
    """Check if a value is missing (None, NaN, empty string, or 'MISSING')."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() in ("", "MISSING"):
        return True
    return False


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float, returning *default* for missing values."""
    if _is_missing(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val: Any) -> Optional[str]:
    """Safely convert to string, returning None for missing values."""
    if _is_missing(val):
        return None
    return str(val)


# --- Bad state definitions ---
BAD_STATES = {"CHARGE_OFF", "DEFAULT", "WORKOUT", "PRE_DEFAULT", "TECHNICAL_DEFAULT"}
GOOD_STATES = {"PAID_CLOSED", "PAID", "OVER_PAID"}
ACTIVE_STATES = {"NORMAL"}
EXCLUDE_STATES = {"REJECTED", "CANCELED", "PENDING_FUNDING", "APPROVED"}

# --- Canonical partner encoding (higher = riskier, based on observed default rates) ---
# HONEYBOOK ~7% bad rate, SPOTON ~11%, PAYSAFE ~13% (riskiest)
PARTNER_ENCODING = {"HONEYBOOK": 0, "SPOTON": 1, "PAYSAFE": 2}


def load_master():
    """Load master features dataset."""
    return pd.read_parquet(DATA_DIR / "master_features.parquet")


def load_default_model():
    """Load pre-filtered default modeling dataset."""
    return pd.read_parquet(DATA_DIR / "default_model.parquet")


def load_funnel():
    """Load application funnel data."""
    return pd.read_parquet(DATA_DIR / "application_funnel.parquet")


def prepare_credit_data(df=None, handle_censoring="exclude_recent"):
    """Prepare data for credit/default modeling.

    Args:
        df: DataFrame (loads default_model.parquet if None)
        handle_censoring: "exclude_recent" drops NORMAL loans with < 6mo seasoning,
                         "include_all" keeps everything (for survival analysis)

    Returns:
        train_df, test_df with 'is_bad' target column added
    """
    if df is None:
        df = load_default_model()

    # Filter to valid loan states
    df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()

    # Create target
    df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)

    # Handle right-censoring
    if handle_censoring == "exclude_recent":
        # Keep: resolved loans (good or bad) + active loans with enough seasoning
        resolved = df["loan_state"].isin(BAD_STATES | GOOD_STATES)
        well_seasoned = df["age"] >= MIN_SEASONING_DAYS
        df = df[resolved | well_seasoned].copy()

    # Time-based split
    train = df[df["signing_date"] < SPLIT_DATE].copy()
    test = df[df["signing_date"] >= SPLIT_DATE].copy()

    return train, test


def get_credit_features(df, feature_set="bureau"):
    """Get feature columns for credit modeling.

    Feature sets:
        "fico_only": Just FICO score
        "bureau": FICO + bureau features (delinquencies, balances, etc.)
        "bureau_qi": Bureau + QI quadrant
        "full": Bureau + QI + partner + derived features
    """
    fico_only = ["fico"]

    bureau = fico_only + [
        "d30", "d60", "inq6", "revutil", "pastdue",
        "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
        "mopmt", "crhist"
    ]

    bureau_qi = bureau + ["qi_encoded", "qi_missing"]

    full = bureau_qi + [
        "partner_encoded", "fico_qi_interaction",
        "paid_ratio", "delin_rate", "delin_severity",
        "payment_burden", "credit_activity"
    ]

    sets = {
        "fico_only": fico_only,
        "bureau": bureau,
        "bureau_qi": bureau_qi,
        "full": full,
    }

    available = [c for c in sets[feature_set] if c in df.columns]
    return available


def engineer_features(df):
    """Add derived features for credit modeling."""
    df = df.copy()

    # QI encoding (ordinal by bad rate: MISSING worst, HFHT best)
    qi_map = {"HFHT": 0, "HFLT": 1, "LFHT": 2, "LFLT": 3, "MISSING": 4}
    df["qi_encoded"] = df["qi"].map(qi_map).fillna(4).astype(int)
    df["qi_missing"] = (df["qi"] == "MISSING").astype(int)

    # Partner encoding (canonical: higher = riskier, based on default rates)
    df["partner_encoded"] = df["partner"].map(PARTNER_ENCODING).fillna(1).astype(int)

    # FICO × QI interaction (the big one — 15.4x lift)
    fico_bin = pd.cut(df["fico"], bins=[0, 550, 600, 650, 700, 900],
                      labels=[0, 1, 2, 3, 4]).astype(float).fillna(0)
    df["fico_qi_interaction"] = fico_bin * 5 + df["qi_encoded"]

    # Bureau-derived ratios
    df["paid_ratio"] = (df["tl_paid"] / df["tl_total"].clip(lower=1)).clip(0, 1)
    df["delin_rate"] = (df["tl_delin"] / df["tl_total"].clip(lower=1)).clip(0, 1)
    df["delin_severity"] = df["d30"] + 2 * df["d60"]  # weighted delinquency

    # Payment burden (monthly payment / total balance proxy)
    total_bal = (df["instbal"].fillna(0) + df["revbal"].fillna(0)).clip(lower=1)
    df["payment_burden"] = df["mopmt"].fillna(0) / total_bal

    # Credit activity (recent inquiries relative to tradelines)
    df["credit_activity"] = df["inq6"] / df["tl_total"].clip(lower=1)

    return df


def load_and_split(split_date=None):
    """Convenience wrapper: load master_features + entity graph, create is_bad, do temporal split.

    Returns train_df, test_df with 'is_bad' target column.
    This is the canonical data loading function for DefaultScorecard training.
    """
    from pathlib import Path
    entity_path = DATA_DIR / "entity_graph_cross.parquet"

    df = load_master()

    # Join entity graph features if available
    if entity_path.exists():
        entity_df = pd.read_parquet(entity_path)
        entity_cols = ['application_id', 'has_prior_bad', 'is_repeat', 'is_cross_entity',
                       'prior_loans_ssn', 'prior_shops_ssn', 'prior_bad_ssn', 'prior_paid_ssn']
        entity_feats = entity_df[[c for c in entity_cols if c in entity_df.columns]].copy()
        if "has_prior_bad" not in df.columns:
            df = df.merge(entity_feats, on='application_id', how='left')
            for col in entity_feats.columns:
                if col != 'application_id' and col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)

    # Filter out excluded states
    if "loan_state" in df.columns:
        df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()

    # Create target
    if "is_bad" not in df.columns:
        df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)

    # Rename shop_qi -> qi and experian_FICO_SCORE -> fico for consistency
    if "shop_qi" in df.columns and "qi" not in df.columns:
        df["qi"] = df["shop_qi"]
    if "experian_FICO_SCORE" in df.columns and "fico" not in df.columns:
        df["fico"] = df["experian_FICO_SCORE"]

    # Temporal split
    if "signing_date" in df.columns:
        df["signing_date"] = pd.to_datetime(df["signing_date"])
    cutoff = pd.Timestamp(split_date) if split_date else SPLIT_DATE
    train = df[df["signing_date"] < cutoff].copy()
    test = df[df["signing_date"] >= cutoff].copy()

    return train, test


def evaluate_model(y_true, y_prob, model_name="Model", threshold=0.5):
    """Comprehensive model evaluation. Returns dict of metrics."""
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1

    # Brier score only valid for probabilities in [0, 1]
    y_prob_arr = np.asarray(y_prob)
    if y_prob_arr.min() >= 0 and y_prob_arr.max() <= 1:
        brier = brier_score_loss(y_true, y_prob_arr)
    else:
        brier = np.nan

    # KS statistic
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)

    # Lift in top decile
    df_eval = pd.DataFrame({"y": y_true, "p": y_prob})
    df_eval["decile"] = pd.qcut(df_eval["p"], 10, labels=False, duplicates="drop")
    top_decile_rate = df_eval[df_eval["decile"] == df_eval["decile"].max()]["y"].mean()
    base_rate = y_true.mean()
    lift_top = top_decile_rate / base_rate if base_rate > 0 else 0

    # Lift in bottom decile (safest)
    bot_decile_rate = df_eval[df_eval["decile"] == df_eval["decile"].min()]["y"].mean()

    metrics = {
        "model": model_name,
        "auc": round(auc, 4),
        "gini": round(gini, 4),
        "ks": round(ks, 4),
        "brier": round(brier, 4),
        "lift_top_decile": round(lift_top, 2),
        "top_decile_bad_rate": round(top_decile_rate, 4),
        "bot_decile_bad_rate": round(bot_decile_rate, 4),
        "base_rate": round(base_rate, 4),
        "n_obs": len(y_true),
        "n_bad": int(y_true.sum()),
    }
    return metrics


def metrics_to_markdown_row(metrics):
    """Convert metrics dict to markdown table row."""
    return (
        f"| {metrics['model']} | {metrics['auc']} | {metrics['gini']} | "
        f"{metrics['ks']} | {metrics['brier']} | {metrics['lift_top_decile']}x | "
        f"{metrics['top_decile_bad_rate']:.1%} | {metrics['bot_decile_bad_rate']:.1%} | "
        f"{metrics['n_obs']:,} | {metrics['n_bad']:,} |"
    )


def print_metrics_table(results_list):
    """Print a formatted metrics comparison table."""
    header = (
        "| Model | AUC | Gini | KS | Brier | Lift@Top | Top Bad% | Bot Bad% | N | Bads |"
    )
    sep = "|-------|-----|------|-----|-------|----------|----------|----------|---|------|"
    print(header)
    print(sep)
    for m in sorted(results_list, key=lambda x: x["auc"], reverse=True):
        print(metrics_to_markdown_row(m))
