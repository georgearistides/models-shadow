#!/usr/bin/env python3
"""
train_all.py — Retrain all models with fixed code and re-serialize.

Trains FraudGate, DefaultScorecard, CreditGrader, and the full Pipeline
from data/master_features.parquet, evaluates on the temporal test set
(signing_date >= 2025-07-01), and saves serialized models to models/.

Usage:
    python3 scripts/models/train_all.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging before importing model modules (they use logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_all")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Add project root to path so imports work
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.fraud_gate import FraudGate
from scripts.models.default_scorecard import DefaultScorecard
from scripts.models.credit_grader import CreditGrader
from scripts.models.pipeline import Pipeline
from scripts.models.model_utils import (
    BAD_STATES, GOOD_STATES, EXCLUDE_STATES, SPLIT_DATE, PARTNER_ENCODING,
)
from scripts.models.feature_engine import validate_no_leaky_features


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load master_features.parquet and prepare train/test splits.

    Returns:
        train_df, test_df — both with 'is_bad' target column.
    """
    parquet_path = DATA_DIR / "master_features.parquet"
    if not parquet_path.exists():
        logger.error(f"Data file not found: {parquet_path}")
        logger.error("Run '/pull-data' first to download from Databricks.")
        sys.exit(1)

    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")

    # Load entity graph features if available
    entity_path = DATA_DIR / "entity_graph_cross.parquet"
    if entity_path.exists():
        logger.info(f"Loading entity graph from {entity_path}")
        entity_df = pd.read_parquet(entity_path)
        entity_cols = ['application_id', 'has_prior_bad', 'is_repeat', 'is_cross_entity',
                       'prior_loans_ssn', 'prior_shops_ssn', 'prior_bad_ssn', 'prior_paid_ssn']
        entity_feats = entity_df[[c for c in entity_cols if c in entity_df.columns]].copy()

        # Join on application_id
        before_cols = len(df.columns)
        df = df.merge(entity_feats, on='application_id', how='left')

        # Fill missing entity features with 0 (no prior history known)
        for col in entity_feats.columns:
            if col != 'application_id' and col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        logger.info(f"Joined entity graph: {before_cols} -> {len(df.columns)} columns, "
                    f"{entity_feats['has_prior_bad'].sum() if 'has_prior_bad' in entity_feats.columns else 0} loans with prior_bad")
    else:
        logger.warning(f"Entity graph not found at {entity_path}. Training without entity features.")

    # Load banking velocity features if available
    banking_path = DATA_DIR / "banking_velocity_features.parquet"
    if banking_path.exists():
        logger.info(f"Loading banking features from {banking_path}")
        banking_df = pd.read_parquet(banking_path)
        banking_cols = ['revision_id', 'avg_balance_30d', 'negative_balance_days_90d', 'nsf_count_90d']
        banking_feats = banking_df[[c for c in banking_cols if c in banking_df.columns]].copy()

        # Convert string columns to numeric (parquet may store as object/string)
        for col in banking_feats.columns:
            if col != 'revision_id' and banking_feats[col].dtype == object:
                banking_feats[col] = pd.to_numeric(banking_feats[col], errors='coerce')

        # Join on revision_id
        before_cols = len(df.columns)
        df = df.merge(banking_feats, on='revision_id', how='left')
        n_joined = df['avg_balance_30d'].notna().sum() if 'avg_balance_30d' in df.columns else 0
        logger.info(f"Joined banking features: {before_cols} -> {len(df.columns)} columns, "
                    f"{n_joined} loans with avg_balance_30d ({n_joined/len(df):.1%} coverage)")
    else:
        logger.warning(f"Banking features not found at {banking_path}. Training without banking features.")

    # Filter out excluded loan states
    if "loan_state" in df.columns:
        before = len(df)
        df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
        logger.info(f"Filtered loan states: {before} -> {len(df)} rows "
                    f"(removed {before - len(df)} excluded states)")

    # Create target variable
    if "is_bad" not in df.columns:
        if "loan_state" not in df.columns:
            logger.error("Neither 'is_bad' nor 'loan_state' found in data.")
            sys.exit(1)
        df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)
        logger.info(f"Created 'is_bad' target: {df['is_bad'].mean():.3%} bad rate")
    else:
        logger.info(f"Using existing 'is_bad' column: {df['is_bad'].mean():.3%} bad rate")

    # Ensure signing_date is datetime
    if "signing_date" in df.columns:
        df["signing_date"] = pd.to_datetime(df["signing_date"])
    else:
        logger.error("'signing_date' column not found — cannot do temporal split.")
        sys.exit(1)

    # Map column names for consistency (FraudGate and DefaultScorecard handle
    # this internally, but we need 'qi' and 'partner' available at the top level)
    if "shop_qi" in df.columns and "qi" not in df.columns:
        df["qi"] = df["shop_qi"]
    if "experian_FICO_SCORE" in df.columns and "fico" not in df.columns:
        df["fico"] = df["experian_FICO_SCORE"]

    # Safety check: no leaky features
    validate_no_leaky_features(df, context="loaded data")

    # Time-based split
    train = df[df["signing_date"] < SPLIT_DATE].copy()
    test = df[df["signing_date"] >= SPLIT_DATE].copy()

    logger.info(f"Train: {len(train)} rows, bad rate = {train['is_bad'].mean():.3%}")
    logger.info(f"Test:  {len(test)} rows, bad rate = {test['is_bad'].mean():.3%}")

    return train, test


# =============================================================================
# TRAINING
# =============================================================================

def train_fraud_gate(train_df, test_df):
    """Train and evaluate FraudGate."""
    logger.info("=" * 60)
    logger.info("TRAINING FRAUD GATE")
    logger.info("=" * 60)

    fg = FraudGate(threshold_profile="moderate")
    fg.fit(train_df)

    # Build entity graph lookup if data available
    entity_path = DATA_DIR / "entity_graph_cross.parquet"
    if entity_path.exists():
        entity_df = pd.read_parquet(entity_path)
        graph_lookup = FraudGate.build_graph_lookup(entity_df, key_col='application_id')
        fg.set_graph_lookup(graph_lookup, key_col='application_id')
        logger.info(f"Loaded entity graph lookup: {len(graph_lookup)} entries")

    # Evaluate on test set
    metrics = fg.evaluate(test_df, y_col="is_bad")

    logger.info(f"FraudGate Test AUC:  {metrics['auc']:.4f}")
    logger.info(f"  Gini:              {metrics['gini']:.4f}")
    logger.info(f"  KS:                {metrics['ks']:.4f}")
    logger.info(f"  Top decile bad%:   {metrics['top_decile_bad_rate']:.2%}")
    logger.info(f"  Bot decile bad%:   {metrics['bot_decile_bad_rate']:.2%}")
    logger.info(f"  Lift@top:          {metrics['lift_top_decile']}x")

    # Decision distribution
    for dec, stats in metrics["decision_stats"].items():
        logger.info(f"  {dec}: {stats['count']} ({stats['pct']}%) "
                    f"bad_rate={stats['bad_rate']:.2%}")

    # Save
    save_path = MODELS_DIR / "fraud_gate.joblib"
    fg.save(save_path)
    logger.info(f"Saved FraudGate to {save_path}")

    return fg, metrics


def train_default_scorecard(train_df, test_df):
    """Train and evaluate DefaultScorecard."""
    logger.info("=" * 60)
    logger.info("TRAINING DEFAULT SCORECARD")
    logger.info("=" * 60)

    ds = DefaultScorecard()
    ds.fit(train_df, y_col="is_bad")

    # Evaluate on test set
    metrics = ds.evaluate(test_df, y_col="is_bad")

    logger.info(f"DefaultScorecard Test AUC (blend): {metrics['auc']:.4f}")
    logger.info(f"  Gini:  {metrics['gini']:.4f}")
    logger.info(f"  KS:    {metrics['ks']:.4f}")
    logger.info(f"  Brier: {metrics['brier']:.4f}")

    # Component AUCs
    for comp, auc_val in metrics["component_aucs"].items():
        logger.info(f"  {comp} AUC: {auc_val}")

    # Save
    save_path = MODELS_DIR / "default_scorecard.joblib"
    ds.save(str(save_path))
    logger.info(f"Saved DefaultScorecard to {save_path}")

    return ds, metrics


def train_credit_grader(train_df, test_df, default_scorecard):
    """Train and evaluate CreditGrader using PDs from the DefaultScorecard.

    Trains two graders:
      1. optimal_iv (primary): boundaries that maximize Information Value
      2. 5grade (fallback): equal-bad-rate quantile boundaries
    """
    logger.info("=" * 60)
    logger.info("TRAINING CREDIT GRADER")
    logger.info("=" * 60)

    # Get calibrated PDs from the DefaultScorecard for the training set
    train_pds = default_scorecard.predict_batch(train_df)["pd"].values
    train_labels = train_df["is_bad"].astype(int).values

    # --- Primary: Optimal-IV grader ---
    logger.info("--- Optimal-IV Grader (primary) ---")
    cg = CreditGrader(scheme="optimal_iv")
    cg.fit(train_pds, y_true=train_labels)

    logger.info(f"Grade boundaries: {cg.boundaries}")
    logger.info(f"Grade labels: {cg.grades}")
    if cg._iv_score is not None:
        logger.info(f"Total IV: {cg._iv_score:.4f}")

    # Evaluate on test set
    test_pds = default_scorecard.predict_batch(test_df)["pd"].values
    partners = test_df["partner"].values if "partner" in test_df.columns else None
    metrics = cg.evaluate(test_pds, test_df["is_bad"].astype(int).values, partners=partners)

    logger.info(f"CreditGrader AUC: {metrics['auc']}")
    logger.info(f"  Monotonic: {metrics['monotonic']}")
    for grade, rate in metrics["grade_bad_rates"].items():
        count = metrics["grade_counts"].get(grade, 0)
        bads = metrics["grade_bads"].get(grade, 0)
        logger.info(f"  {grade}: n={count}, bads={bads}, bad_rate={rate:.2%}")

    # Save primary grader
    save_path = MODELS_DIR / "credit_grader.joblib"
    cg.save(save_path)
    logger.info(f"Saved Optimal-IV CreditGrader to {save_path}")

    # --- Fallback: 5grade quantile grader ---
    logger.info("--- 5grade Quantile Grader (fallback) ---")
    cg_fallback = CreditGrader(scheme="5grade")
    cg_fallback.fit(train_pds)

    fb_metrics = cg_fallback.evaluate(test_pds, test_df["is_bad"].astype(int).values, partners=partners)
    logger.info(f"Fallback 5grade AUC: {fb_metrics['auc']}")
    logger.info(f"  Monotonic: {fb_metrics['monotonic']}")
    for grade, rate in fb_metrics["grade_bad_rates"].items():
        count = fb_metrics["grade_counts"].get(grade, 0)
        logger.info(f"  {grade}: n={count}, bad_rate={rate:.2%}")

    save_path_fb = MODELS_DIR / "credit_grader_5grade.joblib"
    cg_fallback.save(save_path_fb)
    logger.info(f"Saved 5grade CreditGrader to {save_path_fb}")

    return cg, metrics


def train_pipeline(train_df, test_df):
    """Train the full Pipeline (FraudGate + DefaultScorecard + CreditGrader)."""
    logger.info("=" * 60)
    logger.info("TRAINING FULL PIPELINE")
    logger.info("=" * 60)

    pipeline = Pipeline(
        fraud_threshold_profile="moderate",
        grading_scheme="optimal_iv",
    )
    pipeline.fit(train_df, y_col="is_bad")

    # Load entity graph into Pipeline's FraudGate (same as standalone)
    entity_path = DATA_DIR / "entity_graph_cross.parquet"
    if entity_path.exists():
        entity_df = pd.read_parquet(entity_path)
        graph_lookup = FraudGate.build_graph_lookup(entity_df, key_col='application_id')
        pipeline.fraud_gate.set_graph_lookup(graph_lookup, key_col='application_id')
        logger.info(f"Loaded entity graph into Pipeline FraudGate: {len(graph_lookup)} entries")

    # Evaluate on test set
    metrics = pipeline.evaluate(test_df, y_col="is_bad")

    logger.info(f"Pipeline Test Metrics:")
    logger.info(f"  Fraud AUC:   {metrics.get('fraud_auc')}")
    logger.info(f"  Default AUC: {metrics.get('default_auc')}")
    logger.info(f"  Grade AUC:   {metrics.get('grade_auc')}")
    logger.info(f"  Overall AUC: {metrics.get('overall_auc')}")

    # Decision distribution
    logger.info(f"  Decision counts: {metrics.get('decision_counts')}")
    logger.info(f"  Decision bad rates: {metrics.get('decision_bad_rates')}")

    # Grade distribution
    logger.info(f"  Grade counts: {metrics.get('grade_counts')}")
    logger.info(f"  Grade bad rates: {metrics.get('grade_bad_rates')}")

    # Component AUCs
    comp_aucs = metrics.get("component_aucs", {})
    for comp, auc_val in comp_aucs.items():
        logger.info(f"  {comp} AUC: {auc_val}")

    # Save the full pipeline
    pipeline.save(MODELS_DIR)
    logger.info(f"Saved full Pipeline to {MODELS_DIR}/")

    return pipeline, metrics


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_saved_models(test_df):
    """Load serialized models and verify they produce correct predictions."""
    logger.info("=" * 60)
    logger.info("VERIFYING SAVED MODELS")
    logger.info("=" * 60)

    # Load the pipeline
    pipeline = Pipeline.load(MODELS_DIR)
    logger.info(f"Loaded pipeline: {pipeline}")

    # Score a single sample
    sample_row = test_df.iloc[0].to_dict()
    result = pipeline.score(sample_row)
    logger.info(f"Single score result:")
    logger.info(f"  Decision: {result['decision']}")
    logger.info(f"  Fraud score: {result['fraud_score']} ({result['fraud_decision']})")
    logger.info(f"  PD: {result['pd']:.4f}")
    logger.info(f"  Default score: {result['default_score']}")
    logger.info(f"  Grade: {result['grade']}")
    logger.info(f"  Reasons: {result['reasons']}")

    # Batch score a sample
    sample_df = test_df.head(20)
    batch_results = pipeline.score_batch(sample_df)
    logger.info(f"Batch score (20 rows): mean PD={batch_results['pd'].mean():.4f}, "
                f"decisions={batch_results['decision'].value_counts().to_dict()}")

    # Verify individual model loading
    fg = FraudGate.load(MODELS_DIR / "fraud_gate.joblib")
    ds = DefaultScorecard.load(str(MODELS_DIR / "default_scorecard.joblib"))
    cg = CreditGrader.load(MODELS_DIR / "credit_grader.joblib")
    logger.info(f"Individual models loaded successfully:")
    logger.info(f"  FraudGate: {fg}")
    logger.info(f"  DefaultScorecard: {ds}")
    logger.info(f"  CreditGrader: {cg}")

    # Verify AUC matches on full test set
    from sklearn.metrics import roc_auc_score
    full_results = pipeline.score_batch(test_df)
    y_true = test_df["is_bad"].astype(int).values
    reload_auc = roc_auc_score(y_true, full_results["pd"].values)
    fraud_auc = roc_auc_score(y_true, full_results["fraud_score"].values)
    logger.info(f"Verification AUCs (loaded models on full test set):")
    logger.info(f"  Default blend AUC: {reload_auc:.4f}")
    logger.info(f"  Fraud AUC: {fraud_auc:.4f}")

    return pipeline, reload_auc, fraud_auc


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    logger.info("Starting model retraining")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info(f"Models dir: {MODELS_DIR}")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df = load_and_prepare_data()

    # Train individual models
    fg, fg_metrics = train_fraud_gate(train_df, test_df)
    ds, ds_metrics = train_default_scorecard(train_df, test_df)
    cg, cg_metrics = train_credit_grader(train_df, test_df, ds)

    # Train full pipeline (retrains everything internally for consistency)
    pipeline, pipeline_metrics = train_pipeline(train_df, test_df)

    # Verify saved models
    loaded_pipeline, verify_default_auc, verify_fraud_auc = verify_saved_models(test_df)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("")
    logger.info("AUC Summary (Test Set):")
    logger.info(f"  FraudGate:              {fg_metrics['auc']:.4f}")
    logger.info(f"  DefaultScorecard blend: {ds_metrics['auc']:.4f}")
    logger.info(f"    WoE sub-model:        {ds_metrics['component_aucs']['woe_pd']}")
    logger.info(f"    Rule sub-model:       {ds_metrics['component_aucs']['rule_pd']}")
    logger.info(f"    XGB sub-model:        {ds_metrics['component_aucs']['xgb_pd']}")
    logger.info(f"  CreditGrader:           {cg_metrics['auc']}")
    logger.info(f"  Pipeline overall:       {pipeline_metrics.get('overall_auc')}")
    logger.info("")
    logger.info(f"Verification (loaded models): default={verify_default_auc:.4f}, fraud={verify_fraud_auc:.4f}")
    logger.info("")
    logger.info("Saved model files:")
    for p in sorted(MODELS_DIR.glob("*.joblib")):
        size_kb = p.stat().st_size / 1024
        logger.info(f"  {p.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
