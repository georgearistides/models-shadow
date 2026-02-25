#!/usr/bin/env python3
"""
score_portfolio.py — Shadow scoring: score every application in the portfolio,
compare against production decisions, and output a comprehensive comparison.

Uses the unified Pipeline class (FraudGate -> DefaultScorecard -> CreditGrader)
instead of orchestrating individual models.

Usage:
    python3 scripts/score_portfolio.py                       # Use local parquet
    python3 scripts/score_portfolio.py --output results/shadow_scores.csv
    python3 scripts/score_portfolio.py --train-first         # Train models before scoring
    python3 scripts/score_portfolio.py --train-first --output results/shadow_scores.csv
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Project paths (flat layout: all files in same directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

from pipeline import Pipeline
from fraud_gate import FraudGate
from model_utils import BAD_STATES, EXCLUDE_STATES


def load():
    """Load master_features.parquet from the project directory."""
    parquet_path = PROJECT_ROOT / "master_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"master_features.parquet not found at {parquet_path}. "
            "Run pull_data.py first."
        )
    return pd.read_parquet(parquet_path)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BAD_STATES, EXCLUDE_STATES imported from model_utils

MODELS_DIR = PROJECT_ROOT  # flat layout: joblib files alongside code

TIME_SPLIT = "2025-07-01"


# ---------------------------------------------------------------------------
# Slim pipeline support
# ---------------------------------------------------------------------------

def _load_slim_pipeline(models_dir):
    """Load the slim pipeline variant from serialized components.

    The slim pipeline uses DefaultScorecardSlim instead of DefaultScorecard,
    with separately saved FraudGate and CreditGrader. The pipeline_slim.joblib
    is a config dict (not a Pipeline object) that references the component files.

    Returns a Pipeline instance with the slim default scorecard swapped in.
    """
    import joblib
    from fraud_gate import FraudGate
    from default_scorecard_slim import DefaultScorecardSlim
    from credit_grader import CreditGrader

    models_dir = Path(models_dir)
    config_path = models_dir / "pipeline_slim.joblib"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Slim pipeline config not found at {config_path}. "
            "Run scripts/models/train_slim.py first."
        )

    config = joblib.load(config_path)
    components = config["components"]

    # Load sub-models using the filenames stored in config
    fg = FraudGate.load(models_dir / components["fraud_gate"])
    ds = DefaultScorecardSlim.load(str(models_dir / components["default_scorecard"]))
    cg = CreditGrader.load(models_dir / components["credit_grader"])

    # Assemble into a Pipeline object (bypass __init__, mirror Pipeline.load)
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.fraud_gate = fg
    pipeline.default_scorecard = ds
    pipeline.credit_grader = cg
    pipeline.pd_decline = config.get("pd_decline", 0.25)
    pipeline.pd_review = config.get("pd_review", 0.15)
    pipeline.review_grades = set(config.get("review_grades", ["E", "F", "G"]))
    pipeline._fit_timestamp = config.get("fit_timestamp")
    pipeline._train_stats = config.get("train_stats", {})
    pipeline._fitted = True

    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _letter_grade(grade_str):
    """Extract the base letter from a production grade like 'C3' -> 'C'."""
    if pd.isna(grade_str) or grade_str is None:
        return None
    g = str(grade_str).strip()
    if not g:
        return None
    # Production grades are like A1, B2, C3, D, E, F -- take first letter
    return g[0]


# ---------------------------------------------------------------------------
# Pipeline loading / training
# ---------------------------------------------------------------------------

def load_or_train_pipeline(df, train_first=False, slim=False):
    """Load a pre-trained Pipeline from disk, or train fresh if --train-first or missing."""
    if slim:
        print(f"[Pipeline] Loading SLIM pipeline from {MODELS_DIR}")
        return _load_slim_pipeline(MODELS_DIR)

    if not train_first and (MODELS_DIR / "pipeline_config.joblib").exists():
        print(f"[Pipeline] Loading from {MODELS_DIR}")
        return Pipeline.load(str(MODELS_DIR))
    else:
        print("[Pipeline] Training fresh...")
        train = df[df["signing_date"] < TIME_SPLIT].copy()
        train["is_bad"] = train["loan_state"].isin(BAD_STATES).astype(int)
        pipeline = Pipeline()
        pipeline.fit(train, y_col="is_bad")
        pipeline.save(str(MODELS_DIR))
        return pipeline


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_portfolio(df, pipeline):
    """Score every application through the full Pipeline.

    Returns a DataFrame with one row per application and all shadow scores.
    """
    n = len(df)
    print(f"\n[Scoring] Scoring {n:,} applications via Pipeline...")

    # --- Run the entire pipeline in one call ---
    pipeline_results = pipeline.score_batch(df)

    # --- Build output DataFrame ---
    print("  [Build] Assembling results...")

    out = pd.DataFrame(index=df.index)

    # Application identifiers
    out["application_id"] = df["application_id"].values
    out["signing_date"] = df["signing_date"].values
    out["loan_state"] = df["loan_state"].values
    out["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int).values

    # Fraud gate outputs (from Pipeline)
    out["our_fraud_score"] = pipeline_results["fraud_score"].values
    out["our_fraud_normalized"] = pipeline_results["fraud_normalized"].values
    out["our_fraud_decision"] = pipeline_results["fraud_decision"].values

    # Default scorecard outputs (from Pipeline)
    out["our_pd"] = pipeline_results["pd"].values
    out["our_default_score"] = pipeline_results["default_score"].values
    out["our_woe_pd"] = pipeline_results["woe_pd"].values
    out["our_rule_pd"] = pipeline_results["rule_pd"].values
    out["our_xgb_pd"] = pipeline_results["xgb_pd"].values

    # Credit grader outputs (from Pipeline)
    out["our_grade"] = pipeline_results["grade"].values
    out["our_pricing_tier"] = pipeline_results["pricing_tier"].values

    # Pipeline's unified decision (lowercase: approve/review/decline)
    # Capitalize for display consistency with summary output
    out["our_decision"] = pipeline_results["decision"].apply(
        lambda d: d.capitalize() if isinstance(d, str) else d
    ).values

    # Production reference
    out["production_grade"] = df["grade"].values
    out["production_grade_letter"] = df["grade"].apply(_letter_grade).values

    # Comparison flags
    out["agrees_with_production"] = (
        out["our_grade"] == out["production_grade_letter"]
    )
    out["would_have_declined"] = (out["our_decision"] == "Decline")
    out["would_have_caught"] = (
        out["would_have_declined"] & (out["is_bad"] == 1)
    )

    # Context columns
    if "experian_FICO_SCORE" in df.columns:
        out["fico"] = df["experian_FICO_SCORE"].values
    if "shop_qi" in df.columns:
        out["qi"] = df["shop_qi"].values
    if "partner" in df.columns:
        out["partner"] = df["partner"].values

    # Entity graph features (pass through if available)
    entity_feature_cols = ['has_prior_bad', 'is_repeat', 'is_cross_entity',
                           'prior_loans_ssn', 'prior_shops_ssn', 'prior_bad_ssn', 'prior_paid_ssn']
    for col in entity_feature_cols:
        if col in df.columns:
            out[col] = df[col].values

    print(f"  [Done] Scored {n:,} applications.\n")
    return out


# ---------------------------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------------------------

def print_summary(scored_df, df_full):
    """Print comprehensive summary statistics to stdout."""
    n = len(scored_df)
    n_bad = scored_df["is_bad"].sum()
    bad_rate = n_bad / n * 100

    # Time split
    scored_df = scored_df.copy()
    scored_df["signing_date"] = pd.to_datetime(scored_df["signing_date"])
    is_test = scored_df["signing_date"] >= TIME_SPLIT
    test_df = scored_df[is_test]
    train_df = scored_df[~is_test]

    print("=" * 60)
    print("         SHADOW SCORING SUMMARY")
    print("=" * 60)
    print(f"\nPortfolio: {n:,} applications scored")
    print(f"  Train (< {TIME_SPLIT}): {len(train_df):,}")
    print(f"  Test  (>= {TIME_SPLIT}): {len(test_df):,}")
    print(f"  Overall bad rate: {bad_rate:.1f}% ({n_bad:,} bads)")

    # --- Decision Distribution ---
    print(f"\n--- Decision Distribution ---")
    for decision in ["Approve", "Review", "Decline"]:
        mask = scored_df["our_decision"] == decision
        count = mask.sum()
        pct = count / n * 100
        print(f"  {decision:<10} {count:>6,}  ({pct:5.1f}%)")

    # --- Bad Rate by Our Decision ---
    print(f"\n--- Bad Rate by Our Decision ---")
    for decision in ["Approve", "Review", "Decline"]:
        mask = scored_df["our_decision"] == decision
        subset = scored_df[mask]
        if len(subset) > 0:
            br = subset["is_bad"].mean() * 100
            n_bads = subset["is_bad"].sum()
            print(f"  {decision:<10} {br:5.1f}%  ({n_bads:,} bads in {len(subset):,})")
        else:
            print(f"  {decision:<10}   N/A")

    # --- Fraud Gate Decision Breakdown ---
    print(f"\n--- Fraud Gate Decision Breakdown ---")
    for fd in ["pass", "review", "decline"]:
        mask = scored_df["our_fraud_decision"] == fd
        subset = scored_df[mask]
        if len(subset) > 0:
            br = subset["is_bad"].mean() * 100
            print(f"  {fd:<10} {len(subset):>6,}  ({len(subset)/n*100:5.1f}%)  bad rate: {br:.1f}%")

    # --- Comparison vs Production ---
    print(f"\n--- Comparison vs Production ---")
    agree_count = scored_df["agrees_with_production"].sum()
    agree_pct = agree_count / n * 100
    print(f"  Grade agreement rate: {agree_pct:.1f}% ({agree_count:,}/{n:,})")

    # Applications we would decline
    would_decline = scored_df["would_have_declined"].sum()
    would_caught = scored_df["would_have_caught"].sum()
    would_decline_performing = would_decline - would_caught

    print(f"\n  We would DECLINE {would_decline:,} applications that production originated")
    if would_decline > 0:
        print(f"    Of those, {would_caught:,} ({would_caught/would_decline*100:.1f}%) actually defaulted")
        print(f"      -> We would have caught {would_caught:,} defaults")
        print(f"    Of those, {would_decline_performing:,} ({would_decline_performing/would_decline*100:.1f}%) are still performing")
        print(f"      -> False positive rate: {would_decline_performing/would_decline*100:.1f}%")

    # What fraction of all defaults we would have caught
    if n_bad > 0:
        print(f"\n  Default capture: {would_caught:,}/{n_bad:,} = {would_caught/n_bad*100:.1f}% of all defaults")

    # --- Model Performance (Test Set Only) ---
    print(f"\n--- Model Performance (Test Set Only, n={len(test_df):,}) ---")
    if len(test_df) > 0 and test_df["is_bad"].sum() > 0:
        y_test = test_df["is_bad"].values

        # Fraud Gate AUC
        try:
            fraud_auc = roc_auc_score(y_test, test_df["our_fraud_score"].values)
            print(f"  Fraud Gate AUC:       {fraud_auc:.4f}")
        except ValueError:
            print("  Fraud Gate AUC:       N/A (insufficient data)")

        # Default Blend AUC
        try:
            blend_auc = roc_auc_score(y_test, test_df["our_pd"].values)
            print(f"  Default Blend AUC:    {blend_auc:.4f}")
        except ValueError:
            print("  Default Blend AUC:    N/A")

        # Sub-model AUCs
        for name, col in [("WoE", "our_woe_pd"), ("Rule", "our_rule_pd"), ("XGBoost", "our_xgb_pd")]:
            try:
                auc = roc_auc_score(y_test, test_df[col].values)
                print(f"    {name} sub-model:    {auc:.4f}")
            except ValueError:
                print(f"    {name} sub-model:    N/A")

        # Pricing tier AUC (replaces old grade_index AUC)
        try:
            tier_auc = roc_auc_score(y_test, test_df["our_pricing_tier"].values)
            print(f"  Pricing Tier AUC:     {tier_auc:.4f}")
        except ValueError:
            print("  Pricing Tier AUC:     N/A")
    else:
        print("  Insufficient test data for AUC computation.")

    # --- Fraud Gate Flagged Applications ---
    print(f"\n--- Fraud Gate Flagged Applications ---")
    flagged = scored_df[scored_df["our_fraud_decision"].isin(["decline", "review"])]
    if len(flagged) > 0:
        print(f"  {len(flagged):,} applications flagged by Fraud Gate (decline + review)")
        for fd in ["decline", "review"]:
            mask = scored_df["our_fraud_decision"] == fd
            cnt = mask.sum()
            if cnt > 0:
                br = scored_df.loc[mask, "is_bad"].mean() * 100
                print(f"    {fd}: {cnt:,} apps, {br:.1f}% bad rate")
    else:
        print("  No declined/review applications.")

    # --- Grade Migration Matrix ---
    print(f"\n--- Grade Migration Matrix (Production -> Our Grade) ---")
    # Map production grades to letters
    our_grades_sorted = sorted(scored_df["our_grade"].dropna().unique())
    prod_grades_sorted = sorted(scored_df["production_grade_letter"].dropna().unique())

    # Only keep grades that appear in the data
    if our_grades_sorted and prod_grades_sorted:
        cross = pd.crosstab(
            scored_df["production_grade_letter"],
            scored_df["our_grade"],
            margins=False,
        )
        # Reindex to have consistent ordering
        all_grades = ["A", "B", "C", "D", "E", "F", "G"]
        row_grades = [g for g in all_grades if g in cross.index]
        col_grades = [g for g in all_grades if g in cross.columns]
        cross = cross.reindex(index=row_grades, columns=col_grades, fill_value=0)

        # Format header
        header = f"{'Prod':<6}" + "".join(f"Our {g:>3}" for g in col_grades)
        print(f"  {header}")
        print(f"  {'----':<6}" + "".join(f"{'----':>6}" for _ in col_grades))
        for prod_g in row_grades:
            row = f"  {prod_g:<6}"
            for our_g in col_grades:
                row += f"{cross.loc[prod_g, our_g]:>6,}"
            print(row)

    # --- Bad Rate by Our Grade ---
    print(f"\n--- Bad Rate by Our Grade ---")
    for grade in ["A", "B", "C", "D", "E", "F", "G"]:
        mask = scored_df["our_grade"] == grade
        subset = scored_df[mask]
        if len(subset) > 0:
            br = subset["is_bad"].mean() * 100
            print(f"  {grade}: {br:5.1f}% bad rate  ({subset['is_bad'].sum():,} bads in {len(subset):,})")

    # --- Bad Rate by Production Grade ---
    print(f"\n--- Bad Rate by Production Grade (Letter) ---")
    for grade in ["A", "B", "C", "D", "E", "F", "G"]:
        mask = scored_df["production_grade_letter"] == grade
        subset = scored_df[mask]
        if len(subset) > 0:
            br = subset["is_bad"].mean() * 100
            print(f"  {grade}: {br:5.1f}% bad rate  ({subset['is_bad'].sum():,} bads in {len(subset):,})")

    # --- Score Distribution ---
    print(f"\n--- Score Distribution (Default Score 300-850) ---")
    score_stats = scored_df["our_default_score"].describe()
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"  {stat:<6}: {score_stats[stat]:.0f}")

    print("\n" + "=" * 60)
    print("         END OF SUMMARY")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shadow score the portfolio and compare against production decisions."
    )
    parser.add_argument(
        "--output", "-o",
        default="results/shadow_scores.csv",
        help="Path for output CSV (default: results/shadow_scores.csv)",
    )
    parser.add_argument(
        "--train-first",
        action="store_true",
        help="Train pipeline fresh before scoring (overwrites saved models).",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Use the slim pipeline variant (8-feature DefaultScorecardSlim).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # --- Step 1: Load data ---
    print("[Data] Loading master_features from local parquet...")
    df = load()
    print(f"[Data] Loaded {len(df):,} rows x {len(df.columns)} columns")

    # --- Step 1b: Load and join entity graph features ---
    entity_path = PROJECT_ROOT / "entity_graph_cross.parquet"
    if entity_path.exists():
        print(f"[Data] Loading entity graph from {entity_path}")
        entity_df = pd.read_parquet(entity_path)
        entity_cols = ['application_id', 'has_prior_bad', 'is_repeat', 'is_cross_entity',
                       'prior_loans_ssn', 'prior_shops_ssn', 'prior_bad_ssn', 'prior_paid_ssn']
        entity_feats = entity_df[[c for c in entity_cols if c in entity_df.columns]].copy()

        before_cols = len(df.columns)
        df = df.merge(entity_feats, on='application_id', how='left')

        for col in entity_feats.columns:
            if col != 'application_id' and col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        print(f"[Data] Joined entity graph: {before_cols} -> {len(df.columns)} columns, "
              f"{(df['has_prior_bad'] == 1).sum() if 'has_prior_bad' in df.columns else 0} with prior_bad")
    else:
        print(f"[Data] Entity graph not found at {entity_path}. Scoring without entity features.")

    # --- Step 1c: Load and join banking velocity features ---
    banking_path = PROJECT_ROOT / "banking_velocity_features.parquet"
    if banking_path.exists():
        print(f"[Data] Loading banking features from {banking_path}")
        banking_df = pd.read_parquet(banking_path)
        banking_cols = ['revision_id', 'avg_balance_30d', 'negative_balance_days_90d', 'nsf_count_90d']
        banking_feats = banking_df[[c for c in banking_cols if c in banking_df.columns]].copy()
        # Convert string columns to numeric (parquet may store as object/string)
        for col in banking_feats.columns:
            if col != 'revision_id' and banking_feats[col].dtype == object:
                banking_feats[col] = pd.to_numeric(banking_feats[col], errors='coerce')
        before_cols = len(df.columns)
        df = df.merge(banking_feats, on='revision_id', how='left')
        n_joined = df['avg_balance_30d'].notna().sum() if 'avg_balance_30d' in df.columns else 0
        print(f"[Data] Joined banking features: {before_cols} -> {len(df.columns)} columns, "
              f"{n_joined} with avg_balance_30d ({n_joined/len(df):.1%} coverage)")
    else:
        print(f"[Data] Banking features not found at {banking_path}. Scoring without banking features.")

    # --- Step 2: Prepare data ---
    print("[Data] Preparing data...")
    df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
    df["signing_date"] = pd.to_datetime(df["signing_date"])
    print(f"[Data] After excluding {EXCLUDE_STATES}: {len(df):,} rows")

    # Map shop_qi -> qi for models that expect 'qi'
    if "shop_qi" in df.columns and "qi" not in df.columns:
        df["qi"] = df["shop_qi"]

    # --- Step 3: Load or train pipeline ---
    variant = "SLIM" if args.slim else "FULL"
    print(f"\n[Models] Loading/training {variant} pipeline...")
    pipeline = load_or_train_pipeline(df, train_first=args.train_first, slim=args.slim)
    print(f"  Pipeline: {pipeline}")

    # --- Step 3b: Load entity graph lookup into FraudGate ---
    if entity_path.exists() and hasattr(pipeline, 'fraud_gate'):
        entity_df_for_lookup = pd.read_parquet(entity_path)
        graph_lookup = FraudGate.build_graph_lookup(entity_df_for_lookup, key_col='application_id')
        pipeline.fraud_gate.set_graph_lookup(graph_lookup, key_col='application_id')
        print(f"[Models] Loaded entity graph lookup into FraudGate: {len(graph_lookup)} entries")

    # --- Step 4: Score every application ---
    scored_df = score_portfolio(df, pipeline)

    # --- Step 5: Save CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    print(f"[Output] Saved {len(scored_df):,} rows to {output_path}")

    # --- Step 6: Print summary ---
    print_summary(scored_df, df)

    return scored_df


if __name__ == "__main__":
    main()
