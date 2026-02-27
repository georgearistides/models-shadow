#!/usr/bin/env python3
"""
train_v6.py — Train the simplified WoE model and validate.
============================================================
Produces: woe_model.joblib, config.joblib
Validates: AUC >= 0.64, grade monotonicity, calibration

Run: python3 train_v6.py
"""

import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
sys.path.insert(0, str(MODEL_DIR))

from model_utils import load_and_split, BAD_STATES
from feature_engine import build_all_features
from default_model import DefaultModel
from grader import Grader, DEFAULT_BOUNDARIES

# Column aliases: long Experian names → short names used by feature_engine
COLUMN_ALIASES = {
    "experian_delinquencies_thirty_day_count": "d30",
    "experian_delinquencies_sixty_day_count": "d60",
    "experian_inquiries_last_six_months": "inq6",
    "experian_revolving_account_credit_available_percentage": "revutil",
    "experian_balance_total_past_due_amounts": "pastdue",
    "experian_balance_total_installment_accounts": "instbal",
    "experian_balance_total_revolving_accounts": "revbal",
    "experian_tradelines_total_items": "tl_total",
    "experian_tradelines_total_items_paid": "tl_paid",
    "experian_tradelines_total_items_currently_delinquent": "tl_delin",
    "experian_payment_amount_monthly_total": "mopmt",
}


def alias_columns(df):
    """Rename long Experian column names to short names, add crhist."""
    df = df.copy()
    for long_name, short_name in COLUMN_ALIASES.items():
        if long_name in df.columns and short_name not in df.columns:
            df[short_name] = pd.to_numeric(df[long_name], errors="coerce").fillna(0)

    # Credit history length: days since oldest tradeline
    if "experian_tradelines_oldest_date" in df.columns and "crhist" not in df.columns:
        oldest = pd.to_datetime(df["experian_tradelines_oldest_date"], errors="coerce")
        signing = pd.to_datetime(df["signing_date"], errors="coerce")
        df["crhist"] = (signing - oldest).dt.days.fillna(0).clip(lower=0)

    return df


def main():
    print("=" * 60)
    print("  TRAIN v6.0 — WoE-Only Model")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and split data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    train_df, test_df = load_and_split()
    print(f"  Train: {len(train_df):,} loans, {train_df['is_bad'].mean():.1%} bad rate")
    print(f"  Test:  {len(test_df):,} loans, {test_df['is_bad'].mean():.1%} bad rate")

    # ------------------------------------------------------------------
    # 2. Alias columns and engineer features
    # ------------------------------------------------------------------
    print("\n[2/6] Engineering features...")
    train_df = alias_columns(train_df)
    test_df = alias_columns(test_df)
    train_feat = build_all_features(train_df)
    test_feat = build_all_features(test_df)
    print(f"  Features available: {len(train_feat.columns)}")

    # ------------------------------------------------------------------
    # 3. Train WoE model
    # ------------------------------------------------------------------
    print("\n[3/6] Training WoE model...")
    model = DefaultModel()
    model.fit(train_feat, y_col="is_bad")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n[4/6] Evaluating on test set...")
    metrics = model.evaluate(test_feat, y_col="is_bad")

    print(f"\n  Test AUC:           {metrics['auc']}")
    print(f"  Test Gini:          {metrics['gini']}")
    print(f"  Mean predicted PD:  {metrics['mean_pd']}")
    print(f"  Actual bad rate:    {metrics['actual_bad_rate']}")
    print(f"  Pred/actual ratio:  {metrics['pred_actual_ratio']}x")
    print(f"  Features used:      {len(metrics['selected_features'])}")
    print(f"\n  Feature IVs:")
    for feat in metrics['selected_features']:
        print(f"    {feat:<25s} IV={metrics['iv_scores'][feat]:.4f}")

    # Validate AUC threshold
    if metrics['auc'] < 0.64:
        print(f"\n  WARNING: AUC {metrics['auc']} is below 0.64 threshold!")
    else:
        print(f"\n  PASS: AUC {metrics['auc']} >= 0.64")

    # ------------------------------------------------------------------
    # 5. Validate grade monotonicity
    # ------------------------------------------------------------------
    print("\n[5/6] Validating grade monotonicity...")
    grader = Grader()

    test_pds = model.predict_pd_batch(test_feat)
    test_grades = grader.assign_batch(test_pds)

    grade_stats = []
    for grade in ["A", "B", "C", "D"]:
        mask = test_grades == grade
        n = mask.sum()
        if n > 0:
            bad_rate = test_feat.loc[mask, "is_bad"].mean()
            mean_pd = test_pds[mask].mean()
            grade_stats.append({
                "grade": grade, "n": n, "bad_rate": bad_rate, "mean_pd": mean_pd
            })
            print(f"  Grade {grade}: {n:>5,} loans, {bad_rate:>6.1%} bad rate, {mean_pd:.4f} mean PD")

    # Check monotonicity
    bad_rates = [s["bad_rate"] for s in grade_stats]
    is_monotonic = all(bad_rates[i] <= bad_rates[i + 1] for i in range(len(bad_rates) - 1))
    print(f"\n  Grade monotonicity: {'PASS' if is_monotonic else 'FAIL'}")

    if not is_monotonic:
        print("  WARNING: Bad rates are not monotonically increasing by grade!")
        print("  Consider adjusting grade boundaries.")

    # ------------------------------------------------------------------
    # 5b. Also evaluate on full portfolio (train + test)
    # ------------------------------------------------------------------
    print("\n  --- Full portfolio grade distribution ---")
    full_df = pd.concat([train_feat, test_feat])
    full_pds = model.predict_pd_batch(full_df)
    full_grades = grader.assign_batch(full_pds)

    for grade in ["A", "B", "C", "D"]:
        mask = full_grades == grade
        n = mask.sum()
        pct = n / len(full_df) * 100
        bad_rate = full_df.loc[mask, "is_bad"].mean() if mask.any() else 0
        print(f"  Grade {grade}: {n:>6,} ({pct:>5.1f}%), {bad_rate:>6.1%} bad rate")
    print(f"  Total:  {len(full_df):>6,}")

    # ------------------------------------------------------------------
    # 6. Save artifacts
    # ------------------------------------------------------------------
    print("\n[6/6] Saving artifacts...")
    model.save(str(MODEL_DIR / "woe_model.joblib"))

    config = {
        "grader": grader.get_config(),
        "version": "6.0",
        "pd_decline_threshold": 0.25,
        "pd_review_threshold": 0.15,
        "train_auc": metrics['auc'],
        "train_date": str(pd.Timestamp.now()),
    }
    joblib.dump(config, str(MODEL_DIR / "config.joblib"))
    print(f"  Saved: woe_model.joblib, config.joblib")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  AUC: {metrics['auc']} | Features: {len(metrics['selected_features'])} | Monotonic: {is_monotonic}")
    print(f"{'=' * 60}")

    return metrics


if __name__ == "__main__":
    main()
