#!/usr/bin/env python3
"""
SurvivalScorer -- Time-to-Default Survival Model
=================================================
Predicts probability of default within T months using Cox Proportional Hazards.
Complements the binary DefaultScorecard with temporal risk profiles.

Usage:
    from scripts.models.survival_scorer import SurvivalScorer

    scorer = SurvivalScorer()
    df = pd.read_parquet("data/master_features.parquet")
    scorer.fit(df)

    result = scorer.predict({"fico": 620, "qi": "MISSING", "partner": "PAYSAFE"})
    # -> {"pd_6mo": 0.08, "pd_12mo": 0.18, "pd_18mo": 0.25, "risk_tier": "early_default", ...}

    results_df = scorer.predict_batch(test_df)
    metrics = scorer.evaluate(test_df)

    scorer.save("models/survival_scorer.joblib")
    loaded = SurvivalScorer.load("models/survival_scorer.joblib")
"""

import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from scripts.models.model_utils import (
    BAD_STATES, GOOD_STATES, ACTIVE_STATES, EXCLUDE_STATES,
    SPLIT_DATE, PARTNER_ENCODING, _safe_float, _is_missing,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

# Observation date for computing durations (current date)
OBSERVATION_DATE = pd.Timestamp("2026-02-25")

# Days per month conversion
DAYS_PER_MONTH = 30.44

# Column name mapping (raw -> short) -- same as DefaultScorecard
COLUMN_MAP = {
    "experian_delinquencies_thirty_day_count": "d30",
    "experian_delinquencies_sixty_day_count": "d60",
    "experian_delinquencies_thirty_days": "d30",
    "experian_delinquencies_sixty_days": "d60",
    "experian_inquiries_six_months": "inq6",
    "experian_inquiries_last_six_months": "inq6",
    "experian_revolving_account_credit_available_percentage": "revutil",
    "experian_revolving_utilization": "revutil",
    "experian_balance_total_past_due_amounts": "pastdue",
    "experian_past_due_amount": "pastdue",
    "experian_balance_total_installment_accounts": "instbal",
    "experian_installment_balance": "instbal",
    "experian_balance_total_revolving_accounts": "revbal",
    "experian_revolving_balance": "revbal",
    "experian_tradelines_total_items": "tl_total",
    "experian_tradelines_total_items_paid": "tl_paid",
    "experian_tradelines_total_items_currently_delinquent": "tl_delin",
    "experian_delinquent_tradelines": "tl_delin",
    "experian_payment_amount_monthly_total": "mopmt",
    "experian_credit_history_length_months": "crhist",
    "j_latest_fico_score": "fico",
    "experian_FICO_SCORE": "fico",
}

# SAFE features only -- NO leaky features
# NEVER use: has_taktile, has_prove, has_plaid, grade, principal, j_credit_score, partner_encoded
FEATURES = [
    "fico", "d30", "d60", "inq6", "revutil", "pastdue",
    "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
    "mopmt", "crhist",
    "qi_encoded", "qi_missing",
    "fico_qi_interaction",
    "paid_ratio", "delin_rate", "delin_severity",
    "bureau_health", "cash_stress",
]

# Risk tier thresholds — assigned by hazard ratio percentiles from training data.
# These are set during fit() based on the 67th and 33rd percentile of predicted
# partial hazard. During prediction, we compare the application's hazard ratio
# against these thresholds to assign: "early_default", "late_default", "low_risk".
HAZARD_TIER_PERCENTILES = (33, 67)  # Low/medium/high cutoffs

# Prediction time points (in months)
TIME_POINTS_MONTHS = [6, 12, 18]
TIME_POINTS_DAYS = [int(t * DAYS_PER_MONTH) for t in TIME_POINTS_MONTHS]


# =============================================================================
# FEATURE ENGINEERING (consistent with DefaultScorecard)
# =============================================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Experian column names to short names."""
    df = df.copy()
    rename_map = {}
    for raw, short in COLUMN_MAP.items():
        if raw in df.columns and short not in df.columns:
            rename_map[raw] = short
    if rename_map:
        df = df.rename(columns=rename_map)

    # Derive crhist from tradelines oldest date if available
    if "crhist" not in df.columns and "experian_tradelines_oldest_date" in df.columns:
        oldest = pd.to_datetime(df["experian_tradelines_oldest_date"], errors="coerce")
        ref_date = df["signing_date"] if "signing_date" in df.columns else pd.Timestamp(OBSERVATION_DATE)
        ref_date = pd.to_datetime(ref_date, errors="coerce")
        df["crhist"] = ((ref_date - oldest).dt.days / 30.44).clip(lower=0).fillna(0)

    # Normalize shop_qi -> qi
    if "shop_qi" in df.columns and "qi" not in df.columns:
        df["qi"] = df["shop_qi"]

    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build derived features for survival model (mirrors DefaultScorecard pipeline)."""
    df = df.copy()

    # Ensure all needed numeric columns exist (default to NaN/0 for single-row predictions)
    numeric_cols = ["fico", "d30", "d60", "inq6", "revutil", "pastdue",
                    "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
                    "mopmt", "crhist"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # QI encoding (ordinal by bad rate: MISSING worst, HFHT best)
    qi_col = "qi" if "qi" in df.columns else "shop_qi"
    if qi_col in df.columns:
        qi_map = {"HFHT": 0, "HFLT": 1, "LFHT": 2, "LFLT": 3, "MISSING": 4}
        df["qi_encoded"] = df[qi_col].astype(str).str.upper().map(qi_map).fillna(4).astype(int)
        df["qi_missing"] = (
            df[qi_col].isna() | df[qi_col].astype(str).str.upper().isin(["MISSING", "NONE", "NAN"])
        ).astype(int)
    else:
        df["qi_encoded"] = 4
        df["qi_missing"] = 1

    # FICO x QI interaction (15.4x lift -- the strongest signal)
    fico_bin = pd.cut(
        df["fico"], bins=[0, 550, 600, 650, 700, 900],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2)
    df["fico_qi_interaction"] = fico_bin * 5 + df["qi_encoded"]

    # Bureau-derived ratios
    df["paid_ratio"] = (df["tl_paid"] / df["tl_total"].clip(lower=1)).clip(0, 1)
    df["delin_rate"] = (df["tl_delin"] / df["tl_total"].clip(lower=1)).clip(0, 1)
    df["delin_severity"] = df["d30"].fillna(0) + 2 * df["d60"].fillna(0)

    # Balance features
    total_bal = (df["instbal"].fillna(0) + df["revbal"].fillna(0)).clip(lower=1)
    df["payment_burden"] = df["mopmt"].fillna(0) / total_bal

    # Credit activity
    df["credit_activity"] = df["inq6"].fillna(0) / df["tl_total"].clip(lower=1)

    # Bureau health composite
    zero_current_delin = (df["tl_delin"].fillna(0) == 0).astype(int)
    df["bureau_health"] = (
        zero_current_delin * 0.4 +
        (1 - df["credit_activity"].clip(0, 1)) * 0.3 +
        df["paid_ratio"].clip(0, 1) * 0.3
    )

    # Cash stress indicator
    df["cash_stress"] = (
        ((df["fico"] < 580).astype(int) * 2) +
        ((df["pastdue"].fillna(0) > 0).astype(int)) +
        ((df["revutil"].fillna(0) > 80).astype(int))
    )

    return df


def _prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare survival data: compute duration and event indicator.

    Duration = days from signing_date to observation or event.
    For all loans, we use (observation_date - signing_date) as the duration,
    since we don't have exact default dates. For bad loans, this is an upper
    bound on time-to-default. For good/active loans, this is right-censored.

    Event = 1 if defaulted, 0 if censored (active or paid off).
    """
    df = df.copy()

    # Ensure signing_date is datetime
    df["signing_date"] = pd.to_datetime(df["signing_date"], errors="coerce")

    # Compute age in days and months
    df["duration_days"] = (OBSERVATION_DATE - df["signing_date"]).dt.days
    df["duration_months"] = df["duration_days"] / DAYS_PER_MONTH

    # Event indicator: 1 = defaulted, 0 = censored
    df["event"] = df["loan_state"].isin(BAD_STATES).astype(int)

    # Filter: must have positive duration and valid signing date
    df = df[df["duration_days"] > 0].copy()
    df = df[df["signing_date"].notna()].copy()

    return df


# =============================================================================
# SURVIVAL SCORER CLASS
# =============================================================================

class SurvivalScorer:
    """Time-to-default survival model using Cox Proportional Hazards.

    Predicts probability of default within T months for pricing and reserving.
    Complements the binary DefaultScorecard with temporal risk profiles.
    """

    VERSION = "1.0.0"
    FEATURES = FEATURES

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0):
        """Initialize SurvivalScorer.

        Args:
            penalizer: Regularization strength for Cox PH (higher = more regularization).
            l1_ratio: Elastic net mixing (0 = L2 only, 1 = L1 only).
        """
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.cph: Optional[CoxPHFitter] = None
        self.kmf_overall: Optional[KaplanMeierFitter] = None
        self.train_concordance: Optional[float] = None
        self.test_concordance: Optional[float] = None
        self.feature_names: List[str] = []
        self.train_stats: Dict[str, Any] = {}
        self.hazard_thresholds: Optional[tuple] = None  # (q33, q67) from training data
        self._is_fitted = False

    # -----------------------------------------------------------------
    # FIT
    # -----------------------------------------------------------------

    def fit(self, df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None,
            split_date: Optional[str] = None) -> "SurvivalScorer":
        """Fit Cox PH model on training data.

        If test_df is not provided, performs automatic time-based split.

        Args:
            df: DataFrame with raw features + loan_state + signing_date.
                Can be raw master_features.parquet format.
            test_df: Optional separate test DataFrame.
            split_date: Optional split date string (default: 2025-07-01).

        Returns:
            self (for chaining).
        """
        logger.info("SurvivalScorer.fit() starting...")

        # --- Step 1: Normalize columns ---
        df = _normalize_columns(df)

        # --- Step 2: Filter excluded states ---
        if "loan_state" in df.columns:
            df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()

        # --- Step 3: Require FICO ---
        if "fico" in df.columns:
            df = df[df["fico"].notna()].copy()

        # --- Step 4: Engineer features ---
        df = _engineer_features(df)

        # --- Step 5: Prepare survival data (duration + event) ---
        df = _prepare_survival_data(df)

        # --- Step 6: Time-based split ---
        cutoff = pd.Timestamp(split_date) if split_date else SPLIT_DATE
        if test_df is not None:
            test_df = _normalize_columns(test_df)
            test_df = _engineer_features(test_df)
            test_df = _prepare_survival_data(test_df)
            train = df[df["signing_date"] < cutoff].copy()
            test = test_df
        else:
            train = df[df["signing_date"] < cutoff].copy()
            test = df[df["signing_date"] >= cutoff].copy()

        # --- Step 7: Select features available in data ---
        self.feature_names = [f for f in self.FEATURES if f in train.columns]
        logger.info(f"Using {len(self.feature_names)} features: {self.feature_names}")

        # --- Step 8: Prepare training matrix ---
        surv_cols = self.feature_names + ["duration_days", "event"]
        train_surv = train[surv_cols].dropna().copy()
        test_surv = test[surv_cols].dropna().copy()

        # Ensure all features are float
        for col in self.feature_names:
            train_surv[col] = train_surv[col].astype(float)
            test_surv[col] = test_surv[col].astype(float)

        logger.info(f"Train: {len(train_surv):,} loans, {train_surv['event'].sum():,} events "
                     f"({train_surv['event'].mean()*100:.1f}%)")
        logger.info(f"Test:  {len(test_surv):,} loans, {test_surv['event'].sum():,} events "
                     f"({test_surv['event'].mean()*100:.1f}%)")

        # Store stats
        self.train_stats = {
            "n_train": len(train_surv),
            "n_test": len(test_surv),
            "n_events_train": int(train_surv["event"].sum()),
            "n_events_test": int(test_surv["event"].sum()),
            "event_rate_train": round(train_surv["event"].mean(), 4),
            "event_rate_test": round(test_surv["event"].mean(), 4),
            "median_duration_train": round(train_surv["duration_days"].median(), 1),
            "median_duration_test": round(test_surv["duration_days"].median(), 1),
        }

        # --- Step 9: Fit Cox PH model ---
        self.cph = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
        )
        self.cph.fit(
            train_surv,
            duration_col="duration_days",
            event_col="event",
        )

        # --- Step 10: Compute concordance index ---
        self.train_concordance = concordance_index(
            train_surv["duration_days"],
            -self.cph.predict_partial_hazard(train_surv[self.feature_names]),
            train_surv["event"],
        )
        self.test_concordance = concordance_index(
            test_surv["duration_days"],
            -self.cph.predict_partial_hazard(test_surv[self.feature_names]),
            test_surv["event"],
        )

        # --- Step 10b: Compute hazard tier thresholds from training data ---
        train_hazard = self.cph.predict_partial_hazard(train_surv[self.feature_names]).values.flatten()
        q33 = float(np.percentile(train_hazard, HAZARD_TIER_PERCENTILES[0]))
        q67 = float(np.percentile(train_hazard, HAZARD_TIER_PERCENTILES[1]))
        self.hazard_thresholds = (q33, q67)
        logger.info(f"Hazard tier thresholds: low<{q33:.4f}, mid<{q67:.4f}, high>={q67:.4f}")

        # --- Step 11: Fit overall KM estimator (for baseline survival) ---
        self.kmf_overall = KaplanMeierFitter()
        self.kmf_overall.fit(
            train_surv["duration_days"],
            event_observed=train_surv["event"],
        )

        # --- Step 12: Store test data reference for evaluation ---
        self._test_surv = test_surv
        self._train_surv = train_surv
        self._is_fitted = True

        logger.info(f"Train C-index: {self.train_concordance:.4f}")
        logger.info(f"Test  C-index: {self.test_concordance:.4f}")
        print(f"\n{'='*60}")
        print(f"SurvivalScorer v{self.VERSION} -- Fit Complete")
        print(f"{'='*60}")
        print(f"  Features:          {len(self.feature_names)}")
        print(f"  Train:             {self.train_stats['n_train']:,} loans, "
              f"{self.train_stats['n_events_train']:,} events ({self.train_stats['event_rate_train']:.1%})")
        print(f"  Test:              {self.train_stats['n_test']:,} loans, "
              f"{self.train_stats['n_events_test']:,} events ({self.train_stats['event_rate_test']:.1%})")
        print(f"  Train C-index:     {self.train_concordance:.4f}")
        print(f"  Test  C-index:     {self.test_concordance:.4f}")
        print(f"{'='*60}")

        # Print coefficient summary
        self._print_coefficients()

        return self

    def _print_coefficients(self):
        """Print top Cox PH coefficients sorted by absolute value."""
        if self.cph is None:
            return
        summary = self.cph.summary.sort_values("coef", key=abs, ascending=False)
        print(f"\n  Top Coefficients (hazard ratios):")
        print(f"  {'Feature':<25} {'Coef':>8} {'HR':>8} {'p-value':>10} {'Sig':>4}")
        print(f"  {'-'*58}")
        for idx, row in summary.head(15).iterrows():
            direction = "risk+" if row["coef"] > 0 else "risk-"
            sig = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else ""))
            print(f"  {idx:<25} {row['coef']:>8.4f} {row['exp(coef)']:>8.4f} {row['p']:>10.4f} {sig:>4}")

    # -----------------------------------------------------------------
    # PREDICT (single application)
    # -----------------------------------------------------------------

    def predict(self, features: Union[Dict, pd.Series]) -> Dict[str, Any]:
        """Predict survival profile for a single application.

        Args:
            features: Dict or Series with feature values. Can use raw column
                      names (experian_FICO_SCORE) or short names (fico).

        Returns:
            Dict with survival probabilities, PD at time horizons,
            median survival, and risk tier.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert to DataFrame row
        if isinstance(features, dict):
            row_df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            row_df = pd.DataFrame([features.to_dict()])
        else:
            raise TypeError(f"Expected dict or pd.Series, got {type(features)}")

        # Normalize and engineer
        row_df = _normalize_columns(row_df)
        row_df = _engineer_features(row_df)

        return self._predict_row(row_df)

    def _predict_row(self, row_df: pd.DataFrame) -> Dict[str, Any]:
        """Internal prediction on a single-row DataFrame with engineered features."""
        # Fill missing features with 0
        for feat in self.feature_names:
            if feat not in row_df.columns:
                row_df[feat] = 0.0
            else:
                row_df[feat] = row_df[feat].fillna(0.0).astype(float)

        X = row_df[self.feature_names]

        # Predict survival function
        surv_func = self.cph.predict_survival_function(X)

        # Extract survival probabilities at time points
        result = {}
        for months, days in zip(TIME_POINTS_MONTHS, TIME_POINTS_DAYS):
            # Find closest available time index
            idx = surv_func.index.get_indexer([days], method="nearest")[0]
            surv_prob = float(surv_func.iloc[idx, 0])
            result[f"survival_{months}mo"] = round(surv_prob, 4)
            result[f"pd_{months}mo"] = round(1 - surv_prob, 4)

        # Median survival time
        try:
            median_surv_days = float(self.cph.predict_median(X).iloc[0])
            median_surv_months = median_surv_days / DAYS_PER_MONTH
        except Exception:
            median_surv_months = float("inf")
        result["median_survival_months"] = round(median_surv_months, 1) if np.isfinite(median_surv_months) else None

        # Partial hazard (relative risk score)
        hazard = float(self.cph.predict_partial_hazard(X).iloc[0])
        result["hazard_ratio"] = round(hazard, 4)

        # Risk tier assignment (based on training hazard percentiles)
        if self.hazard_thresholds is not None:
            q33, q67 = self.hazard_thresholds
            if hazard >= q67:
                result["risk_tier"] = "early_default"
            elif hazard >= q33:
                result["risk_tier"] = "late_default"
            else:
                result["risk_tier"] = "low_risk"
        else:
            result["risk_tier"] = "unknown"

        return result

    # -----------------------------------------------------------------
    # PREDICT BATCH
    # -----------------------------------------------------------------

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction on a DataFrame.

        Args:
            df: DataFrame with features (raw or short names).

        Returns:
            DataFrame with original index + prediction columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Normalize and engineer
        df_proc = _normalize_columns(df)
        df_proc = _engineer_features(df_proc)

        # Fill missing features
        for feat in self.feature_names:
            if feat not in df_proc.columns:
                df_proc[feat] = 0.0
            else:
                df_proc[feat] = pd.to_numeric(df_proc[feat], errors="coerce").fillna(0.0)

        X = df_proc[self.feature_names].astype(float)

        # Predict survival functions for all rows
        surv_funcs = self.cph.predict_survival_function(X)

        # Extract predictions at each time point
        results = pd.DataFrame(index=df.index)
        for months, days in zip(TIME_POINTS_MONTHS, TIME_POINTS_DAYS):
            idx = surv_funcs.index.get_indexer([days], method="nearest")[0]
            surv_at_t = surv_funcs.iloc[idx, :].values
            results[f"survival_{months}mo"] = np.round(surv_at_t, 4)
            results[f"pd_{months}mo"] = np.round(1 - surv_at_t, 4)

        # Partial hazard
        results["hazard_ratio"] = self.cph.predict_partial_hazard(X).values.flatten()

        # Median survival
        try:
            medians = self.cph.predict_median(X)
            results["median_survival_months"] = (medians.values.flatten() / DAYS_PER_MONTH).round(1)
        except Exception:
            results["median_survival_months"] = np.nan

        # Risk tier (based on training hazard percentiles)
        if self.hazard_thresholds is not None:
            q33, q67 = self.hazard_thresholds
            results["risk_tier"] = np.where(
                results["hazard_ratio"] >= q67, "early_default",
                np.where(results["hazard_ratio"] >= q33, "late_default", "low_risk"),
            )
        else:
            results["risk_tier"] = "unknown"

        return results

    # -----------------------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------------------

    def evaluate(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate model performance.

        Args:
            df: DataFrame to evaluate on. If None, uses the held-out test set from fit().

        Returns:
            Dict with concordance index, Brier scores at 6/12/18mo,
            calibration metrics, and risk tier separation.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if df is not None:
            # Prepare external data
            df = _normalize_columns(df)
            if "loan_state" in df.columns:
                df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
            df = _engineer_features(df)
            df = _prepare_survival_data(df)
            surv_cols = self.feature_names + ["duration_days", "event"]
            eval_data = df[surv_cols].dropna().copy()
            for col in self.feature_names:
                eval_data[col] = eval_data[col].astype(float)
        else:
            eval_data = self._test_surv

        # Concordance index
        ci = concordance_index(
            eval_data["duration_days"],
            -self.cph.predict_partial_hazard(eval_data[self.feature_names]),
            eval_data["event"],
        )

        # Brier scores at time horizons
        brier_scores = {}
        for months, days in zip(TIME_POINTS_MONTHS, TIME_POINTS_DAYS):
            brier = self._brier_score_at_time(eval_data, days)
            brier_scores[f"brier_{months}mo"] = round(brier, 4) if brier is not None else None

        # Calibration: predicted vs actual survival at each time point
        calibration = {}
        for months, days in zip(TIME_POINTS_MONTHS, TIME_POINTS_DAYS):
            cal = self._calibration_at_time(eval_data, days)
            calibration[f"calibration_{months}mo"] = cal

        # Risk tier separation
        tier_stats = self._risk_tier_separation(eval_data)

        # Kaplan-Meier by risk tier
        km_by_tier = self._km_by_risk_tier(eval_data)

        metrics = {
            "concordance_index": round(ci, 4),
            "train_concordance": round(self.train_concordance, 4),
            "n_features": len(self.feature_names),
            "n_eval": len(eval_data),
            "n_events": int(eval_data["event"].sum()),
            "event_rate": round(eval_data["event"].mean(), 4),
            **brier_scores,
            "calibration": calibration,
            "risk_tier_stats": tier_stats,
            "km_by_tier": km_by_tier,
        }

        self._print_evaluation(metrics)
        return metrics

    def _brier_score_at_time(self, data: pd.DataFrame, t_days: int) -> Optional[float]:
        """Compute Brier score at a specific time point.

        Brier score = mean((predicted_survival - actual_status)^2)
        For loans observed beyond t_days: actual status is known.
        For loans censored before t_days: excluded (we don't know their status).
        """
        # Only include loans that were observed long enough
        eligible = data[data["duration_days"] >= t_days].copy()
        if len(eligible) < 20:
            return None

        # Actual: 1 if survived past t_days (event=0 or event happened after t_days)
        # Since our duration = observation age (not exact event time), for event=1 loans
        # we know they defaulted sometime during the observation period.
        # But we don't know exactly when. So we use event indicator as the truth:
        # event=1 means "did not survive" (defaulted during period)
        actual_survived = 1 - eligible["event"].values

        # Predicted survival probability at t_days
        surv_funcs = self.cph.predict_survival_function(eligible[self.feature_names].astype(float))
        idx = surv_funcs.index.get_indexer([t_days], method="nearest")[0]
        predicted_survived = surv_funcs.iloc[idx, :].values

        # Brier score
        brier = float(np.mean((predicted_survived - actual_survived) ** 2))
        return brier

    def _calibration_at_time(self, data: pd.DataFrame, t_days: int) -> Dict[str, float]:
        """Compute calibration at a specific time point."""
        eligible = data[data["duration_days"] >= t_days].copy()
        if len(eligible) < 20:
            return {"predicted_survival": None, "actual_survival": None, "n": 0}

        actual_survival_rate = 1 - eligible["event"].mean()

        surv_funcs = self.cph.predict_survival_function(eligible[self.feature_names].astype(float))
        idx = surv_funcs.index.get_indexer([t_days], method="nearest")[0]
        predicted_survival_rate = float(surv_funcs.iloc[idx, :].mean())

        return {
            "predicted_survival": round(predicted_survival_rate, 4),
            "actual_survival": round(actual_survival_rate, 4),
            "gap": round(abs(predicted_survival_rate - actual_survival_rate), 4),
            "n": len(eligible),
        }

    def _risk_tier_separation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute default rates by predicted risk tier (hazard-based)."""
        X = data[self.feature_names].astype(float)
        hazard = self.cph.predict_partial_hazard(X).values.flatten()

        # Use training-derived thresholds
        if self.hazard_thresholds is not None:
            q33, q67 = self.hazard_thresholds
        else:
            q33 = np.percentile(hazard, 33)
            q67 = np.percentile(hazard, 67)

        tiers = np.where(
            hazard >= q67, "early_default",
            np.where(hazard >= q33, "late_default", "low_risk"),
        )

        tier_data = data[["event"]].copy()
        tier_data["tier"] = tiers

        tier_stats = {}
        for tier in ["early_default", "late_default", "low_risk"]:
            mask = tier_data["tier"] == tier
            if mask.sum() > 0:
                tier_stats[tier] = {
                    "n": int(mask.sum()),
                    "pct": round(mask.mean(), 4),
                    "bad_rate": round(tier_data.loc[mask, "event"].mean(), 4),
                    "n_bad": int(tier_data.loc[mask, "event"].sum()),
                }
        return tier_stats

    def _km_by_risk_tier(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Compute Kaplan-Meier survival estimates by risk tier."""
        X = data[self.feature_names].astype(float)

        # Use hazard ratio percentiles for tier assignment
        hazard = self.cph.predict_partial_hazard(X).values.flatten()
        q33 = np.percentile(hazard, 33)
        q66 = np.percentile(hazard, 66)

        tiers = np.where(hazard <= q33, "low_risk",
                         np.where(hazard <= q66, "medium_risk", "high_risk"))

        km_results = {}
        kmf = KaplanMeierFitter()
        for tier in ["low_risk", "medium_risk", "high_risk"]:
            mask = tiers == tier
            if mask.sum() > 10:
                kmf.fit(
                    data.loc[mask, "duration_days"],
                    event_observed=data.loc[mask, "event"],
                    label=tier,
                )
                tier_result = {"n": int(mask.sum()), "events": int(data.loc[mask, "event"].sum())}
                for months, days in zip(TIME_POINTS_MONTHS, TIME_POINTS_DAYS):
                    try:
                        surv = float(kmf.predict(days))
                        tier_result[f"survival_{months}mo"] = round(surv, 4)
                        tier_result[f"pd_{months}mo"] = round(1 - surv, 4)
                    except Exception:
                        tier_result[f"survival_{months}mo"] = None
                        tier_result[f"pd_{months}mo"] = None
                km_results[tier] = tier_result

        return km_results

    def _print_evaluation(self, metrics: Dict):
        """Print formatted evaluation results."""
        print(f"\n{'='*60}")
        print(f"SurvivalScorer v{self.VERSION} -- Evaluation Results")
        print(f"{'='*60}")
        print(f"  Concordance Index: {metrics['concordance_index']:.4f}")
        print(f"  (Train C-index:    {metrics['train_concordance']:.4f})")
        print(f"  N eval:            {metrics['n_eval']:,}")
        print(f"  N events:          {metrics['n_events']:,} ({metrics['event_rate']:.1%})")

        # Brier scores
        print(f"\n  Brier Scores (lower is better):")
        for months in TIME_POINTS_MONTHS:
            key = f"brier_{months}mo"
            val = metrics.get(key)
            print(f"    {months}-month: {val:.4f}" if val is not None else f"    {months}-month: N/A")

        # Calibration
        print(f"\n  Calibration (predicted vs actual survival):")
        for months in TIME_POINTS_MONTHS:
            key = f"calibration_{months}mo"
            cal = metrics.get(key, {})
            if cal and cal.get("predicted_survival") is not None:
                print(f"    {months}-month: predicted={cal['predicted_survival']:.4f}, "
                      f"actual={cal['actual_survival']:.4f}, gap={cal['gap']:.4f} "
                      f"(n={cal['n']:,})")

        # Risk tier separation
        print(f"\n  Risk Tier Separation:")
        tier_stats = metrics.get("risk_tier_stats", {})
        for tier in ["early_default", "late_default", "low_risk"]:
            ts = tier_stats.get(tier, {})
            if ts:
                print(f"    {tier:<18} n={ts['n']:>5} ({ts['pct']:.1%})  "
                      f"bad_rate={ts['bad_rate']:.1%}  bads={ts['n_bad']}")

        # KM by hazard tier
        print(f"\n  Kaplan-Meier Survival by Hazard Tier:")
        km = metrics.get("km_by_tier", {})
        print(f"    {'Tier':<15} {'N':>6} {'Events':>7} "
              f"{'PD 6mo':>8} {'PD 12mo':>8} {'PD 18mo':>8}")
        print(f"    {'-'*55}")
        for tier in ["low_risk", "medium_risk", "high_risk"]:
            kt = km.get(tier, {})
            if kt:
                pd6 = kt.get("pd_6mo")
                pd12 = kt.get("pd_12mo")
                pd18 = kt.get("pd_18mo")
                print(f"    {tier:<15} {kt['n']:>6} {kt['events']:>7} "
                      f"{pd6:>7.1%} {pd12:>7.1%} {pd18:>7.1%}")

        print(f"{'='*60}")

    # -----------------------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize model to disk.

        Args:
            path: File path for the joblib file.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        save_dict = {
            "version": self.VERSION,
            "cph": self.cph,
            "kmf_overall": self.kmf_overall,
            "feature_names": self.feature_names,
            "penalizer": self.penalizer,
            "l1_ratio": self.l1_ratio,
            "train_concordance": self.train_concordance,
            "test_concordance": self.test_concordance,
            "train_stats": self.train_stats,
            "hazard_thresholds": self.hazard_thresholds,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, path)
        print(f"  Model saved to {path} ({path.stat().st_size / 1024:.1f} KB)")

    @classmethod
    def load(cls, path: str) -> "SurvivalScorer":
        """Load serialized model from disk.

        Args:
            path: File path for the joblib file.

        Returns:
            Fitted SurvivalScorer instance.
        """
        data = joblib.load(path)

        scorer = cls(
            penalizer=data.get("penalizer", 0.1),
            l1_ratio=data.get("l1_ratio", 0.0),
        )
        scorer.cph = data["cph"]
        scorer.kmf_overall = data.get("kmf_overall")
        scorer.feature_names = data["feature_names"]
        scorer.train_concordance = data.get("train_concordance")
        scorer.test_concordance = data.get("test_concordance")
        scorer.train_stats = data.get("train_stats", {})
        scorer.hazard_thresholds = data.get("hazard_thresholds")
        scorer._is_fitted = True

        print(f"  Loaded SurvivalScorer v{data.get('version', '?')} "
              f"({len(scorer.feature_names)} features, "
              f"C-index={scorer.test_concordance})")
        return scorer

    # -----------------------------------------------------------------
    # REPRESENTATIVE PROFILES (for documentation)
    # -----------------------------------------------------------------

    def predict_profiles(self) -> pd.DataFrame:
        """Predict survival for representative borrower profiles.

        Returns:
            DataFrame with profile descriptions and predicted PD at each horizon.
        """
        profiles = [
            {"label": "High-risk: Sub-600 FICO, delinquent, missing QI",
             "fico": 550, "d30": 5, "d60": 2, "inq6": 4, "revutil": 20,
             "pastdue": 500, "instbal": 5000, "revbal": 8000,
             "tl_total": 10, "tl_paid": 3, "tl_delin": 2, "mopmt": 300, "crhist": 36,
             "qi": "MISSING", "partner": "PAYSAFE"},

            {"label": "Mid-risk: FICO 625, moderate bureau, HFLT",
             "fico": 625, "d30": 3, "d60": 1, "inq6": 2, "revutil": 50,
             "pastdue": 100, "instbal": 10000, "revbal": 5000,
             "tl_total": 15, "tl_paid": 8, "tl_delin": 1, "mopmt": 500, "crhist": 72,
             "qi": "HFLT", "partner": "SPOTON"},

            {"label": "Low-risk: FICO 700, clean bureau, HFHT",
             "fico": 700, "d30": 0, "d60": 0, "inq6": 1, "revutil": 35,
             "pastdue": 0, "instbal": 15000, "revbal": 3000,
             "tl_total": 20, "tl_paid": 15, "tl_delin": 0, "mopmt": 400, "crhist": 120,
             "qi": "HFHT", "partner": "HONEYBOOK"},

            {"label": "Stressed: Sub-580 FICO, high delinquency",
             "fico": 560, "d30": 8, "d60": 4, "inq6": 6, "revutil": 95,
             "pastdue": 2000, "instbal": 8000, "revbal": 12000,
             "tl_total": 12, "tl_paid": 2, "tl_delin": 4, "mopmt": 800, "crhist": 48,
             "qi": "LFLT", "partner": "PAYSAFE"},

            {"label": "Pristine: High FICO, no issues, HFHT",
             "fico": 750, "d30": 0, "d60": 0, "inq6": 0, "revutil": 15,
             "pastdue": 0, "instbal": 5000, "revbal": 1000,
             "tl_total": 25, "tl_paid": 20, "tl_delin": 0, "mopmt": 200, "crhist": 180,
             "qi": "HFHT", "partner": "HONEYBOOK"},
        ]

        rows = []
        for p in profiles:
            label = p.pop("label")
            pred = self.predict(p)
            pred["profile"] = label
            rows.append(pred)

        result = pd.DataFrame(rows)
        # Reorder columns
        cols = ["profile"] + [c for c in result.columns if c != "profile"]
        return result[cols]


# =============================================================================
# MAIN -- Run from command line
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    MODELS_DIR = Path(__file__).parent.parent.parent / "models"

    print("Loading master features...")
    df = pd.read_parquet(DATA_DIR / "master_features.parquet")
    print(f"  {len(df):,} rows loaded")

    # Fit
    scorer = SurvivalScorer(penalizer=0.1)
    scorer.fit(df)

    # Evaluate
    print("\n--- Evaluation on test set ---")
    metrics = scorer.evaluate()

    # Profile predictions
    print("\n--- Representative Profile Predictions ---")
    profiles = scorer.predict_profiles()
    for _, row in profiles.iterrows():
        print(f"\n  {row['profile']}")
        print(f"    PD  6mo: {row['pd_6mo']:.1%}  |  PD 12mo: {row['pd_12mo']:.1%}  |  PD 18mo: {row['pd_18mo']:.1%}")
        ms = row.get('median_survival_months')
        ms_str = f"{ms:.1f}" if ms is not None and np.isfinite(ms) else "inf"
        print(f"    Median survival: {ms_str} months  |  Risk tier: {row['risk_tier']}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scorer.save(str(MODELS_DIR / "survival_scorer.joblib"))

    # Load and verify
    loaded = SurvivalScorer.load(str(MODELS_DIR / "survival_scorer.joblib"))
    test_pred = loaded.predict({"fico": 620, "qi": "MISSING"})
    print(f"\n  Save/load round-trip verified. Test predict: PD 12mo = {test_pred['pd_12mo']:.1%}")

    # Edge cases
    print("\n--- Edge Case Tests ---")
    edge_cases = [
        {"name": "Missing all features", "features": {}},
        {"name": "Extreme low FICO", "features": {"fico": 300}},
        {"name": "Extreme high FICO", "features": {"fico": 850, "qi": "HFHT"}},
        {"name": "Only FICO", "features": {"fico": 620}},
    ]
    for ec in edge_cases:
        try:
            pred = loaded.predict(ec["features"])
            print(f"  {ec['name']:<25} PD 12mo={pred['pd_12mo']:.1%}  tier={pred['risk_tier']}")
        except Exception as e:
            print(f"  {ec['name']:<25} ERROR: {e}")

    print("\nDone.")
