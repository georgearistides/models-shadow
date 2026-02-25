#!/usr/bin/env python3
"""
PaymentMonitor — Post-origination portfolio health monitor using NSF/payment signals.

Productionizes the NSF early-warning system discovered in payment_analysis.py.
This is NOT for underwriting — it monitors existing loans after origination using
payment behavior data (NSF returns, payment timing, recovery attempts).

Key finding: NSF patterns are the strongest early-warning signal for loan defaults.
- Any NSF -> 58.7% default rate vs 1.4% baseline (5.8x lift)
- Red tier (3+ flags) = ~70% default rate
- Green tier (0 flags) = 0.7-2.2% default rate
- 9-flag composite score achieves AUC 0.844 (test) / 0.950 (train)

Usage:
    from scripts.models.payment_monitor import PaymentMonitor

    pm = PaymentMonitor()
    pm.fit(df)

    # Single prediction
    result = pm.predict({"nsf_count": 3, "nsf_rate": 0.25, ...})
    # -> {"risk_tier": "red", "risk_score": 67, "flag_count": 5, ...}

    # Batch prediction
    results_df = pm.predict_batch(df)

    # Evaluate
    metrics = pm.evaluate(df, label_col="is_bad")

    # Serialize
    pm.save("models/payment_monitor.joblib")
    pm2 = PaymentMonitor.load("models/payment_monitor.joblib")
"""

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from scripts.models.model_utils import _is_missing, _safe_float, BAD_STATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BAD_STATES imported from model_utils

# The 9 flags from payment_analysis.py composite score
FLAG_DEFINITIONS = {
    "flag_any_nsf": {
        "description": "Loan has at least 1 returned payment (any type)",
        "feature": "nsf_count",
        "threshold": 0,
        "direction": "gt",
        "weight": 1,
    },
    "flag_nsf_rate_high": {
        "description": "NSF rate > 15% across all repayment attempts",
        "feature": "nsf_rate",
        "threshold": 0.15,
        "direction": "gt",
        "weight": 1,
    },
    "flag_early_nsf": {
        "description": "NSF within first 90 days of signing",
        "feature": "nsf_in_first_90d",
        "threshold": 0,
        "direction": "gt",
        "weight": 1,
    },
    "flag_nsf_cluster": {
        "description": "2+ NSFs within any 30-day window",
        "feature": "nsf_cluster_max",
        "threshold": 2,
        "direction": "gte",
        "weight": 1,
    },
    "flag_consec_nsf": {
        "description": "2+ consecutive returned payments",
        "feature": "max_consecutive_nsf",
        "threshold": 2,
        "direction": "gte",
        "weight": 1,
    },
    "flag_has_recovery": {
        "description": "Loan has RECOVERY payment type (entered collections)",
        "feature": "has_recovery",
        "threshold": 0,
        "direction": "gt",
        "weight": 1,
    },
    "flag_recovery_nsf": {
        "description": ">50% of recovery attempts returned (collections failing)",
        "feature": "recovery_nsf_rate",
        "threshold": 0.5,
        "direction": "gt",
        "weight": 1,
        "fill_na": 0,  # if no recovery attempts, not flagged
    },
    "flag_high_cv": {
        "description": "Customer payment amount CV > 1.0",
        "feature": "payment_cv",
        "threshold": 1.0,
        "direction": "gt",
        "weight": 1,
        "fill_na": 0,
    },
    "flag_declining": {
        "description": "Last 3 customer payments declined >20% vs first 3",
        "feature": "payment_decline_pct",
        "threshold": -0.20,
        "direction": "lt",
        "weight": 1,
        "fill_na": 0,
    },
}

# Default tier boundaries (flag_count -> tier)
DEFAULT_TIER_BOUNDARIES = {
    "green": (0, 0),     # 0 flags
    "yellow": (1, 1),    # 1 flag
    "orange": (2, 2),    # 2 flags
    "red": (3, 9),       # 3+ flags
}

# Recommendations per tier
TIER_RECOMMENDATIONS = {
    "green": "No action needed",
    "yellow": "Monitor -- schedule 30-day review",
    "orange": "Alert -- contact borrower, review account",
    "red": "Critical -- initiate early intervention immediately",
}

# All payment feature columns used by the 9 flags
PAYMENT_FEATURE_COLS = [
    "nsf_count", "nsf_rate", "nsf_in_first_90d", "nsf_cluster_max",
    "max_consecutive_nsf", "has_recovery", "recovery_nsf_rate",
    "payment_cv", "payment_decline_pct",
]

# Max composite score (all 9 flags triggered)
MAX_RAW_SCORE = 9


# ---------------------------------------------------------------------------
# PaymentMonitor class
# ---------------------------------------------------------------------------


class PaymentMonitor:
    """Post-origination portfolio health monitor using NSF/payment signals.

    Scores loans based on 9 binary flags derived from payment behavior data.
    Each flag adds 1 point to a composite early-warning score (0-9). Loans are
    assigned to risk tiers (green/yellow/orange/red) based on total flag count.

    The class also provides calibrated default probabilities via isotonic
    regression, mapping the composite score to P(default).

    Parameters
    ----------
    version : str
        Model version identifier.
    tier_boundaries : dict, optional
        Override tier boundaries. Keys are tier names, values are (min, max)
        inclusive flag count tuples.
    flag_weights : dict, optional
        Override flag weights. Keys are flag names from FLAG_DEFINITIONS,
        values are numeric weights.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        version: str = "1.0.0",
        tier_boundaries: Optional[Dict[str, tuple]] = None,
        flag_weights: Optional[Dict[str, float]] = None,
    ):
        self.version = version
        self.tier_boundaries = tier_boundaries or DEFAULT_TIER_BOUNDARIES.copy()
        self.flag_definitions = FLAG_DEFINITIONS.copy()

        # Apply custom weights if provided
        if flag_weights:
            for flag_name, weight in flag_weights.items():
                if flag_name in self.flag_definitions:
                    self.flag_definitions[flag_name]["weight"] = weight
                else:
                    logger.warning(f"Unknown flag name '{flag_name}' in flag_weights -- ignored")

        # Populated by .fit()
        self._fitted = False
        self._calibrator: Optional[IsotonicRegression] = None
        self._train_stats: Optional[Dict[str, Any]] = None
        self._tier_default_rates: Optional[Dict[str, float]] = None
        self._flag_default_rates: Optional[Dict[str, float]] = None
        self._score_distribution: Optional[Dict[int, int]] = None

    # ------------------------------------------------------------------
    # Flag computation (core logic)
    # ------------------------------------------------------------------

    def _compute_flag(self, flag_name: str, flag_def: dict, value: Any) -> int:
        """Evaluate a single flag for a single value.

        Returns 1 if the flag is triggered, 0 otherwise.
        """
        fill_na = flag_def.get("fill_na", None)
        v = _safe_float(value, default=float("nan"))

        if math.isnan(v):
            if fill_na is not None:
                return int(fill_na)
            return 0  # if no fill_na specified, missing values don't trigger flag

        direction = flag_def["direction"]
        threshold = flag_def["threshold"]

        if direction == "gt":
            return 1 if v > threshold else 0
        elif direction == "gte":
            return 1 if v >= threshold else 0
        elif direction == "lt":
            return 1 if v < threshold else 0
        elif direction == "lte":
            return 1 if v <= threshold else 0
        else:
            raise ValueError(f"Unknown direction '{direction}' for flag '{flag_name}'")

    def _compute_all_flags(self, features: dict) -> Dict[str, int]:
        """Compute all 9 flags for a single loan.

        Parameters
        ----------
        features : dict
            Dictionary with payment feature fields.

        Returns
        -------
        dict mapping flag name -> 0/1
        """
        flags = {}
        for flag_name, flag_def in self.flag_definitions.items():
            feature_col = flag_def["feature"]
            value = features.get(feature_col, None)
            flags[flag_name] = self._compute_flag(flag_name, flag_def, value)
        return flags

    def _compute_all_flags_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 9 flags for a DataFrame. Vectorized for performance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with payment feature columns.

        Returns
        -------
        pd.DataFrame with one column per flag (0/1 values)
        """
        flag_df = pd.DataFrame(index=df.index)

        for flag_name, flag_def in self.flag_definitions.items():
            feature_col = flag_def["feature"]
            threshold = flag_def["threshold"]
            direction = flag_def["direction"]
            fill_na = flag_def.get("fill_na", None)

            if feature_col not in df.columns:
                flag_df[flag_name] = 0
                logger.warning(f"Feature '{feature_col}' not found in DataFrame -- flag '{flag_name}' set to 0")
                continue

            series = pd.to_numeric(df[feature_col], errors="coerce")

            if direction == "gt":
                result = (series > threshold).astype(int)
            elif direction == "gte":
                result = (series >= threshold).astype(int)
            elif direction == "lt":
                result = (series < threshold).astype(int)
            elif direction == "lte":
                result = (series <= threshold).astype(int)
            else:
                raise ValueError(f"Unknown direction '{direction}' for flag '{flag_name}'")

            # Handle NaN: fill with fill_na value or 0
            if fill_na is not None:
                result = result.fillna(fill_na).astype(int)
            else:
                result = result.fillna(0).astype(int)

            flag_df[flag_name] = result

        return flag_df

    def _compute_composite_score(self, flag_dict: Dict[str, int]) -> int:
        """Compute weighted composite score from flag dict (single loan)."""
        score = 0
        for flag_name, triggered in flag_dict.items():
            if triggered and flag_name in self.flag_definitions:
                score += self.flag_definitions[flag_name].get("weight", 1)
        return score

    def _compute_composite_score_vectorized(self, flag_df: pd.DataFrame) -> pd.Series:
        """Compute weighted composite score for a DataFrame of flags."""
        weights = pd.Series(
            {fn: fd.get("weight", 1) for fn, fd in self.flag_definitions.items()}
        )
        # Only include flags that exist in the DataFrame
        common = flag_df.columns.intersection(weights.index)
        return flag_df[common].multiply(weights[common]).sum(axis=1).astype(int)

    def _assign_tier(self, flag_count: int) -> str:
        """Assign risk tier based on flag count."""
        for tier, (low, high) in self.tier_boundaries.items():
            if low <= flag_count <= high:
                return tier
        # If flag_count exceeds all defined boundaries, assign to highest tier
        return "red"

    def _assign_tier_vectorized(self, scores: pd.Series) -> pd.Series:
        """Assign risk tiers for a Series of composite scores."""
        conditions = []
        choices = []
        for tier, (low, high) in self.tier_boundaries.items():
            conditions.append((scores >= low) & (scores <= high))
            choices.append(tier)

        result = pd.Series("red", index=scores.index)  # default fallback
        # Apply in reverse order so the first matching condition wins
        for cond, choice in reversed(list(zip(conditions, choices))):
            result = result.where(~cond, choice)
        return result

    def _risk_score_0_100(self, composite_score: int) -> float:
        """Map composite score (0-9) to a 0-100 risk score.

        Linear mapping: 0 flags = 0, 9 flags = 100.
        """
        max_score = sum(fd.get("weight", 1) for fd in self.flag_definitions.values())
        if max_score == 0:
            return 0.0
        return round(100.0 * composite_score / max_score, 1)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _fit_calibrator(self, composite_scores: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression calibrator: composite_score -> P(default).

        Uses the same pattern as DefaultScorecard: isotonic regression with
        y_min=0.0, y_max=1.0, out_of_bounds="clip".
        """
        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._calibrator.fit(composite_scores.astype(float), labels.astype(float))
        logger.info("[Cal] Fitted isotonic calibrator on %d observations", len(labels))

    def _calibrate(self, composite_scores) -> np.ndarray:
        """Apply isotonic calibrator to get P(default).

        Falls back to tier-based empirical rates if calibrator is not fitted.
        """
        if self._calibrator is not None:
            arr = np.asarray(composite_scores, dtype=float).ravel()
            return np.clip(self._calibrator.predict(arr), 0.001, 0.999)

        # Fallback: use tier default rates if available
        if self._tier_default_rates is not None:
            result = []
            for score in np.asarray(composite_scores).ravel():
                tier = self._assign_tier(int(score))
                result.append(self._tier_default_rates.get(tier, 0.09))
            return np.array(result)

        # No calibration data at all -- return base rate
        return np.full(len(np.asarray(composite_scores).ravel()), 0.09)

    # ------------------------------------------------------------------
    # PUBLIC API: FIT
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, label_col: str = "is_bad"):
        """Learn tier boundaries and calibrate from historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Payment features DataFrame with label column.
        label_col : str
            Name of binary target column (1=bad, 0=good).
        """
        logger.info("Fitting PaymentMonitor on %d observations...", len(df))

        # Compute flags
        flag_df = self._compute_all_flags_vectorized(df)
        composite_scores = self._compute_composite_score_vectorized(flag_df)
        tiers = self._assign_tier_vectorized(composite_scores)
        labels = df[label_col].astype(int).values

        # --- Store training statistics ---
        n_bad = int(labels.sum())
        base_rate = float(labels.mean())

        # Score distribution
        score_counts = composite_scores.value_counts().sort_index().to_dict()
        self._score_distribution = {int(k): int(v) for k, v in score_counts.items()}

        # Tier default rates
        tier_rates = {}
        for tier in self.tier_boundaries.keys():
            mask = tiers == tier
            if mask.sum() > 0:
                tier_rates[tier] = float(labels[mask.values].mean())
            else:
                tier_rates[tier] = 0.0
        self._tier_default_rates = tier_rates

        # Flag-level default rates (loans with flag=1 vs flag=0)
        flag_rates = {}
        for flag_name in self.flag_definitions.keys():
            if flag_name in flag_df.columns:
                flagged_mask = flag_df[flag_name] == 1
                flagged_rate = float(labels[flagged_mask.values].mean()) if flagged_mask.sum() > 0 else 0.0
                unflagged_rate = float(labels[~flagged_mask.values].mean()) if (~flagged_mask).sum() > 0 else 0.0
                flag_rates[flag_name] = {
                    "flagged_rate": round(flagged_rate, 4),
                    "unflagged_rate": round(unflagged_rate, 4),
                    "flagged_count": int(flagged_mask.sum()),
                    "lift": round(flagged_rate / base_rate, 2) if base_rate > 0 else 0.0,
                }
        self._flag_default_rates = flag_rates

        # Training stats
        self._train_stats = {
            "n_obs": len(df),
            "n_bad": n_bad,
            "base_rate": round(base_rate, 4),
            "fit_date": datetime.now().isoformat(),
            "score_mean": round(float(composite_scores.mean()), 2),
            "score_std": round(float(composite_scores.std()), 2),
            "tier_distribution": {
                tier: int((tiers == tier).sum())
                for tier in self.tier_boundaries.keys()
            },
        }

        # --- Fit calibrator ---
        self._fit_calibrator(composite_scores.values, labels)

        self._fitted = True
        logger.info(
            "PaymentMonitor fitted: %d obs, %.1f%% bad rate, score mean=%.2f",
            len(df), base_rate * 100, composite_scores.mean(),
        )
        logger.info("Tier default rates: %s", tier_rates)

    # ------------------------------------------------------------------
    # PUBLIC API: PREDICT (single loan)
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> dict:
        """Score a single loan's payment health.

        Parameters
        ----------
        features : dict
            Dictionary with payment feature fields (nsf_count, nsf_rate, etc.)

        Returns
        -------
        dict with keys:
            risk_tier: 'green'|'yellow'|'orange'|'red'
            risk_score: 0-100 (composite score mapped to 0-100 scale)
            flag_count: int (number of triggered flags)
            flags: list of triggered flag descriptions
            default_probability: float (calibrated P(default))
            recommendation: str (action recommendation for this tier)
        """
        if not self._fitted:
            raise RuntimeError("PaymentMonitor has not been fitted. Call fit() first.")
        flags = self._compute_all_flags(features)
        composite = self._compute_composite_score(flags)
        tier = self._assign_tier(composite)
        risk_score = self._risk_score_0_100(composite)

        # Calibrated default probability
        default_prob = float(self._calibrate(np.array([composite]))[0])

        # Build list of triggered flags with descriptions
        triggered = []
        for flag_name, triggered_val in flags.items():
            if triggered_val:
                triggered.append(self.flag_definitions[flag_name]["description"])

        return {
            "risk_tier": tier,
            "risk_score": risk_score,
            "flag_count": composite,
            "flags": triggered,
            "flags_detail": {k: v for k, v in flags.items()},
            "default_probability": round(default_prob, 4),
            "recommendation": TIER_RECOMMENDATIONS.get(tier, "Unknown tier"),
        }

    # ------------------------------------------------------------------
    # PUBLIC API: PREDICT_BATCH (vectorized)
    # ------------------------------------------------------------------

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of loans. Vectorized for performance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with payment feature columns.

        Returns
        -------
        pd.DataFrame with columns:
            risk_tier, risk_score, flag_count, default_probability, recommendation,
            plus one column per flag (0/1).
        """
        if not self._fitted:
            raise RuntimeError("PaymentMonitor has not been fitted. Call fit() first.")
        flag_df = self._compute_all_flags_vectorized(df)
        composite_scores = self._compute_composite_score_vectorized(flag_df)
        tiers = self._assign_tier_vectorized(composite_scores)

        # Risk score 0-100
        max_score = sum(fd.get("weight", 1) for fd in self.flag_definitions.values())
        risk_scores = (100.0 * composite_scores / max_score).round(1) if max_score > 0 else composite_scores * 0.0

        # Calibrated default probabilities
        default_probs = self._calibrate(composite_scores.values)

        # Recommendations
        recommendations = tiers.map(TIER_RECOMMENDATIONS)

        # Build result DataFrame
        result = pd.DataFrame(
            {
                "risk_tier": tiers,
                "risk_score": risk_scores,
                "flag_count": composite_scores,
                "default_probability": np.round(default_probs, 4),
                "recommendation": recommendations,
            },
            index=df.index,
        )

        # Add individual flag columns
        for col in flag_df.columns:
            result[col] = flag_df[col]

        return result

    # ------------------------------------------------------------------
    # PUBLIC API: EVALUATE
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame, label_col: str = "is_bad") -> dict:
        """Evaluate on labeled data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with payment features and label column.
        label_col : str
            Name of binary target column.

        Returns
        -------
        dict with keys: auc, gini, ks, brier, tier_stats, flag_stats,
            confusion_matrix, n_obs, n_bad, base_rate
        """
        results = self.predict_batch(df)
        y_true = df[label_col].astype(int).values

        # AUC on composite score
        try:
            auc = float(roc_auc_score(y_true, results["flag_count"]))
        except ValueError:
            auc = float("nan")

        # AUC on calibrated probability
        try:
            auc_cal = float(roc_auc_score(y_true, results["default_probability"]))
        except ValueError:
            auc_cal = float("nan")

        # Gini
        gini = 2 * auc - 1

        # KS statistic
        try:
            fpr, tpr, _ = roc_curve(y_true, results["flag_count"])
            ks = float(np.max(tpr - fpr))
        except ValueError:
            ks = float("nan")

        # Brier score on calibrated probabilities
        try:
            brier = float(brier_score_loss(y_true, results["default_probability"]))
        except ValueError:
            brier = float("nan")

        # Tier statistics
        tier_stats = {}
        for tier in self.tier_boundaries.keys():
            mask = results["risk_tier"] == tier
            n_tier = int(mask.sum())
            if n_tier > 0:
                tier_bad_rate = float(y_true[mask.values].mean())
                tier_bad_count = int(y_true[mask.values].sum())
            else:
                tier_bad_rate = 0.0
                tier_bad_count = 0
            tier_stats[tier] = {
                "n_loans": n_tier,
                "pct_portfolio": round(n_tier / len(df), 4),
                "bad_rate": round(tier_bad_rate, 4),
                "n_bad": tier_bad_count,
            }

        # Flag-level statistics
        flag_stats = {}
        for flag_name in self.flag_definitions.keys():
            if flag_name in results.columns:
                flagged = results[flag_name] == 1
                n_flagged = int(flagged.sum())
                if n_flagged > 0:
                    flagged_bad_rate = float(y_true[flagged.values].mean())
                else:
                    flagged_bad_rate = 0.0
                base_rate = float(y_true.mean())
                lift = flagged_bad_rate / base_rate if base_rate > 0 else 0.0
                flag_stats[flag_name] = {
                    "n_flagged": n_flagged,
                    "flagged_pct": round(n_flagged / len(df), 4),
                    "bad_rate": round(flagged_bad_rate, 4),
                    "lift": round(lift, 2),
                }

        # Confusion matrix at each tier boundary
        # Use "red" tier as the positive prediction for confusion matrix
        red_mask = (results["risk_tier"] == "red").astype(int).values
        try:
            cm = confusion_matrix(y_true, red_mask).tolist()
        except ValueError:
            cm = None

        return {
            "auc": round(auc, 4),
            "auc_calibrated": round(auc_cal, 4),
            "gini": round(gini, 4),
            "ks": round(ks, 4),
            "brier": round(brier, 4),
            "tier_stats": tier_stats,
            "flag_stats": flag_stats,
            "confusion_matrix_red": cm,
            "n_obs": len(y_true),
            "n_bad": int(y_true.sum()),
            "base_rate": round(float(y_true.mean()), 4),
        }

    # ------------------------------------------------------------------
    # PUBLIC API: PORTFOLIO SUMMARY
    # ------------------------------------------------------------------

    def get_portfolio_summary(self, df: pd.DataFrame) -> dict:
        """Aggregate portfolio health: tier distribution, avg risk, trends.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with payment features. May optionally include
            'signing_date' for trend analysis.

        Returns
        -------
        dict with keys: total_loans, tier_distribution, avg_risk_score,
            avg_default_probability, flag_prevalence, score_histogram,
            monthly_trends (if signing_date present)
        """
        results = self.predict_batch(df)

        # Tier distribution
        tier_dist = {}
        for tier in self.tier_boundaries.keys():
            mask = results["risk_tier"] == tier
            tier_dist[tier] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.mean()), 4),
                "avg_default_prob": round(float(results.loc[mask, "default_probability"].mean()), 4) if mask.sum() > 0 else 0.0,
            }

        # Flag prevalence
        flag_prevalence = {}
        for flag_name in self.flag_definitions.keys():
            if flag_name in results.columns:
                flag_prevalence[flag_name] = {
                    "count": int((results[flag_name] == 1).sum()),
                    "pct": round(float((results[flag_name] == 1).mean()), 4),
                }

        # Score histogram
        score_hist = results["flag_count"].value_counts().sort_index().to_dict()
        score_hist = {int(k): int(v) for k, v in score_hist.items()}

        summary = {
            "total_loans": len(df),
            "tier_distribution": tier_dist,
            "avg_risk_score": round(float(results["risk_score"].mean()), 2),
            "avg_flag_count": round(float(results["flag_count"].mean()), 2),
            "avg_default_probability": round(float(results["default_probability"].mean()), 4),
            "flag_prevalence": flag_prevalence,
            "score_histogram": score_hist,
        }

        # Monthly trends (if signing_date is available)
        if "signing_date" in df.columns:
            df_with_results = df[["signing_date"]].copy()
            df_with_results["risk_tier"] = results["risk_tier"].values
            df_with_results["flag_count"] = results["flag_count"].values
            df_with_results["default_probability"] = results["default_probability"].values
            df_with_results["signing_month"] = pd.to_datetime(df_with_results["signing_date"]).dt.to_period("M").astype(str)

            monthly = df_with_results.groupby("signing_month").agg(
                n_loans=("flag_count", "count"),
                avg_flags=("flag_count", "mean"),
                avg_default_prob=("default_probability", "mean"),
                pct_red=("risk_tier", lambda x: (x == "red").mean()),
                pct_orange_plus=("risk_tier", lambda x: (x.isin(["orange", "red"])).mean()),
            ).round(4)

            summary["monthly_trends"] = monthly.to_dict(orient="index")

        return summary

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialize PaymentMonitor to disk with joblib.

        Parameters
        ----------
        path : str or Path
            File path for the serialized model.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": self.version,
            "tier_boundaries": self.tier_boundaries,
            "flag_definitions": self.flag_definitions,
            "fitted": self._fitted,
            "calibrator": self._calibrator,
            "train_stats": self._train_stats,
            "tier_default_rates": self._tier_default_rates,
            "flag_default_rates": self._flag_default_rates,
            "score_distribution": self._score_distribution,
        }
        joblib.dump(state, path)
        logger.info("Saved PaymentMonitor to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PaymentMonitor":
        """Deserialize a PaymentMonitor from disk.

        Parameters
        ----------
        path : str or Path
            Path to the joblib file.

        Returns
        -------
        PaymentMonitor instance with restored state.
        """
        state = joblib.load(path)

        # Version check (warn, don't error, for backward compatibility)
        saved_version = state.get("version", "unknown")
        if saved_version != cls.VERSION:
            logger.warning(
                "PaymentMonitor version mismatch: saved=%s, current=%s. "
                "Model behavior may differ.",
                saved_version, cls.VERSION,
            )

        pm = cls.__new__(cls)
        pm.version = state.get("version", "1.0.0")
        pm.tier_boundaries = state.get("tier_boundaries", DEFAULT_TIER_BOUNDARIES.copy())
        pm.flag_definitions = state.get("flag_definitions", FLAG_DEFINITIONS.copy())
        pm._fitted = state.get("fitted", False)
        pm._calibrator = state.get("calibrator")
        pm._train_stats = state.get("train_stats")
        pm._tier_default_rates = state.get("tier_default_rates")
        pm._flag_default_rates = state.get("flag_default_rates")
        pm._score_distribution = state.get("score_distribution")
        return pm

    # ------------------------------------------------------------------
    # REPR
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        if self._fitted and self._train_stats:
            return (
                f"PaymentMonitor(v{self.version}, {status}, "
                f"n={self._train_stats['n_obs']}, "
                f"bad_rate={self._train_stats['base_rate']:.1%}, "
                f"9 flags)"
            )
        return f"PaymentMonitor(v{self.version}, {status})"
