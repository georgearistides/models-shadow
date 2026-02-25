#!/usr/bin/env python3
"""
DefaultScorecard — Blended Default Probability Model
=====================================================
Combines three sub-models (WoE Scorecard, Rule Engine, XGBoost) into a
calibrated probability-of-default (PD) estimate with a 300-850 score mapping.

Usage:
    from default_scorecard import DefaultScorecard

    ds = DefaultScorecard()
    ds.fit(train_df, y_col="is_bad")

    result = ds.predict({"fico": 620, "qi": "MISSING", "partner": "PAYSAFE"})
    # -> {"pd": 0.28, "score": 458, "components": {"woe_pd": ..., "rule_pd": ..., "xgb_pd": ...}}

    results_df = ds.predict_batch(test_df)
    metrics = ds.evaluate(test_df, y_col="is_bad")

    ds.save("models/default_scorecard.joblib")
    ds2 = DefaultScorecard.load("models/default_scorecard.joblib")
"""

import logging
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from model_utils import PARTNER_ENCODING

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# =============================================================================
# CONSTANTS
# =============================================================================

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
    # Entity graph features (already short names in data)
    "has_prior_bad": "has_prior_bad",
    "is_repeat": "is_repeat",
    "is_cross_entity": "is_cross_entity",
    "prior_loans_ssn": "prior_loans_ssn",
    "prior_shops_ssn": "prior_shops_ssn",
    # Banking velocity features
    "avg_balance_30d": "avg_balance_30d",
}

# Short names expected by the model internals
SHORT_NAMES = [
    "fico", "d30", "d60", "inq6", "revutil", "pastdue",
    "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
    "mopmt", "crhist", "qi", "partner",
    "has_prior_bad", "is_repeat", "is_cross_entity", "prior_loans_ssn",
    "prior_shops_ssn",
    "avg_balance_30d",
]

# Blend weights
BLEND_WEIGHTS = {"woe": 0.25, "rule": 0.25, "xgb": 0.50}

# Expert bins for WoE computation (domain knowledge > data quantiles)
EXPERT_WOE_BINS = {
    "fico": [0, 570, 600, 620, 650, 670, 700, 740, 800, 900],
}

# WoE feature candidates (order matters for IV selection)
# NOTE: avg_balance_30d removed in V5.2 — real signal but 19.3% coverage causes
# MISSING-bin WoE noise that hurts more than it helps (net -0.013 blend AUC).
# Re-add when coverage reaches ~40%+. Config kept in COLUMN_MAP/GRADER_CONFIGS.
# partner_encoded removed for fair lending compliance (iter 34, AUC impact -0.0014)
WOE_FEATURE_CANDIDATES = [
    "fico", "d30", "d60", "inq6", "revutil", "pastdue",
    "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
    "mopmt", "crhist",
    "qi_encoded", "fico_qi_interaction",
    "paid_ratio", "delin_rate", "bureau_health", "cash_stress",
    "is_repeat", "has_prior_bad",
]

# XGBoost feature list (20 features — pruned 4 zero-SHAP + avg_balance_30d)
# NOTE: avg_balance_30d removed in V5.2 — see WOE_FEATURE_CANDIDATES comment above.
XGB_FEATURES = [
    "fico", "d30", "d60", "inq6", "revutil", "pastdue",
    "instbal", "revbal", "tl_total", "tl_paid",
    "mopmt",
    "qi_encoded", "qi_missing", "fico_qi_interaction",
    "paid_ratio", "delin_rate", "delin_severity", "bureau_health",
    "cash_stress", "util_distance",
]

# Monotonic constraints for XGBoost (pruned 4 zero-SHAP features: has_prior_bad, is_repeat, crhist, tl_delin)
MONOTONE_CONSTRAINTS_MAP = {
    "fico": -1,
    "d30": 1, "d60": 1, "inq6": 1, "pastdue": 1,
    "qi_encoded": 1, "qi_missing": 1,
    "delin_rate": 1, "cash_stress": 1, "delin_severity": 1,
    "bureau_health": -1, "paid_ratio": -1,
    "tl_total": -1, "tl_paid": -1,
    "avg_balance_30d": -1,  # Higher balance = lower risk
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 200,
    "min_child_weight": 50,
    "subsample": 0.9,
    "colsample_bytree": 0.6,
    "reg_lambda": 8.0,
    "reg_alpha": 1.0,
    "eval_metric": "auc",
    "random_state": 42,
}

# Rule engine boundary grader configs
GRADER_CONFIGS = {
    "fico": {"boundaries": [570, 600, 620, 650, 670, 700, 740, 770, 800], "lower_is_riskier": True, "if_missing": 7},
    "d30": {"boundaries": [1, 2, 4, 8, 12], "lower_is_riskier": False, "if_missing": 5},
    "d60": {"boundaries": [1, 2, 4, 8], "lower_is_riskier": False, "if_missing": 5},
    "inq6": {"boundaries": [1, 3, 5, 8, 12], "lower_is_riskier": False, "if_missing": 3},
    "revutil": {"boundaries": [25, 50, 75, 90], "lower_is_riskier": False, "if_missing": 5},
    "tl_total": {"boundaries": [3, 8, 15, 25, 40], "lower_is_riskier": True, "if_missing": 8},
    "crhist": {"boundaries": [365, 1095, 1825, 3650, 5475, 7300], "lower_is_riskier": True, "if_missing": 7},
    "mopmt": {"boundaries": [100, 300, 500, 1000, 2000], "lower_is_riskier": False, "if_missing": 5},
    # Entity graph: is_repeat boundary at 1 (0 = first-timer = riskier, 1 = repeat = safer)
    "is_repeat": {"boundaries": [1], "lower_is_riskier": True, "if_missing": 5},
    "avg_balance_30d": {"boundaries": [500, 2000, 5000, 10000, 25000], "lower_is_riskier": True, "if_missing": 7},
}

# has_prior_bad is handled specially in the rule engine (binary flag -> max risk)
HAS_PRIOR_BAD_RISK_SCORE = 10  # Maximum risk score when borrower had a prior bad loan

QI_RISK = {"HFHT": 1, "LFHT": 3, "HFLT": 3, "LFLT": 4, "MISSING": 10}
# PARTNER_RISK removed for fair lending compliance (iter 34, AUC impact -0.0014)


# =============================================================================
# BOUNDARY GRADER (for rule engine)
# =============================================================================

class BoundaryGrader:
    """Grade a numeric value to a 1-10 risk score using sorted boundary bins.

    NOTE: This is distinct from credit_grader.BoundaryGrader, which maps PD
    to letter grades. This version maps raw feature values to integer risk
    scores (1-10) for the rule engine sub-model.
    """

    def __init__(self, boundaries, lower_is_riskier=True, if_missing=10):
        self.boundaries = sorted(boundaries)
        self.lower_is_riskier = lower_is_riskier
        self.if_missing = if_missing

    def grade(self, value):
        if pd.isna(value):
            return self.if_missing
        if self.lower_is_riskier:
            # Higher value = better (lower score). E.g., FICO.
            score = len(self.boundaries) + 1
            for i, boundary in enumerate(reversed(self.boundaries)):
                if value >= boundary:
                    score = i + 1
                    break
            return score
        else:
            # Higher value = worse (higher score). E.g., delinquencies.
            score = 1
            for i, boundary in enumerate(self.boundaries):
                if value >= boundary:
                    score = i + 2
                else:
                    break
            return score

    def grade_series(self, series):
        # Ensure numeric dtype (banking parquet may store as string/object)
        if series.dtype == object:
            series = pd.to_numeric(series, errors="coerce")
        return series.apply(self.grade)


# =============================================================================
# DEFAULT SCORECARD CLASS
# =============================================================================

class DefaultScorecard:
    """
    Blended default probability scorecard with 3 sub-models:
      1. WoE Scorecard (LogisticRegressionCV on WoE-transformed features)
      2. Rule Engine (boundary graders + weighted composite + logistic mapping)
      3. XGBoost (constrained with monotonic constraints)

    Final PD = weighted blend of sub-model PDs, with optional Platt calibration.
    Score = 300 + (1 - PD) * 550  (300-850 range, higher = safer).
    """

    VERSION = "1.6.0"  # partner_encoded removed for fair lending compliance (iter 34, AUC impact -0.0014)

    def __init__(self, blend_weights=None):
        self.blend_weights = blend_weights or dict(BLEND_WEIGHTS)
        self._fitted = False

        # Sub-model artifacts (populated by .fit())
        self._woe_model = None          # LogisticRegressionCV
        self._woe_transforms = {}       # {feature: (woe_dict, bin_edges)}
        self._woe_features = []         # Selected features (by IV)
        self._woe_iv_scores = {}        # {feature: IV}

        self._rule_graders = {}         # {feature_name: BoundaryGrader}
        self._rule_logistic = None      # LogisticRegression mapping score -> PD

        self._xgb_model = None          # XGBClassifier
        self._xgb_features = []         # Feature list used

        self._calibrator = None         # IsotonicRegression (v1.1+) or LogisticRegression (v1.0)
        self._platt_calibrator = None   # Legacy Platt scaling (kept for comparison)
        self._train_base_rate = None    # For sanity checks

        # v1.1: PAYSAFE post-hoc calibration multiplier
        self._paysafe_multiplier = None  # float, computed from training data
        self._paysafe_train_stats = None # dict with empirical stats used to derive multiplier

    # =========================================================================
    # COLUMN MAPPING
    # =========================================================================

    def _map_columns(self, df):
        """Map long Experian column names to short names if needed."""
        df = df.copy()
        # Check if ALL expected short names already exist — only then skip mapping
        if all(name in df.columns for name in SHORT_NAMES):
            return df

        # Apply column mapping
        rename = {}
        for long_name, short_name in COLUMN_MAP.items():
            if long_name in df.columns and short_name not in df.columns:
                rename[long_name] = short_name
        if rename:
            df = df.rename(columns=rename)
        return df

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    def _engineer_features(self, df):
        """Compute all derived features from raw inputs."""
        df = df.copy()

        # Defensive initialization: ensure all required bureau columns exist
        # (fill with NaN if entirely missing, so downstream code doesn't KeyError)
        numeric_cols = [
            "fico", "d30", "d60", "inq6", "revutil", "pastdue",
            "instbal", "revbal", "tl_total", "tl_paid", "tl_delin",
            "mopmt", "crhist",
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Ensure numeric types for bureau fields (handle Decimal from Spark)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # QI encoding (ordinal by risk: HFHT=0 best, MISSING=4 worst)
        qi_map = {"HFHT": 0, "HFLT": 1, "LFHT": 2, "LFLT": 3, "MISSING": 4}
        if "qi" in df.columns:
            df["qi_encoded"] = df["qi"].map(qi_map).fillna(4).astype(float)
        else:
            df["qi_encoded"] = 4.0
        df["qi_missing"] = (df["qi_encoded"] == 4).astype(int)

        # Partner encoding (canonical: higher = riskier, based on default rates)
        if "partner" in df.columns:
            df["partner_encoded"] = df["partner"].map(PARTNER_ENCODING).fillna(1).astype(float)
        else:
            df["partner_encoded"] = 1.0

        # FICO x QI interaction (the dominant signal: 15.4x lift)
        fico_bins = pd.cut(
            df["fico"],
            bins=[0, 550, 600, 650, 700, 900],
            labels=[0, 1, 2, 3, 4],
        )
        df["fico_qi_interaction"] = fico_bins.astype(float).fillna(2) * 5 + df["qi_encoded"]

        # Bureau ratios (safe division)
        df["paid_ratio"] = (df["tl_paid"].fillna(0) / df["tl_total"].fillna(1).clip(lower=1)).clip(0, 1)
        df["delin_rate"] = (df["tl_delin"].fillna(0) / df["tl_total"].fillna(1).clip(lower=1)).clip(0, 1)
        df["delin_severity"] = df["d30"].fillna(0) + 2 * df["d60"].fillna(0)

        # Bureau health composite
        zero_current_delin = (df["tl_delin"].fillna(0) == 0).astype(float)
        df["bureau_health"] = (
            zero_current_delin * 0.4
            + df["paid_ratio"] * 0.3
            + (1 - df["delin_rate"]) * 0.3
        )

        # Cash stress indicator
        df["cash_stress"] = (
            (df["fico"].fillna(700) < 580).astype(int) * 2
            + (df["pastdue"].fillna(0) > 0).astype(int)
            + (df["revutil"].fillna(0) > 80).astype(int)
            + (df["d60"].fillna(0) > 0).astype(int)
        ).astype(float)

        # Utilization distance from optimal midpoint
        df["util_distance"] = (df["revutil"].fillna(50) - 50).abs()

        # Any delinquency flag
        df["any_delin"] = (
            (df["d30"].fillna(0) > 0) | (df["d60"].fillna(0) > 0)
        ).astype(int)

        # Entity graph features: pass through as-is, default 0 if missing
        # (0 = no prior history known, which is the safe/neutral assumption)
        entity_cols = ["has_prior_bad", "is_repeat", "is_cross_entity",
                       "prior_loans_ssn", "prior_shops_ssn"]
        for col in entity_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0).astype(int)

        # 3-level prior history feature (available for analysis, not used in WoE)
        # Encoding: prior_BAD (42.86% bad), none (~9%), prior_PAID (~8%)
        # NOTE: prior_history was tested in WoE (iter 14) but REVERTED —
        # the 3-level encoding reduces WoE AUC from 0.6611 to 0.6455. The prior_PAID
        # category adds opposing WoE signal that confuses the logistic regression
        # even though IV is higher (0.041 vs 0.036). Binary has_prior_bad is retained.
        df["prior_history"] = "none"
        df.loc[df["is_repeat"] == 1, "prior_history"] = "prior_PAID"
        df.loc[df["has_prior_bad"] == 1, "prior_history"] = "prior_BAD"

        return df

    # =========================================================================
    # SUB-MODEL 1: WoE SCORECARD
    # =========================================================================

    def _compute_woe(self, feature_series, target, n_bins=6, feature_name=None):
        """Compute Weight of Evidence bins for a single feature.

        Returns:
            woe_dict: {bin_label_str: woe_value}
            bin_edges: numpy array of edges (None for categoricals)
            iv: Information Value (scalar)
        """
        bin_edges = None
        used_expert_bins = False

        # Use expert bins if defined for this feature
        if feature_name and feature_name in EXPERT_WOE_BINS:
            series_clean = feature_series.dropna()
            try:
                binned = pd.cut(series_clean, bins=EXPERT_WOE_BINS[feature_name],
                                include_lowest=True, duplicates='drop')
                bins = binned.astype(str)
                # Reindex to match original series, filling NaN with "MISSING"
                bins = bins.reindex(feature_series.index, fill_value="MISSING")
                bins = bins.fillna("MISSING")
                bin_edges = np.array(EXPERT_WOE_BINS[feature_name], dtype=float)
                used_expert_bins = True
                logger.debug(f"Using expert bins for '{feature_name}': {EXPERT_WOE_BINS[feature_name]}")
            except Exception as e:
                logger.warning(f"Expert bins failed for '{feature_name}': {e}, falling back to quantile bins")
                used_expert_bins = False

        if not used_expert_bins:
            if feature_series.dtype == "object" or feature_series.nunique() <= 10:
                bins = feature_series.fillna("MISSING").astype(str)
            else:
                try:
                    bins, bin_edges = pd.qcut(
                        feature_series, q=n_bins, duplicates="drop", retbins=True
                    )
                    bins = bins.astype(str).fillna("MISSING")
                except (ValueError, TypeError):
                    try:
                        bins, bin_edges = pd.cut(
                            feature_series, bins=n_bins, duplicates="drop", retbins=True
                        )
                        bins = bins.astype(str).fillna("MISSING")
                    except (ValueError, TypeError):
                        bins = feature_series.fillna(0).astype(str)

        total_good = (target == 0).sum()
        total_bad = (target == 1).sum()

        woe_dict = {}
        iv = 0.0

        for bin_val in bins.unique():
            mask = bins == bin_val
            n_good = ((target == 0) & mask).sum()
            n_bad = ((target == 1) & mask).sum()

            pct_good = max(n_good, 0.5) / total_good
            pct_bad = max(n_bad, 0.5) / total_bad

            woe = np.log(pct_good / pct_bad)
            iv += (pct_good - pct_bad) * woe

            woe_dict[bin_val] = woe

        return woe_dict, bin_edges, iv

    def _apply_woe_transform(self, series, feature_name):
        """Apply stored WoE transform to a feature series."""
        woe_dict, bin_edges = self._woe_transforms[feature_name]

        if series.dtype == "object" or series.nunique() <= 10:
            bins = series.fillna("MISSING").astype(str)
        elif bin_edges is not None:
            edges = bin_edges.copy()
            edges[0] = -np.inf
            edges[-1] = np.inf
            bins = pd.cut(series, bins=edges, duplicates="drop").astype(str).fillna("MISSING")
        else:
            bins = series.fillna(0).astype(str)

        return bins.map(woe_dict).fillna(0).astype(float)

    def _fit_woe_scorecard(self, df, target):
        """Train the WoE scorecard sub-model."""
        candidates = [f for f in WOE_FEATURE_CANDIDATES if f in df.columns]

        # Compute IV for each feature and store WoE transforms
        iv_scores = {}
        for feat in candidates:
            try:
                woe_dict, bin_edges, iv = self._compute_woe(df[feat], target, n_bins=6, feature_name=feat)
                iv_scores[feat] = iv
                self._woe_transforms[feat] = (woe_dict, bin_edges)
            except Exception as e:
                logger.warning(f"WoE computation failed for feature '{feat}': {e}")
                continue

        self._woe_iv_scores = iv_scores

        # Select features with IV >= 0.02, up to 15
        selected = sorted(
            [(f, iv) for f, iv in iv_scores.items() if iv >= 0.02],
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        self._woe_features = [f for f, _ in selected]

        if not self._woe_features:
            # Fallback: take top 5 by IV regardless of threshold
            selected = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            self._woe_features = [f for f, _ in selected]

        # Build WoE-transformed training matrix
        X_woe = pd.DataFrame(index=df.index)
        for feat in self._woe_features:
            X_woe[f"{feat}_woe"] = self._apply_woe_transform(df[feat], feat)
        X_woe = X_woe.fillna(0)

        # Fit logistic regression
        self._woe_model = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            penalty="l2",
            cv=3,
            scoring="roc_auc",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        self._woe_model.fit(X_woe, target)

    def _predict_woe(self, df):
        """Produce WoE sub-model PD for a DataFrame."""
        X_woe = pd.DataFrame(index=df.index)
        for feat in self._woe_features:
            if feat in df.columns:
                X_woe[f"{feat}_woe"] = self._apply_woe_transform(df[feat], feat)
            else:
                X_woe[f"{feat}_woe"] = 0.0
        X_woe = X_woe.fillna(0)
        return self._woe_model.predict_proba(X_woe)[:, 1]

    # =========================================================================
    # SUB-MODEL 2: RULE ENGINE
    # =========================================================================

    def _fit_rule_engine(self, df, target):
        """Build boundary graders and fit the logistic mapping."""
        # Create graders from config
        self._rule_graders = {
            name: BoundaryGrader(**cfg)
            for name, cfg in GRADER_CONFIGS.items()
        }

        # Compute composite scores for training set
        scores = self._compute_rule_scores(df)

        # Fit logistic mapping: composite score (1-10) -> PD
        self._rule_logistic = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )
        self._rule_logistic.fit(scores.values.reshape(-1, 1), target)

    def _compute_rule_scores(self, df):
        """Compute the weighted composite rule score for a DataFrame."""
        s = pd.DataFrame(index=df.index)

        # Grade numeric features
        for name, grader in self._rule_graders.items():
            if name in df.columns:
                s[f"{name}_score"] = grader.grade_series(df[name])
            else:
                s[f"{name}_score"] = grader.if_missing

        # Categorical scores
        s["qi_score"] = df["qi"].map(QI_RISK).fillna(10) if "qi" in df.columns else 10
        # partner_score removed for fair lending compliance (iter 34)

        # Delinquency composite
        delin_score = s["d30_score"] * 0.6 + s["d60_score"] * 0.4

        # Credit depth composite
        credit_depth_score = s["tl_total_score"] * 0.5 + s["crhist_score"] * 0.5

        # Entity graph scores
        # has_prior_bad: binary flag -> maximum risk score (10) if True, else neutral (1)
        if "has_prior_bad" in df.columns:
            s["prior_bad_score"] = df["has_prior_bad"].apply(
                lambda x: HAS_PRIOR_BAD_RISK_SCORE if x == 1 else 1
            ).astype(float)
        else:
            s["prior_bad_score"] = 1.0

        # Sub-composites
        credit_profile = 0.4 * s["fico_score"] + 0.3 * delin_score + 0.3 * credit_depth_score
        # partner_score removed; reweight QI 0.65, inquiries 0.35
        business_health = 0.65 * s["qi_score"] + 0.35 * s["inq6_score"]
        bureau_detail = (
            0.25 * s["revutil_score"]
            + 0.25 * s["mopmt_score"]
            + 0.25 * s["tl_total_score"]
            + 0.25 * s["crhist_score"]
        )

        # Entity history composite (repeat borrower + prior bad loan)
        # is_repeat graded by BoundaryGrader above; has_prior_bad handled specially
        entity_score = 0.5 * s.get("is_repeat_score", pd.Series(5.0, index=df.index)) + 0.5 * s["prior_bad_score"]

        # Final composite (entity_history gets 10% weight, taken from business_health)
        final = (
            0.40 * credit_profile
            + 0.25 * business_health
            + 0.25 * bureau_detail
            + 0.10 * entity_score
        )
        return final

    def _predict_rule(self, df):
        """Produce rule engine sub-model PD for a DataFrame."""
        scores = self._compute_rule_scores(df)
        return self._rule_logistic.predict_proba(scores.values.reshape(-1, 1))[:, 1]

    # =========================================================================
    # SUB-MODEL 3: XGBOOST
    # =========================================================================

    def _fit_xgboost(self, df, target):
        """Train constrained XGBoost sub-model."""
        import xgboost as xgb

        available = [f for f in XGB_FEATURES if f in df.columns]
        self._xgb_features = available

        X = df[available].copy()
        # Fill NaN with sensible defaults
        X = X.fillna(X.median())

        # Build monotonic constraint tuple
        mono = tuple(MONOTONE_CONSTRAINTS_MAP.get(f, 0) for f in available)

        params = dict(XGB_PARAMS)
        params["monotone_constraints"] = mono

        self._xgb_model = xgb.XGBClassifier(**params)
        self._xgb_model.fit(X, target)

        # Store median fill values for prediction time
        self._xgb_fill_values = X.median().to_dict()

    def _predict_xgb(self, df):
        """Produce XGBoost sub-model PD for a DataFrame."""
        # Select only features present in df; fill missing columns with training medians
        present = [f for f in self._xgb_features if f in df.columns]
        missing = [f for f in self._xgb_features if f not in df.columns]
        X = df[present].copy() if present else pd.DataFrame(index=df.index)
        for col in missing:
            X[col] = self._xgb_fill_values.get(col, 0)
        # Fill NaN with training medians
        for col in self._xgb_features:
            fill_val = self._xgb_fill_values.get(col, 0)
            X[col] = X[col].fillna(fill_val)
        # Ensure column order matches training
        X = X[self._xgb_features]
        # Coerce any object columns to numeric (banking parquet may have string types)
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(
                    self._xgb_fill_values.get(col, 0)
                )
        return self._xgb_model.predict_proba(X)[:, 1]

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    def _fit_calibrator(self, df, target):
        """Fit isotonic regression calibrator using cross-validated OOF predictions.

        v1.5 change: Uses 5-fold cross-validated out-of-fold (OOF) predictions to
        avoid overfitting the isotonic calibrator to the training data. Previously
        (v1.1-v1.4), the isotonic was fit on in-sample predictions, which caused
        the step function to memorize training noise rather than learn the true
        calibration curve. With OOF predictions, the isotonic sees predictions
        made on held-out folds, producing a more generalizable mapping.

        The final calibrator is then re-fit on the full OOF predictions -> targets,
        which gives a robust non-parametric mapping from raw blend scores to PDs.
        """
        from sklearn.model_selection import KFold

        woe_pd = self._predict_woe(df)
        rule_pd = self._predict_rule(df)
        xgb_pd = self._predict_xgb(df)

        raw_blend = (
            self.blend_weights["woe"] * woe_pd
            + self.blend_weights["rule"] * rule_pd
            + self.blend_weights["xgb"] * xgb_pd
        )

        # Cross-validated OOF predictions for calibration
        # Each fold: train sub-models on fold's training portion, predict on fold's
        # held-out portion. This prevents the isotonic from seeing in-sample predictions.
        n = len(df)
        oof_raw_blend = np.full(n, np.nan)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        target_arr = target.values
        df_indexed = df.reset_index(drop=True)
        target_indexed = pd.Series(target_arr)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_indexed)):
            fold_train = df_indexed.iloc[train_idx]
            fold_target = target_indexed.iloc[train_idx]
            fold_val = df_indexed.iloc[val_idx]

            # Train temporary sub-models on this fold's training data
            fold_ds = DefaultScorecard(blend_weights=dict(self.blend_weights))

            # Copy the transforms and models, then retrain WoE + rule + XGB on fold
            try:
                fold_ds._fit_woe_scorecard(fold_train, fold_target)
                fold_ds._fit_rule_engine(fold_train, fold_target)
                fold_ds._fit_xgboost(fold_train, fold_target)

                # Predict on held-out fold
                fold_woe = fold_ds._predict_woe(fold_val)
                fold_rule = fold_ds._predict_rule(fold_val)
                fold_xgb = fold_ds._predict_xgb(fold_val)

                fold_blend = (
                    self.blend_weights["woe"] * fold_woe
                    + self.blend_weights["rule"] * fold_rule
                    + self.blend_weights["xgb"] * fold_xgb
                )
                oof_raw_blend[val_idx] = fold_blend
            except Exception as e:
                logger.warning(f"CV calibration fold {fold_idx} failed: {e}. "
                               "Using in-sample predictions for this fold.")
                oof_raw_blend[val_idx] = raw_blend.values[val_idx] if hasattr(raw_blend, 'values') else raw_blend[val_idx]

        # Check for any NaN (shouldn't happen but be defensive)
        nan_mask = np.isnan(oof_raw_blend)
        if nan_mask.any():
            logger.warning(f"CV calibration: {nan_mask.sum()} NaN OOF predictions, "
                           "falling back to in-sample for those.")
            raw_blend_arr = raw_blend.values if hasattr(raw_blend, 'values') else raw_blend
            oof_raw_blend[nan_mask] = raw_blend_arr[nan_mask]

        # Fit isotonic on OOF predictions (not in-sample)
        self._calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._calibrator.fit(oof_raw_blend, target_arr)

        logger.info(f"[Cal] Isotonic fit on {n} OOF predictions, "
                    f"raw blend range: [{oof_raw_blend.min():.4f}, {oof_raw_blend.max():.4f}]")

        # Keep Platt scaling for comparison / fallback
        self._platt_calibrator = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )
        self._platt_calibrator.fit(oof_raw_blend.reshape(-1, 1), target_arr)

    def _calibrate(self, raw_pd):
        """Apply isotonic regression calibration to raw blended PD.

        v1.5: uses IsotonicRegression fitted on cross-validated OOF predictions.
        Isotonic regression preserves rank ordering while allowing calibrated PDs
        to go as low as the data supports (no artificial sigmoid floor).
        """
        if self._calibrator is None:
            return raw_pd
        arr = np.asarray(raw_pd).ravel()
        calibrated = self._calibrator.predict(arr)
        # Clip to valid probability range (defensive)
        return np.clip(calibrated, 0.001, 0.999)

    def recalibrate(self, df, y_col="is_bad"):
        """Recalibrate the isotonic calibrator on a new dataset without retraining sub-models.

        This is useful when:
          - The original calibration was done on right-censored data and you now
            have a fully seasoned cohort with reliable bad rates.
          - You want to calibrate to a specific population (e.g., 2024H1 holdout).
          - The portfolio mix has shifted and PDs need recalibration to new base rates.

        The sub-models (WoE, Rule, XGB) are NOT retrained -- only the isotonic
        calibrator mapping from raw blend scores to PDs is updated.

        Args:
            df: DataFrame with features and target column. Should be a fully
                seasoned cohort for best results.
            y_col: Name of the binary target column.

        Returns:
            dict with recalibration diagnostics (before/after mean PD, Brier, etc.)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Map columns and engineer features
        df_mapped = self._map_columns(df.copy())

        # Normalize categorical inputs
        for col in ("qi", "shop_qi", "partner"):
            if col in df_mapped.columns:
                mask = df_mapped[col].notna()
                df_mapped.loc[mask, col] = df_mapped.loc[mask, col].astype(str).str.upper()

        df_eng = self._engineer_features(df_mapped)
        target = df_eng[y_col].astype(int)

        # Get raw blend scores from existing sub-models
        woe_pd = self._predict_woe(df_eng)
        rule_pd = self._predict_rule(df_eng)
        xgb_pd = self._predict_xgb(df_eng)

        raw_blend = (
            self.blend_weights["woe"] * woe_pd
            + self.blend_weights["rule"] * rule_pd
            + self.blend_weights["xgb"] * xgb_pd
        )

        # Capture before-recalibration metrics
        old_calibrated = self._calibrate(raw_blend)
        old_mean_pd = float(old_calibrated.mean())
        old_brier = float(brier_score_loss(target.values, old_calibrated))

        # Fit new isotonic calibrator on this dataset
        new_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        new_calibrator.fit(raw_blend, target.values)

        # Capture after-recalibration metrics
        new_calibrated = np.clip(new_calibrator.predict(np.asarray(raw_blend).ravel()), 0.001, 0.999)
        new_mean_pd = float(new_calibrated.mean())
        new_brier = float(brier_score_loss(target.values, new_calibrated))

        # Replace the calibrator
        old_calibrator = self._calibrator
        self._calibrator = new_calibrator

        # Also re-fit Platt for comparison
        self._platt_calibrator = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )
        self._platt_calibrator.fit(np.asarray(raw_blend).reshape(-1, 1), target)

        # Also re-fit PAYSAFE multiplier on this data
        self._fit_paysafe_multiplier(df_eng, target)

        # Update train base rate
        self._train_base_rate = float(target.mean())

        actual_rate = float(target.mean())

        diagnostics = {
            "n_samples": len(df),
            "actual_bad_rate": round(actual_rate, 4),
            "before": {
                "mean_pd": round(old_mean_pd, 4),
                "brier": round(old_brier, 4),
                "calibration_ratio": round(old_mean_pd / max(actual_rate, 0.001), 2),
            },
            "after": {
                "mean_pd": round(new_mean_pd, 4),
                "brier": round(new_brier, 4),
                "calibration_ratio": round(new_mean_pd / max(actual_rate, 0.001), 2),
            },
            "raw_blend_range": [round(float(raw_blend.min()), 4), round(float(raw_blend.max()), 4)],
        }

        logger.info(f"[Recalibrate] n={len(df)}, actual_rate={actual_rate:.4f}")
        logger.info(f"  Before: mean_pd={old_mean_pd:.4f}, brier={old_brier:.4f}, "
                    f"ratio={old_mean_pd / max(actual_rate, 0.001):.2f}")
        logger.info(f"  After:  mean_pd={new_mean_pd:.4f}, brier={new_brier:.4f}, "
                    f"ratio={new_mean_pd / max(actual_rate, 0.001):.2f}")

        return diagnostics

    # =========================================================================
    # PAYSAFE POST-HOC CALIBRATION
    # =========================================================================

    def _fit_paysafe_multiplier(self, df, target):
        """Compute the PAYSAFE post-hoc calibration multiplier from training data.

        Approach: compute the ratio of actual bad rate to mean predicted PD for
        PAYSAFE loans in training data. This directly measures how much the
        model under-predicts for PAYSAFE after isotonic calibration.

        The multiplier = actual_bad_rate_PAYSAFE / mean_predicted_PD_PAYSAFE.
        This is more stable than quintile-stratified ratios when PAYSAFE sample
        sizes are small (n=59 in training).
        """
        # Get calibrated PDs (without PAYSAFE adjustment) for training data
        woe_pd = self._predict_woe(df)
        rule_pd = self._predict_rule(df)
        xgb_pd = self._predict_xgb(df)

        raw_blend = (
            self.blend_weights["woe"] * woe_pd
            + self.blend_weights["rule"] * rule_pd
            + self.blend_weights["xgb"] * xgb_pd
        )
        cal_pd = self._calibrate(raw_blend)

        # Identify PAYSAFE loans
        is_paysafe = (df["partner"] == "PAYSAFE").values if "partner" in df.columns else np.zeros(len(df), dtype=bool)

        n_paysafe = is_paysafe.sum()
        if n_paysafe < 20:
            logger.warning(f"Only {n_paysafe} PAYSAFE loans in training — "
                           "skipping multiplier (too few for reliable estimate)")
            self._paysafe_multiplier = 1.0
            self._paysafe_train_stats = {"n_paysafe": int(n_paysafe), "skipped": True}
            return

        # Primary multiplier: actual / predicted for PAYSAFE in training
        ps_actual_rate = float(target.values[is_paysafe].mean())
        ps_predicted_mean = float(cal_pd[is_paysafe].mean())

        raw_multiplier = ps_actual_rate / max(ps_predicted_mean, 0.01)

        # Apply Bayesian shrinkage to account for small sample size and
        # regression to the mean. With n=59, we expect the test-time ratio
        # to be lower than training. Shrinkage: effective_mult = 1 + k*(raw-1)
        # where k = n/(n+n0) is a credibility factor (n0=60 is the prior weight).
        n0 = 60  # Prior pseudo-count (shrinks toward 1.0)
        shrinkage = n_paysafe / (n_paysafe + n0)
        shrunk_multiplier = 1.0 + shrinkage * (raw_multiplier - 1.0)

        # Cap the multiplier between 1.0 and 3.0 (conservative)
        self._paysafe_multiplier = float(np.clip(shrunk_multiplier, 1.0, 3.0))

        # Also compute quintile-stratified details for diagnostics
        try:
            pd_quintile = pd.qcut(cal_pd, q=5, labels=False, duplicates="drop")
        except ValueError:
            pd_quintile = pd.cut(cal_pd, bins=5, labels=False)

        eval_df = pd.DataFrame({
            "pd": cal_pd,
            "y": target.values,
            "is_paysafe": is_paysafe,
            "quintile": pd_quintile,
        })

        quintile_details = {}
        for q in sorted(eval_df["quintile"].dropna().unique()):
            q_mask = eval_df["quintile"] == q
            ps_mask = q_mask & eval_df["is_paysafe"]
            nps_mask = q_mask & ~eval_df["is_paysafe"]
            n_ps = ps_mask.sum()
            n_nps = nps_mask.sum()
            if n_ps >= 3:
                br_ps = eval_df.loc[ps_mask, "y"].mean()
                br_nps = eval_df.loc[nps_mask, "y"].mean() if n_nps > 0 else 0
                quintile_details[int(q)] = {
                    "n_paysafe": int(n_ps),
                    "n_other": int(n_nps),
                    "bad_rate_paysafe": round(float(br_ps), 4),
                    "bad_rate_other": round(float(br_nps), 4),
                    "ratio": round(float(br_ps / max(br_nps, 0.001)), 3),
                }

        self._paysafe_train_stats = {
            "n_paysafe": int(n_paysafe),
            "n_total": len(eval_df),
            "overall_bad_rate_paysafe": round(float(ps_actual_rate), 4),
            "overall_bad_rate_other": round(float(target.values[~is_paysafe].mean()), 4),
            "mean_predicted_pd_paysafe": round(float(ps_predicted_mean), 4),
            "raw_multiplier": round(float(raw_multiplier), 4),
            "shrinkage_factor": round(float(shrinkage), 4),
            "shrunk_multiplier": round(float(shrunk_multiplier), 4),
            "final_multiplier": round(float(self._paysafe_multiplier), 4),
            "quintile_details": quintile_details,
        }

        logger.info(f"[PAYSAFE] Multiplier = {self._paysafe_multiplier:.3f} "
                     f"(raw={raw_multiplier:.3f}, shrunk={shrunk_multiplier:.3f}, "
                     f"shrinkage={shrinkage:.3f}, n={n_paysafe})")
        logger.info(f"[PAYSAFE] Actual={ps_actual_rate:.3f}, "
                     f"Predicted={ps_predicted_mean:.3f}")

    def _apply_paysafe_adjustment(self, calibrated_pd, partner_series):
        """Apply post-hoc PAYSAFE multiplier to calibrated PDs.

        Args:
            calibrated_pd: numpy array of calibrated PDs.
            partner_series: pandas Series or numpy array of partner names.

        Returns:
            Adjusted PD array (capped at 0.95 to stay in valid probability range).
        """
        if self._paysafe_multiplier is None or self._paysafe_multiplier == 1.0:
            return calibrated_pd

        pd_arr = np.asarray(calibrated_pd).copy()
        partner_arr = np.asarray(partner_series).astype(str)
        is_paysafe = (partner_arr == "PAYSAFE")

        pd_arr[is_paysafe] = pd_arr[is_paysafe] * self._paysafe_multiplier

        # Cap at 0.95 to keep in valid range
        return np.clip(pd_arr, 0.001, 0.95)

    # =========================================================================
    # SCORE MAPPING
    # =========================================================================

    @staticmethod
    def _pd_to_score(pd_value):
        """Map PD to a 300-850 score (higher = safer)."""
        pd_arr = np.asarray(pd_value)
        return np.clip(300 + (1 - pd_arr) * 550, 300, 850).astype(int)

    # =========================================================================
    # PUBLIC API: FIT
    # =========================================================================

    def fit(self, train_df, y_col="is_bad"):
        """
        Train all 3 sub-models, calibrator, and store blend weights.

        Args:
            train_df: DataFrame with raw features and target column.
            y_col: Name of the binary target column.
        """
        df = self._map_columns(train_df)

        # Normalize categorical inputs to uppercase for consistent matching.
        # Must match the normalization in predict_batch() to avoid train/test skew.
        # Preserve NaN/None as actual NaN, not string "NONE" or "NAN".
        for col in ("qi", "shop_qi", "partner"):
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.upper()

        df = self._engineer_features(df)
        target = df[y_col].astype(int)
        self._train_base_rate = target.mean()

        logger.info(f"Training on {len(df)} samples, "
                    f"bad rate = {self._train_base_rate:.3f}")

        # Sub-model 1: WoE Scorecard
        logger.info("[1/3] Fitting WoE Scorecard...")
        self._fit_woe_scorecard(df, target)
        woe_train_pd = self._predict_woe(df)
        woe_auc = roc_auc_score(target, woe_train_pd)
        logger.info(f"  Selected {len(self._woe_features)} features, "
                    f"train AUC = {woe_auc:.4f}")

        # Sub-model 2: Rule Engine
        logger.info("[2/3] Fitting Rule Engine...")
        self._fit_rule_engine(df, target)
        rule_train_pd = self._predict_rule(df)
        rule_auc = roc_auc_score(target, rule_train_pd)
        logger.info(f"  train AUC = {rule_auc:.4f}")

        # Sub-model 3: XGBoost
        logger.info("[3/3] Fitting XGBoost...")
        self._fit_xgboost(df, target)
        xgb_train_pd = self._predict_xgb(df)
        xgb_auc = roc_auc_score(target, xgb_train_pd)
        logger.info(f"  {len(self._xgb_features)} features, "
                    f"train AUC = {xgb_auc:.4f}")

        # Calibrator (v1.5: cross-validated isotonic regression + legacy Platt)
        logger.info("[Cal] Fitting cross-validated isotonic calibrator (v1.5)...")
        self._fit_calibrator(df, target)

        # PAYSAFE post-hoc multiplier (v1.1)
        logger.info("[PAYSAFE] Fitting post-hoc calibration multiplier...")
        self._fit_paysafe_multiplier(df, target)

        # Verify blended train AUC
        raw_blend = (
            self.blend_weights["woe"] * woe_train_pd
            + self.blend_weights["rule"] * rule_train_pd
            + self.blend_weights["xgb"] * xgb_train_pd
        )
        blend_cal = self._calibrate(raw_blend)
        blend_auc = roc_auc_score(target, blend_cal)
        logger.info(f"[Blend] Calibrated blend train AUC = {blend_auc:.4f}")

        self._fitted = True
        logger.info("Fit complete.")

    # =========================================================================
    # PUBLIC API: PREDICT (single)
    # =========================================================================

    def predict(self, features_dict):
        """
        Score a single application.

        Args:
            features_dict: dict of feature values (supports both long and short names).

        Returns:
            {"pd": float, "score": int, "components": {"woe_pd": ..., "rule_pd": ..., "xgb_pd": ...}}
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Normalize categorical inputs to uppercase for consistent matching
        features_dict = dict(features_dict)
        if isinstance(features_dict.get("qi"), str):
            features_dict["qi"] = features_dict["qi"].upper()
        if isinstance(features_dict.get("shop_qi"), str):
            features_dict["shop_qi"] = features_dict["shop_qi"].upper()
        if isinstance(features_dict.get("partner"), str):
            features_dict["partner"] = features_dict["partner"].upper()

        # Convert to single-row DataFrame
        row_df = pd.DataFrame([features_dict])
        result_df = self._predict_internal(row_df)

        return {
            "pd": float(result_df["pd"].iloc[0]),
            "score": int(result_df["score"].iloc[0]),
            "components": {
                "woe_pd": float(result_df["woe_pd"].iloc[0]),
                "rule_pd": float(result_df["rule_pd"].iloc[0]),
                "xgb_pd": float(result_df["xgb_pd"].iloc[0]),
            },
        }

    # =========================================================================
    # PUBLIC API: PREDICT_BATCH
    # =========================================================================

    def predict_batch(self, df):
        """
        Score a DataFrame of applications.

        Args:
            df: DataFrame with raw features.

        Returns:
            DataFrame with columns: pd, score, woe_pd, rule_pd, xgb_pd
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        # Normalize categorical inputs to uppercase for consistent matching.
        # IMPORTANT: Must preserve NaN/None as actual NaN, not convert to
        # string "NONE" or "NAN". Apply .upper() only to non-null values.
        df = df.copy()
        for col in ("qi", "shop_qi", "partner"):
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.upper()

        return self._predict_internal(df)

    def _predict_internal(self, df):
        """Core prediction logic shared by predict() and predict_batch()."""
        df_mapped = self._map_columns(df)
        df_eng = self._engineer_features(df_mapped)

        woe_pd = self._predict_woe(df_eng)
        rule_pd = self._predict_rule(df_eng)
        xgb_pd = self._predict_xgb(df_eng)

        raw_blend = (
            self.blend_weights["woe"] * woe_pd
            + self.blend_weights["rule"] * rule_pd
            + self.blend_weights["xgb"] * xgb_pd
        )
        calibrated_pd = self._calibrate(raw_blend)

        # v1.1: Apply PAYSAFE post-hoc calibration multiplier
        partner_col = df_mapped["partner"] if "partner" in df_mapped.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
        adjusted_pd = self._apply_paysafe_adjustment(calibrated_pd, partner_col)

        score = self._pd_to_score(adjusted_pd)

        result = pd.DataFrame(
            {
                "pd": adjusted_pd,
                "pd_pre_paysafe": calibrated_pd,  # PD before PAYSAFE adjustment
                "score": score,
                "woe_pd": woe_pd,
                "rule_pd": rule_pd,
                "xgb_pd": xgb_pd,
            },
            index=df.index,
        )
        return result

    # =========================================================================
    # PUBLIC API: EVALUATE
    # =========================================================================

    def evaluate(self, test_df, y_col="is_bad"):
        """
        Evaluate on a test set. Returns AUC, Gini, KS, Brier, and per-component AUCs.

        Args:
            test_df: DataFrame with features and target column.
            y_col: Name of the binary target column.

        Returns:
            dict with keys: auc, gini, ks, brier, component_aucs, n_obs, n_bad, base_rate
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        results = self.predict_batch(test_df)
        y_true = test_df[y_col].astype(int).values

        # Blend metrics
        auc = roc_auc_score(y_true, results["pd"])
        gini = 2 * auc - 1
        brier = brier_score_loss(y_true, results["pd"])
        fpr, tpr, _ = roc_curve(y_true, results["pd"])
        ks = float(np.max(tpr - fpr))

        # Component AUCs
        component_aucs = {}
        for comp in ["woe_pd", "rule_pd", "xgb_pd"]:
            try:
                component_aucs[comp] = round(roc_auc_score(y_true, results[comp]), 4)
            except ValueError:
                component_aucs[comp] = None

        return {
            "auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks": round(ks, 4),
            "brier": round(brier, 4),
            "component_aucs": component_aucs,
            "n_obs": len(y_true),
            "n_bad": int(y_true.sum()),
            "base_rate": round(float(y_true.mean()), 4),
        }

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def save(self, path):
        """Save all model artifacts to a single joblib file."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        artifacts = {
            "version": self.VERSION,
            "blend_weights": self.blend_weights,
            "woe_model": self._woe_model,
            "woe_transforms": self._woe_transforms,
            "woe_features": self._woe_features,
            "woe_iv_scores": self._woe_iv_scores,
            "rule_graders": self._rule_graders,
            "rule_logistic": self._rule_logistic,
            "xgb_model": self._xgb_model,
            "xgb_features": self._xgb_features,
            "xgb_fill_values": self._xgb_fill_values,
            "calibrator": self._calibrator,
            "platt_calibrator": self._platt_calibrator,
            "train_base_rate": self._train_base_rate,
            # v1.1: PAYSAFE calibration
            "paysafe_multiplier": self._paysafe_multiplier,
            "paysafe_train_stats": self._paysafe_train_stats,
        }

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, path)
        logger.info(f"Saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a saved DefaultScorecard from disk."""
        artifacts = joblib.load(path)

        # Version check (warn, don't error, for backward compatibility)
        saved_version = artifacts.get("version", "unknown")
        if saved_version != cls.VERSION:
            logger.warning(
                f"DefaultScorecard version mismatch: saved={saved_version}, "
                f"current={cls.VERSION}. Model behavior may differ."
            )

        ds = cls(blend_weights=artifacts.get("blend_weights", None))
        ds._woe_model = artifacts.get("woe_model", None)
        ds._woe_transforms = artifacts.get("woe_transforms", {})
        ds._woe_features = artifacts.get("woe_features", [])
        ds._woe_iv_scores = artifacts.get("woe_iv_scores", {})
        ds._rule_graders = artifacts.get("rule_graders", {})
        ds._rule_logistic = artifacts.get("rule_logistic", None)
        ds._xgb_model = artifacts.get("xgb_model", None)
        ds._xgb_features = artifacts.get("xgb_features", [])
        ds._xgb_fill_values = artifacts.get("xgb_fill_values", {})
        ds._calibrator = artifacts.get("calibrator", None)
        ds._platt_calibrator = artifacts.get("platt_calibrator", None)
        ds._train_base_rate = artifacts.get("train_base_rate", None)
        # v1.1: PAYSAFE calibration (backward-compatible with v1.0 models)
        ds._paysafe_multiplier = artifacts.get("paysafe_multiplier", None)
        ds._paysafe_train_stats = artifacts.get("paysafe_train_stats", None)
        ds._fitted = True

        return ds

    # =========================================================================
    # REPR
    # =========================================================================

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        if self._fitted:
            return (
                f"DefaultScorecard({status}, "
                f"woe_feats={len(self._woe_features)}, "
                f"xgb_feats={len(self._xgb_features)}, "
                f"blend={self.blend_weights})"
            )
        return f"DefaultScorecard({status})"
