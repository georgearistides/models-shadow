"""
Pipeline — Chains FraudGate -> DefaultScorecard -> CreditGrader into a
single scoring pipeline for subprime SMB loan applications.

Two scoring paths:
  1. **Origination** (.score / .score_batch):
     FraudGate -> DefaultScorecard -> CreditGrader -> Decision
  2. **Post-origination monitoring** (.monitor / .monitor_batch):
     PaymentMonitor -> risk tier + flag count  (optional, loaded if available)

All three origination stages always run (shadow mode) so every score is
available for comparison, but the overall decision reflects the cascade logic:
  1. Fraud Gate can decline or flag for review
  2. Default Scorecard produces calibrated PD
  3. Credit Grader maps PD to letter grade
  4. Decision logic combines all signals

Usage:
    from scripts.models.pipeline import Pipeline

    # --- Origination scoring ---
    p = Pipeline()
    p.fit(train_df, y_col="is_bad")

    result = p.score({"fico": 620, "qi": "MISSING", "partner": "PAYSAFE", ...})
    results_df = p.score_batch(test_df)

    # --- Post-origination monitoring ---
    # (available if models/payment_monitor.joblib exists)
    p2 = Pipeline.load("/path/to/models/")
    mon = p2.monitor({"nsf_count": 3, "nsf_rate": 0.25, ...})
    mon_df = p2.monitor_batch(payment_df)

    p.save("/path/to/models/")
    metrics = p.evaluate(test_df, y_col="is_bad")
"""

import logging
import math
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from scripts.models.fraud_gate import FraudGate
from scripts.models.default_scorecard import DefaultScorecard
from scripts.models.credit_grader import CreditGrader
from scripts.models.payment_monitor import PaymentMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

DEFAULT_PD_DECLINE = 0.25
DEFAULT_PD_REVIEW = 0.15
REVIEW_GRADES = {"E", "F", "G"}
MAX_REASONS = 8

VERSION = "1.4.0"  # Review queue NPV optimization (iter 36)

# ---------------------------------------------------------------------------
# Grade E sub-segmentation defaults
# ---------------------------------------------------------------------------
# Based on iteration 32 analysis:
#   HONEYBOOK Grade E = 56.1% bad rate -> AUTO-DECLINE
#   PAYSAFE Grade E with PD <= 40% = 24.6% bad rate -> KEEP at 1.55x repricing
#   Other Grade E with PD <= 40% -> KEEP at 1.55x repricing
#   Grade E with PD > 40% -> DECLINE

DEFAULT_GRADE_E_CONFIG = {
    "enabled": True,
    "honeybook_auto_decline": True,
    "pd_threshold": 0.40,
    "repricing_factor": 1.55,
    "decline_partners": ["HONEYBOOK"],  # extensible list
}

# ---------------------------------------------------------------------------
# Review queue NPV optimization defaults (iteration 36)
# ---------------------------------------------------------------------------
# Based on NPV analysis of review queue:
#   Only 145 loans (2.1%) have PD > 0.36 -> auto-decline (negative NPV)
#   Loans with PD < 0.06 are extremely safe -> auto-approve
#   97.9% of review loans are positive-NPV if auto-approved

DEFAULT_REVIEW_OPTIMIZATION = {
    "enabled": False,  # Off by default, must be explicitly enabled
    "auto_decline_pd": 0.36,  # Auto-decline reviews with PD above this
    "auto_approve_pd": 0.06,  # Auto-approve reviews with PD below this
}

# ---------------------------------------------------------------------------
# Input validation constants
# ---------------------------------------------------------------------------

# Valid QI quadrant values (case-insensitive comparison)
VALID_QI_VALUES = {"HFHT", "HFLT", "LFHT", "LFLT", "MISSING"}

# FICO score valid range (FICO Score 8 range)
FICO_MIN = 300
FICO_MAX = 850

# Minimum required columns for batch scoring — the pipeline can run with
# just FICO (everything else defaults), but these are the columns the
# sub-models actually consume.  We only hard-require j_latest_fico_score
# (or its alias "fico") because all other fields have graceful NaN defaults.
BATCH_REQUIRED_COLUMNS = {"j_latest_fico_score"}
BATCH_FICO_ALIASES = {"j_latest_fico_score", "fico", "experian_FICO_SCORE"}

# Known numeric feature fields (either short or long Experian names)
KNOWN_NUMERIC_FIELDS = {
    "fico", "j_latest_fico_score", "experian_FICO_SCORE",
    "d30", "d60", "inq6", "revutil", "pastdue", "instbal", "revbal",
    "tl_total", "tl_paid", "tl_delin", "mopmt", "crhist",
    "experian_delinquencies_thirty_day_count",
    "experian_delinquencies_sixty_day_count",
    "experian_inquiries_six_months",
    "experian_inquiries_last_six_months",
    "experian_revolving_account_credit_available_percentage",
    "experian_balance_total_past_due_amounts",
    "experian_balance_total_installment_accounts",
    "experian_balance_total_revolving_accounts",
    "experian_tradelines_total_items",
    "experian_tradelines_total_items_paid",
    "experian_tradelines_total_items_currently_delinquent",
    "experian_payment_amount_monthly_total",
    "experian_credit_history_length_months",
    "experian_revolving_utilization",
    "experian_revolving_balance",
    "experian_installment_balance",
    "experian_past_due_amount",
    "experian_delinquent_tradelines",
    "experian_delinquencies_thirty_days",
    "experian_delinquencies_sixty_days",
    "plaid_balance", "plaid_balance_current",
    "giact_bankruptcy_count",
}


class Pipeline:
    """End-to-end scoring pipeline: FraudGate -> DefaultScorecard -> CreditGrader.

    Parameters
    ----------
    fraud_threshold_profile : str
        FraudGate threshold profile ("conservative", "moderate", "aggressive").
    grading_scheme : str
        CreditGrader scheme ("5grade", "6grade", "7grade").
    blend_weights : dict, optional
        Override DefaultScorecard blend weights.
    pd_decline : float
        PD threshold above which the decision is "decline".
    pd_review : float
        PD threshold above which the decision is "review".
    review_grades : set
        Grades that trigger a "review" decision.
    """

    def __init__(
        self,
        fraud_threshold_profile: str = "moderate",
        grading_scheme: str = "optimal_iv",
        blend_weights: Optional[Dict[str, float]] = None,
        pd_decline: float = DEFAULT_PD_DECLINE,
        pd_review: float = DEFAULT_PD_REVIEW,
        review_grades: Optional[set] = None,
        grade_e_config: Optional[Dict[str, Any]] = None,
        review_optimization: Optional[Dict[str, Any]] = None,
    ):
        self.fraud_gate = FraudGate(threshold_profile=fraud_threshold_profile)
        self.default_scorecard = DefaultScorecard(blend_weights=blend_weights)
        self.credit_grader = CreditGrader(scheme=grading_scheme)

        self.pd_decline = pd_decline
        self.pd_review = pd_review
        self.review_grades = review_grades if review_grades is not None else set(REVIEW_GRADES)

        # Grade E sub-segmentation config (optional, defaults to enabled)
        self.grade_e_config = grade_e_config if grade_e_config is not None else dict(DEFAULT_GRADE_E_CONFIG)

        # Review queue NPV optimization (optional, defaults to disabled)
        self.review_optimization = (
            review_optimization if review_optimization is not None
            else dict(DEFAULT_REVIEW_OPTIMIZATION)
        )

        # Optional post-origination monitor (loaded separately, not part of fit)
        self.payment_monitor: Optional[PaymentMonitor] = None

        # Optional survival scorer (loaded separately, enriches score output)
        self.survival_scorer = None

        self._fitted = False
        self._fit_timestamp = None
        self._train_stats = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, y_col: str = "is_bad") -> "Pipeline":
        """Train all three pipeline stages from labeled training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with features and binary target column.
        y_col : str
            Name of the binary target column (1 = bad).

        Returns
        -------
        self
        """
        logger.info(f"Fitting on {len(train_df)} samples...")

        # Stage 1: Fraud Gate (rule-based, optional calibration)
        logger.info("=== Stage 1: Fraud Gate ===")
        self.fraud_gate.fit(train_df)

        # Stage 2: Default Scorecard (trains 3 sub-models + calibrator)
        logger.info("=== Stage 2: Default Scorecard ===")
        self.default_scorecard.fit(train_df, y_col=y_col)

        # Stage 3: Credit Grader (fit boundaries from training PDs)
        logger.info("=== Stage 3: Credit Grader ===")
        train_pds = self.default_scorecard.predict_batch(train_df)["pd"].values
        train_labels = train_df[y_col].astype(int).values
        # Pass y_true for optimal_iv scheme (ignored by other schemes)
        self.credit_grader.fit(train_pds, y_true=train_labels)
        logger.info(f"  Grade boundaries: {self.credit_grader.boundaries}")
        logger.info(f"  Grade labels: {self.credit_grader.grades}")
        if self.credit_grader._iv_score is not None:
            logger.info(f"  Total IV: {self.credit_grader._iv_score:.4f}")

        # Store training statistics
        self._fit_timestamp = datetime.utcnow().isoformat()
        self._train_stats = {
            "n_train": len(train_df),
            "bad_rate": float(train_df[y_col].mean()),
            "mean_pd": float(np.mean(train_pds)),
            "median_pd": float(np.median(train_pds)),
        }

        self._fitted = True
        logger.info("Fit complete. All 3 stages trained.")
        return self

    # ------------------------------------------------------------------
    # score (single application)
    # ------------------------------------------------------------------

    def score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single loan application through the full pipeline.

        All stages always run (shadow mode) so every score is available.
        The overall decision reflects the cascade logic.

        Parameters
        ----------
        features : dict
            Feature dictionary for one application.

        Returns
        -------
        dict with keys:
            decision, fraud_score, fraud_normalized, fraud_decision,
            fraud_reasons, pd, default_score, grade, pricing_tier,
            all_model_scores, reasons

        Raises
        ------
        ValueError
            If features fail validation (wrong type, invalid FICO, etc.).
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        self._check_fitted()

        # Validate and normalize input
        features = self._validate_input(features)

        # --- Stage 1: Fraud Gate ---
        fraud_result = self.fraud_gate.predict(features)
        fraud_score = fraud_result["score"]
        fraud_normalized = fraud_result["normalized_score"]
        fraud_decision = fraud_result["decision"]
        fraud_reasons = fraud_result["reasons"]

        # --- Stage 2: Default Scorecard ---
        default_result = self.default_scorecard.predict(features)
        pd_value = default_result["pd"]
        default_score = default_result["score"]
        all_model_scores = default_result["components"]

        # --- Stage 3: Credit Grader ---
        partner = features.get("partner")
        grade_result = self.credit_grader.grade(pd_value, partner=partner)
        grade = grade_result["grade"]
        pricing_tier = grade_result["pricing_tier"]

        # --- Stage 4: Decision Logic ---
        decision = self._compute_decision(fraud_decision, pd_value, grade)

        # --- Reason Code Aggregation ---
        reasons = self._aggregate_reasons(fraud_reasons, pd_value, grade, features)

        # --- Stage 5: Grade E Sub-Segmentation ---
        grade_e_result = self._apply_grade_e_subseg(
            decision=decision,
            grade=grade,
            pd_value=pd_value,
            partner=partner,
            reasons=reasons,
        )
        decision = grade_e_result["decision"]
        reasons = grade_e_result["reasons"]

        # --- Stage 6: Review Queue NPV Optimization ---
        review_opt_result = self._apply_review_optimization(
            decision=decision,
            pd_value=pd_value,
            reasons=reasons,
        )
        decision = review_opt_result["decision"]
        reasons = review_opt_result["reasons"]

        result = {
            "decision": decision,
            "fraud_score": fraud_score,
            "fraud_normalized": fraud_normalized,
            "fraud_decision": fraud_decision,
            "fraud_reasons": fraud_reasons,
            "pd": pd_value,
            "default_score": default_score,
            "grade": grade,
            "pricing_tier": pricing_tier,
            "all_model_scores": all_model_scores,
            "reasons": reasons,
            "monitoring_available": self.payment_monitor is not None,
            "review_optimization": review_opt_result["review_optimization"],
        }

        # Add repricing flag if applicable
        if grade_e_result["grade_e_repricing"] is not None:
            result["grade_e_repricing"] = grade_e_result["grade_e_repricing"]

        # --- Optional: Survival enrichment ---
        if self.survival_scorer is not None:
            try:
                surv_pred = self.survival_scorer.predict(features)
                result["survival"] = {
                    "hazard_6mo": float(surv_pred.get("pd_6mo", 0.0)),
                    "hazard_12mo": float(surv_pred.get("pd_12mo", 0.0)),
                    "hazard_18mo": float(surv_pred.get("pd_18mo", 0.0)),
                    "risk_tier": surv_pred.get("risk_tier", "unknown"),
                    "median_survival_months": surv_pred.get("median_survival_months"),
                }
            except Exception as e:
                logger.warning("SurvivalScorer prediction failed: %s", e)
                result["survival_available"] = False
        else:
            result["survival_available"] = False

        return result

    # ------------------------------------------------------------------
    # score_batch
    # ------------------------------------------------------------------

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of applications through the full pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with feature columns.

        Returns
        -------
        pd.DataFrame with columns:
            decision, fraud_score, fraud_normalized, fraud_decision,
            pd, default_score, grade, pricing_tier,
            woe_pd, rule_pd, xgb_pd, reasons

        Raises
        ------
        ValueError
            If df is not a DataFrame or is missing required columns.
        RuntimeError
            If the pipeline has not been fitted yet.
        """
        self._check_fitted()

        # Validate batch input
        df = self._validate_dataframe(df)

        # Stage 1: Fraud Gate (batch)
        fraud_df = self.fraud_gate.predict_batch(df)

        # Stage 2: Default Scorecard (batch)
        default_df = self.default_scorecard.predict_batch(df)

        # Stage 3: Credit Grader (batch)
        partners = df["partner"].values if "partner" in df.columns else None
        grade_df = self.credit_grader.grade_batch(default_df["pd"].values, partners=partners)

        # Stage 4: Decision Logic (vectorized)
        decisions = self._compute_decision_batch(
            fraud_decisions=fraud_df["decision"].values,
            pds=default_df["pd"].values,
            grades=grade_df["grade"].values,
        )

        # Stage 5: Grade E Sub-Segmentation (vectorized)
        partner_vals = df["partner"].values if "partner" in df.columns else None
        grade_e_batch = self._apply_grade_e_subseg_batch(
            decisions=decisions,
            grades=grade_df["grade"].values,
            pds=default_df["pd"].values,
            partners=partner_vals,
        )
        decisions = grade_e_batch["decisions"]

        # Stage 6: Review Queue NPV Optimization (vectorized)
        review_opt_batch = self._apply_review_optimization_batch(
            decisions=decisions,
            pds=default_df["pd"].values,
        )
        decisions = review_opt_batch["decisions"]

        # Assemble results
        result = pd.DataFrame(
            {
                "decision": decisions,
                "fraud_score": fraud_df["score"].values,
                "fraud_normalized": fraud_df["normalized_score"].values,
                "fraud_decision": fraud_df["decision"].values,
                "pd": default_df["pd"].values,
                "default_score": default_df["score"].values,
                "grade": grade_df["grade"].values,
                "pricing_tier": grade_df["pricing_tier"].values,
                "woe_pd": default_df["woe_pd"].values,
                "rule_pd": default_df["rule_pd"].values,
                "xgb_pd": default_df["xgb_pd"].values,
                "grade_e_repricing": grade_e_batch["grade_e_repricing"],
                "grade_e_action": grade_e_batch["grade_e_action"],
                "review_optimization": review_opt_batch["review_optimization"],
            },
            index=df.index,
        )

        # --- Optional: Survival enrichment (batch) ---
        if self.survival_scorer is not None:
            try:
                surv_df = self.survival_scorer.predict_batch(df)
                result["survival_hazard_6mo"] = surv_df["pd_6mo"].values
                result["survival_hazard_12mo"] = surv_df["pd_12mo"].values
                result["survival_hazard_18mo"] = surv_df["pd_18mo"].values
                result["survival_risk_tier"] = surv_df["risk_tier"].values
                result["survival_median_months"] = surv_df["median_survival_months"].values
                result["survival_available"] = True
            except Exception as e:
                logger.warning("SurvivalScorer batch prediction failed: %s", e)
                result["survival_available"] = False
        else:
            result["survival_available"] = False

        return result

    # ------------------------------------------------------------------
    # monitor (single loan — post-origination)
    # ------------------------------------------------------------------

    def monitor(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor a single loan's post-origination payment health.

        Uses the PaymentMonitor to evaluate NSF/payment behavior signals and
        return a risk tier (Green/Yellow/Orange/Red) with flag details.

        This is a separate path from .score() — .score() is for origination
        decisioning, .monitor() is for ongoing portfolio health monitoring.

        Parameters
        ----------
        features : dict
            Payment feature dictionary for one loan. Expected keys include
            nsf_count, nsf_rate, nsf_in_first_90d, nsf_cluster_max,
            max_consecutive_nsf, has_recovery, recovery_nsf_rate,
            payment_cv, payment_decline_pct. Also accepts 'loan_id'
            or 'jaris_application_id' for identification.

        Returns
        -------
        dict with keys:
            payment_tier: "Green"|"Yellow"|"Orange"|"Red"
            payment_pd: float (calibrated P(default) from payment signals)
            flag_count: int (number of triggered flags, 0-9)
            flags: list[str] (descriptions of triggered flags)
            loan_id: str (from features, or "unknown")

        Raises
        ------
        RuntimeError
            If PaymentMonitor is not loaded.
        """
        self._check_monitor_loaded()

        # Extract loan identifier
        loan_id = str(
            features.get("loan_id")
            or features.get("jaris_application_id")
            or "unknown"
        )

        # Delegate to PaymentMonitor
        pm_result = self.payment_monitor.predict(features)

        return {
            "payment_tier": pm_result["risk_tier"].capitalize(),
            "payment_pd": pm_result["default_probability"],
            "flag_count": pm_result["flag_count"],
            "flags": pm_result["flags"],
            "loan_id": loan_id,
        }

    # ------------------------------------------------------------------
    # monitor_batch (DataFrame — post-origination)
    # ------------------------------------------------------------------

    def monitor_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Monitor a batch of loans for post-origination payment health.

        Vectorized batch version of .monitor(). Returns a DataFrame with
        one row per loan containing payment tier, calibrated PD, flag count,
        and individual flag columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with payment feature columns (nsf_count, nsf_rate,
            etc.). May optionally include 'loan_id' or
            'jaris_application_id' for identification.

        Returns
        -------
        pd.DataFrame with columns:
            payment_tier: "Green"|"Yellow"|"Orange"|"Red"
            payment_pd: float
            flag_count: int
            loan_id: str
            plus one column per flag (0/1)

        Raises
        ------
        RuntimeError
            If PaymentMonitor is not loaded.
        ValueError
            If df is not a DataFrame or is empty.
        """
        self._check_monitor_loaded()

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"'df' must be a pandas DataFrame, got {type(df).__name__}."
            )
        if len(df) == 0:
            raise ValueError(
                "Empty DataFrame. Provide at least one row to monitor."
            )

        # Delegate to PaymentMonitor batch prediction
        pm_results = self.payment_monitor.predict_batch(df)

        # Build result DataFrame with the pipeline's naming conventions
        result = pd.DataFrame(index=df.index)

        # Capitalize tier names: green -> Green, red -> Red
        result["payment_tier"] = pm_results["risk_tier"].str.capitalize()
        result["payment_pd"] = pm_results["default_probability"]
        result["flag_count"] = pm_results["flag_count"]

        # Loan ID
        if "loan_id" in df.columns:
            result["loan_id"] = df["loan_id"].astype(str)
        elif "jaris_application_id" in df.columns:
            result["loan_id"] = df["jaris_application_id"].astype(str)
        else:
            result["loan_id"] = "unknown"

        # Copy individual flag columns from PaymentMonitor output
        flag_cols = [c for c in pm_results.columns if c.startswith("flag_")]
        for col in flag_cols:
            result[col] = pm_results[col]

        return result

    # ------------------------------------------------------------------
    # monitor helpers
    # ------------------------------------------------------------------

    def _check_monitor_loaded(self):
        """Raise informative error if PaymentMonitor is not loaded."""
        if self.payment_monitor is None:
            raise RuntimeError(
                "PaymentMonitor is not loaded. Either:\n"
                "  1. Place 'payment_monitor.joblib' in the models/ directory "
                "and call Pipeline.load(), or\n"
                "  2. Assign a fitted PaymentMonitor directly: "
                "pipeline.payment_monitor = pm\n\n"
                "The .score() origination path works without it."
            )

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _compute_decision(
        self, fraud_decision: str, pd_value: float, grade: str
    ) -> str:
        """Compute overall decision for a single application."""
        if fraud_decision == "decline":
            return "decline"
        if fraud_decision == "review":
            return "review"
        if pd_value >= self.pd_decline:
            return "decline"
        if pd_value >= self.pd_review:
            return "review"
        if grade in self.review_grades:
            return "review"
        return "approve"

    def _compute_decision_batch(
        self,
        fraud_decisions: np.ndarray,
        pds: np.ndarray,
        grades: np.ndarray,
    ) -> np.ndarray:
        """Vectorized decision computation for batch scoring.

        Matches the exact priority order of _compute_decision:
          1. fraud_decision == "decline" -> "decline"  (highest priority)
          2. fraud_decision == "review"  -> "review"
          3. pd >= pd_decline            -> "decline"
          4. pd >= pd_review             -> "review"
          5. grade in review_grades      -> "review"
          6. default                     -> "approve"

        Uses np.select with conditions in priority order (first match wins).
        """
        conditions = [
            fraud_decisions == "decline",           # 1. Fraud decline
            fraud_decisions == "review",            # 2. Fraud review
            pds >= self.pd_decline,                 # 3. PD decline
            pds >= self.pd_review,                  # 4. PD review
            np.isin(grades, list(self.review_grades)),  # 5. Grade review
        ]
        choices = [
            "decline",  # 1
            "review",   # 2
            "decline",  # 3
            "review",   # 4
            "review",   # 5
        ]
        return np.select(conditions, choices, default="approve")

    # ------------------------------------------------------------------
    # Grade E sub-segmentation
    # ------------------------------------------------------------------

    def _apply_grade_e_subseg(
        self,
        decision: str,
        grade: str,
        pd_value: float,
        partner: Optional[str],
        reasons: List[str],
    ) -> Dict[str, Any]:
        """Apply Grade E sub-segmentation to a single application.

        Returns a dict with potentially overridden decision and reasons,
        plus an optional grade_e_repricing flag.

        If grade_e_config is disabled or grade != "E", returns the inputs
        unchanged (no-op).
        """
        result = {
            "decision": decision,
            "reasons": list(reasons),
            "grade_e_repricing": None,
        }

        cfg = self.grade_e_config
        if not cfg.get("enabled", True):
            return result
        if grade != "E":
            return result

        partner_upper = str(partner).upper() if partner else ""
        decline_partners = [p.upper() for p in cfg.get("decline_partners", [])]
        pd_threshold = cfg.get("pd_threshold", 0.40)
        repricing_factor = cfg.get("repricing_factor", 1.55)

        # Rule 1: Auto-decline for specific partners (e.g., HONEYBOOK)
        if cfg.get("honeybook_auto_decline", True) and partner_upper in decline_partners:
            result["decision"] = "decline"
            reason = f"HONEYBOOK_GRADE_E_AUTO_DECLINE (partner={partner_upper})"
            if reason not in result["reasons"]:
                result["reasons"].append(reason)
            # Re-sort by priority tier and re-cap after adding Grade E reason
            result["reasons"].sort(key=self._reason_priority)
            result["reasons"] = result["reasons"][:MAX_REASONS]
            return result

        # Rule 2: High PD Grade E -> decline
        if pd_value > pd_threshold:
            result["decision"] = "decline"
            reason = f"HIGH_PD_GRADE_E_DECLINE (PD={pd_value:.1%} > {pd_threshold:.0%})"
            if reason not in result["reasons"]:
                result["reasons"].append(reason)
            # Re-sort by priority tier and re-cap after adding Grade E reason
            result["reasons"].sort(key=self._reason_priority)
            result["reasons"] = result["reasons"][:MAX_REASONS]
            return result

        # Rule 3: Low PD Grade E -> keep with repricing flag
        result["grade_e_repricing"] = repricing_factor
        return result

    def _apply_grade_e_subseg_batch(
        self,
        decisions: np.ndarray,
        grades: np.ndarray,
        pds: np.ndarray,
        partners: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Vectorized Grade E sub-segmentation for batch scoring.

        Returns a dict with:
            decisions: np.ndarray of (possibly overridden) decisions
            grade_e_repricing: np.ndarray of float (NaN if not applicable)
            grade_e_action: np.ndarray of str reason codes (empty string if none)
        """
        n = len(decisions)
        out_decisions = decisions.copy()
        repricing = np.full(n, np.nan)
        actions = np.full(n, "", dtype=object)

        cfg = self.grade_e_config
        if not cfg.get("enabled", True):
            return {
                "decisions": out_decisions,
                "grade_e_repricing": repricing,
                "grade_e_action": actions,
            }

        is_grade_e = grades == "E"
        if not is_grade_e.any():
            return {
                "decisions": out_decisions,
                "grade_e_repricing": repricing,
                "grade_e_action": actions,
            }

        pd_threshold = cfg.get("pd_threshold", 0.40)
        repricing_factor = cfg.get("repricing_factor", 1.55)
        decline_partners = [p.upper() for p in cfg.get("decline_partners", [])]

        # Build partner mask
        if partners is not None and cfg.get("honeybook_auto_decline", True):
            partner_upper = np.char.upper(np.asarray(partners, dtype=str))
            is_decline_partner = np.isin(partner_upper, decline_partners)
        else:
            is_decline_partner = np.zeros(n, dtype=bool)

        # Rule 1: Grade E + decline partner -> decline
        mask_partner_decline = is_grade_e & is_decline_partner
        out_decisions = np.where(mask_partner_decline, "decline", out_decisions)
        actions = np.where(mask_partner_decline, "HONEYBOOK_GRADE_E_AUTO_DECLINE", actions)

        # Rule 2: Grade E + high PD (and NOT already declined by partner rule) -> decline
        mask_high_pd = is_grade_e & ~mask_partner_decline & (pds > pd_threshold)
        out_decisions = np.where(mask_high_pd, "decline", out_decisions)
        actions = np.where(mask_high_pd, "HIGH_PD_GRADE_E_DECLINE", actions)

        # Rule 3: Grade E + low PD (remaining) -> keep with repricing
        mask_keep = is_grade_e & ~mask_partner_decline & ~mask_high_pd
        repricing = np.where(mask_keep, repricing_factor, repricing)
        actions = np.where(mask_keep, "GRADE_E_REPRICING", actions)

        return {
            "decisions": out_decisions,
            "grade_e_repricing": repricing,
            "grade_e_action": actions,
        }

    # ------------------------------------------------------------------
    # Review queue NPV optimization
    # ------------------------------------------------------------------

    def set_review_optimization(
        self,
        enabled: bool = True,
        auto_decline_pd: float = 0.36,
        auto_approve_pd: float = 0.06,
    ) -> None:
        """Configure review queue NPV optimization.

        When enabled, loans initially assigned decision="review" are
        re-evaluated based on their calibrated PD:
          - PD > auto_decline_pd  -> auto-decline (negative NPV)
          - PD < auto_approve_pd  -> auto-approve (very safe)
          - otherwise             -> stays in review queue

        Parameters
        ----------
        enabled : bool
            Turn the optimization on (True) or off (False).
        auto_decline_pd : float
            PD threshold above which review loans are auto-declined.
        auto_approve_pd : float
            PD threshold below which review loans are auto-approved.
        """
        if auto_approve_pd >= auto_decline_pd:
            raise ValueError(
                f"auto_approve_pd ({auto_approve_pd}) must be less than "
                f"auto_decline_pd ({auto_decline_pd})."
            )
        self.review_optimization = {
            "enabled": enabled,
            "auto_decline_pd": auto_decline_pd,
            "auto_approve_pd": auto_approve_pd,
        }
        logger.info(
            "Review optimization %s (approve PD<%.2f, decline PD>%.2f)",
            "enabled" if enabled else "disabled",
            auto_approve_pd,
            auto_decline_pd,
        )

    def _apply_review_optimization(
        self,
        decision: str,
        pd_value: float,
        reasons: List[str],
    ) -> Dict[str, Any]:
        """Apply review queue NPV optimization to a single application.

        Only acts on loans with decision="review". If review optimization
        is disabled or the loan is not in review, returns inputs unchanged.

        Returns
        -------
        dict with keys:
            decision: str (possibly changed from "review")
            reasons: list[str] (possibly with appended reason code)
            review_optimization: str ("auto_declined"|"auto_approved"|"unchanged"|"not_applicable")
        """
        cfg = self.review_optimization
        if not cfg.get("enabled", False) or decision != "review":
            return {
                "decision": decision,
                "reasons": list(reasons),
                "review_optimization": "not_applicable",
            }

        auto_decline_pd = cfg.get("auto_decline_pd", 0.36)
        auto_approve_pd = cfg.get("auto_approve_pd", 0.06)

        if pd_value > auto_decline_pd:
            new_reasons = list(reasons)
            reason = f"AUTO_DECLINE_HIGH_PD_REVIEW (PD={pd_value:.1%} > {auto_decline_pd:.0%})"
            if reason not in new_reasons:
                new_reasons.append(reason)
            new_reasons.sort(key=self._reason_priority)
            new_reasons = new_reasons[:MAX_REASONS]
            return {
                "decision": "decline",
                "reasons": new_reasons,
                "review_optimization": "auto_declined",
            }

        if pd_value < auto_approve_pd:
            new_reasons = list(reasons)
            reason = f"AUTO_APPROVE_LOW_PD_REVIEW (PD={pd_value:.1%} < {auto_approve_pd:.0%})"
            if reason not in new_reasons:
                new_reasons.append(reason)
            new_reasons.sort(key=self._reason_priority)
            new_reasons = new_reasons[:MAX_REASONS]
            return {
                "decision": "approve",
                "reasons": new_reasons,
                "review_optimization": "auto_approved",
            }

        return {
            "decision": decision,
            "reasons": list(reasons),
            "review_optimization": "unchanged",
        }

    def _apply_review_optimization_batch(
        self,
        decisions: np.ndarray,
        pds: np.ndarray,
    ) -> Dict[str, Any]:
        """Vectorized review queue NPV optimization for batch scoring.

        Returns
        -------
        dict with keys:
            decisions: np.ndarray of (possibly overridden) decisions
            review_optimization: np.ndarray of str action codes
        """
        n = len(decisions)
        out_decisions = decisions.copy()
        actions = np.full(n, "not_applicable", dtype=object)

        cfg = self.review_optimization
        if not cfg.get("enabled", False):
            return {
                "decisions": out_decisions,
                "review_optimization": actions,
            }

        auto_decline_pd = cfg.get("auto_decline_pd", 0.36)
        auto_approve_pd = cfg.get("auto_approve_pd", 0.06)

        is_review = decisions == "review"
        if not is_review.any():
            return {
                "decisions": out_decisions,
                "review_optimization": actions,
            }

        # Mark review loans as "unchanged" baseline (they are in review)
        actions = np.where(is_review, "unchanged", actions)

        # Auto-decline: review + high PD
        mask_decline = is_review & (pds > auto_decline_pd)
        out_decisions = np.where(mask_decline, "decline", out_decisions)
        actions = np.where(mask_decline, "auto_declined", actions)

        # Auto-approve: review + low PD
        mask_approve = is_review & (pds < auto_approve_pd)
        out_decisions = np.where(mask_approve, "approve", out_decisions)
        actions = np.where(mask_approve, "auto_approved", actions)

        return {
            "decisions": out_decisions,
            "review_optimization": actions,
        }

    # ------------------------------------------------------------------
    # Reason aggregation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Reason priority tiers (lower number = higher priority)
    # ------------------------------------------------------------------

    # Priority mapping: prefix pattern -> tier
    # Tier 1: QI interaction codes (3-6x lift)
    # Tier 2: Entity graph codes (actionable)
    # Tier 3: Financial stress codes (1.7-2.3x lift)
    # Tier 4: Specific bureau codes (1.2-2.0x lift)
    # Tier 5: Model-derived (PD, grade)
    # Tier 6: Generic bureau codes (lowest signal)
    # Tier 7: Provider missingness (lowest value)
    _REASON_PRIORITY = {
        # Tier 1 — QI interaction (highest lift)
        "MISSING_QI_": 1,
        "INTERACTION_": 1,
        "Missing QI": 1,
        # Tier 2 — Entity graph (actionable)
        "Prior default by connected": 2,
        "PRIOR_DEFAULT": 2,
        "Multiple connected entities": 2,
        "CONNECTED_": 2,
        "Cross-entity borrower": 2,
        "CROSS_ENTITY_": 2,
        # Tier 2.5 — Grade E sub-segmentation (actionable, high impact)
        "HONEYBOOK_GRADE_E_AUTO_DECLINE": 2,
        "HIGH_PD_GRADE_E_DECLINE": 2,
        # Tier 2.5 — Review queue NPV optimization (actionable)
        "AUTO_DECLINE_HIGH_PD_REVIEW": 2,
        "AUTO_APPROVE_LOW_PD_REVIEW": 6,  # informational, low priority
        # Tier 3 — Financial stress
        "Negative bank balance": 3,
        "NEGATIVE_BANK_BALANCE": 3,
        "High NSFs": 3,
        "HIGH_NSF": 3,
        "Extended negative balance": 3,
        "Bureau stress": 3,
        "PAYSAFE": 3,
        # Tier 4 — Specific bureau codes
        "High delinquency rate": 4,
        "HIGH_DELINQUENCY_RATE": 4,
        "High inquiries": 4,
        "HIGH_INQUIRIES": 4,
        "GIACT bankruptcy": 4,
        "POOR_BUREAU_HEALTH": 4,
        "Below-average credit": 4,
        "HIGH_MONTHLY_OBLIGATIONS": 4,
        "High monthly payment": 4,
        "HIGH_REVOLVING_BALANCE": 4,
        "Elevated revolving": 4,
        "Zero/minimal tradelines": 4,
        # Tier 5 — Model-derived
        "High default risk": 5,
        "Elevated default risk": 5,
        "Poor credit grade": 5,
        # Tier 6 — Generic bureau
        "Severe delinquencies": 6,
        "SEVERE_DELINQUENCIES": 6,
        "High past due": 6,
        "HIGH_PAST_DUE": 6,
        "Low FICO": 6,
        "FICO_BELOW_600": 6,
        "FICO < 550": 6,
        "FICO 550-600": 6,
        # Tier 7 — Provider missingness
        "Plaid missing": 7,
        "PLAID_MISSING": 7,
        "Prove missing": 7,
        "PROVE_MISSING": 7,
        "All providers missing": 7,
        "Minimal provider coverage": 7,
        "Provider disagreement": 7,
    }

    @classmethod
    def _reason_priority(cls, reason: str) -> int:
        """Return priority tier for a reason string (1=highest, 7=lowest)."""
        for prefix, tier in cls._REASON_PRIORITY.items():
            if reason.startswith(prefix):
                return tier
        return 5  # default to model-derived tier for unrecognized reasons

    def _aggregate_reasons(
        self,
        fraud_reasons: List[str],
        pd_value: float,
        grade: str,
        features: Dict[str, Any],
    ) -> List[str]:
        """Combine top reasons from fraud gate and default risk signals.

        Returns the top MAX_REASONS reasons, sorted by priority tier
        (QI interaction > entity graph > financial stress > bureau >
        model-derived > generic bureau > provider missingness).
        """
        reasons = []

        # Add fraud reasons (already sorted by point contribution)
        for r in fraud_reasons:
            # Strip the point values for cleaner output
            clean = r.split("(+")[0].strip() if "(+" in r else r
            reasons.append(clean)

        # Add default risk reasons based on PD and features
        if pd_value >= self.pd_decline:
            reasons.append(f"High default risk (PD={pd_value:.1%})")
        elif pd_value >= self.pd_review:
            reasons.append(f"Elevated default risk (PD={pd_value:.1%})")

        if grade in self.review_grades:
            reasons.append(f"Poor credit grade ({grade})")

        # Feature-based reasons (only if not already covered by fraud reasons)
        fico = features.get("fico") or features.get("experian_FICO_SCORE") or features.get("j_latest_fico_score")
        fico_val = None
        if fico is not None:
            try:
                fico_val = float(fico)
                if fico_val < 600 and not any("FICO" in r for r in reasons):
                    reasons.append(f"Low FICO ({int(fico_val)})")
            except (ValueError, TypeError):
                pass

        # PAYSAFE reason: only fire when FICO < 650 (conditional)
        partner = features.get("partner")
        if partner == "PAYSAFE" and not any("PAYSAFE" in r for r in reasons):
            if fico_val is not None and fico_val < 650:
                reasons.append("PAYSAFE partner (high-risk segment)")

        qi = features.get("qi") or features.get("shop_qi")
        if qi is not None and str(qi).upper() == "MISSING" and not any("QI" in r.upper() for r in reasons):
            reasons.append("Missing QI")

        # --- Bureau/scorecard feature-based reason codes ---
        # These fire when XGB/WoE feature values indicate elevated risk.
        self._add_scorecard_reasons(reasons, features)

        # Deduplicate while preserving order
        seen = set()
        unique_reasons = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)

        # Sort by priority tier (stable sort preserves within-tier order)
        unique_reasons.sort(key=self._reason_priority)

        return unique_reasons[:MAX_REASONS]

    @staticmethod
    def _add_scorecard_reasons(
        reasons: List[str], features: Dict[str, Any]
    ) -> None:
        """Add reason codes for DefaultScorecard features when values
        indicate elevated risk.

        Thresholds are derived from training data distributions:
        - bureau_health: < 0.35 (bottom quartile, composite of delinquency
          and tradeline health; lower = worse)
        - mopmt: > 3000 (P90; high monthly obligations increase strain)
        - revbal: > 50000 (P90; elevated revolving balances)

        These are computed from raw inputs when the derived features are
        not directly available.
        """
        def _get(key, default=None):
            v = features.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return default
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        # --- POOR_BUREAU_HEALTH ---
        # Try the derived feature first; if absent, compute from components
        bureau_health = _get("bureau_health")
        if bureau_health is None:
            tl_delin = _get("tl_delin", _get("experian_tradelines_total_items_currently_delinquent"))
            tl_total = _get("tl_total", _get("experian_tradelines_total_items"))
            tl_paid = _get("tl_paid", _get("experian_tradelines_total_items_paid"))
            if tl_total is not None and tl_total > 0:
                tl_delin = tl_delin if tl_delin is not None else 0
                tl_paid = tl_paid if tl_paid is not None else 0
                paid_ratio = min(tl_paid / max(tl_total, 1), 1.0)
                delin_rate = min(tl_delin / max(tl_total, 1), 1.0)
                zero_delin = 1.0 if tl_delin == 0 else 0.0
                bureau_health = zero_delin * 0.4 + paid_ratio * 0.3 + (1 - delin_rate) * 0.3

        if bureau_health is not None and bureau_health < 0.35:
            if not any("bureau" in r.lower() and "health" in r.lower() for r in reasons):
                reasons.append("Below-average credit bureau profile")

        # --- HIGH_MONTHLY_OBLIGATIONS ---
        mopmt = _get("mopmt", _get("experian_payment_amount_monthly_total"))
        if mopmt is not None and mopmt > 3000:
            if not any("monthly" in r.lower() and "payment" in r.lower() for r in reasons):
                reasons.append(f"High monthly payment obligations (${mopmt:,.0f})")

        # --- HIGH_REVOLVING_BALANCE ---
        revbal = _get("revbal", _get("experian_balance_total_revolving_accounts"))
        if revbal is not None and revbal > 50000:
            if not any("revolving" in r.lower() for r in reasons):
                reasons.append(f"Elevated revolving credit balances (${revbal:,.0f})")

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self, test_df: pd.DataFrame, y_col: str = "is_bad"
    ) -> Dict[str, Any]:
        """Evaluate the full pipeline on labeled test data.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data with features and binary target column.
        y_col : str
            Name of the binary target column.

        Returns
        -------
        dict with keys:
            fraud_auc, default_auc, grade_auc, overall_auc,
            decision_bad_rates, decision_counts,
            grade_bad_rates, grade_counts,
            n_obs, n_bad, base_rate,
            fraud_metrics, default_metrics, grader_metrics
        """
        self._check_fitted()

        y_true = test_df[y_col].astype(int).values

        # Score all applications
        results = self.score_batch(test_df)

        metrics = {}

        # --- Fraud AUC ---
        try:
            from sklearn.metrics import roc_auc_score
            metrics["fraud_auc"] = round(
                roc_auc_score(y_true, results["fraud_score"].values), 4
            )
        except ValueError:
            metrics["fraud_auc"] = None

        # --- Default AUC ---
        try:
            metrics["default_auc"] = round(
                roc_auc_score(y_true, results["pd"].values), 4
            )
        except ValueError:
            metrics["default_auc"] = None

        # --- Grade AUC ---
        try:
            metrics["grade_auc"] = round(
                roc_auc_score(y_true, results["pricing_tier"].values), 4
            )
        except ValueError:
            metrics["grade_auc"] = None

        # --- Overall AUC (using PD as primary risk score) ---
        metrics["overall_auc"] = metrics["default_auc"]

        # --- Decision distribution and bad rates ---
        decision_bad_rates = {}
        decision_counts = {}
        for dec in ["approve", "review", "decline"]:
            mask = results["decision"].values == dec
            n = int(mask.sum())
            decision_counts[dec] = n
            if n > 0:
                decision_bad_rates[dec] = round(float(y_true[mask].mean()), 4)
            else:
                decision_bad_rates[dec] = 0.0

        metrics["decision_bad_rates"] = decision_bad_rates
        metrics["decision_counts"] = decision_counts

        # --- Grade distribution and bad rates ---
        grade_bad_rates = {}
        grade_counts = {}
        for grade in self.credit_grader.grades:
            mask = results["grade"].values == grade
            n = int(mask.sum())
            grade_counts[grade] = n
            if n > 0:
                grade_bad_rates[grade] = round(float(y_true[mask].mean()), 4)
            else:
                grade_bad_rates[grade] = 0.0

        metrics["grade_bad_rates"] = grade_bad_rates
        metrics["grade_counts"] = grade_counts

        # --- Summary stats ---
        metrics["n_obs"] = len(y_true)
        metrics["n_bad"] = int(y_true.sum())
        metrics["base_rate"] = round(float(y_true.mean()), 4)

        # --- Sub-model component AUCs ---
        component_aucs = {}
        for comp in ["woe_pd", "rule_pd", "xgb_pd"]:
            try:
                component_aucs[comp] = round(
                    roc_auc_score(y_true, results[comp].values), 4
                )
            except (ValueError, KeyError):
                component_aucs[comp] = None
        metrics["component_aucs"] = component_aucs

        # --- Per-stage detailed metrics ---
        try:
            metrics["fraud_metrics"] = self.fraud_gate.evaluate(test_df, y_col=y_col)
        except Exception as e:
            logger.warning("FraudGate evaluation failed: %s", e)
            metrics["fraud_metrics"] = None

        try:
            metrics["default_metrics"] = self.default_scorecard.evaluate(
                test_df, y_col=y_col
            )
        except Exception as e:
            logger.warning("DefaultScorecard evaluation failed: %s", e)
            metrics["default_metrics"] = None

        try:
            partners = test_df["partner"].values if "partner" in test_df.columns else None
            metrics["grader_metrics"] = self.credit_grader.evaluate(
                results["pd"].values, y_true, partners=partners
            )
        except Exception as e:
            logger.warning("CreditGrader evaluation failed: %s", e)
            metrics["grader_metrics"] = None

        return metrics

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """Save the entire pipeline to a directory.

        Creates:
            {dir}/fraud_gate.joblib
            {dir}/default_scorecard.joblib
            {dir}/credit_grader.joblib
            {dir}/pipeline_config.joblib
            {dir}/entity_graph.joblib       (if FraudGate has entity graph)
            {dir}/payment_monitor.joblib    (if PaymentMonitor is loaded)
        """
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.fraud_gate.save(directory / "fraud_gate.joblib")
        self.default_scorecard.save(str(directory / "default_scorecard.joblib"))
        self.credit_grader.save(directory / "credit_grader.joblib")

        # Save entity graph as companion file for portable round-trips
        if self.fraud_gate.graph_loaded:
            entity_graph_data = {
                "graph_lookup": self.fraud_gate._graph_lookup,
                "graph_lookup_key": self.fraud_gate._graph_lookup_key,
            }
            joblib.dump(entity_graph_data, directory / "entity_graph.joblib")
            logger.info(
                "Saved entity graph (%d entries) to %s/entity_graph.joblib",
                len(self.fraud_gate._graph_lookup), directory,
            )

        # Optionally save SurvivalScorer
        has_survival = self.survival_scorer is not None
        if has_survival:
            self.survival_scorer.save(str(directory / "survival_scorer.joblib"))
            logger.info("Saved SurvivalScorer to %s/survival_scorer.joblib", directory)

        # Optionally save PaymentMonitor
        has_monitor = self.payment_monitor is not None
        if has_monitor:
            self.payment_monitor.save(directory / "payment_monitor.joblib")
            logger.info("Saved PaymentMonitor to %s/payment_monitor.joblib", directory)

        config = {
            "version": VERSION,
            "pd_decline": self.pd_decline,
            "pd_review": self.pd_review,
            "review_grades": list(self.review_grades),
            "fit_timestamp": self._fit_timestamp,
            "train_stats": self._train_stats,
            "has_payment_monitor": has_monitor,
            "has_survival_scorer": has_survival,
            "grade_e_config": self.grade_e_config,
            "review_optimization": self.review_optimization,
        }
        joblib.dump(config, directory / "pipeline_config.joblib")

        logger.info(f"Saved to {directory}/")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "Pipeline":
        """Load a complete pipeline from a directory.

        Expects:
            {dir}/fraud_gate.joblib
            {dir}/default_scorecard.joblib
            {dir}/credit_grader.joblib
            {dir}/pipeline_config.joblib

        Optionally loads (if file exists):
            {dir}/payment_monitor.joblib
        """
        directory = Path(directory)

        config = joblib.load(directory / "pipeline_config.joblib")

        # Version check (warn, don't error, for backward compatibility)
        saved_version = config.get("version", "unknown")
        if saved_version != VERSION:
            logger.warning(
                f"Pipeline version mismatch: saved={saved_version}, "
                f"current={VERSION}. Model behavior may differ."
            )

        pipeline = cls.__new__(cls)
        pipeline.fraud_gate = FraudGate.load(directory / "fraud_gate.joblib")
        pipeline.default_scorecard = DefaultScorecard.load(
            str(directory / "default_scorecard.joblib")
        )
        pipeline.credit_grader = CreditGrader.load(directory / "credit_grader.joblib")

        pipeline.pd_decline = config.get("pd_decline", DEFAULT_PD_DECLINE)
        pipeline.pd_review = config.get("pd_review", DEFAULT_PD_REVIEW)
        pipeline.review_grades = set(config.get("review_grades", list(REVIEW_GRADES)))
        pipeline._fit_timestamp = config.get("fit_timestamp")
        pipeline._train_stats = config.get("train_stats", {})
        pipeline._fitted = True

        # Grade E sub-segmentation: backward-compatible load
        # Old pipelines without grade_e_config get the default config
        pipeline.grade_e_config = config.get("grade_e_config", dict(DEFAULT_GRADE_E_CONFIG))

        # Review queue NPV optimization: backward-compatible load
        # Old pipelines without review_optimization get defaults (disabled)
        pipeline.review_optimization = config.get("review_optimization", dict(DEFAULT_REVIEW_OPTIMIZATION))

        # Optionally load PaymentMonitor if the file exists
        monitor_path = directory / "payment_monitor.joblib"
        if monitor_path.exists():
            try:
                pipeline.payment_monitor = PaymentMonitor.load(monitor_path)
                logger.info(
                    "Loaded PaymentMonitor from %s (monitoring enabled)",
                    monitor_path,
                )
            except Exception as e:
                logger.warning(
                    "PaymentMonitor file found at %s but failed to load: %s. "
                    "Monitoring will be unavailable.",
                    monitor_path, e,
                )
                pipeline.payment_monitor = None
        else:
            pipeline.payment_monitor = None
            logger.info(
                "No payment_monitor.joblib found in %s — monitoring disabled. "
                "Origination scoring (.score()) works normally.",
                directory,
            )

        # Optionally load SurvivalScorer if the file exists
        survival_path = directory / "survival_scorer.joblib"
        if survival_path.exists():
            try:
                from scripts.models.survival_scorer import SurvivalScorer
                pipeline.survival_scorer = SurvivalScorer.load(str(survival_path))
                logger.info(
                    "Loaded SurvivalScorer from %s (survival enrichment enabled)",
                    survival_path,
                )
            except Exception as e:
                logger.warning(
                    "SurvivalScorer file found at %s but failed to load: %s. "
                    "Survival enrichment will be unavailable.",
                    survival_path, e,
                )
                pipeline.survival_scorer = None
        else:
            pipeline.survival_scorer = None
            logger.info(
                "No survival_scorer.joblib found in %s -- survival enrichment disabled. "
                "Origination scoring (.score()) works normally.",
                directory,
            )

        logger.info(f"Loaded from {directory}/")
        return pipeline

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_nan(value) -> bool:
        """Check if a value is NaN (float NaN or numpy nan)."""
        if value is None:
            return False
        try:
            return math.isnan(float(value))
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _normalize_none(value):
        """Normalize string 'None' / 'nan' / 'NaN' to actual None."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("none", "nan", "null", ""):
            return None
        return value

    def _validate_input(self, features: Any) -> Dict[str, Any]:
        """Validate and normalize a single-application feature dict.

        Raises ValueError with a descriptive message on hard errors.
        Logs warnings for unusual-but-acceptable inputs.

        Parameters
        ----------
        features : Any
            Expected to be a dict of feature values.

        Returns
        -------
        dict — the (possibly normalized) feature dictionary.
        """
        # ---- 1. Type check ----
        if not isinstance(features, dict):
            raise ValueError(
                f"'features' must be a dict, got {type(features).__name__}. "
                f"Example: {{'fico': 620, 'qi': 'MISSING', 'partner': 'HONEYBOOK'}}"
            )

        if len(features) == 0:
            raise ValueError(
                "Empty feature dict. At minimum, provide 'j_latest_fico_score' "
                "(or 'fico')."
            )

        # Work on a shallow copy so we don't mutate the caller's dict
        feat = dict(features)

        # ---- 2. Normalize string sentinels ----
        for key in list(feat.keys()):
            feat[key] = self._normalize_none(feat[key])

        # ---- 3. FICO validation ----
        # Use explicit None-check rather than truthiness so FICO=0 isn't skipped
        fico_raw = feat.get("j_latest_fico_score")
        if fico_raw is None:
            fico_raw = feat.get("fico")
        if fico_raw is None:
            fico_raw = feat.get("experian_FICO_SCORE")

        if fico_raw is not None and not self._is_nan(fico_raw):
            # Must be numeric
            try:
                fico_val = float(fico_raw)
            except (ValueError, TypeError):
                raise ValueError(
                    f"FICO score must be numeric, got {type(fico_raw).__name__} "
                    f"value '{fico_raw}'. Convert to int/float first."
                )

            # Range check
            if fico_val < FICO_MIN or fico_val > FICO_MAX:
                raise ValueError(
                    f"FICO score {fico_val} is out of valid range "
                    f"[{FICO_MIN}, {FICO_MAX}]."
                )

            # Boundary warnings
            if fico_val == FICO_MIN:
                logger.warning(
                    "FICO score is exactly %d (minimum). This is extremely "
                    "rare — verify this is correct.", FICO_MIN
                )
            elif fico_val == FICO_MAX:
                logger.warning(
                    "FICO score is exactly %d (maximum). Verify this is "
                    "correct.", FICO_MAX
                )

            # Ensure the value is stored as a float (handles string "620" -> 620.0)
            if isinstance(fico_raw, str):
                for fico_key in ("j_latest_fico_score", "fico", "experian_FICO_SCORE"):
                    if fico_key in feat and feat[fico_key] is not None:
                        feat[fico_key] = fico_val
        else:
            # FICO is missing — this is acceptable (sub-models handle missing),
            # but it is unusual, so warn.
            logger.warning(
                "FICO score is missing or NaN. The pipeline will use defaults, "
                "but predictions will be less reliable."
            )

        # ---- 4. QI validation ----
        qi_raw = feat.get("qi") or feat.get("shop_qi")
        if qi_raw is not None and not self._is_nan(qi_raw):
            qi_str = str(qi_raw).strip().upper()
            if qi_str not in VALID_QI_VALUES:
                raise ValueError(
                    f"QI value '{qi_raw}' is not recognized. "
                    f"Must be one of: {sorted(VALID_QI_VALUES)}."
                )
            # Normalize to uppercase
            if "qi" in feat and feat["qi"] is not None:
                feat["qi"] = qi_str
            if "shop_qi" in feat and feat["shop_qi"] is not None:
                feat["shop_qi"] = qi_str

        # ---- 5. Partner validation ----
        partner_raw = feat.get("partner")
        if partner_raw is not None and not self._is_nan(partner_raw):
            if not isinstance(partner_raw, str):
                raise ValueError(
                    f"'partner' must be a string, got {type(partner_raw).__name__} "
                    f"value '{partner_raw}'."
                )

        # ---- 6. Numeric field type validation ----
        errors = []
        for field_name in KNOWN_NUMERIC_FIELDS:
            if field_name in feat:
                val = feat[field_name]
                if val is None or self._is_nan(val):
                    continue  # Missing/NaN is fine — sub-models handle it
                # Try to coerce; error only if impossible
                try:
                    float(val)
                except (ValueError, TypeError):
                    errors.append(
                        f"  - '{field_name}': expected numeric, got "
                        f"{type(val).__name__} value '{val}'"
                    )
        if errors:
            raise ValueError(
                "Non-numeric values found in numeric fields:\n"
                + "\n".join(errors)
            )

        return feat

    def _validate_dataframe(self, df: Any) -> pd.DataFrame:
        """Validate a batch-scoring DataFrame.

        Raises ValueError with a descriptive message if the input is not
        a DataFrame or is missing required columns.

        Parameters
        ----------
        df : Any
            Expected to be a pandas DataFrame.

        Returns
        -------
        pd.DataFrame — the validated DataFrame (unchanged).
        """
        # ---- 1. Type check ----
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"'df' must be a pandas DataFrame, got {type(df).__name__}."
            )

        if len(df) == 0:
            raise ValueError(
                "Empty DataFrame. Provide at least one row to score."
            )

        # ---- 2. Check for at least one FICO alias ----
        has_fico = bool(BATCH_FICO_ALIASES & set(df.columns))
        if not has_fico:
            raise ValueError(
                f"DataFrame must contain at least one FICO column. "
                f"Expected one of: {sorted(BATCH_FICO_ALIASES)}. "
                f"Got columns: {sorted(df.columns.tolist())}"
            )

        # ---- 3. Warn about missing optional-but-useful columns ----
        optional_useful = {"qi", "shop_qi", "partner"}
        present_optional = optional_useful & set(df.columns)
        missing_optional = optional_useful - set(df.columns)
        if missing_optional:
            logger.warning(
                "Optional columns missing from DataFrame: %s. "
                "Pipeline will use defaults for these fields.",
                sorted(missing_optional),
            )

        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Pipeline is not fitted. Call .fit(train_df, y_col) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        monitor_str = ", monitor=loaded" if self.payment_monitor is not None else ""
        survival_str = ", survival=loaded" if self.survival_scorer is not None else ""
        if self._fitted:
            return (
                f"Pipeline({status}, "
                f"fraud={self.fraud_gate}, "
                f"default={self.default_scorecard}, "
                f"grader={self.credit_grader}"
                f"{monitor_str}{survival_str})"
            )
        return f"Pipeline({status}{monitor_str}{survival_str})"
