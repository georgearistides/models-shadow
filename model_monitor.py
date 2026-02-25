"""
ModelMonitor — Model monitoring framework for shadow deployment.

Tracks model health over time by computing stability metrics on batches
of scored applications against reference (training) distributions.

Monitors:
  1. Population Stability Index (PSI) for score distributions
  2. AUC decay tracking against reference performance
  3. Grade drift detection via chi-squared test
  4. Calibration monitoring (predicted PD vs actual default rate by decile)

Usage:
    from model_monitor import ModelMonitor

    monitor = ModelMonitor.from_pipeline("models/")
    report = monitor.analyze(scored_df)
    monitor.save_report(report, "results/monitoring/2025-07-report.json")

Status Levels:
    GREEN   - All metrics within acceptable bounds
    WARNING - One or more metrics in caution range
    RED     - Critical drift detected, immediate attention needed
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sklearn.metrics import roc_auc_score, brier_score_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

AUC_DECAY_ALERT = 0.03

GRADE_DRIFT_P_WARNING = 0.05
GRADE_DRIFT_P_RED = 0.01

CALIBRATION_RATIO_WARNING = 1.5
CALIBRATION_RATIO_RED = 2.0
CALIBRATION_RATIO_LOW_WARNING = 0.5
CALIBRATION_RATIO_LOW_RED = 0.25

BAD_STATES = {"CHARGE_OFF", "DEFAULT", "WORKOUT", "PRE_DEFAULT", "TECHNICAL_DEFAULT"}


# ---------------------------------------------------------------------------
# Helper: PSI computation
# ---------------------------------------------------------------------------

def compute_psi_continuous(reference: np.ndarray, current: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
    """Compute Population Stability Index for continuous scores.

    Uses equal-width bins computed from the reference distribution.
    Adds a small epsilon to avoid log(0) or division by zero.

    Parameters
    ----------
    reference : array-like
        Reference (training) score distribution.
    current : array-like
        Current (production) score distribution.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    dict with keys: psi, bin_psis, bin_edges, ref_pcts, cur_pcts
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Remove NaN/Inf
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return {"psi": np.nan, "bin_psis": [], "bin_edges": [],
                "ref_pcts": [], "cur_pcts": [], "error": "Empty input"}

    # Compute bin edges from reference distribution
    ref_min, ref_max = reference.min(), reference.max()
    # Extend slightly to capture outliers in current
    margin = (ref_max - ref_min) * 0.01 if ref_max > ref_min else 0.01
    bin_edges = np.linspace(ref_min - margin, ref_max + margin, n_bins + 1)

    # Count observations in each bin
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Convert to proportions with epsilon smoothing
    epsilon = 1e-6
    ref_pcts = (ref_counts / len(reference)) + epsilon
    cur_pcts = (cur_counts / len(current)) + epsilon

    # Normalize after epsilon addition
    ref_pcts = ref_pcts / ref_pcts.sum()
    cur_pcts = cur_pcts / cur_pcts.sum()

    # PSI per bin
    bin_psis = (cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)
    total_psi = float(bin_psis.sum())

    return {
        "psi": round(total_psi, 6),
        "bin_psis": [round(float(x), 6) for x in bin_psis],
        "bin_edges": [round(float(x), 4) for x in bin_edges],
        "ref_pcts": [round(float(x), 4) for x in ref_pcts],
        "cur_pcts": [round(float(x), 4) for x in cur_pcts],
    }


def compute_psi_categorical(reference: np.ndarray, current: np.ndarray,
                             categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute Population Stability Index for categorical variables (e.g., grades).

    Uses exact category matching.

    Parameters
    ----------
    reference : array-like
        Reference category values.
    current : array-like
        Current category values.
    categories : list, optional
        Expected categories. If None, union of reference and current is used.

    Returns
    -------
    dict with keys: psi, category_psis, categories, ref_pcts, cur_pcts
    """
    reference = np.asarray(reference)
    current = np.asarray(current)

    if categories is None:
        categories = sorted(set(reference) | set(current))

    epsilon = 1e-6
    n_ref = len(reference)
    n_cur = len(current)

    if n_ref == 0 or n_cur == 0:
        return {"psi": np.nan, "category_psis": {}, "categories": categories,
                "ref_pcts": {}, "cur_pcts": {}, "error": "Empty input"}

    ref_pcts = {}
    cur_pcts = {}
    category_psis = {}
    total_psi = 0.0

    for cat in categories:
        ref_count = float(np.sum(reference == cat))
        cur_count = float(np.sum(current == cat))

        ref_p = (ref_count / n_ref) + epsilon
        cur_p = (cur_count / n_cur) + epsilon

        cat_psi = (cur_p - ref_p) * np.log(cur_p / ref_p)
        total_psi += cat_psi

        ref_pcts[str(cat)] = round(ref_count / n_ref, 4)
        cur_pcts[str(cat)] = round(cur_count / n_cur, 4)
        category_psis[str(cat)] = round(float(cat_psi), 6)

    return {
        "psi": round(total_psi, 6),
        "category_psis": category_psis,
        "categories": [str(c) for c in categories],
        "ref_pcts": ref_pcts,
        "cur_pcts": cur_pcts,
    }


# ---------------------------------------------------------------------------
# ModelMonitor class
# ---------------------------------------------------------------------------

class ModelMonitor:
    """Model monitoring framework for shadow deployment.

    Compares current (production) scoring batches against reference
    (training) distributions to detect model degradation and drift.

    Parameters
    ----------
    reference_scores : dict
        Reference distributions. Keys: 'fraud_score', 'pd', 'grade'.
        Each value is a numpy array of training-set scores.
    reference_auc : float
        AUC on the reference (training or validation) set.
    reference_grade_dist : dict
        Expected grade distribution {grade: proportion}.
    reference_grade_bad_rates : dict
        Expected grade-level bad rates {grade: rate}.
    grade_labels : list
        Ordered grade labels (e.g., ['A', 'B', 'C', 'D', 'F']).
    reference_calibration : dict, optional
        Expected calibration by decile {decile: {'predicted_mean': float, 'actual_rate': float}}.
    pipeline_config : dict, optional
        Pipeline configuration metadata (version, thresholds, etc.).
    """

    def __init__(
        self,
        reference_scores: Dict[str, np.ndarray],
        reference_auc: float,
        reference_grade_dist: Dict[str, float],
        reference_grade_bad_rates: Dict[str, float],
        grade_labels: List[str],
        reference_calibration: Optional[Dict] = None,
        pipeline_config: Optional[Dict] = None,
    ):
        self.reference_scores = reference_scores
        self.reference_auc = reference_auc
        self.reference_grade_dist = reference_grade_dist
        self.reference_grade_bad_rates = reference_grade_bad_rates
        self.grade_labels = grade_labels
        self.reference_calibration = reference_calibration or {}
        self.pipeline_config = pipeline_config or {}

    # ------------------------------------------------------------------
    # Factory: load from a trained pipeline directory
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        model_dir: Union[str, Path],
        data_path: Optional[Union[str, Path]] = None,
    ) -> "ModelMonitor":
        """Create a ModelMonitor by loading a trained pipeline and computing
        reference distributions from the training set.

        Parameters
        ----------
        model_dir : str or Path
            Directory containing pipeline artifacts (*.joblib files).
        data_path : str or Path, optional
            Path to master_features.parquet. Defaults to data/master_features.parquet
            relative to the project root.

        Returns
        -------
        ModelMonitor instance ready for .analyze() calls.
        """
        import sys
        model_dir = Path(model_dir)
        project_root = model_dir

        if data_path is None:
            data_path = project_root / "master_features.parquet"
        data_path = Path(data_path)

        from pipeline import Pipeline
        from model_utils import (
            BAD_STATES as BAD_STATES_UTIL, GOOD_STATES, EXCLUDE_STATES,
            SPLIT_DATE, PARTNER_ENCODING,
        )

        # Load pipeline
        logger.info(f"Loading pipeline from {model_dir}")
        pipeline = Pipeline.load(model_dir)

        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)

        # Prepare data (mirror train_all.py logic)
        if "loan_state" in df.columns:
            df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
        if "is_bad" not in df.columns:
            df["is_bad"] = df["loan_state"].isin(BAD_STATES_UTIL).astype(int)
        df["signing_date"] = pd.to_datetime(df["signing_date"])

        if "shop_qi" in df.columns and "qi" not in df.columns:
            df["qi"] = df["shop_qi"]
        if "experian_FICO_SCORE" in df.columns and "fico" not in df.columns:
            df["fico"] = df["experian_FICO_SCORE"]

        # Split
        train_df = df[df["signing_date"] < SPLIT_DATE].copy()
        logger.info(f"Reference population: {len(train_df)} training samples")

        # Score training set to get reference distributions
        logger.info("Scoring training set for reference distributions...")
        train_results = pipeline.score_batch(train_df)
        y_train = train_df["is_bad"].astype(int).values

        # Reference scores
        reference_scores = {
            "fraud_score": train_results["fraud_score"].values.copy(),
            "pd": train_results["pd"].values.copy(),
        }

        # Reference AUC
        reference_auc = float(roc_auc_score(y_train, train_results["pd"].values))
        logger.info(f"Reference AUC (train): {reference_auc:.4f}")

        # Reference grade distribution and bad rates
        grades = train_results["grade"].values
        grade_labels = list(pipeline.credit_grader.grades)

        ref_grade_dist = {}
        ref_grade_bad_rates = {}
        for g in grade_labels:
            mask = grades == g
            n = int(mask.sum())
            ref_grade_dist[g] = round(n / len(grades), 4) if len(grades) > 0 else 0
            if n > 0:
                ref_grade_bad_rates[g] = round(float(y_train[mask].mean()), 4)
            else:
                ref_grade_bad_rates[g] = 0.0

        # Reference grade values for PSI
        reference_scores["grade"] = grades.copy()

        # Reference calibration by decile
        ref_calibration = cls._compute_calibration_table(
            train_results["pd"].values, y_train
        )

        # Pipeline config
        config = {
            "version": getattr(pipeline, '_version', "1.0.0"),
            "fit_timestamp": pipeline._fit_timestamp,
            "train_stats": pipeline._train_stats,
            "n_train": len(train_df),
            "train_bad_rate": round(float(y_train.mean()), 4),
        }

        logger.info("ModelMonitor initialized from pipeline.")
        return cls(
            reference_scores=reference_scores,
            reference_auc=reference_auc,
            reference_grade_dist=ref_grade_dist,
            reference_grade_bad_rates=ref_grade_bad_rates,
            grade_labels=grade_labels,
            reference_calibration=ref_calibration,
            pipeline_config=config,
        )

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        scored_df: pd.DataFrame,
        y_col: str = "is_bad",
        period_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run all monitoring checks on a batch of scored applications.

        Parameters
        ----------
        scored_df : pd.DataFrame
            DataFrame that must contain columns produced by Pipeline.score_batch():
            'fraud_score', 'pd', 'grade', and optionally the target column.
            If y_col is present, AUC and calibration checks are performed.
        y_col : str
            Name of the binary target column (1 = bad).
        period_label : str, optional
            Label for this monitoring period (e.g., "2025-07").

        Returns
        -------
        dict with keys:
            psi, auc, grade_drift, calibration, overall_status, alerts, metadata
        """
        report = {
            "metadata": {
                "period": period_label or datetime.utcnow().strftime("%Y-%m-%d"),
                "n_scored": len(scored_df),
                "timestamp": datetime.utcnow().isoformat(),
                "reference_auc": round(self.reference_auc, 4),
            },
            "psi": {},
            "auc": {},
            "grade_drift": {},
            "calibration": {},
            "overall_status": "GREEN",
            "alerts": [],
        }

        has_outcomes = y_col in scored_df.columns
        if has_outcomes:
            y_true = scored_df[y_col].astype(int).values
            report["metadata"]["n_bad"] = int(y_true.sum())
            report["metadata"]["bad_rate"] = round(float(y_true.mean()), 4)
        else:
            y_true = None

        # --- 1. PSI Analysis ---
        report["psi"] = self._compute_psi(scored_df)

        # --- 2. AUC Decay ---
        if has_outcomes:
            report["auc"] = self._compute_auc_decay(scored_df, y_true)

        # --- 3. Grade Drift ---
        report["grade_drift"] = self._compute_grade_drift(scored_df, y_true)

        # --- 4. Calibration ---
        if has_outcomes:
            report["calibration"] = self._compute_calibration(scored_df, y_true)

        # --- 5. Determine overall status ---
        report["overall_status"], report["alerts"] = self._determine_status(report)

        return report

    # ------------------------------------------------------------------
    # 1. PSI
    # ------------------------------------------------------------------

    def _compute_psi(self, scored_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute PSI for fraud_score, pd, and grade."""
        psi_results = {}

        # Continuous scores
        for score_name in ["fraud_score", "pd"]:
            if score_name in scored_df.columns and score_name in self.reference_scores:
                ref = self.reference_scores[score_name]
                cur = scored_df[score_name].values
                result = compute_psi_continuous(ref, cur, n_bins=10)
                psi_results[score_name] = result["psi"]
                psi_results[f"{score_name}_detail"] = result
            else:
                psi_results[score_name] = np.nan

        # Categorical: grade
        if "grade" in scored_df.columns and "grade" in self.reference_scores:
            ref_grades = self.reference_scores["grade"]
            cur_grades = scored_df["grade"].values
            result = compute_psi_categorical(ref_grades, cur_grades,
                                              categories=self.grade_labels)
            psi_results["grade"] = result["psi"]
            psi_results["grade_detail"] = result
        else:
            psi_results["grade"] = np.nan

        return psi_results

    # ------------------------------------------------------------------
    # 2. AUC Decay
    # ------------------------------------------------------------------

    def _compute_auc_decay(
        self, scored_df: pd.DataFrame, y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Compare current AUC to reference."""
        result = {
            "current": None,
            "reference": round(self.reference_auc, 4),
            "delta": None,
            "alert": False,
        }

        if "pd" not in scored_df.columns:
            result["error"] = "No 'pd' column in scored_df"
            return result

        # Need at least 2 classes
        if len(np.unique(y_true)) < 2:
            result["error"] = "Only one class present in outcomes"
            return result

        try:
            current_auc = float(roc_auc_score(y_true, scored_df["pd"].values))
            delta = current_auc - self.reference_auc
            alert = delta < -AUC_DECAY_ALERT

            result["current"] = round(current_auc, 4)
            result["delta"] = round(delta, 4)
            result["alert"] = alert

            # Also compute component AUCs if available
            component_aucs = {}
            for comp in ["fraud_score", "woe_pd", "rule_pd", "xgb_pd"]:
                if comp in scored_df.columns:
                    try:
                        component_aucs[comp] = round(
                            float(roc_auc_score(y_true, scored_df[comp].values)), 4
                        )
                    except ValueError:
                        component_aucs[comp] = None
            if component_aucs:
                result["component_aucs"] = component_aucs

        except ValueError as e:
            result["error"] = str(e)

        return result

    # ------------------------------------------------------------------
    # 3. Grade Drift
    # ------------------------------------------------------------------

    def _compute_grade_drift(
        self,
        scored_df: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Detect grade distribution shift using chi-squared test.

        Also tracks per-grade bad rates vs expected if outcomes available.
        """
        from scipy.stats import chi2_contingency, chisquare

        result = {
            "chi2": None,
            "p_value": None,
            "alert": False,
            "current_dist": {},
            "reference_dist": dict(self.reference_grade_dist),
            "grade_bad_rates": {},
            "expected_bad_rates": dict(self.reference_grade_bad_rates),
        }

        if "grade" not in scored_df.columns:
            result["error"] = "No 'grade' column in scored_df"
            return result

        cur_grades = scored_df["grade"].values
        n_cur = len(cur_grades)

        # Current distribution
        for g in self.grade_labels:
            count = int(np.sum(cur_grades == g))
            result["current_dist"][g] = round(count / n_cur, 4) if n_cur > 0 else 0

        # Chi-squared test: observed vs expected counts
        observed = []
        expected = []
        for g in self.grade_labels:
            obs_count = int(np.sum(cur_grades == g))
            exp_count = self.reference_grade_dist.get(g, 0) * n_cur
            observed.append(obs_count)
            expected.append(max(exp_count, 1))  # Avoid zero expected

        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)

        try:
            chi2_stat, p_value = chisquare(observed, f_exp=expected)[:2]
            result["chi2"] = round(float(chi2_stat), 4)
            result["p_value"] = round(float(p_value), 6)
            result["alert"] = p_value < GRADE_DRIFT_P_WARNING
        except Exception as e:
            result["error"] = str(e)

        # Per-grade bad rates (if outcomes available)
        if y_true is not None:
            for g in self.grade_labels:
                mask = cur_grades == g
                n = int(mask.sum())
                if n > 0:
                    actual_rate = float(y_true[mask].mean())
                    expected_rate = self.reference_grade_bad_rates.get(g, 0)
                    ratio = actual_rate / max(expected_rate, 0.001) if expected_rate > 0 else None
                    result["grade_bad_rates"][g] = {
                        "actual": round(actual_rate, 4),
                        "expected": round(expected_rate, 4),
                        "ratio": round(ratio, 4) if ratio is not None else None,
                        "n": n,
                        "n_bad": int(y_true[mask].sum()),
                    }

        return result

    # ------------------------------------------------------------------
    # 4. Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_calibration_table(
        predicted: np.ndarray, actual: np.ndarray, n_deciles: int = 10
    ) -> Dict[int, Dict[str, float]]:
        """Compute calibration table: predicted vs actual by decile.

        Parameters
        ----------
        predicted : array
            Predicted probabilities.
        actual : array
            Binary outcomes (0/1).
        n_deciles : int
            Number of buckets.

        Returns
        -------
        dict of {decile_index: {'predicted_mean', 'actual_rate', 'n', 'n_bad'}}
        """
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=int)

        try:
            decile_labels = pd.qcut(predicted, q=n_deciles,
                                     labels=False, duplicates="drop")
        except ValueError:
            # If too few unique values, use cut instead
            decile_labels = pd.cut(predicted, bins=n_deciles,
                                    labels=False, duplicates="drop")

        cal_table = {}
        for d in sorted(np.unique(decile_labels[~np.isnan(decile_labels)])):
            mask = decile_labels == d
            n = int(mask.sum())
            if n > 0:
                cal_table[int(d)] = {
                    "predicted_mean": round(float(predicted[mask].mean()), 4),
                    "actual_rate": round(float(actual[mask].mean()), 4),
                    "n": n,
                    "n_bad": int(actual[mask].sum()),
                }

        return cal_table

    def _compute_calibration(
        self,
        scored_df: pd.DataFrame,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """Compare predicted PD vs actual default rate by decile.

        Computes Brier score and flags deciles with poor calibration.
        """
        result = {
            "brier": None,
            "worst_decile_ratio": None,
            "worst_decile_idx": None,
            "alert": False,
            "decile_table": {},
            "reference_calibration": self.reference_calibration,
        }

        if "pd" not in scored_df.columns:
            result["error"] = "No 'pd' column in scored_df"
            return result

        predicted = scored_df["pd"].values

        # Brier score
        try:
            brier = float(brier_score_loss(y_true, predicted))
            result["brier"] = round(brier, 4)
        except ValueError as e:
            result["brier_error"] = str(e)

        # Calibration by decile
        cal_table = self._compute_calibration_table(predicted, y_true)
        result["decile_table"] = cal_table

        # Find worst decile ratio (actual / predicted)
        worst_ratio = 1.0  # Start at perfect calibration
        worst_idx = None
        alert_deciles = []

        for d_idx, d_stats in cal_table.items():
            pred_mean = d_stats["predicted_mean"]
            actual_rate = d_stats["actual_rate"]

            if pred_mean > 0.001:
                ratio = actual_rate / pred_mean
            elif actual_rate > 0:
                ratio = float("inf")
            else:
                ratio = 1.0

            d_stats["ratio"] = round(ratio, 4)

            # Track worst ratio (furthest from 1.0 in either direction)
            deviation = abs(ratio - 1.0)
            if deviation > abs(worst_ratio - 1.0):
                worst_ratio = ratio
                worst_idx = d_idx

            # Flag problematic deciles
            if ratio > CALIBRATION_RATIO_WARNING or ratio < CALIBRATION_RATIO_LOW_WARNING:
                alert_deciles.append({
                    "decile": d_idx,
                    "ratio": round(ratio, 4),
                    "predicted": pred_mean,
                    "actual": actual_rate,
                    "n": d_stats["n"],
                })

        result["worst_decile_ratio"] = round(worst_ratio, 4) if worst_idx is not None else None
        result["worst_decile_idx"] = worst_idx
        result["alert_deciles"] = alert_deciles
        result["alert"] = len(alert_deciles) > 0

        return result

    # ------------------------------------------------------------------
    # 5. Overall status determination
    # ------------------------------------------------------------------

    def _determine_status(self, report: Dict[str, Any]) -> tuple:
        """Determine overall status (GREEN/WARNING/RED) and compile alerts."""
        alerts = []
        status = "GREEN"

        # PSI checks
        for score_name in ["fraud_score", "pd", "grade"]:
            psi_val = report["psi"].get(score_name)
            if psi_val is not None and not np.isnan(psi_val):
                if psi_val > PSI_MODERATE:
                    alerts.append(
                        f"CRITICAL: {score_name} PSI = {psi_val:.4f} "
                        f"(> {PSI_MODERATE} = significant shift)"
                    )
                    status = "RED"
                elif psi_val > PSI_STABLE:
                    alerts.append(
                        f"WARNING: {score_name} PSI = {psi_val:.4f} "
                        f"({PSI_STABLE}-{PSI_MODERATE} = moderate shift)"
                    )
                    if status != "RED":
                        status = "WARNING"

        # AUC decay check
        auc_info = report.get("auc", {})
        if auc_info.get("alert"):
            delta = auc_info.get("delta", 0)
            alerts.append(
                f"WARNING: AUC dropped by {abs(delta):.4f} from reference "
                f"(current={auc_info.get('current')}, "
                f"reference={auc_info.get('reference')})"
            )
            if status != "RED":
                status = "WARNING"

        # Grade drift check
        grade_info = report.get("grade_drift", {})
        if grade_info.get("alert"):
            p_val = grade_info.get("p_value")
            if p_val is not None:
                if p_val < GRADE_DRIFT_P_RED:
                    alerts.append(
                        f"CRITICAL: Grade distribution shift detected "
                        f"(chi2={grade_info.get('chi2')}, p={p_val:.6f})"
                    )
                    status = "RED"
                else:
                    alerts.append(
                        f"WARNING: Grade distribution shift detected "
                        f"(chi2={grade_info.get('chi2')}, p={p_val:.6f})"
                    )
                    if status != "RED":
                        status = "WARNING"

        # Calibration check
        cal_info = report.get("calibration", {})
        if cal_info.get("alert"):
            worst = cal_info.get("worst_decile_ratio")
            n_alert = len(cal_info.get("alert_deciles", []))
            if worst is not None:
                if worst > CALIBRATION_RATIO_RED or worst < CALIBRATION_RATIO_LOW_RED:
                    alerts.append(
                        f"CRITICAL: Calibration drift in {n_alert} decile(s), "
                        f"worst actual/predicted ratio = {worst:.2f}"
                    )
                    status = "RED"
                else:
                    alerts.append(
                        f"WARNING: Calibration drift in {n_alert} decile(s), "
                        f"worst actual/predicted ratio = {worst:.2f}"
                    )
                    if status != "RED":
                        status = "WARNING"

        if not alerts:
            alerts.append("All metrics within acceptable bounds.")

        return status, alerts

    # ------------------------------------------------------------------
    # Report I/O
    # ------------------------------------------------------------------

    def save_report(
        self, report: Dict[str, Any], path: Union[str, Path]
    ) -> str:
        """Save a monitoring report as JSON.

        Parameters
        ----------
        report : dict
            Output from .analyze().
        path : str or Path
            Destination file path.

        Returns
        -------
        str : the absolute path the report was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Make report JSON-serializable
        serializable = self._make_serializable(report)

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Report saved to {path}")
        return str(path.resolve())

    @staticmethod
    def _make_serializable(obj):
        """Recursively convert numpy types to Python native for JSON."""
        if isinstance(obj, dict):
            return {k: ModelMonitor._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ModelMonitor._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif obj is np.nan or (isinstance(obj, float) and np.isnan(obj)):
            return None
        else:
            return obj

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: Dict[str, Any]) -> str:
        """Format a monitoring report as a human-readable string.

        Returns
        -------
        str : formatted report text (also prints to stdout).
        """
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL MONITORING REPORT")
        lines.append("=" * 70)

        meta = report.get("metadata", {})
        lines.append(f"Period:     {meta.get('period', 'N/A')}")
        lines.append(f"Scored:     {meta.get('n_scored', 'N/A')} applications")
        lines.append(f"Bad rate:   {meta.get('bad_rate', 'N/A')}")
        lines.append(f"Ref AUC:    {meta.get('reference_auc', 'N/A')}")
        lines.append(f"Status:     {report.get('overall_status', 'N/A')}")
        lines.append("")

        # PSI
        lines.append("--- Population Stability Index (PSI) ---")
        psi = report.get("psi", {})
        for name in ["fraud_score", "pd", "grade"]:
            val = psi.get(name)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                level = "STABLE" if val < PSI_STABLE else ("MODERATE" if val < PSI_MODERATE else "SIGNIFICANT")
                lines.append(f"  {name:15s}: {val:.4f}  [{level}]")
        lines.append("")

        # AUC
        auc = report.get("auc", {})
        if auc.get("current") is not None:
            lines.append("--- AUC Decay ---")
            lines.append(f"  Current AUC:    {auc['current']:.4f}")
            lines.append(f"  Reference AUC:  {auc['reference']:.4f}")
            lines.append(f"  Delta:          {auc['delta']:+.4f}")
            lines.append(f"  Alert:          {auc['alert']}")
            comp_aucs = auc.get("component_aucs", {})
            if comp_aucs:
                lines.append("  Component AUCs:")
                for comp, auc_val in comp_aucs.items():
                    lines.append(f"    {comp}: {auc_val}")
            lines.append("")

        # Grade drift
        grade = report.get("grade_drift", {})
        if grade.get("chi2") is not None:
            lines.append("--- Grade Distribution Drift ---")
            lines.append(f"  Chi-squared:    {grade['chi2']:.4f}")
            lines.append(f"  P-value:        {grade['p_value']:.6f}")
            lines.append(f"  Alert:          {grade['alert']}")
            lines.append("  Distribution (ref -> current):")
            ref_dist = grade.get("reference_dist", {})
            cur_dist = grade.get("current_dist", {})
            for g in sorted(set(ref_dist) | set(cur_dist)):
                ref_pct = ref_dist.get(g, 0)
                cur_pct = cur_dist.get(g, 0)
                lines.append(f"    {g}: {ref_pct:.1%} -> {cur_pct:.1%}")

            grade_brs = grade.get("grade_bad_rates", {})
            if grade_brs:
                lines.append("  Grade bad rates (actual vs expected):")
                for g, info in sorted(grade_brs.items()):
                    actual = info.get("actual", 0)
                    expected = info.get("expected", 0)
                    ratio = info.get("ratio", "N/A")
                    n = info.get("n", 0)
                    lines.append(
                        f"    {g}: actual={actual:.2%} expected={expected:.2%} "
                        f"ratio={ratio} (n={n})"
                    )
            lines.append("")

        # Calibration
        cal = report.get("calibration", {})
        if cal.get("brier") is not None:
            lines.append("--- Calibration ---")
            lines.append(f"  Brier score:        {cal['brier']:.4f}")
            lines.append(f"  Worst decile ratio: {cal.get('worst_decile_ratio')}")
            lines.append(f"  Worst decile idx:   {cal.get('worst_decile_idx')}")
            lines.append(f"  Alert:              {cal['alert']}")
            decile_table = cal.get("decile_table", {})
            if decile_table:
                lines.append("  Decile table:")
                lines.append(f"    {'Dec':>4s}  {'Pred':>7s}  {'Actual':>7s}  "
                             f"{'Ratio':>7s}  {'N':>5s}  {'Bads':>5s}")
                lines.append(f"    {'----':>4s}  {'-------':>7s}  {'-------':>7s}  "
                             f"{'-------':>7s}  {'-----':>5s}  {'-----':>5s}")
                for d_idx in sorted(decile_table.keys(), key=int):
                    d = decile_table[d_idx]
                    flag = " ***" if d.get("ratio", 1) > CALIBRATION_RATIO_WARNING or d.get("ratio", 1) < CALIBRATION_RATIO_LOW_WARNING else ""
                    lines.append(
                        f"    {d_idx:>4d}  {d['predicted_mean']:>7.4f}  "
                        f"{d['actual_rate']:>7.4f}  {d.get('ratio', 'N/A'):>7.4f}  "
                        f"{d['n']:>5d}  {d['n_bad']:>5d}{flag}"
                    )
            lines.append("")

        # Alerts
        lines.append("--- Alerts ---")
        for a in report.get("alerts", []):
            lines.append(f"  {a}")
        lines.append("")
        lines.append("=" * 70)

        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"ModelMonitor(ref_auc={self.reference_auc:.4f}, "
            f"grades={self.grade_labels}, "
            f"ref_n={len(self.reference_scores.get('pd', []))})"
        )
