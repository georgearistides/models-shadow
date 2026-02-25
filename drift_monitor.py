#!/usr/bin/env python3
"""
DriftMonitor — Monthly Calibration & Drift Monitoring for V5.2 Pipeline
========================================================================

Production-ready monitoring script that detects model degradation by comparing
current scoring distributions to training baselines. Builds on the existing
ModelMonitor framework (model_monitor.py) and adds feature-level drift tracking.

Monitors:
  1. PSI (Population Stability Index) for fraud score, PD, and grade distributions
  2. AUC decay — current vs training AUC (when outcomes available)
  3. Calibration drift — predicted PD vs actual bad rate by decile
  4. Grade distribution shift — proportion changes vs training baseline
  5. Feature drift — PSI for top input features (fico, qi, partner, etc.)

Alert Thresholds:
  - PSI > 0.1 = WARNING, PSI > 0.25 = CRITICAL
  - AUC drop > 0.02 = WARNING, > 0.05 = CRITICAL
  - Calibration ratio outside [0.8, 1.2] = WARNING, outside [0.6, 1.5] = CRITICAL

Usage (as module):
    from drift_monitor import DriftMonitor
    monitor = DriftMonitor.from_pipeline("models/")
    report = monitor.run(current_data, outcomes=None)
    report.print_summary()
    report.to_json("results/monitoring/report_2026_02.json")

Usage (CLI):
    python3 scripts/models/drift_monitor.py --data data/master_features.parquet --output results/monitoring/
"""

import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sklearn.metrics import roc_auc_score, brier_score_loss

# Project root (flat layout: all files in same directory)
_PROJECT_ROOT = Path(__file__).resolve().parent

# Reuse PSI functions from the existing ModelMonitor
from model_monitor import (
    compute_psi_continuous,
    compute_psi_categorical,
)
from model_utils import (
    BAD_STATES,
    EXCLUDE_STATES,
    SPLIT_DATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Alert thresholds (per task specification)
# ---------------------------------------------------------------------------

# PSI thresholds
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25

# AUC decay thresholds (magnitude of drop)
AUC_DROP_WARNING = 0.02
AUC_DROP_CRITICAL = 0.05

# Calibration ratio thresholds (actual / predicted)
CAL_RATIO_LOW_CRITICAL = 0.6
CAL_RATIO_LOW_WARNING = 0.8
CAL_RATIO_HIGH_WARNING = 1.2
CAL_RATIO_HIGH_CRITICAL = 1.5

# Top features to monitor for drift
MONITORED_FEATURES_CONTINUOUS = ["fico"]
MONITORED_FEATURES_CATEGORICAL = ["qi", "partner"]

# Additional continuous bureau features to monitor
MONITORED_BUREAU_FEATURES = ["d30", "revutil"]

# ---------------------------------------------------------------------------
# Operational drift thresholds (from V5.3 shadow scoring data)
# ---------------------------------------------------------------------------
# These catch failure modes that AUC monitoring misses, e.g. QI outage
# causing approve rate to collapse from 54% to 0.7% while AUC stays stable.

OPERATIONAL_THRESHOLDS = {
    "approve_rate": {
        "expected_range": (0.45, 0.65),
        "warning_range": (0.35, 0.75),
        "critical_range": (0.10, 0.85),
        "critical_floor": 0.10,  # Below this is always CRITICAL (QI outage signature)
    },
    "review_rate": {
        "expected_range": (0.30, 0.55),
        "warning_range": (0.20, 0.65),
        "critical_range": (0.10, 0.80),
    },
    "decline_rate": {
        "expected_range": (0.02, 0.10),
        "warning_range": (0.01, 0.15),
        "critical_range": (0.00, 0.25),  # Only upper critical matters
    },
    "mean_pd": {
        "expected_range": (0.06, 0.14),
        "warning_range": (0.03, 0.20),
        "critical_range": (0.01, 0.30),
    },
}


# ---------------------------------------------------------------------------
# DriftReport — container for monitoring results
# ---------------------------------------------------------------------------

class DriftReport:
    """Container for drift monitoring results with print and export methods.

    Attributes
    ----------
    metadata : dict
        Period, timestamp, counts.
    score_psi : dict
        PSI results for fraud_score, pd, grade.
    feature_psi : dict
        PSI results for monitored input features.
    auc_decay : dict
        AUC comparison (current vs reference).
    calibration : dict
        Decile-level calibration analysis.
    grade_shift : dict
        Grade distribution comparison.
    operational_drift : dict
        Operational metric drift (approve rate, mean PD, etc.).
    overall_status : str
        GREEN / WARNING / CRITICAL.
    alerts : list of str
        Human-readable alert messages.
    """

    def __init__(
        self,
        metadata: Dict[str, Any],
        score_psi: Dict[str, Any],
        feature_psi: Dict[str, Any],
        auc_decay: Dict[str, Any],
        calibration: Dict[str, Any],
        grade_shift: Dict[str, Any],
        overall_status: str,
        alerts: List[str],
        operational_drift: Optional[Dict[str, Any]] = None,
    ):
        self.metadata = metadata
        self.score_psi = score_psi
        self.feature_psi = feature_psi
        self.auc_decay = auc_decay
        self.calibration = calibration
        self.grade_shift = grade_shift
        self.operational_drift = operational_drift or {}
        self.overall_status = overall_status
        self.alerts = alerts

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    def print_summary(self) -> str:
        """Print a human-readable monitoring summary to stdout.

        Returns the formatted text string.
        """
        lines = []
        lines.append("=" * 72)
        lines.append("  MONTHLY DRIFT MONITORING REPORT")
        lines.append("=" * 72)

        # Metadata
        meta = self.metadata
        lines.append(f"  Period:        {meta.get('period', 'N/A')}")
        lines.append(f"  Timestamp:     {meta.get('timestamp', 'N/A')}")
        lines.append(f"  Applications:  {meta.get('n_current', 'N/A'):,}")
        lines.append(f"  Training ref:  {meta.get('n_training', 'N/A'):,}")
        if meta.get("current_bad_rate") is not None:
            lines.append(f"  Bad rate:      {meta['current_bad_rate']:.2%} "
                         f"(training: {meta.get('training_bad_rate', 0):.2%})")
        else:
            lines.append(f"  Bad rate:      N/A (no outcomes)")
        lines.append(f"  Status:        {self.overall_status}")
        lines.append("")

        # --- Score PSI ---
        lines.append("--- Score Distribution Stability (PSI) ---")
        for name in ["fraud_score", "pd", "grade"]:
            val = self.score_psi.get(name, {}).get("psi")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                level = self._psi_level(val)
                lines.append(f"  {name:15s}: {val:.4f}  [{level}]")
            else:
                lines.append(f"  {name:15s}: N/A")
        lines.append("")

        # --- Feature PSI ---
        lines.append("--- Feature Drift (PSI) ---")
        for feat_name, feat_info in self.feature_psi.items():
            val = feat_info.get("psi")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                level = self._psi_level(val)
                lines.append(f"  {feat_name:15s}: {val:.4f}  [{level}]")
            else:
                lines.append(f"  {feat_name:15s}: N/A")
        lines.append("")

        # --- AUC Decay ---
        auc = self.auc_decay
        if auc.get("current_auc") is not None:
            lines.append("--- AUC Decay ---")
            lines.append(f"  Current AUC:    {auc['current_auc']:.4f}")
            lines.append(f"  Reference AUC:  {auc['reference_auc']:.4f}")
            delta = auc.get("delta", 0)
            lines.append(f"  Delta:          {delta:+.4f}")
            level = "OK"
            if abs(delta) >= AUC_DROP_CRITICAL:
                level = "CRITICAL"
            elif abs(delta) >= AUC_DROP_WARNING:
                level = "WARNING"
            lines.append(f"  Status:         [{level}]")

            comp = auc.get("component_aucs", {})
            if comp:
                lines.append("  Component AUCs:")
                for c_name, c_auc in comp.items():
                    lines.append(f"    {c_name:12s}: {c_auc:.4f}" if c_auc else f"    {c_name:12s}: N/A")
            lines.append("")
        elif auc.get("error"):
            lines.append("--- AUC Decay ---")
            lines.append(f"  Skipped: {auc['error']}")
            lines.append("")

        # --- Calibration ---
        cal = self.calibration
        if cal.get("brier") is not None:
            lines.append("--- Calibration Drift ---")
            lines.append(f"  Brier score:            {cal['brier']:.4f}")
            lines.append(f"  Mean calibration ratio: {cal.get('mean_ratio', 'N/A')}")
            lines.append(f"  Worst decile ratio:     {cal.get('worst_ratio', 'N/A')}"
                         f" (decile {cal.get('worst_decile_idx', '?')})")

            n_warning = cal.get("n_warning_deciles", 0)
            n_critical = cal.get("n_critical_deciles", 0)
            lines.append(f"  Warning deciles:        {n_warning}")
            lines.append(f"  Critical deciles:       {n_critical}")

            decile_table = cal.get("decile_table", {})
            if decile_table:
                lines.append("")
                lines.append(f"  {'Dec':>4s}  {'Pred':>7s}  {'Actual':>7s}  "
                             f"{'Ratio':>7s}  {'N':>6s}  {'Bads':>5s}  {'Flag':>8s}")
                lines.append(f"  {'----':>4s}  {'-------':>7s}  {'-------':>7s}  "
                             f"{'-------':>7s}  {'------':>6s}  {'-----':>5s}  {'--------':>8s}")
                for d_idx in sorted(decile_table.keys(), key=lambda x: int(x)):
                    d = decile_table[d_idx]
                    ratio = d.get("ratio", 1.0)
                    flag = ""
                    if ratio < CAL_RATIO_LOW_CRITICAL or ratio > CAL_RATIO_HIGH_CRITICAL:
                        flag = "CRITICAL"
                    elif ratio < CAL_RATIO_LOW_WARNING or ratio > CAL_RATIO_HIGH_WARNING:
                        flag = "WARNING"
                    ratio_str = f"{ratio:.4f}" if isinstance(ratio, (int, float)) else str(ratio)
                    lines.append(
                        f"  {int(d_idx):>4d}  {d['predicted_mean']:>7.4f}  "
                        f"{d['actual_rate']:>7.4f}  {ratio_str:>7s}  "
                        f"{d['n']:>6,}  {d['n_bad']:>5,}  {flag:>8s}"
                    )
            lines.append("")
        elif cal.get("error"):
            lines.append("--- Calibration Drift ---")
            lines.append(f"  Skipped: {cal['error']}")
            lines.append("")

        # --- Grade Shift ---
        gs = self.grade_shift
        if gs.get("current_dist"):
            lines.append("--- Grade Distribution Shift ---")
            ref_dist = gs.get("reference_dist", {})
            cur_dist = gs.get("current_dist", {})
            all_grades = sorted(set(list(ref_dist.keys()) + list(cur_dist.keys())))
            lines.append(f"  {'Grade':>6s}  {'Train':>7s}  {'Current':>7s}  {'Delta':>7s}")
            lines.append(f"  {'------':>6s}  {'-------':>7s}  {'-------':>7s}  {'-------':>7s}")
            for g in all_grades:
                ref_p = ref_dist.get(g, 0)
                cur_p = cur_dist.get(g, 0)
                delta = cur_p - ref_p
                lines.append(f"  {g:>6s}  {ref_p:>7.1%}  {cur_p:>7.1%}  {delta:>+7.1%}")

            grade_brs = gs.get("grade_bad_rates", {})
            if grade_brs:
                lines.append("")
                lines.append("  Grade bad rates (actual vs expected):")
                for g, info in sorted(grade_brs.items()):
                    actual = info.get("actual", 0)
                    expected = info.get("expected", 0)
                    ratio = info.get("ratio")
                    n = info.get("n", 0)
                    ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"
                    lines.append(
                        f"    {g}: actual={actual:.2%} expected={expected:.2%} "
                        f"ratio={ratio_str} (n={n:,})"
                    )
            lines.append("")

        # --- Operational Drift ---
        op = self.operational_drift
        if op:
            lines.append("--- Operational Drift (Circuit Breakers) ---")
            op_status = op.get("status", "N/A")
            lines.append(f"  Status: {op_status}")

            metrics = op.get("metrics", {})
            for metric_name in ["approve_rate", "review_rate", "decline_rate", "mean_pd"]:
                m = metrics.get(metric_name, {})
                if m:
                    value = m.get("value")
                    level = m.get("level", "OK")
                    expected = m.get("expected_range", [])
                    if value is not None:
                        if metric_name == "mean_pd":
                            val_str = f"{value:.4f}"
                            exp_str = f"[{expected[0]:.4f}, {expected[1]:.4f}]" if expected else "N/A"
                        else:
                            val_str = f"{value:.1%}"
                            exp_str = f"[{expected[0]:.1%}, {expected[1]:.1%}]" if expected else "N/A"
                        lines.append(f"  {metric_name:15s}: {val_str:>7s}  expected {exp_str}  [{level}]")
                    else:
                        lines.append(f"  {metric_name:15s}: N/A")

            op_alerts = op.get("alerts", [])
            if op_alerts:
                lines.append("  Alerts:")
                for a in op_alerts:
                    lines.append(f"    {a}")
            lines.append("")

        # --- Alerts ---
        lines.append("--- Alerts ---")
        if self.alerts:
            for a in self.alerts:
                lines.append(f"  {a}")
        else:
            lines.append("  No alerts. All metrics within acceptable bounds.")
        lines.append("")
        lines.append("=" * 72)

        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_json(self, path: Union[str, Path]) -> str:
        """Save the full report as a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.

        Returns
        -------
        str : absolute path the report was written to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "score_psi": self.score_psi,
            "feature_psi": self.feature_psi,
            "auc_decay": self.auc_decay,
            "calibration": self.calibration,
            "grade_shift": self.grade_shift,
            "operational_drift": self.operational_drift,
            "overall_status": self.overall_status,
            "alerts": self.alerts,
        }

        serializable = _make_serializable(data)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        logger.info(f"Report saved to {path}")
        return str(path.resolve())

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a plain dictionary."""
        return _make_serializable({
            "metadata": self.metadata,
            "score_psi": self.score_psi,
            "feature_psi": self.feature_psi,
            "auc_decay": self.auc_decay,
            "calibration": self.calibration,
            "grade_shift": self.grade_shift,
            "operational_drift": self.operational_drift,
            "overall_status": self.overall_status,
            "alerts": self.alerts,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _psi_level(psi_val: float) -> str:
        if psi_val > PSI_CRITICAL:
            return "CRITICAL"
        elif psi_val > PSI_WARNING:
            return "WARNING"
        else:
            return "STABLE"


# ---------------------------------------------------------------------------
# DriftMonitor — main monitoring class
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Monthly calibration and drift monitoring for the V5.2 pipeline.

    Computes training baselines at initialization, then compares any
    current dataset against those baselines to detect drift.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted Pipeline instance.
    train_scores : pd.DataFrame
        Pipeline.score_batch() output on training data.
    train_labels : np.ndarray
        Binary target (is_bad) for training data.
    train_features : pd.DataFrame
        Raw feature values from training data (for feature drift baselines).
    grade_labels : list of str
        Ordered grade labels from the CreditGrader.
    """

    def __init__(
        self,
        pipeline,
        train_scores: pd.DataFrame,
        train_labels: np.ndarray,
        train_features: pd.DataFrame,
        grade_labels: List[str],
    ):
        self.pipeline = pipeline
        self.grade_labels = grade_labels

        # --- Store training baselines ---
        self._ref_fraud_scores = train_scores["fraud_score"].values.copy()
        self._ref_pd = train_scores["pd"].values.copy()
        self._ref_grades = train_scores["grade"].values.copy()
        self._ref_labels = train_labels.copy()

        # Reference AUC
        try:
            self._ref_auc = float(roc_auc_score(train_labels, train_scores["pd"].values))
        except ValueError:
            self._ref_auc = None

        # Reference grade distribution
        self._ref_grade_dist = {}
        n_train = len(self._ref_grades)
        for g in grade_labels:
            mask = self._ref_grades == g
            self._ref_grade_dist[g] = round(int(mask.sum()) / max(n_train, 1), 4)

        # Reference grade bad rates
        self._ref_grade_bad_rates = {}
        for g in grade_labels:
            mask = self._ref_grades == g
            n = int(mask.sum())
            if n > 0:
                self._ref_grade_bad_rates[g] = round(float(train_labels[mask].mean()), 4)
            else:
                self._ref_grade_bad_rates[g] = 0.0

        # Reference calibration table
        self._ref_calibration = self._compute_calibration_table(
            train_scores["pd"].values, train_labels
        )

        # Reference feature distributions (for feature drift)
        self._ref_features = {}
        # Continuous features
        for feat in MONITORED_FEATURES_CONTINUOUS + MONITORED_BUREAU_FEATURES:
            if feat in train_features.columns:
                vals = train_features[feat].dropna().values.astype(float)
                if len(vals) > 0:
                    self._ref_features[feat] = {
                        "type": "continuous",
                        "values": vals.copy(),
                    }
        # Categorical features
        for feat in MONITORED_FEATURES_CATEGORICAL:
            if feat in train_features.columns:
                vals = train_features[feat].fillna("MISSING").astype(str).values
                categories = sorted(set(vals))
                self._ref_features[feat] = {
                    "type": "categorical",
                    "values": vals.copy(),
                    "categories": categories,
                }

        # Store training metadata
        self._n_train = n_train
        self._train_bad_rate = round(float(train_labels.mean()), 4)

        # Training stats dict — used by check_operational_drift() and also
        # settable externally for lightweight / test usage without a full pipeline.
        self.training_stats = {}

        logger.info(
            f"DriftMonitor initialized: {n_train} training samples, "
            f"ref_auc={self._ref_auc}, {len(self._ref_features)} features tracked"
        )

    # ------------------------------------------------------------------
    # Factory: from_pipeline
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        model_dir: Union[str, Path],
        data_path: Optional[Union[str, Path]] = None,
    ) -> "DriftMonitor":
        """Create a DriftMonitor by loading a trained pipeline and computing
        reference distributions from the training set (pre-2025-07-01).

        Parameters
        ----------
        model_dir : str or Path
            Directory containing pipeline artifacts (*.joblib files).
        data_path : str or Path, optional
            Path to master_features.parquet. Defaults to data/master_features.parquet.

        Returns
        -------
        DriftMonitor instance ready for .run() calls.
        """
        model_dir = Path(model_dir)
        project_root = model_dir

        if data_path is None:
            data_path = project_root / "master_features.parquet"
        data_path = Path(data_path)

        from pipeline import Pipeline

        # Load pipeline
        logger.info("Loading pipeline from %s", model_dir)
        pipeline = Pipeline.load(model_dir)

        # Load data
        logger.info("Loading data from %s", data_path)
        df = pd.read_parquet(data_path)

        # Prepare data (same logic as train_all.py / score_portfolio.py)
        if "loan_state" in df.columns:
            df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
        if "is_bad" not in df.columns:
            df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)
        df["signing_date"] = pd.to_datetime(df["signing_date"])

        if "shop_qi" in df.columns and "qi" not in df.columns:
            df["qi"] = df["shop_qi"]
        if "experian_FICO_SCORE" in df.columns and "fico" not in df.columns:
            df["fico"] = df["experian_FICO_SCORE"]

        # Join entity graph if available
        entity_path = project_root / "entity_graph_cross.parquet"
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

        # Split into training set
        train_df = df[df["signing_date"] < SPLIT_DATE].copy()
        y_train = train_df["is_bad"].astype(int).values

        logger.info("Scoring %d training samples for baselines...", len(train_df))
        train_scores = pipeline.score_batch(train_df)

        grade_labels = list(pipeline.credit_grader.grades)

        return cls(
            pipeline=pipeline,
            train_scores=train_scores,
            train_labels=y_train,
            train_features=train_df,
            grade_labels=grade_labels,
        )

    # ------------------------------------------------------------------
    # run() — main analysis
    # ------------------------------------------------------------------

    def run(
        self,
        current_data: pd.DataFrame,
        outcomes: Optional[np.ndarray] = None,
        period_label: Optional[str] = None,
    ) -> DriftReport:
        """Run drift monitoring on current data.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current period's data (raw features — will be scored through pipeline).
        outcomes : np.ndarray, optional
            Binary target (is_bad) for the current period.
            If None and 'is_bad' or 'loan_state' is in current_data, it will be derived.
        period_label : str, optional
            Label for this monitoring period. Defaults to current year-month.

        Returns
        -------
        DriftReport with all metrics computed.
        """
        if period_label is None:
            period_label = datetime.utcnow().strftime("%Y-%m")

        # Prepare current data
        current_df = current_data.copy()
        if "shop_qi" in current_df.columns and "qi" not in current_df.columns:
            current_df["qi"] = current_df["shop_qi"]
        if "experian_FICO_SCORE" in current_df.columns and "fico" not in current_df.columns:
            current_df["fico"] = current_df["experian_FICO_SCORE"]

        # Derive outcomes if possible
        has_outcomes = False
        y_true = outcomes
        if y_true is None:
            if "is_bad" in current_df.columns:
                y_true = current_df["is_bad"].astype(int).values
                has_outcomes = True
            elif "loan_state" in current_df.columns:
                y_true = current_df["loan_state"].isin(BAD_STATES).astype(int).values
                has_outcomes = True
        else:
            has_outcomes = True

        # Score current data through the pipeline
        logger.info("Scoring %d current applications...", len(current_df))
        current_scores = self.pipeline.score_batch(current_df)

        # Build metadata
        metadata = {
            "period": period_label,
            "timestamp": datetime.utcnow().isoformat(),
            "n_current": len(current_df),
            "n_training": self._n_train,
            "training_bad_rate": self._train_bad_rate,
            "reference_auc": round(self._ref_auc, 4) if self._ref_auc else None,
        }
        if has_outcomes:
            metadata["current_bad_rate"] = round(float(y_true.mean()), 4)
            metadata["current_n_bad"] = int(y_true.sum())

        # --- 1. Score PSI ---
        score_psi = self._compute_score_psi(current_scores)

        # --- 2. Feature PSI ---
        feature_psi = self._compute_feature_psi(current_df)

        # --- 3. AUC Decay ---
        auc_decay = self._compute_auc_decay(current_scores, y_true, has_outcomes)

        # --- 4. Calibration Drift ---
        calibration = self._compute_calibration_drift(current_scores, y_true, has_outcomes)

        # --- 5. Grade Shift ---
        grade_shift = self._compute_grade_shift(current_scores, y_true, has_outcomes)

        # --- 6. Operational Drift (circuit breakers) ---
        operational_drift = self.check_operational_drift(current_scores)

        # --- 7. Determine overall status and alerts ---
        overall_status, alerts = self._determine_status(
            score_psi, feature_psi, auc_decay, calibration, grade_shift,
            operational_drift=operational_drift,
        )

        return DriftReport(
            metadata=metadata,
            score_psi=score_psi,
            feature_psi=feature_psi,
            auc_decay=auc_decay,
            calibration=calibration,
            grade_shift=grade_shift,
            overall_status=overall_status,
            alerts=alerts,
            operational_drift=operational_drift,
        )

    # ------------------------------------------------------------------
    # run_from_parquet() — convenience for CLI
    # ------------------------------------------------------------------

    def run_from_parquet(
        self,
        data_path: Union[str, Path],
        period_label: Optional[str] = None,
        test_only: bool = False,
    ) -> DriftReport:
        """Load data from parquet and run drift monitoring.

        Parameters
        ----------
        data_path : str or Path
            Path to parquet file with raw features.
        period_label : str, optional
            Label for the monitoring period.
        test_only : bool
            If True, only analyze the test set (signing_date >= 2025-07-01).

        Returns
        -------
        DriftReport
        """
        data_path = Path(data_path)
        logger.info("Loading data from %s", data_path)
        df = pd.read_parquet(data_path)

        # Prepare data
        if "loan_state" in df.columns:
            df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
        if "is_bad" not in df.columns and "loan_state" in df.columns:
            df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)
        df["signing_date"] = pd.to_datetime(df["signing_date"])

        # Join entity graph if available
        project_root = Path(data_path).parent
        entity_path = project_root / "entity_graph_cross.parquet"
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

        if test_only:
            df = df[df["signing_date"] >= SPLIT_DATE].copy()
            logger.info("Filtered to test set: %d rows", len(df))

        return self.run(df, period_label=period_label)

    # ------------------------------------------------------------------
    # Operational Drift (circuit breaker monitoring)
    # ------------------------------------------------------------------

    def check_operational_drift(
        self,
        current_scores_df: pd.DataFrame,
        thresholds: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Check for operational metric drift that AUC monitoring would miss.

        Monitors approve rate, review rate, decline rate, and mean PD against
        expected ranges derived from V5.3 shadow scoring.  These catch failure
        modes like a QI outage that collapses approve rate from 54% to 0.7%
        while AUC stays stable.

        Parameters
        ----------
        current_scores_df : pd.DataFrame
            Must contain ``decision`` column (values: approve/review/decline)
            and ``pd`` column (predicted probability of default).
        thresholds : dict, optional
            Override default ``OPERATIONAL_THRESHOLDS``.

        Returns
        -------
        dict with keys:
            status : str  — "OK", "WARNING", or "CRITICAL"
            metrics : dict — per-metric value / level / expected range
            alerts : list[str] — human-readable alert messages
        """
        if thresholds is None:
            thresholds = OPERATIONAL_THRESHOLDS

        result: Dict[str, Any] = {
            "status": "OK",
            "metrics": {},
            "alerts": [],
        }

        alerts: List[str] = []
        status = "OK"

        def escalate(new_status: str):
            nonlocal status
            severity = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
            if severity.get(new_status, 0) > severity.get(status, 0):
                status = new_status

        # ----- Decision rate metrics -----
        if "decision" in current_scores_df.columns:
            decisions = current_scores_df["decision"].astype(str).str.lower()
            n_total = len(decisions)

            if n_total > 0:
                for decision_value, metric_name in [
                    ("approve", "approve_rate"),
                    ("review", "review_rate"),
                    ("decline", "decline_rate"),
                ]:
                    rate = float((decisions == decision_value).sum()) / n_total
                    thresh = thresholds.get(metric_name, {})
                    expected = thresh.get("expected_range", (0, 1))
                    warn_range = thresh.get("warning_range", (0, 1))
                    crit_range = thresh.get("critical_range", (0, 1))
                    critical_floor = thresh.get("critical_floor")

                    level = "OK"

                    # Special floor check (QI outage signature)
                    if critical_floor is not None and rate < critical_floor:
                        level = "CRITICAL"
                        alerts.append(
                            f"CRITICAL: {metric_name} = {rate:.1%} "
                            f"(below {critical_floor:.0%} floor — possible QI outage)"
                        )
                        escalate("CRITICAL")
                    elif rate < crit_range[0] or rate > crit_range[1]:
                        level = "CRITICAL"
                        alerts.append(
                            f"CRITICAL: {metric_name} = {rate:.1%} "
                            f"(outside critical range [{crit_range[0]:.1%}, {crit_range[1]:.1%}])"
                        )
                        escalate("CRITICAL")
                    elif rate < warn_range[0] or rate > warn_range[1]:
                        level = "WARNING"
                        alerts.append(
                            f"WARNING: {metric_name} = {rate:.1%} "
                            f"(outside warning range [{warn_range[0]:.1%}, {warn_range[1]:.1%}])"
                        )
                        escalate("WARNING")

                    result["metrics"][metric_name] = {
                        "value": round(rate, 4),
                        "level": level,
                        "expected_range": list(expected),
                    }
        else:
            result["metrics"]["decision_error"] = "No 'decision' column in data"

        # ----- Mean PD metric -----
        if "pd" in current_scores_df.columns:
            pd_values = current_scores_df["pd"].dropna().astype(float)
            if len(pd_values) > 0:
                mean_pd = float(pd_values.mean())
                thresh = thresholds.get("mean_pd", {})
                expected = thresh.get("expected_range", (0, 1))
                warn_range = thresh.get("warning_range", (0, 1))
                crit_range = thresh.get("critical_range", (0, 1))

                level = "OK"
                if mean_pd < crit_range[0] or mean_pd > crit_range[1]:
                    level = "CRITICAL"
                    alerts.append(
                        f"CRITICAL: mean_pd = {mean_pd:.4f} "
                        f"(outside critical range [{crit_range[0]:.4f}, {crit_range[1]:.4f}])"
                    )
                    escalate("CRITICAL")
                elif mean_pd < warn_range[0] or mean_pd > warn_range[1]:
                    level = "WARNING"
                    alerts.append(
                        f"WARNING: mean_pd = {mean_pd:.4f} "
                        f"(outside warning range [{warn_range[0]:.4f}, {warn_range[1]:.4f}])"
                    )
                    escalate("WARNING")

                result["metrics"]["mean_pd"] = {
                    "value": round(mean_pd, 4),
                    "level": level,
                    "expected_range": list(expected),
                }
        else:
            result["metrics"]["pd_error"] = "No 'pd' column in data"

        result["status"] = status
        result["alerts"] = alerts
        return result

    # ------------------------------------------------------------------
    # 1. Score PSI
    # ------------------------------------------------------------------

    def _compute_score_psi(self, current_scores: pd.DataFrame) -> Dict[str, Any]:
        """Compute PSI for fraud_score, pd, and grade distributions."""
        results = {}

        # Continuous: fraud_score
        if "fraud_score" in current_scores.columns:
            psi_result = compute_psi_continuous(
                self._ref_fraud_scores,
                current_scores["fraud_score"].values,
                n_bins=10,
            )
            results["fraud_score"] = psi_result
        else:
            results["fraud_score"] = {"psi": None, "error": "Column not found"}

        # Continuous: pd
        if "pd" in current_scores.columns:
            psi_result = compute_psi_continuous(
                self._ref_pd,
                current_scores["pd"].values,
                n_bins=10,
            )
            results["pd"] = psi_result
        else:
            results["pd"] = {"psi": None, "error": "Column not found"}

        # Categorical: grade
        if "grade" in current_scores.columns:
            psi_result = compute_psi_categorical(
                self._ref_grades,
                current_scores["grade"].values,
                categories=self.grade_labels,
            )
            results["grade"] = psi_result
        else:
            results["grade"] = {"psi": None, "error": "Column not found"}

        return results

    # ------------------------------------------------------------------
    # 2. Feature PSI
    # ------------------------------------------------------------------

    def _compute_feature_psi(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute PSI for top monitored input features."""
        results = {}

        for feat_name, ref_info in self._ref_features.items():
            feat_type = ref_info["type"]

            if feat_name not in current_df.columns:
                results[feat_name] = {"psi": None, "error": f"Column '{feat_name}' not in current data"}
                continue

            if feat_type == "continuous":
                cur_vals = current_df[feat_name].dropna().values.astype(float)
                if len(cur_vals) == 0:
                    results[feat_name] = {"psi": None, "error": "All values null in current data"}
                    continue
                psi_result = compute_psi_continuous(ref_info["values"], cur_vals, n_bins=10)
                results[feat_name] = psi_result

            elif feat_type == "categorical":
                cur_vals = current_df[feat_name].fillna("MISSING").astype(str).values
                psi_result = compute_psi_categorical(
                    ref_info["values"], cur_vals,
                    categories=ref_info.get("categories"),
                )
                results[feat_name] = psi_result

        return results

    # ------------------------------------------------------------------
    # 3. AUC Decay
    # ------------------------------------------------------------------

    def _compute_auc_decay(
        self,
        current_scores: pd.DataFrame,
        y_true: Optional[np.ndarray],
        has_outcomes: bool,
    ) -> Dict[str, Any]:
        """Compare current AUC to training reference."""
        result = {
            "current_auc": None,
            "reference_auc": round(self._ref_auc, 4) if self._ref_auc else None,
            "delta": None,
            "component_aucs": {},
        }

        if not has_outcomes or y_true is None:
            result["error"] = "No outcomes available for AUC computation"
            return result

        if len(np.unique(y_true)) < 2:
            result["error"] = "Only one class in outcomes (need both 0 and 1)"
            return result

        if "pd" not in current_scores.columns:
            result["error"] = "No 'pd' column in scores"
            return result

        try:
            current_auc = float(roc_auc_score(y_true, current_scores["pd"].values))
            result["current_auc"] = round(current_auc, 4)
            if self._ref_auc is not None:
                result["delta"] = round(current_auc - self._ref_auc, 4)
        except ValueError as e:
            result["error"] = str(e)
            return result

        # Component AUCs
        for comp in ["fraud_score", "woe_pd", "rule_pd", "xgb_pd"]:
            if comp in current_scores.columns:
                try:
                    result["component_aucs"][comp] = round(
                        float(roc_auc_score(y_true, current_scores[comp].values)), 4
                    )
                except ValueError:
                    result["component_aucs"][comp] = None

        return result

    # ------------------------------------------------------------------
    # 4. Calibration Drift
    # ------------------------------------------------------------------

    def _compute_calibration_drift(
        self,
        current_scores: pd.DataFrame,
        y_true: Optional[np.ndarray],
        has_outcomes: bool,
    ) -> Dict[str, Any]:
        """Compare predicted PD vs actual bad rate by decile."""
        result = {
            "brier": None,
            "mean_ratio": None,
            "worst_ratio": None,
            "worst_decile_idx": None,
            "n_warning_deciles": 0,
            "n_critical_deciles": 0,
            "decile_table": {},
            "reference_calibration": self._ref_calibration,
        }

        if not has_outcomes or y_true is None:
            result["error"] = "No outcomes available for calibration"
            return result

        if "pd" not in current_scores.columns:
            result["error"] = "No 'pd' column in scores"
            return result

        predicted = current_scores["pd"].values

        # Brier score
        try:
            result["brier"] = round(float(brier_score_loss(y_true, predicted)), 4)
        except ValueError as e:
            result["brier_error"] = str(e)

        # Calibration table
        cal_table = self._compute_calibration_table(predicted, y_true)
        result["decile_table"] = cal_table

        # Analyze each decile
        ratios = []
        worst_ratio = 1.0
        worst_idx = None
        n_warning = 0
        n_critical = 0

        for d_idx, d_stats in cal_table.items():
            pred_mean = d_stats["predicted_mean"]
            actual_rate = d_stats["actual_rate"]

            if pred_mean > 0.001:
                ratio = actual_rate / pred_mean
            elif actual_rate > 0:
                ratio = float("inf")
            else:
                ratio = 1.0

            d_stats["ratio"] = round(ratio, 4) if np.isfinite(ratio) else None
            ratios.append(ratio)

            # Track worst
            deviation = abs(ratio - 1.0)
            if deviation > abs(worst_ratio - 1.0):
                worst_ratio = ratio
                worst_idx = d_idx

            # Count alert deciles
            if (ratio < CAL_RATIO_LOW_CRITICAL or ratio > CAL_RATIO_HIGH_CRITICAL):
                n_critical += 1
            elif (ratio < CAL_RATIO_LOW_WARNING or ratio > CAL_RATIO_HIGH_WARNING):
                n_warning += 1

        finite_ratios = [r for r in ratios if np.isfinite(r)]
        result["mean_ratio"] = round(float(np.mean(finite_ratios)), 4) if finite_ratios else None
        result["worst_ratio"] = round(worst_ratio, 4) if worst_idx is not None and np.isfinite(worst_ratio) else None
        result["worst_decile_idx"] = worst_idx
        result["n_warning_deciles"] = n_warning
        result["n_critical_deciles"] = n_critical

        return result

    # ------------------------------------------------------------------
    # 5. Grade Shift
    # ------------------------------------------------------------------

    def _compute_grade_shift(
        self,
        current_scores: pd.DataFrame,
        y_true: Optional[np.ndarray],
        has_outcomes: bool,
    ) -> Dict[str, Any]:
        """Compare grade proportions and bad rates to training baseline."""
        result = {
            "current_dist": {},
            "reference_dist": dict(self._ref_grade_dist),
            "grade_bad_rates": {},
            "expected_bad_rates": dict(self._ref_grade_bad_rates),
        }

        if "grade" not in current_scores.columns:
            result["error"] = "No 'grade' column in scores"
            return result

        cur_grades = current_scores["grade"].values
        n_cur = len(cur_grades)

        # Current distribution
        for g in self.grade_labels:
            count = int(np.sum(cur_grades == g))
            result["current_dist"][g] = round(count / max(n_cur, 1), 4)

        # Grade bad rates (if outcomes available)
        if has_outcomes and y_true is not None:
            for g in self.grade_labels:
                mask = cur_grades == g
                n = int(mask.sum())
                if n > 0:
                    actual_rate = float(y_true[mask].mean())
                    expected_rate = self._ref_grade_bad_rates.get(g, 0)
                    ratio = actual_rate / max(expected_rate, 0.001)
                    result["grade_bad_rates"][g] = {
                        "actual": round(actual_rate, 4),
                        "expected": round(expected_rate, 4),
                        "ratio": round(ratio, 4),
                        "n": n,
                        "n_bad": int(y_true[mask].sum()),
                    }

        return result

    # ------------------------------------------------------------------
    # 6. Status determination
    # ------------------------------------------------------------------

    def _determine_status(
        self,
        score_psi: Dict[str, Any],
        feature_psi: Dict[str, Any],
        auc_decay: Dict[str, Any],
        calibration: Dict[str, Any],
        grade_shift: Dict[str, Any],
        operational_drift: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Determine overall status (GREEN/WARNING/CRITICAL) and compile alerts."""
        alerts = []
        status = "GREEN"

        def escalate(new_status):
            nonlocal status
            severity = {"GREEN": 0, "WARNING": 1, "CRITICAL": 2}
            if severity.get(new_status, 0) > severity.get(status, 0):
                status = new_status

        # --- Score PSI checks ---
        for score_name in ["fraud_score", "pd", "grade"]:
            psi_info = score_psi.get(score_name, {})
            psi_val = psi_info.get("psi")
            if psi_val is not None and isinstance(psi_val, (int, float)) and np.isfinite(psi_val):
                if psi_val > PSI_CRITICAL:
                    alerts.append(
                        f"CRITICAL: {score_name} PSI = {psi_val:.4f} "
                        f"(> {PSI_CRITICAL} threshold)"
                    )
                    escalate("CRITICAL")
                elif psi_val > PSI_WARNING:
                    alerts.append(
                        f"WARNING: {score_name} PSI = {psi_val:.4f} "
                        f"(> {PSI_WARNING} threshold)"
                    )
                    escalate("WARNING")

        # --- Feature PSI checks ---
        for feat_name, feat_info in feature_psi.items():
            psi_val = feat_info.get("psi")
            if psi_val is not None and isinstance(psi_val, (int, float)) and np.isfinite(psi_val):
                if psi_val > PSI_CRITICAL:
                    alerts.append(
                        f"CRITICAL: Feature '{feat_name}' PSI = {psi_val:.4f} "
                        f"(> {PSI_CRITICAL} threshold)"
                    )
                    escalate("CRITICAL")
                elif psi_val > PSI_WARNING:
                    alerts.append(
                        f"WARNING: Feature '{feat_name}' PSI = {psi_val:.4f} "
                        f"(> {PSI_WARNING} threshold)"
                    )
                    escalate("WARNING")

        # --- AUC decay checks ---
        delta = auc_decay.get("delta")
        if delta is not None and isinstance(delta, (int, float)):
            drop = -delta  # positive drop = degradation
            if drop >= AUC_DROP_CRITICAL:
                alerts.append(
                    f"CRITICAL: AUC dropped by {drop:.4f} "
                    f"(current={auc_decay.get('current_auc')}, "
                    f"reference={auc_decay.get('reference_auc')})"
                )
                escalate("CRITICAL")
            elif drop >= AUC_DROP_WARNING:
                alerts.append(
                    f"WARNING: AUC dropped by {drop:.4f} "
                    f"(current={auc_decay.get('current_auc')}, "
                    f"reference={auc_decay.get('reference_auc')})"
                )
                escalate("WARNING")

        # --- Calibration checks ---
        cal_worst = calibration.get("worst_ratio")
        if cal_worst is not None and isinstance(cal_worst, (int, float)):
            if cal_worst < CAL_RATIO_LOW_CRITICAL or cal_worst > CAL_RATIO_HIGH_CRITICAL:
                alerts.append(
                    f"CRITICAL: Calibration worst decile ratio = {cal_worst:.4f} "
                    f"(outside [{CAL_RATIO_LOW_CRITICAL}, {CAL_RATIO_HIGH_CRITICAL}])"
                )
                escalate("CRITICAL")
            elif cal_worst < CAL_RATIO_LOW_WARNING or cal_worst > CAL_RATIO_HIGH_WARNING:
                alerts.append(
                    f"WARNING: Calibration worst decile ratio = {cal_worst:.4f} "
                    f"(outside [{CAL_RATIO_LOW_WARNING}, {CAL_RATIO_HIGH_WARNING}])"
                )
                escalate("WARNING")

        n_cal_critical = calibration.get("n_critical_deciles", 0)
        n_cal_warning = calibration.get("n_warning_deciles", 0)
        if n_cal_critical > 0:
            alerts.append(
                f"CRITICAL: {n_cal_critical} decile(s) have calibration ratio "
                f"outside [{CAL_RATIO_LOW_CRITICAL}, {CAL_RATIO_HIGH_CRITICAL}]"
            )
            escalate("CRITICAL")
        elif n_cal_warning > 0:
            alerts.append(
                f"WARNING: {n_cal_warning} decile(s) have calibration ratio "
                f"outside [{CAL_RATIO_LOW_WARNING}, {CAL_RATIO_HIGH_WARNING}]"
            )
            escalate("WARNING")

        # --- Operational drift checks ---
        if operational_drift:
            op_status = operational_drift.get("status", "OK")
            op_alerts = operational_drift.get("alerts", [])
            alerts.extend(op_alerts)
            escalate(op_status)

        return status, alerts

    # ------------------------------------------------------------------
    # Calibration helper (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_calibration_table(
        predicted: np.ndarray, actual: np.ndarray, n_deciles: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Compute calibration table: predicted vs actual by decile."""
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=int)

        try:
            decile_labels = pd.qcut(predicted, q=n_deciles,
                                     labels=False, duplicates="drop")
        except ValueError:
            decile_labels = pd.cut(predicted, bins=n_deciles,
                                    labels=False, duplicates="drop")

        cal_table = {}
        for d in sorted(np.unique(decile_labels[~np.isnan(decile_labels)])):
            mask = decile_labels == d
            n = int(mask.sum())
            if n > 0:
                cal_table[str(int(d))] = {
                    "predicted_mean": round(float(predicted[mask].mean()), 4),
                    "actual_rate": round(float(actual[mask].mean()), 4),
                    "n": n,
                    "n_bad": int(actual[mask].sum()),
                }

        return cal_table

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        ref_auc_str = f"{self._ref_auc:.4f}" if self._ref_auc else "N/A"
        return (
            f"DriftMonitor(ref_auc={ref_auc_str}, "
            f"n_train={self._n_train}, "
            f"grades={self.grade_labels}, "
            f"features={list(self._ref_features.keys())})"
        )


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monthly drift monitoring for the V5.2 loan default pipeline."
    )
    parser.add_argument(
        "--data", "-d",
        default="data/master_features.parquet",
        help="Path to data parquet file (default: data/master_features.parquet)",
    )
    parser.add_argument(
        "--models", "-m",
        default="models/",
        help="Path to pipeline model directory (default: models/)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for JSON report (default: results/monitoring/)",
    )
    parser.add_argument(
        "--period",
        default=None,
        help="Period label for the report (default: current year-month)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only analyze the test set (signing_date >= 2025-07-01)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    models_path = Path(args.models)
    if not models_path.is_absolute():
        models_path = project_root / models_path

    # Initialize monitor
    monitor = DriftMonitor.from_pipeline(str(models_path), data_path=str(data_path))

    # Run analysis
    report = monitor.run_from_parquet(
        str(data_path),
        period_label=args.period,
        test_only=args.test_only,
    )

    # Print summary
    report.print_summary()

    # Save JSON report
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        output_dir = project_root / "results" / "monitoring"

    output_dir.mkdir(parents=True, exist_ok=True)
    period = args.period or datetime.utcnow().strftime("%Y_%m")
    json_path = output_dir / f"drift_report_{period}.json"
    saved_path = report.to_json(json_path)
    print(f"\n[DriftMonitor] Report saved to {saved_path}")


if __name__ == "__main__":
    main()
