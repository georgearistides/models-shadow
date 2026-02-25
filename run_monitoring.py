#!/usr/bin/env python3
"""
run_monitoring.py — Combined Monthly Portfolio Monitoring Report
================================================================

Runs both DriftMonitor and PaymentMonitor and produces a unified monthly
monitoring report (text + JSON). This is the single entry point for all
post-origination monitoring.

DriftMonitor checks:
  - Score PSI (fraud, PD, grade)
  - AUC decay detection
  - Calibration drift
  - Grade distribution drift
  - Feature drift (FICO, QI, partner, bureau)

PaymentMonitor checks:
  - NSF flag analysis
  - Tier distribution (Green/Yellow/Orange/Red)
  - Early warning alerts
  - Calibrated default probabilities

CLI usage:
  python3 scripts/models/run_monitoring.py
  python3 scripts/models/run_monitoring.py --scores results/shadow_scores_v5.csv
  python3 scripts/models/run_monitoring.py --json results/monitoring/monthly_report.json
  python3 scripts/models/run_monitoring.py --month 2025-09

Exit codes:
  0 = OK (GREEN)
  1 = WARNING
  2 = CRITICAL
"""

import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.models.model_utils import BAD_STATES, EXCLUDE_STATES, SPLIT_DATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STATUS_GREEN = "GREEN"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"

EXIT_OK = 0
EXIT_WARNING = 1
EXIT_CRITICAL = 2

STATUS_EXIT_MAP = {
    STATUS_GREEN: EXIT_OK,
    STATUS_WARNING: EXIT_WARNING,
    STATUS_CRITICAL: EXIT_CRITICAL,
}


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
# Data loading utilities
# ---------------------------------------------------------------------------

def _find_latest_shadow_scores(project_root: Path) -> Optional[Path]:
    """Find the latest shadow scores CSV by modification time."""
    results_dir = project_root / "results"
    if not results_dir.exists():
        return None

    candidates = sorted(
        results_dir.glob("shadow_scores*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_shadow_scores(path: Path) -> pd.DataFrame:
    """Load shadow scores CSV and normalize column names."""
    df = pd.read_csv(path, parse_dates=["signing_date"])

    # Normalize 'our_*' columns to match pipeline output names
    rename_map = {}
    for col in df.columns:
        if col.startswith("our_"):
            rename_map[col] = col[4:]  # strip 'our_' prefix
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure is_bad exists
    if "is_bad" not in df.columns and "loan_state" in df.columns:
        df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)

    return df


def _load_payment_data(project_root: Path) -> Optional[pd.DataFrame]:
    """Load payment features if available."""
    payment_path = project_root / "data" / "payment_features.parquet"
    if payment_path.exists():
        try:
            return pd.read_parquet(payment_path)
        except Exception as e:
            logger.warning("Failed to load payment features: %s", e)
    return None


def _load_pipeline(model_dir: Path):
    """Load the pipeline from model directory. Returns None on failure."""
    try:
        from scripts.models.pipeline import Pipeline
        pipeline = Pipeline.load(model_dir)
        return pipeline
    except Exception as e:
        logger.warning("Failed to load pipeline from %s: %s", model_dir, e)
        return None


def _load_payment_monitor(model_dir: Path):
    """Load PaymentMonitor from model directory. Returns None on failure."""
    monitor_path = model_dir / "payment_monitor.joblib"
    if not monitor_path.exists():
        logger.info("No payment_monitor.joblib found in %s", model_dir)
        return None
    try:
        from scripts.models.payment_monitor import PaymentMonitor
        return PaymentMonitor.load(monitor_path)
    except Exception as e:
        logger.warning("Failed to load PaymentMonitor: %s", e)
        return None


# ---------------------------------------------------------------------------
# DriftMonitor section runner
# ---------------------------------------------------------------------------

def _run_drift_checks(
    scores_df: pd.DataFrame,
    pipeline,
    project_root: Path,
    month_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Run DriftMonitor checks and return structured results.

    Returns a dict with keys: score_psi, feature_psi, auc_decay,
    calibration, grade_shift, overall_status, alerts, metadata.
    """
    try:
        from scripts.models.drift_monitor import DriftMonitor
    except ImportError as e:
        return {
            "error": f"Could not import DriftMonitor: {e}",
            "overall_status": STATUS_WARNING,
            "alerts": [f"DriftMonitor import failed: {e}"],
        }

    # Load training data for reference baselines
    data_path = project_root / "data" / "master_features.parquet"
    if not data_path.exists():
        return {
            "error": f"Training data not found at {data_path}",
            "overall_status": STATUS_WARNING,
            "alerts": ["Training data not found -- cannot compute drift baselines"],
        }

    try:
        model_dir = project_root / "models"
        monitor = DriftMonitor.from_pipeline(str(model_dir), data_path=str(data_path))
    except Exception as e:
        return {
            "error": f"Failed to initialize DriftMonitor: {e}",
            "overall_status": STATUS_WARNING,
            "alerts": [f"DriftMonitor init failed: {e}"],
        }

    # Prepare current data for drift analysis
    current_df = pd.read_parquet(data_path)
    if "loan_state" in current_df.columns:
        current_df = current_df[~current_df["loan_state"].isin(EXCLUDE_STATES)].copy()
    if "is_bad" not in current_df.columns and "loan_state" in current_df.columns:
        current_df["is_bad"] = current_df["loan_state"].isin(BAD_STATES).astype(int)
    current_df["signing_date"] = pd.to_datetime(current_df["signing_date"])

    # Join entity graph if available
    entity_path = project_root / "data" / "entity_graph_cross.parquet"
    if entity_path.exists():
        entity_df = pd.read_parquet(entity_path)
        entity_cols = ['application_id', 'has_prior_bad', 'is_repeat', 'is_cross_entity',
                       'prior_loans_ssn', 'prior_shops_ssn', 'prior_bad_ssn', 'prior_paid_ssn']
        entity_feats = entity_df[[c for c in entity_cols if c in entity_df.columns]].copy()
        if "has_prior_bad" not in current_df.columns:
            current_df = current_df.merge(entity_feats, on='application_id', how='left')
            for col in entity_feats.columns:
                if col != 'application_id' and col in current_df.columns:
                    current_df[col] = current_df[col].fillna(0).astype(int)

    # Apply month filter: use test data (post-split) if no specific month
    if month_filter:
        current_df["signing_month"] = current_df["signing_date"].dt.strftime("%Y-%m")
        current_df = current_df[current_df["signing_month"] == month_filter].copy()
        period_label = month_filter
    else:
        # Default to test set
        current_df = current_df[current_df["signing_date"] >= SPLIT_DATE].copy()
        period_label = "test-set"

    if len(current_df) == 0:
        return {
            "error": "No data available after filtering",
            "overall_status": STATUS_WARNING,
            "alerts": ["No data available for drift analysis after filtering"],
        }

    try:
        report = monitor.run(current_df, period_label=period_label)
        return report.to_dict()
    except Exception as e:
        return {
            "error": f"DriftMonitor.run() failed: {e}",
            "overall_status": STATUS_WARNING,
            "alerts": [f"Drift analysis failed: {e}"],
        }


# ---------------------------------------------------------------------------
# PaymentMonitor section runner
# ---------------------------------------------------------------------------

def _run_payment_checks(
    payment_df: Optional[pd.DataFrame],
    payment_monitor,
    month_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Run PaymentMonitor checks and return structured results.

    Returns a dict with keys: tier_distribution, high_risk_alerts,
    early_warning, evaluation, overall_status, alerts.
    """
    result = {
        "available": False,
        "tier_distribution": {},
        "high_risk_alerts": 0,
        "early_warning": 0,
        "evaluation": {},
        "overall_status": STATUS_GREEN,
        "alerts": [],
    }

    if payment_df is None:
        result["error"] = "No payment data available"
        return result

    if payment_monitor is None:
        result["error"] = "PaymentMonitor not loaded (no payment_monitor.joblib)"
        return result

    # Apply month filter
    df = payment_df.copy()
    if month_filter and "signing_date" in df.columns:
        df["signing_date"] = pd.to_datetime(df["signing_date"])
        df["signing_month"] = df["signing_date"].dt.strftime("%Y-%m")
        df = df[df["signing_month"] == month_filter].copy()

    if len(df) == 0:
        result["error"] = "No payment data after filtering"
        return result

    result["available"] = True
    result["n_loans"] = len(df)

    # Check how many loans have customer payment data
    has_payment_data = False
    for col in ["nsf_count", "cust_payments", "total_payments"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 0:
                has_payment_data = True
                break

    if not has_payment_data:
        result["error"] = "No non-null payment features found"
        return result

    # Track loans with zero customer payments (67% expected)
    if "cust_payments" in df.columns:
        zero_cust = (df["cust_payments"].fillna(0) == 0).sum()
        result["zero_customer_payments"] = int(zero_cust)
        result["pct_zero_customer_payments"] = round(float(zero_cust / len(df)), 4)

    try:
        # Run batch predictions
        pm_results = payment_monitor.predict_batch(df)

        # Tier distribution
        tier_dist = {}
        for tier in ["green", "yellow", "orange", "red"]:
            mask = pm_results["risk_tier"] == tier
            count = int(mask.sum())
            pct = round(float(count / len(df)), 4) if len(df) > 0 else 0.0
            tier_dist[tier] = {"count": count, "pct": pct}
        result["tier_distribution"] = tier_dist

        # High-risk alerts (red tier)
        red_count = tier_dist.get("red", {}).get("count", 0)
        result["high_risk_alerts"] = red_count

        # Early warning: loans in yellow tier
        yellow_count = tier_dist.get("yellow", {}).get("count", 0)
        result["early_warning"] = yellow_count

        # Orange tier count
        orange_count = tier_dist.get("orange", {}).get("count", 0)

        # Average risk metrics
        result["avg_flag_count"] = round(float(pm_results["flag_count"].mean()), 2)
        result["avg_default_probability"] = round(
            float(pm_results["default_probability"].mean()), 4
        )

        # Run evaluation if labels available
        if "is_bad" in df.columns and df["is_bad"].nunique() > 1:
            try:
                eval_metrics = payment_monitor.evaluate(df, label_col="is_bad")
                result["evaluation"] = {
                    "auc": eval_metrics.get("auc"),
                    "auc_calibrated": eval_metrics.get("auc_calibrated"),
                    "gini": eval_metrics.get("gini"),
                    "brier": eval_metrics.get("brier"),
                    "tier_stats": eval_metrics.get("tier_stats"),
                }
            except Exception as e:
                result["evaluation_error"] = str(e)

        # Generate alerts based on thresholds
        alerts = []
        overall = STATUS_GREEN

        # Red tier > 5% is a warning, > 10% is critical
        red_pct = tier_dist.get("red", {}).get("pct", 0)
        if red_pct > 0.10:
            alerts.append(
                f"CRITICAL: Red tier at {red_pct:.1%} "
                f"({red_count} loans) -- exceeds 10% threshold"
            )
            overall = STATUS_CRITICAL
        elif red_pct > 0.05:
            alerts.append(
                f"WARNING: Red tier at {red_pct:.1%} "
                f"({red_count} loans) -- exceeds 5% threshold"
            )
            overall = STATUS_WARNING

        # Orange+Red > 15% is a warning
        orange_red_pct = (orange_count + red_count) / max(len(df), 1)
        if orange_red_pct > 0.15:
            alerts.append(
                f"WARNING: Orange+Red tiers at {orange_red_pct:.1%} "
                f"({orange_count + red_count} loans)"
            )
            if overall != STATUS_CRITICAL:
                overall = STATUS_WARNING

        result["overall_status"] = overall
        result["alerts"] = alerts

    except Exception as e:
        result["error"] = f"PaymentMonitor prediction failed: {e}"
        result["alerts"] = [f"Payment monitoring failed: {e}"]
        result["overall_status"] = STATUS_WARNING

    return result


# ---------------------------------------------------------------------------
# Score-based drift checks (uses pre-computed shadow scores)
# ---------------------------------------------------------------------------

def _run_score_drift_from_csv(
    scores_df: pd.DataFrame,
    month_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute basic score drift metrics from shadow scores CSV.

    This is a lightweight alternative to full DriftMonitor when we only
    have pre-computed scores (not raw features + pipeline).
    """
    from scripts.models.model_monitor import compute_psi_continuous, compute_psi_categorical

    result = {
        "score_psi": {},
        "auc": {},
        "calibration": {},
        "grade_distribution": {},
        "overall_status": STATUS_GREEN,
        "alerts": [],
    }

    df = scores_df.copy()
    df["signing_date"] = pd.to_datetime(df["signing_date"])

    # Split into reference (train) and current (test)
    ref_df = df[df["signing_date"] < SPLIT_DATE].copy()
    cur_df = df[df["signing_date"] >= SPLIT_DATE].copy()

    # Apply month filter to current if specified
    if month_filter:
        cur_df["signing_month"] = cur_df["signing_date"].dt.strftime("%Y-%m")
        cur_df = cur_df[cur_df["signing_month"] == month_filter].copy()

    if len(ref_df) == 0 or len(cur_df) == 0:
        result["error"] = (
            f"Insufficient data: {len(ref_df)} reference, {len(cur_df)} current"
        )
        return result

    result["n_reference"] = len(ref_df)
    result["n_current"] = len(cur_df)

    alerts = []
    overall = STATUS_GREEN

    def escalate(new_status):
        nonlocal overall
        severity = {STATUS_GREEN: 0, STATUS_WARNING: 1, STATUS_CRITICAL: 2}
        if severity.get(new_status, 0) > severity.get(overall, 0):
            overall = new_status

    # --- Score PSI ---
    for score_col, display_name in [
        ("fraud_score", "Fraud Score"),
        ("pd", "PD Score"),
    ]:
        if score_col in ref_df.columns and score_col in cur_df.columns:
            ref_vals = ref_df[score_col].dropna().values.astype(float)
            cur_vals = cur_df[score_col].dropna().values.astype(float)
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                try:
                    psi_info = compute_psi_continuous(ref_vals, cur_vals, n_bins=10)
                    psi_val = psi_info.get("psi", 0)
                    level = "STABLE"
                    if psi_val > 0.25:
                        level = "CRITICAL"
                        alerts.append(f"CRITICAL: {display_name} PSI = {psi_val:.4f}")
                        escalate(STATUS_CRITICAL)
                    elif psi_val > 0.10:
                        level = "WARNING"
                        alerts.append(f"WARNING: {display_name} PSI = {psi_val:.4f}")
                        escalate(STATUS_WARNING)
                    result["score_psi"][score_col] = {
                        "psi": round(psi_val, 4),
                        "level": level,
                    }
                except Exception as e:
                    result["score_psi"][score_col] = {"error": str(e)}

    # Grade PSI (categorical)
    if "grade" in ref_df.columns and "grade" in cur_df.columns:
        ref_grades = ref_df["grade"].dropna().astype(str).values
        cur_grades = cur_df["grade"].dropna().astype(str).values
        all_grades = sorted(set(list(ref_grades) + list(cur_grades)))
        try:
            psi_info = compute_psi_categorical(ref_grades, cur_grades, categories=all_grades)
            psi_val = psi_info.get("psi", 0)
            level = "STABLE"
            if psi_val > 0.25:
                level = "CRITICAL"
                alerts.append(f"CRITICAL: Grade PSI = {psi_val:.4f}")
                escalate(STATUS_CRITICAL)
            elif psi_val > 0.10:
                level = "WARNING"
                alerts.append(f"WARNING: Grade PSI = {psi_val:.4f}")
                escalate(STATUS_WARNING)
            result["score_psi"]["grade"] = {"psi": round(psi_val, 4), "level": level}
        except Exception as e:
            result["score_psi"]["grade"] = {"error": str(e)}

    # --- AUC Performance ---
    if "is_bad" in df.columns and "pd" in df.columns:
        from sklearn.metrics import roc_auc_score

        # Reference AUC
        ref_y = ref_df["is_bad"].astype(int).values
        ref_pd = ref_df["pd"].astype(float).values
        try:
            ref_auc = float(roc_auc_score(ref_y, ref_pd))
        except ValueError:
            ref_auc = None

        # Current AUC
        cur_y = cur_df["is_bad"].astype(int).values
        cur_pd = cur_df["pd"].astype(float).values
        try:
            cur_auc = float(roc_auc_score(cur_y, cur_pd))
        except ValueError:
            cur_auc = None

        result["auc"]["reference"] = round(ref_auc, 4) if ref_auc else None
        result["auc"]["current"] = round(cur_auc, 4) if cur_auc else None

        if ref_auc and cur_auc:
            delta = cur_auc - ref_auc
            result["auc"]["delta"] = round(delta, 4)
            drop = -delta
            if drop >= 0.05:
                alerts.append(
                    f"CRITICAL: AUC dropped by {drop:.4f} "
                    f"(ref={ref_auc:.4f}, current={cur_auc:.4f})"
                )
                escalate(STATUS_CRITICAL)
            elif drop >= 0.02:
                alerts.append(
                    f"WARNING: AUC dropped by {drop:.4f} "
                    f"(ref={ref_auc:.4f}, current={cur_auc:.4f})"
                )
                escalate(STATUS_WARNING)

        # Fraud score AUC
        if "fraud_score" in df.columns:
            try:
                fraud_ref_auc = float(roc_auc_score(ref_y, ref_df["fraud_score"].values))
                fraud_cur_auc = float(roc_auc_score(cur_y, cur_df["fraud_score"].values))
                result["auc"]["fraud_reference"] = round(fraud_ref_auc, 4)
                result["auc"]["fraud_current"] = round(fraud_cur_auc, 4)
                result["auc"]["fraud_delta"] = round(fraud_cur_auc - fraud_ref_auc, 4)

                fraud_drop = -(fraud_cur_auc - fraud_ref_auc)
                if fraud_drop >= 0.05:
                    alerts.append(
                        f"CRITICAL: FraudGate AUC dropped by {fraud_drop:.4f}"
                    )
                    escalate(STATUS_CRITICAL)
                elif fraud_drop >= 0.02:
                    alerts.append(
                        f"WARNING: FraudGate AUC dropped by {fraud_drop:.4f}"
                    )
                    escalate(STATUS_WARNING)
            except (ValueError, TypeError):
                pass

    # --- Calibration ---
    if "is_bad" in cur_df.columns and "pd" in cur_df.columns:
        cur_y = cur_df["is_bad"].astype(int).values
        cur_pd = cur_df["pd"].astype(float).values

        pred_mean = float(np.mean(cur_pd))
        actual_mean = float(np.mean(cur_y))
        cal_ratio = pred_mean / max(actual_mean, 0.001) if actual_mean > 0 else None

        result["calibration"]["predicted_mean"] = round(pred_mean, 4)
        result["calibration"]["actual_mean"] = round(actual_mean, 4)
        result["calibration"]["ratio"] = round(cal_ratio, 4) if cal_ratio else None

        if cal_ratio is not None:
            if cal_ratio < 0.6 or cal_ratio > 1.5:
                alerts.append(
                    f"CRITICAL: Calibration ratio = {cal_ratio:.4f} "
                    f"(pred={pred_mean:.4f}, actual={actual_mean:.4f})"
                )
                escalate(STATUS_CRITICAL)
            elif cal_ratio < 0.8 or cal_ratio > 1.2:
                alerts.append(
                    f"WARNING: Calibration ratio = {cal_ratio:.4f} "
                    f"(pred={pred_mean:.4f}, actual={actual_mean:.4f})"
                )
                escalate(STATUS_WARNING)

        # Per-grade calibration
        if "grade" in cur_df.columns:
            grade_cal = {}
            for g in sorted(cur_df["grade"].dropna().unique()):
                g_mask = cur_df["grade"] == g
                n = int(g_mask.sum())
                if n > 0:
                    g_pred = float(cur_df.loc[g_mask, "pd"].astype(float).mean())
                    g_actual = float(cur_df.loc[g_mask, "is_bad"].astype(int).mean())
                    grade_cal[g] = {
                        "predicted": round(g_pred, 4),
                        "actual": round(g_actual, 4),
                        "n": n,
                    }
            result["calibration"]["per_grade"] = grade_cal

    # --- Grade Distribution ---
    if "grade" in cur_df.columns:
        grade_dist = cur_df["grade"].value_counts(normalize=True).round(4).to_dict()
        result["grade_distribution"]["current"] = grade_dist

        if "grade" in ref_df.columns:
            ref_grade_dist = ref_df["grade"].value_counts(normalize=True).round(4).to_dict()
            result["grade_distribution"]["reference"] = ref_grade_dist

    result["overall_status"] = overall
    result["alerts"] = alerts

    # --- Operational Drift (circuit breakers) ---
    # Uses the same DriftMonitor.check_operational_drift() logic
    try:
        from scripts.models.drift_monitor import DriftMonitor, OPERATIONAL_THRESHOLDS

        # Create a lightweight DriftMonitor just for operational checks.
        # We don't need a full pipeline — just the check_operational_drift method.
        # Use __new__ to bypass __init__ which requires a pipeline.
        _dummy = object.__new__(DriftMonitor)
        _dummy.training_stats = {}

        op_result = _dummy.check_operational_drift(cur_df)
        result["operational_drift"] = op_result

        # Merge operational alerts into main alerts
        op_alerts = op_result.get("alerts", [])
        result["alerts"].extend(op_alerts)
        op_status = op_result.get("status", "OK")
        op_sev = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
        if op_sev.get(op_status, 0) > op_sev.get(result["overall_status"], 0):
            result["overall_status"] = {"OK": STATUS_GREEN, "WARNING": STATUS_WARNING, "CRITICAL": STATUS_CRITICAL}.get(op_status, STATUS_GREEN)

    except Exception as e:
        result["operational_drift"] = {"error": str(e), "status": "OK", "metrics": {}, "alerts": []}
        logger.warning("Operational drift check failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Combined report formatting
# ---------------------------------------------------------------------------

def _format_text_report(
    drift_results: Dict[str, Any],
    payment_results: Dict[str, Any],
    period: str,
    timestamp: str,
    scores_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Format the combined monitoring report as human-readable text.

    Returns (text_report, overall_status).
    """
    lines = []
    action_items = []

    # Determine combined status
    drift_status = drift_results.get("overall_status", STATUS_GREEN)
    payment_status = payment_results.get("overall_status", STATUS_GREEN)
    severity = {STATUS_GREEN: 0, STATUS_WARNING: 1, STATUS_CRITICAL: 2}
    overall_status = drift_status if severity.get(drift_status, 0) >= severity.get(payment_status, 0) else payment_status

    # Count alerts
    drift_alerts = drift_results.get("alerts", [])
    payment_alerts = payment_results.get("alerts", [])
    n_warnings = sum(1 for a in drift_alerts + payment_alerts if "WARNING" in a)
    n_criticals = sum(1 for a in drift_alerts + payment_alerts if "CRITICAL" in a)

    # === Header ===
    lines.append("=== Monthly Portfolio Monitoring Report ===")
    lines.append(f"Period: {period}")
    lines.append(f"Generated: {timestamp}")
    if scores_path:
        lines.append(f"Source: {scores_path}")
    lines.append("")

    # === Score Drift ===
    lines.append("--- Score Drift ---")
    score_psi = drift_results.get("score_psi", {})

    for score_name, display in [("fraud_score", "Fraud Score PSI"),
                                 ("pd", "PD Score PSI"),
                                 ("grade", "Grade Drift PSI")]:
        info = score_psi.get(score_name, {})
        psi_val = info.get("psi")
        if psi_val is not None:
            level = info.get("level", "OK")
            lines.append(f"{display}: {psi_val:.4f} [{level}]")
        elif "error" in info:
            lines.append(f"{display}: N/A ({info['error']})")
        else:
            lines.append(f"{display}: N/A")

    lines.append(f"Overall: {drift_status}")
    lines.append("")

    # === AUC Performance ===
    lines.append("--- AUC Performance ---")
    auc_info = drift_results.get("auc", {})

    if auc_info.get("fraud_current") is not None:
        fraud_auc = auc_info["fraud_current"]
        fraud_delta = auc_info.get("fraud_delta", 0)
        fraud_level = "OK"
        if abs(fraud_delta) >= 0.05:
            fraud_level = "CRITICAL"
        elif abs(fraud_delta) >= 0.02:
            fraud_level = "WARNING"

        if fraud_delta < 0:
            lines.append(f"FraudGate AUC: {fraud_auc:.4f} [{fraud_level}: {fraud_delta:+.4f} from baseline]")
            if fraud_level != "OK":
                action_items.append(f"Monitor FraudGate AUC decay ({fraud_delta:+.4f} from baseline)")
        else:
            lines.append(f"FraudGate AUC: {fraud_auc:.4f} [{fraud_level}]")
    else:
        lines.append("FraudGate AUC: N/A")

    if auc_info.get("current") is not None:
        default_auc = auc_info["current"]
        default_delta = auc_info.get("delta", 0)
        default_level = "OK"
        if abs(default_delta) >= 0.05:
            default_level = "CRITICAL"
        elif abs(default_delta) >= 0.02:
            default_level = "WARNING"

        if default_delta < 0:
            lines.append(f"DefaultScorecard AUC: {default_auc:.4f} [{default_level}: {default_delta:+.4f} from baseline]")
            if default_level != "OK":
                action_items.append(f"Monitor DefaultScorecard AUC decay ({default_delta:+.4f} from baseline)")
        else:
            lines.append(f"DefaultScorecard AUC: {default_auc:.4f} [{default_level}]")
    else:
        lines.append("DefaultScorecard AUC: N/A")
    lines.append("")

    # === Calibration ===
    lines.append("--- Calibration ---")
    cal = drift_results.get("calibration", {})
    pred_mean = cal.get("predicted_mean")
    actual_mean = cal.get("actual_mean")
    cal_ratio = cal.get("ratio")

    if pred_mean is not None and actual_mean is not None and cal_ratio is not None:
        cal_level = "OK"
        if cal_ratio < 0.6 or cal_ratio > 1.5:
            cal_level = "CRITICAL"
        elif cal_ratio < 0.8 or cal_ratio > 1.2:
            cal_level = "WARNING"
        lines.append(
            f"PD Mean Predicted: {pred_mean:.1%} | "
            f"Actual: {actual_mean:.1%} | "
            f"Ratio: {cal_ratio:.2f} [{cal_level}]"
        )
        if cal_level != "OK":
            action_items.append(f"Review calibration drift (ratio={cal_ratio:.2f})")
    else:
        lines.append("PD Calibration: N/A (no outcomes)")

    # Per-grade calibration
    per_grade = cal.get("per_grade", {})
    if per_grade:
        for grade, ginfo in sorted(per_grade.items()):
            g_pred = ginfo.get("predicted", 0)
            g_actual = ginfo.get("actual", 0)
            g_level = "OK"
            if g_pred > 0.001:
                g_ratio = g_actual / g_pred
                if g_ratio < 0.6 or g_ratio > 1.5:
                    g_level = "CRITICAL"
                elif g_ratio < 0.8 or g_ratio > 1.2:
                    g_level = "WARNING"
            lines.append(f"Grade {grade}: pred {g_pred:.1%} | actual {g_actual:.1%} [{g_level}]")
    lines.append("")

    # === Payment Health ===
    lines.append("--- Payment Health ---")
    if payment_results.get("available"):
        tier_dist = payment_results.get("tier_distribution", {})
        green_pct = tier_dist.get("green", {}).get("pct", 0)
        yellow_pct = tier_dist.get("yellow", {}).get("pct", 0)
        orange_pct = tier_dist.get("orange", {}).get("pct", 0)
        red_pct = tier_dist.get("red", {}).get("pct", 0)

        lines.append(
            f"Green: {green_pct:.1%} | "
            f"Yellow: {yellow_pct:.1%} | "
            f"Orange: {orange_pct:.1%} | "
            f"Red: {red_pct:.1%}"
        )

        red_count = payment_results.get("high_risk_alerts", 0)
        yellow_count = payment_results.get("early_warning", 0)
        lines.append(f"High-Risk Alerts: {red_count} loans in Red tier")
        lines.append(f"Early Warning: {yellow_count} loans in Yellow tier")

        if payment_results.get("zero_customer_payments") is not None:
            zero_pct = payment_results.get("pct_zero_customer_payments", 0)
            lines.append(f"Zero Customer Payments: {zero_pct:.1%} of portfolio")

        # Payment AUC if available
        eval_info = payment_results.get("evaluation", {})
        if eval_info.get("auc"):
            lines.append(f"Payment Monitor AUC: {eval_info['auc']:.4f}")

        if red_count > 0:
            action_items.append(
                f"Review {red_count} loans in Red tier -- initiate early intervention"
            )
    elif payment_results.get("error"):
        lines.append(f"Unavailable: {payment_results['error']}")
    else:
        lines.append("Unavailable: No payment data")
    lines.append("")

    # === Operational Drift ===
    op_drift = drift_results.get("operational_drift", {})
    if op_drift and op_drift.get("metrics"):
        lines.append("--- Operational Drift (Circuit Breakers) ---")
        op_status = op_drift.get("status", "OK")
        lines.append(f"Status: {op_status}")

        metrics = op_drift.get("metrics", {})
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
                    lines.append(f"{metric_name:15s}: {val_str:>7s}  expected {exp_str}  [{level}]")

        op_alerts = op_drift.get("alerts", [])
        if op_alerts:
            for a in op_alerts:
                lines.append(f"  {a}")
            # Add operational alerts as action items
            for a in op_alerts:
                if "CRITICAL" in a:
                    action_items.append(a)
                elif "WARNING" in a:
                    action_items.append(a)
        lines.append("")

        # Factor operational status into overall
        op_sev = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
        op_status_mapped = {"OK": STATUS_GREEN, "WARNING": STATUS_WARNING, "CRITICAL": STATUS_CRITICAL}.get(op_status, STATUS_GREEN)
        overall_sev = {STATUS_GREEN: 0, STATUS_WARNING: 1, STATUS_CRITICAL: 2}
        if overall_sev.get(op_status_mapped, 0) > overall_sev.get(overall_status, 0):
            overall_status = op_status_mapped

    # === Summary ===
    lines.append("--- Summary ---")
    alert_summary_parts = []
    if n_criticals > 0:
        alert_summary_parts.append(f"{n_criticals} CRITICAL")
    if n_warnings > 0:
        alert_summary_parts.append(f"{n_warnings} WARNING")
    alert_suffix = f" ({', '.join(alert_summary_parts)})" if alert_summary_parts else ""
    lines.append(f"Status: {overall_status}{alert_suffix}")

    if action_items:
        lines.append("Action Items:")
        for item in action_items:
            lines.append(f"- {item}")
    else:
        lines.append("Action Items: None -- all metrics within acceptable bounds")

    lines.append("")
    return "\n".join(lines), overall_status


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_monitoring(
    scores_path: Optional[str] = None,
    json_path: Optional[str] = None,
    month: Optional[str] = None,
    model_dir: Optional[str] = None,
    verbose: bool = False,
) -> int:
    """Run combined monitoring and return exit code.

    Parameters
    ----------
    scores_path : str, optional
        Path to shadow scores CSV. Auto-detected if not provided.
    json_path : str, optional
        Path to write JSON output.
    month : str, optional
        Filter to specific month (YYYY-MM).
    model_dir : str, optional
        Path to model directory. Defaults to models/.
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    int : exit code (0=OK, 1=WARNING, 2=CRITICAL)
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

    project_root = _PROJECT_ROOT
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    period = month or datetime.now().strftime("%Y-%m")

    # --- Resolve paths ---
    if model_dir:
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    else:
        model_path = project_root / "models"

    # --- Load shadow scores ---
    scores_df = None
    resolved_scores_path = None
    if scores_path:
        sp = Path(scores_path)
        if not sp.is_absolute():
            sp = project_root / sp
        if sp.exists():
            print(f"[Monitoring] Loading scores from {sp}")
            scores_df = _load_shadow_scores(sp)
            resolved_scores_path = str(sp)
        else:
            print(f"[Monitoring] WARNING: Scores file not found: {sp}")
    else:
        sp = _find_latest_shadow_scores(project_root)
        if sp:
            print(f"[Monitoring] Auto-detected latest scores: {sp}")
            scores_df = _load_shadow_scores(sp)
            resolved_scores_path = str(sp)
        else:
            print("[Monitoring] WARNING: No shadow scores found in results/")

    # --- Load payment data ---
    print("[Monitoring] Loading payment data...")
    payment_df = _load_payment_data(project_root)
    if payment_df is not None:
        print(f"[Monitoring] Payment data: {len(payment_df):,} loans")
    else:
        print("[Monitoring] No payment data available")

    # --- Load PaymentMonitor ---
    payment_monitor = _load_payment_monitor(model_path)
    if payment_monitor:
        print(f"[Monitoring] PaymentMonitor loaded: {payment_monitor}")
    else:
        print("[Monitoring] PaymentMonitor not available")

    # --- Run drift checks ---
    print("[Monitoring] Running score drift analysis...")
    if scores_df is not None:
        drift_results = _run_score_drift_from_csv(scores_df, month_filter=month)
    else:
        drift_results = {
            "error": "No shadow scores available",
            "score_psi": {},
            "auc": {},
            "calibration": {},
            "grade_distribution": {},
            "overall_status": STATUS_WARNING,
            "alerts": ["No shadow scores available for drift analysis"],
        }

    # --- Run payment checks ---
    print("[Monitoring] Running payment health analysis...")
    payment_results = _run_payment_checks(payment_df, payment_monitor, month_filter=month)

    # --- Format report ---
    text_report, overall_status = _format_text_report(
        drift_results=drift_results,
        payment_results=payment_results,
        period=period,
        timestamp=timestamp,
        scores_path=resolved_scores_path,
    )

    # Print text report
    print()
    print(text_report)

    # --- JSON output ---
    combined_json = {
        "metadata": {
            "period": period,
            "timestamp": timestamp,
            "scores_path": resolved_scores_path,
            "overall_status": overall_status,
        },
        "drift": _make_serializable(drift_results),
        "operational_drift": _make_serializable(drift_results.get("operational_drift", {})),
        "payment": _make_serializable(payment_results),
        "alerts": drift_results.get("alerts", []) + payment_results.get("alerts", []),
    }

    if json_path:
        jp = Path(json_path)
        if not jp.is_absolute():
            jp = project_root / jp
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w") as f:
            json.dump(combined_json, f, indent=2, default=str)
        print(f"[Monitoring] JSON report saved to {jp}")

    # Always save to default location
    default_json_dir = project_root / "results" / "monitoring"
    default_json_dir.mkdir(parents=True, exist_ok=True)
    default_json_path = default_json_dir / f"combined_report_{period.replace('-', '_')}.json"
    with open(default_json_path, "w") as f:
        json.dump(combined_json, f, indent=2, default=str)
    print(f"[Monitoring] JSON report saved to {default_json_path}")

    exit_code = STATUS_EXIT_MAP.get(overall_status, EXIT_WARNING)
    print(f"[Monitoring] Exit code: {exit_code} ({overall_status})")

    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Combined Monthly Portfolio Monitoring Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/models/run_monitoring.py
  python3 scripts/models/run_monitoring.py --scores results/shadow_scores_v5.csv
  python3 scripts/models/run_monitoring.py --json results/monitoring/monthly_report.json
  python3 scripts/models/run_monitoring.py --month 2025-09
  python3 scripts/models/run_monitoring.py --verbose

Exit codes:
  0 = OK (GREEN)
  1 = WARNING
  2 = CRITICAL
        """,
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Path to shadow scores CSV (default: auto-detect latest)",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Path for JSON report output (always saves to results/monitoring/ too)",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="Filter to specific month (YYYY-MM format)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Path to model directory (default: models/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    exit_code = run_monitoring(
        scores_path=args.scores,
        json_path=args.json,
        month=args.month,
        model_dir=args.models,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
