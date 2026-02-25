# Databricks notebook source
# MAGIC %pip install xgboost lightgbm -q

# COMMAND ----------

"""
run_shadow.py — One-click shadow scoring notebook for Databricks.

Open this file in a Databricks notebook and Run All. It will:
1. Load the pipeline (5 models) and all data
2. Score every loan in the portfolio (14K applications)
3. Save results to a persistent CSV + JSON report
4. Print a full performance report inline

Results are saved to:
  - shadow_scores_YYYY-MM-DD.csv  (per-loan scores, ~30 columns)
  - shadow_report_YYYY-MM-DD.json (aggregate metrics, machine-readable)
  - shadow_report_latest.json     (always points to most recent run)

These persist in the repo workspace directory between runs.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Setup paths — works both locally (__file__ exists) and on Databricks (it doesn't)
# ---------------------------------------------------------------------------
try:
    REPO_DIR = Path(__file__).resolve().parent
except NameError:
    # Databricks notebook: __file__ is not defined
    REPO_DIR = Path("/Workspace/Repos/george@jaris.io/models-shadow")
sys.path.insert(0, str(REPO_DIR))

from pipeline import Pipeline
from fraud_gate import FraudGate
from model_utils import BAD_STATES, EXCLUDE_STATES

SPLIT_DATE = "2025-07-01"
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
RESULTS_DIR = REPO_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load pipeline
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"  SHADOW SCORING RUN — {RUN_DATE}")
print("=" * 70)

print("\n[1/5] Loading pipeline...")
pipeline = Pipeline.load(str(REPO_DIR))
print(f"  {pipeline}")

# ---------------------------------------------------------------------------
# 2. Load and prepare data
# ---------------------------------------------------------------------------
print("\n[2/5] Loading data...")
df = pd.read_parquet(REPO_DIR / "master_features.parquet")
print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")

# Join entity graph
entity_path = REPO_DIR / "entity_graph_cross.parquet"
if entity_path.exists():
    entity_df = pd.read_parquet(entity_path)
    entity_cols = ['application_id', 'has_prior_bad', 'is_repeat', 'is_cross_entity',
                   'prior_loans_ssn', 'prior_shops_ssn', 'prior_bad_ssn', 'prior_paid_ssn']
    entity_feats = entity_df[[c for c in entity_cols if c in entity_df.columns]].copy()
    df = df.merge(entity_feats, on='application_id', how='left')
    for col in entity_feats.columns:
        if col != 'application_id' and col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    n_prior_bad = (df['has_prior_bad'] == 1).sum() if 'has_prior_bad' in df.columns else 0
    print(f"  Entity graph joined: {n_prior_bad} loans with prior_bad")

# Filter and prep
df = df[~df["loan_state"].isin(EXCLUDE_STATES)].copy()
df["signing_date"] = pd.to_datetime(df["signing_date"])
df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)
if "shop_qi" in df.columns and "qi" not in df.columns:
    df["qi"] = df["shop_qi"]
print(f"  After filtering: {len(df):,} loans, {df['is_bad'].sum():,} bads ({df['is_bad'].mean():.1%})")

# Load entity graph into FraudGate
if entity_path.exists():
    entity_df_lookup = pd.read_parquet(entity_path)
    graph_lookup = FraudGate.build_graph_lookup(entity_df_lookup, key_col='application_id')
    pipeline.fraud_gate.set_graph_lookup(graph_lookup, key_col='application_id')
    print(f"  Entity graph loaded into FraudGate: {len(graph_lookup)} entries")

# ---------------------------------------------------------------------------
# 3. Score entire portfolio
# ---------------------------------------------------------------------------
print("\n[3/5] Scoring portfolio...")
results = pipeline.score_batch(df)
print(f"  Scored {len(results):,} applications")

# Build output DataFrame
scored = pd.DataFrame(index=df.index)
scored["application_id"] = df["application_id"].values
scored["signing_date"] = df["signing_date"].values
scored["loan_state"] = df["loan_state"].values
scored["is_bad"] = df["is_bad"].values

# Pipeline outputs
scored["decision"] = results["decision"].apply(lambda d: d.capitalize() if isinstance(d, str) else d).values
scored["fraud_score"] = results["fraud_score"].values
scored["fraud_normalized"] = results["fraud_normalized"].values
scored["fraud_decision"] = results["fraud_decision"].values
scored["pd"] = results["pd"].values
scored["default_score"] = results["default_score"].values
scored["woe_pd"] = results["woe_pd"].values
scored["rule_pd"] = results["rule_pd"].values
scored["xgb_pd"] = results["xgb_pd"].values
scored["grade"] = results["grade"].values
scored["pricing_tier"] = results["pricing_tier"].values

# Context
if "experian_FICO_SCORE" in df.columns:
    scored["fico"] = df["experian_FICO_SCORE"].values
if "shop_qi" in df.columns:
    scored["qi"] = df["shop_qi"].values
if "partner" in df.columns:
    scored["partner"] = df["partner"].values
if "grade" in df.columns:
    scored["production_grade"] = df["grade"].values
    scored["production_grade_letter"] = df["grade"].apply(
        lambda g: str(g)[0] if pd.notna(g) and str(g).strip() else None
    ).values

# Entity features
for col in ['has_prior_bad', 'is_repeat', 'is_cross_entity', 'prior_bad_ssn']:
    if col in df.columns:
        scored[col] = df[col].values

# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------
print("\n[4/5] Saving results...")
csv_path = RESULTS_DIR / f"shadow_scores_{RUN_DATE}.csv"
scored.to_csv(csv_path, index=False)
print(f"  CSV: {csv_path} ({len(scored):,} rows)")

# ---------------------------------------------------------------------------
# 5. Compute and save report
# ---------------------------------------------------------------------------
print("\n[5/5] Computing report...")

n = len(scored)
n_bad = int(scored["is_bad"].sum())
is_test = scored["signing_date"] >= SPLIT_DATE
test = scored[is_test]
train = scored[~is_test]

report = {
    "run_date": RUN_DATE,
    "run_timestamp": datetime.now().isoformat(),
    "pipeline": str(pipeline),
    "data": {
        "total_loans": n,
        "train_loans": len(train),
        "test_loans": len(test),
        "total_bads": n_bad,
        "bad_rate": round(n_bad / n, 4),
        "date_range": f"{scored['signing_date'].min()} to {scored['signing_date'].max()}",
    },
    "decisions": {},
    "model_performance": {},
    "grades": {},
    "fraud_gate": {},
    "production_comparison": {},
}

# --- Decisions ---
for dec in ["Approve", "Review", "Decline"]:
    mask = scored["decision"] == dec
    subset = scored[mask]
    report["decisions"][dec] = {
        "count": int(mask.sum()),
        "pct": round(float(mask.sum() / n), 4),
        "bad_rate": round(float(subset["is_bad"].mean()), 4) if len(subset) > 0 else None,
        "bads": int(subset["is_bad"].sum()) if len(subset) > 0 else 0,
    }

# --- Model Performance (test set) ---
if len(test) > 0 and test["is_bad"].sum() > 0:
    y_test = test["is_bad"].values
    perf = {}
    for name, col in [("fraud_gate", "fraud_score"), ("default_blend", "pd"),
                       ("woe", "woe_pd"), ("rule", "rule_pd"), ("xgb", "xgb_pd"),
                       ("pricing_tier", "pricing_tier")]:
        try:
            auc = float(roc_auc_score(y_test, test[col].values))
            perf[name] = {"auc": round(auc, 4), "gini": round(2 * auc - 1, 4)}
        except (ValueError, KeyError):
            perf[name] = {"auc": None, "gini": None}

    # Brier score for PD calibration
    try:
        brier = float(brier_score_loss(y_test, test["pd"].values))
        perf["default_blend"]["brier"] = round(brier, 4)
    except (ValueError, KeyError):
        pass

    # KS statistic
    try:
        fpr, tpr, _ = roc_curve(y_test, test["pd"].values)
        ks = float(np.max(tpr - fpr))
        perf["default_blend"]["ks"] = round(ks, 4)
    except (ValueError, KeyError):
        pass

    # Calibration ratio
    pred_mean = float(test["pd"].mean())
    actual_mean = float(test["is_bad"].mean())
    perf["calibration"] = {
        "predicted_mean_pd": round(pred_mean, 4),
        "actual_bad_rate": round(actual_mean, 4),
        "ratio": round(pred_mean / max(actual_mean, 0.001), 4),
    }

    report["model_performance"] = perf
    report["model_performance"]["test_n"] = len(test)
    report["model_performance"]["test_bads"] = int(test["is_bad"].sum())

# --- Grades ---
for grade in ["A", "B", "C", "D", "E"]:
    mask = scored["grade"] == grade
    subset = scored[mask]
    if len(subset) > 0:
        report["grades"][grade] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.sum() / n), 4),
            "bad_rate": round(float(subset["is_bad"].mean()), 4),
            "bads": int(subset["is_bad"].sum()),
            "mean_pd": round(float(subset["pd"].mean()), 4),
        }

# --- Fraud Gate ---
for fd in ["pass", "review", "decline"]:
    mask = scored["fraud_decision"] == fd
    subset = scored[mask]
    if len(subset) > 0:
        report["fraud_gate"][fd] = {
            "count": int(mask.sum()),
            "pct": round(float(mask.sum() / n), 4),
            "bad_rate": round(float(subset["is_bad"].mean()), 4),
        }

# --- Production Comparison ---
if "production_grade_letter" in scored.columns:
    scored_with_prod = scored.dropna(subset=["production_grade_letter"])
    if len(scored_with_prod) > 0:
        agree = (scored_with_prod["grade"] == scored_with_prod["production_grade_letter"]).sum()
        report["production_comparison"]["grade_agreement_rate"] = round(float(agree / len(scored_with_prod)), 4)
        report["production_comparison"]["n_compared"] = len(scored_with_prod)

        # Would-have-declined analysis
        would_decline = (scored["decision"] == "Decline")
        would_caught = would_decline & (scored["is_bad"] == 1)
        report["production_comparison"]["would_decline"] = int(would_decline.sum())
        report["production_comparison"]["would_catch_defaults"] = int(would_caught.sum())
        if n_bad > 0:
            report["production_comparison"]["default_capture_rate"] = round(float(would_caught.sum() / n_bad), 4)

# Save JSON report
json_path = RESULTS_DIR / f"shadow_report_{RUN_DATE}.json"
with open(json_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  JSON: {json_path}")

# Also save as latest
latest_path = RESULTS_DIR / "shadow_report_latest.json"
with open(latest_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  Latest: {latest_path}")

# ---------------------------------------------------------------------------
# Print full report
# ---------------------------------------------------------------------------
print("\n")
print("=" * 70)
print(f"  SHADOW SCORING RESULTS — {RUN_DATE}")
print("=" * 70)

print(f"\nPortfolio: {n:,} loans scored")
print(f"  Train (< {SPLIT_DATE}): {len(train):,}")
print(f"  Test  (>= {SPLIT_DATE}): {len(test):,}")
print(f"  Bad rate: {n_bad/n:.1%} ({n_bad:,} bads)")

print(f"\n--- Pipeline Decisions ---")
for dec in ["Approve", "Review", "Decline"]:
    d = report["decisions"].get(dec, {})
    count = d.get("count", 0)
    pct = d.get("pct", 0)
    br = d.get("bad_rate")
    bads = d.get("bads", 0)
    br_str = f"{br:.1%}" if br is not None else "N/A"
    print(f"  {dec:<10} {count:>6,}  ({pct:5.1%})  bad rate: {br_str}  ({bads} bads)")

print(f"\n--- Model Performance (Test Set, n={len(test):,}) ---")
perf = report.get("model_performance", {})
for name, display in [("fraud_gate", "Fraud Gate"), ("default_blend", "Default Blend"),
                       ("woe", "  WoE sub-model"), ("rule", "  Rule sub-model"),
                       ("xgb", "  XGBoost sub-model"), ("pricing_tier", "Pricing Tier")]:
    m = perf.get(name, {})
    auc = m.get("auc")
    if auc is not None:
        extra = ""
        if name == "default_blend":
            ks = m.get("ks")
            brier = m.get("brier")
            if ks: extra += f"  KS={ks:.4f}"
            if brier: extra += f"  Brier={brier:.4f}"
        print(f"  {display:<22} AUC={auc:.4f}  Gini={m.get('gini', 0):.4f}{extra}")

cal = perf.get("calibration", {})
if cal:
    print(f"\n  Calibration: predicted={cal.get('predicted_mean_pd', 0):.1%}, "
          f"actual={cal.get('actual_bad_rate', 0):.1%}, "
          f"ratio={cal.get('ratio', 0):.2f}")

print(f"\n--- Grade Distribution & Bad Rates ---")
print(f"  {'Grade':<7} {'Count':>7} {'Pct':>7} {'Bad Rate':>10} {'Bads':>6} {'Mean PD':>9}")
print(f"  {'-----':<7} {'-----':>7} {'---':>7} {'--------':>10} {'----':>6} {'-------':>9}")
for grade in ["A", "B", "C", "D", "E"]:
    g = report["grades"].get(grade, {})
    if g:
        print(f"  {grade:<7} {g['count']:>7,} {g['pct']:>7.1%} {g['bad_rate']:>10.1%} {g['bads']:>6,} {g['mean_pd']:>9.2%}")

# Grade spread
grade_rates = [report["grades"][g]["bad_rate"] for g in ["A", "E"] if g in report["grades"]]
if len(grade_rates) == 2 and grade_rates[0] > 0:
    spread = grade_rates[1] / grade_rates[0]
    print(f"\n  Grade spread (E/A): {spread:.1f}x")

print(f"\n--- Fraud Gate Breakdown ---")
for fd in ["pass", "review", "decline"]:
    fg = report["fraud_gate"].get(fd, {})
    if fg:
        print(f"  {fd:<10} {fg['count']:>6,}  ({fg['pct']:5.1%})  bad rate: {fg['bad_rate']:.1%}")

pc = report.get("production_comparison", {})
if pc:
    print(f"\n--- vs Production ---")
    print(f"  Grade agreement: {pc.get('grade_agreement_rate', 0):.1%}")
    print(f"  Would decline: {pc.get('would_decline', 0):,} loans")
    print(f"  Would catch: {pc.get('would_catch_defaults', 0):,} defaults "
          f"({pc.get('default_capture_rate', 0):.1%} of all defaults)")

print(f"\n--- Saved Files ---")
print(f"  {csv_path}")
print(f"  {json_path}")
print(f"  {latest_path}")
print(f"\n{'=' * 70}")
print(f"  RUN COMPLETE")
print(f"{'=' * 70}")
