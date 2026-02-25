# Databricks notebook source
# MAGIC %pip install xgboost lightgbm lifelines "scikit-learn>=1.8" -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

"""
run_live_shadow.py — Incremental shadow scoring micro-batch.

Runs every 15 minutes via scheduled job. On each run:
1. Checks which applications have already been scored (from Delta table)
2. Pulls only NEW applications from Databricks
3. Scores them through the shadow pipeline
4. Appends results to shadow_scores Delta table
5. Logs the run

Shadow scores go to: george_sandbox.shadow_scores (Delta table on prod)
Run log goes to:     george_sandbox.shadow_run_log (Delta table on prod)

These tables are created automatically on first run. Only George has
write access to george_sandbox. Nothing touches production tables.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
REPO_DIR = Path("/Workspace/Repos/george@jaris.io/models-shadow")
sys.path.insert(0, str(REPO_DIR))

from pipeline import Pipeline
from fraud_gate import FraudGate
from model_utils import BAD_STATES, EXCLUDE_STATES

RUN_TS = datetime.now()
RUN_ID = RUN_TS.strftime("%Y%m%d_%H%M%S")

# Shadow output tables (George's personal sandbox — no production impact)
SHADOW_CATALOG = "hive_metastore"
SHADOW_SCHEMA = "george_sandbox"
SCORES_TABLE = f"{SHADOW_CATALOG}.{SHADOW_SCHEMA}.shadow_scores"
LOG_TABLE = f"{SHADOW_CATALOG}.{SHADOW_SCHEMA}.shadow_run_log"

print(f"[{RUN_ID}] Shadow scoring micro-batch starting")

# ---------------------------------------------------------------------------
# 1. Ensure sandbox schema exists
# ---------------------------------------------------------------------------
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SHADOW_CATALOG}.{SHADOW_SCHEMA}")

# ---------------------------------------------------------------------------
# 2. Find already-scored application IDs
# ---------------------------------------------------------------------------
try:
    already_scored = spark.sql(f"SELECT DISTINCT application_id FROM {SCORES_TABLE}").toPandas()
    scored_ids = set(already_scored["application_id"].astype(str))
    print(f"[{RUN_ID}] Already scored: {len(scored_ids):,} applications")
except Exception:
    # Table doesn't exist yet — first run
    scored_ids = set()
    print(f"[{RUN_ID}] First run — no existing scores")

# ---------------------------------------------------------------------------
# 3. Pull new applications from Databricks
# ---------------------------------------------------------------------------
query = """
SELECT
    l.application_id,
    CAST(l.signing_date AS STRING) as signing_date,
    l.state as loan_state,
    CAST(l.principal/100 AS DOUBLE) as principal,
    o.partner,
    o.shop_id,
    DATEDIFF(CURRENT_DATE(), l.signing_date) as age,
    CAST(crm.score AS DOUBLE) as experian_FICO_SCORE,
    CAST(ccs.delinquencies_thirty_day_count AS DOUBLE) as d30,
    CAST(ccs.delinquencies_sixty_day_count AS DOUBLE) as d60,
    CAST(ccs.inquiries_last_six_months AS DOUBLE) as inq6,
    CAST(ccs.revolving_account_credit_available_percentage AS DOUBLE) as revutil,
    CAST(ccs.balance_total_past_due_amounts AS DOUBLE) as pastdue,
    CAST(ccs.balance_total_installment_accounts AS DOUBLE) as instbal,
    CAST(ccs.balance_total_revolving_accounts AS DOUBLE) as revbal,
    CAST(ccs.tradelines_total_items AS DOUBLE) as tl_total,
    CAST(ccs.tradelines_total_items_paid AS DOUBLE) as tl_paid,
    CAST(ccs.tradelines_total_items_currently_delinquent AS DOUBLE) as tl_delin,
    CAST(ccs.payment_amount_monthly_total AS DOUBLE) as mopmt,
    DATEDIFF(l.signing_date, ccs.tradelines_oldest_date) as crhist,
    COALESCE(q.qi, 'MISSING') as shop_qi,
    l.grade as prod_grade
FROM gold_prod.analytics.loan_databricks l
LEFT JOIN gold_prod.analytics.loan_offers_databricks o
    ON l.application_id = o.application_id
    AND o.product_name IN ('flex_loan', 'installment_loan')
LEFT JOIN (
    SELECT target_application_id, id as rid
    FROM (
        SELECT id, target_application_id,
               ROW_NUMBER() OVER (PARTITION BY target_application_id ORDER BY fetched_at DESC) as rn
        FROM silver_prod.bureau.reports
        WHERE provider_type = 'EXPERIAN_CIS_CREDIT_REPORT'
    ) s WHERE rn = 1
) cci ON l.application_id = cci.target_application_id
LEFT JOIN silver_prod.bureau.consumer_credit_risk_models crm
    ON cci.rid = crm.report_id AND crm.model_type = 'FICO_RISK_MODEL_V8'
LEFT JOIN silver_prod.bureau.consumer_credit_summary_statistics ccs
    ON cci.rid = ccs.report_id
LEFT JOIN (
    SELECT shop_id, qi,
           ROW_NUMBER() OVER (PARTITION BY shop_id ORDER BY updated_at DESC) as rn
    FROM bronze_prod.prod_modern_treasury_analytics.shop_qi_history
) q ON o.shop_id = q.shop_id AND q.rn = 1
WHERE l.loan_type IN ('FIXED_FEE_FLEX', 'FIXED_FEE_INSTALLMENT')
    AND l.signing_date >= '2024-01-01'
    AND crm.score IS NOT NULL
    AND l.state NOT IN ('REJECTED', 'CANCELED', 'PENDING_FUNDING', 'APPROVED')
"""

df_all = spark.sql(query).toPandas()

# Fix Spark Decimal types
for col in df_all.columns:
    if df_all[col].dtype == object and col not in ['application_id', 'signing_date', 'loan_state',
                                                     'partner', 'shop_id', 'shop_qi', 'prod_grade']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

df_all["signing_date"] = pd.to_datetime(df_all["signing_date"])

# Filter to only new (unscored) applications
df = df_all[~df_all["application_id"].astype(str).isin(scored_ids)].copy()

print(f"[{RUN_ID}] Total in DB: {len(df_all):,}, new to score: {len(df):,}")

if len(df) == 0:
    # Nothing new — log and exit
    print(f"[{RUN_ID}] No new applications. Done.")
    log_df = spark.createDataFrame([{
        "run_id": RUN_ID,
        "run_timestamp": str(RUN_TS),
        "total_in_db": len(df_all),
        "new_scored": 0,
        "status": "NO_NEW_DATA",
    }])
    log_df.write.mode("append").saveAsTable(LOG_TABLE)
    dbutils.notebook.exit("NO_NEW_DATA")

# ---------------------------------------------------------------------------
# 4. Prepare and score
# ---------------------------------------------------------------------------
df["is_bad"] = df["loan_state"].isin(BAD_STATES).astype(int)
if "shop_qi" in df.columns and "qi" not in df.columns:
    df["qi"] = df["shop_qi"]

# Alias columns for pipeline compatibility
if "experian_FICO_SCORE" in df.columns and "fico" not in df.columns:
    df["fico"] = df["experian_FICO_SCORE"]

# Load entity graph
entity_path = REPO_DIR / "entity_graph_cross.parquet"
if entity_path.exists():
    entity_df = pd.read_parquet(entity_path)
    entity_cols = ['application_id', 'has_prior_bad', 'is_repeat', 'is_cross_entity',
                   'prior_loans_ssn', 'prior_bad_ssn']
    entity_feats = entity_df[[c for c in entity_cols if c in entity_df.columns]].copy()
    df = df.merge(entity_feats, on='application_id', how='left')
    for col in entity_feats.columns:
        if col != 'application_id' and col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

# Load pipeline
print(f"[{RUN_ID}] Loading pipeline...")
pipeline = Pipeline.load(str(REPO_DIR))

if entity_path.exists():
    entity_df_lookup = pd.read_parquet(entity_path)
    graph_lookup = FraudGate.build_graph_lookup(entity_df_lookup, key_col='application_id')
    pipeline.fraud_gate.set_graph_lookup(graph_lookup, key_col='application_id')

# Score
print(f"[{RUN_ID}] Scoring {len(df):,} new applications...")
results = pipeline.score_batch(df)

# ---------------------------------------------------------------------------
# 5. Build output and write to Delta
# ---------------------------------------------------------------------------
scored = pd.DataFrame()
scored["application_id"] = df["application_id"].values
scored["run_id"] = RUN_ID
scored["scored_at"] = str(RUN_TS)
scored["signing_date"] = df["signing_date"].astype(str).values
scored["loan_state"] = df["loan_state"].values
scored["is_bad"] = df["is_bad"].values
scored["fico"] = df["experian_FICO_SCORE"].values if "experian_FICO_SCORE" in df.columns else None
scored["qi"] = df["shop_qi"].values if "shop_qi" in df.columns else None
scored["partner"] = df["partner"].values if "partner" in df.columns else None

# Shadow scores
scored["shadow_decision"] = results["decision"].apply(
    lambda d: d.capitalize() if isinstance(d, str) else d).values
scored["shadow_fraud_score"] = results["fraud_score"].values.astype(float)
scored["shadow_fraud_decision"] = results["fraud_decision"].values
scored["shadow_pd"] = results["pd"].values.astype(float)
scored["shadow_grade"] = results["grade"].values
scored["shadow_pricing_tier"] = results["pricing_tier"].values.astype(float)

# Production reference
scored["prod_grade"] = df["prod_grade"].values
scored["prod_grade_letter"] = df["prod_grade"].apply(
    lambda g: str(g)[0] if pd.notna(g) and str(g).strip() else None).values

# Entity
scored["has_prior_bad"] = df["has_prior_bad"].values.astype(int) if "has_prior_bad" in df.columns else 0

# Write to Delta
scored_spark = spark.createDataFrame(scored)
scored_spark.write.mode("append").saveAsTable(SCORES_TABLE)
print(f"[{RUN_ID}] Wrote {len(scored):,} rows to {SCORES_TABLE}")

# ---------------------------------------------------------------------------
# 6. Log the run
# ---------------------------------------------------------------------------
n_bad = int(scored["is_bad"].sum())
n = len(scored)

# Quick AUC if enough data
auc_val = None
if n > 50 and scored["is_bad"].nunique() > 1:
    try:
        from sklearn.metrics import roc_auc_score
        auc_val = round(float(roc_auc_score(scored["is_bad"].values, scored["shadow_pd"].values)), 4)
    except Exception:
        pass

log_entry = {
    "run_id": RUN_ID,
    "run_timestamp": str(RUN_TS),
    "total_in_db": len(df_all),
    "new_scored": n,
    "cumulative_scored": len(scored_ids) + n,
    "new_bads": n_bad,
    "new_bad_rate": round(n_bad / max(n, 1), 4),
    "auc_this_batch": auc_val,
    "decisions_approve": int((scored["shadow_decision"] == "Approve").sum()),
    "decisions_review": int((scored["shadow_decision"] == "Review").sum()),
    "decisions_decline": int((scored["shadow_decision"] == "Decline").sum()),
    "status": "OK",
}

log_df = spark.createDataFrame([log_entry])
log_df.write.mode("append").saveAsTable(LOG_TABLE)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print(f"  SHADOW MICRO-BATCH COMPLETE — {RUN_ID}")
print(f"{'=' * 60}")
print(f"  New applications scored: {n:,}")
print(f"  Cumulative scored:       {len(scored_ids) + n:,}")
print(f"  Decisions: Approve={log_entry['decisions_approve']}, "
      f"Review={log_entry['decisions_review']}, "
      f"Decline={log_entry['decisions_decline']}")
if n_bad > 0:
    print(f"  Bad rate (this batch): {n_bad/n:.1%}")
if auc_val:
    print(f"  AUC (this batch): {auc_val}")
print(f"  Results in: {SCORES_TABLE}")
print(f"  Log in:     {LOG_TABLE}")
print(f"{'=' * 60}")
