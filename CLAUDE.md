# Shadow Scoring Pipeline — V5.7

Self-contained deployment of the Jaris loan default risk scoring pipeline. This repo runs independently — no dependency on the hack-week research repo.

## What This Is

A 3-model pipeline that scores loan applications for fraud risk, default probability, and credit grade. Runs in shadow mode alongside production — scores everything but doesn't affect real decisions.

**V5.7 Models:**
- FraudGate v1.2.0: Rule-based fraud scoring (AUC 0.657)
- DefaultScorecard v1.6.0: WoE + XGB + Rule blend (AUC 0.660)
- CreditGrader v1.1.0: 5-grade A-E (19.4x spread)
- SurvivalScorer v1.0.0: Cox PH time-to-default (C-index 0.634)
- PaymentMonitor: Post-origination NSF monitoring (AUC 0.844)

## RULES

### Safety
- **This is SHADOW MODE** — scores are for comparison only, not for production decisions
- **NEVER modify production tables** — only read from dataproducts, write shadow scores locally or to a dedicated shadow table
- **NEVER use `jobs/runs/submit`** — use Command Execution API or SQL Statement API only
- **Databricks dataproducts is READ ONLY** — warehouse `3d2cc6b2a32c1b72`

### Code
- **Do NOT install packages** — use `python3 -m pip` if absolutely necessary (never `pip3`)
- **Do NOT modify model artifacts (.joblib files)** unless retraining — they are the trained models
- **partner_encoded was REMOVED from DefaultScorecard** for fair lending — do not re-add it
- **Taktile rules are DISABLED** — temporal confounding makes them harmful. Do not re-enable.

## Quick Start

```bash
# Score the full portfolio (reads from local parquet or Databricks)
python3 score_portfolio.py

# Run monitoring
python3 run_monitoring.py

# Start API server
uvicorn api:app --host 0.0.0.0 --port 8000

# Retrain all models from fresh data
python3 train_all.py

# Refresh entity graph from Databricks
python3 build_graph_lookup.py

# Refresh data from Databricks
python3 pull_data.py --all
```

## Scoring a Single Application

```python
from pipeline import Pipeline
p = Pipeline.load(".")  # loads all .joblib files from current directory
result = p.score({
    "fico": 620,
    "qi": "MISSING",
    "partner": "PAYSAFE",
    "d30": 2,
    "d60": 0,
    "inq6": 3,
    "revutil": 75,
    "pastdue": 500,
    "instbal": 10000,
    "revbal": 8000,
    "tl_total": 12,
    "tl_paid": 8,
    "tl_delin": 2,
    "mopmt": 450,
    "crhist": 1825,  # DAYS not months
})
# Returns: {decision, fraud_score, pd, grade, reasons, ...}
```

## Deploying to Databricks

### Option 1: Upload as workspace files
1. Push this repo to GitHub
2. In Databricks prod workspace, go to Repos → Add Repo → paste GitHub URL
3. Open a notebook, add the repo to sys.path, import and run

### Option 2: Upload files directly
1. Use Databricks CLI: `databricks workspace import_dir ./models-shadow /Users/george@jaris.io/models-shadow --profile prod`
2. Or drag-and-drop in the Databricks UI

### Scheduling
Create a Databricks job (NOT `jobs/runs/submit` — use the UI) that:
1. Runs `score_portfolio.py` weekly
2. Runs `run_monitoring.py` monthly
3. Writes shadow scores to a Delta table (create in prod workspace)

## File Reference

| File | Purpose |
|------|---------|
| pipeline.py | Pipeline v1.3.0 — chains all models |
| fraud_gate.py | FraudGate v1.2.0 — rule-based fraud scoring |
| default_scorecard.py | DefaultScorecard v1.6.0 — 3-sub-model blend |
| credit_grader.py | CreditGrader v1.1.0 — PD to letter grade |
| survival_scorer.py | SurvivalScorer v1.0.0 — time-to-default |
| payment_monitor.py | PaymentMonitor — post-origination NSF alerts |
| drift_monitor.py | DriftMonitor — PSI + calibration + approve-rate |
| review_optimizer.py | ReviewOptimizer — prioritized review queue |
| score_portfolio.py | Batch shadow scoring with CSV output |
| run_monitoring.py | Combined monitoring CLI |
| train_all.py | Retrain all models end-to-end |
| pull_data.py | Refresh data from Databricks |
| build_graph_lookup.py | Refresh entity graph from Databricks |
| api.py | FastAPI endpoints |
| feature_engine.py | Feature engineering pipeline |
| model_utils.py | Shared constants and utilities |
| *.joblib | Serialized model artifacts (21 files) |

## Databricks Connection

```bash
# Query dataproducts (read-only)
databricks --profile default api post /api/2.0/sql/statements \
  --json '{"warehouse_id":"3d2cc6b2a32c1b72","statement":"SELECT count(*) FROM gold_prod.analytics.loan_databricks","wait_timeout":"50s"}'
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Pipeline throughput | 42,000 apps/sec |
| Memory footprint | ~500 MB |
| Model artifacts size | 5.3 MB total |
| Recommended config | Moderate-Tight: 60% approve, 4.1% bad rate |
| Stress tests passing | 305/312 (97.8%) |

## Full Documentation

The complete 2,700-line master guide, visuals, research papers, and all supporting docs are in the `hack-week` repo at `~/Desktop/hack-week/MASTER_MODEL_GUIDE.md`.
