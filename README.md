# Shadow Scoring Deployment Package

Self-contained directory. Upload everything in this folder to wherever you want to run shadow scoring.

## What's Here

- **15 Python files** — all model classes, scoring, monitoring, API
- **21 joblib files** — serialized model artifacts
- **2 parquet files** — master_features (14K loans) + entity_graph (10K entries)
- **requirements.txt** — pinned dependencies

## Quick Start

```bash
# Install deps (if not already available)
pip install -r requirements.txt

# Score the full portfolio
python score_portfolio.py

# Run monitoring
python run_monitoring.py

# Start the API
uvicorn api:app --host 0.0.0.0 --port 8000

# Retrain models (if you have fresh data)
python train_all.py
```

## On Databricks

Upload this entire directory to a Databricks workspace path (e.g., `/Users/george@jaris.io/deploy/`). Then in a notebook:

```python
import sys
sys.path.insert(0, '/Workspace/Users/george@jaris.io/deploy/')

from pipeline import Pipeline
p = Pipeline.load('/Workspace/Users/george@jaris.io/deploy/')
result = p.score({"fico": 620, "qi": "MISSING", "partner": "PAYSAFE"})
print(result)
```

For scheduled shadow scoring, create a Databricks job that runs `score_portfolio.py` weekly.

## File Manifest

### Model Classes
| File | What |
|------|------|
| pipeline.py | Pipeline v1.3.0 — chains FraudGate + DefaultScorecard + CreditGrader |
| fraud_gate.py | FraudGate v1.2.0 — rule-based fraud scoring |
| default_scorecard.py | DefaultScorecard v1.6.0 — WoE + XGB + Rule blend |
| credit_grader.py | CreditGrader v1.1.0 — PD to letter grade |
| survival_scorer.py | SurvivalScorer v1.0.0 — Cox PH time-to-default |
| payment_monitor.py | PaymentMonitor — post-origination NSF monitoring |
| drift_monitor.py | DriftMonitor — PSI, AUC decay, calibration |
| review_optimizer.py | ReviewOptimizer — prioritized review queue |

### Operations
| File | What |
|------|------|
| score_portfolio.py | Batch shadow scoring with CSV output |
| run_monitoring.py | Combined drift + payment monitoring CLI |
| train_all.py | Retrain all models end-to-end |
| pull_data.py | Refresh data from Databricks |
| api.py | FastAPI endpoints (+ Dockerfile pattern in source) |
| build_graph_lookup.py | Refresh entity graph from Databricks |

### Shared
| File | What |
|------|------|
| feature_engine.py | Feature engineering pipeline |
| model_utils.py | Constants, data loading, evaluation |
| __init__.py | Package init |
