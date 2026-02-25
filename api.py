"""
REST API for the Jaris Risk Model Pipeline.

FastAPI wrapper around the Pipeline class providing:
  - POST /score         -- Score a single loan application
  - POST /score/batch   -- Score multiple applications (max 1000)
  - POST /monitor       -- Post-origination payment monitoring
  - GET  /health        -- Basic health check
  - GET  /health/models -- Detailed model component health

Run with:
    python3 -m uvicorn scripts.models.api:app --host 0.0.0.0 --port 8000

Or directly:
    python3 scripts/models/api.py
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_VERSION = "0.1.0"
MAX_BATCH_SIZE = 1000
MODEL_DIR = "models/"

VALID_QI_VALUES = {"HFHT", "HFLT", "LFHT", "LFLT", "MISSING"}


# ---------------------------------------------------------------------------
# Pydantic Models — Requests
# ---------------------------------------------------------------------------


class QIEnum(str, Enum):
    HFHT = "HFHT"
    HFLT = "HFLT"
    LFHT = "LFHT"
    LFLT = "LFLT"
    MISSING = "MISSING"


class ApplicationRequest(BaseModel):
    """Input features for a single loan application."""

    # -- Core identifiers / required fields --
    fico: float = Field(
        ..., ge=300, le=850, description="FICO credit score (300-850)"
    )
    qi: str = Field(
        ...,
        description="QI classification: HFHT, HFLT, LFHT, LFLT, or MISSING",
    )
    partner: str = Field(
        ..., min_length=1, description="Partner/origination channel name"
    )

    # -- Bureau features (with sensible defaults) --
    pastdue: float = Field(
        0, ge=0, description="Total past-due amount in dollars"
    )
    d30: float = Field(
        0, ge=0, description="Count of 30-day delinquencies"
    )
    d60: float = Field(
        0, ge=0, description="Count of 60-day delinquencies"
    )
    delin_rate: Optional[float] = Field(
        None, ge=0, le=1, description="Delinquency rate (0-1)"
    )
    bal_to_limit: Optional[float] = Field(
        None, ge=0, description="Balance-to-limit ratio"
    )
    rev_avail_pct: Optional[float] = Field(
        None, ge=0, le=1, description="Revolving credit available percentage"
    )
    revbal: Optional[float] = Field(
        None, ge=0, description="Total revolving balance"
    )
    mopmt: Optional[float] = Field(
        None, ge=0, description="Monthly payment amount"
    )
    revutil: Optional[float] = Field(
        None, ge=0, description="Revolving utilization ratio"
    )
    instbal: Optional[float] = Field(
        None, ge=0, description="Total installment balance"
    )
    tl_total: Optional[float] = Field(
        None, ge=0, description="Total tradelines"
    )
    tl_paid: Optional[float] = Field(
        None, ge=0, description="Paid tradelines"
    )
    tl_delin: Optional[float] = Field(
        None, ge=0, description="Delinquent tradelines"
    )
    crhist: Optional[float] = Field(
        None, ge=0, description="Credit history length in months"
    )
    inq6: Optional[float] = Field(
        None, ge=0, description="Inquiries in last 6 months"
    )

    # -- Derived / engineered features --
    cash_stress: Optional[float] = Field(
        None, description="Cash stress indicator"
    )
    bureau_health: Optional[float] = Field(
        None, description="Bureau health composite score"
    )

    # -- Business features --
    years_in_business: Optional[float] = Field(
        None, ge=0, description="Years in business"
    )

    # -- Entity graph features --
    has_prior_bad: Optional[int] = Field(
        None, ge=0, le=1, description="Has prior bad loan (0/1)"
    )
    is_repeat: Optional[int] = Field(
        None, ge=0, le=1, description="Is repeat borrower (0/1)"
    )

    # -- Banking features --
    nsf_count: Optional[float] = Field(
        None, ge=0, description="NSF return count"
    )
    negative_balance_days_90d: Optional[float] = Field(
        None, ge=0, description="Days with negative balance in last 90d"
    )
    plaid_balance: Optional[float] = Field(
        None, description="Plaid checking account balance"
    )

    @field_validator("qi")
    @classmethod
    def validate_qi(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in VALID_QI_VALUES:
            raise ValueError(
                f"QI must be one of {sorted(VALID_QI_VALUES)}, got '{v}'"
            )
        return upper

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by Pipeline.score()."""
        d = self.model_dump(exclude_none=True)
        return d


class BatchScoreRequest(BaseModel):
    """Batch scoring request: list of application feature dicts."""

    applications: List[ApplicationRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of applications to score (max {MAX_BATCH_SIZE})",
    )


class MonitorRequest(BaseModel):
    """Post-origination monitoring features for a single loan."""

    loan_id: Optional[str] = Field(
        None, description="Loan or application identifier"
    )
    jaris_application_id: Optional[str] = Field(
        None, description="Jaris application ID (alternative to loan_id)"
    )

    # -- NSF / payment features --
    nsf_count: float = Field(
        0, ge=0, description="Total NSF/returned payment count"
    )
    nsf_rate: float = Field(
        0, ge=0, le=1, description="NSF rate across all repayment attempts"
    )
    nsf_in_first_90d: Optional[int] = Field(
        None, ge=0, le=1, description="Had NSF in first 90 days (0/1)"
    )
    nsf_cluster_max: Optional[float] = Field(
        None, ge=0, description="Max NSFs in a single cluster"
    )
    max_consecutive_nsf: Optional[float] = Field(
        None, ge=0, description="Max consecutive NSF returns"
    )
    has_recovery: Optional[int] = Field(
        None, ge=0, le=1, description="Has recovery attempt (0/1)"
    )
    recovery_nsf_rate: Optional[float] = Field(
        None, ge=0, le=1, description="NSF rate during recovery period"
    )
    payment_cv: Optional[float] = Field(
        None, ge=0, description="Payment amount coefficient of variation"
    )
    payment_decline_pct: Optional[float] = Field(
        None, ge=0, le=1, description="Percentage of declining payments"
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to the dict format expected by Pipeline.monitor()."""
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# Pydantic Models — Responses
# ---------------------------------------------------------------------------


class ScoreResponse(BaseModel):
    """Full pipeline scoring result for one application."""

    request_id: str = Field(description="Unique request identifier")
    decision: str = Field(description="Overall decision: approve, review, decline")
    fraud_score: float = Field(description="Raw fraud score (points)")
    fraud_normalized: float = Field(description="Normalized fraud score (0-1)")
    fraud_decision: str = Field(description="Fraud gate decision: pass, review, decline")
    fraud_reasons: List[str] = Field(description="Fraud-specific reason codes")
    pd: float = Field(description="Calibrated probability of default")
    default_score: float = Field(description="Default scorecard score (300-850 scale)")
    grade: str = Field(description="Credit grade letter (A-G)")
    pricing_tier: int = Field(description="Pricing tier number")
    all_model_scores: Dict[str, Any] = Field(
        description="Component model scores (woe_pd, rule_pd, xgb_pd)"
    )
    reasons: List[str] = Field(description="Top reason codes for the decision")
    monitoring_available: bool = Field(
        description="Whether post-origination monitoring is loaded"
    )


class BatchScoreResponse(BaseModel):
    """Batch scoring results."""

    request_id: str
    count: int = Field(description="Number of applications scored")
    results: List[ScoreResponse]


class MonitorResponse(BaseModel):
    """Post-origination monitoring result for one loan."""

    request_id: str
    payment_tier: str = Field(description="Risk tier: Green, Yellow, Orange, Red")
    payment_pd: float = Field(description="Calibrated PD from payment signals")
    flag_count: int = Field(description="Number of triggered flags (0-9)")
    flags: List[str] = Field(description="Descriptions of triggered flags")
    loan_id: str = Field(description="Loan identifier")


class HealthResponse(BaseModel):
    """Basic health check response."""

    status: str = Field(description="Service status: healthy or degraded")
    api_version: str = Field(description="API version")
    pipeline_version: str = Field(description="Pipeline model version")
    models_loaded: Dict[str, bool] = Field(
        description="Which model components are loaded"
    )
    uptime_seconds: float = Field(description="Seconds since API startup")
    last_prediction_time: Optional[str] = Field(
        description="ISO timestamp of last prediction"
    )
    total_predictions: int = Field(description="Total predictions served")


class ModelHealthResponse(BaseModel):
    """Detailed model component health."""

    fraud_gate: Dict[str, Any] = Field(description="FraudGate details")
    default_scorecard: Dict[str, Any] = Field(description="DefaultScorecard details")
    credit_grader: Dict[str, Any] = Field(description="CreditGrader details")
    payment_monitor: Dict[str, Any] = Field(description="PaymentMonitor details")
    pipeline: Dict[str, Any] = Field(description="Pipeline-level config")


class ErrorResponse(BaseModel):
    """Standard error response."""

    request_id: str
    error: str
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------


class AppState:
    """Mutable application state — holds the loaded pipeline and metrics."""

    def __init__(self):
        self.pipeline = None
        self.startup_time: Optional[float] = None
        self.last_prediction_time: Optional[str] = None
        self.total_predictions: int = 0


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    # Startup
    logger.info("Loading pipeline from %s ...", MODEL_DIR)
    try:
        from pipeline import Pipeline

        state.pipeline = Pipeline.load(MODEL_DIR)
        state.startup_time = time.time()
        logger.info("Pipeline loaded successfully.")
    except FileNotFoundError:
        logger.warning(
            "Model files not found in %s. "
            "The /score and /monitor endpoints will return 503 until models are loaded. "
            "Health endpoints will still work.",
            MODEL_DIR,
        )
        state.startup_time = time.time()
    except Exception:
        logger.exception("Failed to load pipeline.")
        state.startup_time = time.time()

    yield  # --- app is running ---

    # Shutdown
    logger.info("Shutting down API. Total predictions served: %d", state.total_predictions)
    state.pipeline = None


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Jaris Risk Model API",
    description=(
        "REST API for the Jaris subprime SMB loan risk pipeline. "
        "Provides origination scoring (FraudGate + DefaultScorecard + CreditGrader) "
        "and optional post-origination payment monitoring."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware — request ID injection
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Inject a unique request ID into every request for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_pipeline() -> None:
    """Raise 503 if the pipeline is not loaded."""
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Model files may be missing from the models/ directory.",
        )


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/score",
    response_model=ScoreResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Model scoring error"},
        503: {"description": "Pipeline not loaded"},
    },
    summary="Score a single loan application",
    tags=["Scoring"],
)
async def score_application(body: ApplicationRequest, request: Request):
    """Score a single loan application through the full pipeline.

    Runs FraudGate -> DefaultScorecard -> CreditGrader -> Decision logic
    and returns the combined result with reason codes.
    """
    _require_pipeline()
    request_id = _get_request_id(request)

    try:
        features = body.to_feature_dict()
        result = state.pipeline.score(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Scoring error for request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal scoring error: {type(exc).__name__}: {exc}",
        )

    state.last_prediction_time = datetime.now(timezone.utc).isoformat()
    state.total_predictions += 1

    return ScoreResponse(
        request_id=request_id,
        decision=result["decision"],
        fraud_score=result["fraud_score"],
        fraud_normalized=result["fraud_normalized"],
        fraud_decision=result["fraud_decision"],
        fraud_reasons=result.get("fraud_reasons", []),
        pd=result["pd"],
        default_score=result["default_score"],
        grade=result["grade"],
        pricing_tier=result["pricing_tier"],
        all_model_scores=result.get("all_model_scores", {}),
        reasons=result.get("reasons", []),
        monitoring_available=result.get("monitoring_available", False),
    )


@app.post(
    "/score/batch",
    response_model=BatchScoreResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Model scoring error"},
        503: {"description": "Pipeline not loaded"},
    },
    summary="Score multiple loan applications",
    tags=["Scoring"],
)
async def score_batch(body: BatchScoreRequest, request: Request):
    """Score a batch of loan applications (max 1000).

    Each application is scored independently through the full pipeline.
    Results are returned in the same order as the input.
    """
    _require_pipeline()
    request_id = _get_request_id(request)

    results = []
    errors = []

    for i, app_req in enumerate(body.applications):
        try:
            features = app_req.to_feature_dict()
            result = state.pipeline.score(features)
            results.append(
                ScoreResponse(
                    request_id=f"{request_id}:{i}",
                    decision=result["decision"],
                    fraud_score=result["fraud_score"],
                    fraud_normalized=result["fraud_normalized"],
                    fraud_decision=result["fraud_decision"],
                    fraud_reasons=result.get("fraud_reasons", []),
                    pd=result["pd"],
                    default_score=result["default_score"],
                    grade=result["grade"],
                    pricing_tier=result["pricing_tier"],
                    all_model_scores=result.get("all_model_scores", {}),
                    reasons=result.get("reasons", []),
                    monitoring_available=result.get("monitoring_available", False),
                )
            )
        except ValueError as exc:
            errors.append(f"Application [{i}]: {exc}")
        except Exception as exc:
            logger.exception("Batch scoring error at index %d, request %s", i, request_id)
            errors.append(f"Application [{i}]: {type(exc).__name__}: {exc}")

    if errors and not results:
        raise HTTPException(
            status_code=400,
            detail=f"All applications failed validation:\n" + "\n".join(errors),
        )

    if errors:
        logger.warning(
            "Batch request %s: %d succeeded, %d failed. Failures: %s",
            request_id, len(results), len(errors), errors,
        )

    state.last_prediction_time = datetime.now(timezone.utc).isoformat()
    state.total_predictions += len(results)

    return BatchScoreResponse(
        request_id=request_id,
        count=len(results),
        results=results,
    )


@app.post(
    "/monitor",
    response_model=MonitorResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Model error"},
        503: {"description": "Pipeline or PaymentMonitor not loaded"},
    },
    summary="Monitor a loan post-origination",
    tags=["Monitoring"],
)
async def monitor_loan(body: MonitorRequest, request: Request):
    """Monitor a single loan using post-origination payment/NSF signals.

    Requires PaymentMonitor to be loaded (payment_monitor.joblib in models/).
    Returns risk tier (Green/Yellow/Orange/Red), calibrated PD, and triggered flags.
    """
    _require_pipeline()
    request_id = _get_request_id(request)

    if state.pipeline.payment_monitor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "PaymentMonitor is not loaded. Place payment_monitor.joblib "
                "in the models/ directory and restart the server."
            ),
        )

    try:
        features = body.to_feature_dict()
        result = state.pipeline.monitor(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Monitor error for request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Monitoring error: {type(exc).__name__}: {exc}",
        )

    state.last_prediction_time = datetime.now(timezone.utc).isoformat()
    state.total_predictions += 1

    return MonitorResponse(
        request_id=request_id,
        payment_tier=result["payment_tier"],
        payment_pd=result["payment_pd"],
        flag_count=result["flag_count"],
        flags=result["flags"],
        loan_id=result.get("loan_id", "unknown"),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    tags=["Health"],
)
async def health_check():
    """Return basic health status, uptime, and prediction count."""
    from pipeline import VERSION as PIPELINE_VERSION

    pipeline_loaded = state.pipeline is not None
    uptime = time.time() - state.startup_time if state.startup_time else 0

    models_loaded = {
        "pipeline": pipeline_loaded,
        "fraud_gate": False,
        "default_scorecard": False,
        "credit_grader": False,
        "payment_monitor": False,
    }

    if pipeline_loaded:
        models_loaded["fraud_gate"] = state.pipeline.fraud_gate is not None
        models_loaded["default_scorecard"] = state.pipeline.default_scorecard is not None
        models_loaded["credit_grader"] = state.pipeline.credit_grader is not None
        models_loaded["payment_monitor"] = state.pipeline.payment_monitor is not None

    status = "healthy" if pipeline_loaded else "degraded"

    return HealthResponse(
        status=status,
        api_version=API_VERSION,
        pipeline_version=PIPELINE_VERSION if pipeline_loaded else "not loaded",
        models_loaded=models_loaded,
        uptime_seconds=round(uptime, 2),
        last_prediction_time=state.last_prediction_time,
        total_predictions=state.total_predictions,
    )


@app.get(
    "/health/models",
    response_model=ModelHealthResponse,
    summary="Detailed model component health",
    tags=["Health"],
)
async def model_health():
    """Return detailed version and configuration info for each model component."""
    from pipeline import VERSION as PIPELINE_VERSION

    pipeline = state.pipeline

    # FraudGate details
    fraud_info: Dict[str, Any] = {"loaded": False}
    if pipeline and pipeline.fraud_gate:
        fg = pipeline.fraud_gate
        fraud_info = {
            "loaded": True,
            "version": getattr(fg, "VERSION", "unknown"),
            "threshold_profile": getattr(fg, "threshold_profile", "unknown"),
            "has_graph_lookup": getattr(fg, "_graph_lookup", None) is not None,
        }

    # DefaultScorecard details
    default_info: Dict[str, Any] = {"loaded": False}
    if pipeline and pipeline.default_scorecard:
        ds = pipeline.default_scorecard
        default_info = {
            "loaded": True,
            "version": getattr(ds, "VERSION", "unknown"),
            "blend_weights": getattr(ds, "blend_weights", {}),
            "n_features": len(getattr(ds, "feature_names_", [])) if hasattr(ds, "feature_names_") else "unknown",
        }

    # CreditGrader details
    grader_info: Dict[str, Any] = {"loaded": False}
    if pipeline and pipeline.credit_grader:
        cg = pipeline.credit_grader
        grader_info = {
            "loaded": True,
            "version": getattr(cg, "VERSION", "unknown"),
            "scheme": getattr(cg, "scheme", "unknown"),
            "grades": getattr(cg, "grades", []),
            "boundaries": [
                round(float(b), 4)
                for b in (
                    getattr(cg, "boundaries", None)
                    if getattr(cg, "boundaries", None) is not None
                    else []
                )
            ],
        }

    # PaymentMonitor details
    monitor_info: Dict[str, Any] = {"loaded": False, "available": False}
    if pipeline and pipeline.payment_monitor:
        pm = pipeline.payment_monitor
        monitor_info = {
            "loaded": True,
            "available": True,
            "version": getattr(pm, "VERSION", "unknown"),
            "n_flags": len(getattr(pm, "flag_definitions", {})) if hasattr(pm, "flag_definitions") else 9,
        }

    # Pipeline-level config
    pipeline_info: Dict[str, Any] = {"loaded": pipeline is not None}
    if pipeline:
        pipeline_info.update({
            "version": PIPELINE_VERSION,
            "pd_decline_threshold": pipeline.pd_decline,
            "pd_review_threshold": pipeline.pd_review,
            "review_grades": sorted(pipeline.review_grades),
            "fit_timestamp": pipeline._fit_timestamp,
            "train_stats": pipeline._train_stats,
        })

    return ModelHealthResponse(
        fraud_gate=fraud_info,
        default_scorecard=default_info,
        credit_grader=grader_info,
        payment_monitor=monitor_info,
        pipeline=pipeline_info,
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions -- return structured JSON."""
    request_id = _get_request_id(request)
    logger.exception("Unhandled exception for request %s", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": "Internal server error",
            "detail": f"{type(exc).__name__}: {exc}",
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "scripts.models.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
