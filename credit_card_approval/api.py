"""
api.py
------
FastAPI backend for the Credit Card Approval Prediction system.

Endpoints
---------
  GET  /                    Health check
  GET  /health              Detailed health + model metadata
  POST /predict             Single applicant prediction
  POST /predict/batch       Batch prediction (JSON array)
  POST /predict/csv         Batch prediction (CSV file upload)
  GET  /model/info          Model metadata (name, F1, features …)
  GET  /model/comparison    All-model performance comparison table

Run:
    uvicorn api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import io
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from config import COMPARISON_PATH, METADATA_PATH, VALID_RANGES, VALID_CATEGORIES
from logger import get_logger
from utils  import load_metadata, load_batch_csv
from predictor import CreditCardPredictor

log = get_logger("api")

# ── Global predictor (loaded once at startup) ─────────────────────────────────
_predictor: CreditCardPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts; release on shutdown."""
    global _predictor
    log.info("API starting – loading model …")
    try:
        _predictor = CreditCardPredictor.load()
        log.info("Model loaded successfully.")
    except FileNotFoundError as e:
        log.error("Model not found: %s  –  run python train.py first.", e)
    yield
    log.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Approval Prediction API",
    description=(
        "Production-grade REST API for credit card application decisions. "
        "Powered by an XGBoost / LightGBM ensemble with SHAP explainability."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ────────────────────────────────────────────────

class ApplicantIn(BaseModel):
    """Single applicant input schema with validation."""

    age:                        int   = Field(..., ge=18,      le=75,       description="Applicant age (18–75)")
    income:                     float = Field(..., ge=10_000,  le=500_000,  description="Annual income in USD")
    employment_status:          str   = Field(...,                          description="Employed | Self-Employed | Unemployed | Retired")
    credit_score:               int   = Field(..., ge=300,     le=850,      description="FICO credit score (300–850)")
    years_employed:             float = Field(..., ge=0,       le=50,       description="Years in current/last job")
    existing_loans:             int   = Field(..., ge=0,       le=10,       description="Number of active loans")
    loan_amount:                float = Field(..., ge=0,       le=200_000,  description="Total outstanding loan balance (USD)")
    repayment_history:          str   = Field(...,                          description="Good | Average | Poor")
    education:                  str   = Field(...,                          description="High School | Bachelor | Master | PhD")
    num_credit_cards:           int   = Field(..., ge=0,       le=15,       description="Number of credit cards held")
    debt_to_income:             float = Field(..., ge=0.0,     le=1.0,      description="Debt-to-income ratio (0–1)")
    credit_utilization:         float = Field(..., ge=0.0,     le=1.0,      description="Credit utilization ratio (0–1)")
    monthly_expenses:           float = Field(..., ge=500,     le=20_000,   description="Monthly expenses in USD")
    savings_balance:            float = Field(..., ge=0,       le=1_000_000,description="Total savings / liquid assets (USD)")
    months_since_last_default:  int   = Field(..., ge=0,       le=120,      description="Months since last default (0 = never defaulted)")
    marital_status:             str   = Field(...,                          description="Single | Married | Divorced | Widowed")
    housing_status:             str   = Field(...,                          description="Own | Rent | Mortgage | Other")

    @field_validator("employment_status")
    @classmethod
    def validate_employment(cls, v: str) -> str:
        allowed = VALID_CATEGORIES["employment_status"]
        if v not in allowed:
            raise ValueError(f"employment_status must be one of {allowed}")
        return v

    @field_validator("repayment_history")
    @classmethod
    def validate_repayment(cls, v: str) -> str:
        allowed = VALID_CATEGORIES["repayment_history"]
        if v not in allowed:
            raise ValueError(f"repayment_history must be one of {allowed}")
        return v

    @field_validator("education")
    @classmethod
    def validate_education(cls, v: str) -> str:
        allowed = VALID_CATEGORIES["education"]
        if v not in allowed:
            raise ValueError(f"education must be one of {allowed}")
        return v

    @field_validator("marital_status")
    @classmethod
    def validate_marital(cls, v: str) -> str:
        allowed = VALID_CATEGORIES["marital_status"]
        if v not in allowed:
            raise ValueError(f"marital_status must be one of {allowed}")
        return v

    @field_validator("housing_status")
    @classmethod
    def validate_housing(cls, v: str) -> str:
        allowed = VALID_CATEGORIES["housing_status"]
        if v not in allowed:
            raise ValueError(f"housing_status must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_dti_consistency(self) -> "ApplicantIn":
        if self.existing_loans == 0 and self.loan_amount > 0:
            raise ValueError("loan_amount must be 0 when existing_loans is 0.")
        return self


class PredictionOut(BaseModel):
    decision:    str
    probability: float
    risk_level:  str
    risk_colour: str
    risk_description: str
    reasons:     list[str]
    latency_ms:  float


class HealthOut(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str | None
    version:      str


# ── Middleware: request logging ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    log.info("%s %s  →  %d  (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


# ── Exception handler ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _require_model() -> CreditCardPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run python train.py first.",
        )
    return _predictor


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Credit Card Approval Prediction API v2.0", "docs": "/docs"}


@app.get("/health", response_model=HealthOut, tags=["Health"])
async def health():
    meta = load_metadata(METADATA_PATH)
    return HealthOut(
        status       = "ok" if _predictor is not None else "degraded",
        model_loaded = _predictor is not None,
        model_name   = meta.get("model_name"),
        version      = "2.0.0",
    )


@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
async def predict_single(applicant: ApplicantIn):
    """Predict credit card approval for a single applicant."""
    predictor = _require_model()
    data      = applicant.model_dump()

    result = predictor.predict(data, validate=False)   # Pydantic already validated

    return PredictionOut(
        decision         = result["decision"],
        probability      = result["probability"],
        risk_level       = result["risk"]["level"],
        risk_colour      = result["risk"]["colour"],
        risk_description = result["risk"]["description"],
        reasons          = result["reasons"],
        latency_ms       = result["latency_ms"],
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(applicants: list[ApplicantIn]):
    """Predict for a list of applicants (JSON array, max 1000)."""
    if len(applicants) > 1000:
        raise HTTPException(status_code=400, detail="Batch limit is 1,000 applicants per request.")

    predictor = _require_model()
    records   = [a.model_dump() for a in applicants]
    result_df = predictor.predict_batch(records, validate=False)

    return {
        "count"  : len(result_df),
        "results": result_df[["decision", "probability", "risk_level", "reasons"]].to_dict(orient="records"),
    }


@app.post("/predict/csv", tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV file and get batch predictions back as JSON.

    The CSV must contain the same columns as the single-predict schema.
    Missing columns will be imputed; extra columns are ignored.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    predictor = _require_model()

    try:
        contents = await file.read()
        df, warnings = load_batch_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) > 5000:
        raise HTTPException(status_code=400, detail="CSV batch limit is 5,000 rows per request.")

    result_df = predictor.predict_batch(df, validate=True)

    return {
        "count"    : len(result_df),
        "warnings" : warnings,
        "results"  : result_df[["decision", "probability", "risk_level", "reasons"]].to_dict(orient="records"),
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return stored metadata about the trained model."""
    meta = load_metadata(METADATA_PATH)
    if not meta:
        raise HTTPException(status_code=404, detail="Model metadata not found. Run python train.py.")
    return meta


@app.get("/model/comparison", tags=["Model"])
async def model_comparison():
    """Return the full model performance comparison table."""
    if not COMPARISON_PATH.exists():
        raise HTTPException(status_code=404, detail="Comparison results not found. Run python train.py.")
    df = pd.read_csv(COMPARISON_PATH)
    return {
        "models": df.to_dict(orient="records"),
        "best"  : df.iloc[0]["Model"],
    }
