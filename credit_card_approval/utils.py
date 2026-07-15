"""
utils.py
--------
Shared utility functions used across the project:
  - Input validation
  - Applicant dict → DataFrame conversion
  - Risk scoring helpers
  - Report helpers
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import VALID_RANGES, VALID_CATEGORIES, ALL_FEATURES, NUMERICAL_COLS, CATEGORICAL_COLS
from logger import get_logger

log = get_logger(__name__)


# ── Input Validation ─────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when applicant input fails validation."""


def validate_applicant(data: dict) -> dict:
    """Validate and coerce a raw applicant dict.

    Checks
    ------
    - All required fields are present (missing ones filled with NaN, a warning is logged).
    - Numerical fields are within allowed ranges.
    - Categorical fields contain allowed values.

    Returns
    -------
    dict  Cleaned applicant dict safe to pass to ``Preprocessor.transform``.

    Raises
    ------
    ValidationError  If a value is out of range or an unrecognised category is given.
    """
    cleaned: dict[str, Any] = {}
    errors: list[str] = []

    # ── Numerical fields ──────────────────────────────────────────────────
    for col in NUMERICAL_COLS:
        raw = data.get(col, None)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            log.warning("Field '%s' missing — will be imputed.", col)
            cleaned[col] = np.nan
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be numeric, got '{raw}'.")
            continue

        lo, hi = VALID_RANGES[col]
        if not (lo <= val <= hi):
            errors.append(f"'{col}' = {val} is out of range [{lo}, {hi}].")
        else:
            cleaned[col] = val

    # ── Categorical fields ────────────────────────────────────────────────
    for col in CATEGORICAL_COLS:
        raw = data.get(col, None)
        if raw is None:
            log.warning("Field '%s' missing — will be imputed.", col)
            cleaned[col] = np.nan
            continue
        val = str(raw).strip()
        allowed = VALID_CATEGORIES[col]
        if val not in allowed:
            errors.append(f"'{col}' = '{val}' not in allowed values {allowed}.")
        else:
            cleaned[col] = val

    if errors:
        raise ValidationError("Validation failed:\n  • " + "\n  • ".join(errors))

    return cleaned


def applicant_to_df(applicant: dict) -> pd.DataFrame:
    """Convert a validated applicant dict to a single-row DataFrame."""
    row = {col: applicant.get(col, np.nan) for col in ALL_FEATURES}
    return pd.DataFrame([row])


# ── Risk Classification ───────────────────────────────────────────────────────

def probability_to_risk(prob: float) -> dict:
    """Map approval probability to a risk tier with colour and description.

    Returns
    -------
    dict with keys: level, colour, description
    """
    if prob >= 0.80:
        return {"level": "Low",       "colour": "#28a745", "description": "Excellent credit profile. Very likely to be approved."}
    elif prob >= 0.60:
        return {"level": "Medium",    "colour": "#ffc107", "description": "Decent profile. Approval likely with minor concerns."}
    elif prob >= 0.40:
        return {"level": "High",      "colour": "#fd7e14", "description": "Borderline profile. Several risk factors present."}
    else:
        return {"level": "Very High", "colour": "#dc3545", "description": "High-risk profile. Multiple disqualifying factors."}


# ── Rejection / Approval Reasons ─────────────────────────────────────────────

def build_decision_reasons(applicant: dict, decision: str, shap_values: list | None = None) -> list[str]:
    """Generate human-readable reasons for the decision.

    Uses SHAP values when available; falls back to rule-based heuristics.

    Parameters
    ----------
    applicant   : validated applicant dict
    decision    : "Approved" or "Rejected"
    shap_values : list of (feature_name, shap_value) tuples sorted by |shap_value|

    Returns
    -------
    list[str]  Up to 5 plain-English reason strings.
    """
    reasons: list[str] = []

    if shap_values:
        # Use SHAP: top drivers in the direction of the decision
        sign = 1 if decision == "Approved" else -1
        drivers = [(f, v) for f, v in shap_values if v * sign > 0][:3]
        for feat, val in drivers:
            reasons.append(_shap_to_sentence(feat, val, decision))
    else:
        # Rule-based fallback
        cs  = applicant.get("credit_score", 650)
        dti = applicant.get("debt_to_income", 0.3)
        rh  = applicant.get("repayment_history", "Average")
        emp = applicant.get("employment_status", "Employed")
        cu  = applicant.get("credit_utilization", 0.5)

        if decision == "Approved":
            if cs and cs >= 700:    reasons.append(f"Strong credit score ({cs:.0f})")
            if dti and dti < 0.30:  reasons.append(f"Low debt-to-income ratio ({dti:.0%})")
            if rh == "Good":        reasons.append("Excellent repayment history")
            if emp == "Employed":   reasons.append("Stable employment status")
            if cu and cu < 0.30:    reasons.append(f"Low credit utilization ({cu:.0%})")
        else:
            if cs and cs < 600:     reasons.append(f"Low credit score ({cs:.0f} < 600)")
            if dti and dti > 0.50:  reasons.append(f"High debt-to-income ratio ({dti:.0%})")
            if rh == "Poor":        reasons.append("Poor repayment history")
            if emp == "Unemployed": reasons.append("No stable income source")
            if cu and cu > 0.70:    reasons.append(f"High credit utilization ({cu:.0%})")

    return reasons[:5] if reasons else ["Insufficient data to generate detailed reasons."]


def _shap_to_sentence(feature: str, shap_val: float, decision: str) -> str:
    direction = "positively" if shap_val > 0 else "negatively"
    label = feature.replace("_", " ").title()
    return f"{label} {direction} influenced the {decision} decision (impact: {abs(shap_val):.3f})"


# ── JSON Metadata ─────────────────────────────────────────────────────────────

def save_metadata(meta: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log.info("Metadata saved → %s", path)


def load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── CSV Batch Helper ──────────────────────────────────────────────────────────

def load_batch_csv(file_obj) -> tuple[pd.DataFrame, list[str]]:
    """Load an uploaded CSV file and return (df, list_of_warnings).

    Fills missing columns with NaN and warns about extra columns.
    """
    df = pd.read_csv(file_obj)
    warnings: list[str] = []

    missing_cols = [c for c in ALL_FEATURES if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in ALL_FEATURES and c != "approved"]

    if missing_cols:
        warnings.append(f"Missing columns (will be imputed): {missing_cols}")
        for col in missing_cols:
            df[col] = np.nan

    if extra_cols:
        warnings.append(f"Extra columns ignored: {extra_cols}")

    return df[ALL_FEATURES], warnings
