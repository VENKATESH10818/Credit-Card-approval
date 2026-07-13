"""
predictor.py
------------
CreditCardPredictor — the single prediction interface used by both the
FastAPI backend and the Streamlit UI.

  predictor.predict(applicant_dict)         → result dict
  predictor.predict_batch(list_of_dicts)    → list of result dicts
  predictor.explain(applicant_dict)         → SHAP-based reason list
  predictor.save() / CreditCardPredictor.load()
"""

from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd
import joblib

from config import MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH
from utils  import validate_applicant, applicant_to_df, probability_to_risk, build_decision_reasons
from logger import get_logger

log = get_logger(__name__)


class CreditCardPredictor:
    """Wraps the trained model + preprocessor into one predict-able object.

    Usage
    -----
    After training::

        predictor = CreditCardPredictor(model, preprocessor, X_train)
        predictor.save()

    Later::

        predictor = CreditCardPredictor.load()
        result    = predictor.predict({...})
    """

    def __init__(self, model=None, preprocessor=None, X_train: np.ndarray | None = None):
        self.model        = model
        self.preprocessor = preprocessor
        self.X_train      = X_train     # kept for SHAP background samples

    # ──────────────────────────────────────────────────────────────────────────
    # Single prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, applicant: dict, validate: bool = True) -> dict:
        """Predict credit card approval for one applicant.

        Parameters
        ----------
        applicant : dict  Raw or pre-validated applicant data.
        validate  : bool  Whether to run input validation (default True).

        Returns
        -------
        dict with keys:
            decision     : "Approved" | "Rejected"
            probability  : float  (0–1)
            risk         : dict   {level, colour, description}
            reasons      : list[str]
            latency_ms   : float  inference time in milliseconds
        """
        self._check_loaded()
        t0 = time.perf_counter()

        if validate:
            applicant = validate_applicant(applicant)

        df = applicant_to_df(applicant)
        X  = self.preprocessor.transform(df)

        prob     = float(self.model.predict_proba(X)[0, 1])
        decision = "Approved" if prob >= 0.50 else "Rejected"
        risk     = probability_to_risk(prob)

        # Build reasons (SHAP when available, rule-based fallback)
        shap_pairs = self._shap_pairs(X)
        reasons    = build_decision_reasons(applicant, decision, shap_pairs)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info("Prediction: %s (prob=%.4f, risk=%s, %.1f ms)", decision, prob, risk["level"], latency_ms)

        return {
            "decision"   : decision,
            "probability": round(prob, 4),
            "risk"       : risk,
            "reasons"    : reasons,
            "latency_ms" : latency_ms,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Batch prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict_batch(self, applicants: list[dict], validate: bool = True) -> pd.DataFrame:
        """Predict for a list/DataFrame of applicants.

        Parameters
        ----------
        applicants : list[dict] or pd.DataFrame

        Returns
        -------
        pd.DataFrame with original fields + decision / probability / risk_level / reasons columns.
        """
        self._check_loaded()

        if isinstance(applicants, pd.DataFrame):
            records = applicants.to_dict(orient="records")
        else:
            records = applicants

        results = []
        for i, app in enumerate(records):
            try:
                if validate:
                    app = validate_applicant(app)
                r = self.predict(app, validate=False)
                r["row"] = i
                results.append(r)
            except Exception as e:
                log.warning("Row %d failed validation: %s", i, e)
                results.append({
                    "row": i, "decision": "Error", "probability": None,
                    "risk": {"level": "Unknown", "colour": "#95A5A6", "description": str(e)},
                    "reasons": [str(e)], "latency_ms": 0,
                })

        # Build summary DataFrame
        rows = []
        for r in results:
            rows.append({
                "decision"   : r["decision"],
                "probability": r.get("probability"),
                "risk_level" : r["risk"]["level"],
                "reasons"    : " | ".join(r.get("reasons", [])),
            })

        out_df = pd.DataFrame(records).copy()
        meta_df = pd.DataFrame(rows)
        return pd.concat([out_df.reset_index(drop=True), meta_df], axis=1)

    # ──────────────────────────────────────────────────────────────────────────
    # SHAP explanation helper
    # ──────────────────────────────────────────────────────────────────────────

    def _shap_pairs(self, X_instance: np.ndarray) -> list[tuple[str, float]]:
        """Return SHAP-based (feature, value) pairs or empty list."""
        try:
            from explainability import explain_prediction
            if self.X_train is not None:
                return explain_prediction(
                    self.model, X_instance, self.X_train,
                    self.preprocessor.feature_names,
                )
        except Exception as e:
            log.debug("SHAP skipped: %s", e)
        return []

    def explain(self, applicant: dict) -> list[tuple[str, float]]:
        """Public method: return full SHAP explanation for one applicant."""
        applicant = validate_applicant(applicant)
        X         = self.preprocessor.transform(applicant_to_df(applicant))
        return self._shap_pairs(X)

    # ──────────────────────────────────────────────────────────────────────────
    # Persist / load
    # ──────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        joblib.dump(self.model,        MODEL_PATH)
        joblib.dump(self.preprocessor, PREPROCESSOR_PATH)
        if self.X_train is not None:
            joblib.dump(self.X_train, MODEL_PATH.parent / "X_train_background.joblib")
        log.info("Model saved → %s", MODEL_PATH)
        log.info("Preprocessor saved → %s", PREPROCESSOR_PATH)
        print(f"  [Saved] Model        → {MODEL_PATH}")
        print(f"  [Saved] Preprocessor → {PREPROCESSOR_PATH}")

    @classmethod
    def load(cls) -> "CreditCardPredictor":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run  python train.py  first."
            )
        model        = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        bg_path = MODEL_PATH.parent / "X_train_background.joblib"
        X_train = joblib.load(bg_path) if bg_path.exists() else None

        log.info("Model loaded ← %s", MODEL_PATH)
        return cls(model, preprocessor, X_train)

    # ──────────────────────────────────────────────────────────────────────────

    def _check_loaded(self) -> None:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not loaded. Call CreditCardPredictor.load().")
