"""
explainability.py
-----------------
Model explainability using SHAP:

  - Global summary plot  (bar + beeswarm)
  - Per-applicant waterfall / force plot
  - ``explain_prediction()``  → sorted list of (feature, shap_value) tuples
    used by the UI to generate human-readable reasons

Gracefully degrades if shap is not installed.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import REPORTS_DIR, RANDOM_SEED
from logger import get_logger

log = get_logger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    log.warning("shap not installed. Explainability features will be unavailable.")


# ═══════════════════════════════════════════════════════════════════════════════
# Explainer factory
# ═══════════════════════════════════════════════════════════════════════════════

def _get_explainer(model, X_background: np.ndarray):
    """Return the most appropriate SHAP explainer for the given model."""
    model_type = type(model).__name__

    tree_types = (
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
        "XGBClassifier",
        "LGBMClassifier",
        "ExtraTreesClassifier",
    )

    if model_type in tree_types:
        return shap.TreeExplainer(model)
    else:
        # Use a small background sample for KernelExplainer (slow but general)
        background = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(model.predict_proba, background)


# ═══════════════════════════════════════════════════════════════════════════════
# Global SHAP plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_shap_summary(
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> None:
    """Generate and save SHAP beeswarm + bar summary plots.

    Parameters
    ----------
    model         : fitted estimator
    X_train       : training array (used for background + computing values)
    feature_names : column names matching X_train columns
    max_samples   : cap sample count to keep it fast
    """
    if not _HAS_SHAP:
        log.warning("shap not available – skipping SHAP summary plots.")
        print("  [SHAP] skipped (install shap: pip install shap)")
        return

    log.info("Computing global SHAP values …")
    print("  [SHAP] Computing global values …", end=" ", flush=True)

    idx = np.random.default_rng(RANDOM_SEED).choice(
        len(X_train), size=min(max_samples, len(X_train)), replace=False
    )
    X_sample = X_train[idx]

    try:
        explainer   = _get_explainer(model, X_train)
        shap_values = explainer.shap_values(X_sample)

        # TreeExplainer returns list[array] for multi-class; take class-1 values
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        # ── Beeswarm (summary) ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Summary – Feature Impact on Approval", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path_bee = REPORTS_DIR / "13_shap_beeswarm.png"
        plt.savefig(path_bee, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  [Plot] {path_bee}")

        # ── Bar (mean |SHAP|) ───────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            sv, X_sample,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
        )
        plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path_bar = REPORTS_DIR / "14_shap_importance.png"
        plt.savefig(path_bar, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] {path_bar}")

        log.info("Global SHAP plots saved.")

    except Exception as e:
        log.error("SHAP global plot failed: %s", e)
        print(f"  [SHAP] warning: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Per-applicant explanation
# ═══════════════════════════════════════════════════════════════════════════════

def explain_prediction(
    model,
    X_instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
) -> list[tuple[str, float]]:
    """Compute SHAP values for a single applicant and return sorted feature impacts.

    Parameters
    ----------
    model         : fitted estimator
    X_instance    : shape (1, n_features)
    X_background  : training array used as background for KernelExplainer
    feature_names : list of feature column names

    Returns
    -------
    list of (feature_name, shap_value) sorted by abs(shap_value) descending.
    An empty list is returned if shap is not available.
    """
    if not _HAS_SHAP:
        return []

    try:
        explainer   = _get_explainer(model, X_background)
        shap_values = explainer.shap_values(X_instance)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]          # class-1 shap for first (only) sample
        else:
            sv = shap_values[0]

        pairs = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return pairs

    except Exception as e:
        log.error("SHAP explain_prediction failed: %s", e)
        return []


def plot_waterfall(
    model,
    X_instance: np.ndarray,
    X_background: np.ndarray,
    feature_names: list[str],
    applicant_id: str = "Applicant",
) -> plt.Figure | None:
    """Generate a SHAP waterfall plot for a single prediction.

    Returns a matplotlib Figure (caller must close it), or None on failure.
    """
    if not _HAS_SHAP:
        return None

    try:
        explainer   = _get_explainer(model, X_background)
        shap_values = explainer(X_instance)

        # Handle multi-output
        if shap_values.values.ndim == 3:
            vals     = shap_values.values[0, :, 1]
            base_val = shap_values.base_values[0, 1]
        else:
            vals     = shap_values.values[0]
            base_val = shap_values.base_values[0]

        exp = shap.Explanation(
            values      = vals,
            base_values = base_val,
            data        = X_instance[0],
            feature_names = feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(exp, show=False)
        plt.title(f"SHAP Waterfall — {applicant_id}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    except Exception as e:
        log.error("SHAP waterfall failed: %s", e)
        return None
