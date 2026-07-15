"""
train.py
--------
Full ML pipeline orchestrator. Auto-detects cloud environment and
uses lightweight settings to complete training within Streamlit Cloud
free-tier limits (~3 min).
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.metrics         import f1_score, accuracy_score

from config          import (
    DATA_PATH, REPORTS_DIR, RANDOM_SEED, TEST_SIZE,
    METADATA_PATH, COMPARISON_PATH, MODEL_PATH,
)
from logger           import get_logger
from generate_dataset import generate_dataset
from preprocessing    import run_eda, Preprocessor, apply_smote, select_features
from models           import (
    train_all_models, evaluate_all_models,
    plot_comparison, plot_confusion_matrices, plot_roc_curves,
    tune_model,
)
from predictor        import CreditCardPredictor
from utils            import save_metadata

log = get_logger(__name__)

# ── Detect cloud environment ──────────────────────────────────────────────────
IS_CLOUD = (
    os.environ.get("HOME", "").startswith("/home/adminuser")   # Streamlit Cloud
    or os.environ.get("STREAMLIT_SHARING_MODE") == "true"
    or os.environ.get("IS_CLOUD", "false").lower() == "true"
)

CLOUD_SETTINGS = {
    "n_samples"   : 2000,   # smaller dataset
    "tuning_iter" : 8,      # fewer RandomizedSearchCV iterations
    "skip_shap"   : True,   # skip slow SHAP plots
    "skip_models" : ["LightGBM", "Voting Ensemble"],  # skip heaviest models
}

LOCAL_SETTINGS = {
    "n_samples"   : 6000,
    "tuning_iter" : 20,
    "skip_shap"   : False,
    "skip_models" : [],
}

SETTINGS = CLOUD_SETTINGS if IS_CLOUD else LOCAL_SETTINGS


def main() -> None:
    t_start = time.perf_counter()

    print("\n" + "█"*65)
    print("   CREDIT CARD APPROVAL PREDICTION — TRAINING PIPELINE v2")
    if IS_CLOUD:
        print("   [CLOUD MODE] Using lightweight settings for fast training")
    print("█"*65)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    # Always regenerate on cloud to avoid stale v1 data
    if not DATA_PATH.exists() or IS_CLOUD:
        log.info("Generating dataset  n=%d …", SETTINGS["n_samples"])
        df = generate_dataset(n_samples=SETTINGS["n_samples"])
    else:
        df = pd.read_csv(DATA_PATH)
        # Regenerate if missing new features
        expected_cols = {"credit_utilization", "savings_balance", "marital_status"}
        if not expected_cols.issubset(df.columns):
            log.info("Old dataset detected — regenerating with new features …")
            df = generate_dataset(n_samples=SETTINGS["n_samples"])
        else:
            log.info("Dataset loaded  rows=%d", len(df))
            print(f"\n[Data] Loaded {len(df):,} rows from {DATA_PATH}")

    # ── 2. EDA (skip slow pairplot on cloud) ──────────────────────────────────
    if IS_CLOUD:
        # Run only the two fastest EDA plots on cloud
        import matplotlib.pyplot as _plt
        counts = df["approved"].value_counts().sort_index()
        fig, ax = _plt.subplots(figsize=(5, 3))
        ax.bar(["Rejected", "Approved"], counts.values, color=["#E74C3C", "#2ECC71"])
        ax.set_title("Target Distribution")
        fig.savefig(str(REPORTS_DIR / "01_target_distribution.png"), dpi=100, bbox_inches="tight")
        _plt.close(fig)
        print("  [EDA] Fast plot done (cloud mode).")
    else:
        run_eda(df)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PREPROCESSING")
    print("="*65)
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    print(f"  Processed shape : X={X.shape}, y={y.shape}")

    # ── 4. Feature Selection ──────────────────────────────────────────────────
    select_features(X, y, preprocessor.feature_names, top_n=10)

    # ── 5. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"\n  Train : {len(X_train):,}   Test : {len(X_test):,}")

    # ── 6. SMOTE ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SMOTE OVERSAMPLING")
    print("="*65)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── 7. Train Models ───────────────────────────────────────────────────────
    # On cloud, skip heavy models to save time
    skip = SETTINGS["skip_models"]
    models_all = train_all_models(X_train_bal, y_train_bal)
    models = {k: v for k, v in models_all.items() if k not in skip}

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    results_df = evaluate_all_models(models, X_test, y_test)
    plot_comparison(results_df)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)

    results_df.to_csv(COMPARISON_PATH, index=False)
    print(f"\n  [Saved] {COMPARISON_PATH}")

    # ── 9. Tune Best Model ────────────────────────────────────────────────────
    best_name   = results_df.iloc[0]["Model"]
    tune_target = best_name if best_name != "Voting Ensemble" else results_df.iloc[1]["Model"]

    # Override tuning iterations for cloud
    import config as _cfg
    _orig_iter = _cfg.TUNING_ITER
    _cfg.TUNING_ITER = SETTINGS["tuning_iter"]

    tuned    = tune_model(tune_target, X_train_bal, y_train_bal)
    _cfg.TUNING_ITER = _orig_iter  # restore

    y_pred_tuned = tuned.predict(X_test)
    tuned_f1     = f1_score(y_test, y_pred_tuned)
    tuned_acc    = accuracy_score(y_test, y_pred_tuned)

    base_f1     = float(results_df[results_df["Model"] == tune_target]["F1-Score"].iloc[0])
    final_model = tuned if tuned_f1 >= base_f1 else models[tune_target]
    final_name  = tune_target
    final_f1    = max(tuned_f1, base_f1)
    print(f"  Final model: {final_name}  (F1={final_f1:.4f})")

    # ── 10. SHAP (skip on cloud) ──────────────────────────────────────────────
    if not SETTINGS["skip_shap"]:
        from explainability import plot_shap_summary
        print("\n" + "="*65)
        print("  SHAP GLOBAL EXPLAINABILITY")
        print("="*65)
        plot_shap_summary(final_model, X_train_bal, preprocessor.feature_names)
    else:
        print("\n  [SHAP] Skipped in cloud mode.")

    # ── 11. Save Artefacts ────────────────────────────────────────────────────
    predictor = CreditCardPredictor(final_model, preprocessor, X_train_bal)
    predictor.save()

    elapsed = round(time.perf_counter() - t_start, 1)

    metadata = {
        "model_name"      : final_name,
        "f1_score"        : round(final_f1, 4),
        "accuracy"        : round(tuned_acc, 4),
        "features"        : preprocessor.feature_names,
        "n_train"         : int(len(X_train_bal)),
        "n_test"          : int(len(X_test)),
        "training_time_s" : elapsed,
        "smote_applied"   : True,
        "cloud_mode"      : IS_CLOUD,
        "all_models"      : results_df.to_dict(orient="records"),
    }
    save_metadata(metadata, METADATA_PATH)

    # ── Smoke Test ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SMOKE TEST")
    print("="*65)
    test_app = {
        "age": 35, "income": 75000, "employment_status": "Employed",
        "credit_score": 700, "years_employed": 8, "existing_loans": 1,
        "loan_amount": 5000, "repayment_history": "Good", "education": "Bachelor",
        "num_credit_cards": 2, "debt_to_income": 0.07, "credit_utilization": 0.20,
        "monthly_expenses": 2500, "savings_balance": 15000,
        "months_since_last_default": 0, "marital_status": "Married",
        "housing_status": "Rent",
    }
    r = predictor.predict(test_app)
    print(f"  Test prediction → {r['decision']} (prob={r['probability']:.2%})")

    print("\n" + "█"*65)
    print(f"  PIPELINE COMPLETE  ({elapsed}s)")
    print("█"*65)


if __name__ == "__main__":
    main()
