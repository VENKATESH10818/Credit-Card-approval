"""
train.py
--------
Full ML pipeline orchestrator:

  1.  Load / generate dataset
  2.  EDA
  3.  Preprocess  (impute → encode → scale)
  4.  Feature selection
  5.  Train / test split
  6.  Apply SMOTE on training set
  7.  Train all 7 models
  8.  Evaluate & compare
  9.  Hyperparameter-tune the best model
  10. SHAP global explanation plots
  11. Save model artefacts + metadata

Run:
    python train.py
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

# ── Local modules ─────────────────────────────────────────────────────────────
from config          import (
    DATA_PATH, REPORTS_DIR, RANDOM_SEED, TEST_SIZE,
    METADATA_PATH, COMPARISON_PATH, MODEL_PATH,
)
from logger          import get_logger
from generate_dataset import generate_dataset
from preprocessing   import run_eda, Preprocessor, apply_smote, select_features
from models          import (
    train_all_models, evaluate_all_models,
    plot_comparison, plot_confusion_matrices, plot_roc_curves,
    tune_model,
)
from explainability  import plot_shap_summary
from predictor       import CreditCardPredictor
from utils           import save_metadata

log = get_logger(__name__)


def main() -> None:
    t_start = time.perf_counter()

    print("\n" + "█"*65)
    print("   CREDIT CARD APPROVAL PREDICTION — TRAINING PIPELINE v2")
    print("█"*65)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        log.info("Dataset not found. Generating …")
        df = generate_dataset()
    else:
        df = pd.read_csv(DATA_PATH)
        log.info("Dataset loaded  rows=%d", len(df))
        print(f"\n[Data] Loaded {len(df):,} rows from {DATA_PATH}")

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    run_eda(df)

    # ── 3. Preprocessing ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PREPROCESSING")
    print("="*65)
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    print(f"  Processed shape : X={X.shape}, y={y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class balance   : {dict(zip(unique.tolist(), counts.tolist()))}")

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
    models = train_all_models(X_train_bal, y_train_bal)

    # ── 8. Evaluate ───────────────────────────────────────────────────────────
    results_df = evaluate_all_models(models, X_test, y_test)
    plot_comparison(results_df)
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)

    results_df.to_csv(COMPARISON_PATH, index=False)
    print(f"\n  [Saved] {COMPARISON_PATH}")

    # ── 9. Tune Best Model ────────────────────────────────────────────────────
    best_name = results_df.iloc[0]["Model"]
    print(f"\n  Best model (F1): {best_name}")

    # Don't tune Voting Ensemble — tune its best constituent instead
    tune_target = best_name if best_name != "Voting Ensemble" else results_df.iloc[1]["Model"]
    tuned = tune_model(tune_target, X_train_bal, y_train_bal)

    # Re-evaluate tuned model
    y_pred_tuned = tuned.predict(X_test)
    tuned_f1     = f1_score(y_test, y_pred_tuned)
    tuned_acc    = accuracy_score(y_test, y_pred_tuned)

    print(f"\n  Tuned {tune_target}  →  F1={tuned_f1:.4f}  Acc={tuned_acc:.4f}")
    log.info("Tuned %s  F1=%.4f  Acc=%.4f", tune_target, tuned_f1, tuned_acc)

    # Use tuned model only if it improves F1
    base_f1 = float(results_df[results_df["Model"] == tune_target]["F1-Score"].iloc[0])
    final_model = tuned if tuned_f1 >= base_f1 else models[tune_target]
    final_name  = tune_target
    final_f1    = max(tuned_f1, base_f1)
    print(f"  Final model: {final_name}  (F1={final_f1:.4f})")

    # ── 10. SHAP Plots ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SHAP GLOBAL EXPLAINABILITY")
    print("="*65)
    plot_shap_summary(final_model, X_train_bal, preprocessor.feature_names)

    # ── 11. Save Artefacts ────────────────────────────────────────────────────
    predictor = CreditCardPredictor(final_model, preprocessor, X_train_bal)
    predictor.save()

    elapsed = round(time.perf_counter() - t_start, 1)

    metadata = {
        "model_name"       : final_name,
        "f1_score"         : round(final_f1, 4),
        "accuracy"         : round(tuned_acc, 4),
        "features"         : preprocessor.feature_names,
        "n_train"          : int(len(X_train_bal)),
        "n_test"           : int(len(X_test)),
        "training_time_s"  : elapsed,
        "smote_applied"    : True,
        "all_models"       : results_df.to_dict(orient="records"),
    }
    save_metadata(metadata, METADATA_PATH)

    # ── Smoke Test ────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SMOKE TEST — Sample Predictions")
    print("="*65)
    smoke_applicants = [
        {   # Strong
            "age": 45, "income": 130_000, "employment_status": "Employed",
            "credit_score": 790, "years_employed": 14, "existing_loans": 0,
            "loan_amount": 0, "repayment_history": "Good", "education": "Master",
            "num_credit_cards": 3, "debt_to_income": 0.0, "credit_utilization": 0.12,
            "monthly_expenses": 3500, "savings_balance": 85_000,
            "months_since_last_default": 0, "marital_status": "Married",
            "housing_status": "Own",
        },
        {   # Weak
            "age": 23, "income": 18_000, "employment_status": "Unemployed",
            "credit_score": 325, "years_employed": 0, "existing_loans": 4,
            "loan_amount": 22_000, "repayment_history": "Poor", "education": "High School",
            "num_credit_cards": 1, "debt_to_income": 0.88, "credit_utilization": 0.92,
            "monthly_expenses": 2000, "savings_balance": 200,
            "months_since_last_default": 6, "marital_status": "Single",
            "housing_status": "Rent",
        },
    ]

    for i, app in enumerate(smoke_applicants, 1):
        r = predictor.predict(app)
        print(f"\n  Applicant #{i}  →  {r['decision']}  "
              f"(prob={r['probability']:.2%}, risk={r['risk']['level']}, "
              f"{r['latency_ms']} ms)")
        for reason in r["reasons"]:
            print(f"    • {reason}")

    print("\n" + "█"*65)
    print(f"  PIPELINE COMPLETE  ({elapsed}s)")
    print("█"*65)
    print(f"\n  Reports  → {REPORTS_DIR}")
    print(f"  Models   → {MODEL_PATH.parent}")
    print(f"\n  Launch app:")
    print(f"    streamlit run app.py")
    print(f"  Launch API:")
    print(f"    uvicorn api:app --reload --port 8000")


if __name__ == "__main__":
    main()
