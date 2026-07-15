"""
config.py
---------
Central configuration for the entire application.
All paths, model parameters, and feature definitions live here.
"""

import os
from pathlib import Path

# ── Directory Layout ────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── File Paths ───────────────────────────────────────────────────────────────
DATA_PATH        = DATA_DIR / "credit_card_applications.csv"
MODEL_PATH       = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METADATA_PATH    = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH  = REPORTS_DIR / "model_comparison.csv"
LOG_FILE         = LOGS_DIR / "app.log"

# ── Dataset ──────────────────────────────────────────────────────────────────
N_SAMPLES   = 6000
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# ── Feature Definitions ──────────────────────────────────────────────────────
NUMERICAL_COLS = [
    "age",
    "income",
    "credit_score",
    "years_employed",
    "existing_loans",
    "loan_amount",
    "num_credit_cards",
    "debt_to_income",
    "credit_utilization",      # NEW
    "monthly_expenses",         # NEW
    "savings_balance",          # NEW
    "months_since_last_default",# NEW
]

CATEGORICAL_COLS = [
    "employment_status",
    "repayment_history",
    "education",
    "marital_status",           # NEW
    "housing_status",           # NEW
]

TARGET_COL = "approved"

ALL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS

# ── Valid Input Ranges (used for validation) ──────────────────────────────────
VALID_RANGES = {
    "age"                       : (18, 75),
    "income"                    : (10_000, 500_000),
    "credit_score"              : (300, 850),
    "years_employed"            : (0, 50),
    "existing_loans"            : (0, 10),
    "loan_amount"               : (0, 200_000),
    "num_credit_cards"          : (0, 15),
    "debt_to_income"            : (0.0, 1.0),
    "credit_utilization"        : (0.0, 1.0),
    "monthly_expenses"          : (500, 20_000),
    "savings_balance"           : (0, 1_000_000),
    "months_since_last_default" : (0, 120),
}

VALID_CATEGORIES = {
    "employment_status" : ["Employed", "Self-Employed", "Unemployed", "Retired"],
    "repayment_history" : ["Good", "Average", "Poor"],
    "education"         : ["High School", "Bachelor", "Master", "PhD"],
    "marital_status"    : ["Single", "Married", "Divorced", "Widowed"],
    "housing_status"    : ["Own", "Rent", "Mortgage", "Other"],
}

# ── Model Training ────────────────────────────────────────────────────────────
CV_FOLDS        = 5
TUNING_ITER     = 20          # RandomizedSearchCV iterations
SMOTE_STRATEGY  = "auto"
SCORING_METRIC  = "f1"
