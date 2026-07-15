"""
generate_dataset.py
-------------------
Generates a realistic synthetic credit card applicant dataset with
17 features (12 numerical + 5 categorical) and saves it to
data/credit_card_applications.csv.

New real-world features vs. v1
-------------------------------
  credit_utilization        – ratio of used credit to total credit limit
  monthly_expenses          – estimated monthly spending (USD)
  savings_balance           – total savings / liquid assets (USD)
  months_since_last_default – 0 if no default history
  marital_status            – Single / Married / Divorced / Widowed
  housing_status            – Own / Rent / Mortgage / Other

Run standalone:
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd

from config import (
    DATA_PATH, N_SAMPLES, RANDOM_SEED,
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL,
)
from logger import get_logger

log = get_logger(__name__)


def generate_dataset(
    n_samples: int = N_SAMPLES,
    save_path=DATA_PATH,
) -> pd.DataFrame:
    """Generate synthetic applicant data and persist it.

    Parameters
    ----------
    n_samples : int
    save_path : str | Path

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(RANDOM_SEED)
    log.info("Generating synthetic dataset  n=%d …", n_samples)

    # ── Demographics ────────────────────────────────────────────────────────
    age = rng.integers(18, 76, size=n_samples)

    income = np.round(
        rng.lognormal(mean=10.9, sigma=0.55, size=n_samples).clip(10_000, 500_000), 2
    )

    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired"],
        size=n_samples, p=[0.55, 0.20, 0.15, 0.10],
    )

    marital_status = rng.choice(
        ["Single", "Married", "Divorced", "Widowed"],
        size=n_samples, p=[0.35, 0.45, 0.15, 0.05],
    )

    housing_status = rng.choice(
        ["Own", "Rent", "Mortgage", "Other"],
        size=n_samples, p=[0.25, 0.35, 0.32, 0.08],
    )

    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n_samples, p=[0.28, 0.42, 0.22, 0.08],
    )

    # ── Credit Profile ───────────────────────────────────────────────────────
    # Credit score: positively correlated with income & education
    edu_bonus = np.where(education == "PhD", 40,
                np.where(education == "Master", 25,
                np.where(education == "Bachelor", 10, 0)))

    base_score = (income / 500_000) * 380 + 300 + edu_bonus
    credit_score = np.clip(
        np.round(base_score + rng.normal(0, 65, n_samples)).astype(int), 300, 850
    )

    years_employed = np.where(
        employment_status == "Unemployed", 0.0,
        np.round(rng.uniform(0, 38, n_samples), 1),
    )

    existing_loans = rng.integers(0, 8, size=n_samples)

    loan_amount = np.round(
        np.where(
            existing_loans == 0, 0.0,
            rng.uniform(500, 150_000, n_samples),
        ), 2
    )

    repayment_history = rng.choice(
        ["Good", "Average", "Poor"],
        size=n_samples, p=[0.50, 0.30, 0.20],
    )

    num_credit_cards = rng.integers(0, 12, size=n_samples)

    # ── New Real-World Features ───────────────────────────────────────────────
    # Credit utilization: lower is better; positively skewed
    credit_utilization = np.round(
        np.clip(rng.beta(a=2, b=5, size=n_samples), 0.0, 1.0), 4
    )

    # Monthly expenses: correlated with income
    monthly_expenses = np.round(
        np.clip(income / 12 * rng.uniform(0.3, 0.85, n_samples), 500, 20_000), 2
    )

    # Savings balance: log-normal, correlated with income
    savings_balance = np.round(
        np.clip(
            rng.lognormal(mean=9.5, sigma=1.2, size=n_samples),
            0, 1_000_000
        ), 2
    )

    # Months since last default: 0 = no default; otherwise 1-120 months ago
    has_default = rng.random(n_samples) < 0.18
    months_since_last_default = np.where(
        has_default, rng.integers(1, 121, size=n_samples), 0
    )

    # Derived: debt-to-income
    debt_to_income = np.round(
        np.clip(loan_amount / np.where(income == 0, 1, income), 0, 1), 4
    )

    # ── Approval Score (rule-based + Gaussian noise) ─────────────────────────
    score = np.zeros(n_samples, dtype=float)

    # Credit score  (0–1 contribution, weight 0.28)
    score += (credit_score - 300) / 550 * 0.28

    # Income        (log-normalised, weight 0.16)
    score += np.log1p(income) / np.log1p(500_000) * 0.16

    # Repayment history (weight 0.18)
    score += np.vectorize({"Good": 0.18, "Average": 0.09, "Poor": 0.00}.get)(repayment_history)

    # Employment     (weight 0.12)
    score += np.vectorize(
        {"Employed": 0.12, "Self-Employed": 0.08, "Retired": 0.06, "Unemployed": 0.00}.get
    )(employment_status)

    # Debt-to-income penalty (weight -0.10)
    score -= debt_to_income * 0.10

    # Credit utilization penalty (weight -0.08)
    score -= credit_utilization * 0.08

    # Default history penalty
    score -= np.where(months_since_last_default > 0,
                      0.10 * np.exp(-months_since_last_default / 36), 0)

    # Savings bonus (weight 0.06)
    score += np.log1p(savings_balance) / np.log1p(1_000_000) * 0.06

    # Housing stability bonus
    score += np.vectorize(
        {"Own": 0.04, "Mortgage": 0.03, "Rent": 0.01, "Other": 0.00}.get
    )(housing_status)

    # Noise
    score += rng.normal(0, 0.04, n_samples)
    score  = np.clip(score, 0, 1)

    approved = (score >= 0.50).astype(int)

    # ── Inject ~5 % Missing Values ────────────────────────────────────────────
    def _inject_missing(arr, frac=0.05):
        mask = rng.random(n_samples) < frac
        result = arr.astype(object)
        result[mask] = np.nan
        return result

    income_m              = _inject_missing(income)
    credit_score_m        = _inject_missing(credit_score)
    employment_status_m   = _inject_missing(np.array(employment_status, dtype=object))
    repayment_history_m   = _inject_missing(np.array(repayment_history, dtype=object))
    credit_utilization_m  = _inject_missing(credit_utilization)
    savings_balance_m     = _inject_missing(savings_balance)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "age"                       : age,
        "income"                    : income_m,
        "employment_status"         : employment_status_m,
        "credit_score"              : credit_score_m,
        "years_employed"            : years_employed,
        "existing_loans"            : existing_loans,
        "loan_amount"               : loan_amount,
        "repayment_history"         : repayment_history_m,
        "education"                 : education,
        "num_credit_cards"          : num_credit_cards,
        "debt_to_income"            : debt_to_income,
        "credit_utilization"        : credit_utilization_m,
        "monthly_expenses"          : monthly_expenses,
        "savings_balance"           : savings_balance_m,
        "months_since_last_default" : months_since_last_default,
        "marital_status"            : marital_status,
        "housing_status"            : housing_status,
        TARGET_COL                  : approved,
    })

    os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
    df.to_csv(str(save_path), index=False)

    approval_rate = df[TARGET_COL].mean() * 100
    log.info(
        "Dataset saved → %s  (%d rows, %.1f%% approval rate)",
        save_path, len(df), approval_rate,
    )
    print(f"[Dataset] {len(df):,} rows | Approval rate: {approval_rate:.1f}% | Saved → {save_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
