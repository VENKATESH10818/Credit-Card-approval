"""
preprocessing.py
----------------
Full preprocessing pipeline:
  1. EDA  – summary stats + 6 diagnostic plots saved to reports/
  2. Preprocessor class  – impute → encode → scale  (fit / transform)
  3. SMOTE oversampling  – balance the training set
  4. Feature selection   – Random Forest importances + correlation

Dependencies: scikit-learn, imbalanced-learn, seaborn, matplotlib
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute          import SimpleImputer
from imblearn.over_sampling  import SMOTE

from config import (
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL,
    REPORTS_DIR, RANDOM_SEED, SMOTE_STRATEGY,
)
from logger import get_logger

log = get_logger(__name__)

# ── Plot palette ──────────────────────────────────────────────────────────────
PALETTE = {"approved": "#2ECC71", "rejected": "#E74C3C"}
sns.set_theme(style="whitegrid", palette="muted")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EDA
# ═══════════════════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame) -> None:
    """Print summary statistics and save diagnostic plots to reports/."""
    log.info("Running EDA …")
    print("\n" + "="*65)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*65)

    n, p = df.shape
    approved_pct = df[TARGET_COL].mean() * 100
    print(f"\n  Rows: {n:,}  |  Columns: {p}  |  Approval rate: {approved_pct:.1f}%")

    print("\n── Missing Values ──")
    miss = df.isnull().sum()
    miss_pct = miss / n * 100
    miss_df = pd.DataFrame({"Count": miss, "Pct (%)": miss_pct.round(2)})[miss > 0]
    print(miss_df.to_string() if not miss_df.empty else "  None")

    print("\n── Numerical Summary ──")
    print(df[NUMERICAL_COLS].describe().round(2).to_string())

    print("\n── Categorical Value Counts ──")
    for col in CATEGORICAL_COLS:
        print(f"\n  {col}:\n{df[col].value_counts(dropna=False).to_string()}")

    _plot_target_distribution(df)
    _plot_numerical_distributions(df)
    _plot_categorical_vs_target(df)
    _plot_correlation_heatmap(df)
    _plot_boxplots(df)
    _plot_pairplot_key_features(df)

    log.info("EDA complete.")


def _save(fig: plt.Figure, fname: str) -> None:
    path = REPORTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.debug("Plot saved → %s", path)
    print(f"  [Plot] {path}")


def _plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Rejected", "Approved"]
    colors = ["#E74C3C", "#2ECC71"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(ax1.patches, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 str(val), ha="center", fontweight="bold")
    ax1.set_title("Approval Count", fontsize=12)
    ax1.set_ylabel("Count")

    ax2.pie(counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=140, wedgeprops=dict(edgecolor="white"))
    ax2.set_title("Approval Distribution", fontsize=12)

    fig.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_target_distribution.png")


def _plot_numerical_distributions(df: pd.DataFrame) -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(NUMERICAL_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_COLS):
        for val, color, label in [(1, "#2ECC71", "Approved"), (0, "#E74C3C", "Rejected")]:
            data = df.loc[df[TARGET_COL] == val, col].dropna()
            axes[i].hist(data, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
        axes[i].set_title(col, fontsize=10)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions by Approval Status", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_numerical_distributions.png")


def _plot_categorical_vs_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(CATEGORICAL_COLS), figsize=(20, 5))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, hue=TARGET_COL, order=order, ax=ax,
                      palette={0: "#E74C3C", 1: "#2ECC71"})
        ax.set_title(f"{col}", fontsize=11)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Approved", labels=["No", "Yes"], fontsize=8)
    fig.suptitle("Categorical Features vs Approval Status", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_categorical_vs_target.png")


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num_df = df[NUMERICAL_COLS + [TARGET_COL]].dropna()
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_correlation_heatmap.png")


def _plot_boxplots(df: pd.DataFrame) -> None:
    n_cols = 4
    n_rows = int(np.ceil(len(NUMERICAL_COLS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_COLS):
        data = [
            df.loc[df[TARGET_COL] == 0, col].dropna(),
            df.loc[df[TARGET_COL] == 1, col].dropna(),
        ]
        bp = axes[i].boxplot(data, labels=["Rejected", "Approved"],
                             patch_artist=True, notch=True)
        bp["boxes"][0].set_facecolor("#E74C3C")
        bp["boxes"][1].set_facecolor("#2ECC71")
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        axes[i].set_title(col, fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Approval Status (Box Plots)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "05_boxplots.png")


def _plot_pairplot_key_features(df: pd.DataFrame) -> None:
    """Pairplot of the 4 most predictive numerical features."""
    key = ["credit_score", "income", "debt_to_income", "credit_utilization", TARGET_COL]
    sample = df[key].dropna().sample(min(800, len(df)), random_state=RANDOM_SEED)
    sample[TARGET_COL] = sample[TARGET_COL].map({0: "Rejected", 1: "Approved"})

    g = sns.pairplot(sample, hue=TARGET_COL,
                     palette={"Approved": "#2ECC71", "Rejected": "#E74C3C"},
                     plot_kws={"alpha": 0.4, "s": 20})
    g.figure.suptitle("Pairplot – Key Features", y=1.02, fontsize=13, fontweight="bold")
    _save(g.figure, "06_pairplot_key_features.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Preprocessor
# ═══════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    """Impute → Encode → Scale pipeline.

    Call ``fit_transform(df)`` on the training DataFrame (including target column).
    Call ``transform(df)`` on new data (target column optional).
    """

    def __init__(self):
        self.num_imputer     = SimpleImputer(strategy="median")
        self.cat_imputer     = SimpleImputer(strategy="most_frequent")
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler          = StandardScaler()
        self.feature_names: list[str] = []
        self._fitted         = False

    # ── fit_transform ────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit all transformers and return (X_processed, y)."""
        df = df.copy()
        y  = df.pop(TARGET_COL).values.astype(int)

        log.info("Fitting preprocessor …")

        # Impute
        df[NUMERICAL_COLS]   = self.num_imputer.fit_transform(df[NUMERICAL_COLS])
        df[CATEGORICAL_COLS] = self.cat_imputer.fit_transform(df[CATEGORICAL_COLS])

        # Encode
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.feature_names = NUMERICAL_COLS + CATEGORICAL_COLS
        X = df[self.feature_names].values.astype(float)
        X = self.scaler.fit_transform(X)
        self._fitted = True

        log.info("Preprocessor fitted. Shape: X=%s", X.shape)
        return X, y

    # ── transform ────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")
        df = df.copy()
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        # Ensure all expected columns exist
        for col in NUMERICAL_COLS + CATEGORICAL_COLS:
            if col not in df.columns:
                df[col] = np.nan

        df[NUMERICAL_COLS]   = self.num_imputer.transform(df[NUMERICAL_COLS])
        df[CATEGORICAL_COLS] = self.cat_imputer.transform(df[CATEGORICAL_COLS])

        for col in CATEGORICAL_COLS:
            le = self.label_encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )

        return self.scaler.transform(df[self.feature_names].values.astype(float))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SMOTE Oversampling
# ═══════════════════════════════════════════════════════════════════════════════

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the training set using SMOTE.

    Returns
    -------
    X_resampled, y_resampled
    """
    unique, counts = np.unique(y_train, return_counts=True)
    log.info("Before SMOTE: %s", dict(zip(unique, counts)))
    print(f"  Before SMOTE: {dict(zip(unique.tolist(), counts.tolist()))}")

    smote = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_SEED, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    unique2, counts2 = np.unique(y_res, return_counts=True)
    log.info("After  SMOTE: %s", dict(zip(unique2, counts2)))
    print(f"  After  SMOTE: {dict(zip(unique2.tolist(), counts2.tolist()))}")

    return X_res, y_res


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Feature Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> list[str]:
    """Rank features by Random Forest importance and save a bar chart.

    Returns
    -------
    list[str]  Top *top_n* feature names.
    """
    log.info("Running feature selection …")
    rf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)

    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    print("\n── Feature Importances (Random Forest) ──")
    print(imp.round(4).to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    colors  = ["#E74C3C" if i < top_n else "#BDC3C7" for i in range(len(imp))]
    imp.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Feature Importances — Random Forest", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    _save(fig, "07_feature_importances.png")

    top = imp.head(top_n).index.tolist()
    log.info("Top %d features: %s", top_n, top)
    print(f"\n  Top {top_n} features: {top}")
    return top
