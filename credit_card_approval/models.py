"""
models.py
---------
Defines, trains, evaluates, and tunes 7 classifiers:

  Baseline
  --------
  1. Logistic Regression
  2. Decision Tree

  Ensemble / Boosting
  -------------------
  3. Random Forest
  4. Gradient Boosting
  5. XGBoost
  6. LightGBM

  Meta / Stacking
  ---------------
  7. Voting Ensemble  (LR + RF + XGB soft-vote)

Public API
----------
  train_all_models(X, y)              → dict[name, fitted_model]
  evaluate_all_models(models, X, y)   → pd.DataFrame of metrics
  tune_model(name, X, y)              → best_estimator
  plot_comparison(results_df)
  plot_confusion_matrices(models, X, y)
  plot_roc_curves(models, X, y)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
    roc_curve,
)
from scipy.stats import randint, uniform

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from config import CV_FOLDS, TUNING_ITER, SCORING_METRIC, REPORTS_DIR, RANDOM_SEED
from logger import get_logger

log = get_logger(__name__)

CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

def _build_base_models() -> dict:
    models: dict = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_SEED,
            class_weight="balanced", n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=RANDOM_SEED,
        ),
    }

    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=6, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_SEED,
            scale_pos_weight=1, n_jobs=-1,
        )
    else:
        log.warning("xgboost not installed – skipping XGBoost model.")

    if _HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=6, random_state=RANDOM_SEED,
            class_weight="balanced", n_jobs=-1,
            verbose=-1,
        )
    else:
        log.warning("lightgbm not installed – skipping LightGBM model.")

    # Voting ensemble uses whichever strong models are available
    voters = [
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1)),
    ]
    if _HAS_XGB:
        voters.append(("xgb", XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, n_jobs=-1,
        )))

    models["Voting Ensemble"] = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Fit all models and return a dict of {name: fitted_estimator}."""
    models = _build_base_models()
    fitted: dict = {}

    print("\n" + "="*65)
    print("  MODEL TRAINING")
    print("="*65)

    for name, model in models.items():
        log.info("Training: %s", name)
        print(f"\n  [{name}] training …", end=" ", flush=True)
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV, scoring=SCORING_METRIC, n_jobs=-1,
        )
        print(f"done  |  CV {SCORING_METRIC.upper()} = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        log.info("%s → CV %s = %.4f ± %.4f", name, SCORING_METRIC, cv_scores.mean(), cv_scores.std())
        fitted[name] = model

    return fitted


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_all_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compute metrics for every model and return a comparison DataFrame."""
    print("\n" + "="*65)
    print("  MODEL EVALUATION  (held-out test set)")
    print("="*65)

    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba") else None
        )

        row = {
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred), 4),
            "F1-Score" : round(f1_score(y_test, y_pred), 4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else 0.0,
        }
        rows.append(row)

        print(f"\n  ── {name} ──")
        print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    results_df = (
        pd.DataFrame(rows)
        .sort_values("F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "="*65)
    print("  PERFORMANCE SUMMARY (sorted by F1-Score)")
    print("="*65)
    print(results_df.to_string(index=False))

    log.info("Evaluation complete. Best model: %s", results_df.iloc[0]["Model"])
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Hyperparameter Tuning  (RandomizedSearchCV)
# ═══════════════════════════════════════════════════════════════════════════════

# Parameter distributions for RandomizedSearchCV
_PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "Logistic Regression": {
        "C"      : uniform(0.001, 100),
        "penalty": ["l2"],
        "solver" : ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth"        : [None, 4, 6, 8, 12, 16],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf" : randint(1, 10),
        "criterion"        : ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators"     : randint(100, 500),
        "max_depth"        : [None, 10, 20, 30],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf" : randint(1, 5),
        "max_features"     : ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators"  : randint(100, 400),
        "learning_rate" : uniform(0.01, 0.3),
        "max_depth"     : randint(3, 8),
        "subsample"     : uniform(0.6, 0.4),
        "min_samples_leaf": randint(1, 10),
    },
    "XGBoost": {
        "n_estimators"   : randint(100, 400),
        "learning_rate"  : uniform(0.01, 0.3),
        "max_depth"      : randint(3, 10),
        "subsample"      : uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma"          : uniform(0, 0.5),
        "reg_alpha"      : uniform(0, 1),
        "reg_lambda"     : uniform(0, 1),
    },
    "LightGBM": {
        "n_estimators"  : randint(100, 400),
        "learning_rate" : uniform(0.01, 0.3),
        "max_depth"     : randint(3, 10),
        "num_leaves"    : randint(20, 100),
        "subsample"     : uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha"     : uniform(0, 1),
        "reg_lambda"    : uniform(0, 1),
    },
}


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """RandomizedSearchCV for the given model.  Returns best fitted estimator."""
    base_models = _build_base_models()

    if model_name not in base_models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(base_models.keys())}")

    param_dist = _PARAM_DISTRIBUTIONS.get(model_name)
    if not param_dist:
        log.warning("No param distribution for '%s'. Returning base model.", model_name)
        base_models[model_name].fit(X_train, y_train)
        return base_models[model_name]

    print("\n" + "="*65)
    print(f"  HYPERPARAMETER TUNING  →  {model_name}")
    print("="*65)
    print(f"  Iterations : {TUNING_ITER}   |   CV folds : {CV_FOLDS}")

    search = RandomizedSearchCV(
        base_models[model_name],
        param_distributions=param_dist,
        n_iter=TUNING_ITER,
        cv=CV,
        scoring=SCORING_METRIC,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best params : {search.best_params_}")
    print(f"  Best CV F1  : {search.best_score_:.4f}")
    log.info("Tuning done. Best CV %s = %.4f  params=%s",
             SCORING_METRIC, search.best_score_, search.best_params_)

    _plot_tuning_results(search, model_name)
    return search.best_estimator_


def _plot_tuning_results(search: RandomizedSearchCV, model_name: str) -> None:
    cv_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=True)
        .tail(20)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(len(cv_df)),
        cv_df["mean_test_score"].values,
        xerr=cv_df["std_test_score"].values,
        color="#3498DB", alpha=0.8, edgecolor="white",
    )
    labels = [str(p) for p in cv_df["params"].values]
    ax.set_yticks(range(len(cv_df)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Mean CV F1-Score")
    ax.set_title(f"Tuning — {model_name} (Top 20 combos)", fontsize=12, fontweight="bold")
    ax.axvline(search.best_score_, color="red", linewidth=1.5, linestyle="--", label=f"Best: {search.best_score_:.4f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f"09_tuning_{model_name.lower().replace(' ', '_')}.png"
    path  = REPORTS_DIR / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.debug("Tuning plot saved → %s", path)
    print(f"  [Plot] {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Visualisations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing Accuracy / Precision / Recall / F1 / ROC-AUC."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.copy()
    for m in metrics:
        plot_df[m] = pd.to_numeric(plot_df[m], errors="coerce")

    x     = np.arange(len(plot_df))
    width = 0.14
    colors = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, plot_df[metric], width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(plot_df["Model"], rotation=15, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = REPORTS_DIR / "10_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {path}")


def plot_confusion_matrices(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Grid of confusion matrices for all models."""
    n     = len(models)
    n_col = min(3, n)
    n_row = int(np.ceil(n / n_col))

    fig, axes = plt.subplots(n_row, n_col, figsize=(6 * n_col, 5 * n_row))
    axes = np.array(axes).flatten()

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        disp   = ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = REPORTS_DIR / "11_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {path}")


def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Overlaid ROC curves for all models that support predict_proba."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]

    for (name, model), color in zip(models.items(), colors):
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc  = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "12_roc_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {path}")
