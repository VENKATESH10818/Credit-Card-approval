"""
app.py
------
Production-grade Streamlit UI for Credit Card Approval Prediction.

Pages (sidebar navigation)
---------------------------
  1. Single Prediction   – form with sliders/dropdowns, instant decision
  2. Batch Predictions   – CSV upload, downloadable results table
  3. Model Dashboard     – performance comparison, all training plots
  4. About               – project info, API docs link

Run:
    streamlit run app.py
"""

import os, sys, io, time, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config     import COMPARISON_PATH, METADATA_PATH, REPORTS_DIR, VALID_CATEGORIES, MODEL_PATH
from utils      import load_metadata, load_batch_csv
from logger     import get_logger

log = get_logger("streamlit_app")


# ── Auto-train if no model exists (Streamlit Cloud compatibility) ─────────────
def _auto_train() -> None:
    """Run the full training pipeline automatically when no model is found.
    This makes the app self-contained on Streamlit Cloud without needing
    a separate train step.
    """
    import subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    st.info("⏳ No trained model found. Training now — this takes 2–4 minutes on first run…")
    progress = st.progress(0, text="Starting training pipeline…")

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            progress.progress(100, text="Training complete!")
            st.success("✅ Model trained successfully! Refreshing…")
            st.rerun()
        else:
            progress.empty()
            st.error("Training failed. See details below.")
            st.code(result.stderr[-3000:] if result.stderr else result.stdout[-3000:])
    except subprocess.TimeoutExpired:
        progress.empty()
        st.error("Training timed out (>10 min). Try a smaller dataset or upgrade your deployment plan.")
    except Exception as e:
        progress.empty()
        st.error(f"Training error: {e}")


from predictor import CreditCardPredictor

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CreditAI — Approval Prediction",
    page_icon  = "💳",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #F0F4F8; }
[data-testid="stSidebar"]          { background: #1A237E; }
[data-testid="stSidebar"] * { color: #E8EAF6 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #C5CAE9 !important; }

/* ── Cards ── */
.card {
    background: white; border-radius: 12px; padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08); margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg,#1565C0,#1E88E5);
    color: white; border-radius: 12px; padding: 20px;
    text-align: center; box-shadow: 0 4px 15px rgba(21,101,192,.3);
}
.metric-card h1 { font-size:2.4rem; margin:0; }
.metric-card p  { margin:0; opacity:.85; font-size:.95rem; }

/* ── Decision banners ── */
.approved-banner {
    background: linear-gradient(135deg,#1B5E20,#2E7D32);
    color: white; border-radius: 12px; padding: 24px;
    text-align: center; box-shadow: 0 4px 15px rgba(46,125,50,.4);
}
.rejected-banner {
    background: linear-gradient(135deg,#B71C1C,#C62828);
    color: white; border-radius: 12px; padding: 24px;
    text-align: center; box-shadow: 0 4px 15px rgba(198,40,40,.4);
}
.approved-banner h1, .rejected-banner h1 { font-size:2rem; margin:0; }
.approved-banner p,  .rejected-banner p  { margin:4px 0; opacity:.9; }

/* ── Risk badges ── */
.badge-low       { background:#2E7D32; color:white; padding:5px 14px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-medium    { background:#F57F17; color:white; padding:5px 14px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-high      { background:#E65100; color:white; padding:5px 14px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-very-high { background:#B71C1C; color:white; padding:5px 14px; border-radius:20px; font-weight:700; display:inline-block; }

/* ── Reason items ── */
.reason-item {
    background:#F8F9FA; border-left:4px solid #1565C0;
    padding:10px 14px; border-radius:0 8px 8px 0; margin:6px 0;
    font-size:.93rem;
}
.reason-positive { border-left-color:#2E7D32; }
.reason-negative { border-left-color:#C62828; }

/* ── Section headers ── */
.section-header {
    font-size:1.3rem; font-weight:700; color:#1A237E;
    border-bottom:2px solid #1565C0; padding-bottom:6px; margin-bottom:16px;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_predictor():
    return CreditCardPredictor.load()
# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_badge_html(level: str) -> str:
    css = {
        "Low": "badge-low", "Medium": "badge-medium",
        "High": "badge-high", "Very High": "badge-very-high",
    }.get(level, "badge-medium")
    return f'<span class="{css}">{level} Risk</span>'


def gauge_chart(probability: float, decision: str) -> plt.Figure:
    """Half-donut probability gauge."""
    fig, ax = plt.subplots(figsize=(3.8, 2.2), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("white")

    theta_bg   = np.linspace(np.pi, 0, 200)
    theta_fill = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    color      = "#2E7D32" if decision == "Approved" else "#C62828"

    ax.plot(np.cos(theta_bg),   np.sin(theta_bg),   color="#E8EAF6", linewidth=18)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color,     linewidth=18)

    ax.text(0, -0.05, f"{probability:.1%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color=color)
    ax.text(0, -0.42, "Approval Probability", ha="center", va="center",
            fontsize=8, color="#78909C")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.3); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def styled_reasons(reasons: list[str], decision: str) -> None:
    css_class = "reason-positive" if decision == "Approved" else "reason-negative"
    icon      = "✅" if decision == "Approved" else "⚠️"
    for r in reasons:
        st.markdown(f'<div class="reason-item {css_class}">{icon} {r}</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
def page_single_prediction(predictor: CreditCardPredictor) -> None:
    st.markdown('<p class="section-header">💳 Single Applicant Prediction</p>', unsafe_allow_html=True)

    # ── Form ─────────────────────────────────────────────────────────────────
    with st.form("applicant_form"):
        st.markdown("#### 👤 Personal Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age            = st.slider("Age", 18, 75, 35)
            marital_status = st.selectbox("Marital Status", VALID_CATEGORIES["marital_status"])
        with c2:
            income         = st.number_input("Annual Income ($)", 10_000, 500_000, 60_000, step=1_000)
            housing_status = st.selectbox("Housing Status", VALID_CATEGORIES["housing_status"])
        with c3:
            education         = st.selectbox("Education", VALID_CATEGORIES["education"])
            employment_status = st.selectbox("Employment Status", VALID_CATEGORIES["employment_status"])

        st.markdown("---")
        st.markdown("#### 💰 Financial Profile")
        c4, c5, c6 = st.columns(3)
        with c4:
            credit_score   = st.slider("Credit Score", 300, 850, 680)
            years_employed = st.slider("Years Employed", 0.0, 50.0, 5.0, 0.5)
        with c5:
            existing_loans    = st.selectbox("Existing Loans", list(range(11)), index=1)
            loan_amount       = st.number_input("Outstanding Loan ($)", 0, 200_000, 5_000, step=500)
            repayment_history = st.selectbox("Repayment History", VALID_CATEGORIES["repayment_history"])
        with c6:
            num_credit_cards  = st.slider("Credit Cards Held", 0, 15, 2)
            monthly_expenses  = st.number_input("Monthly Expenses ($)", 500, 20_000, 2_500, step=100)
            savings_balance   = st.number_input("Savings Balance ($)", 0, 1_000_000, 10_000, step=500)

        st.markdown("---")
        st.markdown("#### 📊 Credit Behaviour")
        c7, c8, c9 = st.columns(3)
        with c7:
            credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.30, 0.01,
                                           help="Ratio of used credit to total limit")
        with c8:
            debt_to_income = round(loan_amount / income if income > 0 else 0.0, 4)
            st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2%}",
                      help="Auto-calculated from income and loan amount")
        with c9:
            months_since_last_default = st.slider("Months Since Last Default", 0, 120, 0,
                                                  help="0 means no default history")

        submitted = st.form_submit_button("🔍 Predict Approval", use_container_width=True, type="primary")

    # ── Result ────────────────────────────────────────────────────────────────
    if submitted:
        applicant = dict(
            age=age, income=float(income), employment_status=employment_status,
            credit_score=credit_score, years_employed=float(years_employed),
            existing_loans=existing_loans, loan_amount=float(loan_amount),
            repayment_history=repayment_history, education=education,
            num_credit_cards=num_credit_cards, debt_to_income=debt_to_income,
            credit_utilization=float(credit_utilization),
            monthly_expenses=float(monthly_expenses),
            savings_balance=float(savings_balance),
            months_since_last_default=months_since_last_default,
            marital_status=marital_status, housing_status=housing_status,
        )

        with st.spinner("Analyzing application…"):
            result = predictor.predict(applicant, validate=True)

        decision    = result["decision"]
        probability = result["probability"]
        risk        = result["risk"]
        reasons     = result["reasons"]

        st.markdown("---")
        r1, r2, r3 = st.columns([1.4, 1, 1])

        with r1:
            banner_class = "approved-banner" if decision == "Approved" else "rejected-banner"
            icon         = "✅" if decision == "Approved" else "❌"
            st.markdown(f"""
            <div class="{banner_class}">
                <h1>{icon} {decision}</h1>
                <p>Confidence: {probability:.1%}</p>
                <p style="font-size:.85rem;opacity:.8">Processed in {result['latency_ms']} ms</p>
            </div>""", unsafe_allow_html=True)

        with r2:
            fig = gauge_chart(probability, decision)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        with r3:
            st.markdown(f"**Risk Assessment**")
            st.markdown(risk_badge_html(risk["level"]), unsafe_allow_html=True)
            st.markdown(f"<small>{risk['description']}</small>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📋 Decision Reasons")
        styled_reasons(reasons, decision)

        # Applicant summary table
        st.markdown("---")
        st.markdown("#### 📄 Submitted Application Summary")
        summary_data = {
            "Age": str(age), "Income": f"${income:,}",
            "Employment": employment_status, "Credit Score": str(credit_score),
            "Years Employed": str(years_employed), "Existing Loans": str(existing_loans),
            "Outstanding Debt": f"${loan_amount:,}", "Repayment History": repayment_history,
            "Education": education, "Credit Cards": str(num_credit_cards),
            "Credit Utilization": f"{credit_utilization:.1%}",
            "Debt-to-Income": f"{debt_to_income:.2%}",
            "Monthly Expenses": f"${monthly_expenses:,}",
            "Savings Balance": f"${savings_balance:,}",
            "Months Since Default": str(months_since_last_default),
            "Marital Status": marital_status, "Housing": housing_status,
        }
        rows = [{"Field": k, "Value": v} for k, v in summary_data.items()]
        half = len(rows) // 2
        col_a, col_b = st.columns(2)
        with col_a:
            st.dataframe(pd.DataFrame(rows[:half]), hide_index=True, use_container_width=True)
        with col_b:
            st.dataframe(pd.DataFrame(rows[half:]), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Batch Predictions
# ═══════════════════════════════════════════════════════════════════════════════
def page_batch_predictions(predictor: CreditCardPredictor) -> None:
    st.markdown('<p class="section-header">📂 Batch Predictions</p>', unsafe_allow_html=True)

    st.info(
        "Upload a CSV file with applicant data. Required columns match the single-prediction form. "
        "Missing columns will be auto-imputed. Download predictions as CSV when done."
    )

    # Template download
    template_cols = [
        "age","income","employment_status","credit_score","years_employed",
        "existing_loans","loan_amount","repayment_history","education",
        "num_credit_cards","debt_to_income","credit_utilization","monthly_expenses",
        "savings_balance","months_since_last_default","marital_status","housing_status",
    ]
    sample_rows = [
        [35,75000,"Employed",720,8,1,5000,"Good","Bachelor",2,0.07,0.20,2500,15000,0,"Married","Rent"],
        [28,32000,"Self-Employed",580,4,2,12000,"Average","High School",1,0.38,0.55,1800,3000,14,"Single","Rent"],
        [52,145000,"Retired",790,0,0,0,"Good","Master",4,0.0,0.08,4000,120000,0,"Married","Own"],
    ]
    template_df = pd.DataFrame(sample_rows, columns=template_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download CSV Template",
        data=csv_template, file_name="applicant_template.csv", mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Applicant CSV", type=["csv"])

    if uploaded is not None:
        with st.spinner("Processing batch…"):
            try:
                df_in, warnings = load_batch_csv(uploaded)
                if warnings:
                    for w in warnings:
                        st.warning(w)

                result_df = predictor.predict_batch(df_in, validate=True)
                st.success(f"✅ Processed {len(result_df):,} applicants.")

                # ── Summary KPIs ──────────────────────────────────────────
                approved  = (result_df["decision"] == "Approved").sum()
                rejected  = (result_df["decision"] == "Rejected").sum()
                avg_prob  = result_df["probability"].mean()

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total",      f"{len(result_df):,}")
                k2.metric("Approved",   f"{approved:,}",  delta=f"{approved/len(result_df):.1%}")
                k3.metric("Rejected",   f"{rejected:,}",  delta=f"-{rejected/len(result_df):.1%}")
                k4.metric("Avg Confidence", f"{avg_prob:.1%}")

                # ── Risk distribution pie ─────────────────────────────────
                st.markdown("---")
                col_chart, col_table = st.columns([1, 2])
                with col_chart:
                    risk_counts = result_df["risk_level"].value_counts()
                    fig, ax = plt.subplots(figsize=(4, 4))
                    colors_map = {"Low":"#2E7D32","Medium":"#F57F17",
                                  "High":"#E65100","Very High":"#B71C1C","Unknown":"#90A4AE"}
                    colors = [colors_map.get(r,"#90A4AE") for r in risk_counts.index]
                    ax.pie(risk_counts.values, labels=risk_counts.index, colors=colors,
                           autopct="%1.1f%%", startangle=140,
                           wedgeprops=dict(edgecolor="white"))
                    ax.set_title("Risk Distribution", fontweight="bold")
                    st.pyplot(fig)
                    plt.close(fig)

                with col_table:
                    st.markdown("##### Results Preview")
                    display_cols = ["decision","probability","risk_level","reasons"]
                    available    = [c for c in display_cols if c in result_df.columns]
                    st.dataframe(result_df[available].head(50), use_container_width=True, hide_index=True)

                # ── Download ──────────────────────────────────────────────
                csv_out = result_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Full Results CSV",
                    data=csv_out, file_name="predictions.csv", mime="text/csv",
                )

            except Exception as e:
                st.error(f"Processing failed: {e}")
                log.error("Batch prediction error: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_dashboard() -> None:
    st.markdown('<p class="section-header">📊 Model Performance Dashboard</p>', unsafe_allow_html=True)

    meta = load_metadata(METADATA_PATH)

    # ── Metadata KPIs ─────────────────────────────────────────────────────────
    if meta:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f'<div class="metric-card"><h1>{meta.get("model_name","—")}</h1><p>Best Model</p></div>',
                    unsafe_allow_html=True)
        k2.markdown(f'<div class="metric-card"><h1>{float(meta.get("f1_score",0)):.2%}</h1><p>F1-Score</p></div>',
                    unsafe_allow_html=True)
        k3.markdown(f'<div class="metric-card"><h1>{float(meta.get("accuracy",0)):.2%}</h1><p>Accuracy</p></div>',
                    unsafe_allow_html=True)
        k4.markdown(f'<div class="metric-card"><h1>{meta.get("n_train","—"):,}</h1><p>Training Samples</p></div>',
                    unsafe_allow_html=True)

    # ── Model Comparison Table ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏆 All Models Comparison")
    if COMPARISON_PATH.exists():
        df = pd.read_csv(COMPARISON_PATH)
        best_idx = int(df["F1-Score"].idxmax())

        def highlight_best(row):
            return ["background-color:#C8E6C9;font-weight:bold"
                    if row.name == best_idx else "" for _ in row]

        fmt = {c: "{:.2%}" for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
               if c in df.columns}
        st.dataframe(
            df.style.apply(highlight_best, axis=1).format(fmt),
            use_container_width=True, hide_index=True,
        )
        st.caption("🟢 Green row = best model by F1-Score")
    else:
        st.warning("Run `python train.py` to generate model results.")

    # ── Training Report Plots ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Training Report Visualisations")

    plot_catalogue = {
        "Target Distribution"        : "01_target_distribution.png",
        "Numerical Distributions"    : "02_numerical_distributions.png",
        "Categorical vs Target"      : "03_categorical_vs_target.png",
        "Correlation Heatmap"        : "04_correlation_heatmap.png",
        "Box Plots"                  : "05_boxplots.png",
        "Pairplot (Key Features)"    : "06_pairplot_key_features.png",
        "Feature Importances"        : "07_feature_importances.png",
        "Model Comparison"           : "10_model_comparison.png",
        "Confusion Matrices"         : "11_confusion_matrices.png",
        "ROC Curves"                 : "12_roc_curves.png",
        "SHAP Beeswarm"              : "13_shap_beeswarm.png",
        "SHAP Feature Importance"    : "14_shap_importance.png",
    }

    available = {k: REPORTS_DIR / v for k, v in plot_catalogue.items()
                 if (REPORTS_DIR / v).exists()}

    if available:
        keys   = list(available.keys())
        cols   = st.columns(2)
        for i, (title, path) in enumerate(available.items()):
            with cols[i % 2]:
                st.image(str(path), caption=title, use_column_width=True)
    else:
        st.info("Run `python train.py` to generate training plots.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ═══════════════════════════════════════════════════════════════════════════════
def page_about() -> None:
    st.markdown('<p class="section-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>💳 CreditAI — Credit Card Approval Prediction System</h3>
    <p>
    A production-ready machine learning system that predicts credit card application
    outcomes based on applicant financial and demographic data.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🤖 Models Used")
        models = [
            ("Logistic Regression",  "Interpretable baseline"),
            ("Decision Tree",        "Rule-based classifier"),
            ("Random Forest",        "Bagging ensemble"),
            ("Gradient Boosting",    "Sequential boosting"),
            ("XGBoost",              "Regularised gradient boosting"),
            ("LightGBM",             "Leaf-wise boosting (fast)"),
            ("Voting Ensemble",      "Soft-vote meta-model"),
        ]
        for name, desc in models:
            st.markdown(f"- **{name}** — {desc}")

        st.markdown("#### 🔬 Techniques")
        st.markdown("""
        - SMOTE oversampling for class imbalance
        - RandomizedSearchCV hyperparameter tuning
        - SHAP values for explainability
        - Rotating log files for observability
        - Pydantic request validation in FastAPI
        """)

    with col2:
        st.markdown("#### 🗂️ Features (17 total)")
        features = [
            "Age", "Annual Income", "Employment Status", "Credit Score",
            "Years Employed", "Existing Loans", "Loan Amount",
            "Repayment History", "Education", "Num Credit Cards",
            "Debt-to-Income Ratio", "Credit Utilization ★",
            "Monthly Expenses ★", "Savings Balance ★",
            "Months Since Last Default ★", "Marital Status ★", "Housing Status ★",
        ]
        for f in features:
            st.markdown(f"- {f}")
        st.caption("★ = newly added real-world features")

    st.markdown("---")
    st.markdown("#### 🚀 API Endpoints")
    st.code("""
POST /predict          Single applicant prediction
POST /predict/batch    Batch prediction (JSON)
POST /predict/csv      Batch prediction (CSV upload)
GET  /model/info       Model metadata
GET  /model/comparison All-model comparison
GET  /health           Health check

Start API: uvicorn api:app --reload --port 8000
Docs:      http://localhost:8000/docs
    """, language="text")

    st.markdown("#### 📦 Deployment Options")
    options = [
        ("Streamlit Cloud", "https://streamlit.io/cloud", "Push repo → auto deploy app.py"),
        ("Render",          "https://render.com",         "Free web service for FastAPI/Streamlit"),
        ("Railway",         "https://railway.app",        "One-click Python deployments"),
        ("AWS EC2",         "https://aws.amazon.com/ec2", "Full control, production-grade"),
        ("Hugging Face Spaces", "https://huggingface.co/spaces", "Free GPU/CPU Streamlit hosting"),
    ]
    for name, url, desc in options:
        st.markdown(f"- **[{name}]({url})** — {desc}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Sidebar navigation + routing
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── Auto-train check (Streamlit Cloud / first run) ────────────────────────
    if not MODEL_PATH.exists():
        _auto_train()
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 💳 CreditAI")
        st.markdown("*Credit Card Approval System*")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=["🔍 Single Prediction", "📂 Batch Predictions",
                     "📊 Model Dashboard", "ℹ️ About"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        # Model status
        meta = load_metadata(METADATA_PATH)
        if meta:
            st.markdown("**Model Status**")
            st.success(f"✅ {meta.get('model_name','Unknown')}")
            st.caption(f"F1: {float(meta.get('f1_score',0)):.2%}  |  "
                       f"Acc: {float(meta.get('accuracy',0)):.2%}")
        else:
            st.warning("⚠️ No trained model found.\nRun `python train.py`")

        st.markdown("---")
        st.caption("Built with scikit-learn, XGBoost,\nLightGBM & Streamlit")

    # ── Load model for prediction pages ──────────────────────────────────────
    predictor = None
    if page in ("🔍 Single Prediction", "📂 Batch Predictions"):
        try:
            predictor = load_predictor()
        except FileNotFoundError:
            st.error("Model not found. Please run `python train.py` first.")
            st.stop()
    # ── Route ─────────────────────────────────────────────────────────────────
    if page == "🔍 Single Prediction":
        page_single_prediction(predictor)
    elif page == "📂 Batch Predictions":
        page_batch_predictions(predictor)
    elif page == "📊 Model Dashboard":
        page_model_dashboard()
    else:
        page_about()


if __name__ == "__main__":
    main()
