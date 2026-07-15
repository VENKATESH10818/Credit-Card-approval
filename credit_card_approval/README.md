# 💳 CreditAI — Credit Card Approval Prediction System

A production-ready ML system that predicts credit card application outcomes
with explainable AI, a modern Streamlit UI, and a FastAPI REST backend.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates data, trains 7 models, tunes the best one)
python train.py

# 3. Launch the Streamlit app
streamlit run app.py

# 4. (Optional) Launch the FastAPI backend
uvicorn api:app --reload --port 8000
# API docs → http://localhost:8000/docs
```

---

## 📁 Project Structure

```
credit_card_approval/
├── config.py              # Central config (paths, features, ranges)
├── logger.py              # Rotating file + console logger
├── utils.py               # Validation, risk scoring, batch helpers
├── generate_dataset.py    # Synthetic dataset generator (17 features)
├── preprocessing.py       # EDA, imputation, encoding, scaling, SMOTE
├── models.py              # 7 classifiers + RandomizedSearchCV tuning
├── explainability.py      # SHAP global + per-applicant explanations
├── predictor.py           # CreditCardPredictor (predict/batch/explain)
├── train.py               # Full pipeline orchestrator
├── api.py                 # FastAPI REST backend
├── app.py                 # Streamlit production UI
├── requirements.txt
├── Procfile               # Heroku / Render deployment
├── render.yaml            # Render.com multi-service config
└── .streamlit/
    └── config.toml        # Streamlit theme + server settings
```

---

## 🤖 Models

| Model               | Type                  |
|---------------------|-----------------------|
| Logistic Regression | Linear baseline       |
| Decision Tree       | Rule-based            |
| Random Forest       | Bagging ensemble      |
| Gradient Boosting   | Boosting (sklearn)    |
| XGBoost             | Regularised boosting  |
| LightGBM            | Leaf-wise boosting    |
| Voting Ensemble     | Soft-vote meta-model  |

The best model by F1-score is automatically selected and hyperparameter-tuned
with `RandomizedSearchCV` (40 iterations, stratified 5-fold CV).

---

## 📊 Features (17)

**Numerical (12):** age, income, credit_score, years_employed, existing_loans,
loan_amount, num_credit_cards, debt_to_income, credit_utilization,
monthly_expenses, savings_balance, months_since_last_default

**Categorical (5):** employment_status, repayment_history, education,
marital_status, housing_status

---

## 🔌 API Endpoints

```
POST /predict           Single applicant → decision + probability + reasons
POST /predict/batch     JSON array       → bulk predictions
POST /predict/csv       CSV upload       → bulk predictions + download
GET  /model/info        Trained model metadata
GET  /model/comparison  All-model comparison table
GET  /health            Health check
```

---

## 🌐 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point

### Render
```bash
# Uses render.yaml — just connect your GitHub repo
```

### Locally with Docker (optional)
```bash
docker build -t creditai .
docker run -p 8501:8501 creditai
```

---

## 🧪 Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), SHAP
- **API:** FastAPI + Pydantic v2 + Uvicorn
- **UI:** Streamlit
- **Logging:** Python `logging` with `RotatingFileHandler`
