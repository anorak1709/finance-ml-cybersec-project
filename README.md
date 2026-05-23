# Home Credit Default Risk — Probability of Default Model

A machine learning pipeline that predicts the **Probability of Default (PD)** for loan applicants who lack traditional credit history. Built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (307,511 applications), this project delivers a calibrated LightGBM model, SHAP-based interpretability, and an analyst-readable risk narrative.

This is the **PD pillar** of the Basel III / RBI Expected Loss framework:

```
Expected Loss = PD × LGD × EAD
```

## Why This Matters

Most traditional credit scoring models rely on established credit bureau history. Millions of potential borrowers are excluded simply because they lack that history. This model uses alternative data — employment details, income ratios, asset ownership, and aggregated bureau records from prior loans — to produce a calibrated probability that a given applicant will default.

## Dataset

The Home Credit Default Risk competition dataset from Kaggle. Place the CSV files in `data/`.

| File | Description |
|------|-------------|
| `application_train.csv` | 307,511 loan applications with 122 features and a binary `TARGET` (1 = default, 0 = repaid) |
| `application_test.csv` | Scoring set (no target) |
| `bureau.csv` | 1.7M prior credit records from other lenders |
| `bureau_balance.csv` | 27M monthly payment status records for bureau credits |
| `HomeCredit_columns_description.csv` | Data dictionary |

Base default rate is **8.07%**, making this a heavily imbalanced classification problem.

## Pipeline

### 1. Exploratory Data Analysis ([01_eda.ipynb](notebooks/01_eda.ipynb))

- Audits data shape, types, and memory footprint
- Quantifies target imbalance (8% default rate) and explains why accuracy is a useless metric here
- Maps missingness across 122 columns (41 columns are >50% missing)
- Detects the `DAYS_EMPLOYED = 365,243` sentinel (18% of rows — encodes pensioners/unemployed, who actually default less)
- Analyzes `EXT_SOURCE_1/2/3` — external bureau scores that are the single strongest predictors
- Checks categorical cardinality and bureau coverage (99.4% of applicants have prior bureau records)

### 2. Feature Engineering ([02_feature_engineering.ipynb](notebooks/02_feature_engineering.ipynb))

- Fixes the DAYS_EMPLOYED sentinel (replace with NaN + binary flag)
- Engineers 14 application-level features: income ratios, loan affordability ratios, employment-to-age ratio, EXT_SOURCE aggregates (mean, min, max, std, product)
- Aggregates `bureau_balance` (27M rows) into per-bureau summaries: months tracked, max days-past-due, delinquency count
- Aggregates `bureau` (1.7M rows) into per-applicant summaries, separately for all/active/closed credits: debt totals, overdue amounts, credit counts
- Merges everything into a single **307,511 x 241** feature matrix saved as `outputs/features.parquet`

### 3. Modeling ([03_modeling.ipynb](notebooks/03_modeling.ipynb))

- Trains LightGBM with `scale_pos_weight = 11.39` (no SMOTE, no undersampling)
- 3-fold stratified cross-validation with early stopping on PR-AUC
- Applies **isotonic calibration** on out-of-fold predictions so that raw scores become true probabilities (mean predicted PD matches the 8.07% actual default rate exactly)
- Refits a champion model on the full dataset and serializes it to `outputs/champion.joblib`

### 4. Interpretability ([04_interpretability.ipynb](notebooks/04_interpretability.ipynb))

- Computes SHAP values on a stratified 20K-row sample
- Generates global explanation plots (bar chart, beeswarm) and dependence plots for the top 5 drivers
- Produces three local waterfall explanations: a safe applicant (PD = 0.0%), a risky applicant (PD = 100%), and a borderline applicant (PD = 10%)
- Writes `outputs/MODEL_README.md` — an analyst-facing document with risk-factor narratives, PD bucket guidance, monitoring recommendations, and fairness caveats

## Model Performance

Out-of-fold metrics (honest, leak-free evaluation):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **PR-AUC** | 0.259 | ~3.2x the trivial 0.08 baseline; primary metric for imbalanced data |
| **ROC-AUC** | 0.771 | 77% chance of ranking a defaulter above a non-defaulter |
| **Gini** | 0.541 | Industry shorthand (2xAUC - 1); >0.40 is usable for underwriting |
| **KS** | 0.406 | Max separation between defaulter/non-defaulter distributions; >0.30 is acceptable |
| **Brier (calibrated)** | 0.067 | Calibration reduced this from 0.177 to 0.067 (62% improvement) |

## Top Risk Drivers (SHAP)

| Rank | Feature | Meaning |
|------|---------|---------|
| 1 | EXT_SOURCE_MEAN | Average of three external bureau scores — higher means safer |
| 2 | CREDIT_ANNUITY_RATIO | Loan amount / monthly payment (proxy for loan term) — longer = riskier |
| 3 | CREDIT_GOODS_RATIO | Loan amount / goods price — borrowing more than goods cost = riskier |
| 4 | ORGANIZATION_TYPE | Employer category — self-employed and certain industries skew riskier |
| 5 | EXT_SOURCE_MIN | Worst bureau score — a single bad score matters |

## Output Artifacts

| File | Description |
|------|-------------|
| `outputs/champion.joblib` | Serialized LightGBM model + isotonic calibrator |
| `outputs/features.parquet` | 241-column feature matrix (reusable for retraining) |
| `outputs/oof_predictions.parquet` | Calibrated PD for every training applicant |
| `outputs/MODEL_README.md` | Analyst-facing risk narrative and monitoring guide |
| `outputs/reliability.png` | Calibration reliability diagram |
| `outputs/shap_global_bar.png` | Top 20 features by mean SHAP contribution |
| `outputs/shap_global_beeswarm.png` | SHAP distribution and direction |
| `outputs/shap_dependence_top5.png` | How top 5 features influence predictions |
| `outputs/shap_local_*.png` | Waterfall explanations for safe, risky, borderline applicants |

## Repo Structure

```
data/                  # Raw CSVs (not committed)
notebooks/
  01_eda.ipynb
  02_feature_engineering.ipynb
  03_modeling.ipynb
  04_interpretability.ipynb
outputs/               # Serialized models, plots, metrics, parquet intermediates
src/utils.py           # Shared helpers (data loaders, metric functions, constants)
requirements.txt       # Python dependencies
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Place the Kaggle dataset CSVs in data/

# Run notebooks in order
jupyter nbconvert --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --execute --inplace notebooks/02_feature_engineering.ipynb
jupyter nbconvert --execute --inplace notebooks/03_modeling.ipynb
jupyter nbconvert --execute --inplace notebooks/04_interpretability.ipynb
```

## Scoring a New Applicant

```python
import joblib

artifacts = joblib.load("outputs/champion.joblib")
model = artifacts["model"]
calibrator = artifacts["calibrator"]
feature_names = artifacts["feature_names"]

# new_applicant must go through the same feature engineering pipeline
X_new = new_applicant_features[feature_names]
raw_score = model.predict(X_new)
pd_score = calibrator.predict(raw_score)  # calibrated PD — use this, not raw_score
```

## Known Limitations

- Only uses `application_train` + `bureau` + `bureau_balance`. Adding `previous_application`, POS_CASH, credit_card, and installments tables would likely improve ROC-AUC by ~0.02.
- No hyperparameter tuning beyond hand-picked defaults. Bayesian optimization would add another 0.005-0.015 ROC-AUC.
- No temporal validation (dataset lacks usable timestamps). In production, validate on a forward time window before promoting any retrain.
- `CODE_GENDER` is among the top SHAP drivers. Many jurisdictions prohibit using gender in credit decisions — see `outputs/MODEL_README.md` for fairness guidance.

## Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, matplotlib, seaborn
