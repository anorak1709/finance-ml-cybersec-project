<<<<<<< HEAD
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
=======
The problem
Home Credit wants to predict which loan applicants will default (fail to repay). Most applicants lack traditional credit history, so the model uses alternative data. The output is a Probability of Default (PD) between 0 and 1 — not a yes/no, but a calibrated probability that feeds into the banking risk formula Expected Loss = PD × LGD × EAD.

Step 1 — EDA (Notebook 01)
Goal: Understand the raw data before touching it.

Loaded application_train.csv — 307,511 loan applications with 122 columns. Each row is one applicant. The TARGET column says whether they defaulted (1) or repaid (0).
Checked class imbalance — Only 8.07% of applicants defaulted. This means a dumb model predicting "everyone repays" gets 92% accuracy but is useless. This is why the project uses PR-AUC and KS instead of accuracy.
Mapped missingness — 67 of 122 columns have missing values. 41 columns are more than 50% missing (mostly building/housing detail columns). LightGBM handles NaN natively, so no manual imputation needed.
Found the DAYS_EMPLOYED trap — 18% of rows have DAYS_EMPLOYED = 365,243 (roughly 1,000 years). This is a sentinel meaning "not employed / pensioner," not an actual value. Leaving it as-is would corrupt every calculation involving employment duration. These people actually default less (5.4% vs 8.7%), so the sentinel carries real signal.
Analyzed EXT_SOURCE_1/2/3 — These are scores from external credit bureaus, normalized to [0,1]. They're the single most predictive features in the dataset (correlations of -0.16 to -0.18 with default). Defaulters have visibly lower EXT_SOURCE scores. EXT_SOURCE_1 is 56% missing, but the other two are mostly complete.
Checked categorical columns — 16 categorical columns, all low-cardinality (mostly under 10 unique values). Can be fed directly to LightGBM as category dtype without one-hot encoding.
Checked bureau coverage — 99.4% of applicants have at least one prior credit record in the bureau table, so bureau-derived features will have broad coverage.
Step 2 — Feature Engineering (Notebook 02)
Goal: Transform raw tables into a single applicant-level feature matrix.

Fixed the DAYS_EMPLOYED sentinel — Replaced 365,243 with NaN and created a binary flag DAYS_EMPLOYED_ANOM (1 = was sentinel). This preserves the signal without poisoning numeric calculations.
Created application-level ratios — Domain-meaningful features a credit analyst would recognize:
CREDIT_INCOME_RATIO = loan amount / income (how leveraged is the applicant?)
ANNUITY_INCOME_RATIO = monthly payment / income (debt service burden)
CREDIT_ANNUITY_RATIO = loan amount / monthly payment (proxy for loan term in years)
CREDIT_GOODS_RATIO = loan amount / goods price (how much markup/insurance is bundled?)
EMPLOYED_AGE_RATIO = employment duration / age (job stability relative to life stage)
INCOME_PER_PERSON = income / family size
Created EXT_SOURCE aggregates — Since EXT_SOURCE_1/2/3 are the strongest features, the project engineered combinations: mean, min, max, std, product, and a count of how many are missing. These capture the overall bureau signal even when one source is missing.
Aggregated bureau_balance — This table has one row per (bureau record × month), with payment status codes (on time, 1 month late, 2 months late, etc.). Collapsed to per-bureau-record summaries: total months tracked, max days-past-due, how many months had any delinquency.
Aggregated bureau — This table has one row per prior credit record. Merged in the bureau_balance summaries, then collapsed to per-applicant: count of prior credits, total debt, max overdue amount, mean credit duration, etc. Did this three times — once for all credits, once for only active credits, once for only closed credits. Active-credit aggregates capture current leverage; closed-credit aggregates capture repayment track record.
Merged everything — Left-joined all bureau aggregates onto the application table by SK_ID_CURR. Added two final ratios: active debt as a share of total debt, and active credit count as a share of total count.
Result: 307,511 rows × 241 columns. Saved as features.parquet (86 MB). The 16 object columns were cast to pandas category dtype for LightGBM.
Sanity checks — Verified no rows were lost in the join, TARGET has no NaN, SK_ID_CURR is unique, and default rate is still 8.07%.
Step 3 — Modeling (Notebook 03)
Goal: Train a calibrated LightGBM model and measure its performance honestly.

Loaded the feature matrix — 239 features (241 minus TARGET and SK_ID_CURR). Set scale_pos_weight = 11.39 (ratio of non-defaulters to defaulters) so the model pays 11x more attention to defaults during training.
Set hyperparameters — Conservative settings to avoid overfitting on the small minority class: num_leaves=31, min_child_samples=100, strong regularization (reg_alpha=0.1, reg_lambda=0.1), 80% feature and row subsampling.
Ran 3-fold stratified cross-validation — Each fold trains on ~205K rows and validates on ~102K. Stratified means each fold preserves the 8% default rate. Early stopping on validation PR-AUC prevents overfitting — training stops when the validation metric doesn't improve for 50 rounds. Best iterations: 197, 248, 405.
Collected out-of-fold (OOF) predictions — Every row gets a prediction from the fold where it was in the validation set. This gives an honest, leak-free score for every applicant.
Measured OOF performance:
PR-AUC: 0.2586 — Looks low, but this is normal for 8% base rate problems. It means the model concentrates defaults toward the top of its ranking.
ROC-AUC: 0.7705 — The model has 77% chance of ranking a random defaulter above a random non-defaulter. Decent for tabular credit data.
Gini: 0.5410 — Industry-standard measure (= 2×AUC − 1). A Gini above 0.4 is considered good for credit scoring.
KS: 0.4062 — Maximum separation between the cumulative distributions of defaulters and non-defaulters. >0.3 is acceptable in practice.
Brier: 0.1769 — Mean squared error of predicted probabilities. High because scale_pos_weight inflates raw scores.
Applied isotonic calibration — The raw LightGBM scores are good rankings but not true probabilities (mean raw score was 0.377 vs actual 8.07% default rate). Isotonic regression learned a monotonic mapping from raw scores to calibrated PDs on the OOF data. After calibration:
Mean predicted PD = 0.0807 (matches actual default rate exactly)
Brier score dropped from 0.1769 → 0.0670 (major improvement)
Ranking metrics (PR-AUC, ROC-AUC, KS) stayed the same (isotonic is monotonic, so ranks don't change)
Identified top 20 features by LightGBM gain:
EXT_SOURCE_MEAN dominated (377K gain — 2.5x the next feature)
Then ORGANIZATION_TYPE, CREDIT_ANNUITY_RATIO, CREDIT_GOODS_RATIO, OCCUPATION_TYPE
Bureau features like EXT_SOURCE_MIN, employment and age features also ranked highly
Refitted champion model on full data — Used the median best iteration (248 rounds) to train on all 307K rows. Saved the model, calibrator, OOF predictions, and feature importances.
Step 4 — Interpretability (Notebook 04)
Goal: Explain the model in terms a credit risk analyst can understand and act on.

Computed SHAP values — Used TreeExplainer on a stratified 20K-row sample (preserving the 8% default rate). SHAP decomposes every prediction into per-feature contributions: "this applicant's PD is 12% because their EXT_SOURCE_MEAN pushed it up by +3%, their CREDIT_ANNUITY_RATIO pushed it down by -1%," etc.
Global SHAP bar chart — Shows the top 20 features ranked by mean absolute SHAP value (average contribution magnitude across all applicants):
EXT_SOURCE_MEAN (0.386) — by far the dominant driver. This is the average of the three external bureau scores.
CREDIT_ANNUITY_RATIO (0.129) — proxy for loan term; longer terms = higher risk
CREDIT_GOODS_RATIO (0.123) — how much the loan exceeds the goods price
ORGANIZATION_TYPE (0.123) — employer type
EXT_SOURCE_MIN (0.095) — worst bureau score matters
Global SHAP beeswarm — Shows not just magnitude but direction. For example: low EXT_SOURCE_MEAN (red dots on the right) pushes predictions toward default; high values (blue) push toward repayment.
Dependence plots for top 5 — Shows how each feature's SHAP contribution varies with its value. For example, EXT_SOURCE_MEAN below ~0.3 has a strong positive SHAP (pushes toward default), while above ~0.5 it has a strong negative SHAP (pushes toward repayment).
Three local explanations (waterfall plots):
Safe applicant (PD = 0.000): High EXT_SOURCE scores, good credit ratios → every feature pushed the prediction toward "will repay"
Risky applicant (PD = 1.000): Low EXT_SOURCE scores, bad ratios → every feature pushed toward "will default." This person actually did default.
Borderline applicant (PD = 0.100): Mixed signals — some features push toward risk, others toward safety, landing near the decision boundary.
Saved the top-20 SHAP table and generated MODEL_README.md — an analyst-readable document explaining the risk factors, metrics, and monitoring guidance.
What was produced
Artifact	Purpose
features.parquet (86 MB)	The 241-column feature matrix, reusable for retraining
champion.joblib (900 KB)	Serialized LightGBM model + isotonic calibrator
oof_predictions.parquet (4.4 MB)	Every applicant's calibrated PD, for threshold analysis
MODEL_README.md	Analyst-facing risk narrative
6 PNG plots	Reliability diagram, SHAP global/dependence/local explanations
2 CSV files	Feature importance and SHAP top-20 rankings
>>>>>>> 7bbe780568077cc4bbc0f0927959bc493341f881
