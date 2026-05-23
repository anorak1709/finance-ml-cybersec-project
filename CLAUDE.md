# Home Credit Default Risk — Probability of Default (PD) Model

## Problem
Build a supervised ML model that outputs a **Probability of Default (PD ∈ [0,1])** for loan applicants who lack traditional credit history. This is the first pillar of the Basel III / RBI Expected Loss framework (`EL = PD × LGD × EAD`). The deliverable is something a credit risk analyst — not just an ML engineer — can read, trust, and monitor.

## Dataset
Home Credit Default Risk (Kaggle). Located in `data/`.

- `application_train.csv` — ~307K rows, 122 cols, contains `TARGET` (1 = default, 0 = repaid). **Base default rate ≈ 8%** → imbalanced.
- `application_test.csv` — scoring set, no target.
- `bureau.csv` + `bureau_balance.csv` — prior credit bureau records (other lenders).
- `previous_application.csv` — prior Home Credit applications.
- `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv` — monthly behavioral history on prior loans.
- `HomeCredit_columns_description.csv` — data dictionary. **Consult before inventing feature meanings.**

Join key: `SK_ID_CURR` (current application) and `SK_ID_BUREAU` / `SK_ID_PREV` for the auxiliary tables.

## Repo Layout
```
data/                # raw CSVs (do not commit if large; check .gitignore)
notebooks/
  01_eda.ipynb              # data audit, target balance, missingness, leakage checks
  02_feature_engineering.ipynb  # aggregate auxiliary tables → application level
  03_modeling.ipynb         # baseline + tuned model, CV, calibration
  04_interpretability.ipynb # SHAP global + local, risk-factor narrative
outputs/             # serialized models, SHAP plots, metric tables, README artifacts
src/utils.py         # shared helpers (loaders, metrics, plotting)
```

## Modeling Principles
- **Accuracy is meaningless here** (92% baseline by predicting all-zero). Use:
  - **PR-AUC** (primary — handles imbalance)
  - **ROC-AUC** (secondary, comparable across literature)
  - **KS statistic** (industry standard for credit scorecards)
  - **Calibration** (Brier score, reliability curve) — PD must be a *probability*, not a ranking score
  - **Gini = 2·AUC − 1** for stakeholder reporting
- **Imbalance handling**: prefer `scale_pos_weight` / class weights over naive oversampling; SMOTE only if justified and only on training folds.
- **Validation**: stratified K-fold on `TARGET`. No target leakage from aggregates computed on the full dataset.
- **Calibration**: isotonic or Platt scaling on a held-out fold before producing PDs.

## Interpretability Requirements
- Global SHAP feature importance (top 20).
- Local SHAP examples: one approved-looking applicant, one rejected-looking, one borderline.
- Map top features back to **business meaning** (e.g., `EXT_SOURCE_*` = external bureau scores; `DAYS_EMPLOYED` = job tenure — watch for the 365243 sentinel).
- Fairness / monitoring notes: which features could proxy protected attributes; how to detect PSI / drift in production.

## Locked Scope Decisions (2026-05-22)
- **Data scope**: `application_train` + `bureau` + `bureau_balance`. Skip `previous_application`, POS_CASH, credit_card, installments for v1.
- **Model**: LightGBM as the single champion. No logistic regression baseline unless added later.
- **Imbalance**: `scale_pos_weight` (= negatives/positives ≈ 11.4). No SMOTE, no undersampling.
- **Deliverables (all four required)**:
  1. Trained LightGBM model serialized to `outputs/`.
  2. SHAP global feature importance + dependence plots for top 5 + 2–3 local explanations.
  3. Analyst-readable README (risk-factor narrative, metric explanations, monitoring guidance).
  4. Notebooks 01–04 runnable top-to-bottom.
- **Env**: dependencies declared in `requirements.txt`; user installs manually.

## Non-Goals
- Not building LGD or EAD models.
- Not deploying a service — final artifact is notebooks + saved model + analyst-readable README.
- No deep learning unless tabular DL clearly beats GBM (it won't on this dataset).

## Conventions
- Python, pandas, scikit-learn, LightGBM/XGBoost, SHAP.
- Set seeds everywhere (`RANDOM_STATE = 42`).
- Persist intermediates as parquet under `outputs/` to keep notebooks fast.
- Notebook cells should be re-runnable top-to-bottom without manual fiddling.
