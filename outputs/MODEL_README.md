# PD Model — Analyst README

**Model:** LightGBM, calibrated with isotonic regression
**Trained on:** Home Credit Default Risk — `application_train` (307,511 rows) + `bureau` + `bureau_balance`
**Output:** `PD ∈ [0, 1]` per applicant — the Probability of Default within the loan tenor
**Champion artifact:** `outputs/champion.joblib`
**Last trained:** 2026-05-22

---

## What this model is for

This is the **PD pillar** of the Basel III / RBI Expected Loss framework:

```
Expected Loss = PD × LGD × EAD
```

It does not estimate Loss Given Default or Exposure at Default. It only answers: *given everything we know about an applicant at the point of underwriting, what is the probability they will default?* Pricing, provisioning, and approval rules sit downstream of this score.

---

## Headline performance (5-fold cross-validation, out-of-fold)

| Metric | Value | What it tells you |
|---|---|---|
| **PR-AUC** | 0.263 | Mean precision across all recall levels. Baseline = base default rate (0.08). We are ~3.3× the trivial baseline. |
| **ROC-AUC** | 0.773 | Probability the model ranks a random defaulter above a random non-defaulter. 0.70+ is bankable; 0.80+ is best-in-class. |
| **Gini** | 0.546 | `2·AUC − 1`. Industry shorthand for ROC-AUC. Anything >0.40 is usable for underwriting. |
| **KS statistic** | 0.409 | Max separation between cumulative TPR and FPR. >0.30 = usable, >0.40 = strong. This is what your CRO will ask for. |
| **Brier score (calibrated)** | 0.067 | Mean squared error of predicted PDs. Lower is better; <0.10 on imbalanced data is good. Isotonic calibration cut this from 0.164 → 0.067 (≈60% reduction). |

**Population calibration check:** mean predicted PD = 0.0807, actual default rate = 0.0807 — perfectly aligned at the portfolio level.

### Why not accuracy?
Predicting "no one defaults" gives 92% accuracy and zero business value. We refuse to report it.

---

## Top risk factors (SHAP, mean |contribution|)

| Rank | Feature | Plain-English meaning | Direction |
|---|---|---|---|
| 1 | `EXT_SOURCE_MEAN` | Average of three external bureau scores | Higher score → lower PD |
| 2 | `CREDIT_ANNUITY_RATIO` | `AMT_CREDIT / AMT_ANNUITY` ≈ loan duration in years | Longer terms → higher PD |
| 3 | `ORGANIZATION_TYPE` | Employer category (Business, Government, Self-employed, etc.) | Self-employed and certain industries skew riskier |
| 4 | `CREDIT_GOODS_RATIO` | Loan amount vs. price of goods financed | Borrowing more than goods cost → higher PD |
| 5 | `OWN_CAR_AGE` | Age of applicant's car (if any) | Older car → marginally higher PD (proxy for income stability) |
| 6 | `CODE_GENDER` | M/F | **Flagged — see Fairness section** |
| 7 | `NAME_EDUCATION_TYPE` | Highest completed education | Higher education → lower PD |
| 8 | `AMT_ANNUITY` | Monthly loan instalment | Larger instalments → higher PD |
| 9–12 | `EXT_SOURCE_*` (1, 2, 3, MIN, MAX) | Individual external bureau scores | All higher = lower PD |

The story is unsurprising: **external bureau scores carry most of the signal**, followed by loan-affordability ratios, then employer/education stability proxies.

See [shap_global_beeswarm.png](shap_global_beeswarm.png) for the full distribution and [shap_dependence_top5.png](shap_dependence_top5.png) for how each driver behaves across its value range.

---

## How to read a PD score

| PD bucket | Suggested treatment | Approx. share of applicants |
|---|---|---|
| < 0.02 | Pre-approve, best pricing tier | ~25% |
| 0.02 – 0.08 | Standard approval, standard pricing | ~40% |
| 0.08 – 0.20 | Manual review or risk-based pricing | ~25% |
| 0.20 – 0.40 | Decline by default; approve only with collateral / co-applicant | ~7% |
| > 0.40 | Decline | ~3% |

These cut-offs are **starting points** — finalize against your loss tolerance and capital cost. The model gives a probability; the cut-offs are a business decision.

For three concrete examples (safe / risky / borderline), see [shap_local_safe.png](shap_local_safe.png), [shap_local_risky.png](shap_local_risky.png), [shap_local_borderline.png](shap_local_borderline.png).

---

## Monitoring in production

A model that performs well today can silently degrade as the borrower mix shifts. Recommended monitoring cadence:

### Daily / weekly
- **PSI (Population Stability Index)** on the score distribution vs. the training distribution.
  - PSI < 0.10: stable
  - 0.10 – 0.25: investigate
  - \> 0.25: retrain
- **PSI on top 10 SHAP features individually.** Catches input drift before it shows up in the score.

### Monthly
- **Vintage analysis** — observed default rate by approval cohort vs. predicted PD at the cohort's mean.
- **Calibration check** — bin PDs into deciles, plot predicted vs. realized default rate.
- **Approval rate by PD bucket** — has the funnel shifted?

### Quarterly
- **Full retrain** with the latest 12 months of mature loans (typically loans seasoned ≥6 months so the default label is reliable).
- **Champion-challenger** test before swapping the production model.

### Triggers for emergency review
- Score PSI > 0.25 in any week.
- Mean PD on new applications moves by > 20% week-over-week.
- Observed-vs-predicted default rate diverges by > 30% in any decile for two consecutive months.

---

## Fairness and regulatory caveats

- **`CODE_GENDER` is among the top 10 SHAP drivers.** Many jurisdictions (US ECOA, EU equality directives) prohibit using gender directly in credit decisions. Before deploying, you must either:
  1. Drop `CODE_GENDER` from features and retrain (expect a small AUC drop), OR
  2. Keep it but run an adverse-impact analysis and document a legal basis — typically not available for retail credit.

  Even if `CODE_GENDER` is dropped, **proxy variables** (occupation type, organization type) may correlate with gender. A separate fairness audit on a holdout segmented by protected attributes is required.

- **`ORGANIZATION_TYPE` and `OCCUPATION_TYPE`** can proxy for ethnicity, caste, or nationality depending on geography. Audit before deployment.

- **Model card:** maintain a document listing intended use, exclusions (e.g., business loans, secured credit), training data window, known limitations, and review history. RBI's Working Group on Digital Lending guidelines (2022) make this effectively mandatory for NBFCs.

- **Explainability for adverse action notices:** when declining an applicant, regulators (and customers) increasingly demand a reason. Use per-applicant SHAP values to surface the top 2–3 negative contributors as the decline rationale.

---

## Data quirks the model handles (so you don't have to re-discover them)

- **`DAYS_EMPLOYED == 365243`** (~18% of training rows) encodes "never employed / pensioner". We NaN it and add a binary flag — pensioners actually have *lower* default rates than working applicants, so the flag is signal, not noise.
- **High missingness in the building/material columns** (>50% missing in `*_AVG`, `*_MEDI`, `*_MODE`). LightGBM handles NaN natively; we did not impute.
- **`EXT_SOURCE_1` is ~57% missing.** Critical — many applicants are thin-file for the most-predictive bureau. The model leans more on `EXT_SOURCE_2` and `EXT_SOURCE_3` for those applicants.
- **Bureau coverage is 99.4%** of applicants. Bureau aggregates (active credit count, sum of overdue amounts, max DPD) materially improve the model.

---

## How to score a new applicant

```python
import joblib
import pandas as pd

artifacts = joblib.load("outputs/champion.joblib")
model = artifacts["model"]
calibrator = artifacts["calibrator"]
feature_names = artifacts["feature_names"]

# new_applicant must go through the same FE pipeline as notebooks 01 + 02
X_new = new_applicant_features[feature_names]
raw_score = model.predict(X_new)
pd_score = calibrator.predict(raw_score)  # this is the PD to use
```

**Do not** ship `raw_score` to downstream systems. It is a ranking, not a probability — the isotonic step is what makes it calibrated.

---

## Reproducibility

- Random seed: `42` (set in `src/utils.py` as `RANDOM_STATE`).
- Notebook order: `01_eda` → `02_feature_engineering` → `03_modeling` → `04_interpretability`.
- All intermediates persisted to `outputs/`. To rebuild from scratch, delete `outputs/*` and re-run the notebooks top-to-bottom.
- Dependencies pinned in `requirements.txt`.

## Known limitations

- **No previous_application / POS_CASH / credit_card / installments tables used.** Kaggle winners gain another ~0.02 ROC-AUC from these. Adding them is the obvious v2 lift.
- **No hyperparameter tuning beyond hand-picked defaults.** Bayesian search over `num_leaves`, `learning_rate`, `min_child_samples` would likely add another 0.005–0.015 ROC-AUC.
- **No temporal validation.** The Home Credit dataset has no timestamps usable for out-of-time validation. In production you must validate on a forward window before promoting any retrain.
- **No challenger model.** A logistic-regression scorecard with WOE binning is the regulator-friendly baseline you'd want to maintain in parallel.
