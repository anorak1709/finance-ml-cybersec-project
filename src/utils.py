"""Shared helpers for the Home Credit PD project.

Kept intentionally small: data loading, the DAYS_EMPLOYED sentinel fix, and
the credit-risk metric suite (PR-AUC, ROC-AUC, KS, Brier, Gini).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DAYS_EMPLOYED_SENTINEL = 365243


def load_application(kind: str = "train") -> pd.DataFrame:
    """Load application_train.csv or application_test.csv."""
    path = DATA_DIR / f"application_{kind}.csv"
    return pd.read_csv(path)


def load_bureau() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(DATA_DIR / "bureau.csv"),
        pd.read_csv(DATA_DIR / "bureau_balance.csv"),
    )


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the 365243 sentinel in DAYS_EMPLOYED with NaN.

    365243 days ≈ 1000 years, used to encode 'never employed' / pensioners.
    Leaving it as a number poisons distance- and tree-based features alike.
    """
    if "DAYS_EMPLOYED" in df.columns:
        df = df.copy()
        df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype("int8")
        df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan
    return df


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between cumulative TPR and FPR.

    Industry standard for credit scorecards. 0.3+ is usable, 0.4+ is strong.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)[::-1]
    y_true = y_true[order]
    pos = (y_true == 1).cumsum() / max((y_true == 1).sum(), 1)
    neg = (y_true == 0).cumsum() / max((y_true == 0).sum(), 1)
    return float(np.max(pos - neg))


def credit_risk_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """The metric suite we report for every model / fold."""
    roc = roc_auc_score(y_true, y_score)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc),
        "gini": float(2 * roc - 1),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
