"""Model evaluation metrics for credit scoring.

- KS statistic (Kolmogorov-Smirnov): max separation between good/bad CDFs
- Gini coefficient: 2*AUC - 1
- PR-AUC: area under precision-recall curve
- scorecard_report: comprehensive metrics dict
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


def ks_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic.

    Measures the maximum separation between cumulative distributions
    of good and bad samples. Range [0, 1], higher = better separation.

    Interpretation (banking convention):
        < 0.20 — poor
        0.20-0.30 — fair
        0.30-0.40 — good
        0.40-0.50 — excellent
        > 0.50 — suspect (overfitting or leakage)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    # Sort predictions descending
    idx = np.argsort(y_pred)[::-1]
    y_true = y_true[idx]

    n_bad = (y_true == 1).sum()
    n_good = (y_true == 0).sum()

    if n_bad == 0 or n_good == 0:
        return 0.0

    cum_bad = np.cumsum(y_true == 1) / n_bad
    cum_good = np.cumsum(y_true == 0) / n_good

    return np.max(np.abs(cum_bad - cum_good))


def gini_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1.

    Range [-1, 1], 0 = random, 1 = perfect.
    """
    auc_val = roc_auc_score(y_true, y_pred)
    return 2 * auc_val - 1


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC AUC."""
    return roc_auc_score(y_true, y_pred)


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area under Precision-Recall curve.

    More informative than ROC AUC for imbalanced datasets.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def scorecard_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
    label: str = "",
) -> Dict:
    """Generate comprehensive model performance report."""
    auc = roc_auc_score(y_true, y_pred)
    ks = ks_score(y_true, y_pred)
    gini = 2 * auc - 1
    pr = pr_auc(y_true, y_pred)

    report = {
        "label": label,
        "AUC": round(auc, 4),
        "KS": round(ks, 4),
        "Gini": round(gini, 4),
        "PR_AUC": round(pr, 4),
        "n_samples": len(y_true),
        "default_rate": float(y_true.mean()),
    }

    # Score-band default rates
    if y_score is not None:
        bands = [(300, 450), (450, 520), (520, 580), (580, 640), (640, 700), (700, 760), (760, 850)]
        band_rates = {}
        for low, high in bands:
            mask = (y_score >= low) & (y_score < high)
            if mask.sum() > 0:
                band_rates[f"{low}-{high}"] = {
                    "count": int(mask.sum()),
                    "default_rate": round(float(y_true[mask].mean()), 4),
                }
        report["score_bands"] = band_rates

    return report


def roc_data(y_true: np.ndarray, y_pred: np.ndarray):
    """Return FPR, TPR, thresholds for ROC plotting."""
    return roc_curve(y_true, y_pred)
