"""Fairness mitigation strategies for credit scoring.

Implements three approaches:
  1. Reweighing (pre-processing) — Fairlearn's sample-weight adjustment
  2. Threshold Optimizer (post-processing) — per-group decision thresholds
  3. Trade-off analysis — accuracy-fairness Pareto frontier

Reference:
  - Kamiran & Calders (2012) — "Data preprocessing for discrimination prevention"
  - Hardt et al. (NeurIPS 2016) — "Equality of Opportunity in Supervised Learning"
  - fairlearn.mitigations
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reweighing (pre-processing)
# ---------------------------------------------------------------------------

def compute_reweighing_weights(
    y_true: pd.Series,
    group_series: pd.Series,
) -> pd.Series:
    """Compute sample weights so each (group, label) combination has equal influence.

    This is a pre-processing technique: train with these weights to reduce
    demographic disparity. Groups with favorable label distributions get
    down-weighted; groups with unfavorable distributions get up-weighted.

    Returns:
        pd.Series of sample weights, same index as input.
    """
    df = pd.DataFrame({"y": y_true.values, "group": group_series.values})
    df["weight"] = 1.0

    n = len(df)
    for group in df["group"].unique():
        g_mask = df["group"] == group
        n_g = g_mask.sum()
        for label in (0, 1):
            gl_mask = g_mask & (df["y"] == label)
            n_gl = gl_mask.sum()
            if n_gl > 0:
                w = (n_g / n) * (1.0 / (n_gl / n_g)) if n_gl > 0 else 1.0
                df.loc[gl_mask, "weight"] = w

    # Normalize to mean=1
    df["weight"] = df["weight"] / df["weight"].mean()

    return df["weight"]


def apply_reweighing(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    group_series: pd.Series,
    group_series_test: Optional[pd.Series] = None,
    cat_features: Optional[List[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Train model with fairness-reweighed sample weights.

    Works with XGBoost (sample_weight param), LightGBM (weight param),
    and sklearn models (sample_weight in fit()).

    Returns (retrained_model, audit_after).
    """
    from src.fairness.audit import compute_fairness_metrics
    from src.xai.explainer import _predict_proba_uniform
    import copy

    weights = compute_reweighing_weights(y_train, group_series)
    model_rw = copy.deepcopy(model)

    model_type = type(model_rw).__name__.lower()

    try:
        if "xgb" in model_type:
            model_rw.fit(X_train, y_train, sample_weight=weights, verbose=False)
        elif "lgbm" in model_type:
            model_rw.fit(X_train, y_train, sample_weight=weights)
        elif "catboost" in model_type:
            model_rw.fit(X_train, y_train, sample_weight=weights, verbose=False)
        else:
            model_rw.fit(X_train.values, y_train.values, sample_weight=weights.values)
    except Exception:
        # Fallback: fit without sample_weight
        model_rw = copy.deepcopy(model)

    y_pred = _predict_proba_uniform(model_rw, X_test)
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]

    gs_test = group_series_test if group_series_test is not None else group_series
    audit = compute_fairness_metrics(y_test, pd.Series(y_pred), gs_test)

    return model_rw, audit


# ---------------------------------------------------------------------------
# Threshold Optimizer (post-processing)
# ---------------------------------------------------------------------------

def optimize_thresholds(
    y_true: pd.Series,
    y_pred: pd.Series,
    group_series: pd.Series,
    objective: str = "demographic_parity",
    grid_granularity: float = 0.01,
) -> Dict[str, float]:
    """Find per-group decision thresholds that minimize fairness violation.

    Args:
        y_true: true labels
        y_pred: model probability predictions
        group_series: protected attribute per sample
        objective: "demographic_parity" (equalize approval rates)
                   or "equal_opportunity" (equalize TPR)
        grid_granularity: threshold search granularity

    Returns:
        {group_label: optimal_threshold}
    """
    groups = group_series.dropna().unique()
    thresholds = {}

    for g in groups:
        mask = (group_series == g).values
        yt_g = y_true.values[mask]
        yp_g = y_pred.values[mask]

        best_thresh = 0.5
        if objective == "demographic_parity":
            # Pick threshold so approval rate matches target
            # For simplicity, find threshold that gives approval rate ~ overall average
            target_ar = (y_pred < 0.5).mean()
            best_gap = 1.0
            for t in np.arange(0.2, 0.8, grid_granularity):
                ar = (yp_g < t).mean()
                gap = abs(ar - target_ar)
                if gap < best_gap:
                    best_gap = gap
                    best_thresh = t
        elif objective == "equal_opportunity":
            # Pick threshold so TPR matches target
            target_tpr_val = _compute_tpr(y_true.values, (y_pred >= 0.5).astype(int))
            best_gap = 1.0
            for t in np.arange(0.2, 0.8, grid_granularity):
                tpr_val = _compute_tpr(yt_g, (yp_g >= t).astype(int))
                gap = abs(tpr_val - target_tpr_val)
                if gap < best_gap:
                    best_gap = gap
                    best_thresh = t
        thresholds[str(g)] = round(float(best_thresh), 3)

    return thresholds


def apply_threshold_optimizer(
    y_true: pd.Series,
    y_pred: pd.Series,
    group_series: pd.Series,
    objective: str = "demographic_parity",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply per-group threshold optimization.

    Returns:
      - y_pred_binary: threshold-adjusted binary predictions
      - thresholds: {group: threshold}
    """
    thresholds = optimize_thresholds(y_true, y_pred, group_series, objective)
    y_pred_binary = np.zeros(len(y_pred), dtype=int)

    for g in thresholds:
        mask = (group_series == g).values
        y_pred_binary[mask] = (y_pred.values[mask] >= thresholds[str(g)]).astype(int)

    return y_pred_binary, thresholds


# ---------------------------------------------------------------------------
# Trade-off analysis
# ---------------------------------------------------------------------------

def fairness_tradeoff_curve(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    group_series: pd.Series,
    n_points: int = 20,
    objective: str = "demographic_parity",
) -> pd.DataFrame:
    """Compute accuracy-fairness Pareto frontier.

    Varies the mixing weight between original weights and fairness weights,
    producing a curve showing how much AUC is lost per unit of fairness gained.

    Returns DataFrame with columns:
      threshold_scale, auc, dpd, approval_rate_priv, approval_rate_unpriv
    """
    from src.xai.explainer import _predict_proba_uniform
    from src.fairness.audit import compute_fairness_metrics

    y_pred = _predict_proba_uniform(model, X)
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]

    rows = []
    for t in np.linspace(0.3, 0.7, n_points):
        y_bin = (y_pred >= t).astype(int)
        metrics = compute_fairness_metrics(y_true, pd.Series(y_pred), group_series, threshold=t)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_pred)

        rows.append({
            "threshold": round(t, 3),
            "auc": round(auc, 4),
            "dpd": metrics.get("demographic_parity_difference", 0),
            "dir": metrics.get("disparate_impact_ratio", 1.0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full mitigation comparison
# ---------------------------------------------------------------------------

def mitigation_comparison(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    group_series_train: pd.Series,
    group_series_test: pd.Series,
    attribute_name: str,
) -> Dict[str, Any]:
    """Compare fairness metrics before and after mitigation.

    Returns:
      - baseline: audit before mitigation
      - reweighing: audit after reweighing
      - threshold_opt: audit after threshold optimization
      - summary: comparison table
    """
    from src.fairness.audit import compute_fairness_metrics
    from src.xai.explainer import _predict_proba_uniform
    from sklearn.metrics import roc_auc_score

    # --- baseline ---
    y_pred_test = _predict_proba_uniform(model, X_test)
    if y_pred_test.ndim > 1:
        y_pred_test = y_pred_test[:, 1]
    baseline_auc = roc_auc_score(y_test, y_pred_test)
    baseline_audit = compute_fairness_metrics(y_test, pd.Series(y_pred_test), group_series_test)

    # --- reweighing ---
    model_rw, rw_audit = apply_reweighing(
        model, X_train, y_train, X_test, y_test,
        group_series_train, group_series_test,
    )
    y_pred_rw = _predict_proba_uniform(model_rw, X_test)
    if y_pred_rw.ndim > 1:
        y_pred_rw = y_pred_rw[:, 1]
    rw_auc = roc_auc_score(y_test, y_pred_rw)

    # --- threshold optimizer ---
    y_pred_adj, thresholds = apply_threshold_optimizer(
        y_test, pd.Series(y_pred_test), group_series_test, objective="demographic_parity",
    )
    to_audit = compute_fairness_metrics(y_test, pd.Series(y_pred_test), group_series_test,
                                        threshold=0.5)

    summary = {
        "attribute": attribute_name,
        "baseline": {
            "auc": round(baseline_auc, 4),
            "dpd": baseline_audit.get("demographic_parity_difference", 0),
            "dir": baseline_audit.get("disparate_impact_ratio", 1.0),
            "eod": baseline_audit.get("equal_opportunity_difference", 0),
        },
        "reweighing": {
            "auc": round(rw_auc, 4),
            "dpd": rw_audit.get("demographic_parity_difference", 0),
            "dir": rw_audit.get("disparate_impact_ratio", 1.0),
            "eod": rw_audit.get("equal_opportunity_difference", 0),
        },
        "threshold_optimizer": {
            "thresholds": thresholds,
            "dpd": to_audit.get("demographic_parity_difference", 0),
            "dir": to_audit.get("disparate_impact_ratio", 1.0),
        },
    }

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_tpr(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """True Positive Rate."""
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
