"""Fairness audit for credit scoring models.

Computes fairness metrics across protected attributes:
  - Demographic Parity Difference (DPD): approval-rate gap
  - Equal Opportunity Difference (EOD): true-positive-rate gap
  - Equalized Odds Difference: TPR + FPR gap
  - Disparate Impact Ratio (DIR): < 0.8 triggers US EEOC "80% rule"

Protected attributes (from LendingClub data):
  - income_band: binned annual_inc
  - age_proxy: binned emp_length (employment years — proxy for age)
  - region: addr_state (if joined back from raw data)

Reference:
  - Hardt et al. (NeurIPS 2016) — "Equality of Opportunity"
  - Fairlearn documentation
  - US ECOA / EEOC 80% rule
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Protected attribute creation
# ---------------------------------------------------------------------------

def create_protected_attributes(
    df: pd.DataFrame,
    income_col: str = "annual_inc",
    emp_length_col: str = "emp_length",
    region_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create categorical protected-attribute columns from raw features.

    Args:
        df: DataFrame with applicant features
        income_col: annual income column name
        emp_length_col: employment length column name
        region_col: state/region column name (e.g., addr_state)

    Returns:
        DataFrame with added columns: income_band, age_proxy, (region)
    """
    df = df.copy()

    # Income bands
    if income_col in df.columns:
        df["income_band"] = pd.cut(
            df[income_col].clip(lower=0),
            bins=[0, 30000, 60000, 100000, 200000, np.inf],
            labels=["<30K", "30-60K", "60-100K", "100-200K", ">200K"],
        )

    # Age proxy from employment length (in years)
    if emp_length_col in df.columns:
        # emp_length may be string ("10+ years", "< 1 year") or numeric
        emp_map = {
            "< 1 year": 0,
            "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
            "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8,
            "9 years": 9, "10+ years": 10,
        }
        df["emp_years"] = df[emp_length_col].map(emp_map)
        if df["emp_years"].isnull().any():
            # Fallback: try numeric conversion
            numeric_vals = pd.to_numeric(df[emp_length_col], errors="coerce")
            df.loc[df["emp_years"].isnull(), "emp_years"] = numeric_vals
        df["emp_years"] = df["emp_years"].fillna(-1)

        df["age_proxy"] = pd.cut(
            df["emp_years"],
            bins=[-2, 2, 5, 10, 15, 100],
            labels=["0-2yr", "3-5yr", "6-10yr", "11-15yr", ">15yr"],
        )

    # Region (if available)
    if region_col and region_col in df.columns:
        df["region"] = df[region_col]

    return df


def suggest_privileged_groups(X: pd.DataFrame, attr: str) -> Tuple[List, str]:
    """Suggest which group(s) to consider as privileged for a protected attribute.

    Heuristic: pick the group with the lowest default rate (most advantaged).
    """
    if attr not in X.columns:
        raise ValueError(f"Attribute '{attr}' not in DataFrame")

    # Find labels with the most samples (dominant group)
    counts = X[attr].value_counts()
    if len(counts) < 2:
        raise ValueError(f"Need at least 2 groups for '{attr}', found {len(counts)}")

    # Default is the largest group (conservative choice for fairness analysis)
    privileged = counts.index[0]
    all_groups = counts.index.tolist()

    return all_groups, privileged


# ---------------------------------------------------------------------------
# Fairness metrics (mirror fairlearn but explicit)
# ---------------------------------------------------------------------------

def compute_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    group_series: pd.Series,
    threshold: float = 0.5,
    privileged_group: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute per-group and aggregate fairness metrics.

    Args:
        y_true: binary labels (1 = default/bad)
        y_pred: model predicted probabilities
        group_series: protected-attribute category per sample
        threshold: decision threshold for predict_proba → binary prediction
        privileged_group: label of privileged group (default: most frequent)

    Returns dict with:
      - per_group: {group: {count, default_rate, approval_rate, tpr, fpr, precision}}
      - demographic_parity_difference
      - equal_opportunity_difference
      - equalized_odds_difference
      - disparate_impact_ratio
    """
    # Normalise indices so boolean masks align regardless of caller's index
    y_true = pd.Series(y_true.values if hasattr(y_true, "values") else y_true)
    y_pred = pd.Series(y_pred.values if hasattr(y_pred, "values") else y_pred)
    group_series = pd.Series(group_series.values if hasattr(group_series, "values") else group_series)

    y_pred_binary = (y_pred >= threshold).astype(int)

    # approved = predicted non-default (0), rejected = predicted default (1)
    # NOTE: For credit scoring, "positive outcome" = approved (y_pred_binary == 0)
    # Fairness is about approval gaps between groups

    groups = group_series.dropna().unique()
    if privileged_group is None:
        privileged_group = group_series.value_counts().index[0]

    per_group = {}
    for g in groups:
        mask = (group_series == g) & y_true.notna()
        if mask.sum() < 10:
            continue

        yt_g = y_true[mask].values
        yp_g = y_pred_binary[mask] if isinstance(y_pred_binary, np.ndarray) else y_pred_binary[mask].values

        n = len(yt_g)
        n_bad = int(yt_g.sum())
        n_approved = int((yp_g == 0).sum())

        tp = int(((yt_g == 1) & (yp_g == 1)).sum())
        fp = int(((yt_g == 0) & (yp_g == 1)).sum())
        tn = int(((yt_g == 0) & (yp_g == 0)).sum())
        fn = int(((yt_g == 1) & (yp_g == 0)).sum())

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        per_group[str(g)] = {
            "count": n,
            "default_rate": round(float(n_bad / n), 4),
            "approval_rate": round(float(n_approved / n), 4),
            "tpr": round(float(tpr), 4),
            "fpr": round(float(fpr), 4),
            "precision": round(float(precision), 4),
        }

    if len(per_group) < 2:
        return {"error": f"Only {len(per_group)} groups with sufficient data", "per_group": per_group}

    approval_rates = [v["approval_rate"] for v in per_group.values()]
    tprs = [v["tpr"] for v in per_group.values()]
    fprs = [v["fpr"] for v in per_group.values()]

    priv_ar = per_group[str(privileged_group)]["approval_rate"]
    priv_tpr = per_group[str(privileged_group)]["tpr"]
    priv_fpr = per_group[str(privileged_group)]["fpr"]

    unpriv = [g for g in per_group if g != str(privileged_group)]
    unpriv_ar = max((per_group[g]["approval_rate"] for g in unpriv), default=priv_ar)
    unpriv_tpr = max((per_group[g]["tpr"] for g in unpriv), default=priv_tpr)
    unpriv_fpr = max((per_group[g]["fpr"] for g in unpriv), default=priv_fpr)

    dpd = priv_ar - min(per_group[g]["approval_rate"] for g in per_group)
    eod = priv_tpr - min(per_group[g]["tpr"] for g in per_group)
    eodd = max(abs(priv_tpr - v["tpr"]) for v in per_group.values()) + \
           max(abs(priv_fpr - v["fpr"]) for v in per_group.values())

    # Disparate Impact Ratio = min(approval_rate) / max(approval_rate)
    min_ar = min(approval_rates)
    max_ar = max(approval_rates)
    dir_score = min_ar / max_ar if max_ar > 0 else 1.0

    return {
        "per_group": per_group,
        "privileged_group": str(privileged_group),
        "demographic_parity_difference": round(float(dpd), 4),
        "equal_opportunity_difference": round(float(eod), 4),
        "equalized_odds_difference": round(float(eodd), 4),
        "disparate_impact_ratio": round(float(dir_score), 4),
        "dir_violation": bool(dir_score < 0.8),
        "dpd_violation": bool(abs(dpd) > 0.1),
        "eod_violation": bool(abs(eod) > 0.1),
    }


# ---------------------------------------------------------------------------
# Full fairness audit
# ---------------------------------------------------------------------------

def fairness_audit(
    model: Any,
    X: pd.DataFrame,
    y_true: pd.Series,
    protected_attrs: List[str],
    threshold: float = 0.5,
    predict_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    """Run fairness audit across multiple protected attributes.

    Args:
        model: trained model
        X: feature DataFrame (must include protected-attribute columns)
        y_true: true labels
        protected_attrs: column names for fairness grouping
        threshold: probability threshold for classification
        predict_fn: optional custom predict_proba function

    Returns dict with:
      - global_metrics: overall performance
      - per_attribute: {attr: fairness_metrics}
      - summary: worst violations across all attributes
    """
    from src.xai.explainer import _predict_proba_uniform
    import numpy as np

    # Filter X to only the features the model was trained on.
    # StackingEnsemble exposes base_models; individual sklearn/lgbm models
    # store feature names in feature_names_in_ or feature_name_.
    def _model_feature_names(m):
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
        if hasattr(m, "feature_name_"):
            return list(m.feature_name_)
        if hasattr(m, "base_models") and m.base_models:
            return _model_feature_names(m.base_models[0])
        return None

    feat_cols = _model_feature_names(model)
    X_for_pred = X[feat_cols] if feat_cols is not None else X.select_dtypes(include=[np.number])

    if predict_fn is None:
        y_pred = _predict_proba_uniform(model, X_for_pred)
    else:
        y_pred = predict_fn(model, X_for_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]

    results = {"global": {"n_samples": len(y_true), "default_rate": round(float(y_true.mean()), 4)}}

    per_attr = {}
    violations = []

    for attr in protected_attrs:
        if attr not in X.columns:
            per_attr[attr] = {"error": f"Column '{attr}' not found in data"}
            continue

        groups = X[attr].dropna().unique()
        counts = X[attr].value_counts()
        dominant = counts.index[0]

        metrics = compute_fairness_metrics(
            y_true, y_pred, X[attr], threshold=threshold, privileged_group=dominant,
        )

        per_attr[attr] = metrics
        if "dir_violation" in metrics:
            if metrics["dir_violation"]:
                violations.append(f"{attr}: DIR={metrics['disparate_impact_ratio']:.3f} (< 0.8)")
            if metrics["dpd_violation"]:
                violations.append(f"{attr}: DPD={metrics['demographic_parity_difference']:.3f} (> 0.1)")
            if metrics["eod_violation"]:
                violations.append(f"{attr}: EOD={metrics['equal_opportunity_difference']:.3f} (> 0.1)")

    results["per_attribute"] = per_attr
    results["summary"] = {
        "n_attributes_checked": len(protected_attrs),
        "n_violations": len(violations),
        "violations": violations,
        "overall_pass": len(violations) == 0,
    }

    return results


def compare_model_fairness(
    model_results: Dict[str, Dict],
) -> pd.DataFrame:
    """Compare fairness metrics across multiple models/strategies.

    Args:
        model_results: {model_name: fairness_audit_result}

    Returns:
        DataFrame with one row per (model, attribute) pair
    """
    rows = []
    for model_name, audit in model_results.items():
        for attr, metrics in audit.get("per_attribute", {}).items():
            if "error" in metrics:
                continue
            rows.append({
                "model": model_name,
                "attribute": attr,
                "DPD": metrics.get("demographic_parity_difference"),
                "EOD": metrics.get("equal_opportunity_difference"),
                "EODD": metrics.get("equalized_odds_difference"),
                "DIR": metrics.get("disparate_impact_ratio"),
                "DPD_violation": metrics.get("dpd_violation"),
                "DIR_violation": metrics.get("dir_violation"),
            })
    return pd.DataFrame(rows)
