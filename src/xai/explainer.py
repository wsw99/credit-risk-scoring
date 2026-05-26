"""Explainable AI wrappers for credit scoring models.

Global explanations:
  - SHAP summary (bar, dot, violin), feature importance (gain/permutation)
  - Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE)

Local explanations:
  - SHAP waterfall / force plot for single predictions
  - LIME local surrogate explanations
  - DiCE counterfactual explanations ("what would change the decision?")

All wrappers accept tree models (XGBoost, LightGBM, CatBoost, RandomForest)
or the Stage 3 stacking ensemble via a uniform predict_proba interface.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# SHAP global explanations
# ---------------------------------------------------------------------------

def explain_shap(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 20,
    method: str = "tree",
    nsamples: int = 200,
) -> Dict[str, Any]:
    """Compute SHAP values for a tree model.

    Uses TreeExplainer for XGBoost/LGBM/CB/RF (fast, exact).
    Falls back to KernelExplainer for non-tree models.

    Returns dict with:
      - shap_values: np.ndarray (n_samples, n_features)
      - expected_value: float (base value)
      - feature_names: list[str]
      - mean_abs_shap: pd.Series sorted descending
    """
    import shap

    model_type = type(model).__name__.lower()

    if method == "tree" and _is_tree_model(model):
        # Try model-agnostic approach first
        try:
            # LightGBM/CatBoost/XGBoost need feature names to match
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        except Exception:
            # Fallback: pass numpy array
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.values)

        # CatBoost and some versions return a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        # KernelExplainer — sample background
        bg = shap.sample(X, min(nsamples, len(X)), random_state=42)
        explainer = shap.KernelExplainer(
            lambda x: _predict_proba_uniform(model, pd.DataFrame(x, columns=X.columns)),
            bg.values if hasattr(bg, "values") else bg,
        )
        shap_values = explainer.shap_values(X[:nsamples], nsamples=nsamples // 10)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        expected_value = explainer.expected_value

    feature_names = list(X.columns)
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    return {
        "shap_values": shap_values,
        "expected_value": float(expected_value),
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
    }


def shap_feature_importance(shap_result: Dict, top_k: int = 20) -> pd.DataFrame:
    """Return top-k features ranked by mean |SHAP| value."""
    s = shap_result["mean_abs_shap"]
    df = s.head(top_k).reset_index()
    df.columns = ["feature", "mean_abs_shap"]
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# LIME local explanations
# ---------------------------------------------------------------------------

def explain_lime(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int = 0,
    num_features: int = 10,
    num_samples: int = 5000,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate LIME explanation for a single instance.

    Returns:
      - lime_explanation: the LIME Explanation object
      - local_exp: dict mapping class index → list of (feature, weight)
      - instance: the explained row as dict
    """
    import lime
    import lime.lime_tabular

    if class_names is None:
        class_names = ["Fully Paid", "Default"]

    # Drop non-numeric columns — LIME can't perturb them and they'd cause model failures
    if hasattr(X, "select_dtypes"):
        X = X.select_dtypes(include=[np.number])

    # Use numpy array for LIME; drop zero-variance columns to avoid truncnorm scale=0 error
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    all_feature_names = list(X.columns)

    col_std = X_np.std(axis=0)
    keep = col_std > 0

    # constant values for dropped columns — needed to reconstruct full input for model
    dropped_const = {
        name: X_np[0, i]
        for i, (name, k) in enumerate(zip(all_feature_names, keep)) if not k
    }

    if not keep.all():
        X_np_lime = X_np[:, keep]
        feature_names = [f for f, k in zip(all_feature_names, keep) if k]
    else:
        X_np_lime = X_np
        feature_names = all_feature_names

    def _predict_fn(x):
        df = pd.DataFrame(x, columns=feature_names)
        for col_name, val in dropped_const.items():
            df[col_name] = val
        df = df[all_feature_names]  # restore original column order
        return _predict_proba_uniform(model, df)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_np_lime,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=False,
        sample_around_instance=True,
        random_state=42,
    )

    instance = X_np_lime[instance_idx]
    exp = explainer.explain_instance(
        instance,
        _predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    _p = _predict_proba_uniform(model, X.iloc[[instance_idx]])
    predicted_proba = float(_p[0, 1] if _p.ndim == 2 else _p[0])

    return {
        "lime_explanation": exp,
        "local_exp": exp.as_list(),
        "instance": dict(zip(all_feature_names, X.iloc[instance_idx].values)),
        "predicted_proba": predicted_proba,
    }


# ---------------------------------------------------------------------------
# DiCE counterfactual explanations
# ---------------------------------------------------------------------------

def explain_dice(
    model: Any,
    X: pd.DataFrame,
    instance_idx: int = 0,
    num_counterfactuals: int = 3,
    desired_class: int = 0,  # "Fully Paid" / approved
    features_to_vary: Optional[List[str]] = None,
    permitted_range: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Generate DiCE counterfactual explanations.

    Answers: "What would need to change for this applicant to be approved?"

    Args:
        model: trained tree model
        X: feature DataFrame
        instance_idx: which row to explain
        num_counterfactuals: how many CF examples to generate
        desired_class: 0 = approved (Fully Paid), 1 = rejected (Default)
        features_to_vary: subset of features DiCE may modify
        permitted_range: {feature: [min, max]} constraints

    Returns:
        dict with cfs (counterfactual instances), original, and changes
    """
    import dice_ml
    from dice_ml import Dice

    feature_names = list(X.columns)

    # DiCE needs a DataFrame with the target column
    query_instance = X.iloc[[instance_idx]].copy()

    # Build DiCE data object
    d = dice_ml.Data(
        dataframe=pd.concat([X, pd.Series(np.zeros(len(X)), name="dummy_target")], axis=1),
        continuous_features=_get_continuous_features(X),
        outcome_name="dummy_target",
    )

    # Wrap model for DiCE
    backend = "sklearn"
    m = dice_ml.Model(model=ModelAdapter(model, feature_names), backend=backend)

    exp = Dice(d, m, method="random")

    if features_to_vary is None:
        features_to_vary = feature_names

    cf = exp.generate_counterfactuals(
        query_instance.drop(columns=["dummy_target"] if "dummy_target" in query_instance.columns else []),
        total_CFs=num_counterfactuals,
        desired_class=desired_class,
        features_to_vary=features_to_vary,
        permitted_range=permitted_range,
    )

    return {
        "counterfactuals": cf,
        "original_instance": X.iloc[instance_idx].to_dict(),
        "query_idx": instance_idx,
        "desired_class": desired_class,
    }


# ---------------------------------------------------------------------------
# PDP / ICE
# ---------------------------------------------------------------------------

def compute_pdp(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_points: int = 30,
    ice: bool = False,
) -> Dict[str, Any]:
    """Compute Partial Dependence for a single feature.

    Args:
        model: trained model
        X: feature DataFrame
        feature: column name
        grid_points: number of grid values
        ice: if True, also return ICE curves

    Returns:
        dict with grid, pdp_values, (optional ice_values)
    """
    col = X[feature]
    grid = np.linspace(col.quantile(0.01), col.quantile(0.99), grid_points)

    X_copy = X.copy()
    pdp_vals = np.zeros(grid_points)
    ice_vals = np.zeros((len(X), grid_points)) if ice else None

    for i, val in enumerate(grid):
        X_copy[feature] = val
        preds = _predict_proba_uniform(model, X_copy)
        pdp_vals[i] = preds.mean()
        if ice:
            ice_vals[:, i] = preds

    return {
        "feature": feature,
        "grid": grid,
        "pdp_values": pdp_vals,
        "ice_values": ice_vals,
    }


def compute_pdp_multi(
    model: Any,
    X: pd.DataFrame,
    features: List[str],
    grid_points: int = 30,
) -> Dict[str, Dict]:
    """Compute PDP for multiple features."""
    return {f: compute_pdp(model, X, f, grid_points, ice=False) for f in features}


# ---------------------------------------------------------------------------
# Feature interaction (SHAP)
# ---------------------------------------------------------------------------

def shap_interaction_values(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 10,
    sample_size: int = 500,
) -> Dict[str, Any]:
    """Compute SHAP interaction values for top features.

    Returns interaction matrix for a sample of instances.
    """
    import shap

    X_sample = X.sample(min(sample_size, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X_sample)

    if isinstance(interaction_values, list):
        interaction_values = interaction_values[1]

    # Average absolute interaction over samples
    mean_interaction = np.abs(interaction_values).mean(axis=0)

    return {
        "interaction_matrix": mean_interaction,
        "feature_names": list(X.columns),
        "sample_size": len(X_sample),
    }


# ---------------------------------------------------------------------------
# Model adapter for DiCE (sklearn-compatible wrapper)
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Wrap a non-sklearn tree model to expose sklearn-style API for DiCE."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 2) probability array."""
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        p1 = _predict_proba_uniform(self.model, X_df)
        if p1.ndim == 1:
            p1 = p1.reshape(-1, 1)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_tree_model(model: Any) -> bool:
    """Check if model is a tree-based model supported by SHAP TreeExplainer."""
    name = type(model).__name__.lower()
    tree_types = (
        "xgbclassifier", "xgboost", "booster",
        "lgbmclassifier", "catboostclassifier", "catboost",
        "randomforestclassifier",
        "gradientboostingclassifier",
        "extratreesclassifier",
        "histgradientboostingclassifier",
    )
    return any(t in name for t in tree_types)


def _predict_proba_uniform(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict probability of positive class.

    Handles sklearn, XGBoost, LightGBM, CatBoost, and StackingEnsemble.
    """
    model_type = type(model).__name__.lower()

    try:
        # StackingEnsemble
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)

        if "catboost" in model_type:
            return model.predict_proba(X)[:, 1]

        # Default sklearn-style
        proba = model.predict_proba(X)
        return proba if proba.ndim == 1 else proba[:, 1]
    except Exception:
        # Last resort: raw values
        return model.predict_proba(X.values)[:, 1]


def _get_continuous_features(X: pd.DataFrame) -> List[str]:
    """Return list of continuous (numeric) feature names."""
    return [c for c in X.columns if X[c].dtype in ("float64", "float32", "int64", "int32")
            and X[c].nunique() > 10]
