"""Tree-based models for credit risk scoring.

Supports: RandomForest, XGBoost, LightGBM, CatBoost.
Includes: default-param training, Optuna hyperparameter optimization,
class-imbalance strategies, and stacking/blending ensembles.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb_tuner  # for log_evaluation callback

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    target: str = "is_bad",
    drop_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Split feature matrix and target, separating numeric vs categorical.

    Drops: target, addr_state, zip_code (fairness proxies), text columns,
    and any explicitly requested columns.
    """
    always_drop = {
        target, "issue_year", "issue_d", "loan_status",
        "addr_state", "zip_code", "emp_title", "title", "desc",
        "id", "member_id", "url", "grade", "sub_grade",
    }
    always_drop.update(drop_cols or [])

    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols].copy()
    y = df[target].copy().astype(int)

    # Separate dtypes
    if cat_cols is None:
        cat_cols = [c for c in X.columns if X[c].dtype in ("object", "category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Label-encode categoricals to integers
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        # Fill NaN with a sentinel string so LabelEncoder doesn't break
        vals = X[c].fillna("__MISSING__").astype(str)
        X[c] = le.fit_transform(vals)
        encoders[c] = le

    return X, y, num_cols, cat_cols


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "rf": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 50,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "xgb": {
        "n_estimators": 2000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 4.0,
        "tree_method": "hist",
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": -1,
    },
    "lgb": {
        "n_estimators": 2000,
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "is_unbalance": True,
        "metric": "auc",
        "early_stopping_rounds": 50,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    },
    "cb": {
        "iterations": 2000,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "border_count": 128,
        "eval_metric": "AUC",
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": 0,
    },
}


def make_model(name: str, extra_params: Optional[Dict[str, Any]] = None) -> Any:
    """Create a tree model instance with sensible defaults."""
    params = dict(MODEL_DEFAULTS.get(name, {}))
    if extra_params:
        params.update(extra_params)

    if name == "rf":
        return RandomForestClassifier(**params)

    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)

    if name == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)

    if name == "cb":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)

    raise ValueError(f"Unknown model: {name}. Choose rf/xgb/lgb/cb.")


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    cat_features: Optional[List[str]] = None,
) -> Any:
    """Fit a model with optional early stopping on validation set.

    Handles model-specific fit() signatures (XGBoost/LightGBM/CatBoost
    pass eval_set; sklearn RF ignores it).
    """
    model_name = type(model).__name__.lower()

    if model_name == "xgbclassifier":
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return model

    if model_name == "lgbmclassifier":
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        model.fit(X_train, y_train, eval_set=eval_set,
                  callbacks=[lgb_tuner.log_evaluation(50)])
        return model

    if model_name == "catboostclassifier":
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            cat_features=cat_features or [],
            verbose=False,
        )
        return model

    # sklearn RF
    model.fit(X_train.values, y_train.values)
    return model


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict probability of positive class (is_bad=1)."""
    model_name = type(model).__name__.lower()

    if model_name == "catboostclassifier":
        return model.predict_proba(X)[:, 1]

    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return model.predict_proba(X.values)[:, 1]


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimization
# ---------------------------------------------------------------------------

OPTUNA_SPACES: Dict[str, Dict[str, Any]] = {
    "xgb": {
        "learning_rate": ("float", 0.01, 0.3, True),
        "max_depth": ("int", 3, 10),
        "min_child_weight": ("int", 1, 10),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_alpha": ("float", 0.0, 1.0),
        "reg_lambda": ("float", 0.0, 1.0),
        "scale_pos_weight": ("float", 1.0, 10.0),
    },
    "lgb": {
        "learning_rate": ("float", 0.01, 0.3, True),
        "num_leaves": ("int", 15, 255),
        "min_child_samples": ("int", 5, 100),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.6, 1.0),
        "reg_alpha": ("float", 0.0, 1.0),
        "reg_lambda": ("float", 0.0, 1.0),
    },
    "cb": {
        "learning_rate": ("float", 0.01, 0.3, True),
        "depth": ("int", 3, 10),
        "l2_leaf_reg": ("float", 1.0, 10.0),
        "border_count": ("int", 32, 255),
    },
}


def _suggest_param(trial, name: str, spec: tuple):
    """Suggest a single parameter for an Optuna trial."""
    kind = spec[0]
    low, high = spec[1], spec[2]
    log = spec[3] if len(spec) > 3 else False

    if kind == "float":
        return trial.suggest_float(name, low, high, log=log)
    if kind == "int":
        return trial.suggest_int(name, low, high)
    raise ValueError(f"Unknown param kind: {kind}")


def _make_optuna_model(model_name: str, trial, extra: Optional[Dict] = None):
    """Build a model instance with trial-suggested hyperparameters."""
    space = OPTUNA_SPACES.get(model_name, {})
    trial_params = {k: _suggest_param(trial, k, v) for k, v in space.items()}

    defaults = dict(MODEL_DEFAULTS.get(model_name, {}))
    params = {**defaults, **trial_params}
    if extra:
        params.update(extra)

    # Remove early-stopping for optuna trials (handled separately)
    params.pop("early_stopping_rounds", None)

    return make_model(model_name, params)


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    cat_features: Optional[List[str]] = None,
    extra_params: Optional[Dict] = None,
    timeout: Optional[int] = None,
    save_dir: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Run Optuna hyperparameter optimization for a tree model.

    If save_dir is provided, saves the best model as '{model_name}_optuna_stage3.pkl'.
    Returns (best_model, best_params, study_trials_df).
    """
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = _make_optuna_model(model_name, trial, extra_params)
        train_model(
            model, X_train, y_train, X_val, y_val,
            cat_features=cat_features,
        )
        preds = predict_proba(model, X_val)
        return roc_auc_score(y_val, preds)

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Re-train best model with full config (including early stopping)
    best_params = study.best_params
    defaults = dict(MODEL_DEFAULTS.get(model_name, {}))
    full_params = {**defaults, **best_params}
    if extra_params:
        full_params.update(extra_params)

    best_model = make_model(model_name, full_params)
    train_model(
        best_model, X_train, y_train, X_val, y_val,
        cat_features=cat_features,
    )

    # Save model to disk so kernel crash won't require retraining
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{model_name}_optuna_stage3.pkl'
        save_model(best_model, save_path)
        print(f'  Model saved to {save_path}')

    trials_df = study.trials_dataframe().sort_values("value", ascending=False)

    return best_model, best_params, trials_df


# ---------------------------------------------------------------------------
# Class imbalance strategies
# ---------------------------------------------------------------------------

def get_imbalance_params(model_name: str, strategy: str, y_train: pd.Series) -> Dict[str, Any]:
    """Return model-specific params for a given imbalance-handling strategy.

    Strategies:
      - none: no adjustment
      - balanced: class_weight='balanced' (RF) or is_unbalance=True (LGB)
      - scale_pos_weight: computed from class ratio for XGB/CB
      - smote: apply SMOTE oversampling (handled externally)
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    ratio = n_neg / n_pos

    if strategy == "none":
        return {}

    if strategy == "balanced":
        if model_name == "rf":
            return {"class_weight": "balanced"}
        if model_name == "xgb":
            return {"scale_pos_weight": ratio}
        if model_name == "lgb":
            return {"is_unbalance": True}
        if model_name == "cb":
            return {"scale_pos_weight": ratio}
        return {}

    if strategy == "scale_pos_weight":
        if model_name == "lgb":
            return {"class_weight": "balanced"}
        if model_name == "rf":
            return {"class_weight": "balanced"}
        return {"scale_pos_weight": ratio}

    return {}


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

@dataclass
class StackingEnsemble:
    """Two-level stacking: base models → meta-learner (LR)."""

    base_models: List[Any]
    meta_model: Any = None
    base_names: List[str] = field(default_factory=list)
    cv_folds: int = 5

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
    ):
        """Train meta-learner on validation-set predictions from already-trained base models.

        Base models are assumed already trained — no retraining. Meta-learner (LR)
        is fit on val-set predictions, which is fast and avoids overfitting to train.
        """
        self.base_names = [type(m).__name__ for m in self.base_models]
        self.meta_model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)

        if X_val is not None and y_val is not None:
            meta_X = np.column_stack([
                predict_proba(m, X_val) for m in self.base_models
            ])
            meta_y = y_val
        else:
            # Fallback: use train predictions (slightly optimistic bias but fast)
            meta_X = np.column_stack([
                predict_proba(m, X_train) for m in self.base_models
            ])
            meta_y = y_train

        self.meta_model.fit(meta_X, meta_y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Stacked probability predictions."""
        base_preds = np.column_stack([
            predict_proba(m, X) for m in self.base_models
        ])
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def predict_score(self, X: pd.DataFrame, base_score=600, pdo=20, base_odds=50) -> np.ndarray:
        """Convert probabilities to credit scores (300-850 scale)."""
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        odds = (1 - proba) / proba
        return np.clip(offset + factor * np.log(odds), 300, 850)


def blend_predictions(
    model_preds: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Weighted average blending of model predictions.

    If weights is None, equal-weight blending is used.
    """
    if weights is None:
        weights = {k: 1.0 for k in model_preds}

    total_w = sum(weights.values())
    blended = sum(preds * weights[name] for name, preds in model_preds.items())
    return blended / total_w


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

from src.evaluation.metrics import auc_score, ks_score, gini_score, pr_auc


def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str = "",
    fit_time: float = 0.0,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics across all splits."""
    preds = {name: predict_proba(model, X) for name, X in [
        ("train", X_train), ("val", X_val), ("test", X_test)
    ]}
    ys = {"train": y_train, "val": y_val, "test": y_test}

    result = {"model": label, "fit_time_sec": round(fit_time, 1)}
    for split in ("train", "val", "test"):
        yt, yp = ys[split].values, preds[split]
        result[f"{split}_auc"] = round(auc_score(yt, yp), 4)
        result[f"{split}_ks"] = round(ks_score(yt, yp), 4)
        result[f"{split}_gini"] = round(gini_score(yt, yp), 4)
        result[f"{split}_pr_auc"] = round(pr_auc(yt, yp), 4)

    return result


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert list of evaluation dicts to a sorted comparison DataFrame."""
    df = pd.DataFrame(results)
    if "test_auc" in df.columns:
        df = df.sort_values("test_auc", ascending=False)
    return df.reset_index(drop=True)


def save_model(model: Any, path: str | Path):
    """Save a model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> Any:
    """Load a model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
