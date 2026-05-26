"""XAI evaluation metrics.

Evaluates explanation quality along three axes:
  1. Faithfulness — does the explanation reflect what the model actually uses?
  2. Stability — do similar inputs produce similar explanations?
  3. Comprehensibility — is the explanation human-readable?

Reference:
  - "Towards A Rigorous Science of Interpretable Machine Learning" (Doshi-Velez & Kim 2017)
  - "Sanity Checks for Saliency Maps" (Adebayo et al. NeurIPS 2018)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Faithfulness: perturbation-based
# ---------------------------------------------------------------------------

def faithfulness_perturbation(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    shap_values: np.ndarray,
    top_k: int = 10,
    perturbation_ratio: float = 0.1,
    n_repeats: int = 10,
) -> Dict[str, Any]:
    """Measure faithfulness by removing top-k SHAP features and measuring prediction change.

    High-faithfulness explanation: removing top features → large prediction change.
    Low-faithfulness explanation: removing top features → small/no change (explanation lies).

    Returns:
      - top_k_drop: mean |Δprediction| when dropping top-k features
      - random_k_drop: mean |Δprediction| when dropping k random features (control)
      - faithfulness_score: top_k_drop / random_k_drop (> 1 means faithful)
      - per_run: list of (top_drop, random_drop) per repeat
    """
    from src.xai.explainer import _predict_proba_uniform

    n = len(feature_names)
    k = min(top_k, n)

    feat_rank = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    top_features = [feature_names[i] for i in feat_rank[:k]]

    top_drops = []
    random_drops = []

    for _ in range(n_repeats):
        X_perturbed = X.copy()

        # Drop top-k features (set to median)
        for f in top_features:
            X_perturbed[f] = X[f].median()

        base_preds = _predict_proba_uniform(model, X)
        pert_preds = _predict_proba_uniform(model, X_perturbed)
        if base_preds.ndim > 1:
            base_preds = base_preds[:, 1]
        if pert_preds.ndim > 1:
            pert_preds = pert_preds[:, 1]
        top_drops.append(np.abs(base_preds - pert_preds).mean())

        # Drop k random features (control)
        rand_features = np.random.choice(feature_names, size=k, replace=False)
        X_rand = X.copy()
        for f in rand_features:
            X_rand[f] = X[f].median()

        rand_preds = _predict_proba_uniform(model, X_rand)
        if rand_preds.ndim > 1:
            rand_preds = rand_preds[:, 1]
        random_drops.append(np.abs(base_preds - rand_preds).mean())

    top_mean, top_std = np.mean(top_drops), np.std(top_drops)
    rand_mean, rand_std = np.mean(random_drops), np.std(random_drops)
    score = top_mean / rand_mean if rand_mean > 1e-10 else 1.0

    return {
        "top_k_drop": round(top_mean, 6),
        "top_k_drop_std": round(top_std, 6),
        "random_k_drop": round(rand_mean, 6),
        "random_k_drop_std": round(rand_std, 6),
        "faithfulness_score": round(score, 4),
        "top_features": top_features,
        "k": k,
        "n_repeats": n_repeats,
    }


def faithfulness_monotonic(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    shap_values: np.ndarray,
    increments: Tuple[int, ...] = (1, 3, 5, 10, 20),
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """Check monotonicity: dropping more important features → monotonically larger Δpred.

    A faithful explanation should show that removing the top-1 feature matters more
    than removing the top-20 (since importance decays).

    Returns dict with:
      - curve: {k: mean_Δpred} for each k
      - spearman_r: rank correlation between k rank and Δpred rank
    """
    from src.xai.explainer import _predict_proba_uniform

    feat_rank = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    ranked_features = [feature_names[i] for i in feat_rank]
    base_preds = _predict_proba_uniform(model, X)
    if base_preds.ndim > 1:
        base_preds = base_preds[:, 1]

    curve = {}
    for k in increments:
        if k > len(ranked_features):
            break
        drops = []
        for _ in range(n_repeats):
            X_pert = X.copy()
            for f in ranked_features[:k]:
                X_pert[f] = X[f].median()
            pert_preds = _predict_proba_uniform(model, X_pert)
            if pert_preds.ndim > 1:
                pert_preds = pert_preds[:, 1]
            drops.append(np.abs(base_preds - pert_preds).mean())
        curve[k] = round(np.mean(drops), 6)

    ks = sorted(curve.keys())
    vals = [curve[k] for k in ks]
    sp_r, sp_p = spearmanr(range(len(ks)), vals)

    return {
        "curve": curve,
        "spearman_r": round(sp_r, 4),
        "spearman_p": round(sp_p, 4),
        "is_monotonic": bool(sp_r > 0.7 and sp_p < 0.05),
    }


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def explanation_stability(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    n_samples: int = 100,
    noise_scale: float = 0.01,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Measure SHAP stability under small input perturbations.

    For each of n_samples test instances, add small Gaussian noise (±1% of std)
    and compare the top-k feature ranking before vs after perturbation.

    Returns:
      - mean_jaccard: mean Jaccard similarity of top-k sets (0-1, higher = more stable)
      - mean_rank_corr: mean Spearman rank correlation of top-k ordering
    """
    import shap

    rng = np.random.RandomState(42)
    n_total = min(n_samples, len(X))
    idxs = rng.choice(len(X), size=n_total, replace=False)

    explainer = shap.TreeExplainer(model)

    jaccards = []
    rank_corrs = []

    for idx in idxs:
        x_orig = X.iloc[idx:idx+1].copy()
        sv_orig = _get_shap_for_instance(explainer, x_orig)
        if sv_orig is None:
            continue

        top_orig = set(np.argsort(np.abs(sv_orig[0]))[::-1][:top_k])

        # Add noise
        x_noisy = x_orig.copy()
        for c in x_noisy.columns:
            std = x_noisy[c].std()
            if std and std > 0:
                noise = rng.normal(0, noise_scale * std, size=1)
                x_noisy[c] = float(x_noisy[c].iloc[0]) + noise[0]

        sv_noisy = _get_shap_for_instance(explainer, x_noisy)
        if sv_noisy is None:
            continue

        top_noisy = set(np.argsort(np.abs(sv_noisy[0]))[::-1][:top_k])

        # Jaccard
        jaccard = len(top_orig & top_noisy) / len(top_orig | top_noisy) if top_orig | top_noisy else 1.0
        jaccards.append(jaccard)

        # Rank correlation
        try:
            sr, _ = spearmanr(np.abs(sv_orig[0]), np.abs(sv_noisy[0]))
            rank_corrs.append(sr if not np.isnan(sr) else 1.0)
        except Exception:
            rank_corrs.append(1.0)

    return {
        "mean_jaccard": round(np.mean(jaccards), 4) if jaccards else 0,
        "jaccard_std": round(np.std(jaccards), 4) if jaccards else 0,
        "mean_rank_corr": round(np.mean(rank_corrs), 4) if rank_corrs else 0,
        "rank_corr_std": round(np.std(rank_corrs), 4) if rank_corrs else 0,
        "n_tested": len(jaccards),
    }


# ---------------------------------------------------------------------------
# Comprehensibility
# ---------------------------------------------------------------------------

def comprehensibility_score(
    shap_result: Dict,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Heuristic comprehensibility scoring.

    Checks:
      - sparsity: how much of total |SHAP| is in top-k features (>0.7 = good)
      - concentration: is top-1 feature < 50% of total? (diverse drivers)
      - tail: how many features have near-zero contribution?
    """
    s = shap_result["mean_abs_shap"]
    total = s.sum()
    top_k_ratio = s.head(top_k).sum() / total if total > 0 else 0
    top1_ratio = s.iloc[0] / total if total > 0 else 0
    n_near_zero = (s < 0.01 * total).sum() if total > 0 else 0

    score = 0
    if top_k_ratio > 0.7:
        score += 2
    elif top_k_ratio > 0.5:
        score += 1
    if top1_ratio < 0.5:
        score += 1
    if n_near_zero > 0:
        score += 1
    score = min(score, 5)

    return {
        "score": score,
        "top_k_concentration": round(top_k_ratio, 4),
        "top1_ratio": round(top1_ratio, 4),
        "n_near_zero_features": int(n_near_zero),
        "interpretation": _comprehensibility_label(score),
    }


def _comprehensibility_label(score: int) -> str:
    if score >= 4:
        return "Excellent — top features dominate, diverse drivers, clear signal"
    if score >= 3:
        return "Good — acceptable concentration, moderate interpretability"
    if score >= 2:
        return "Fair — explanation is somewhat diffuse"
    return "Poor — explanation is spread across many features, hard to interpret"


# ---------------------------------------------------------------------------
# Full XAI evaluation report
# ---------------------------------------------------------------------------

def xai_evaluation_report(
    model: Any,
    X: pd.DataFrame,
    shap_result: Dict,
    top_k: int = 10,
    n_samples: int = 100,
) -> Dict[str, Any]:
    """Generate a comprehensive XAI evaluation report."""
    print("Computing XAI evaluation metrics...")
    print(f"  Faithfulness (perturbation)...")
    faith = faithfulness_perturbation(
        model, X, shap_result["feature_names"], shap_result["shap_values"],
        top_k=top_k, perturbation_ratio=0.1, n_repeats=10,
    )

    print(f"  Faithfulness (monotonic)...")
    faith_mono = faithfulness_monotonic(
        model, X, shap_result["feature_names"], shap_result["shap_values"],
    )

    print(f"  Stability...")
    stab = explanation_stability(
        model, X, shap_result["feature_names"],
        n_samples=n_samples, top_k=top_k,
    )

    print(f"  Comprehensibility...")
    comp = comprehensibility_score(shap_result, top_k=top_k)

    return {
        "faithfulness": faith,
        "faithfulness_monotonic": faith_mono,
        "stability": stab,
        "comprehensibility": comp,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_shap_for_instance(explainer, X_instance: pd.DataFrame) -> Optional[np.ndarray]:
    """Get SHAP values for a single instance, handling edge cases."""
    try:
        sv = explainer.shap_values(X_instance)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        return sv
    except Exception:
        try:
            sv = explainer.shap_values(X_instance.values)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            return sv
        except Exception:
            return None
