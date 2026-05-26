"""Explainable AI: SHAP, LIME, DiCE counterfactual explanations."""

from src.xai.explainer import (
    explain_shap,
    shap_feature_importance,
    explain_lime,
    explain_dice,
    compute_pdp,
    compute_pdp_multi,
    shap_interaction_values,
    ModelAdapter,
)

from src.xai.evaluation import (
    faithfulness_perturbation,
    faithfulness_monotonic,
    explanation_stability,
    comprehensibility_score,
    xai_evaluation_report,
)
