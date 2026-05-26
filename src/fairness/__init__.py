"""Fairness audit and mitigation for credit scoring models.

Metrics:
  - Demographic Parity Difference (DPD)
  - Equal Opportunity Difference (EOD)
  - Equalized Odds Difference
  - Disparate Impact Ratio (DIR)

Mitigation:
  - Reweighing (pre-processing)
  - Threshold Optimizer (post-processing)
  - Trade-off analysis
"""

from src.fairness.audit import (
    create_protected_attributes,
    compute_fairness_metrics,
    fairness_audit,
    compare_model_fairness,
    suggest_privileged_groups,
)

from src.fairness.mitigation import (
    compute_reweighing_weights,
    apply_reweighing,
    optimize_thresholds,
    apply_threshold_optimizer,
    fairness_tradeoff_curve,
    mitigation_comparison,
)
