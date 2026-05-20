"""Calibration diagnostics and Population Stability Index (PSI).

- PSI: measures distribution shift between reference and comparison population
- Score distribution plots
- Default rate by score band
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Population Stability Index.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Interpretation:
        < 0.1  — no significant shift
        0.1-0.2 — moderate shift, investigate
        > 0.2  — significant shift, model may need recalibration

    Parameters
    ----------
    expected : np.ndarray
        Reference distribution (e.g., train scores/probabilities).
    actual : np.ndarray
        Comparison distribution (e.g., test scores/probabilities).
    bins : int
        Number of bins.
    strategy : str
        'quantile' (equal-frequency) or 'uniform' (equal-width).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(np.concatenate([expected, actual]), quantiles)
        bin_edges = np.unique(bin_edges)  # deduplicate
    else:
        all_vals = np.concatenate([expected, actual])
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    if len(bin_edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Avoid zero proportions
    exp_props = np.clip(exp_counts / exp_counts.sum(), 1e-10, 1)
    act_props = np.clip(act_counts / act_counts.sum(), 1e-10, 1)

    psi_val = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
    return float(psi_val)


def score_distribution_stats(
    scores_train: np.ndarray,
    scores_test: np.ndarray,
    scores_val: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Summary statistics for score distributions."""
    stats = {
        "dataset": ["Train", "Test"],
        "count": [len(scores_train), len(scores_test)],
        "mean": [float(np.mean(scores_train)), float(np.mean(scores_test))],
        "std": [float(np.std(scores_train)), float(np.std(scores_test))],
        "min": [float(np.min(scores_train)), float(np.min(scores_test))],
        "p5": [float(np.percentile(scores_train, 5)), float(np.percentile(scores_test, 5))],
        "p25": [float(np.percentile(scores_train, 25)), float(np.percentile(scores_test, 25))],
        "median": [float(np.median(scores_train)), float(np.median(scores_test))],
        "p75": [float(np.percentile(scores_train, 75)), float(np.percentile(scores_test, 75))],
        "p95": [float(np.percentile(scores_train, 95)), float(np.percentile(scores_test, 95))],
        "max": [float(np.max(scores_train)), float(np.max(scores_test))],
    }

    if scores_val is not None:
        stats["dataset"].append("Val")
        stats["count"].append(len(scores_val))
        stats["mean"].append(float(np.mean(scores_val)))
        stats["std"].append(float(np.std(scores_val)))
        stats["min"].append(float(np.min(scores_val)))
        stats["p5"].append(float(np.percentile(scores_val, 5)))
        stats["p25"].append(float(np.percentile(scores_val, 25)))
        stats["median"].append(float(np.median(scores_val)))
        stats["p75"].append(float(np.percentile(scores_val, 75)))
        stats["p95"].append(float(np.percentile(scores_val, 95)))
        stats["max"].append(float(np.max(scores_val)))

    return pd.DataFrame(stats)


def default_rate_by_score_band(
    scores: np.ndarray,
    y_true: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Calculate default rate for each score band."""
    _, bin_edges = pd.qcut(scores, q=bins, retbins=True, duplicates="drop")
    bin_edges = np.unique(np.round(bin_edges, -1))  # round to nearest 10

    bands = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= low) & (scores < high)
        if mask.sum() > 0:
            bands.append({
                "score_low": int(low),
                "score_high": int(high),
                "count": int(mask.sum()),
                "pct": round(float(mask.sum()) / len(scores) * 100, 1),
                "default_rate": round(float(y_true[mask].mean()) * 100, 2),
                "cum_pct": round(float(mask.sum()) / len(scores) * 100, 1),
            })

    df = pd.DataFrame(bands)
    df["cum_pct"] = df["pct"].cumsum()
    return df
