"""Feature selection for scorecard development.

- iv_filter: keep features with IV above threshold
- vif_filter: remove features with VIF above threshold (iteratively)
- correlation_filter: among highly correlated pairs, keep the one with higher IV
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def iv_filter(
    iv_series: pd.Series,
    threshold: float = 0.02,
    return_dropped: bool = False,
) -> List[str] | Tuple[List[str], List[str]]:
    """Keep features with Information Value >= threshold.

    IV interpretation:
        < 0.02  — unpredictive
        0.02-0.1 — weak
        0.1-0.3  — medium
        > 0.3    — strong
    """
    keep = iv_series[iv_series >= threshold].index.tolist()
    dropped = iv_series[iv_series < threshold].index.tolist()

    if return_dropped:
        return keep, dropped
    return keep


def vif_filter(
    df: pd.DataFrame,
    threshold: float = 5.0,
    return_dropped: bool = False,
) -> List[str] | Tuple[List[str], List[str], pd.DataFrame]:
    """Iteratively remove features with Variance Inflation Factor > threshold.

    VIF = 1 / (1 - R²) where R² from regressing feature on all others.
    VIF > 5 (or 10) indicates high multicollinearity.
    """
    features = list(df.columns)
    dropped = []
    vif_history = []

    while True:
        vif = _calculate_vif(df[features])
        vif_history.append(vif)

        max_vif_feat = vif.idxmax()
        max_vif_val = vif.max()

        if max_vif_val <= threshold:
            break
        if len(features) <= 2:
            break

        features.remove(max_vif_feat)
        dropped.append(max_vif_feat)

    vif_final = vif_history[-1] if vif_history else pd.Series(dtype=float)

    if return_dropped:
        return features, dropped, vif_final
    return features


def _calculate_vif(df: pd.DataFrame) -> pd.Series:
    """Compute VIF for each column in df."""
    df = df.dropna()
    vif = {}
    for i, col in enumerate(df.columns):
        y = df[col].values
        X = df.drop(columns=col).values
        if X.shape[1] == 0:
            vif[col] = 1.0
            continue
        try:
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            vif[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except Exception:
            vif[col] = float("inf")
    return pd.Series(vif)


def correlation_filter(
    df: pd.DataFrame,
    threshold: float = 0.7,
    iv_dict: Dict[str, float] = None,
    return_dropped: bool = False,
) -> List[str] | Tuple[List[str], List[str]]:
    """Among pairs with |corr| > threshold, keep the feature with higher IV.

    Parameters
    ----------
    df : pd.DataFrame
        WOE-transformed DataFrame.
    threshold : float
        Correlation coefficient threshold.
    iv_dict : dict
        Feature name -> IV mapping for tie-breaking.
    """
    corr = df.corr().abs()
    features = list(df.columns)
    dropped = []

    if iv_dict is None:
        iv_dict = {f: 1.0 for f in features}

    # Upper triangle: check each pair
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if f1 in dropped or f2 in dropped:
                continue
            if corr.loc[f1, f2] > threshold:
                iv1 = iv_dict.get(f1, 0)
                iv2 = iv_dict.get(f2, 0)
                # Drop the one with lower IV
                if iv1 >= iv2:
                    dropped.append(f2)
                else:
                    dropped.append(f1)

    keep = [f for f in features if f not in dropped]

    if return_dropped:
        return keep, dropped
    return keep
