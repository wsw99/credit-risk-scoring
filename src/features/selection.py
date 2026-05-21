"""Feature selection & validation for scorecard development.

- iv_filter: keep features with IV above threshold
- vif_filter: remove features with VIF above threshold (iteratively)
- correlation_filter: among highly correlated pairs, keep the one with higher IV
- adjacent_bin_merge_test: chi-square test on neighboring bins (flag over-binning)
- woe_consistency_check: compare WOE trends between train and val
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
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


def adjacent_bin_merge_test(
    binner,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Chi-square test on each pair of adjacent bins per feature.

    A high p-value (> alpha) means the two adjacent bins are not
    significantly different — they should be merged. This prevents
    over-binning and satisfies regulatory expectations.

    Parameters
    ----------
    binner : WOEBinner
        A fitted WOEBinner instance.
    alpha : float
        Significance threshold. Adjacent bins with p > alpha are flagged.

    Returns
    -------
    pd.DataFrame with columns: feature, bin_pair, chi2_stat, p_value, should_merge
    """
    results = []
    for feat in binner.features_:
        if feat not in binner._binners:
            continue

        optb = binner._binners[feat]
        table = optb.binning_table.build()

        # Exclude meta-rows: Special, Missing, Totals
        bins = table[~table["Bin"].isin(["Special", "Missing", "Totals"])]
        if len(bins) < 2:
            continue

        for i in range(len(bins) - 1):
            row_a = bins.iloc[i]
            row_b = bins.iloc[i + 1]

            contingency = [
                [row_a["Non-event"], row_a["Event"]],
                [row_b["Non-event"], row_b["Event"]],
            ]

            try:
                chi2, p_value, _, _ = chi2_contingency(contingency)
            except Exception:
                chi2, p_value = np.nan, np.nan

            results.append({
                "feature": feat,
                "bin_pair": f"{row_a['Bin']}  |  {row_b['Bin']}",
                "count_a": int(row_a["Count"]),
                "count_b": int(row_b["Count"]),
                "event_rate_a": round(float(row_a["Event rate"]), 4),
                "event_rate_b": round(float(row_b["Event rate"]), 4),
                "chi2_stat": round(float(chi2), 2) if not np.isnan(chi2) else np.nan,
                "p_value": round(float(p_value), 4) if not np.isnan(p_value) else np.nan,
                "should_merge": p_value > alpha if not np.isnan(p_value) else False,
            })

    return pd.DataFrame(results)


def woe_consistency_check(
    binner,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str = "is_bad",
    corr_threshold: float = 0.9,
) -> pd.DataFrame:
    """Compare WOE trend direction between train and validation sets.

    For each feature, compute WOE values on train and val using trainʼs
    bin boundaries. Features with low WOE correlation (< corr_threshold)
    or sign reversal across train/val are flagged as unstable.

    Parameters
    ----------
    binner : WOEBinner
        Fitted binner (on train data).
    train_df : pd.DataFrame
        Training data.
    val_df : pd.DataFrame
        Validation data.
    target : str
        Binary target column name.
    corr_threshold : float
        WOE correlation below which a feature is flagged.

    Returns
    -------
    pd.DataFrame with columns: feature, train_woe, val_woe, correlation, stable
    """
    results = []
    for feat in binner.features_:
        if feat not in binner._binners:
            continue

        optb = binner._binners[feat]
        table = optb.binning_table.build()
        bins = table[~table["Bin"].isin(["Special", "Missing", "Totals"])]

        train_woe_raw = bins["WoE"].values
        # Ensure all WOE values are float (optbinning may return strings for edge cases)
        try:
            train_woe = np.array([float(w) if not isinstance(w, (int, float)) else float(w) for w in train_woe_raw])
        except (ValueError, TypeError):
            continue
        if len(train_woe) < 2:
            continue

        # Compute val WOE using train bin boundaries
        val_woe = _compute_woe_on_dataset(optb, val_df[feat].values, val_df[target].values)

        # Align lengths (val may have fewer bins due to empty bins)
        min_len = min(len(train_woe), len(val_woe))
        if min_len < 2:
            continue

        train_woe = train_woe[:min_len]
        val_woe = val_woe[:min_len]

        try:
            corr = np.corrcoef(train_woe, val_woe)[0, 1]
        except Exception:
            corr = np.nan

        # Check sign consistency: all signs should match
        sign_match = all(
            (tw > 0) == (vw > 0)
            for tw, vw in zip(train_woe, val_woe)
            if abs(tw) > 0.01 and abs(vw) > 0.01
        )

        stable = (not np.isnan(corr) and corr >= corr_threshold) and sign_match

        results.append({
            "feature": feat,
            "train_woe": [round(float(x), 4) for x in train_woe],
            "val_woe": [round(float(x), 4) for x in val_woe.astype(float)],
            "woe_correlation": round(float(corr), 4) if not np.isnan(corr) else np.nan,
            "sign_consistent": sign_match,
            "stable": stable,
        })

    return pd.DataFrame(results)


def _compute_woe_on_dataset(optb, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Recompute WOE for each bin on a new dataset using pre-fitted bin boundaries.

    Uses bin labels (strings) returned by optb.transform(metric='bins') to
    group samples and recompute WOE per bin in the same order as train.
    """
    # Drop NaN target rows
    valid = ~np.isnan(target.astype(float))
    values = values[valid]
    target = target[valid].astype(int)

    if len(values) == 0:
        return np.array([])

    # Get bin label for each sample
    bin_labels = optb.transform(values, metric="bins")

    # Get the train bin table to determine bin order
    train_table = optb.binning_table.build()
    train_bins = train_table[~train_table["Bin"].isin(["Special", "Missing", "Totals"])]
    bin_order = train_bins["Bin"].tolist()

    total_good = (target == 0).sum()
    total_bad = (target == 1).sum()

    woe_per_bin = []
    for bin_label in bin_order:
        mask = bin_labels == bin_label
        n_good = (target[mask] == 0).sum()
        n_bad = (target[mask] == 1).sum()

        dist_good = n_good / total_good if total_good > 0 else 0.0
        dist_bad = n_bad / total_bad if total_bad > 0 else 0.0

        if dist_bad == 0.0 and dist_good == 0.0:
            woe = 0.0
        elif dist_bad == 0.0:
            woe = np.log(max(dist_good, 1e-10) / 1e-10)
        elif dist_good == 0.0:
            woe = np.log(1e-10 / max(dist_bad, 1e-10))
        else:
            woe = np.log(dist_good / dist_bad)

        woe_per_bin.append(woe)

    return np.array(woe_per_bin)
