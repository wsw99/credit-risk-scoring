"""WOE binning engine wrapping optbinning.OptimalBinning.

Provides:
- WOEBinner.fit(df, target, features) — fit optimal bins for each feature
- WOEBinner.transform(df) — replace raw values with bin index
- binning_table_ — per-feature bin boundaries, WOE, IV summary
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from optbinning import OptimalBinning


class WOEBinner:
    """Optimal monotonic WOE binning for scorecard development.

    Usage:
        binner = WOEBinner(max_bins=8, min_bin_size=0.02)
        binner.fit(train_df, target='is_bad', features=feature_list)
        binned_train = binner.transform(train_df)
        binning_table = binner.binning_table_
    """

    def __init__(
        self,
        max_bins: int = 5,
        min_bin_size: float = 0.02,
        monotonic_trend: str = "auto",
        min_iv: float = 0.01,
        max_pvalue: float = 0.05,
    ):
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.min_iv = min_iv
        self.max_pvalue = max_pvalue
        self._binners: dict = {}
        self.binning_table_: Optional[pd.DataFrame] = None
        self.features_: List[str] = []
        self.iv_: pd.Series = None

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> "WOEBinner":
        """Fit optimal binning for each numeric/categorical feature.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.
        target : str
            Binary target column name.
        features : list of str, optional
            Feature columns to bin. If None, uses all numeric columns.
        categorical_features : list of str, optional
            Columns to treat as categorical (dtype object is auto-detected).
        """
        df = df.copy()

        if features is None:
            exclude = {target, "issue_year", "issue_d", "loan_status", "addr_state", "zip_code"}
            features = [c for c in df.columns if c not in exclude]

        if categorical_features is None:
            categorical_features = []

        # Separate numeric vs categorical
        numeric_features = [
            f for f in features
            if f not in categorical_features
            and df[f].dtype in ("int64", "float64", "int32", "float32")
        ]

        candidate_features = numeric_features + categorical_features
        iv_values = {}
        all_tables = []
        self._failed_features = {}

        for feat in candidate_features:
            dtype = "categorical" if feat in categorical_features else "numerical"
            x = df[feat].values
            y = df[target].values

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    optb = OptimalBinning(
                        name=feat,
                        dtype=dtype,
                        max_n_bins=self.max_bins,
                        min_bin_size=self.min_bin_size,
                        monotonic_trend=self.monotonic_trend,
                        min_iv=self.min_iv,
                        max_pvalue=self.max_pvalue,
                    )
                    optb.fit(x, y)
                    self._binners[feat] = optb
                    iv_values[feat] = optb.binning_table.build()["IV"].sum()

                    table = optb.binning_table.build()
                    table["Feature"] = feat
                    all_tables.append(table)
                except Exception as e1:
                    # Fall back to fewer bins, relaxed constraints
                    try:
                        optb = OptimalBinning(
                            name=feat,
                            dtype=dtype,
                            max_n_bins=3,
                            min_bin_size=0.05,
                            monotonic_trend="auto_asc_desc",
                        )
                        optb.fit(x, y)
                        self._binners[feat] = optb
                        iv_values[feat] = optb.binning_table.build()["IV"].sum()
                        table = optb.binning_table.build()
                        table["Feature"] = feat
                        all_tables.append(table)
                    except Exception as e2:
                        self._failed_features[feat] = str(e2)

        self.features_ = list(self._binners.keys())
        self.iv_ = pd.Series(iv_values).sort_values(ascending=False)
        self.binning_table_ = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

        if self._failed_features:
            print(f"WOE binning: {len(self._failed_features)} features failed to bin:")
            for f, err in self._failed_features.items():
                print(f"  {f}: {err}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace raw feature values with bin indices (0, 1, 2...).

        Missing values are mapped to NaN in the output (optbinning bin index = -1 or separate).
        """
        df = df.copy()
        for feat in self.features_:
            if feat not in self._binners or feat not in df.columns:
                continue
            x = df[feat].values
            try:
                df[feat + "_bin"] = self._binners[feat].transform(x, metric="bins")
            except Exception:
                df[feat + "_bin"] = np.nan
        return df

    def get_woe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace raw feature values with WOE values."""
        df = df.copy()
        for feat in self.features_:
            if feat not in self._binners or feat not in df.columns:
                continue
            x = df[feat].values
            try:
                df[feat + "_woe"] = self._binners[feat].transform(x, metric="woe")
            except Exception:
                df[feat + "_woe"] = np.nan
        return df
