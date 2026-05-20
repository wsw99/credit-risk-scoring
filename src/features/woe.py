"""WOE transformer that converts raw features to Weight-of-Evidence values.

Uses the bins learned by WOEBinner to transform train/val/test consistently.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.features.binning import WOEBinner


class WOETransformer:
    """Transform raw features to WOE-encoded matrix for logistic regression.

    Usage:
        transformer = WOETransformer(binner)
        woe_train = transformer.fit_transform(train_df, target='is_bad')
        woe_val = transformer.transform(val_df)
    """

    def __init__(self, binner: WOEBinner):
        self.binner = binner
        self.woe_features_: List[str] = []
        self.iv_: pd.Series = None

    def fit_transform(self, df: pd.DataFrame, target: str = "is_bad") -> pd.DataFrame:
        """Transform training data using the pre-fitted binner.

        The binner must already be fitted before calling this method.
        Returns a WOE-encoded DataFrame with the target column appended.
        """
        self.woe_features_ = [f + "_woe" for f in self.binner.features_]
        self.iv_ = self.binner.iv_

        woe_df = self.binner.get_woe(df)
        woe_cols = [c for c in woe_df.columns if c.endswith("_woe")]
        result = pd.DataFrame(index=df.index)
        for col in woe_cols:
            base = col.replace("_woe", "")
            if base in self.binner.features_:
                result[base] = woe_df[col]
        result[target] = df[target].values
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using pre-fitted bins."""
        woe_df = self.binner.get_woe(df)
        woe_cols = [c for c in woe_df.columns if c.endswith("_woe")]
        result = pd.DataFrame(index=df.index)
        for col in woe_cols:
            base_name = col.replace("_woe", "")
            if base_name in self.binner.features_:
                result[base_name] = woe_df[col]
        return result
