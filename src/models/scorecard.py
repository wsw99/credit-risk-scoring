"""Classical credit scorecard: WOE binning → IV/VIF selection → LR → 300-850 scale.

The Scorecard class orchestrates the full pipeline and produces a
bank-grade scorecard table with per-bin point contributions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.binning import WOEBinner
from src.features.selection import correlation_filter, iv_filter, vif_filter
from src.features.woe import WOETransformer


class Scorecard:
    """End-to-end classical credit scorecard.

    Usage:
        sc = Scorecard(base_score=600, base_odds=50, pdo=20)
        sc.fit(train_df, val_df=None)
        scores = sc.predict_score(test_df)
        probs = sc.predict_proba(test_df)
        print(sc.scorecard_table_)
    """

    def __init__(
        self,
        base_score: float = 600,
        base_odds: float = 50,
        pdo: float = 20,
        iv_threshold: float = 0.02,
        vif_threshold: float = 5.0,
        corr_threshold: float = 0.7,
        max_bins: int = 6,
        min_bin_size: float = 0.02,
        lr_C: float = 1.0,
        exclude_features: Optional[List[str]] = None,
    ):
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo
        self.iv_threshold = iv_threshold
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.lr_C = lr_C
        self.exclude_features = exclude_features or []

        # Scorecard scale parameters
        self.factor_ = pdo / np.log(2)
        self.offset_ = base_score - self.factor_ * np.log(base_odds)

        # Internal state
        self.binner_: Optional[WOEBinner] = None
        self.transformer_: Optional[WOETransformer] = None
        self.lr_: Optional[LogisticRegression] = None
        self.final_features_: List[str] = []
        self.iv_: pd.Series = None
        self.scorecard_table_: Optional[pd.DataFrame] = None
        self.coef_: pd.Series = None
        self.intercept_: float = 0.0

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        target: str = "is_bad",
    ) -> "Scorecard":
        """Run the full scorecard pipeline.

        Steps:
        1. Feature list preparation (exclude non-features, grade/sub_grade)
        2. WOE binning on train set
        3. IV filtering
        4. WOE transformation
        5. VIF + correlation filtering
        6. Logistic regression
        7. Score scaling
        """
        del val_df  # reserved for future stepwise / early-stopping

        df = train_df.copy()

        # 1. Prepare feature list
        exclude = {
            target, "issue_year", "issue_d", "loan_status",
            "addr_state", "zip_code", "grade", "sub_grade",
            "emp_title", "title", "desc", "id", "member_id", "url",
        }
        exclude.update(self.exclude_features)
        features = [c for c in df.columns if c not in exclude]

        # Separate categorical features
        cat_features = [c for c in features if df[c].dtype == "object" or df[c].dtype == "category"]

        print(f"Starting scorecard with {len(features)} features "
              f"({len(cat_features)} categorical, {len(features) - len(cat_features)} numeric)")

        # 2. WOE binning
        self.binner_ = WOEBinner(
            max_bins=self.max_bins,
            min_bin_size=self.min_bin_size,
        )
        self.binner_.fit(
            df,
            target=target,
            features=features,
            categorical_features=cat_features,
        )

        n_binned = len(self.binner_.features_)
        print(f"Successfully binned {n_binned}/{len(features)} features")

        # 3. IV filtering
        keep_iv, dropped_iv = iv_filter(self.binner_.iv_, threshold=self.iv_threshold, return_dropped=True)
        print(f"IV filter (threshold={self.iv_threshold}): kept {len(keep_iv)}, dropped {len(dropped_iv)}")

        if dropped_iv:
            for f in dropped_iv[:5]:
                print(f"  Dropped (low IV): {f} (IV={self.binner_.iv_[f]:.4f})")
            if len(dropped_iv) > 5:
                print(f"  ... and {len(dropped_iv) - 5} more")

        # Update binner to keep only IV-passed features
        self.binner_.features_ = [f for f in self.binner_.features_ if f in keep_iv]
        self.binner_.iv_ = self.binner_.iv_[keep_iv]
        self.iv_ = self.binner_.iv_.copy()  # store on self for feature importance

        # 4. WOE transformation
        self.transformer_ = WOETransformer(self.binner_)
        woe_train = self.transformer_.fit_transform(df, target=target)

        # Drop target from woe for VIF check
        woe_features = woe_train.drop(columns=target)

        # 5. VIF + correlation filtering
        keep_vif, dropped_vif, vif_final = vif_filter(
            woe_features, threshold=self.vif_threshold, return_dropped=True
        )
        print(f"VIF filter (threshold={self.vif_threshold}): kept {len(keep_vif)}, dropped {len(dropped_vif)}")

        keep_corr, dropped_corr = correlation_filter(
            woe_features[keep_vif],
            threshold=self.corr_threshold,
            iv_dict=self.binner_.iv_.to_dict(),
            return_dropped=True,
        )
        print(f"Correlation filter (threshold={self.corr_threshold}): "
              f"kept {len(keep_corr)}, dropped {len(dropped_corr)}")

        self.final_features_ = keep_corr
        print(f"Final features for LR: {len(self.final_features_)}")

        # 6. Logistic regression
        X_train = woe_train[self.final_features_].dropna()
        idx = X_train.index
        y_train = woe_train.loc[idx, target]

        self.lr_ = LogisticRegression(
            penalty="l2",
            C=self.lr_C,
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        self.lr_.fit(X_train, y_train)

        self.coef_ = pd.Series(self.lr_.coef_[0], index=self.final_features_)
        self.intercept_ = self.lr_.intercept_[0]

        # 7. Build scorecard table
        self._build_scorecard_table()

        print(f"\nScorecard complete.")
        print(f"  Features: {len(self.final_features_)}")
        print(f"  Train score range: [{self.predict_score(train_df).min():.0f}, "
              f"{self.predict_score(train_df).max():.0f}]")

        return self

    def _build_scorecard_table(self):
        """Build per-bin point contributions table."""
        rows = []
        n = len(self.final_features_)

        for feat in self.final_features_:
            if feat not in self.binner_._binners:
                continue

            optb = self.binner_._binners[feat]
            table = optb.binning_table.build()
            beta = self.coef_[feat]

            for _, row in table.iterrows():
                woe_raw = row.get("WoE", np.nan)
                # Skip bins where WoE is not a valid number (e.g. "Special", "Missing" meta-rows)
                try:
                    woe = float(woe_raw)
                except (ValueError, TypeError):
                    continue

                # points_i = -beta_i * WOE_i * factor  (standard formula)
                points = -woe * beta * self.factor_

                rows.append({
                    "feature": feat,
                    "bin": row["Bin"],
                    "count": row["Count"],
                    "count_pct": row["Count (%)"],
                    "non_event": row["Non-event"],
                    "event": row["Event"],
                    "event_rate": row["Event rate"],
                    "woe": woe,
                    "iv": row.get("IV", 0),
                    "coefficient": beta,
                    "points": np.round(points, 1),
                })

        # Adjust intercept contribution
        base_points = self.offset_ - self.factor_ * self.intercept_
        rows.append({
            "feature": "(Intercept)",
            "bin": "-",
            "count": 0,
            "count_pct": 0,
            "non_event": 0,
            "event": 0,
            "event_rate": 0,
            "woe": 0,
            "iv": 0,
            "coefficient": self.intercept_,
            "points": np.round(base_points, 1),
        })

        self.scorecard_table_ = pd.DataFrame(rows)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of default (is_bad=1)."""
        woe_df = self.transformer_.transform(df)
        # Ensure column order matches training
        X = woe_df[self.final_features_].values
        # Fill NaN with 0 (neutral WOE)
        X = np.nan_to_num(X, nan=0.0)
        return self.lr_.predict_proba(X)[:, 1]

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Convert probability to score (higher = lower risk).

        Standard formula: score = offset + factor * ln(good:bad odds)
        """
        proba = self.predict_proba(df)
        proba = np.clip(proba, 1e-10, 1 - 1e-10)
        odds_good_to_bad = (1 - proba) / proba
        score = self.offset_ + self.factor_ * np.log(odds_good_to_bad)
        return np.clip(score, 300, 850)

    def get_feature_importance(self) -> pd.DataFrame:
        """Feature importance based on absolute coefficient * IV."""
        importance = []
        for feat in self.final_features_:
            beta = self.coef_.get(feat, 0)
            iv = self.iv_.get(feat, 0) if self.iv_ is not None else 0
            importance.append({
                "feature": feat,
                "coefficient": beta,
                "abs_coef": abs(beta),
                "iv": iv,
                "importance": abs(beta) * iv,
            })
        df = pd.DataFrame(importance)
        df = df.sort_values("importance", ascending=False)
        return df

    def save(self, path: str | Path):
        """Save the scorecard to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Scorecard":
        """Load a scorecard from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
