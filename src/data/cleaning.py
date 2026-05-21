"""Data cleaning pipeline for LendingClub accepted loans.

Key steps:
  1. define_labels — keep only definite-outcome loans, create binary target
  2. drop_high_missingness — remove columns with >threshold missing
  3. drop_leaky_features — remove post-issuance observable features
  4. drop_id_artifact_columns — remove IDs, URLs, free-text with high missing
  5. apply_temporal_split — split by issue date (train 2014-2016, val 2017, test 2018)
"""

import pandas as pd
import numpy as np

CLEANING_STATS: dict = {}


def define_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with definite outcomes; create binary target `is_bad`.

    - Good (0): Fully Paid
    - Bad (1): Charged Off, Default
    - Excluded: Current, Late (all stages), In Grace Period, policy-violation rows
    """
    good_labels = {"Fully Paid"}
    bad_labels = {"Charged Off", "Default"}
    exclude_labels = {
        "Current",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "In Grace Period",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    }

    n_before = len(df)

    mask_outcome = df["loan_status"].isin(good_labels | bad_labels)
    df = df[mask_outcome].copy()

    df["is_bad"] = df["loan_status"].isin(bad_labels).astype(int)

    n_after = len(df)
    CLEANING_STATS["rows_before_label"] = n_before
    CLEANING_STATS["rows_after_label"] = n_after
    CLEANING_STATS["rows_removed_label"] = n_before - n_after
    CLEANING_STATS["bad_rate"] = df["is_bad"].mean()

    return df


def drop_id_artifact_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not features: IDs, URLs, free-text with near-100% missing."""
    artifact_cols = [
        "id",
        "member_id",
        "url",
        "desc",            # free-text loan description, 94% missing
        "emp_title",       # free-text job title, high cardinality — drop for now
        "title",           # free-text loan title
        "zip_code",        # redundant with addr_state, first-3-digits only
    ]
    drop = [c for c in artifact_cols if c in df.columns]
    CLEANING_STATS["artifact_cols_dropped"] = len(drop)
    return df.drop(columns=drop, errors="ignore")


def drop_high_missingness(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Drop columns with missing rate above threshold."""
    missing_rate = df.isnull().mean()
    drop = missing_rate[missing_rate > threshold].index.tolist()
    CLEANING_STATS["high_missing_cols_dropped"] = len(drop)
    CLEANING_STATS["high_missing_cols"] = drop
    return df.drop(columns=drop, errors="ignore")


def drop_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop features that are only observable after loan issuance."""
    leaky = [
        # Post-issuance payment / recovery
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        # Last / next payment dates and amounts
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        # Last credit pull (may be post-issuance or contemporaneous)
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        # Debt settlement / hardship — post-issuance events
        "debt_settlement_flag",
        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
        "hardship_flag",
        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "hardship_start_date",
        "hardship_end_date",
        "hardship_amount",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
        "deferral_term",
        "payment_plan_start_date",
        "orig_projected_additional_accrued_interest",
        # Policy code (metadata)
        "policy_code",
    ]
    drop = [c for c in leaky if c in df.columns]
    CLEANING_STATS["leaky_cols_dropped"] = len(drop)
    CLEANING_STATS["leaky_cols"] = drop
    return df.drop(columns=drop, errors="ignore")


def apply_temporal_split(
    df: pd.DataFrame,
    train_years: tuple = (2007, 2014),
    val_year: int = 2015,
    test_year: int = 2016,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by loan issue year (strict out-of-time).

    This mirrors real deployment: train on past, validate on recent, test on future.
    """
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_dt"].dt.year

    train = df[df["issue_year"].between(*train_years)].copy()
    val = df[df["issue_year"] == val_year].copy()
    test = df[df["issue_year"] == test_year].copy()

    # Drop temporary date columns
    for part in (train, val, test):
        part.drop(columns=["issue_dt"], inplace=True, errors="ignore")

    CLEANING_STATS["train_rows"] = len(train)
    CLEANING_STATS["val_rows"] = len(val)
    CLEANING_STATS["test_rows"] = len(test)

    total_removed = (
        CLEANING_STATS.get("artifact_cols_dropped", 0)
        + CLEANING_STATS.get("high_missing_cols_dropped", 0)
        + CLEANING_STATS.get("leaky_cols_dropped", 0)
    )
    CLEANING_STATS["total_removed"] = total_removed

    return train, val, test


def drop_split_useless_columns(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    nan_threshold: float = 0.8,
    constant_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop columns that are near-useless in *any* of the three splits.

    Phase 1 (global) drop_high_missingness catches columns with >80% NaN on the
    full dataset.  But a column could have e.g. 73% NaN globally yet be 100% NaN
    in train (field added after 2014).  This pass checks each split individually
    and removes any column that exceeds the threshold in any split.

    Parameters
    ----------
    nan_threshold : float
        Drop if NaN rate >= this in *any* split (default 0.8, matching global).
    constant_threshold : float
        Drop if the most frequent single value accounts for >= this fraction.
    """
    splits = {"train": train, "val": val, "test": test}

    # --- NaN check per split ---
    nan_drop: set = set()
    for name, part in splits.items():
        rate = part.isnull().mean()
        bad = rate[rate >= nan_threshold].index.tolist()
        if bad:
            CLEANING_STATS[f"split_nan_{name}"] = bad
        nan_drop.update(bad)

    # --- Near-constant check per split ---
    constant_drop: set = set()
    for name, part in splits.items():
        for col in part.columns:
            if col in nan_drop or col in constant_drop:
                continue
            try:
                dominant = part[col].value_counts(normalize=True, dropna=False).iloc[0]
                if dominant >= constant_threshold:
                    constant_drop.add(col)
                    CLEANING_STATS.setdefault(f"split_constant_{name}", []).append(col)
            except Exception:
                pass

    drop = sorted(nan_drop | constant_drop)

    CLEANING_STATS["split_useless_cols_dropped"] = len(drop)
    CLEANING_STATS["split_useless_cols"] = drop
    CLEANING_STATS["split_useless_nan"] = sorted(nan_drop)
    CLEANING_STATS["split_useless_constant"] = sorted(constant_drop)

    return (
        train.drop(columns=drop, errors="ignore"),
        val.drop(columns=drop, errors="ignore"),
        test.drop(columns=drop, errors="ignore"),
    )
