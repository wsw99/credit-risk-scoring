"""Sweep max_bins to quantify WOE discretization loss."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.features.binning import WOEBinner
from src.features.woe import WOETransformer
from src.features.selection import iv_filter, vif_filter, correlation_filter
import warnings
warnings.filterwarnings('ignore')

train = pd.read_parquet('data/processed/train.parquet')
val   = pd.read_parquet('data/processed/val.parquet')
test  = pd.read_parquet('data/processed/test.parquet')

target = 'is_bad'
exclude = {target, 'issue_year', 'issue_d', 'loan_status',
           'addr_state', 'zip_code', 'grade', 'sub_grade',
           'emp_title', 'title', 'desc', 'id', 'member_id', 'url'}
exclude = {c for c in exclude if c in train.columns}
features = [c for c in train.columns if c not in exclude]
cat_features = [c for c in features if train[c].dtype == 'object']

print(f'Features: {len(features)} ({len(cat_features)} cat, {len(features) - len(cat_features)} num)')
print()

# Raw LR baseline (no binning, with proper scaling)
from sklearn.preprocessing import StandardScaler

num_features = [c for c in features if c not in cat_features]
X_train_r = train[num_features].copy()
X_test_r = test[num_features].copy()
for col in num_features:
    med = X_train_r[col].median()
    X_train_r[col] = X_train_r[col].fillna(med)
    X_test_r[col] = X_test_r[col].fillna(med)
# Also label-encode cats
for col in cat_features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    mapping = {v: i for i, v in enumerate(train[col].value_counts().index)}
    train[col] = train[col].map(mapping).fillna(-1)
    test[col] = test[col].map(mapping).fillna(-1)
    X_train_r[col] = train[col]
    X_test_r[col] = test[col]

scaler = StandardScaler()
X_train_rs = scaler.fit_transform(X_train_r)
X_test_rs = scaler.transform(X_test_r)

lr_raw = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
lr_raw.fit(X_train_rs, train[target])
raw_auc = roc_auc_score(test[target], lr_raw.predict_proba(X_test_rs)[:, 1])
print(f'Raw LR (no binning, {len(num_features) + len(cat_features)} features, standardized): Test AUC = {raw_auc:.4f}')
print()

# Sweep max_bins
for max_bins in [3, 5, 8, 12, 20]:
    binner = WOEBinner(max_bins=max_bins, min_bin_size=0.02)
    binner.fit(train, target=target, features=features, categorical_features=cat_features)

    n_binned = len(binner.features_)

    # IV filter
    keep_iv, _ = iv_filter(binner.iv_, threshold=0.02, return_dropped=True)
    binner.features_ = [f for f in binner.features_ if f in keep_iv]
    binner.iv_ = binner.iv_[keep_iv]

    # WOE transform
    transformer = WOETransformer(binner)
    woe_train = transformer.fit_transform(train, target=target)
    woe_test = transformer.transform(test)

    woe_feats = woe_train.drop(columns=target)

    # VIF + correlation
    keep_vif, _, _ = vif_filter(woe_feats, threshold=5.0, return_dropped=True)
    keep_corr, _ = correlation_filter(
        woe_feats[keep_vif], threshold=0.7,
        iv_dict=binner.iv_.to_dict(), return_dropped=True
    )

    final_features = keep_corr

    # LR
    X_tr = woe_train[final_features].fillna(0).values
    X_te = woe_test[final_features].fillna(0).values
    y_tr = train[target]
    y_te = test[target]

    lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
    lr.fit(X_tr, y_tr)
    woe_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

    # Count bins per feature
    avg_bins = np.mean([len(binner._binners[f].splits) + 1 for f in final_features if f in binner._binners])

    print(f'max_bins={max_bins:2d}: {n_binned} binned → IV={len(keep_iv)} → final={len(final_features):2d}  '
          f'Test AUC={woe_auc:.4f}  Δ={woe_auc - raw_auc:+.4f}  avg_bins={avg_bins:.1f}')
