"""Comprehensive Stage 1 vs Stage 2 comparison on identical test set.

Answers: Why does Stage 2 AUC (0.704) < Stage 1 AUC (0.714)?
"""
import sys
sys.path.insert(0, '.')
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Load data ──────────────────────────────────────────────────────
train = pd.read_parquet('data/processed/train.parquet')
val   = pd.read_parquet('data/processed/val.parquet')
test  = pd.read_parquet('data/processed/test.parquet')

print('=' * 80)
print('DATA CONSISTENCY CHECK')
print('=' * 80)
print(f'Train: {train.shape[0]:,} rows x {train.shape[1]} cols  bad_rate={train["is_bad"].mean():.4f}')
print(f'Val:   {val.shape[0]:,} rows x {val.shape[1]} cols  bad_rate={val["is_bad"].mean():.4f}')
print(f'Test:  {test.shape[0]:,} rows x {test.shape[1]} cols  bad_rate={test["is_bad"].mean():.4f}')
print(f'Test row hash: {hash(tuple(test.index))}')  # proves test is identical

# ── Prepare features ───────────────────────────────────────────────
exclude_cols = {'is_bad', 'issue_year', 'issue_d', 'loan_status',
                'addr_state', 'zip_code', 'grade', 'sub_grade',
                'emp_title', 'title', 'desc', 'id', 'member_id', 'url'}
# Only keep features present in data
exclude_cols = {c for c in exclude_cols if c in train.columns}

# Numeric features (no one-hot encoding, for fair comparison with WOE)
numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols]
cat_cols = [c for c in train.columns
            if c not in exclude_cols and c not in numeric_cols]

print(f'\nNumeric features: {len(numeric_cols)}')
print(f'Categorical features: {len(cat_cols)} (will be label-encoded)')

# ── Handle categoricals (label encoding, same for both stages) ─────
def label_encode_fit(train, val, test, cat_cols):
    """Fit on train, transform all."""
    mappings = {}
    for col in cat_cols:
        train[col] = train[col].astype(str)
        val[col] = val[col].astype(str)
        test[col] = test[col].astype(str)
        uniq = train[col].value_counts().index
        mapping = {v: i for i, v in enumerate(uniq)}
        mappings[col] = mapping
        train[col] = train[col].map(mapping).fillna(-1)
        val[col] = val[col].map(mapping).fillna(-1)
        test[col] = test[col].map(mapping).fillna(-1)
    return mappings

train_le = train.copy()
val_le = val.copy()
test_le = test.copy()
mappings = label_encode_fit(train_le, val_le, test_le, [c for c in cat_cols if c in train_le.columns])

all_features_s1 = numeric_cols + [c for c in cat_cols if c in train_le.columns]
print(f'Stage 1 total features: {len(all_features_s1)}')

# ── Stage 1: Raw LR ────────────────────────────────────────────────
print('\n' + '=' * 80)
print('STAGE 1: RAW LOGISTIC REGRESSION')
print('=' * 80)

# Impute missing with median
X_train_s1 = train_le[all_features_s1].copy()
X_val_s1 = val_le[all_features_s1].copy()
X_test_s1 = test_le[all_features_s1].copy()

for col in all_features_s1:
    med = X_train_s1[col].median()
    X_train_s1[col] = X_train_s1[col].fillna(med)
    X_val_s1[col] = X_val_s1[col].fillna(med)
    X_test_s1[col] = X_test_s1[col].fillna(med)

y_train = train_le['is_bad']
y_val = val_le['is_bad']
y_test = test_le['is_bad']

# Scale
scaler = StandardScaler()
X_train_s1_s = scaler.fit_transform(X_train_s1)
X_val_s1_s = scaler.transform(X_val_s1)
X_test_s1_s = scaler.transform(X_test_s1)

lr1 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
lr1.fit(X_train_s1_s, y_train)

s1_train_auc = roc_auc_score(y_train, lr1.predict_proba(X_train_s1_s)[:, 1])
s1_val_auc = roc_auc_score(y_val, lr1.predict_proba(X_val_s1_s)[:, 1])
s1_test_auc = roc_auc_score(y_test, lr1.predict_proba(X_test_s1_s)[:, 1])

print(f'Stage 1 LR — Train AUC: {s1_train_auc:.4f}  Val AUC: {s1_val_auc:.4f}  Test AUC: {s1_test_auc:.4f}')

# ── Stage 2: Load scorecard ────────────────────────────────────────
print('\n' + '=' * 80)
print('STAGE 2: WOE SCORECARD')
print('=' * 80)

sc = pickle.load(open('results/models/scorecard_stage2.pkl', 'rb'))

s2_train_auc = roc_auc_score(y_train, sc.predict_proba(train))
s2_val_auc = roc_auc_score(y_val, sc.predict_proba(val))
s2_test_auc = roc_auc_score(y_test, sc.predict_proba(test))

print(f'Stage 2 WOE — Train AUC: {s2_train_auc:.4f}  Val AUC: {s2_val_auc:.4f}  Test AUC: {s2_test_auc:.4f}')

# ── Diagnostic 1: Stage 1 LR with ONLY Stage 2's 26 features ──────
print('\n' + '=' * 80)
print('DIAGNOSTIC 1: Stage 1 LR with only Stage 2\'s 26 features')
print('=' * 80)

s2_features = sc.final_features_
# Map WOE feature names back to raw feature names
s2_raw_features = [f.replace('_woe', '') for f in s2_features]
# Keep only features present in data
s2_raw_features_avail = [f for f in s2_raw_features if f in all_features_s1]
print(f'Stage 2 uses {len(s2_features)} WOE features')
print(f'Mapped to {len(s2_raw_features_avail)} raw features (present in data)')

X_train_s2f = X_train_s1[s2_raw_features_avail].copy()
X_val_s2f = X_val_s1[s2_raw_features_avail].copy()
X_test_s2f = X_test_s1[s2_raw_features_avail].copy()

# Use same imputation
for col in s2_raw_features_avail:
    med = X_train_s1[col].median()
    X_train_s2f[col] = X_train_s2f[col].fillna(med)
    X_val_s2f[col] = X_val_s2f[col].fillna(med)
    X_test_s2f[col] = X_test_s2f[col].fillna(med)

scaler2 = StandardScaler()
X_train_s2f_s = scaler2.fit_transform(X_train_s2f)
X_val_s2f_s = scaler2.transform(X_val_s2f)
X_test_s2f_s = scaler2.transform(X_test_s2f)

lr2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
lr2.fit(X_train_s2f_s, y_train)

s1_subset_train = roc_auc_score(y_train, lr2.predict_proba(X_train_s2f_s)[:, 1])
s1_subset_val = roc_auc_score(y_val, lr2.predict_proba(X_val_s2f_s)[:, 1])
s1_subset_test = roc_auc_score(y_test, lr2.predict_proba(X_test_s2f_s)[:, 1])

print(f'Raw LR with 26 features — Train: {s1_subset_train:.4f}  Val: {s1_subset_val:.4f}  Test: {s1_subset_test:.4f}')
print(f'Gap from Stage 1 (full {len(all_features_s1)} features): {s1_subset_test - s1_test_auc:+.4f}')

# ── Diagnostic 2: WOE encoding vs raw values (LR on WOE vs LR on raw) ──
print('\n' + '=' * 80)
print('DIAGNOSTIC 2: WOE encoding effect (same 26 features, LR on WOE vs raw)')
print('=' * 80)

# Get WOE-transformed data
woe_train = sc.transformer_.transform(train)
woe_test = sc.transformer_.transform(test)

# Drop NaN rows
woe_train_clean = woe_train[s2_features].dropna()
idx_train = woe_train_clean.index
y_train_clean = train.loc[idx_train, 'is_bad']

woe_test_clean = woe_test[s2_features].dropna()
idx_test = woe_test_clean.index
y_test_clean = test.loc[idx_test, 'is_bad']

# Fill NaN with 0 (neutral WOE) for both
X_woe_train = woe_train[s2_features].fillna(0).values
X_woe_test = woe_test[s2_features].fillna(0).values

lr_woe = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
lr_woe.fit(X_woe_train, y_train)
woe_lr_test = roc_auc_score(y_test, lr_woe.predict_proba(X_woe_test)[:, 1])
print(f'LR on WOE values (26 features) — Test AUC: {woe_lr_test:.4f}')
print(f'Scorecard LR on WOE     — Test AUC: {s2_test_auc:.4f}')

# ── Diagnostic 3: Feature reduction impact ─────────────────────────
print('\n' + '=' * 80)
print('DIAGNOSTIC 3: Impact breakdown')
print('=' * 80)

print(f'{"Component":<45} {"Test AUC":>10} {"Δ from baseline":>15}')
print('-' * 72)
print(f'{"A. Stage 1 Raw LR (all features)":<45} {s1_test_auc:>10.4f} {"(baseline)":>15}')
print(f'{"B. Raw LR (only 26 WOE-selected features)":<45} {s1_subset_test:>10.4f} {s1_subset_test - s1_test_auc:>+15.4f}')
print(f'{"C. Stage 2 WOE Scorecard (WOE + LR)":<45} {s2_test_auc:>10.4f} {s2_test_auc - s1_test_auc:>+15.4f}')

# Decompose the gap
feature_selection_loss = s1_subset_test - s1_test_auc  # B - A
woe_discretization_loss = s2_test_auc - s1_subset_test  # C - B
total_loss = s2_test_auc - s1_test_auc

print(f'\n--- AUC Decomposition ---')
print(f'Feature selection (89 → 26): {feature_selection_loss:+.4f}')
print(f'WOE discretization (raw → bins): {woe_discretization_loss:+.4f}')
print(f'Total Stage 2 vs Stage 1 gap:     {total_loss:+.4f}')

# ── Diagnostic 4: Feature overlap analysis ─────────────────────────
print('\n' + '=' * 80)
print('DIAGNOSTIC 4: Top features in Stage 1 vs Stage 2')
print('=' * 80)

# Stage 1 feature importance (abs coefficient)
s1_coefs = pd.Series(lr1.coef_[0], index=all_features_s1)
s1_top = s1_coefs.abs().sort_values(ascending=False).head(20)
print('\nStage 1 top 20 features (by |coef|):')
for i, (feat, coef) in enumerate(s1_top.items()):
    in_s2 = '✓' if feat in s2_raw_features_avail else '✗'
    print(f'  {i+1:2d}. {feat:<35s} |coef|={coef:.4f}  in Stage2: {in_s2}')

# Stage 2 features NOT in Stage 1 top 20
s2_feat_set = set(s2_raw_features_avail)
s1_top20_set = set(s1_top.index)
missing_from_s1_top = s2_feat_set - s1_top20_set
print(f'\nStage 2 features NOT in Stage 1 top 20 ({len(missing_from_s1_top)}):')
for f in sorted(missing_from_s1_top):
    rank = s1_coefs.abs().sort_values(ascending=False).tolist().index(abs(s1_coefs[f])) + 1
    print(f'  {f:<35s} Stage1 rank={rank}, |coef|={abs(s1_coefs[f]):.4f}')

# Features dropped by IV/VIF that had high Stage 1 importance
s1_strong = set(s1_top.head(20).index)
dropped_strong = s1_strong - s2_feat_set
print(f'\nTop-20 Stage 1 features DROPPED by Stage 2 ({len(dropped_strong)}):')
for f in sorted(dropped_strong, key=lambda x: abs(s1_coefs[x]), reverse=True):
    print(f'  {f:<35s} Stage1 |coef|={abs(s1_coefs[f]):.4f}, rank={s1_top.index.tolist().index(f)+1}')

print('\n' + '=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'Stage 1 Test AUC: {s1_test_auc:.4f} ({len(all_features_s1)} features, raw values)')
print(f'Stage 2 Test AUC: {s2_test_auc:.4f} ({len(s2_features)} features, WOE-binned)')
print(f'Gap: {total_loss:+.4f}')
print(f'\nPrimary cause: {"feature reduction" if abs(feature_selection_loss) > abs(woe_discretization_loss) else "WOE discretization"}')
