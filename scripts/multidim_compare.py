"""Multi-dimensional Stage 1 vs Stage 2 comparison."""
import sys
sys.path.insert(0, '.')
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss
)
from scipy.stats import ks_2samp

train = pd.read_parquet('data/processed/train.parquet')
val   = pd.read_parquet('data/processed/val.parquet')
test  = pd.read_parquet('data/processed/test.parquet')
target = 'is_bad'

# ── Prepare data for Stage 1 ───────────────────────────────────────
exclude = {target, 'issue_year', 'issue_d', 'loan_status',
           'addr_state', 'zip_code', 'grade', 'sub_grade'}
exclude = {c for c in exclude if c in train.columns}
num_features = [c for c in train.select_dtypes(include=[np.number]).columns if c not in exclude]
cat_features = [c for c in train.columns if c not in exclude and c not in num_features]

def prep_data(train, val, test, cat_features, num_features):
    t, v, te = train.copy(), val.copy(), test.copy()
    for col in num_features:
        med = t[col].median()
        for d in [t, v, te]:
            d[col] = d[col].fillna(med)
    for col in cat_features:
        for d in [t, v, te]:
            d[col] = d[col].astype(str)
        mapping = {val: i for i, val in enumerate(t[col].value_counts().index)}
        for d in [t, v, te]:
            d[col] = d[col].map(mapping).fillna(-1)
    all_cols = num_features + [c for c in cat_features if c in t.columns]
    return t, v, te, all_cols

t1, v1, te1, all_cols = prep_data(train, val, test, cat_features, num_features)

X_tr = t1[all_cols].values
X_va = v1[all_cols].values
X_te = te1[all_cols].values
y_tr = t1[target].values
y_va = v1[target].values
y_te = te1[target].values

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

lr1 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000, random_state=42)
lr1.fit(X_tr_s, y_tr)

p1_tr = lr1.predict_proba(X_tr_s)[:,1]
p1_va = lr1.predict_proba(X_va_s)[:,1]
p1_te = lr1.predict_proba(X_te_s)[:,1]

# ── Stage 2 ────────────────────────────────────────────────────────
sc = pickle.load(open('results/models/scorecard_stage2.pkl', 'rb'))
p2_tr = sc.predict_proba(train)
p2_va = sc.predict_proba(val)
p2_te = sc.predict_proba(test)

# Score distributions
s2_tr = sc.predict_score(train)
s2_va = sc.predict_score(val)
s2_te = sc.predict_score(test)

# Convert S1 probabilities to pseudo-scores (same scale, for comparability)
factor = sc.factor_
offset = sc.offset_
s1_te = offset + factor * np.log(np.clip((1 - p1_te) / np.clip(p1_te, 1e-10, 1), 1e-10, 1e10))
s1_tr = offset + factor * np.log(np.clip((1 - p1_tr) / np.clip(p1_tr, 1e-10, 1), 1e-10, 1e10))
s1_va = offset + factor * np.log(np.clip((1 - p1_va) / np.clip(p1_va, 1e-10, 1), 1e-10, 1e10))

# ── Compute all metrics ────────────────────────────────────────────
def ks(y_true, y_pred):
    return ks_2samp(y_pred[y_true == 0], y_pred[y_true == 1]).statistic

def psi(expected_scores, actual_scores, bins=10):
    """Population Stability Index."""
    breaks = np.percentile(expected_scores, np.linspace(0, 100, bins + 1))
    breaks[0], breaks[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected_scores, bins=breaks)[0] / len(expected_scores)
    actual_pct = np.histogram(actual_scores, bins=breaks)[0] / len(actual_scores)
    expected_pct = np.clip(expected_pct, 1e-6, 1)
    actual_pct = np.clip(actual_pct, 1e-6, 1)
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

def gini(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1

def score_band_monotonicity(scores, y_true, n_bins=10):
    """Check default rate monotonicity across score bands. Returns % of bands that are monotonic."""
    breaks = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
    drates = []
    for i in range(n_bins):
        mask = (scores >= breaks[i]) & (scores < breaks[i+1]) if i < n_bins - 1 else (scores >= breaks[i])
        if mask.sum() > 0:
            drates.append(y_true[mask].mean())
    # Check monotonic decrease (higher score = lower default rate)
    diffs = np.diff(drates)
    return (diffs <= 0).mean()

def train_test_gap(train_metric, test_metric):
    return train_metric - test_metric

results = {}
for name, (y_tr, p_tr, y_va, p_va, y_te, p_te, s_tr, s_va, s_te) in [
    ('Stage 1 (Raw LR)', (y_tr, p1_tr, y_va, p1_va, y_te, p1_te, s1_tr, s1_va, s1_te)),
    ('Stage 2 (WOE Scorecard)', (y_tr, p2_tr, y_va, p2_va, y_te, p2_te, s2_tr, s2_va, s2_te)),
]:
    m = {}
    m['AUC']         = roc_auc_score(y_te, p_te)
    m['KS']          = ks(y_te, p_te)
    m['Gini']        = gini(y_te, p_te)
    m['PR-AUC']      = average_precision_score(y_te, p_te)
    m['Brier Score'] = brier_score_loss(y_te, p_te)
    m['Log Loss']    = log_loss(y_te, p_te)
    m['PSI (Train→Test)'] = psi(s_tr, s_te)
    m['PSI (Train→Val)']  = psi(s_tr, s_va)
    m['Train AUC']   = roc_auc_score(y_tr, p_tr)
    m['Val AUC']     = roc_auc_score(y_va, p_va)
    m['Train-Test AUC Gap'] = roc_auc_score(y_tr, p_tr) - roc_auc_score(y_te, p_te)
    m['Score Band Monotonicity'] = score_band_monotonicity(s_te, y_te)
    m['Score Mean (Test)'] = s_te.mean()
    m['Score Std (Test)']  = s_te.std()
    results[name] = m

# ── Print comparison table ─────────────────────────────────────────
print('=' * 90)
print('MULTI-DIMENSIONAL COMPARISON: Stage 1 vs Stage 2')
print('=' * 90)

# Group metrics by category
categories = {
    'Discrimination (higher = better)': ['AUC', 'KS', 'Gini', 'PR-AUC'],
    'Calibration (lower = better)': ['Brier Score', 'Log Loss'],
    'Stability (lower = better)': ['PSI (Train→Test)', 'PSI (Train→Val)', 'Train-Test AUC Gap'],
    'Robustness': ['Score Band Monotonicity'],
    'Complexity': [],
}

for cat, metrics in categories.items():
    print(f'\n--- {cat} ---')
    header = f'{"Metric":<30} {"Stage 1":>12} {"Stage 2":>12} {"Δ (S2-S1)":>14} {"Winner":>10}'
    print(header)
    print('-' * len(header))
    for metric in metrics:
        v1 = results['Stage 1 (Raw LR)'][metric]
        v2 = results['Stage 2 (WOE Scorecard)'][metric]
        delta = v2 - v1
        # Determine winner
        if 'lower = better' in cat:
            winner = 'Stage 1' if delta > 0 else 'Stage 2' if delta < 0 else 'Tie'
        else:
            winner = 'Stage 2' if delta > 0 else 'Stage 1' if delta < 0 else 'Tie'
        print(f'{metric:<30} {v1:>12.4f} {v2:>12.4f} {delta:>+14.4f} {winner:>10}')

# Complexity row
print(f'\n{"Feature count":<30} {len(all_cols):>12d} {len(sc.final_features_):>12d} {"—":>14} {"Stage 2":>10}')

# Per-year stability
print(f'\n--- Per-Year AUC Stability ---')
print(f'{"Year":<10} {"S1 AUC":>10} {"S2 AUC":>10} {"Δ":>10}')
years_data = [
    ('Train', y_tr, p1_tr, p2_tr),
    ('Val', y_va, p1_va, p2_va),
    ('Test', y_te, p1_te, p2_te),
]
for yr, y, prob1, prob2 in years_data:
    a1 = roc_auc_score(y, prob1)
    a2 = roc_auc_score(y, prob2)
    print(f'{yr:<10} {a1:>10.4f} {a2:>10.4f} {a2-a1:>+10.4f}')

# ── Overall summary ─────────────────────────────────────────────────
print(f'\n{"=" * 90}')
print('SUMMARY: Stage 2 vs Stage 1 tradeoffs')
print(f'{"=" * 90}')
wins_s1 = 0
wins_s2 = 0
for cat, metrics in categories.items():
    for metric in metrics:
        v1 = results['Stage 1 (Raw LR)'][metric]
        v2 = results['Stage 2 (WOE Scorecard)'][metric]
        if 'lower = better' in cat:
            if v2 < v1: wins_s2 += 1
            elif v1 < v2: wins_s1 += 1
        else:
            if v2 > v1: wins_s2 += 1
            elif v1 > v2: wins_s1 += 1
print(f'Metrics where Stage 1 wins: {wins_s1}')
print(f'Metrics where Stage 2 wins: {wins_s2}')
print()
print('Stage 2 advantage: interpretability (per-bin scorecard), regulatory compliance,')
print('  PSI monitoring, cleaner feature set with no data leakage')
print('Stage 1 advantage: raw continuous values preserve fine-grained signal')
print('  at the cost of a black-box model that regulators would reject')
