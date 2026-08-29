"""
S6E3 EXP4 - OptimalBinning WoE Feature Experiment
===================================================
Source: Kaggle Notebook by alpayabbaszade (AUC 0.9136 standalone)
Approach:
  - OptimalBinning 1D: Fit on original IBM dataset, transform to monotonic WoE
  - OptimalBinning 2D: Joint binning of feature pairs, captures interactions
  - Test if these WoE features add signal on top of V4 baseline + EXP3 winners

Methodology: 4-stage evaluation
  Stage 1: Batch screening (2-fold fast)
  Stage 2: Individual ablation (top batches)
  Stage 3: Greedy forward selection
  Stage 4: 5-fold confirmation
"""
# !pip install optbinning -q

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# OptimalBinning
from optbinning import OptimalBinning, OptimalBinning2D

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
ORIG_PATH  = "/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = 'Churn'
SEED = 42

LGB_PARAMS = {
    'n_estimators': 3000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'verbose': -1,
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + V4 BASELINE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("S6E3 EXP4 - OptimalBinning WoE Feature Experiment")
print("=" * 80)
t0_all = time.time()

print("\n[LOAD]")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)

train[TARGET] = train[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
orig[TARGET]  = orig[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
if 'customerID' in orig.columns:
    orig.drop(columns=['customerID'], inplace=True)

print(f"Train: {train.shape} | Orig: {orig.shape}")

CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
ALL_FEATURES = CATS + NUMS  # 19 features total

# ═══════════════════════════════════════════════════════════════════════════════
# V4 BASELINE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[PREP] Building V4 baseline features...")

v4_features = list(CATS) + list(NUMS)

# 1. Frequency Encoding
for col in NUMS:
    freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
    for df in [train, test, orig]:
        df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
    v4_features.append(f'FREQ_{col}')

# 2. Arithmetic Interactions
for df in [train, test, orig]:
    df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
    df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
    df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
v4_features += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

# 3. Service Counts
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for df in [train, test, orig]:
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
    df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
v4_features += ['service_count', 'has_internet', 'has_phone']

# 4. ORIG_proba mapping
for col in CATS + NUMS:
    tmp = orig.groupby(col)[TARGET].mean()
    _name = f"ORIG_proba_{col}"
    for df in [train, test]:
        df[_name] = df[col].map(tmp).fillna(0.5).astype('float32')
    v4_features.append(_name)

# Label encode categoricals for LightGBM (simpler than V4's full pipeline for EXP)
le_dict = {}
for col in CATS:
    le = LabelEncoder()
    le.fit(pd.concat([train[col].astype(str), test[col].astype(str), orig[col].astype(str)]))
    for df in [train, test, orig]:
        df[col] = le.transform(df[col].astype(str))
    le_dict[col] = le

# 5. EXP3 v3 winners (9 distribution features)
print("[PREP] Adding EXP3 v3 distribution features...")

# Helper: percentile rank against a reference distribution
def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return np.searchsorted(ref_sorted, values) / len(ref_sorted)

# Helper: z-score against a reference distribution
def zscore_against(values, reference):
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values))
    return (values - mu) / sigma

# Churner / Non-churner distributions from original
orig_churner_tc = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
orig_tc = orig['TotalCharges'].values

exp3_features = []

for df in [train, test]:
    tc = df['TotalCharges'].values

    # 1. pctrank_nonchurner_TotalCharges
    df['pctrank_nonchurner_TotalCharges'] = pctrank_against(tc, orig_nonchurner_tc).astype('float32')
    # 2. zscore_churn_gap_TotalCharges
    df['zscore_churn_gap_TotalCharges'] = (np.abs(zscore_against(tc, orig_churner_tc)) - np.abs(zscore_against(tc, orig_nonchurner_tc))).astype('float32')
    # 3. pctrank_churn_gap_TotalCharges
    df['pctrank_churn_gap_TotalCharges'] = (pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)).astype('float32')
    # 4. resid_mean_InternetService_MonthlyCharges
    is_map = orig.groupby('InternetService')['MonthlyCharges'].mean()
    df['resid_mean_InternetService_MonthlyCharges'] = (df['MonthlyCharges'] - df['InternetService'].map(is_map).fillna(0)).astype('float32')
    # 5. cond_pctrank_InternetService_TotalCharges
    vals = np.zeros(len(df), dtype='float32')
    for cat_val in orig['InternetService'].unique():
        mask_df = (df['InternetService'] == le_dict['InternetService'].transform([str(cat_val)])[0]) if isinstance(cat_val, str) else (df['InternetService'] == cat_val)
        ref = orig.loc[orig['InternetService'] == cat_val, 'TotalCharges'].values
        if len(ref) > 0 and mask_df.sum() > 0:
            vals[mask_df] = pctrank_against(df.loc[mask_df, 'TotalCharges'].values, ref)
    df['cond_pctrank_InternetService_TotalCharges'] = vals
    # 6. zscore_nonchurner_TotalCharges
    df['zscore_nonchurner_TotalCharges'] = zscore_against(tc, orig_nonchurner_tc).astype('float32')
    # 7. pctrank_orig_TotalCharges
    df['pctrank_orig_TotalCharges'] = pctrank_against(tc, orig_tc).astype('float32')
    # 8. pctrank_churner_TotalCharges
    df['pctrank_churner_TotalCharges'] = pctrank_against(tc, orig_churner_tc).astype('float32')
    # 9. cond_pctrank_Contract_TotalCharges
    vals = np.zeros(len(df), dtype='float32')
    for cat_val in orig['Contract'].unique():
        mask_df = (df['Contract'] == le_dict['Contract'].transform([str(cat_val)])[0]) if isinstance(cat_val, str) else (df['Contract'] == cat_val)
        ref = orig.loc[orig['Contract'] == cat_val, 'TotalCharges'].values
        if len(ref) > 0 and mask_df.sum() > 0:
            vals[mask_df] = pctrank_against(df.loc[mask_df, 'TotalCharges'].values, ref)
    df['cond_pctrank_Contract_TotalCharges'] = vals

exp3_features = [
    'pctrank_nonchurner_TotalCharges', 'zscore_churn_gap_TotalCharges',
    'pctrank_churn_gap_TotalCharges', 'resid_mean_InternetService_MonthlyCharges',
    'cond_pctrank_InternetService_TotalCharges', 'zscore_nonchurner_TotalCharges',
    'pctrank_orig_TotalCharges', 'pctrank_churner_TotalCharges',
    'cond_pctrank_Contract_TotalCharges'
]

baseline_features = v4_features + exp3_features
print(f"V4 + EXP3 Baseline: {len(baseline_features)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# OPTBINNING FEATURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[GEN] Generating OptBinning features...")

# We need the ORIGINAL (pre-label-encoded) categoricals for OptBinning
# Reload orig for fitting OptBinning (it needs string categoricals)
orig_raw = pd.read_csv(ORIG_PATH)
orig_raw[TARGET] = orig_raw[TARGET].map({'No': 0, 'Yes': 1}).astype(int)
orig_raw['TotalCharges'] = pd.to_numeric(orig_raw['TotalCharges'], errors='coerce')
orig_raw['TotalCharges'].fillna(orig_raw['TotalCharges'].median(), inplace=True)
if 'customerID' in orig_raw.columns:
    orig_raw.drop(columns=['customerID'], inplace=True)

# Also need raw train/test for OptBinning transforms
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

# 1D WoE features
print("\n  [A] 1D OptimalBinning WoE...")
woe_1d_features = []
iv_scores = {}

raw_cats = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
raw_nums = ['tenure', 'MonthlyCharges', 'TotalCharges']
raw_all = raw_cats + raw_nums

for col in raw_all:
    try:
        dtype = 'categorical' if orig_raw[col].dtype == 'object' or col == 'SeniorCitizen' else 'numerical'

        # For SeniorCitizen (int but categorical), convert to string
        fit_x = orig_raw[col].copy()
        train_x = train_raw[col].copy()
        test_x  = test_raw[col].copy()

        if dtype == 'categorical':
            fit_x = fit_x.astype(str)
            train_x = train_x.astype(str)
            test_x  = test_x.astype(str)

        encoder = OptimalBinning(name=col, dtype=dtype, solver='cp')
        encoder.fit(fit_x.values, orig_raw[TARGET].values)

        # Get IV
        try:
            binning_table = encoder.binning_table
            iv = binning_table.build()['IV'].iloc[-1]  # Total IV is last row
            iv_scores[col] = iv
        except:
            iv_scores[col] = 0.0

        feat_name = f'woe1d_{col}'
        train[feat_name] = encoder.transform(train_x.values, metric='woe').astype('float32')
        test[feat_name]  = encoder.transform(test_x.values, metric='woe').astype('float32')
        woe_1d_features.append(feat_name)
    except Exception as e:
        print(f"    SKIP {col}: {e}")

print(f"  Generated {len(woe_1d_features)} 1D WoE features")
if iv_scores:
    print("  Top 5 by IV:")
    for col, iv in sorted(iv_scores.items(), key=lambda x: -x[1])[:5]:
        print(f"    {col}: IV={iv:.4f}")

# 2D WoE features (joint binning of feature pairs)
print("\n  [B] 2D OptimalBinning WoE (feature interactions)...")
woe_2d_features = []

# Use top features by IV for 2D (to limit computation)
top_iv_features = sorted(iv_scores.keys(), key=lambda x: -iv_scores.get(x, 0))[:10]
print(f"  Using top 10 by IV for 2D: {top_iv_features}")

feature_pairs = list(combinations(top_iv_features, 2))
print(f"  Testing {len(feature_pairs)} feature pairs...")

for (f1, f2) in feature_pairs:
    try:
        dtype_x = 'categorical' if orig_raw[f1].dtype == 'object' or f1 == 'SeniorCitizen' else 'numerical'
        dtype_y = 'categorical' if orig_raw[f2].dtype == 'object' or f2 == 'SeniorCitizen' else 'numerical'

        fit_x = orig_raw[f1].astype(str) if dtype_x == 'categorical' else orig_raw[f1]
        fit_y = orig_raw[f2].astype(str) if dtype_y == 'categorical' else orig_raw[f2]
        tr_x = train_raw[f1].astype(str) if dtype_x == 'categorical' else train_raw[f1]
        tr_y = train_raw[f2].astype(str) if dtype_y == 'categorical' else train_raw[f2]
        te_x = test_raw[f1].astype(str) if dtype_x == 'categorical' else test_raw[f1]
        te_y = test_raw[f2].astype(str) if dtype_y == 'categorical' else test_raw[f2]

        encoder_2d = OptimalBinning2D(
            name_x=f1, name_y=f2,
            dtype_x=dtype_x, dtype_y=dtype_y,
            solver='cp'
        )
        encoder_2d.fit(fit_x.values, fit_y.values, orig_raw[TARGET].values)

        feat_name = f'woe2d_{f1}_{f2}'
        train[feat_name] = encoder_2d.transform(tr_x.values, tr_y.values, metric='woe').astype('float32')
        test[feat_name]  = encoder_2d.transform(te_x.values, te_y.values, metric='woe').astype('float32')
        woe_2d_features.append(feat_name)
    except Exception as e:
        pass  # Some pairs may fail

print(f"  Generated {len(woe_2d_features)} 2D WoE features")

# Clean up raw data
del train_raw, test_raw, orig_raw
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_features(feature_list, n_folds=2, label=""):
    """Quick LightGBM evaluation with n-fold CV"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    y = train[TARGET].values

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        X_tr = train.loc[tr_idx, feature_list].copy()
        y_tr = y[tr_idx]
        X_va = train.loc[va_idx, feature_list].copy()
        y_va = y[va_idx]

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)]
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        del model, X_tr, X_va
        gc.collect()

    return roc_auc_score(y, oof)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: BATCH SCREENING (2-fold fast)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 1: BATCH SCREENING (2-fold)")
print("=" * 80)

baseline_auc = evaluate_features(baseline_features, n_folds=2, label="V4+EXP3 Baseline")
print(f"\nV4+EXP3 Baseline (2-fold): {baseline_auc:.5f}")

batches = {
    'A_WoE_1D': woe_1d_features,
    'B_WoE_2D': woe_2d_features,
    'C_WoE_All': woe_1d_features + woe_2d_features,
}

print(f"\n{'Batch':<25} {'#':>5} {'AUC':>10} {'Delta':>10} {'Verdict'}")
print("-" * 65)

promising_batches = []
for name, feats in batches.items():
    if not feats:
        print(f"{name:<25} {0:>5} {'N/A':>10} {'N/A':>10} ⛔ Empty")
        continue
    auc = evaluate_features(baseline_features + feats, n_folds=2)
    delta = auc - baseline_auc
    verdict = "🔶 Promising" if delta > 0.00005 else ("❌ Hurts" if delta < -0.00010 else "⚠️  Neutral")
    print(f"{name:<25} {len(feats):>5} {auc:>10.5f} {delta:>+10.5f} {verdict}")
    if delta > -0.00020:  # Keep anything that doesn't badly hurt
        promising_batches.append((name, feats))

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: INDIVIDUAL ABLATION (2-fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 2: INDIVIDUAL ABLATION (2-fold)")
print("=" * 80)

# Test all WoE features individually
all_woe_features = woe_1d_features + woe_2d_features
print(f"\nTesting {len(all_woe_features)} features individually...")

individual_results = []
print(f"\n{'Feature':<50} {'AUC':>10} {'Delta':>10} {'V'}")
print("-" * 80)

for feat in all_woe_features:
    auc = evaluate_features(baseline_features + [feat], n_folds=2)
    delta = auc - baseline_auc
    verdict = "🔶" if delta > 0.00005 else ("❌" if delta < -0.00010 else "~")
    print(f"{feat:<50} {auc:>10.5f} {delta:>+10.5f} {verdict}")
    individual_results.append((feat, auc, delta))

# Sort by delta
individual_results.sort(key=lambda x: -x[2])

# Top features
top_features = [(f, a, d) for f, a, d in individual_results if d > 0.00005]
print(f"\n--- TOP FEATURES (Δ > +0.00005) [{len(top_features)}] ---")
for f, a, d in top_features:
    print(f"  {f}: {d:+.5f}")

# Worst hurters
worst = [(f, a, d) for f, a, d in individual_results if d < -0.00010]
print(f"\n--- WORST HURTERS [{len(worst)}] ---")
for f, a, d in worst:
    print(f"  {f}: {d:+.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: GREEDY FORWARD SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 3: GREEDY FORWARD SELECTION")
print("=" * 80)

candidates = [f for f, a, d in individual_results if d > 0.00003]
if not candidates:
    candidates = [f for f, a, d in individual_results[:5]]  # Take top 5 anyway

print(f"Candidates: {len(candidates)}")

selected = []
current_best = baseline_auc

for feat, _, _ in sorted([(f, a, d) for f, a, d in individual_results if f in candidates], key=lambda x: -x[2]):
    test_set = baseline_features + selected + [feat]
    auc = evaluate_features(test_set, n_folds=2)
    if auc > current_best:
        selected.append(feat)
        current_best = auc
        print(f"  ✅ ADD {feat}: {auc:.5f} (Δ = {auc - baseline_auc:+.5f})")
    else:
        print(f"  ❌ SKIP {feat}: {auc:.5f} (no improvement over {current_best:.5f})")

if not selected:
    print("\n  ⚠️ No features survived greedy selection.")
    print(f"\nTotal Time: {(time.time() - t0_all) / 60:.1f} min")
    exit()

print(f"\nSelected {len(selected)} features: {selected}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: 5-FOLD CONFIRMATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"STAGE 4: 5-FOLD CONFIRMATION (with {len(selected)} selected features)")
print("=" * 80)

# 5-fold baseline
print("\n5-fold V4+EXP3 Baseline...")
skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y = train[TARGET].values
oof_base = np.zeros(len(train))
for fold_idx, (tr_idx, va_idx) in enumerate(skf5.split(train, y)):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        train.loc[tr_idx, baseline_features], y[tr_idx],
        eval_set=[(train.loc[va_idx, baseline_features], y[va_idx])],
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)]
    )
    oof_base[va_idx] = model.predict_proba(train.loc[va_idx, baseline_features])[:, 1]
    del model; gc.collect()

base_5f = roc_auc_score(y, oof_base)
fold_scores_base = []
for _, va_idx in skf5.split(train, y):
    fold_scores_base.append(roc_auc_score(y[va_idx], oof_base[va_idx]))
print(f"  V4+EXP3 Baseline (5-fold): {base_5f:.5f}  Folds: {' | '.join(f'{s:.5f}' for s in fold_scores_base)}")

# 5-fold with selected features
print(f"\n5-fold V4+EXP3 + Selected ({len(selected)} features)...")
all_feat = baseline_features + selected
oof_sel = np.zeros(len(train))
for fold_idx, (tr_idx, va_idx) in enumerate(skf5.split(train, y)):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        train.loc[tr_idx, all_feat], y[tr_idx],
        eval_set=[(train.loc[va_idx, all_feat], y[va_idx])],
        callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)]
    )
    oof_sel[va_idx] = model.predict_proba(train.loc[va_idx, all_feat])[:, 1]
    del model; gc.collect()

sel_5f = roc_auc_score(y, oof_sel)
fold_scores_sel = []
for _, va_idx in skf5.split(train, y):
    fold_scores_sel.append(roc_auc_score(y[va_idx], oof_sel[va_idx]))
print(f"  V4+EXP3 + Selected (5-fold): {sel_5f:.5f}  Folds: {' | '.join(f'{s:.5f}' for s in fold_scores_sel)}")

delta_5f = sel_5f - base_5f
print(f"\n  FINAL DELTA: {delta_5f:+.5f}")
print(f"  Selected Features: {selected}")

if delta_5f > 0.00005:
    print(f"\n  ✅ CONFIRMED: {len(selected)} OptBinning features genuinely improve V4+EXP3!")
elif delta_5f > 0:
    print(f"\n  ⚠️ MARGINAL: Tiny improvement, may not be reliable.")
else:
    print(f"\n  ❌ REJECTED: OptBinning WoE features do NOT improve V4+EXP3 baseline.")

total_time = (time.time() - t0_all) / 60
print(f"\nTotal Time: {total_time:.1f} min")
