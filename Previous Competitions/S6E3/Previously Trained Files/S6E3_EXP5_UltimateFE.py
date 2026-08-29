"""
S6E3 EXP5 - Ultimate Feature Discovery (Final FE Search)
==========================================================
Goal: Exhaustively test EVERY remaining FE direction before moving to model diversity.
      After this, we should never need to come back to FE.

What we've ALREADY tried (EXP1-4):
  ❌ Risk flags, CLV, RFM, cross-interactions, GroupBy stats
  ❌ WoE (simple + OptBinning 1D + 2D)
  ❌ Multi-stat TE, conditional churn stats
  ✅ TotalCharges distribution features (9 winners in V6)

What we're testing NOW (10 new batches):
  A. MonthlyCharges distributions (pctrank/zscore vs churner/non-churner)
  B. Tenure distributions
  C. Derived numerical distributions (charges_dev, avg_monthly, ratio)
  D. More conditional groups (PaymentMethod, SeniorCitizen, PaperlessBilling)
  E. 3-way conditional pctrank (Contract × InternetService)
  F. Quantile distance features (distance to Q10/Q25/Q50/Q75/Q90)
  G. KDE density ratio (P(x|churn=1) / P(x|churn=0))
  H. KMeans cluster features from original
  I. Polynomial interactions between winning distribution features
  J. Nearest-neighbor distance to original churners/non-churners

Evaluation: 4-stage pipeline
  Stage 1: Batch screening (2-fold)
  Stage 2: Individual ablation (promising batches)
  Stage 3: Greedy forward selection
  Stage 4: 5-fold confirmation
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

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
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0: return np.zeros(len(values), dtype='float32')
    return ((values - mu) / sigma).astype('float32')

def evaluate_features(train_data, y, feature_list, n_folds=2):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_data))
    for _, (tr_idx, va_idx) in enumerate(skf.split(train_data, y)):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            train_data.loc[tr_idx, feature_list], y[tr_idx],
            eval_set=[(train_data.loc[va_idx, feature_list], y[va_idx])],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)]
        )
        oof[va_idx] = model.predict_proba(train_data.loc[va_idx, feature_list])[:, 1]
        del model; gc.collect()
    return roc_auc_score(y, oof)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("S6E3 EXP5 - Ultimate Feature Discovery")
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

# ═══════════════════════════════════════════════════════════════════════════════
# V6 BASELINE FEATURES (V4 + 9 EXP3 distribution features)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[PREP] Building V6 baseline...")

v6_features = list(CATS) + list(NUMS)

# V4 Core
for col in NUMS:
    freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
    for df in [train, test, orig]:
        df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
    v6_features.append(f'FREQ_{col}')

for df in [train, test, orig]:
    df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
    df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
    df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
v6_features += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for df in [train, test, orig]:
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
    df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
v6_features += ['service_count', 'has_internet', 'has_phone']

for col in CATS + NUMS:
    tmp = orig.groupby(col)[TARGET].mean()
    _name = f"ORIG_proba_{col}"
    for df in [train, test]:
        df[_name] = df[col].map(tmp).fillna(0.5).astype('float32')
    v6_features.append(_name)

# Label encode categoricals
le_dict = {}
for col in CATS:
    le = LabelEncoder()
    le.fit(pd.concat([train[col].astype(str), test[col].astype(str), orig[col].astype(str)]))
    for df in [train, test, orig]:
        df[col] = le.transform(df[col].astype(str))
    le_dict[col] = le

# EXP3 v3 winners (9 distribution features already in V6)
orig_churner_tc    = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
orig_nonchurner_tc = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
orig_tc = orig['TotalCharges'].values
orig_is_mc_mean = orig.groupby('InternetService')['MonthlyCharges'].mean()

for df in [train, test]:
    tc = df['TotalCharges'].values
    df['pctrank_nonchurner_TC']  = pctrank_against(tc, orig_nonchurner_tc)
    df['pctrank_churner_TC']     = pctrank_against(tc, orig_churner_tc)
    df['pctrank_orig_TC']        = pctrank_against(tc, orig_tc)
    df['zscore_churn_gap_TC']    = (np.abs(zscore_against(tc, orig_churner_tc)) - np.abs(zscore_against(tc, orig_nonchurner_tc))).astype('float32')
    df['zscore_nonchurner_TC']   = zscore_against(tc, orig_nonchurner_tc)
    df['pctrank_churn_gap_TC']   = (pctrank_against(tc, orig_churner_tc) - pctrank_against(tc, orig_nonchurner_tc)).astype('float32')
    df['resid_IS_MC'] = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc_mean).fillna(0)).astype('float32')
    
    vals = np.zeros(len(df), dtype='float32')
    for cat_val in orig['InternetService'].unique():
        mask = df['InternetService'] == cat_val
        ref = orig.loc[orig['InternetService'] == cat_val, 'TotalCharges'].values
        if len(ref) > 0 and mask.sum() > 0:
            vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
    df['cond_pctrank_IS_TC'] = vals
    
    vals = np.zeros(len(df), dtype='float32')
    for cat_val in orig['Contract'].unique():
        mask = df['Contract'] == cat_val
        ref = orig.loc[orig['Contract'] == cat_val, 'TotalCharges'].values
        if len(ref) > 0 and mask.sum() > 0:
            vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
    df['cond_pctrank_C_TC'] = vals

v6_dist = [
    'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
    'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
    'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
]
v6_features += v6_dist
print(f"V6 Baseline: {len(v6_features)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE 10 NEW FEATURE BATCHES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[GEN] Generating 10 feature batches...")

# Original sub-distributions
orig_churner_mc    = orig.loc[orig[TARGET] == 1, 'MonthlyCharges'].values
orig_nonchurner_mc = orig.loc[orig[TARGET] == 0, 'MonthlyCharges'].values
orig_mc = orig['MonthlyCharges'].values
orig_churner_t     = orig.loc[orig[TARGET] == 1, 'tenure'].values
orig_nonchurner_t  = orig.loc[orig[TARGET] == 0, 'tenure'].values
orig_t = orig['tenure'].values

batches = {}

# ─── BATCH A: MonthlyCharges distributions ────────────────────────────────────
batch_a = []
for df in [train, test]:
    mc = df['MonthlyCharges'].values
    df['pctrank_nonchurner_MC']  = pctrank_against(mc, orig_nonchurner_mc)
    df['pctrank_churner_MC']     = pctrank_against(mc, orig_churner_mc)
    df['pctrank_orig_MC']        = pctrank_against(mc, orig_mc)
    df['zscore_churn_gap_MC']    = (np.abs(zscore_against(mc, orig_churner_mc)) - np.abs(zscore_against(mc, orig_nonchurner_mc))).astype('float32')
    df['zscore_nonchurner_MC']   = zscore_against(mc, orig_nonchurner_mc)
    df['pctrank_churn_gap_MC']   = (pctrank_against(mc, orig_churner_mc) - pctrank_against(mc, orig_nonchurner_mc)).astype('float32')
batch_a = ['pctrank_nonchurner_MC', 'pctrank_churner_MC', 'pctrank_orig_MC',
           'zscore_churn_gap_MC', 'zscore_nonchurner_MC', 'pctrank_churn_gap_MC']
batches['A_MC_Dist'] = batch_a

# ─── BATCH B: Tenure distributions ────────────────────────────────────────────
batch_b = []
for df in [train, test]:
    t = df['tenure'].values.astype(float)
    df['pctrank_nonchurner_T']  = pctrank_against(t, orig_nonchurner_t)
    df['pctrank_churner_T']     = pctrank_against(t, orig_churner_t)
    df['pctrank_orig_T']        = pctrank_against(t, orig_t)
    df['zscore_churn_gap_T']    = (np.abs(zscore_against(t, orig_churner_t)) - np.abs(zscore_against(t, orig_nonchurner_t))).astype('float32')
    df['zscore_nonchurner_T']   = zscore_against(t, orig_nonchurner_t)
    df['pctrank_churn_gap_T']   = (pctrank_against(t, orig_churner_t) - pctrank_against(t, orig_nonchurner_t)).astype('float32')
batch_b = ['pctrank_nonchurner_T', 'pctrank_churner_T', 'pctrank_orig_T',
           'zscore_churn_gap_T', 'zscore_nonchurner_T', 'pctrank_churn_gap_T']
batches['B_Tenure_Dist'] = batch_b

# ─── BATCH C: Derived numerical distributions ─────────────────────────────────
batch_c = []
orig_cd = (orig['TotalCharges'] - orig['tenure'] * orig['MonthlyCharges']).values
orig_am = (orig['TotalCharges'] / (orig['tenure'] + 1)).values
orig_ratio = (orig['MonthlyCharges'] / (orig['TotalCharges'] + 1)).values
orig_cd_ch = orig_cd[orig[TARGET] == 1]
orig_cd_nc = orig_cd[orig[TARGET] == 0]
orig_am_ch = orig_am[orig[TARGET] == 1]
orig_am_nc = orig_am[orig[TARGET] == 0]
orig_ratio_ch = orig_ratio[orig[TARGET] == 1]
orig_ratio_nc = orig_ratio[orig[TARGET] == 0]

for df in [train, test]:
    cd = df['charges_deviation'].values
    am = df['avg_monthly_charges'].values
    ratio = df['monthly_to_total_ratio'].values
    df['pctrank_churn_gap_CD'] = (pctrank_against(cd, orig_cd_ch) - pctrank_against(cd, orig_cd_nc)).astype('float32')
    df['pctrank_orig_CD']      = pctrank_against(cd, orig_cd)
    df['zscore_churn_gap_CD']  = (np.abs(zscore_against(cd, orig_cd_ch)) - np.abs(zscore_against(cd, orig_cd_nc))).astype('float32')
    df['pctrank_churn_gap_AM'] = (pctrank_against(am, orig_am_ch) - pctrank_against(am, orig_am_nc)).astype('float32')
    df['pctrank_orig_AM']      = pctrank_against(am, orig_am)
    df['pctrank_churn_gap_RT'] = (pctrank_against(ratio, orig_ratio_ch) - pctrank_against(ratio, orig_ratio_nc)).astype('float32')
batch_c = ['pctrank_churn_gap_CD', 'pctrank_orig_CD', 'zscore_churn_gap_CD',
           'pctrank_churn_gap_AM', 'pctrank_orig_AM', 'pctrank_churn_gap_RT']
batches['C_Derived_Dist'] = batch_c

# ─── BATCH D: More conditional groups ─────────────────────────────────────────
batch_d = []
extra_cats = ['PaymentMethod', 'SeniorCitizen', 'PaperlessBilling', 'Partner', 'Dependents']
for cat_col in extra_cats:
    for num_col, num_ref_ch, num_ref_nc in [
        ('TotalCharges', orig_churner_tc, orig_nonchurner_tc),
        ('MonthlyCharges', orig_churner_mc, orig_nonchurner_mc)
    ]:
        feat_name = f'cond_pctrank_{cat_col[:3]}_{num_col[:2]}'
        for df in [train, test]:
            vals = np.zeros(len(df), dtype='float32')
            for cat_val in orig[cat_col].unique():
                mask = df[cat_col] == cat_val
                ref = orig.loc[orig[cat_col] == cat_val, num_col].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, num_col].values, ref)
            df[feat_name] = vals
        batch_d.append(feat_name)
        
    # Residual from group mean
    for num_col in ['TotalCharges', 'MonthlyCharges']:
        feat_name = f'resid_{cat_col[:3]}_{num_col[:2]}'
        grp_mean = orig.groupby(cat_col)[num_col].mean()
        for df in [train, test]:
            df[feat_name] = (df[num_col] - df[cat_col].map(grp_mean).fillna(0)).astype('float32')
        batch_d.append(feat_name)
batches['D_MoreCondGroups'] = batch_d

# ─── BATCH E: 3-way conditional pctrank ────────────────────────────────────────
batch_e = []
cross_pairs = [('Contract', 'InternetService'), ('Contract', 'PaymentMethod'), 
               ('InternetService', 'PaymentMethod')]
for cat1, cat2 in cross_pairs:
    feat_name = f'cond3_pctrank_{cat1[:3]}x{cat2[:3]}_TC'
    for df in [train, test]:
        vals = np.zeros(len(df), dtype='float32')
        for c1 in orig[cat1].unique():
            for c2 in orig[cat2].unique():
                mask_orig = (orig[cat1] == c1) & (orig[cat2] == c2)
                mask_df   = (df[cat1] == c1) & (df[cat2] == c2)
                ref = orig.loc[mask_orig, 'TotalCharges'].values
                if len(ref) > 2 and mask_df.sum() > 0:
                    vals[mask_df] = pctrank_against(df.loc[mask_df, 'TotalCharges'].values, ref)
        df[feat_name] = vals
    batch_e.append(feat_name)
    
    # Same for MonthlyCharges
    feat_name2 = f'cond3_pctrank_{cat1[:3]}x{cat2[:3]}_MC'
    for df in [train, test]:
        vals = np.zeros(len(df), dtype='float32')
        for c1 in orig[cat1].unique():
            for c2 in orig[cat2].unique():
                mask_orig = (orig[cat1] == c1) & (orig[cat2] == c2)
                mask_df   = (df[cat1] == c1) & (df[cat2] == c2)
                ref = orig.loc[mask_orig, 'MonthlyCharges'].values
                if len(ref) > 2 and mask_df.sum() > 0:
                    vals[mask_df] = pctrank_against(df.loc[mask_df, 'MonthlyCharges'].values, ref)
        df[feat_name2] = vals
    batch_e.append(feat_name2)
batches['E_3WayCond'] = batch_e

# ─── BATCH F: Quantile distance features ──────────────────────────────────────
batch_f = []
for num_col, ch_ref, nc_ref in [
    ('TotalCharges', orig_churner_tc, orig_nonchurner_tc),
    ('MonthlyCharges', orig_churner_mc, orig_nonchurner_mc),
]:
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(ch_ref, q_val)
        nc_q = np.quantile(nc_ref, q_val)
        feat_ch = f'dist_{num_col[:2]}_ch_{q_label}'
        feat_nc = f'dist_{num_col[:2]}_nc_{q_label}'
        feat_gap = f'qdist_gap_{num_col[:2]}_{q_label}'
        for df in [train, test]:
            df[feat_ch]  = np.abs(df[num_col] - ch_q).astype('float32')
            df[feat_nc]  = np.abs(df[num_col] - nc_q).astype('float32')
            df[feat_gap] = (df[feat_nc] - df[feat_ch]).astype('float32')  # negative = closer to churners
        batch_f += [feat_ch, feat_nc, feat_gap]
batches['F_QuantileDist'] = batch_f

# ─── BATCH G: KDE density ratio ───────────────────────────────────────────────
batch_g = []
from scipy.stats import gaussian_kde
for num_col, ch_ref, nc_ref in [
    ('TotalCharges', orig_churner_tc, orig_nonchurner_tc),
    ('MonthlyCharges', orig_churner_mc, orig_nonchurner_mc),
    ('tenure', orig_churner_t.astype(float), orig_nonchurner_t.astype(float)),
]:
    try:
        kde_ch = gaussian_kde(ch_ref)
        kde_nc = gaussian_kde(nc_ref)
        feat_name = f'kde_logratio_{num_col[:2]}'
        for df in [train, test]:
            x = df[num_col].values.astype(float)
            p_ch = kde_ch(x) + 1e-10
            p_nc = kde_nc(x) + 1e-10
            df[feat_name] = np.log(p_ch / p_nc).astype('float32')
        batch_g.append(feat_name)
    except:
        pass
batches['G_KDE_Ratio'] = batch_g

# ─── BATCH H: KMeans cluster features ─────────────────────────────────────────
batch_h = []
cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
orig_cluster_data = orig[cluster_features].values.astype(float)
# Standardize
c_mean = orig_cluster_data.mean(axis=0)
c_std  = orig_cluster_data.std(axis=0) + 1e-8

for n_clusters in [3, 5, 8]:
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit((orig_cluster_data - c_mean) / c_std)
    
    feat_name = f'cluster_{n_clusters}'
    for df in [train, test]:
        df_data = df[cluster_features].values.astype(float)
        df[feat_name] = km.predict((df_data - c_mean) / c_std).astype('float32')
    batch_h.append(feat_name)
    
    # Distance to nearest cluster center
    feat_dist = f'cluster_dist_{n_clusters}'
    for df in [train, test]:
        df_data = df[cluster_features].values.astype(float)
        dists = km.transform((df_data - c_mean) / c_std)
        df[feat_dist] = dists.min(axis=1).astype('float32')
    batch_h.append(feat_dist)
    
    # Churn rate per cluster from original
    orig_clusters = km.predict((orig_cluster_data - c_mean) / c_std)
    cluster_churn = {}
    for c in range(n_clusters):
        mask = orig_clusters == c
        if mask.sum() > 0:
            cluster_churn[c] = orig.loc[mask, TARGET].mean()
        else:
            cluster_churn[c] = 0.5
    feat_cr = f'cluster_churnrate_{n_clusters}'
    for df in [train, test]:
        df[feat_cr] = df[feat_name].map(cluster_churn).fillna(0.5).astype('float32')
    batch_h.append(feat_cr)

batches['H_Clusters'] = batch_h

# ─── BATCH I: Polynomial interactions between winning dist features ────────────
batch_i = []
dist_pairs = [
    ('pctrank_nonchurner_TC', 'pctrank_nonchurner_MC'),
    ('pctrank_nonchurner_TC', 'pctrank_orig_T'),
    ('zscore_churn_gap_TC', 'zscore_churn_gap_MC'),
    ('pctrank_churn_gap_TC', 'pctrank_churn_gap_MC'),
    ('pctrank_orig_TC', 'pctrank_orig_MC'),
    ('pctrank_nonchurner_TC', 'zscore_churn_gap_TC'),
]
for f1, f2 in dist_pairs:
    if f1 in train.columns and f2 in train.columns:
        feat_mul = f'{f1[:15]}_x_{f2[:15]}'
        feat_div = f'{f1[:15]}_d_{f2[:15]}'
        for df in [train, test]:
            df[feat_mul] = (df[f1] * df[f2]).astype('float32')
            df[feat_div] = (df[f1] / (df[f2] + 1e-6)).astype('float32')
        batch_i += [feat_mul, feat_div]
batches['I_PolyInteract'] = batch_i

# ─── BATCH J: Nearest-neighbor distance to original churners/non-churners ─────
batch_j = []
nn_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
orig_nn = orig[nn_features].values.astype(float)
orig_nn_norm = (orig_nn - c_mean) / c_std

ch_mask = orig[TARGET].values == 1
nc_mask = orig[TARGET].values == 0

nn_ch = NearestNeighbors(n_neighbors=5).fit(orig_nn_norm[ch_mask])
nn_nc = NearestNeighbors(n_neighbors=5).fit(orig_nn_norm[nc_mask])

for df in [train, test]:
    df_nn = df[nn_features].values.astype(float)
    df_nn_norm = (df_nn - c_mean) / c_std
    
    dist_ch, _ = nn_ch.kneighbors(df_nn_norm)
    dist_nc, _ = nn_nc.kneighbors(df_nn_norm)
    
    df['nn5_dist_churner']     = dist_ch.mean(axis=1).astype('float32')
    df['nn5_dist_nonchurner']  = dist_nc.mean(axis=1).astype('float32')
    df['nn5_dist_ratio']       = (dist_ch.mean(axis=1) / (dist_nc.mean(axis=1) + 1e-6)).astype('float32')
    df['nn5_dist_gap']         = (dist_nc.mean(axis=1) - dist_ch.mean(axis=1)).astype('float32')
    df['nn1_dist_churner']     = dist_ch[:, 0].astype('float32')
    df['nn1_dist_nonchurner']  = dist_nc[:, 0].astype('float32')
batch_j = ['nn5_dist_churner', 'nn5_dist_nonchurner', 'nn5_dist_ratio', 
           'nn5_dist_gap', 'nn1_dist_churner', 'nn1_dist_nonchurner']
batches['J_NearestNeighbor'] = batch_j

# Print summary
total_new = 0
for name, feats in batches.items():
    print(f"  {name}: {len(feats)} features")
    total_new += len(feats)
print(f"  TOTAL: {total_new} new features")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: BATCH SCREENING (2-fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 1: BATCH SCREENING (2-fold)")
print("=" * 80)

y = train[TARGET].values
baseline_auc = evaluate_features(train, y, v6_features, n_folds=2)
print(f"\nV6 Baseline (2-fold): {baseline_auc:.5f}")

print(f"\n{'Batch':<25} {'#':>5} {'AUC':>10} {'Delta':>10} {'Verdict'}")
print("-" * 65)

promising_batches = {}
for name, feats in batches.items():
    if not feats:
        print(f"{name:<25} {0:>5} {'N/A':>10} {'N/A':>10} ⛔ Empty")
        continue
    auc = evaluate_features(train, y, v6_features + feats, n_folds=2)
    delta = auc - baseline_auc
    if delta > 0.00010:
        verdict = "🔶 Promising"
    elif delta > 0.00000:
        verdict = "⚠️  Neutral"
    elif delta > -0.00010:
        verdict = "⚠️  Neutral"
    else:
        verdict = "❌ Hurts"
    print(f"{name:<25} {len(feats):>5} {auc:>10.5f} {delta:>+10.5f} {verdict}")
    promising_batches[name] = feats

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: INDIVIDUAL ABLATION (all features from all batches, 2-fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 2: INDIVIDUAL ABLATION (2-fold)")
print("=" * 80)

all_new_features = []
for feats in batches.values():
    all_new_features.extend(feats)
print(f"\nTesting {len(all_new_features)} features individually...")

individual_results = []
print(f"\n{'Feature':<50} {'AUC':>10} {'Delta':>10} {'V'}")
print("-" * 80)

for feat in all_new_features:
    auc = evaluate_features(train, y, v6_features + [feat], n_folds=2)
    delta = auc - baseline_auc
    verdict = "🔶" if delta > 0.00005 else ("❌" if delta < -0.00010 else "~")
    print(f"{feat:<50} {auc:>10.5f} {delta:>+10.5f} {verdict}")
    individual_results.append((feat, auc, delta))

individual_results.sort(key=lambda x: -x[2])

top_features = [(f, a, d) for f, a, d in individual_results if d > 0.00005]
print(f"\n--- TOP FEATURES (Δ > +0.00005) [{len(top_features)}] ---")
for f, a, d in top_features:
    print(f"  {f}: {d:+.5f}")

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
if len(candidates) < 3:
    candidates = [f for f, a, d in individual_results[:10]]
print(f"Candidates: {len(candidates)}")

selected = []
current_best = baseline_auc

for feat, _, _ in sorted([(f, a, d) for f, a, d in individual_results if f in candidates], key=lambda x: -x[2]):
    test_set = v6_features + selected + [feat]
    auc = evaluate_features(train, y, test_set, n_folds=2)
    if auc > current_best:
        selected.append(feat)
        current_best = auc
        print(f"  ✅ ADD {feat}: {auc:.5f} (Δ = {auc - baseline_auc:+.5f})")
    else:
        print(f"  ❌ SKIP {feat}: {auc:.5f} (no improvement over {current_best:.5f})")

if not selected:
    print("\n  ⚠️ No features survived greedy selection.")
    print(f"\n  CONCLUSION: V6's feature set is already optimal. No further FE needed.")
    print(f"\nTotal Time: {(time.time() - t0_all) / 60:.1f} min")
else:
    print(f"\nSelected {len(selected)} features: {selected}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 4: 5-FOLD CONFIRMATION
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"STAGE 4: 5-FOLD CONFIRMATION (with {len(selected)} selected features)")
    print("=" * 80)

    # 5-fold baseline
    print("\n5-fold V6 Baseline...")
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_base = np.zeros(len(train))
    for fold_idx, (tr_idx, va_idx) in enumerate(skf5.split(train, y)):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            train.loc[tr_idx, v6_features], y[tr_idx],
            eval_set=[(train.loc[va_idx, v6_features], y[va_idx])],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)]
        )
        oof_base[va_idx] = model.predict_proba(train.loc[va_idx, v6_features])[:, 1]
        del model; gc.collect()

    base_5f = roc_auc_score(y, oof_base)
    fold_scores_base = []
    for _, va_idx in skf5.split(train, y):
        fold_scores_base.append(roc_auc_score(y[va_idx], oof_base[va_idx]))
    print(f"  V6 Baseline (5-fold): {base_5f:.5f}  Folds: {' | '.join(f'{s:.5f}' for s in fold_scores_base)}")

    # 5-fold with selected features
    print(f"\n5-fold V6 + Selected ({len(selected)} features)...")
    all_feat = v6_features + selected
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
    print(f"  V6 + Selected (5-fold): {sel_5f:.5f}  Folds: {' | '.join(f'{s:.5f}' for s in fold_scores_sel)}")

    delta_5f = sel_5f - base_5f
    print(f"\n  FINAL DELTA: {delta_5f:+.5f}")
    print(f"  Selected Features: {selected}")

    if delta_5f > 0.00005:
        print(f"\n  ✅ CONFIRMED: {len(selected)} new features improve V6!")
    elif delta_5f > 0:
        print(f"\n  ⚠️ MARGINAL: Tiny improvement, may not be reliable.")
    else:
        print(f"\n  ❌ REJECTED: Features do NOT improve V6 baseline.")

total_time = (time.time() - t0_all) / 60
print(f"\nTotal Time: {total_time:.1f} min")
