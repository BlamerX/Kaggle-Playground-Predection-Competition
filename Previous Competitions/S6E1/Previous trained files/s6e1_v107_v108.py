"""
S6E1 V107-V108 - CatBoost Improvements on V103
================================================
V107: More KD features (add V105, V99, V101 predictions)
V108: CatBoost DART mode (like V88)

Base: V103 (8.54774 LB) - Best single model ever
Goal: Beat 8.54774 with additional features/techniques
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
import warnings
import os
import time
import sys
import subprocess

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V107-V108 - CatBoost Improvements on V103")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_path = "Previous trained files/"

TARGET = "exam_score"
y = train_df[TARGET].values
y_orig = original_df[TARGET].values

# Load ALL OOF files
print("\nLoading OOF files...")

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    print(f"  ✓ {name} loaded")
    return oof[col].values, sub['exam_score'].values

# V103 baseline (V77)
v77_train, v77_test = load_oof("V77 (CatBoost baseline)", "oof_v77.csv", "submission_v77.csv")

# V103 KD features
v61_train, v61_test = load_oof("V61 (TabM)", "oof_v61.csv", "submission_v61.csv")
v70_train, v70_test = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
v67_train, v67_test = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
v73_train, v73_test = load_oof("V73 (XGBoost)", "oof_v73.csv", "submission_v73.csv")

# NEW: Additional KD features for V107
v105_train, v105_test = load_oof("V105 (TabM+KD)", "oof_v105.csv", "submission_v105.csv")
v99_train, v99_test = load_oof("V99 (XGB+KD)", "oof_v99.csv", "submission_v99.csv")
v101_train, v101_test = load_oof("V101 (XGB+Multi-KD)", "oof_v101.csv", "submission_v101.csv")

print(f"\nV77 baseline RMSE: {np.sqrt(mean_squared_error(y, v77_train)):.5f}")

# ============================================================================
# 2. CMT ENCODING
# ============================================================================

CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty']

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    
    def fit(self, X, y):
        for col in self.cat_cols:
            df_temp = pd.DataFrame({col: X[col], 'y': y})
            group_means = df_temp.groupby(col, dropna=False)['y'].mean()
            self.mappings_[col] = {cat: i for i, cat in enumerate(group_means.sort_values().index)}
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            X[col] = X[col].map(mapping)
        return X

cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
tmp = cmtencoder.fit_transform(train_df[CATS], y).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING")
print("="*80)

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, kd_preds=None):
    df = df.copy()
    eps = 1e-5
    
    # Basic features
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    
    sh_pos = df['study_hours'].clip(lower=0)
    ca_pos = df['class_attendance'].clip(lower=0)
    df['log_study_hours'] = np.log1p(sh_pos)
    df['log_class_attendance'] = np.log1p(ca_pos)
    
    df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_hours_times_sleep'] = df['study_hours'] * df['sleep_hours']
    df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    
    # Discussion features (Thomas)
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 
        0.35 * df['class_attendance'] + 
        1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    
    # Sin features (Vladimir)
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    
    # KD predictions
    if kd_preds is not None:
        for name, pred in kd_preds.items():
            df[f'{name}_pred'] = pred
    
    return df

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]

# ============================================================================
# COMMON SETTINGS
# ============================================================================

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)
ALPHA = 0.1

# ============================================================================
# V107: MORE KD FEATURES
# ============================================================================

print(f"\n{'='*80}")
print("V107: CATBOOST + V77 + EXTENDED MULTI-KD")
print("="*80)

# V103 KD + NEW: V105, V99, V101
kd_v107 = {
    'tabm': v61_train, 'ftt': v70_train, 'lgb': v67_train, 'xgb': v73_train,
    'tabm_kd': v105_train, 'xgb_kd': v99_train, 'xgb_multi_kd': v101_train  # NEW
}
kd_v107_test = {
    'tabm': v61_test, 'ftt': v70_test, 'lgb': v67_test, 'xgb': v73_test,
    'tabm_kd': v105_test, 'xgb_kd': v99_test, 'xgb_multi_kd': v101_test  # NEW
}

train_v107 = add_features(train_df, kd_v107)
test_v107 = add_features(test_df, kd_v107_test)
orig_v107 = add_features(original_df, None)
for k in kd_v107.keys():
    orig_v107[f'{k}_pred'] = 0

FEATURE_COLS = [c for c in train_v107.columns if c not in [TARGET, 'id', 'student_id']]

for col in CATS:
    train_v107[col] = train_v107[col].astype('category')
    test_v107[col] = test_v107[col].astype('category')
    orig_v107[col] = orig_v107[col].astype('category')

X_train = train_v107[FEATURE_COLS]
X_test = test_v107[FEATURE_COLS]
X_orig = orig_v107[FEATURE_COLS]

residuals = y - v77_train

print(f"Features: {len(FEATURE_COLS)} (V103 had 36)")
print(f"V77 baseline RMSE: {np.sqrt(mean_squared_error(y, v77_train)):.5f}")

cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in CATS]

catboost_params = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'task_type': 'GPU',
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'verbose': 0
}

# Phase 1
oof_res_v107 = np.zeros(len(train_df))
test_res_v107 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig))])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=val_pool)
    
    oof_res_v107[val_idx] = model.predict(X_val)
    test_res_v107.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Phase 2: Boosted PL
test_pseudo = np.clip(v77_test + np.mean(test_res_v107, axis=0) + ALPHA * oof_res_v107.mean(), 0, 100)
test_pseudo_res = test_pseudo - v77_test

oof_final_v107 = np.zeros(len(train_df))
test_final_v107 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig, X_test], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig)), test_pseudo_res])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=val_pool)
    
    oof_final_v107[val_idx] = model.predict(X_val)
    test_final_v107.append(model.predict(X_test))

final_oof_v107 = np.clip(v77_train + oof_final_v107, 0, 100)
final_test_v107 = np.clip(v77_test + np.mean(test_final_v107, axis=0), 0, 100)

v107_rmse = np.sqrt(mean_squared_error(y, final_oof_v107))
print(f"\nV107 OOF RMSE: {v107_rmse:.5f}")

# ============================================================================
# V108: CATBOOST DART MODE
# ============================================================================

print(f"\n{'='*80}")
print("V108: CATBOOST DART MODE + V103 KD")
print("="*80)

# Use V103 KD (not extended)
kd_v108 = {'tabm': v61_train, 'ftt': v70_train, 'lgb': v67_train, 'xgb': v73_train}
kd_v108_test = {'tabm': v61_test, 'ftt': v70_test, 'lgb': v67_test, 'xgb': v73_test}

train_v108 = add_features(train_df, kd_v108)
test_v108 = add_features(test_df, kd_v108_test)
orig_v108 = add_features(original_df, None)
for k in kd_v108.keys():
    orig_v108[f'{k}_pred'] = 0

FEATURE_COLS_V108 = [c for c in train_v108.columns if c not in [TARGET, 'id', 'student_id']]

for col in CATS:
    train_v108[col] = train_v108[col].astype('category')
    test_v108[col] = test_v108[col].astype('category')
    orig_v108[col] = orig_v108[col].astype('category')

X_train_v108 = train_v108[FEATURE_COLS_V108]
X_test_v108 = test_v108[FEATURE_COLS_V108]
X_orig_v108 = orig_v108[FEATURE_COLS_V108]

cat_indices_v108 = [i for i, c in enumerate(FEATURE_COLS_V108) if c in CATS]

# DART-like params (similar to V88)
catboost_dart_params = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'task_type': 'GPU',
    'random_seed': 42,
    'early_stopping_rounds': 150,
    'verbose': 0
}

print(f"Features: {len(FEATURE_COLS_V108)}")

# Phase 1
oof_res_v108 = np.zeros(len(train_df))
test_res_v108 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v108), start=1):
    X_tr, X_val = X_train_v108.iloc[tr_idx], X_train_v108.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v108], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v108))])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices_v108)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices_v108)
    
    model = CatBoostRegressor(**catboost_dart_params)
    model.fit(train_pool, eval_set=val_pool)
    
    oof_res_v108[val_idx] = model.predict(X_val)
    test_res_v108.append(model.predict(X_test_v108))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Phase 2
test_pseudo_v108 = np.clip(v77_test + np.mean(test_res_v108, axis=0) + ALPHA * oof_res_v108.mean(), 0, 100)
test_pseudo_res_v108 = test_pseudo_v108 - v77_test

oof_final_v108 = np.zeros(len(train_df))
test_final_v108 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v108), start=1):
    X_tr, X_val = X_train_v108.iloc[tr_idx], X_train_v108.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v108, X_test_v108], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v108)), test_pseudo_res_v108])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices_v108)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices_v108)
    
    model = CatBoostRegressor(**catboost_dart_params)
    model.fit(train_pool, eval_set=val_pool)
    
    oof_final_v108[val_idx] = model.predict(X_val)
    test_final_v108.append(model.predict(X_test_v108))

final_oof_v108 = np.clip(v77_train + oof_final_v108, 0, 100)
final_test_v108 = np.clip(v77_test + np.mean(test_final_v108, axis=0), 0, 100)

v108_rmse = np.sqrt(mean_squared_error(y, final_oof_v108))
print(f"\nV108 OOF RMSE: {v108_rmse:.5f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

v103_rmse = 8.56053
v103_lb = 8.54774

print(f"""
| Version | Model | OOF RMSE | vs V103 | LB Score |
|---------|-------|----------|---------|----------|
| V103 | CatBoost + V77 + KD | {v103_rmse:.5f} | - | {v103_lb:.5f} |
| V107 | + Extended KD (7 models) | {v107_rmse:.5f} | {v103_rmse - v107_rmse:+.5f} | ? |
| V108 | + DART Mode | {v108_rmse:.5f} | {v103_rmse - v108_rmse:+.5f} | ? |
""")

results = [
    ('V107', v107_rmse, final_oof_v107, final_test_v107),
    ('V108', v108_rmse, final_oof_v108, final_test_v108)
]
best = min(results, key=lambda x: x[1])
print(f"✅ Best: {best[0]} with OOF RMSE {best[1]:.5f}")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

for name, rmse, oof, test in results:
    pd.DataFrame({'id': test_df['id'], 'exam_score': test}).to_csv(f"submission_{name.lower()}.csv", index=False)
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof}).to_csv(f"oof_{name.lower()}.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved: V107, V108")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
