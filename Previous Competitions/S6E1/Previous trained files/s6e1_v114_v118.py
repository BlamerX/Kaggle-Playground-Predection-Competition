"""
S6E1 V114-V118 - XGBoost & LightGBM Improvements
=================================================
V114: XGBoost DART mode + Multi-KD (base: V101)
V115: XGBoost + Ridge meta-feature (base: V101)
V116: XGBoost + Binned features (base: V101)
V117: LightGBM DART mode only (base: V67)
V118: LightGBM + Ridge meta only (base: V67)

Goal: Diversify single models beyond CatBoost
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V114-V118 - XGBoost & LightGBM Improvements")
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

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    print(f"  ✓ {name} loaded")
    return oof[col].values, sub['exam_score'].values

print("\nLoading OOF files...")

# Baselines - use BEST versions
v101_train, v101_test = load_oof("V101 (XGBoost BEST)", "oof_v101.csv", "submission_v101.csv")
v67_train, v67_test = load_oof("V67 (LightGBM BEST)", "oof_v67.csv", "submission_v67.csv")

# KD features
v61_train, v61_test = load_oof("V61 (TabM)", "oof_v61.csv", "submission_v61.csv")
v70_train, v70_test = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
v77_train, v77_test = load_oof("V77 (CatBoost)", "oof_v77.csv", "submission_v77.csv")

print(f"\nV101 XGB baseline RMSE: {np.sqrt(mean_squared_error(y, v101_train)):.5f}")
print(f"V67 LGB baseline RMSE: {np.sqrt(mean_squared_error(y, v67_train)):.5f}")

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

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, kd_preds=None, add_binned=False):
    df = df.copy()
    eps = 1e-5
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
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 0.35 * df['class_attendance'] + 1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    
    if add_binned:
        df['study_bin'] = pd.cut(df['study_hours'], bins=5, labels=False)
        df['sleep_bin'] = pd.cut(df['sleep_hours'], bins=5, labels=False)
        df['attendance_bin'] = pd.cut(df['class_attendance'], bins=5, labels=False)
    
    if kd_preds is not None:
        for name, pred in kd_preds.items():
            df[f'{name}_pred'] = pred
    return df

# ============================================================================
# COMMON SETTINGS
# ============================================================================

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)
ALPHA = 0.1

# ============================================================================
# V114: XGBoost DART + Multi-KD
# ============================================================================

print(f"\n{'='*80}")
print("V114: XGBoost DART + Multi-KD")
print("="*80)

kd_xgb = {'tabm': v61_train, 'ftt': v70_train, 'lgb': v67_train, 'cat': v77_train}
kd_xgb_test = {'tabm': v61_test, 'ftt': v70_test, 'lgb': v67_test, 'cat': v77_test}

train_v114 = add_features(train_df, kd_xgb)
test_v114 = add_features(test_df, kd_xgb_test)
orig_v114 = add_features(original_df, None)
for k in kd_xgb.keys():
    orig_v114[f'{k}_pred'] = 0

FEATURE_COLS_V114 = [c for c in train_v114.columns if c not in [TARGET, 'id', 'student_id'] + CATS]
for col in CATS:
    train_v114[col] = train_v114[col].astype('category')
    test_v114[col] = test_v114[col].astype('category')
    orig_v114[col] = orig_v114[col].astype('category')

X_train_v114 = train_v114[FEATURE_COLS_V114 + CATS]
X_test_v114 = test_v114[FEATURE_COLS_V114 + CATS]
X_orig_v114 = orig_v114[FEATURE_COLS_V114 + CATS]
residuals_xgb = y - v101_train

print(f"Features: {len(FEATURE_COLS_V114 + CATS)}")

# XGBoost DART params
xgb_dart_params = {
    'n_estimators': 5000,
    'learning_rate': 0.02,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_lambda': 5,
    'reg_alpha': 0.1,
    'booster': 'dart',
    'sample_type': 'weighted',
    'rate_drop': 0.1,
    'enable_categorical': True,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'early_stopping_rounds': 100
}

v114_start = time.time()
oof_v114 = np.zeros(len(train_df))
test_v114_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v114), start=1):
    X_tr, X_val = X_train_v114.iloc[tr_idx], X_train_v114.iloc[val_idx]
    res_tr, res_val = residuals_xgb[tr_idx], residuals_xgb[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v114], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v114))])
    
    model = xgb.XGBRegressor(**xgb_dart_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)], verbose=False)
    
    oof_v114[val_idx] = model.predict(X_val)
    test_v114_preds.append(model.predict(X_test_v114))
    
    if fold % 5 == 0:
        print(f"  Fold {fold}/{N_FOLDS} done")

final_oof_v114 = np.clip(v101_train + oof_v114, 0, 100)
final_test_v114 = np.clip(v101_test + np.mean(test_v114_preds, axis=0), 0, 100)
v114_rmse = np.sqrt(mean_squared_error(y, final_oof_v114))
v114_time = (time.time() - v114_start) / 60
print(f"V114 OOF RMSE: {v114_rmse:.5f} ({v114_time:.1f} min)")

# ============================================================================
# V115: XGBoost + Ridge Meta
# ============================================================================

print(f"\n{'='*80}")
print("V115: XGBoost + Ridge Meta")
print("="*80)

# Create Ridge meta-feature
NUM_COLS = [c for c in train_v114.columns if c not in CATS + [TARGET, 'id', 'student_id']]
ridge_oof = np.zeros(len(train_df))
ridge_test = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_v114), start=1):
    X_tr = train_v114.iloc[tr_idx][NUM_COLS].values
    X_val = train_v114.iloc[val_idx][NUM_COLS].values
    y_tr = y[tr_idx]
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_tr, y_tr)
    ridge_oof[val_idx] = ridge.predict(X_val)
    ridge_test.append(ridge.predict(test_v114[NUM_COLS].values))

ridge_test_pred = np.mean(ridge_test, axis=0)
print(f"Ridge OOF RMSE: {np.sqrt(mean_squared_error(y, ridge_oof)):.5f}")

# Add Ridge to features
train_v115 = train_v114.copy()
train_v115['ridge_pred'] = ridge_oof
test_v115 = test_v114.copy()
test_v115['ridge_pred'] = ridge_test_pred
orig_v115 = orig_v114.copy()
orig_v115['ridge_pred'] = 0

FEATURE_COLS_V115 = FEATURE_COLS_V114 + ['ridge_pred']
X_train_v115 = train_v115[FEATURE_COLS_V115 + CATS]
X_test_v115 = test_v115[FEATURE_COLS_V115 + CATS]
X_orig_v115 = orig_v115[FEATURE_COLS_V115 + CATS]

print(f"Features: {len(FEATURE_COLS_V115 + CATS)}")

v115_start = time.time()
oof_v115 = np.zeros(len(train_df))
test_v115_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v115), start=1):
    X_tr, X_val = X_train_v115.iloc[tr_idx], X_train_v115.iloc[val_idx]
    res_tr, res_val = residuals_xgb[tr_idx], residuals_xgb[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v115], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v115))])
    
    model = xgb.XGBRegressor(**xgb_dart_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)], verbose=False)
    
    oof_v115[val_idx] = model.predict(X_val)
    test_v115_preds.append(model.predict(X_test_v115))
    
    if fold % 5 == 0:
        print(f"  Fold {fold}/{N_FOLDS} done")

final_oof_v115 = np.clip(v101_train + oof_v115, 0, 100)
final_test_v115 = np.clip(v101_test + np.mean(test_v115_preds, axis=0), 0, 100)
v115_rmse = np.sqrt(mean_squared_error(y, final_oof_v115))
v115_time = (time.time() - v115_start) / 60
print(f"V115 OOF RMSE: {v115_rmse:.5f} ({v115_time:.1f} min)")

# ============================================================================
# V116: XGBoost + Binned Features
# ============================================================================

print(f"\n{'='*80}")
print("V116: XGBoost + Binned Features")
print("="*80)

train_v116 = add_features(train_df, kd_xgb, add_binned=True)
test_v116 = add_features(test_df, kd_xgb_test, add_binned=True)
orig_v116 = add_features(original_df, None, add_binned=True)
for k in kd_xgb.keys():
    orig_v116[f'{k}_pred'] = 0

FEATURE_COLS_V116 = [c for c in train_v116.columns if c not in [TARGET, 'id', 'student_id'] + CATS]
for col in CATS:
    train_v116[col] = train_v116[col].astype('category')
    test_v116[col] = test_v116[col].astype('category')
    orig_v116[col] = orig_v116[col].astype('category')

X_train_v116 = train_v116[FEATURE_COLS_V116 + CATS]
X_test_v116 = test_v116[FEATURE_COLS_V116 + CATS]
X_orig_v116 = orig_v116[FEATURE_COLS_V116 + CATS]

print(f"Features: {len(FEATURE_COLS_V116 + CATS)}")

v116_start = time.time()
oof_v116 = np.zeros(len(train_df))
test_v116_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v116), start=1):
    X_tr, X_val = X_train_v116.iloc[tr_idx], X_train_v116.iloc[val_idx]
    res_tr, res_val = residuals_xgb[tr_idx], residuals_xgb[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v116], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v116))])
    
    model = xgb.XGBRegressor(**xgb_dart_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)], verbose=False)
    
    oof_v116[val_idx] = model.predict(X_val)
    test_v116_preds.append(model.predict(X_test_v116))
    
    if fold % 5 == 0:
        print(f"  Fold {fold}/{N_FOLDS} done")

final_oof_v116 = np.clip(v101_train + oof_v116, 0, 100)
final_test_v116 = np.clip(v101_test + np.mean(test_v116_preds, axis=0), 0, 100)
v116_rmse = np.sqrt(mean_squared_error(y, final_oof_v116))
v116_time = (time.time() - v116_start) / 60
print(f"V116 OOF RMSE: {v116_rmse:.5f} ({v116_time:.1f} min)")

# ============================================================================
# V117: LightGBM DART Mode
# ============================================================================

print(f"\n{'='*80}")
print("V117: LightGBM DART Mode")
print("="*80)

# LightGBM features (no KD - V104 proved it hurts)
train_v117 = add_features(train_df, None)
test_v117 = add_features(test_df, None)
orig_v117 = add_features(original_df, None)

FEATURE_COLS_V117 = [c for c in train_v117.columns if c not in [TARGET, 'id', 'student_id'] + CATS]
for col in CATS:
    train_v117[col] = train_v117[col].astype('category')
    test_v117[col] = test_v117[col].astype('category')
    orig_v117[col] = orig_v117[col].astype('category')

X_train_v117 = train_v117[FEATURE_COLS_V117 + CATS]
X_test_v117 = test_v117[FEATURE_COLS_V117 + CATS]
X_orig_v117 = orig_v117[FEATURE_COLS_V117 + CATS]
residuals_lgb = y - v67_train

print(f"Features: {len(FEATURE_COLS_V117 + CATS)}")

lgb_dart_params = {
    'n_estimators': 5000,
    'learning_rate': 0.02,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_lambda': 3,
    'reg_alpha': 0.1,
    'boosting_type': 'dart',
    'drop_rate': 0.1,
    'random_state': 42,
    'verbose': -1
}

v117_start = time.time()
oof_v117 = np.zeros(len(train_df))
test_v117_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v117), start=1):
    X_tr, X_val = X_train_v117.iloc[tr_idx], X_train_v117.iloc[val_idx]
    res_tr, res_val = residuals_lgb[tr_idx], residuals_lgb[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v117], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v117))])
    
    model = lgb.LGBMRegressor(**lgb_dart_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_v117[val_idx] = model.predict(X_val)
    test_v117_preds.append(model.predict(X_test_v117))
    
    if fold % 5 == 0:
        print(f"  Fold {fold}/{N_FOLDS} done")

final_oof_v117 = np.clip(v67_train + oof_v117, 0, 100)
final_test_v117 = np.clip(v67_test + np.mean(test_v117_preds, axis=0), 0, 100)
v117_rmse = np.sqrt(mean_squared_error(y, final_oof_v117))
v117_time = (time.time() - v117_start) / 60
print(f"V117 OOF RMSE: {v117_rmse:.5f} ({v117_time:.1f} min)")

# ============================================================================
# V118: LightGBM + Ridge Meta
# ============================================================================

print(f"\n{'='*80}")
print("V118: LightGBM + Ridge Meta")
print("="*80)

# Create Ridge for LightGBM features
ridge_lgb_oof = np.zeros(len(train_df))
ridge_lgb_test = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_v117), start=1):
    X_tr = train_v117.iloc[tr_idx][FEATURE_COLS_V117].values
    X_val = train_v117.iloc[val_idx][FEATURE_COLS_V117].values
    y_tr = y[tr_idx]
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_tr, y_tr)
    ridge_lgb_oof[val_idx] = ridge.predict(X_val)
    ridge_lgb_test.append(ridge.predict(test_v117[FEATURE_COLS_V117].values))

ridge_lgb_test_pred = np.mean(ridge_lgb_test, axis=0)

train_v118 = train_v117.copy()
train_v118['ridge_pred'] = ridge_lgb_oof
test_v118 = test_v117.copy()
test_v118['ridge_pred'] = ridge_lgb_test_pred
orig_v118 = orig_v117.copy()
orig_v118['ridge_pred'] = 0

FEATURE_COLS_V118 = FEATURE_COLS_V117 + ['ridge_pred']
X_train_v118 = train_v118[FEATURE_COLS_V118 + CATS]
X_test_v118 = test_v118[FEATURE_COLS_V118 + CATS]
X_orig_v118 = orig_v118[FEATURE_COLS_V118 + CATS]

print(f"Features: {len(FEATURE_COLS_V118 + CATS)}")

v118_start = time.time()
oof_v118 = np.zeros(len(train_df))
test_v118_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v118), start=1):
    X_tr, X_val = X_train_v118.iloc[tr_idx], X_train_v118.iloc[val_idx]
    res_tr, res_val = residuals_lgb[tr_idx], residuals_lgb[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v118], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v118))])
    
    model = lgb.LGBMRegressor(**lgb_dart_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_v118[val_idx] = model.predict(X_val)
    test_v118_preds.append(model.predict(X_test_v118))
    
    if fold % 5 == 0:
        print(f"  Fold {fold}/{N_FOLDS} done")

final_oof_v118 = np.clip(v67_train + oof_v118, 0, 100)
final_test_v118 = np.clip(v67_test + np.mean(test_v118_preds, axis=0), 0, 100)
v118_rmse = np.sqrt(mean_squared_error(y, final_oof_v118))
v118_time = (time.time() - v118_start) / 60
print(f"V118 OOF RMSE: {v118_rmse:.5f} ({v118_time:.1f} min)")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"""
| Version | Model | Strategy | OOF RMSE | Time |
|---------|-------|----------|----------|------|
| V101 | XGBoost base | Multi-KD | 8.55902 | - |
| **V114** | XGBoost | DART + Multi-KD | {v114_rmse:.5f} | {v114_time:.1f} min |
| **V115** | XGBoost | + Ridge meta | {v115_rmse:.5f} | {v115_time:.1f} min |
| **V116** | XGBoost | + Binned | {v116_rmse:.5f} | {v116_time:.1f} min |
| V67 | LightGBM base | Boosted PL | 8.59019 | - |
| **V117** | LightGBM | DART mode | {v117_rmse:.5f} | {v117_time:.1f} min |
| **V118** | LightGBM | + Ridge meta | {v118_rmse:.5f} | {v118_time:.1f} min |
""")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

for name, oof, test in [
    ('v114', final_oof_v114, final_test_v114),
    ('v115', final_oof_v115, final_test_v115),
    ('v116', final_oof_v116, final_test_v116),
    ('v117', final_oof_v117, final_test_v117),
    ('v118', final_oof_v118, final_test_v118)
]:
    pd.DataFrame({'id': test_df['id'], 'exam_score': test}).to_csv(f"submission_{name}.csv", index=False)
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof}).to_csv(f"oof_{name}.csv", index=False)
    print(f"  ✓ {name} saved")

elapsed = (time.time() - start_time) / 60
print(f"\nTotal time: {elapsed:.1f} minutes")
print("="*80)
