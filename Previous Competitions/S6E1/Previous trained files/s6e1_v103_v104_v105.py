"""
S6E1 V103-V106 - Multi-Model Single Models with Best Baselines
================================================================
Replicating V101 success with different model architectures.

V103: CatBoost + V77 baseline + Multi-KD (TabM+FTT+LGB+XGB)
V104: LightGBM + V67 baseline + Multi-KD (TabM+FTT+XGB)
V105: TabM + V61 baseline + Multi-KD (XGB+FTT+LGB)
V106: FTT + V70 baseline + Multi-KD (TabM+XGB+LGB)

Each uses best OOF for that model type + Discussion FE + KD features.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import os
import time
import sys
import subprocess

# Try to import DL models (auto-install if needed)
try:
    import skorch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skorch", "-q"])

try:
    import torch
    from pytabkit import TabM_D_Regressor, FTT_D_Regressor
    HAVE_PYTABKIT = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
        from pytabkit import TabM_D_Regressor, FTT_D_Regressor
        HAVE_PYTABKIT = True
    except:
        HAVE_PYTABKIT = False
        print("⚠️ pytabkit not available - V105 will be skipped")

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V103-V106 - Multi-Model with Best Baselines")
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

# Load ALL OOF files needed
print("\nLoading OOF files...")

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    print(f"  ✓ {name} loaded")
    return oof[col].values, sub['exam_score'].values

# Baselines for each model type
v77_train, v77_test = load_oof("V77 (CatBoost best)", "oof_v77.csv", "submission_v77.csv")
v67_train, v67_test = load_oof("V67 (LightGBM best)", "oof_v67.csv", "submission_v67.csv")
v61_train, v61_test = load_oof("V61 (TabM best)", "oof_v61.csv", "submission_v61.csv")
v70_train, v70_test = load_oof("V70 (FTT best)", "oof_v70.csv", "submission_v70.csv")

# KD features (other model predictions)
v73_train, v73_test = load_oof("V73 (XGBoost)", "oof_v73.csv", "submission_v73.csv")

print(f"\nBaseline RMSEs:")
print(f"  V77 (CatBoost): {np.sqrt(mean_squared_error(y, v77_train)):.5f}")
print(f"  V67 (LightGBM): {np.sqrt(mean_squared_error(y, v67_train)):.5f}")
print(f"  V61 (TabM):     {np.sqrt(mean_squared_error(y, v61_train)):.5f}")
print(f"  V70 (FTT):      {np.sqrt(mean_squared_error(y, v70_train)):.5f}")

# ============================================================================
# 2. CMT ENCODING (same as V77)
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
# 3. FEATURE ENGINEERING (V77 style + Discussion FE + KD)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING")
print("="*80)

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, cmt_cols, kd_preds=None):
    """Feature engineering with optional KD predictions."""
    df = df.copy()
    eps = 1e-5
    
    # Basic features (same as V77)
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
    
    # KD predictions as features
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

# ============================================================================
# V103: CATBOOST + V77 BASELINE + MULTI-KD
# ============================================================================

print(f"\n{'='*80}")
print("V103: CATBOOST + V77 BASELINE + MULTI-KD")
print("="*80)

# Add KD features for CatBoost (all other models)
kd_v103 = {'tabm': v61_train, 'ftt': v70_train, 'lgb': v67_train, 'xgb': v73_train}
kd_v103_test = {'tabm': v61_test, 'ftt': v70_test, 'lgb': v67_test, 'xgb': v73_test}

train_v103 = add_features(train_df, cmt_cols, kd_v103)
test_v103 = add_features(test_df, cmt_cols, kd_v103_test)
orig_v103 = add_features(original_df, cmt_cols, None)
for k in kd_v103.keys():
    orig_v103[f'{k}_pred'] = 0

FEATURE_COLS = [c for c in train_v103.columns if c not in [TARGET, 'id', 'student_id']]

for col in CATS:
    train_v103[col] = train_v103[col].astype('category')
    test_v103[col] = test_v103[col].astype('category')
    orig_v103[col] = orig_v103[col].astype('category')

X_train = train_v103[FEATURE_COLS]
X_test = test_v103[FEATURE_COLS]
X_orig = orig_v103[FEATURE_COLS]

# Residuals from V77 baseline
residuals_v103 = y - v77_train

print(f"Features: {len(FEATURE_COLS)}")
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

# Phase 1: Train on residuals
oof_res_v103 = np.zeros(len(train_df))
test_res_v103 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals_v103[tr_idx], residuals_v103[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig))])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=val_pool)
    
    oof_res_v103[val_idx] = model.predict(X_val)
    test_res_v103.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Phase 2: Boosted PL
ALPHA = 0.1
test_pseudo = np.clip(v77_test + np.mean(test_res_v103, axis=0) + ALPHA * oof_res_v103.mean(), 0, 100)
test_pseudo_res = test_pseudo - v77_test

oof_final_v103 = np.zeros(len(train_df))
test_final_v103 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals_v103[tr_idx], residuals_v103[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig, X_test], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig)), test_pseudo_res])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_pool, eval_set=val_pool)
    
    oof_final_v103[val_idx] = model.predict(X_val)
    test_final_v103.append(model.predict(X_test))

final_oof_v103 = np.clip(v77_train + oof_final_v103, 0, 100)
final_test_v103 = np.clip(v77_test + np.mean(test_final_v103, axis=0), 0, 100)

v103_rmse = np.sqrt(mean_squared_error(y, final_oof_v103))
print(f"\nV103 OOF RMSE: {v103_rmse:.5f}")

# ============================================================================
# V104: LIGHTGBM + V67 BASELINE + MULTI-KD
# ============================================================================

print(f"\n{'='*80}")
print("V104: LIGHTGBM + V67 BASELINE + MULTI-KD")
print("="*80)

# Add KD features for LightGBM
kd_v104 = {'tabm': v61_train, 'ftt': v70_train, 'xgb': v73_train, 'catboost': v77_train}
kd_v104_test = {'tabm': v61_test, 'ftt': v70_test, 'xgb': v73_test, 'catboost': v77_test}

train_v104 = add_features(train_df, cmt_cols, kd_v104)
test_v104 = add_features(test_df, cmt_cols, kd_v104_test)
orig_v104 = add_features(original_df, cmt_cols, None)
for k in kd_v104.keys():
    orig_v104[f'{k}_pred'] = 0

FEATURE_COLS_V104 = [c for c in train_v104.columns if c not in [TARGET, 'id', 'student_id']]

for col in CATS:
    train_v104[col] = train_v104[col].astype('category')
    test_v104[col] = test_v104[col].astype('category')
    orig_v104[col] = orig_v104[col].astype('category')

X_train_v104 = train_v104[FEATURE_COLS_V104]
X_test_v104 = test_v104[FEATURE_COLS_V104]
X_orig_v104 = orig_v104[FEATURE_COLS_V104]

residuals_v104 = y - v67_train

print(f"Features: {len(FEATURE_COLS_V104)}")
print(f"V67 baseline RMSE: {np.sqrt(mean_squared_error(y, v67_train)):.5f}")

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 5000,
    'max_depth': 8,
    'num_leaves': 64,
    'learning_rate': 0.02,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.1,
    'reg_lambda': 5.0,
    'min_data_in_leaf': 50,
    'device': 'cpu',
    'verbose': -1,
    'random_state': 1003
}

# Phase 1
oof_res_v104 = np.zeros(len(train_df))
test_res_v104 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v104), start=1):
    X_tr, X_val = X_train_v104.iloc[tr_idx], X_train_v104.iloc[val_idx]
    res_tr, res_val = residuals_v104[tr_idx], residuals_v104[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v104], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v104))])
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_res_v104[val_idx] = model.predict(X_val)
    test_res_v104.append(model.predict(X_test_v104))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Phase 2: Boosted PL
test_pseudo_v104 = np.clip(v67_test + np.mean(test_res_v104, axis=0) + ALPHA * oof_res_v104.mean(), 0, 100)
test_pseudo_res_v104 = test_pseudo_v104 - v67_test

oof_final_v104 = np.zeros(len(train_df))
test_final_v104 = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v104), start=1):
    X_tr, X_val = X_train_v104.iloc[tr_idx], X_train_v104.iloc[val_idx]
    res_tr, res_val = residuals_v104[tr_idx], residuals_v104[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig_v104, X_test_v104], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v104)), test_pseudo_res_v104])
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_comb, res_comb, eval_set=[(X_val, res_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    
    oof_final_v104[val_idx] = model.predict(X_val)
    test_final_v104.append(model.predict(X_test_v104))

final_oof_v104 = np.clip(v67_train + oof_final_v104, 0, 100)
final_test_v104 = np.clip(v67_test + np.mean(test_final_v104, axis=0), 0, 100)

v104_rmse = np.sqrt(mean_squared_error(y, final_oof_v104))
print(f"\nV104 OOF RMSE: {v104_rmse:.5f}")

# ============================================================================
# V105: TABM + V61 BASELINE + MULTI-KD
# ============================================================================

v105_rmse = None
final_oof_v105 = None
final_test_v105 = None

if HAVE_PYTABKIT:
    print(f"\n{'='*80}")
    print("V105: TABM + V61 BASELINE + MULTI-KD")
    print("="*80)
    
    # Add KD features for TabM (other model predictions)
    kd_v105 = {'xgb': v73_train, 'ftt': v70_train, 'lgb': v67_train, 'catboost': v77_train}
    kd_v105_test = {'xgb': v73_test, 'ftt': v70_test, 'lgb': v67_test, 'catboost': v77_test}
    
    train_v105 = add_features(train_df, cmt_cols, kd_v105)
    test_v105 = add_features(test_df, cmt_cols, kd_v105_test)
    orig_v105 = add_features(original_df, cmt_cols, None)
    for k in kd_v105.keys():
        orig_v105[f'{k}_pred'] = 0
    
    # Prepare data for TabM (numerical only)
    FEATURE_COLS_V105 = [c for c in train_v105.columns if c not in [TARGET, 'id', 'student_id']]
    
    # Encode categoricals for TabM
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    scaler = StandardScaler()
    
    NUMS = [c for c in FEATURE_COLS_V105 if c not in CATS]
    
    encoder.fit(train_v105[CATS])
    scaler.fit(train_v105[NUMS])
    
    def preprocess_tabm(df):
        df = df.copy()
        df[CATS] = encoder.transform(df[CATS])
        df[NUMS] = scaler.transform(df[NUMS])
        return df[FEATURE_COLS_V105]
    
    X_train_v105 = preprocess_tabm(train_v105)
    X_test_v105 = preprocess_tabm(test_v105)
    X_orig_v105 = preprocess_tabm(orig_v105)
    
    residuals_v105 = y - v61_train
    
    print(f"Features: {len(FEATURE_COLS_V105)}")
    print(f"V61 baseline RMSE: {np.sqrt(mean_squared_error(y, v61_train)):.5f}")
    
    tabm_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 100,
        'batch_size': 256,
        'n_blocks': 5,
        'patience': 4,
        'weight_decay': 1e-2,
        'random_state': 42,
        'verbosity': 0
    }
    
    # Phase 1
    oof_res_v105 = np.zeros(len(train_df))
    test_res_v105 = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v105), start=1):
        X_tr, X_val = X_train_v105.iloc[tr_idx], X_train_v105.iloc[val_idx]
        res_tr, res_val = residuals_v105[tr_idx], residuals_v105[val_idx]
        
        X_comb = pd.concat([X_tr, X_orig_v105], axis=0)
        res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v105))])
        
        model = TabM_D_Regressor(**tabm_params)
        model.fit(X_comb, res_comb, X_val, res_val)
        
        oof_res_v105[val_idx] = model.predict(X_val).flatten()
        test_res_v105.append(model.predict(X_test_v105).flatten())
        
        if fold % 5 == 0:
            print(f"  Fold {fold} done")
    
    # Phase 2: Boosted PL
    test_pseudo_v105 = np.clip(v61_test + np.mean(test_res_v105, axis=0) + ALPHA * oof_res_v105.mean(), 0, 100)
    test_pseudo_res_v105 = test_pseudo_v105 - v61_test
    
    oof_final_v105 = np.zeros(len(train_df))
    test_final_v105 = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_v105), start=1):
        X_tr, X_val = X_train_v105.iloc[tr_idx], X_train_v105.iloc[val_idx]
        res_tr, res_val = residuals_v105[tr_idx], residuals_v105[val_idx]
        
        X_comb = pd.concat([X_tr, X_orig_v105, X_test_v105], axis=0)
        res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_v105)), test_pseudo_res_v105])
        
        model = TabM_D_Regressor(**tabm_params)
        model.fit(X_comb, res_comb, X_val, res_val)
        
        oof_final_v105[val_idx] = model.predict(X_val).flatten()
        test_final_v105.append(model.predict(X_test_v105).flatten())
    
    final_oof_v105 = np.clip(v61_train + oof_final_v105, 0, 100)
    final_test_v105 = np.clip(v61_test + np.mean(test_final_v105, axis=0), 0, 100)
    
    v105_rmse = np.sqrt(mean_squared_error(y, final_oof_v105))
    print(f"\nV105 OOF RMSE: {v105_rmse:.5f}")
else:
    print("\n⚠️ V105 skipped - pytabkit not available")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

v101_rmse = 8.55902
v101_lb = 8.54860

print(f"""
| Version | Model | Baseline | OOF RMSE | vs V101 | LB Score |
|---------|-------|----------|----------|---------|----------|
| V101 | XGBoost | V73 | {v101_rmse:.5f} | - | {v101_lb:.5f} |
| V103 | CatBoost | V77 | {v103_rmse:.5f} | {v101_rmse - v103_rmse:+.5f} | ? |
| V104 | LightGBM | V67 | {v104_rmse:.5f} | {v101_rmse - v104_rmse:+.5f} | ? |""")

if v105_rmse is not None:
    print(f"| V105 | TabM | V61 | {v105_rmse:.5f} | {v101_rmse - v105_rmse:+.5f} | ? |")

results = [
    ('V103', v103_rmse, final_oof_v103, final_test_v103),
    ('V104', v104_rmse, final_oof_v104, final_test_v104)
]
if v105_rmse is not None:
    results.append(('V105', v105_rmse, final_oof_v105, final_test_v105))

best = min(results, key=lambda x: x[1])
print(f"\n✅ Best: {best[0]} with OOF RMSE {best[1]:.5f}")

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

print(f"\nFiles saved: {', '.join([r[0] for r in results])}")
print(f"Total time: {elapsed:.1f} minutes")
print("\n⚠️ V106 (FTT) is in separate file: s6e1_v106.py")
print("="*80)
