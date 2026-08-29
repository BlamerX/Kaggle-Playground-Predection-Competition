"""
S6E1 V123-V127 - Recursive Knowledge Distillation
==================================================
All models learn from ALL other models' OOFs.
Based on V110 architecture (best single model).

V123: CatBoost + V101,V105,V70,V67,V73,V122 KD
V124: XGBoost + V110,V105,V70,V67,V77,V122 KD
V125: TabM + V110,V101,V70,V67,V73,V122 KD
V126: LightGBM + V110,V101,V105,V70,V73,V122 KD
V127: FTT + V110,V101,V105,V67,V77,V122 KD

Target: 8.52 LB (Stage 1 of Phase 3)
"""

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import time
import os
import sys
import subprocess

warnings.filterwarnings("ignore")

# Install skorch if needed (required by pytabkit)
try:
    import skorch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skorch", "-q"])

# Try to import pytabkit (auto-install if needed)
HAVE_PYTABKIT = False
try:
    import torch
    from pytabkit import TabM_D_Regressor, FTT_D_Regressor
    HAVE_PYTABKIT = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
        import torch
        from pytabkit import TabM_D_Regressor, FTT_D_Regressor
        HAVE_PYTABKIT = True
    except:
        print("⚠️ pytabkit not available - V125/V127 will be skipped")

print("="*80)
print("S6E1 V123-V127 - Recursive Knowledge Distillation")
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
    except_pytabkit = False
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_path = "Previous trained files/"
    except_pytabkit = True

TARGET = "exam_score"
y = train_df[TARGET].values
n_samples = len(y)

# ============================================================================
# 2. LOAD ALL OOF FILES
# ============================================================================

print("\nLoading OOF files...")

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    oof_vals = oof[col].values
    sub_vals = sub['exam_score'].values
    rmse = np.sqrt(mean_squared_error(y, oof_vals))
    print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
    return oof_vals, sub_vals

# Load all required OOFs
oofs = {}
subs = {}

oofs['v110'], subs['v110'] = load_oof("V110 (CatBoost DART)", "oof_v110.csv", "submission_v110.csv")
oofs['v101'], subs['v101'] = load_oof("V101 (XGBoost)", "oof_v101.csv", "submission_v101.csv")
oofs['v105'], subs['v105'] = load_oof("V105 (TabM)", "oof_v105.csv", "submission_v105.csv")
oofs['v70'], subs['v70'] = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
oofs['v67'], subs['v67'] = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
oofs['v73'], subs['v73'] = load_oof("V73 (XGBoost Base)", "oof_v73.csv", "submission_v73.csv")
oofs['v77'], subs['v77'] = load_oof("V77 (CatBoost Base)", "oof_v77.csv", "submission_v77.csv")
oofs['v122'], subs['v122'] = load_oof("V122 (Best Ensemble)", "oof_v122.csv", "submission_v122.csv")
oofs['v61'], subs['v61'] = load_oof("V61 (TabM Base)", "oof_v61.csv", "submission_v61.csv")

# ============================================================================
# 3. CMT ENCODING (FROM V110)
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
# 4. FEATURE ENGINEERING (FROM V110)
# ============================================================================

print("\nPreparing features...")

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, kd_preds=None):
    """Feature engineering matching V110"""
    df = df.copy()
    eps = 1e-5
    
    # Squared features
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    
    # Log features
    sh_pos = df['study_hours'].clip(lower=0)
    ca_pos = df['class_attendance'].clip(lower=0)
    df['log_study_hours'] = np.log1p(sh_pos)
    df['log_class_attendance'] = np.log1p(ca_pos)
    
    # Interaction features
    df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_hours_times_sleep'] = df['study_hours'] * df['sleep_hours']
    df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    
    # Manual formula
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 0.35 * df['class_attendance'] + 1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    
    # High study flag
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    
    # Sinusoidal features
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    
    # KD predictions
    if kd_preds is not None:
        for name, pred in kd_preds.items():
            df[f'{name}_pred'] = pred
    
    return df

# V110 used these KD preds for base features
kd_train_base = {'tabm': oofs['v61'], 'ftt': oofs['v70'], 'lgb': oofs['v67'], 'xgb': oofs['v73']}
kd_test_base = {'tabm': subs['v61'], 'ftt': subs['v70'], 'lgb': subs['v67'], 'xgb': subs['v73']}

train_eng = add_features(train_df, kd_train_base)
test_eng = add_features(test_df, kd_test_base)
orig_eng = add_features(original_df, None)
for k in kd_train_base.keys():
    orig_eng[f'{k}_pred'] = 0

FEATURE_COLS = [c for c in train_eng.columns if c not in [TARGET, 'id', 'student_id']]
for col in CATS:
    train_eng[col] = train_eng[col].astype('category')
    test_eng[col] = test_eng[col].astype('category')
    orig_eng[col] = orig_eng[col].astype('category')

X_train = train_eng[FEATURE_COLS]
X_test = test_eng[FEATURE_COLS]
X_orig = orig_eng[FEATURE_COLS]

# Residual target (from V110)
residuals = y - oofs['v77']
cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in CATS]

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  V77 baseline RMSE: {np.sqrt(mean_squared_error(y, oofs['v77'])):.5f}")

# ============================================================================
# 5. V123: CATBOOST + EXTENDED KD (MATCHING V110 PARAMS)
# ============================================================================

def run_catboost_kd():
    print(f"\n{'='*80}")
    print("V123: CATBOOST + ALL KD (V101,V105,V70,V67,V73,V122)")
    print("="*80)
    
    # Add extra KD features for this version
    extra_kd_train = {'v101': oofs['v101'], 'v105': oofs['v105'], 'v122': oofs['v122']}
    extra_kd_test = {'v101': subs['v101'], 'v105': subs['v105'], 'v122': subs['v122']}
    
    X_tr_kd = X_train.copy()
    X_te_kd = X_test.copy()
    X_or_kd = X_orig.copy()
    
    for name, pred in extra_kd_train.items():
        X_tr_kd[f'{name}_pred'] = pred
        X_te_kd[f'{name}_pred'] = extra_kd_test[name]
        X_or_kd[f'{name}_pred'] = 0
    
    feature_cols_kd = X_tr_kd.columns.tolist()
    cat_idx_kd = [i for i, c in enumerate(feature_cols_kd) if c in CATS]
    
    start_time = time.time()
    N_FOLDS = 10
    SEED = 42
    
    # V110 DART params
    catboost_params = {
        'iterations': 5000,
        'learning_rate': 0.02,
        'depth': 6,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'task_type': 'GPU',
        'early_stopping_rounds': 150,
        'verbose': 0,
        'random_seed': SEED
    }
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_res = np.zeros(n_samples)
    test_res = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_kd), start=1):
        print(f"  Fold {fold}/{N_FOLDS}...", end=" ")
        fold_start = time.time()
        
        X_tr, X_val = X_tr_kd.iloc[tr_idx], X_tr_kd.iloc[val_idx]
        res_tr, res_val = residuals[tr_idx], residuals[val_idx]
        
        # Concat original data (from V110)
        X_comb = pd.concat([X_tr, X_or_kd], axis=0)
        res_comb = np.concatenate([res_tr, np.zeros(len(X_or_kd))])
        
        train_pool = Pool(X_comb, res_comb, cat_features=cat_idx_kd)
        val_pool = Pool(X_val, res_val, cat_features=cat_idx_kd)
        
        model = CatBoostRegressor(**catboost_params)
        model.fit(train_pool, eval_set=val_pool)
        
        oof_res[val_idx] = model.predict(X_val)
        test_res.append(model.predict(X_te_kd))
        
        fold_rmse = np.sqrt(mean_squared_error(res_val, oof_res[val_idx]))
        print(f"Res RMSE: {fold_rmse:.5f} ({time.time()-fold_start:.1f}s)")
    
    # Final predictions = V77 baseline + residuals
    oof_preds = oofs['v77'] + oof_res
    test_preds = subs['v77'] + np.mean(test_res, axis=0)
    
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n  V123 OOF RMSE: {final_rmse:.5f} (Time: {elapsed:.1f} min)")
    
    return oof_preds, test_preds, final_rmse

# ============================================================================
# 6. V124: XGBOOST + ALL KD
# ============================================================================

def run_xgboost_kd():
    print(f"\n{'='*80}")
    print("V124: XGBOOST + ALL KD (V110,V105,V70,V67,V77,V122)")
    print("="*80)
    
    # Add extra KD features
    extra_kd_train = {'v110': oofs['v110'], 'v105': oofs['v105'], 'v122': oofs['v122'], 'v77': oofs['v77']}
    extra_kd_test = {'v110': subs['v110'], 'v105': subs['v105'], 'v122': subs['v122'], 'v77': subs['v77']}
    
    X_tr_kd = X_train.copy()
    X_te_kd = X_test.copy()
    
    for name, pred in extra_kd_train.items():
        X_tr_kd[f'{name}_kd'] = pred
        X_te_kd[f'{name}_kd'] = extra_kd_test[name]
    
    # Convert categories to numeric for XGBoost
    for col in CATS:
        X_tr_kd[col] = X_tr_kd[col].cat.codes
        X_te_kd[col] = X_te_kd[col].cat.codes
    
    start_time = time.time()
    N_FOLDS = 10
    SEED = 42
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(n_samples)
    test_preds = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_kd), start=1):
        print(f"  Fold {fold}/{N_FOLDS}...", end=" ")
        fold_start = time.time()
        
        X_tr, X_val = X_tr_kd.iloc[tr_idx], X_tr_kd.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.5,
            reg_alpha=0.1,
            reg_lambda=5,
            min_child_weight=5,
            random_state=SEED,
            tree_method='hist',
            device='cuda',
            early_stopping_rounds=50,
            eval_metric='rmse',
            verbosity=0
        )
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_te_kd) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"RMSE: {fold_rmse:.5f} ({time.time()-fold_start:.1f}s)")
    
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n  V124 OOF RMSE: {final_rmse:.5f} (Time: {elapsed:.1f} min)")
    
    return oof_preds, test_preds, final_rmse

# ============================================================================
# 7. V125: TABM + ALL KD
# ============================================================================

def run_tabm_kd():
    print(f"\n{'='*80}")
    print("V125: TABM + ALL KD (V110,V101,V70,V67,V73,V122)")
    print("="*80)
    
    if not HAVE_PYTABKIT:
        print("  SKIPPED: pytabkit not available")
        return None, None, None
    
    # Add extra KD features
    extra_kd_train = {'v110': oofs['v110'], 'v101': oofs['v101'], 'v122': oofs['v122']}
    extra_kd_test = {'v110': subs['v110'], 'v101': subs['v101'], 'v122': subs['v122']}
    
    X_tr_kd = X_train.copy()
    X_te_kd = X_test.copy()
    
    for name, pred in extra_kd_train.items():
        X_tr_kd[f'{name}_kd'] = pred
        X_te_kd[f'{name}_kd'] = extra_kd_test[name]
    
    # Convert categoricals to numeric for TabM
    for col in CATS:
        X_tr_kd[col] = X_tr_kd[col].cat.codes
        X_te_kd[col] = X_te_kd[col].cat.codes
    
    # Standardize features
    scaler = StandardScaler()
    NUMS = [c for c in X_tr_kd.columns if c not in CATS]
    X_tr_kd[NUMS] = scaler.fit_transform(X_tr_kd[NUMS])
    X_te_kd[NUMS] = scaler.transform(X_te_kd[NUMS])
    
    start_time = time.time()
    N_FOLDS = 5  # TabM is slow, use 5 folds
    SEED = 42
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(n_samples)
    test_preds = np.zeros(len(X_test))
    
    tabm_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 100,
        'batch_size': 256,
        'n_blocks': 5,
        'patience': 4,
        'weight_decay': 1e-2,
        'random_state': SEED,
        'verbosity': 0
    }
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_kd), start=1):
        print(f"  Fold {fold}/{N_FOLDS}...", end=" ")
        fold_start = time.time()
        
        X_tr, X_val = X_tr_kd.iloc[tr_idx], X_tr_kd.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = TabM_D_Regressor(**tabm_params)
        model.fit(X_tr, y_tr, X_val, y_val)
        
        oof_preds[val_idx] = model.predict(X_val).flatten()
        test_preds += model.predict(X_te_kd).flatten() / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"RMSE: {fold_rmse:.5f} ({time.time()-fold_start:.1f}s)")
    
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n  V125 OOF RMSE: {final_rmse:.5f} (Time: {elapsed:.1f} min)")
    
    return oof_preds, test_preds, final_rmse

# ============================================================================
# 8. V126: LIGHTGBM + ALL KD
# ============================================================================

def run_lightgbm_kd():
    print(f"\n{'='*80}")
    print("V126: LIGHTGBM + ALL KD (V110,V101,V105,V70,V73,V122)")
    print("="*80)
    
    # Add extra KD features
    extra_kd_train = {'v110': oofs['v110'], 'v101': oofs['v101'], 'v105': oofs['v105'], 'v122': oofs['v122']}
    extra_kd_test = {'v110': subs['v110'], 'v101': subs['v101'], 'v105': subs['v105'], 'v122': subs['v122']}
    
    X_tr_kd = X_train.copy()
    X_te_kd = X_test.copy()
    
    for name, pred in extra_kd_train.items():
        X_tr_kd[f'{name}_kd'] = pred
        X_te_kd[f'{name}_kd'] = extra_kd_test[name]
    
    # Convert categories to numeric for LightGBM
    for col in CATS:
        X_tr_kd[col] = X_tr_kd[col].cat.codes
        X_te_kd[col] = X_te_kd[col].cat.codes
    
    start_time = time.time()
    N_FOLDS = 10
    SEED = 42
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(n_samples)
    test_preds = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_kd), start=1):
        print(f"  Fold {fold}/{N_FOLDS}...", end=" ")
        fold_start = time.time()
        
        X_tr, X_val = X_tr_kd.iloc[tr_idx], X_tr_kd.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=20000,
            learning_rate=0.015,
            num_leaves=128,
            max_depth=12,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.01,
            reg_lambda=1.0,
            min_data_in_leaf=50,
            random_state=SEED,
            device='cpu',  # V67 uses CPU
            verbose=-1,
            n_jobs=-1
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_te_kd) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"RMSE: {fold_rmse:.5f} ({time.time()-fold_start:.1f}s)")
    
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n  V126 OOF RMSE: {final_rmse:.5f} (Time: {elapsed:.1f} min)")
    
    return oof_preds, test_preds, final_rmse

# ============================================================================
# 9. V127: FTT + ALL KD
# ============================================================================

def run_ftt_kd():
    print(f"\n{'='*80}")
    print("V127: FTT + ALL KD (V110,V101,V105,V67,V77,V122)")
    print("="*80)
    
    if not HAVE_PYTABKIT:
        print("  SKIPPED: pytabkit not available")
        return None, None, None
    
    # Add extra KD features
    extra_kd_train = {'v110': oofs['v110'], 'v101': oofs['v101'], 'v105': oofs['v105'], 'v122': oofs['v122']}
    extra_kd_test = {'v110': subs['v110'], 'v101': subs['v101'], 'v105': subs['v105'], 'v122': subs['v122']}
    
    X_tr_kd = X_train.copy()
    X_te_kd = X_test.copy()
    
    for name, pred in extra_kd_train.items():
        X_tr_kd[f'{name}_kd'] = pred
        X_te_kd[f'{name}_kd'] = extra_kd_test[name]
    
    # Convert categoricals to numeric for FTT
    for col in CATS:
        X_tr_kd[col] = X_tr_kd[col].cat.codes
        X_te_kd[col] = X_te_kd[col].cat.codes
    
    # Standardize features
    scaler = StandardScaler()
    NUMS = [c for c in X_tr_kd.columns if c not in CATS]
    X_tr_kd[NUMS] = scaler.fit_transform(X_tr_kd[NUMS])
    X_te_kd[NUMS] = scaler.transform(X_te_kd[NUMS])
    
    start_time = time.time()
    N_FOLDS = 5  # FTT is slow, use 5 folds
    SEED = 42
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(n_samples)
    test_preds = np.zeros(len(X_test))
    
    ftt_params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 256,
        'random_state': SEED,
        'verbosity': 0
    }
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_kd), start=1):
        print(f"  Fold {fold}/{N_FOLDS}...", end=" ")
        fold_start = time.time()
        
        X_tr, X_val = X_tr_kd.iloc[tr_idx], X_tr_kd.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = FTT_D_Regressor(**ftt_params)
        model.fit(X_tr, y_tr, X_val, y_val)
        
        oof_preds[val_idx] = model.predict(X_val).flatten()
        test_preds += model.predict(X_te_kd).flatten() / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"RMSE: {fold_rmse:.5f} ({time.time()-fold_start:.1f}s)")
    
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n  V127 OOF RMSE: {final_rmse:.5f} (Time: {elapsed:.1f} min)")
    
    return oof_preds, test_preds, final_rmse

# ============================================================================
# 10. RUN ALL MODELS
# ============================================================================

results = {}

# V123: CatBoost
oof_v123, test_v123, rmse_v123 = run_catboost_kd()
if oof_v123 is not None:
    results['V123'] = {'oof': oof_v123, 'test': test_v123, 'rmse': rmse_v123}
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v123}).to_csv("oof_v123.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': test_v123}).to_csv("submission_v123.csv", index=False)

# V124: XGBoost
oof_v124, test_v124, rmse_v124 = run_xgboost_kd()
if oof_v124 is not None:
    results['V124'] = {'oof': oof_v124, 'test': test_v124, 'rmse': rmse_v124}
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v124}).to_csv("oof_v124.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': test_v124}).to_csv("submission_v124.csv", index=False)

# V125: TabM
oof_v125, test_v125, rmse_v125 = run_tabm_kd()
if oof_v125 is not None:
    results['V125'] = {'oof': oof_v125, 'test': test_v125, 'rmse': rmse_v125}
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v125}).to_csv("oof_v125.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': test_v125}).to_csv("submission_v125.csv", index=False)

# V126: LightGBM
oof_v126, test_v126, rmse_v126 = run_lightgbm_kd()
if oof_v126 is not None:
    results['V126'] = {'oof': oof_v126, 'test': test_v126, 'rmse': rmse_v126}
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v126}).to_csv("oof_v126.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': test_v126}).to_csv("submission_v126.csv", index=False)

# V127: FTT
oof_v127, test_v127, rmse_v127 = run_ftt_kd()
if oof_v127 is not None:
    results['V127'] = {'oof': oof_v127, 'test': test_v127, 'rmse': rmse_v127}
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v127}).to_csv("oof_v127.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': test_v127}).to_csv("submission_v127.csv", index=False)

# ============================================================================
# 11. SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print("""
| Version | Model | OOF RMSE | Status |
|---------|-------|----------|--------|""")

for name, data in results.items():
    print(f"| {name} | - | {data['rmse']:.5f} | ✅ |")

print(f"""
Reference:
  V110 (Best Single): 8.55927 OOF → 8.54708 LB
  V122 (Best Ensemble): 8.55763 OOF → 8.54693 LB

Next Steps:
  1. Submit each V123-V127 to get LB scores
  2. Run V128 (Ridge Stack on V123-V127)
  3. Run V129 (LightGBM Stack on V123-V127 + V122)
  4. Run V130 (Final HillClimber) → Target 8.52
""")

print("="*80)
