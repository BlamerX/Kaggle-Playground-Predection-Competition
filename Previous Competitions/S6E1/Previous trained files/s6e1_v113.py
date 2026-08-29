"""
S6E1 V113 - TabM + Extended KD (Add V110)
==========================================
V105: TabM + Multi-KD → 8.54963 LB (best TabM)
V113: V105 + V110 (best CatBoost) prediction as additional KD

Base: V105 (8.54963 LB)
Goal: Add V110 prediction to improve TabM
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
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
print("S6E1 V113 - TabM + Extended KD (Add V110)")
print("="*80)

# Auto-install pytabkit
try:
    from pytabkit import TabM_D_Regressor
    HAVE_PYTABKIT = True
except ImportError:
    print("Installing pytabkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    try:
        from pytabkit import TabM_D_Regressor
        HAVE_PYTABKIT = True
    except ImportError:
        HAVE_PYTABKIT = False
        print("Warning: pytabkit not available")

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

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    print(f"  ✓ {name} loaded")
    return oof[col].values, sub['exam_score'].values

print("\nLoading OOF files...")

# V105 baseline (V61)
v61_train, v61_test = load_oof("V61 (TabM baseline)", "oof_v61.csv", "submission_v61.csv")

# V105 original KD features
v70_train, v70_test = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
v67_train, v67_test = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
v73_train, v73_test = load_oof("V73 (XGBoost)", "oof_v73.csv", "submission_v73.csv")

# NEW: Add V110 (best CatBoost) as additional KD
v110_train, v110_test = load_oof("V110 (CatBoost DART 5-seed)", "oof_v110.csv", "submission_v110.csv")

print(f"\nV61 baseline RMSE: {np.sqrt(mean_squared_error(y, v61_train)):.5f}")

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
    if kd_preds is not None:
        for name, pred in kd_preds.items():
            df[f'{name}_pred'] = pred
    return df

# Extended KD: V105 features + V110
kd_train = {
    'ftt': v70_train, 'lgb': v67_train, 'xgb': v73_train,
    'v110': v110_train  # NEW: best CatBoost
}
kd_test = {
    'ftt': v70_test, 'lgb': v67_test, 'xgb': v73_test,
    'v110': v110_test  # NEW
}

train_eng = add_features(train_df, kd_train)
test_eng = add_features(test_df, kd_test)
orig_eng = add_features(original_df, None)
for k in kd_train.keys():
    orig_eng[f'{k}_pred'] = 0

FEATURE_COLS = [c for c in train_eng.columns if c not in [TARGET, 'id', 'student_id'] + CATS]
print(f"Features: {len(FEATURE_COLS)} (V105 had 35, +1 V110)")

# ============================================================================
# V113: TABM + EXTENDED KD
# ============================================================================

if HAVE_PYTABKIT:
    print(f"\n{'='*80}")
    print("V113: TABM + EXTENDED KD (V110)")
    print("="*80)
    
    # TabM preprocessing
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat = ord_enc.fit_transform(train_eng[CATS])
    test_cat = ord_enc.transform(test_eng[CATS])
    orig_cat = ord_enc.transform(orig_eng[CATS])
    
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_eng[FEATURE_COLS])
    test_num = scaler.transform(test_eng[FEATURE_COLS])
    orig_num = scaler.transform(orig_eng[FEATURE_COLS])
    
    X_train_full = np.hstack([train_num, train_cat])
    X_test_full = np.hstack([test_num, test_cat])
    X_orig_full = np.hstack([orig_num, orig_cat])
    
    residuals = y - v61_train
    
    N_FOLDS = 10
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)
    ALPHA = 0.1
    
    tabm_params = {
        'n_epochs': 100,
        'patience': 15,
        'batch_size': 256,
        'device': 'cuda',
        'verbosity': 0
    }
    
    # Phase 1
    print("\n--- Phase 1: Training on residuals ---")
    oof_res = np.zeros(len(train_df))
    test_res = []
    phase1_start = time.time()
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        fold_start = time.time()
        X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
        res_tr, res_val = residuals[tr_idx], residuals[val_idx]
        
        X_comb = np.vstack([X_tr, X_orig_full])
        res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_full))])
        
        print(f"  Fold {fold}/{N_FOLDS}: Training TabM on {len(X_comb)} samples...", end=" ", flush=True)
        model = TabM_D_Regressor(**tabm_params)
        model.fit(X_comb, res_comb, X_val, res_val)
        
        oof_res[val_idx] = model.predict(X_val)
        test_res.append(model.predict(X_test_full))
        
        fold_time = (time.time() - fold_start) / 60
        val_rmse = np.sqrt(mean_squared_error(res_val, oof_res[val_idx]))
        print(f"done in {fold_time:.1f} min, val_rmse={val_rmse:.5f}")
    
    phase1_time = (time.time() - phase1_start) / 60
    print(f"\nPhase 1 complete: {phase1_time:.1f} min total")
    
    # Phase 2: Boosted PL
    print("\n--- Phase 2: Boosted Pseudo-Labeling ---")
    test_pseudo = np.clip(v61_test + np.mean(test_res, axis=0) + ALPHA * oof_res.mean(), 0, 100)
    test_pseudo_res = test_pseudo - v61_test
    
    oof_final = np.zeros(len(train_df))
    test_final = []
    phase2_start = time.time()
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):
        fold_start = time.time()
        X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
        res_tr, res_val = residuals[tr_idx], residuals[val_idx]
        
        X_comb = np.vstack([X_tr, X_orig_full, X_test_full])
        res_comb = np.concatenate([res_tr, np.zeros(len(X_orig_full)), test_pseudo_res])
        
        print(f"  Fold {fold}/{N_FOLDS}: Training TabM on {len(X_comb)} samples (w/ PL)...", end=" ", flush=True)
        model = TabM_D_Regressor(**tabm_params)
        model.fit(X_comb, res_comb, X_val, res_val)
        
        oof_final[val_idx] = model.predict(X_val)
        test_final.append(model.predict(X_test_full))
        
        fold_time = (time.time() - fold_start) / 60
        val_rmse = np.sqrt(mean_squared_error(res_val, oof_final[val_idx]))
        print(f"done in {fold_time:.1f} min, val_rmse={val_rmse:.5f}")
    
    phase2_time = (time.time() - phase2_start) / 60
    print(f"\nPhase 2 complete: {phase2_time:.1f} min total")
    
    final_oof = np.clip(v61_train + oof_final, 0, 100)
    final_test = np.clip(v61_test + np.mean(test_final, axis=0), 0, 100)
    v113_rmse = np.sqrt(mean_squared_error(y, final_oof))
    
    # ============================================================================
    # RESULTS
    # ============================================================================
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"""
| Version | Model | OOF RMSE | LB Score |
|---------|-------|----------|----------|
| V105 | TabM + V61 + 4-KD | 8.56382 | 8.54963 |
| **V113** | **TabM + V61 + 5-KD (V110)** | **{v113_rmse:.5f}** | **?** |
""")
    
    # ============================================================================
    # SAVE
    # ============================================================================
    
    pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v113.csv", index=False)
    pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v113.csv", index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved: submission_v113.csv, oof_v113.csv")
    print(f"Total time: {elapsed:.1f} minutes")
    print("="*80)
else:
    print("ERROR: pytabkit not available. Cannot run V113.")
