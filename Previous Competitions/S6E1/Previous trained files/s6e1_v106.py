"""
S6E1 V106 - FT-Transformer + V70 Baseline + Multi-KD
======================================================
FTT training takes ~60+ minutes, so this is a separate file.

V106: FTT + V70 baseline + Multi-KD (TabM+XGB+LGB+CatBoost predictions)
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
import numpy as np
import warnings
import os
import time
import sys
import subprocess

# Auto-install dependencies if needed
try:
    import skorch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skorch", "-q"])

try:
    import torch
    from pytabkit import FTT_D_Regressor
    HAVE_FTT = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
        import torch
        from pytabkit import FTT_D_Regressor
        HAVE_FTT = True
    except:
        HAVE_FTT = False
        print("❌ ERROR: pytabkit required for V106. Install with: pip install pytabkit")
        exit(1)

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V106 - FT-Transformer + V70 Baseline + Multi-KD")
print("="*80)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

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

# Load OOF files
print("\nLoading OOF files...")

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    print(f"  ✓ {name} loaded")
    return oof[col].values, sub['exam_score'].values

# V70 baseline (best FTT)
v70_train, v70_test = load_oof("V70 (FTT best)", "oof_v70.csv", "submission_v70.csv")

# KD features
v61_train, v61_test = load_oof("V61 (TabM)", "oof_v61.csv", "submission_v61.csv")
v73_train, v73_test = load_oof("V73 (XGBoost)", "oof_v73.csv", "submission_v73.csv")
v67_train, v67_test = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
v77_train, v77_test = load_oof("V77 (CatBoost)", "oof_v77.csv", "submission_v77.csv")

print(f"\nV70 baseline RMSE: {np.sqrt(mean_squared_error(y, v70_train)):.5f}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING")
print("="*80)

CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty']

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

# Add KD features
kd_train = {'tabm': v61_train, 'xgb': v73_train, 'lgb': v67_train, 'catboost': v77_train}
kd_test = {'tabm': v61_test, 'xgb': v73_test, 'lgb': v67_test, 'catboost': v77_test}

train_eng = add_features(train_df, kd_train)
test_eng = add_features(test_df, kd_test)
orig_eng = add_features(original_df, None)
for k in kd_train.keys():
    orig_eng[f'{k}_pred'] = 0

FEATURE_COLS = [c for c in train_eng.columns if c not in [TARGET, 'id', 'student_id']]
NUMS = [c for c in FEATURE_COLS if c not in CATS]

# Preprocessing
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()

encoder.fit(train_eng[CATS])
scaler.fit(train_eng[NUMS])

def preprocess(df):
    df = df.copy()
    df[CATS] = encoder.transform(df[CATS])
    df[NUMS] = scaler.transform(df[NUMS])
    return df[FEATURE_COLS]

X_train = preprocess(train_eng)
X_test = preprocess(test_eng)
X_orig = preprocess(orig_eng)

print(f"Features: {len(FEATURE_COLS)}")

# Residuals from V70 baseline
residuals = y - v70_train

# ============================================================================
# 3. FTT TRAINING
# ============================================================================

print(f"\n{'='*80}")
print("V106: FTT + V70 BASELINE + MULTI-KD")
print("="*80)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)
ALPHA = 0.1

ftt_params = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'random_state': 42,
    'verbosity': 0,
    'batch_size': 256,
}

# Phase 1: Train on residuals
print("\nPhase 1: Training on residuals...")
oof_res = np.zeros(len(train_df))
test_res = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig))])
    
    model = FTT_D_Regressor(**ftt_params)
    model.fit(X_comb, res_comb, X_val, res_val)
    
    oof_res[val_idx] = model.predict(X_val).flatten()
    test_res.append(model.predict(X_test).flatten())
    
    print(f"  Fold {fold}/10 done")

phase1_rmse = np.sqrt(mean_squared_error(y, v70_train + oof_res))
print(f"\nPhase 1 OOF RMSE: {phase1_rmse:.5f}")

# Phase 2: Boosted PL
print("\nPhase 2: Boosted Pseudo-Labels...")
test_pseudo = np.clip(v70_test + np.mean(test_res, axis=0) + ALPHA * oof_res.mean(), 0, 100)
test_pseudo_res = test_pseudo - v70_test

oof_final = np.zeros(len(train_df))
test_final = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig, X_test], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig)), test_pseudo_res])
    
    model = FTT_D_Regressor(**ftt_params)
    model.fit(X_comb, res_comb, X_val, res_val)
    
    oof_final[val_idx] = model.predict(X_val).flatten()
    test_final.append(model.predict(X_test).flatten())
    
    print(f"  Fold {fold}/10 done")

final_oof = np.clip(v70_train + oof_final, 0, 100)
final_test = np.clip(v70_test + np.mean(test_final, axis=0), 0, 100)

v106_rmse = np.sqrt(mean_squared_error(y, final_oof))
print(f"\nV106 OOF RMSE: {v106_rmse:.5f}")

# ============================================================================
# RESULTS
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
| **V106** | **FTT** | **V70** | **{v106_rmse:.5f}** | **{v101_rmse - v106_rmse:+.5f}** | **?** |
""")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v106.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v106.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved: submission_v106.csv, oof_v106.csv")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
