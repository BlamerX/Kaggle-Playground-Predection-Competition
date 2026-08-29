"""
S6E1 V26 - Larger TabM (FAILED)
================================
Attempted: tabm_k=48, d_embedding=32
Result: 8.57376 LB (WORSE than V25's 8.56226)

LESSON: Larger model = MORE OVERFITTING
        OOF looked better (8.613 vs 8.615) but LB was worse (+0.0115)
        V25 (32/24) remains the optimal configuration.
"""

import os
import gc
import sys
import subprocess
import random
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Install pytabkit if needed
try:
    from pytabkit import TabM_D_Regressor
except ImportError:
    print("Installing pytabkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Regressor

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "V26_TabM_larger"
    SEED = 42
    N_FOLDS = 5  # Was 5-fold screening
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TabM Params (OVERFITTED!)
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 1e-3
    TABM_K = 48           # Too large - caused overfitting
    D_EMBEDDING = 32      # Too large - caused overfitting
    DROPOUT = 0.11
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(CFG.SEED)
print(f"Setup complete. Device: {CFG.DEVICE}")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("S6E1 V26 - Larger TabM (FAILED EXPERIMENT)")
print("="*80)

train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")
print(f"Orig:  {original_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\nFeature Engineering...")

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_engineered_features(df):
    df_temp = df.copy()
    
    # Polynomials
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    
    # Log transforms
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
    
    # Magic Formula
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )
    
    # Ordinal encoding
    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2}).fillna(1)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2}).fillna(1)
    
    # Interactions
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']

    # Cast base features to string for embeddings
    for col in BASE_COLS:
        df_temp[col] = df_temp[col].astype(str)
        
    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(original_df)

CATS = BASE_COLS
NUMS = [
    'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared',
    'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
    'feature_formula',
    'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
    'study_hours_times_attendance', 'study_hours_times_sleep'
]

print(f"{len(CATS)} Categories, {len(NUMS)} Numerics")

# ============================================================================
# 4. PREPROCESSING
# ============================================================================

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()

encoder.fit(train_eng[CATS])
scaler.fit(train_eng[NUMS])

def preprocess(df_eng):
    cats_encoded = pd.DataFrame(encoder.transform(df_eng[CATS]), columns=CATS, index=df_eng.index)
    nums_scaled = pd.DataFrame(scaler.transform(df_eng[NUMS]), columns=NUMS, index=df_eng.index)
    return pd.concat([nums_scaled, cats_encoded], axis=1)

X = preprocess(train_eng)
X_test = preprocess(test_eng)
X_original = preprocess(orig_eng)

y = train_df[CFG.TARGET].values
y_original = original_df[CFG.TARGET].values

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

print("\nTraining TabM (LARGER - 48/32)...")
print(f"Config: tabm_k={CFG.TABM_K}, d_embedding={CFG.D_EMBEDDING}, dropout={CFG.DROPOUT}")

tabm_params = {
    'device': CFG.DEVICE,
    'random_state': 100,
    'verbosity': 0,
    'arch_type': 'tabm-mini-normal',
    'tabm_k': CFG.TABM_K,
    'num_emb_type': 'pwl',
    'd_embedding': CFG.D_EMBEDDING, 
    'batch_size': CFG.BATCH_SIZE, 
    'lr': CFG.LR, 
    'n_epochs': CFG.EPOCHS,
    'dropout': CFG.DROPOUT,
    'd_block': 256, 
    'n_blocks': 5,
    'patience': 4,
    'weight_decay': 1e-2,
}

kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_predictions = np.zeros(len(X))
test_predictions = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    X_tr_aug = pd.concat([X_tr, X_original], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_original], axis=0)
    
    model = TabM_D_Regressor(**tabm_params)
    model.fit(X_tr_aug, y_tr_aug, X_val, y_val, cat_col_names=CATS)
    
    fold_pred = np.clip(model.predict(X_val), 0, 100)
    oof_predictions[val_idx] = fold_pred
    
    fold_test_pred = np.clip(model.predict(X_test), 0, 100)
    test_predictions.append(fold_test_pred)
    
    rmse = np.sqrt(mean_squared_error(y_val, fold_pred))
    fold_scores.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.5f}")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================================
# 6. RESULTS
# ============================================================================

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

print(f"\nOOF RMSE: {oof_rmse:.5f}")
print(f"V25 Baseline: 8.60407 (10-fold)")
print(f"\n⚠️ WARNING: Larger model may overfit!")

# Save
oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_predictions})
oof_df.to_csv('oof_v26_larger_tabm.csv', index=False)

avg_test_preds = np.mean(test_predictions, axis=0)
submission_df['exam_score'] = avg_test_preds
submission_df.to_csv('submission_v26_larger_tabm.csv', index=False)

print("\nSaved: oof_v26_larger_tabm.csv, submission_v26_larger_tabm.csv")
print(f"\n{'='*80}")
print("V26 COMPLETE - RESULT: 8.57376 LB (WORSE than V25)")
print("="*80)
