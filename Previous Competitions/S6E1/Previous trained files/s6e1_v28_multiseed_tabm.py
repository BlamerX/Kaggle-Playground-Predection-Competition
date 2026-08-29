"""
S6E1 V28 - Multi-seed TabM (3 seeds)
=====================================
Based on V25 TabM with 3 seeds averaged (42, 100, 200)

Results:
- OOF RMSE: 8.59671 (-0.00736 vs V25)
- LB Score: 8.65178 (+0.08952 vs V25) - OVERFIT!

Lesson: Multi-seed averaging hurt generalization. V25 single seed remains best.
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

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION (EXACT V25)
# ============================================================================

class CFG:
    EXP_ID = "V28_MultiSeed_TabM"
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TabM Params (EXACT V25)
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3
    TABM_K = 32
    D_EMBEDDING = 24
    DROPOUT = 0.11
    
    # EXPERIMENT: Multiple seeds
    SEEDS = [42, 100, 200]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(CFG.SEED)
print(f"Setup complete. Device: {CFG.DEVICE}")

# Install pytabkit
print("\nInstalling pytabkit...")
subprocess.run(["pip", "install", "-q", "pytabkit"], check=True)
from pytabkit import TabM_D_Regressor
print("pytabkit installed!")

# ============================================================================
# 2. DATA LOADING (EXACT V25)
# ============================================================================

print("\n" + "="*80)
print("S6E1 V28 - Multi-seed TabM (3 seeds)")
print("="*80)

train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")
print(f"Orig:  {original_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING (EXACT V25)
# ============================================================================

print("\nFeature Engineering (Dual Representation)...")

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_engineered_features(df):
    df_temp = df.copy()
    
    # Trigonometric patterns
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')

    # Non-linear transforms
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
        
    # Magic Formula
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # Cast base features to string for embeddings
    for col in BASE_COLS:
        df_temp[col] = df_temp[col].astype(str)
        
    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(original_df)

CATS = BASE_COLS
NUMS = [col for col in train_eng.columns if col not in CATS + [CFG.TARGET, 'id', 'student_id']]

print(f"{len(CATS)} Categories, {len(NUMS)} Numerics")

# ============================================================================
# 4. PREPROCESSING (EXACT V25)
# ============================================================================

print("\nPreprocessing...")

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
# 5. MULTI-SEED TRAINING (EXPERIMENT)
# ============================================================================

print(f"\n{'='*80}")
print(f"TRAINING TabM with {len(CFG.SEEDS)} seeds: {CFG.SEEDS}")
print("="*80)

tabm_params = {
    'device': CFG.DEVICE,
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

all_oof = []
all_test = []

for seed in CFG.SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print("="*60)
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_predictions = np.zeros(len(X))
    test_predictions = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Data Augmentation
        X_tr_aug = pd.concat([X_tr, X_original], axis=0)
        y_tr_aug = np.concatenate([y_tr, y_original], axis=0)
        
        params = tabm_params.copy()
        params['random_state'] = seed
        
        model = TabM_D_Regressor(**params)
        model.fit(X_tr_aug, y_tr_aug, X_val, y_val, cat_col_names=CATS)
        
        fold_pred = np.clip(model.predict(X_val), 0, 100)
        oof_predictions[val_idx] = fold_pred
        
        fold_test_pred = np.clip(model.predict(X_test), 0, 100)
        test_predictions.append(fold_test_pred)
        
        rmse = np.sqrt(mean_squared_error(y_val, fold_pred))
        fold_scores.append(rmse)
        print(f"  Fold {fold+1} RMSE: {rmse:.5f}")
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    seed_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"\nSeed {seed} OOF RMSE: {seed_oof_rmse:.5f}")
    
    all_oof.append(oof_predictions)
    all_test.append(np.mean(test_predictions, axis=0))

# ============================================================================
# 6. AVERAGE SEEDS
# ============================================================================

avg_oof = np.mean(all_oof, axis=0)
avg_test = np.mean(all_test, axis=0)

final_oof_rmse = np.sqrt(mean_squared_error(y, avg_oof))

# ============================================================================
# 7. RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("V28 RESULTS - Multi-seed TabM")
print("="*80)

BASELINE = 8.60407  # V25 single seed

print(f"\nPer-seed OOF RMSE:")
for i, seed in enumerate(CFG.SEEDS):
    seed_rmse = np.sqrt(mean_squared_error(y, all_oof[i]))
    print(f"  Seed {seed}: {seed_rmse:.5f}")

print(f"\nAveraged OOF RMSE: {final_oof_rmse:.5f}")
print(f"V25 Baseline:      {BASELINE:.5f}")
print(f"Delta vs V25:      {final_oof_rmse - BASELINE:+.5f}")

if final_oof_rmse < BASELINE - 0.001:
    print("\n  ✅ IMPROVEMENT! Multi-seed averaging helps!")
elif final_oof_rmse > BASELINE + 0.001:
    print("\n  ❌ WORSE. Unexpected.")
else:
    print("\n  ≈ SIMILAR. Marginal variance reduction.")

# Save OOF
oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': avg_oof})
oof_df.to_csv('oof_v28.csv', index=False)

# Save Submission
submission_df['exam_score'] = avg_test
submission_df.to_csv('submission_v28.csv', index=False)

print("\nSaved: oof_v28.csv, submission_v28.csv")
print("="*80)
