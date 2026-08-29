"""
S6E1 V30 - 5-Seed TabM
=======================
Extends V28 (3 seeds) to 5 seeds for reduced variance.
Uses EXACT same architecture and FE as V25/V28.

Seeds: 42, 100, 200, 314, 777
Expected runtime: ~5-6 hours on T4 GPU (10 folds × 5 seeds = 50 models)

Baseline to beat: V28 = 8.56178 LB (3 seeds: 42, 100, 200)
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
# 1. CONFIGURATION (EXACT SAME AS V25/V28)
# ============================================================================

class CFG:
    EXP_ID = "V30_5Seed_TabM"
    CV_SEED = 42  # For KFold consistency
    N_FOLDS = 10  # Back to 10-fold (5-fold proved worse in experiment)
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 5 Seeds for multi-seed averaging (replacing 200 which performed worst in V28)
    # V28 Results: Seed 42=8.60263, Seed 100=8.60407, Seed 200=8.60839 (worst)
    MODEL_SEEDS = [42, 100, 314, 777, 1003]
    
    # TabM Params (EXACT SAME as V25/V28 - more_capacity config)
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3
    TABM_K = 32           # Same as V25
    D_EMBEDDING = 24      # Same as V25
    DROPOUT = 0.11        # Same as V25
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

print(f"Setup complete. Device: {CFG.DEVICE}")
print(f"Seeds to use: {CFG.MODEL_SEEDS}")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("S6E1 V30 - 5-Seed TabM")
print("="*80)

train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")
print(f"Orig:  {original_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING (EXACT SAME AS V25)
# ============================================================================

print("\nFeature Engineering (Dual Representation - V25 EXACT)...")

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_engineered_features(df):
    df_temp = df.copy()
    
    # Trigonometric patterns (SAME as V25)
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')

    # Non-linear transforms (SAME as V25)
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
        
    # Magic Formula (SAME as V25)
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # Cast base features to string for embeddings (SAME as V25)
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
# 4. PREPROCESSING (EXACT SAME AS V25)
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
# 5. MODEL TRAINING (5 Seeds × 10 Folds = 50 Models)
# ============================================================================

print("\nTraining TabM (5 Seeds × 10 Folds = 50 Models)...")
print(f"Config: tabm_k={CFG.TABM_K}, d_embedding={CFG.D_EMBEDDING}, dropout={CFG.DROPOUT}")

# Storage for all seeds
all_seed_oof = {}
all_seed_test = {}

for seed_idx, model_seed in enumerate(CFG.MODEL_SEEDS):
    print(f"\n{'='*40}")
    print(f"Seed {seed_idx+1}/{len(CFG.MODEL_SEEDS)}: {model_seed}")
    print(f"{'='*40}")
    
    seed_everything(model_seed)
    
    tabm_params = {
        'device': CFG.DEVICE,
        'random_state': model_seed,  # Use current seed
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
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.CV_SEED)  # Same CV splits for all seeds
    oof_predictions = np.zeros(len(X))
    test_predictions = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Data Augmentation (Train + Original) - SAME as V25
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
    
    seed_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"Seed {model_seed} OOF RMSE: {seed_oof_rmse:.5f}")
    
    all_seed_oof[model_seed] = oof_predictions.copy()
    all_seed_test[model_seed] = np.mean(test_predictions, axis=0)

# ============================================================================
# 6. RESULTS & SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("MULTI-SEED RESULTS")
print("="*80)

# Individual seed OOF scores
for seed, oof in all_seed_oof.items():
    seed_rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"Seed {seed} OOF: {seed_rmse:.5f}")

# Averaged OOF
avg_oof = np.mean([oof for oof in all_seed_oof.values()], axis=0)
avg_oof_rmse = np.sqrt(mean_squared_error(y, avg_oof))

# Averaged Test
avg_test = np.mean([test for test in all_seed_test.values()], axis=0)

print(f"\n5-Seed Averaged OOF RMSE: {avg_oof_rmse:.5f}")
print(f"V28 Baseline (3-seed):    8.59671 (OOF) / 8.56178 (LB)")
print(f"Delta vs V28 OOF:         {avg_oof_rmse - 8.59671:+.5f}")

# Save OOF
oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': avg_oof})
oof_df.to_csv('oof_v30_5seed_tabm.csv', index=False)

# Save Submission
submission_df['exam_score'] = avg_test
submission_df.to_csv('submission_v30_5seed_tabm.csv', index=False)

print("\nSaved: oof_v30_5seed_tabm.csv, submission_v30_5seed_tabm.csv")
print(f"\n{'='*80}")
print(f"V30 COMPLETE - 5-Seed TabM")
print(f"If OOF < 8.597, submit to LB to beat V28 (8.56178)")
print("="*80)
