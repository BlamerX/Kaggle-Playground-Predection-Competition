"""
S6E1 V24 - TabM Dual Representation (Deep Learning)
===================================================
Replicating the high-scoring helper notebook strategy (8.56240 LB).

Key Strategy:
1.  Model: TabM (Table Model) from pytabkit.
2.  Dual Representation:
    - Numeric features are scaled and used directly.
    - Categorical features are ordinal encoded.
    - CRITICAL: All base features (even numeric like 'age') are ALSO treated as categorical
      to allow the model to learn embeddings for specific values.
3.  Data Augmentation: original_df is added to the TRAINING set of each fold.

Dependencies:
    pip install pytabkit
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

# Try importing pytabkit, handle if missing
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
    EXP_ID = "V24_TabM_DualRep"
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TabM Params (from helper notebook)
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3
    TABM_K = 24
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

print("\nLoad Data...")
train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train: {train_df.shape}")
print(f"Test:  {test_df.shape}")
print(f"Orig:  {original_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING (DUAL REPRESENTATION)
# ============================================================================

print("\nFeature Engineering (Dual Representation)...")

# Base features to be treated as Categorical (The "Face" of artifacts)
# Including numeric cols here effectively bins them by unique value for embedding
BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_engineered_features(df):
    df_temp = df.copy()
    
    # --- 1. Numerical Perspective (The Trends) ---
    # Trigonometric patterns (cyclic nature)
    # 12 is possibly arbitrary or related to months/hours, strictly keeping helper logic
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 100).astype('float32') # Fixed: attendance is 0-100 usually
    
    # helper notebook had /12 for attendance too? Let's check logic.
    # Actually checking helper notebook again:  df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12)
    # This seems odd for attendance (0-100), but if it works for them... 
    # Wait, the prompt provided content says: / 12 for both. I will stick to their logic exactly to reproduce score.
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32') 

    # Non-linear transforms
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
        
    # The "Magic Formula" (Linear Backbone from previous winning notebook ideas)
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # --- 2. Categorical Perspective (The Artifacts) ---
    # Cast base features to string to force Embedding learning in TabM
    # This creates the "Dual" nature -> Numeric value exists in scaled columns, 
    # Discrete value exists in categorical columns.
    for col in BASE_COLS:
        df_temp[col] = df_temp[col].astype(str)
        
    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(original_df)

# Define Feature Groups
CATS = BASE_COLS
# Numeric cols are everything else except target/id/CATS
NUMS = [col for col in train_eng.columns if col not in CATS + [CFG.TARGET, 'id', 'student_id']]

print(f"Directed Cast: {len(CATS)} Categories (embeddings), {len(NUMS)} Numerics (scaled).")

# ============================================================================
# 4. PREPROCESSING (ENCODING & SCALING)
# ============================================================================

print("\nPreprocessing...")

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()

# Fit only on TRAIN (competition data)
encoder.fit(train_eng[CATS])
scaler.fit(train_eng[NUMS])

def preprocess(df_eng):
    # Important: Reconstruct DataFrame to keep columns aligned
    cats_encoded = pd.DataFrame(encoder.transform(df_eng[CATS]), columns=CATS, index=df_eng.index)
    nums_scaled = pd.DataFrame(scaler.transform(df_eng[NUMS]), columns=NUMS, index=df_eng.index)
    return pd.concat([nums_scaled, cats_encoded], axis=1)

X = preprocess(train_eng)
X_test = preprocess(test_eng)
X_original = preprocess(orig_eng)

y = train_df[CFG.TARGET].values
y_original = original_df[CFG.TARGET].values

# ============================================================================
# 5. MODEL TRAINING (TabM)
# ============================================================================

print("\nTraining TabM (10-Fold)...")

tabm_params = {
    'device': CFG.DEVICE,
    'random_state': 100,
    'verbosity': 0,
    'arch_type': 'tabm-mini-normal',
    'tabm_k': CFG.TABM_K,
    'num_emb_type': 'pwl',
    'd_embedding': 16, 
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
    # Data Splitting
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # Data Augmentation (Train + Original)
    # Augment training set with original data, but keep validation pure competition data
    X_tr_aug = pd.concat([X_tr, X_original], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_original], axis=0)
    
    # Init Model
    # Note: TabM_D_Regressor might need to be imported inside loop if memory is tight, 
    # but usually fine outside. 
    try:
        model = TabM_D_Regressor(**tabm_params)
        model.fit(
            X_tr_aug, y_tr_aug,
            X_val, y_val,
            cat_col_names=CATS
        )
        
        # Inference
        fold_p = model.predict(X_val)
        # TabM usually clips internally or returns valid range, but for safety:
        fold_p = np.clip(fold_p, 0, 100)
        
        oof_predictions[val_idx] = fold_p
        
        # Test inference
        fold_test_p = model.predict(X_test)
        fold_test_p = np.clip(fold_test_p, 0, 100)
        test_predictions.append(fold_test_p)
        
        rmse = np.sqrt(mean_squared_error(y_val, fold_p))
        fold_scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.5f}")
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except NameError:
        print("TabM_D_Regressor not defined. Skipping fold (install pytabkit)")
        break

# ============================================================================
# 6. RESULTS & SUBMISSION
# ============================================================================

if len(fold_scores) > 0:
    mean_rmse = np.mean(fold_scores)
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    
    print(f"\nAverage Fold RMSE: {mean_rmse:.5f}")
    print(f"OOF RMSE:          {oof_rmse:.5f}")
    
    # Save OOF
    oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_predictions})
    oof_df.to_csv('oof_v24_tabm.csv', index=False)
    
    # Save Submission
    avg_test_preds = np.mean(test_predictions, axis=0)
    submission_df['exam_score'] = avg_test_preds
    submission_df.to_csv('submission_v24_tabm.csv', index=False)
    
    print("\nSaved: oof_v24_tabm.csv, submission_v24_tabm.csv")
    print(f"Please submit to Leaderboard to verify improvement over V23 (8.56367)")
else:
    print("\nTraining failed (likely missing pytabkit library).")
