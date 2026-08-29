"""
S6E1 V27 - FT-Transformer (Feature Tokenizer Transformer)
==========================================================
Different deep learning approach from TabM. Uses Transformer attention 
on tabular data for both categorical and numerical features.

Key Differences from TabM:
- FT-Transformer: Uses self-attention on feature embeddings
- TabM: Uses mixture of experts approach

Both available in pytabkit library.
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

# Install dependencies
import subprocess
import sys

# skorch is required by pytabkit FTT
try:
    import skorch
except ImportError:
    print("Installing skorch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skorch", "-q"])
    import skorch

try:
    from pytabkit import FTT_D_Regressor
except ImportError:
    print("Installing pytabkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import FTT_D_Regressor

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(CFG.SEED)
print("="*70)
print("V27 FT-Transformer (Feature Tokenizer Transformer)")
print("="*70)
print(f"Device: {CFG.DEVICE}")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Orig: {original_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING (Same as V25 TabM for fair comparison)
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
# 4. PREPROCESSING
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
# 5. MODEL TRAINING (FT-Transformer)
# ============================================================================

print("\n" + "="*70)
print("Training FT-Transformer (10-Fold CV)")
print("="*70)

kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_predictions = np.zeros(len(X))
test_predictions = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/{CFG.N_FOLDS} ---")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    # Data Augmentation (Train + Original)
    X_tr_aug = pd.concat([X_tr, X_original], axis=0)
    y_tr_aug = np.concatenate([y_tr, y_original], axis=0)
    
    # FTT with minimal params - let defaults handle architecture
    model = FTT_D_Regressor(
        device=CFG.DEVICE,
        random_state=100,
        verbosity=1,
    )
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
# 6. RESULTS & SUBMISSION
# ============================================================================

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

print("\n" + "="*70)
print("V27 FT-TRANSFORMER RESULTS")
print("="*70)
print(f"Average Fold RMSE: {np.mean(fold_scores):.5f}")
print(f"OOF RMSE:          {oof_rmse:.5f}")
print(f"V25 TabM:          8.60407")
print(f"Delta vs V25:      {oof_rmse - 8.60407:+.5f}")

# Save OOF
oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_predictions})
oof_df.to_csv('oof_v27_ftt.csv', index=False)

# Save Submission
avg_test_preds = np.mean(test_predictions, axis=0)
submission_df['exam_score'] = avg_test_preds
submission_df.to_csv('submission_v27_ftt.csv', index=False)

print("\nSaved: oof_v27_ftt.csv, submission_v27_ftt.csv")
print(f"\n{'='*70}")
print("V27 COMPLETE - FT-Transformer")
print("="*70)
