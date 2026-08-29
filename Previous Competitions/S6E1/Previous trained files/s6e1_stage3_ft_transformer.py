
"""
S6E1 Stage 3 - FT-Transformer Hybrid Model
==========================================
Model: FT-Transformer (Feature Tokenizer Transformer) via pytabkit
Features: Hybrid (V28 Base + Stage 3 Golden Features)
Architecture: Multi-seed (42, 100, 200)
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

# Install dependencies if needed
try:
    import skorch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skorch", "-q"])

try:
    from pytabkit import FTT_D_Regressor
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import FTT_D_Regressor

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "Stage3_FTT_Hybrid"
    SEEDS = [42, 100, 200]
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

seed_everything(CFG.SEEDS[0])
print(f"Setup complete. Device: {CFG.DEVICE}")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("S6E1 Stage 3 - FT-Transformer Hybrid")
print("="*80)

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")

print(f"Train: {train_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING (Hybrid: V28 Base + Stage 3 Golden)
# ============================================================================

print("\nFeature Engineering (Hybrid)...")

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_hybrid_features(df):
    df_temp = df.copy()
    
    # --- V28 BASE FEATURES ---
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

    # --- STAGE 3 GOLDEN FEATURES ---
    # 1. Z-Score / Aggregation interactions
    if 'study_hours' in df_temp.columns and 'internet_access' in df_temp.columns:
        grp = df_temp.groupby('internet_access')['study_hours']
        mean_map = grp.transform('mean')
        std_map = grp.transform('std')
        
        df_temp['study_hours_minus_internet_access_mean'] = df_temp['study_hours'] - mean_map
        df_temp['study_hours_zscore_internet_access'] = (df_temp['study_hours'] - mean_map) / (std_map + 1e-6)
        
    # 2. Target Encoding Surrogate
    if 'class_attendance' in df_temp.columns and 'course' in df_temp.columns:
        df_temp['class_attendance_by_course_mean'] = df_temp.groupby('course')['class_attendance'].transform('mean')

    # 3. Digits
    for col in ['study_hours', 'class_attendance']:
        if col in df_temp.columns:
            df_temp[f'{col}_decimal'] = (df_temp[col] * 10).astype(int) % 10
            df_temp[f'{col}_digit_0'] = (df_temp[col].abs().astype(int) % 10)

    # Neural Nets prefer String categories for Embeddings
    for col in BASE_COLS:
        df_temp[col] = df_temp[col].astype(str)
        
    return df_temp

train_eng = add_hybrid_features(train_df)
test_eng = add_hybrid_features(test_df)
orig_eng = add_hybrid_features(original_df)

CATS = BASE_COLS
NUMS = [col for col in train_eng.columns if col not in CATS + [CFG.TARGET, 'id', 'student_id']]

print(f"{len(CATS)} Categories, {len(NUMS)} Numerics")

# ============================================================================
# 4. PREPROCESSING (Standard Scaling for NN)
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
# 5. MULTI-SEED TRAINING
# ============================================================================

print(f"\n{'='*80}")
print(f"TRAINING FT-Transformer with {len(CFG.SEEDS)} seeds: {CFG.SEEDS}")
print("="*80)

all_oof = []
all_test = []

for seed in CFG.SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print("="*60)
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
    oof_predictions = np.zeros(len(X))
    test_predictions = []
    
    # FTT Parameters (Auto-tuned by pytabkit usually, but defaults are good)
    model_params = {
        'device': CFG.DEVICE,
        'random_state': seed,
        'verbosity': 0,
        'batch_size': 256,
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Data Augmentation
        X_tr_aug = pd.concat([X_tr, X_original], axis=0)
        y_tr_aug = np.concatenate([y_tr, y_original], axis=0)
        
        # FT-Transformer
        model = FTT_D_Regressor(**model_params)
        model.fit(X_tr_aug, y_tr_aug, X_val, y_val, cat_col_names=CATS)
        
        fold_pred = np.clip(model.predict(X_val), 0, 100)
        oof_predictions[val_idx] = fold_pred
        
        fold_test_pred = np.clip(model.predict(X_test), 0, 100)
        test_predictions.append(fold_test_pred)
        
        rmse = np.sqrt(mean_squared_error(y_val, fold_pred))
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
# 6. RESULTS
# ============================================================================

avg_oof = np.mean(all_oof, axis=0)
avg_test = np.mean(all_test, axis=0)
final_rmse = np.sqrt(mean_squared_error(y, avg_oof))

print(f"\n{'='*80}")
print(f"FINAL 3-SEED FT-TRANSFORMER OOF RMSE: {final_rmse:.5f}")
print("="*80)

# Save
submission = pd.read_csv("Dataset/test.csv" if not os.path.exists('/kaggle/input') else "/kaggle/input/playground-series-s6e1/test.csv", usecols=['id'])
submission['exam_score'] = avg_test
submission.to_csv("submission_stage3_ftt.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': avg_oof})
oof_df.to_csv("oof_stage3_ftt.csv", index=False)

print("Saved files: submission_stage3_ftt.csv, oof_stage3_ftt.csv")
