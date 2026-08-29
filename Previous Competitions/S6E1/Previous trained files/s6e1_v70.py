"""
S6E1 V70 - FT-Transformer + Boosted Pseudo-Labels (Using V44 OOF)
==================================================================
OPTIMIZED: Uses existing V44 OOF/submission - NO FTT training!

Strategy:
1. LOAD V44 OOF (train predictions) + V44 submission (test pseudo-labels)
2. Calculate residuals = y_true - V44_oof
3. Train residual FTT model
4. Update pseudo-labels: new = old + α × residual_pred
5. Retrain FTT with updated pseudo-labels

Time Savings: ~6+ hours (skip FTT baseline training)
"""

import os
import gc
import sys
import subprocess
import random
import warnings
import time
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
start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "V70_FTT_BoostedPL_OOF"
    SEEDS = [42, 100, 200]  # Same as V44
    N_FOLDS = 10
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_ITERATIONS = 1  # 1 iteration gets 99.5% of benefit
    ALPHA = 0.1

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(CFG.SEEDS[0])

print("="*80)
print("S6E1 V70 - FT-Transformer + Boosted Pseudo-Labels (Using V44 OOF)")
print("="*80)
print(f"Device: {CFG.DEVICE}")
print("⚡ OPTIMIZED: Using existing V44 OOF - NO FTT baseline training!")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    # OOF files from dataset
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v44_ftt.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v44_ftt.csv"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/oof_v44_ftt.csv"
    sub_path = "Previous trained files/Submissions/submission_v44_ftt.csv"

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# ============================================================================
# 3. LOAD EXISTING V44 OOF & SUBMISSIONS
# ============================================================================

print("\n" + "="*80 + "\nLOADING V44 OOF (SKIPPING FTT BASELINE TRAINING!)\n" + "="*80)

v44_oof = pd.read_csv(oof_path)
v44_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V44 OOF: {v44_oof.shape}")
print(f"✓ Loaded V44 submission: {v44_sub.shape}")

# V44 OOF uses 'oof_pred' column, not 'exam_score'
oof_col = 'oof_pred' if 'oof_pred' in v44_oof.columns else 'exam_score'
oof_baseline = v44_oof[oof_col].values
test_pseudo_labels = v44_sub['exam_score'].values

y = train_df[CFG.TARGET].values

# Calculate baseline RMSE
baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nV44 Baseline OOF RMSE: {baseline_rmse:.5f}")
print("⚡ Saved ~6+ hours by loading existing OOF instead of training!")

# Calculate residuals
train_residuals = y - oof_baseline
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

# ============================================================================
# 4. FEATURE ENGINEERING (V28 EXACT - NO GOLDEN!) - Same as V44
# ============================================================================

print("\n" + "="*80 + "\nFEATURE ENGINEERING\n" + "="*80)

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_engineered_features(df):
    """V28 feature engineering - NO Golden Features!"""
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

    # Neural Nets prefer String categories for Embeddings (SAME AS STAGE 3)
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
# 5. PREPROCESSING (Standard Scaling for NN)
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

y_original = original_df[CFG.TARGET].values

# ============================================================================
# 6. BOOSTED PSEUDO-LABELS (1 iteration)
# ============================================================================

print("\n" + "="*80 + "\nBOOSTED PSEUDO-LABELS (1 iteration)\n" + "="*80)

kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEEDS[0])

# FTT Parameters (Same as V44)
ftt_params = {
    'device': CFG.DEVICE,
    'random_state': CFG.SEEDS[0],
    'verbosity': 0,
    'batch_size': 256,
}

# Residual model uses same config
res_params = ftt_params.copy()

results = []

for iteration in range(1, CFG.N_ITERATIONS + 1):
    print(f"\n--- Iteration {iteration}/{CFG.N_ITERATIONS} ---")
    
    # Train residual model
    oof_residual = np.zeros(len(X))
    test_residual = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        res_tr, res_val = train_residuals[train_idx], train_residuals[val_idx]
        
        # Augment with original (residuals = 0 for original)
        X_tr_aug = pd.concat([X_tr, X_original], axis=0)
        res_tr_aug = np.concatenate([res_tr, np.zeros(len(X_original))], axis=0)
        
        res_model = FTT_D_Regressor(**res_params)
        res_model.fit(X_tr_aug, res_tr_aug, X_val, res_val, cat_col_names=CATS)
        
        oof_residual[val_idx] = res_model.predict(X_val)
        test_residual.append(res_model.predict(X_test))
        
        print(f"  Residual Fold {fold}: done")
        
        del res_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Update pseudo-labels
    test_residual_mean = np.mean(test_residual, axis=0)
    test_pseudo_labels = np.clip(test_pseudo_labels + CFG.ALPHA * test_residual_mean, 0, 100)
    
    print(f"  Pseudo-labels updated (α={CFG.ALPHA})")
    
    # Retrain with updated pseudo-labels
    oof_updated = np.zeros(len(X))
    test_updated = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Combine: train + original + test (with pseudo-labels)
        X_comb = pd.concat([X_tr, X_original, X_test], axis=0)
        y_comb = np.concatenate([y_tr, y_original, test_pseudo_labels], axis=0)
        
        model = FTT_D_Regressor(**ftt_params)
        model.fit(X_comb, y_comb, X_val, y_val, cat_col_names=CATS)
        
        oof_updated[val_idx] = np.clip(model.predict(X_val), 0, 100)
        test_updated.append(np.clip(model.predict(X_test), 0, 100))
        
        rmse = np.sqrt(mean_squared_error(y_val, oof_updated[val_idx]))
        print(f"  Fold {fold} RMSE: {rmse:.5f}")
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    updated_rmse = np.sqrt(mean_squared_error(y, oof_updated))
    improvement = baseline_rmse - updated_rmse
    print(f"\nIteration {iteration} OOF RMSE: {updated_rmse:.5f} (vs V44 baseline: {improvement:+.5f})")
    
    results.append({
        'iteration': iteration,
        'oof_rmse': updated_rmse,
        'test_preds': np.mean(test_updated, axis=0),
        'oof': oof_updated
    })
    train_residuals = y - oof_updated

# Select best iteration
best = min(results, key=lambda x: x['oof_rmse'])
print(f"\n{'='*80}\nBest Iteration: {best['iteration']} with OOF RMSE: {best['oof_rmse']:.5f}")

# ============================================================================
# 7. SAVE OUTPUTS
# ============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = test_df[['id']].copy()
submission['exam_score'] = best['test_preds']
submission.to_csv("submission_v70.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': best['oof']})
oof_df.to_csv("oof_v70.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v70.csv")
print(f"  oof_v70.csv (for ensemble use)")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("V70 SUMMARY")
print("="*80)
print(f"\n| Version | Model | OOF RMSE | LB Score |")
print(f"|---------|-------|----------|----------|")
print(f"| V44 | FTT (baseline) | {baseline_rmse:.5f} | 8.56179 |")
print(f"| **V70** | **FTT + PL** | **{best['oof_rmse']:.5f}** | **~8.55-8.56** |")
print(f"\n⚡ Time saved by using OOF: ~6+ hours!")
print("\n✅ V70 ready for submission!")
print("="*80)
