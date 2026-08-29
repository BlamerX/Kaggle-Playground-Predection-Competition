"""
S6E1 V55-V56 - TabM + Row-wise Sorted Features (V55) + Target Signal Decomposition (V56)
=========================================================================================
COMBINED EXPERIMENT: Runs both V55 and V56 in one script, outputs separate OOF files.

V55: Row-wise Sorted Features (from S4E5 1st Place)
V56: Target Signal Decomposition (from S4E5 1st Place)

Both use V61 OOF as baseline (OOF-leveraged approach).

Sources:
- V61: s6e1_v61.py
- S4E5 1st Place: https://www.kaggle.com/c/playground-series-s4e5/discussion/509043
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
    from pytabkit import TabM_D_Regressor
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Regressor

warnings.filterwarnings('ignore')
start_time = time.time()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class CFG:
    EXP_ID = "V55_V56_TabM_S4E5_Techniques"
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_ITERATIONS = 1
    ALPHA = 0.1
    
    # Target Signal Decomposition scale (from S4E5: 0.1 for mean, or 0.005 for sum of 20 features)
    TARGET_DECOMP_SCALE = 0.1  # We use mean * 0.1

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(CFG.SEED)

print("="*80)
print("S6E1 V55-V56 - TabM + S4E5 Techniques (Row-wise Sorted + Target Decomposition)")
print("="*80)
print(f"Device: {CFG.DEVICE}")
print("⚡ Using V61 OOF (OOF-leveraged approach)")

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v61.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v61.csv"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/oof_v61.csv"
    sub_path = "Previous trained files/Submissions/submission_v61.csv"

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 3. LOAD V61 OOF & SUBMISSIONS
# =============================================================================

print("\n" + "="*80 + "\nLOADING V61 OOF (SKIPPING BASELINE TRAINING!)\n" + "="*80)

v61_oof = pd.read_csv(oof_path)
v61_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V61 OOF: {v61_oof.shape}")
print(f"✓ Loaded V61 submission: {v61_sub.shape}")

oof_baseline = v61_oof['exam_score'].values
test_pseudo_labels_base = v61_sub['exam_score'].values

y = train_df[CFG.TARGET].values

baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nV61 Baseline OOF RMSE: {baseline_rmse:.5f}")

train_residuals = y - oof_baseline
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80 + "\nFEATURE ENGINEERING\n" + "="*80)

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

NUMERIC_COLS = ['study_hours', 'class_attendance', 'sleep_hours', 'age']

def add_engineered_features(df):
    """V61 feature engineering (base)"""
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
    
    return df_temp

def add_row_wise_sorted_features(df, numeric_cols):
    """
    V55: Row-wise Sorted Features (S4E5 1st Place)
    Sort numerical values within each row to capture distribution patterns.
    """
    df_temp = df.copy()
    
    # Get numeric values and sort row-wise
    numeric_values = df_temp[numeric_cols].values
    sorted_values = np.sort(numeric_values, axis=1)
    
    # Create sorted feature columns
    for i in range(len(numeric_cols)):
        df_temp[f'sorted_feat_{i}'] = sorted_values[:, i]
    
    # Additional row-wise statistics
    df_temp['row_mean'] = np.mean(numeric_values, axis=1)
    df_temp['row_std'] = np.std(numeric_values, axis=1)
    df_temp['row_max'] = np.max(numeric_values, axis=1)
    df_temp['row_min'] = np.min(numeric_values, axis=1)
    df_temp['row_range'] = df_temp['row_max'] - df_temp['row_min']
    
    return df_temp

def add_target_signal_decomposition(df, numeric_cols, scale=0.1):
    """
    V56: Target Signal Decomposition (S4E5 1st Place)
    Calculate the linear component that will be subtracted from target.
    Returns the linear component for each row.
    """
    # Calculate row mean of numeric features
    linear_component = df[numeric_cols].mean(axis=1) * scale
    return linear_component

# Apply base engineering
train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(original_df)

# =============================================================================
# 5. EXPERIMENT V55: Row-wise Sorted Features
# =============================================================================

print("\n" + "="*80)
print("EXPERIMENT V55: Row-wise Sorted Features")
print("="*80)

# Add row-wise sorted features
train_v55 = add_row_wise_sorted_features(train_eng, NUMERIC_COLS)
test_v55 = add_row_wise_sorted_features(test_eng, NUMERIC_COLS)
orig_v55 = add_row_wise_sorted_features(orig_eng, NUMERIC_COLS)

print(f"V55 added features: sorted_feat_0-3, row_mean, row_std, row_max, row_min, row_range")
print(f"V55 train shape: {train_v55.shape}")

# Neural Nets prefer String categories for Embeddings
CATS = BASE_COLS
for col in CATS:
    train_v55[col] = train_v55[col].astype(str)
    test_v55[col] = test_v55[col].astype(str)
    orig_v55[col] = orig_v55[col].astype(str)

NUMS_V55 = [col for col in train_v55.columns if col not in CATS + [CFG.TARGET, 'id', 'student_id']]
print(f"V55: {len(CATS)} Categories, {len(NUMS_V55)} Numerics")

# Preprocessing
encoder_v55 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler_v55 = StandardScaler()

encoder_v55.fit(train_v55[CATS])
scaler_v55.fit(train_v55[NUMS_V55])

def preprocess_v55(df_eng):
    cats_encoded = pd.DataFrame(encoder_v55.transform(df_eng[CATS]), columns=CATS, index=df_eng.index)
    nums_scaled = pd.DataFrame(scaler_v55.transform(df_eng[NUMS_V55]), columns=NUMS_V55, index=df_eng.index)
    return pd.concat([nums_scaled, cats_encoded], axis=1)

X_v55 = preprocess_v55(train_v55)
X_test_v55 = preprocess_v55(test_v55)
X_original_v55 = preprocess_v55(orig_v55)

y_original = original_df[CFG.TARGET].values

# TabM params (same as V61)
tabm_params = {
    'device': CFG.DEVICE,
    'verbosity': 0,
    'arch_type': 'tabm-mini-normal',
    'tabm_k': 32,
    'num_emb_type': 'pwl',
    'd_embedding': 24, 
    'batch_size': 256, 
    'lr': 1e-3, 
    'n_epochs': 100,
    'dropout': 0.11,
    'd_block': 256, 
    'n_blocks': 5,
    'patience': 4,
    'weight_decay': 1e-2,
    'random_state': CFG.SEED,
}

res_params = tabm_params.copy()
res_params['n_epochs'] = 50

kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

# V55 Training
print("\n--- V55 Training ---")
test_pseudo_labels_v55 = test_pseudo_labels_base.copy()

# Train residual model
oof_residual_v55 = np.zeros(len(X_v55))
test_residual_v55 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_v55), 1):
    X_tr, X_val = X_v55.iloc[train_idx], X_v55.iloc[val_idx]
    res_tr, res_val = train_residuals[train_idx], train_residuals[val_idx]
    
    X_tr_aug = pd.concat([X_tr, X_original_v55], axis=0)
    res_tr_aug = np.concatenate([res_tr, np.zeros(len(X_original_v55))], axis=0)
    
    res_model = TabM_D_Regressor(**res_params)
    res_model.fit(X_tr_aug, res_tr_aug, X_val, res_val, cat_col_names=CATS)
    
    oof_residual_v55[val_idx] = res_model.predict(X_val)
    test_residual_v55.append(res_model.predict(X_test_v55))
    
    print(f"  V55 Residual Fold {fold}: done")
    
    del res_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Update pseudo-labels
test_residual_mean_v55 = np.mean(test_residual_v55, axis=0)
test_pseudo_labels_v55 = np.clip(test_pseudo_labels_v55 + CFG.ALPHA * test_residual_mean_v55, 0, 100)

# Retrain with updated pseudo-labels
oof_v55 = np.zeros(len(X_v55))
test_v55_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_v55), 1):
    X_tr, X_val = X_v55.iloc[train_idx], X_v55.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    X_comb = pd.concat([X_tr, X_original_v55, X_test_v55], axis=0)
    y_comb = np.concatenate([y_tr, y_original, test_pseudo_labels_v55], axis=0)
    
    model = TabM_D_Regressor(**tabm_params)
    model.fit(X_comb, y_comb, X_val, y_val, cat_col_names=CATS)
    
    oof_v55[val_idx] = np.clip(model.predict(X_val), 0, 100)
    test_v55_preds.append(np.clip(model.predict(X_test_v55), 0, 100))
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_v55[val_idx]))
    print(f"  V55 Fold {fold} RMSE: {rmse:.5f}")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

v55_rmse = np.sqrt(mean_squared_error(y, oof_v55))
v55_improvement = baseline_rmse - v55_rmse
print(f"\n✅ V55 OOF RMSE: {v55_rmse:.5f} (vs V61: {v55_improvement:+.5f})")

# =============================================================================
# 6. EXPERIMENT V56: Target Signal Decomposition
# =============================================================================

print("\n" + "="*80)
print("EXPERIMENT V56: Target Signal Decomposition")
print("="*80)

# Use base features (same as V61)
for col in CATS:
    train_eng[col] = train_eng[col].astype(str)
    test_eng[col] = test_eng[col].astype(str)
    orig_eng[col] = orig_eng[col].astype(str)

NUMS_V56 = [col for col in train_eng.columns if col not in CATS + [CFG.TARGET, 'id', 'student_id']]

encoder_v56 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler_v56 = StandardScaler()

encoder_v56.fit(train_eng[CATS])
scaler_v56.fit(train_eng[NUMS_V56])

def preprocess_v56(df_eng):
    cats_encoded = pd.DataFrame(encoder_v56.transform(df_eng[CATS]), columns=CATS, index=df_eng.index)
    nums_scaled = pd.DataFrame(scaler_v56.transform(df_eng[NUMS_V56]), columns=NUMS_V56, index=df_eng.index)
    return pd.concat([nums_scaled, cats_encoded], axis=1)

X_v56 = preprocess_v56(train_eng)
X_test_v56 = preprocess_v56(test_eng)
X_original_v56 = preprocess_v56(orig_eng)

# Calculate linear component for target decomposition
linear_train = add_target_signal_decomposition(train_df, NUMERIC_COLS, CFG.TARGET_DECOMP_SCALE)
linear_test = add_target_signal_decomposition(test_df, NUMERIC_COLS, CFG.TARGET_DECOMP_SCALE)
linear_orig = add_target_signal_decomposition(original_df, NUMERIC_COLS, CFG.TARGET_DECOMP_SCALE)

print(f"Linear component stats: mean={linear_train.mean():.4f}, std={linear_train.std():.4f}")

# Decomposed target = y - linear_component
y_decomposed = y - linear_train.values
y_orig_decomposed = y_original - linear_orig.values

# Decomposed residuals (from V61 OOF, also decomposed)
decomposed_baseline = oof_baseline - linear_train.values
decomposed_residuals = y_decomposed - decomposed_baseline

print(f"Decomposed target stats: mean={y_decomposed.mean():.4f}, std={y_decomposed.std():.4f}")

# V56 Training
print("\n--- V56 Training ---")
test_pseudo_labels_v56 = test_pseudo_labels_base - linear_test.values  # Decompose pseudo-labels too

# Train residual model on decomposed residuals
oof_residual_v56 = np.zeros(len(X_v56))
test_residual_v56 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_v56), 1):
    X_tr, X_val = X_v56.iloc[train_idx], X_v56.iloc[val_idx]
    res_tr, res_val = decomposed_residuals[train_idx], decomposed_residuals[val_idx]
    
    X_tr_aug = pd.concat([X_tr, X_original_v56], axis=0)
    res_tr_aug = np.concatenate([res_tr, np.zeros(len(X_original_v56))], axis=0)
    
    res_model = TabM_D_Regressor(**res_params)
    res_model.fit(X_tr_aug, res_tr_aug, X_val, res_val, cat_col_names=CATS)
    
    oof_residual_v56[val_idx] = res_model.predict(X_val)
    test_residual_v56.append(res_model.predict(X_test_v56))
    
    print(f"  V56 Residual Fold {fold}: done")
    
    del res_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Update pseudo-labels (still decomposed)
test_residual_mean_v56 = np.mean(test_residual_v56, axis=0)
test_pseudo_labels_v56 = test_pseudo_labels_v56 + CFG.ALPHA * test_residual_mean_v56

# Retrain with updated decomposed pseudo-labels
oof_v56_decomposed = np.zeros(len(X_v56))
test_v56_decomposed = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_v56), 1):
    X_tr, X_val = X_v56.iloc[train_idx], X_v56.iloc[val_idx]
    y_tr_decomp = y_decomposed[train_idx]
    y_val_decomp = y_decomposed[val_idx]
    
    X_comb = pd.concat([X_tr, X_original_v56, X_test_v56], axis=0)
    y_comb = np.concatenate([y_tr_decomp, y_orig_decomposed, test_pseudo_labels_v56], axis=0)
    
    model = TabM_D_Regressor(**tabm_params)
    model.fit(X_comb, y_comb, X_val, y_val_decomp, cat_col_names=CATS)
    
    oof_v56_decomposed[val_idx] = model.predict(X_val)
    test_v56_decomposed.append(model.predict(X_test_v56))
    
    # Calculate RMSE on original scale
    oof_v56_original = oof_v56_decomposed[val_idx] + linear_train.values[val_idx]
    rmse = np.sqrt(mean_squared_error(y[val_idx], oof_v56_original))
    print(f"  V56 Fold {fold} RMSE: {rmse:.5f}")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Add back linear component for final predictions
oof_v56 = np.clip(oof_v56_decomposed + linear_train.values, 0, 100)
test_v56_preds_decomposed = np.mean(test_v56_decomposed, axis=0)
test_v56_preds = np.clip(test_v56_preds_decomposed + linear_test.values, 0, 100)

v56_rmse = np.sqrt(mean_squared_error(y, oof_v56))
v56_improvement = baseline_rmse - v56_rmse
print(f"\n✅ V56 OOF RMSE: {v56_rmse:.5f} (vs V61: {v56_improvement:+.5f})")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

# V55 outputs
submission_v55 = test_df[['id']].copy()
submission_v55['exam_score'] = np.mean(test_v55_preds, axis=0)
submission_v55.to_csv("submission_v55.csv", index=False)

oof_v55_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v55})
oof_v55_df.to_csv("oof_v55.csv", index=False)

# V56 outputs
submission_v56 = test_df[['id']].copy()
submission_v56['exam_score'] = test_v56_preds
submission_v56.to_csv("submission_v56.csv", index=False)

oof_v56_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v56})
oof_v56_df.to_csv("oof_v56.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved:")
print(f"  V55: submission_v55.csv, oof_v55.csv")
print(f"  V56: submission_v56.csv, oof_v56.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

# =============================================================================
# 8. FINAL COMPARISON
# =============================================================================

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"\n| Version | Technique | OOF RMSE | vs V61 |")
print(f"|---------|-----------|----------|--------|")
print(f"| V61 | Baseline (TabM + PL) | {baseline_rmse:.5f} | — |")
print(f"| **V55** | + Row-wise Sorted | **{v55_rmse:.5f}** | {v55_improvement:+.5f} |")
print(f"| **V56** | + Target Decomposition | **{v56_rmse:.5f}** | {v56_improvement:+.5f} |")

# Determine winner
if v55_rmse < v56_rmse and v55_rmse < baseline_rmse:
    print(f"\n🏆 V55 (Row-wise Sorted) is the BEST with OOF {v55_rmse:.5f}")
    print("   Submit submission_v55.csv to Kaggle!")
elif v56_rmse < v55_rmse and v56_rmse < baseline_rmse:
    print(f"\n🏆 V56 (Target Decomposition) is the BEST with OOF {v56_rmse:.5f}")
    print("   Submit submission_v56.csv to Kaggle!")
elif baseline_rmse <= min(v55_rmse, v56_rmse):
    print(f"\n⚠️ V61 baseline is still the best. Neither technique helped.")
else:
    print(f"\n🔍 Results are mixed. Try submitting both to Kaggle to verify.")

print("\n" + "="*80)
print("✅ V55-V56 Experiment Complete!")
print("="*80)
