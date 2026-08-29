"""
S6E1 V47 - Clean Ridge Stack (All No-Golden Models)
====================================================
Uses ONLY models without Golden Features for better LB generalization.

Models:
- V28 TabM (8.56178 LB) - Best single model
- V34 XGB (8.56352 LB) - Best XGBoost
- V44 FTT (8.56179 LB) - Best FT-Transformer
- V45 ResNet (8.57707 LB) - Best ResNet
- V46 LGB (8.58266 LB) - Best LightGBM

Expected: Better than V43 (8.55253) due to cleaner base models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*80)
print("S6E1 V47 - Clean Ridge Stack (All No-Golden Models)")
print("="*80)

# ============================================================================
# 1. PATHS
# ============================================================================

# Kaggle paths
if os.path.exists('/kaggle/input'):
    BASE = '/kaggle/input/playground-series-s6e1'
    OOF_BASE = '/kaggle/input/s6e1-oof-files/'
    SUB_BASE = '/kaggle/input/s6e1-submission-files/'
    print("Environment: KAGGLE")
else:
    BASE = 'Dataset'
    OOF_BASE = 'Previous trained files/OOF/'
    SUB_BASE = 'Previous trained files/Submissions/'
    print("Environment: LOCAL")

train_file = os.path.join(BASE, 'train.csv')
test_file = os.path.join(BASE, 'test.csv')

# ============================================================================
# 2. DEFINE MODELS (ALL NO-GOLDEN!)
# ============================================================================

MODELS = {
    "V28_TabM": {
        "oof": OOF_BASE + "oof_v28.csv",
        "sub": SUB_BASE + "submission_v28.csv",
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.597,
        "lb": 8.56178
    },
    "V34_XGB": {
        "oof": OOF_BASE + "oof_v34.csv",
        "sub": SUB_BASE + "submission_v34.csv",
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.601,
        "lb": 8.56352
    },
    "V44_FTT": {
        "oof": OOF_BASE + "oof_v44_ftt.csv",
        "sub": SUB_BASE + "submission_v44_ftt.csv",
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.605,
        "lb": 8.56179
    },
    "V45_ResNet": {
        "oof": OOF_BASE + "oof_v45_resnet.csv",
        "sub": SUB_BASE + "submission_v45_resnet.csv",
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.616,
        "lb": 8.57707
    },
    "V46_LGB": {
        "oof": OOF_BASE + "oof_v46_lgb.csv",
        "sub": SUB_BASE + "submission_v46_lgb.csv",
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.622,
        "lb": 8.58266
    }
}

# ============================================================================
# 3. LOAD DATA
# ============================================================================

train_df = pd.read_csv(train_file)
y = train_df['exam_score'].values
n_train = len(y)

print(f"\nTrain samples: {n_train}")
print(f"Models to stack: {len(MODELS)}")

# ============================================================================
# 4. LOAD OOF AND SUBMISSION PREDICTIONS
# ============================================================================

print(f"\n{'='*80}")
print("LOADING OOF AND SUBMISSION FILES")
print("="*80)

oof_matrix = []
sub_matrix = []
model_names = []

for name, config in MODELS.items():
    oof_path = config["oof"]
    sub_path = config["sub"]
    
    # Check if files exist
    if not os.path.exists(oof_path):
        print(f"  ❌ {name}: OOF not found at {oof_path}")
        continue
    if not os.path.exists(sub_path):
        print(f"  ❌ {name}: SUB not found at {sub_path}")
        continue
    
    # Load OOF
    oof_df = pd.read_csv(oof_path)
    oof_col = config["oof_col"]
    if oof_col not in oof_df.columns:
        oof_col = oof_df.columns[-1]  # Fallback to last column
    oof_preds = oof_df[oof_col].values
    
    # Load submission
    sub_df = pd.read_csv(sub_path)
    sub_col = config["sub_col"]
    if sub_col not in sub_df.columns:
        sub_col = sub_df.columns[-1]
    sub_preds = sub_df[sub_col].values
    
    # Validate OOF
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    expected = config["expected_oof"]
    
    if abs(oof_rmse - expected) > 0.05:
        print(f"  ⚠️ {name}: OOF={oof_rmse:.5f} (expected {expected:.3f})")
    else:
        print(f"  ✅ {name}: OOF={oof_rmse:.5f}, LB={config['lb']:.5f}")
    
    oof_matrix.append(oof_preds)
    sub_matrix.append(sub_preds)
    model_names.append(name)

# Convert to numpy arrays
X_oof = np.column_stack(oof_matrix)
X_sub = np.column_stack(sub_matrix)

print(f"\nLoaded {len(model_names)} models: {model_names}")
print(f"OOF shape: {X_oof.shape}, Sub shape: {X_sub.shape}")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("CORRELATION ANALYSIS")
print("="*80)

corr_matrix = np.corrcoef(X_oof.T)
print("\nModel Correlations:")
print("="*60)
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        corr = corr_matrix[i, j]
        print(f"  {model_names[i]} ↔ {model_names[j]}: {corr:.4f}")

avg_corr = (np.sum(corr_matrix) - len(model_names)) / (len(model_names) * (len(model_names) - 1))
print(f"\nAverage correlation: {avg_corr:.4f}")

# ============================================================================
# 6. RIDGE STACKING
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE STACKING")
print("="*80)

kf = KFold(n_splits=10, shuffle=True, random_state=1003)

oof_stack = np.zeros(n_train)
sub_stack = np.zeros(len(X_sub))

alphas = np.logspace(-3, 3, 50)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_oof), 1):
    X_tr, X_val = X_oof[train_idx], X_oof[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    ridge.fit(X_tr, y_tr)
    
    val_preds = ridge.predict(X_val)
    oof_stack[val_idx] = val_preds
    
    sub_stack += ridge.predict(X_sub) / 10
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"  Fold {fold:2d} RMSE: {fold_rmse:.5f} (alpha={ridge.alpha_:.4f})")

# Final OOF score
oof_rmse = np.sqrt(mean_squared_error(y, oof_stack))
print(f"\nRidge Stack OOF RMSE: {oof_rmse:.5f}")

# ============================================================================
# 7. FEATURE WEIGHTS ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("MODEL WEIGHTS (from final Ridge)")
print("="*80)

# Train final Ridge on all data
ridge_final = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
ridge_final.fit(X_oof, y)

weights = ridge_final.coef_
weight_sum = np.sum(np.abs(weights))
normalized_weights = np.abs(weights) / weight_sum * 100

print("\nModel Contributions:")
for name, weight, norm_weight in sorted(zip(model_names, weights, normalized_weights), 
                                         key=lambda x: -abs(x[1])):
    print(f"  {name:15s}: {weight:+.4f} ({norm_weight:.1f}%)")

# ============================================================================
# 8. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("FINAL RESULTS")
print("="*80)

print(f"\nV47 Clean Stack OOF RMSE: {oof_rmse:.5f}")
print(f"V43 Stack OOF RMSE: 8.58561")
print(f"Improvement vs V43: {8.58561 - oof_rmse:+.5f}")

# Save submission
sub_df = pd.read_csv(test_file, usecols=['id'])
sub_df['exam_score'] = sub_stack
sub_df.to_csv("submission_v47_stack.csv", index=False)

# Save OOF
oof_df = pd.DataFrame({'id': train_df['id'], 'oof_pred': oof_stack})
oof_df.to_csv("oof_v47_stack.csv", index=False)

print(f"\nSaved: submission_v47_stack.csv, oof_v47_stack.csv")
