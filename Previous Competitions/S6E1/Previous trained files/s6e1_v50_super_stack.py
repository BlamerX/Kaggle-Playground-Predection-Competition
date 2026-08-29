"""
S6E1 V50 - Super Stack with Diversity Models
=============================================
Combines ALL models including KNN and SVR for maximum diversity:
- V28 TabM (DL)
- V34 XGBoost (GBDT)
- V44 FT-Transformer (DL)
- V45 ResNet (DL)
- V46 LightGBM (GBDT)
- V48 KNN (Diversity)
- V49 SVR (Diversity)

Uses RidgeCV to find optimal weights.
"""

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

print("="*80)
print("S6E1 V50 - Super Stack with Diversity Models")
print("="*80)

# ============================================================================
# 1. PATHS
# ============================================================================

if os.path.exists('/kaggle/input'):
    print("Environment: KAGGLE")
    oof_path = '/kaggle/input/'  # Update with correct dataset paths
    sub_path = '/kaggle/input/'
    train_path = '/kaggle/input/playground-series-s6e1/train.csv'
    test_path = '/kaggle/input/playground-series-s6e1/test.csv'
else:
    print("Environment: LOCAL")
    oof_path = "Previous trained files/OOF/"
    sub_path = "Previous trained files/Submissions/"
    train_path = "Dataset/train.csv"
    test_path = "Dataset/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGET = 'exam_score'
y_train = train_df[TARGET].values

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# ============================================================================
# 2. LOAD ALL OOF AND SUBMISSION FILES
# ============================================================================

print(f"\n{'='*80}")
print("LOADING OOF AND SUBMISSION FILES")
print("="*80)

# Model configurations
models = {
    'V28_TabM': {'oof': 'oof_v28.csv', 'sub': 'submission_v28.csv'},
    'V34_XGB': {'oof': 'oof_v34.csv', 'sub': 'submission_v34.csv'},
    'V44_FTT': {'oof': 'oof_v44_ftt.csv', 'sub': 'submission_v44_ftt.csv'},
    'V45_ResNet': {'oof': 'oof_v45_resnet.csv', 'sub': 'submission_v45_resnet.csv'},
    'V46_LGB': {'oof': 'oof_v46_lgb.csv', 'sub': 'submission_v46_lgb.csv'},
    'V48_KNN': {'oof': 'oof_v48_knn.csv', 'sub': 'submission_v48_knn.csv'},
    'V49_SVR': {'oof': 'oof_v49_svr.csv', 'sub': 'submission_v49_svr.csv'},
}

oof_preds = {}
test_preds = {}

for name, files in models.items():
    try:
        # Try local path first
        oof_file = oof_path + files['oof']
        sub_file = sub_path + files['sub']
        
        if os.path.exists(oof_file):
            oof_df = pd.read_csv(oof_file)
            sub_df = pd.read_csv(sub_file)
        else:
            # Try current directory
            oof_df = pd.read_csv(files['oof'])
            sub_df = pd.read_csv(files['sub'])
        
        # Handle different column names
        oof_col = 'oof_pred' if 'oof_pred' in oof_df.columns else 'exam_score'
        sub_col = 'exam_score' if 'exam_score' in sub_df.columns else sub_df.columns[-1]
        
        oof_preds[name] = oof_df[oof_col].values
        test_preds[name] = sub_df[sub_col].values
        
        oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"✅ {name}: OOF RMSE = {oof_rmse:.5f}")
        
    except Exception as e:
        print(f"❌ {name}: Failed to load - {e}")

print(f"\nLoaded {len(oof_preds)} models successfully")

if len(oof_preds) < 5:
    print("⚠️ Warning: Less than 5 models loaded. Check file paths.")

# ============================================================================
# 3. ANALYZE CORRELATIONS
# ============================================================================

print(f"\n{'='*80}")
print("CORRELATION ANALYSIS")
print("="*80)

oof_df = pd.DataFrame(oof_preds)
corr_matrix = oof_df.corr()

print("\nOOF Prediction Correlations:")
print(corr_matrix.round(3))

# Find average correlation for each model
avg_corr = corr_matrix.mean()
print("\nAverage correlation (higher = less diverse):")
for name in avg_corr.sort_values().index:
    print(f"  {name}: {avg_corr[name]:.4f}")

# ============================================================================
# 4. RIDGE STACKING
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE STACKING")
print("="*80)

# Prepare features
X_stack = np.column_stack([oof_preds[name] for name in oof_preds.keys()])
X_test_stack = np.column_stack([test_preds[name] for name in test_preds.keys()])

# RidgeCV
alphas = np.logspace(-3, 3, 20)
ridge = RidgeCV(alphas=alphas, cv=10, scoring='neg_root_mean_squared_error')
ridge.fit(X_stack, y_train)

print(f"\nRidge best alpha: {ridge.alpha_:.4f}")
print(f"\nModel weights (coefficients):")
for i, name in enumerate(oof_preds.keys()):
    print(f"  {name}: {ridge.coef_[i]:.4f}")
print(f"  Intercept: {ridge.intercept_:.4f}")

# OOF predictions
oof_stacked = ridge.predict(X_stack)
oof_stacked = np.clip(oof_stacked, 0, 100)

# Test predictions
test_stacked = ridge.predict(X_test_stack)
test_stacked = np.clip(test_stacked, 0, 100)

# Calculate RMSE
stack_rmse = np.sqrt(mean_squared_error(y_train, oof_stacked))
print(f"\n{'='*80}")
print(f"V50 SUPER STACK OOF RMSE: {stack_rmse:.5f}")
print("="*80)

# ============================================================================
# 5. COMPARE WITH BASELINES
# ============================================================================

print(f"\n{'='*80}")
print("COMPARISON WITH BASELINES")
print("="*80)

print(f"\n| Model | OOF RMSE |")
print(f"|-------|----------|")
for name in oof_preds.keys():
    rmse = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
    print(f"| {name} | {rmse:.5f} |")
print(f"| **V50 Super Stack** | **{stack_rmse:.5f}** |")
print(f"| V47 Clean Stack (baseline) | 8.58607 |")

improvement = 8.58607 - stack_rmse
print(f"\nImprovement vs V47: {improvement:+.5f}")

# ============================================================================
# 6. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

# Save submission
submission = pd.read_csv(test_path, usecols=['id'])
submission['exam_score'] = test_stacked
submission.to_csv("submission_v50_super_stack.csv", index=False)

# Save OOF
oof_out = pd.DataFrame({'id': train_df['id'], 'oof_pred': oof_stacked})
oof_out.to_csv("oof_v50_super_stack.csv", index=False)

print(f"Saved: submission_v50_super_stack.csv, oof_v50_super_stack.csv")
print(f"\nV50 Super Stack OOF RMSE: {stack_rmse:.5f}")
