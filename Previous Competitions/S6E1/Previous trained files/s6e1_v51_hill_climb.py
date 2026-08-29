"""
S6E1 V51 - Diverse Ridge Stack (Extended Model Selection)
===========================================================
Based on research: Ridge is optimal for linear blending.
Strategy: Add more DIVERSE models with good LB scores.

Key insight: We need models with DIFFERENT feature engineering,
not just the best single models which are highly correlated.

Selected Models (based on public_scores.md analysis):
- TabM variants: V28 (best), V30 (5-seed)
- XGB variants: V34 (best), V23 (different FE), V32 (1-seed)
- FTT variants: V44 (best), V27 (different implementation)
- ResNet: V45 (best)
- LGB: V46 (best), V33 (different)
- Stage 3 models: S3_XGB, S3_FTT (Golden features - different!)
"""

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("="*80)
print("S6E1 V51 - Diverse Ridge Stack (Extended Model Selection)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

if os.path.exists('/kaggle/input'):
    print("Environment: KAGGLE")
    train_path = '/kaggle/input/playground-series-s6e1/train.csv'
    test_path = '/kaggle/input/playground-series-s6e1/test.csv'
    oof_path = ''
    sub_path = ''
else:
    print("Environment: LOCAL")
    train_path = "Dataset/train.csv"
    test_path = "Dataset/test.csv"
    oof_path = "Previous trained files/OOF/"
    sub_path = "Previous trained files/Submissions/"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGET = 'exam_score'
y_train = train_df[TARGET].values

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# ============================================================================
# 2. EXTENDED MODEL SELECTION
# ============================================================================

print(f"\n{'='*80}")
print("LOADING DIVERSE MODELS")
print("="*80)

# Strategically selected models for diversity
# Format: name -> (oof_file, sub_file, LB_score, notes)
model_configs = {
    # Core best models (NO Golden)
    'V28_TabM': ('oof_v28.csv', 'submission_v28.csv', 8.56178, 'Best TabM'),
    'V34_XGB': ('oof_v34.csv', 'submission_v34.csv', 8.56352, 'Best XGB'),
    'V44_FTT': ('oof_v44_ftt.csv', 'submission_v44_ftt.csv', 8.56179, 'Best FTT'),
    'V45_ResNet': ('oof_v45_resnet.csv', 'submission_v45_resnet.csv', 8.57707, 'Best ResNet'),
    'V46_LGB': ('oof_v46_lgb.csv', 'submission_v46_lgb.csv', 8.58266, 'Best LGB'),
    
    # Additional TabM variants (different configs)
    'V30_TabM_5seed': ('oof_v30_5seed_tabm.csv', 'submission_v30_5seed_tabm.csv', 8.56231, '5-seed variant'),
    
    # Additional XGB variants (different FE)
    'V23_XGB': ('oof_v23.csv', 'submission_v23.csv', 8.56367, 'CMT + params'),
    'V32_XGB': ('oof_v32.csv', 'submission_v32.csv', 8.56355, '1-seed baseline'),
    
    # Different FTT implementation
    'V27_FTT': ('oof_v27_ftt.csv', 'submission_v27_ftt.csv', 8.56507, 'pytabkit FTT'),
    
    # Stage 3 models (WITH Golden Features = different!)
    'S3_XGB': ('oof_stage3_xgb.csv', 'submission_stage3_xgb.csv', 8.56393, 'Golden FE'),
    'S3_FTT': ('oof_stage3_ftt.csv', 'submission_stage3_ftt.csv', 8.56379, 'Golden FE'),
    'S3_LGB': ('oof_stage3_lgb.csv', 'submission_stage3_lgb.csv', 8.58278, 'Golden FE'),
}

oof_preds = {}
test_preds = {}
loaded_models = []

for name, (oof_file, sub_file, expected_lb, notes) in model_configs.items():
    try:
        oof_full = oof_path + oof_file
        sub_full = sub_path + sub_file
        
        if os.path.exists(oof_full):
            oof_df = pd.read_csv(oof_full)
            sub_df = pd.read_csv(sub_full)
        else:
            oof_df = pd.read_csv(oof_file)
            sub_df = pd.read_csv(sub_file)
        
        oof_col = 'oof_pred' if 'oof_pred' in oof_df.columns else 'exam_score'
        sub_col = 'exam_score' if 'exam_score' in sub_df.columns else sub_df.columns[-1]
        
        oof_preds[name] = oof_df[oof_col].values
        test_preds[name] = sub_df[sub_col].values
        
        oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"✅ {name}: OOF={oof_rmse:.5f}, LB={expected_lb:.5f} - {notes}")
        loaded_models.append(name)
        
    except Exception as e:
        print(f"❌ {name}: {e}")

print(f"\nLoaded {len(loaded_models)} models")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("CORRELATION ANALYSIS")
print("="*80)

oof_df = pd.DataFrame(oof_preds)
corr_matrix = oof_df.corr()

# Find models with lowest average correlation (most diverse)
avg_corr = corr_matrix.mean().sort_values()
print("\nAverage Correlation (lower = more diverse):")
for name in avg_corr.index:
    print(f"  {name}: {avg_corr[name]:.4f}")

# ============================================================================
# 4. RIDGE STACKING
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE STACKING ({} models)".format(len(loaded_models)))
print("="*80)

X_stack = np.column_stack([oof_preds[name] for name in loaded_models])
X_test_stack = np.column_stack([test_preds[name] for name in loaded_models])

# RidgeCV with broad alpha range
alphas = np.logspace(-4, 4, 50)
ridge = RidgeCV(alphas=alphas, cv=10, scoring='neg_root_mean_squared_error')
ridge.fit(X_stack, y_train)

print(f"\nRidge best alpha: {ridge.alpha_:.4f}")
print(f"\nModel weights:")
for i, name in enumerate(loaded_models):
    weight = ridge.coef_[i]
    status = "✅" if weight > 0.01 else "⚠️" if weight > -0.01 else "❌"
    print(f"  {status} {name}: {weight:.4f}")
print(f"  Intercept: {ridge.intercept_:.4f}")

# OOF predictions
oof_stacked = ridge.predict(X_stack)
oof_stacked = np.clip(oof_stacked, 19.6, 100.0)

# Test predictions
test_stacked = ridge.predict(X_test_stack)
test_stacked = np.clip(test_stacked, 19.6, 100.0)

stack_rmse = np.sqrt(mean_squared_error(y_train, oof_stacked))

print(f"\n{'='*80}")
print(f"V51 DIVERSE STACK OOF RMSE: {stack_rmse:.5f}")
print("="*80)

# ============================================================================
# 5. COMPARISON
# ============================================================================

print(f"\n{'='*80}")
print("COMPARISON WITH PREVIOUS BEST")
print("="*80)

print(f"\n| Method | Models | OOF RMSE |")
print(f"|--------|--------|----------|")
print(f"| V47 Ridge (5 models) | 5 | 8.58607 |")
print(f"| V50 Super (7 models) | 7 | 8.58586 |")
print(f"| **V51 Diverse** | **{len(loaded_models)}** | **{stack_rmse:.5f}** |")

improvement = 8.58607 - stack_rmse
print(f"\nImprovement vs V47: {improvement:+.5f}")

# ============================================================================
# 6. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

submission = pd.read_csv(test_path, usecols=['id'])
submission['exam_score'] = test_stacked
submission.to_csv("submission_v51_diverse_stack.csv", index=False)

oof_out = pd.DataFrame({'id': train_df['id'], 'oof_pred': oof_stacked})
oof_out.to_csv("oof_v51_diverse_stack.csv", index=False)

print(f"Saved: submission_v51_diverse_stack.csv, oof_v51_diverse_stack.csv")
print(f"\nFinal V51 OOF RMSE: {stack_rmse:.5f}")
