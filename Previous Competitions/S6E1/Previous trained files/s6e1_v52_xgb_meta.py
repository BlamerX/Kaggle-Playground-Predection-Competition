"""
S6E1 V52 - Maximum OOF Ridge Stack (All Available Models)
==========================================================
Strategy: Include ALL available OOF files for maximum diversity.
Ridge will automatically zero out unhelpful models.

We have 46 OOF files - use them all!
"""

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import glob
import os

np.random.seed(42)

print("="*80)
print("S6E1 V52 - Maximum OOF Ridge Stack (All Available Models)")
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
n_train = len(train_df)
n_test = len(test_df)

print(f"Train samples: {n_train}")
print(f"Test samples: {n_test}")

# ============================================================================
# 2. LOAD ALL AVAILABLE OOF FILES
# ============================================================================

print(f"\n{'='*80}")
print("LOADING ALL OOF FILES")
print("="*80)

# All single models (not stacks) - selected based on good LB performance
all_models = {
    # TabM variants
    'V28_TabM': ('oof_v28.csv', 'submission_v28.csv'),
    'V30_TabM_5s': ('oof_v30_5seed_tabm.csv', 'submission_v30_5seed_tabm.csv'),
    'V25_TabM': ('oof_v25_tabm.csv', 'submission_v25_tabm.csv'),
    'V24_TabM': ('oof_v24_tabm.csv', 'submission_v24_tabm.csv'),
    'V19_TabM': ('oof_v19_tabm.csv', 'submission_v19_tabm.csv'),
    
    # XGBoost variants
    'V34_XGB': ('oof_v34.csv', 'submission_v34.csv'),
    'V32_XGB': ('oof_v32.csv', 'submission_v32.csv'),
    'V23_XGB': ('oof_v23.csv', 'submission_v23.csv'),
    'V29_XGB_3s': ('oof_v29_multiseed_xgb.csv', 'submission_v29_multiseed_xgb.csv'),
    'V31_XGB': ('oof_v31.csv', 'submission_v31.csv'),
    'V20_XGB': ('oof_v20.csv', 'submission_v20.csv'),
    'V22_XGB': ('oof_v22.csv', 'submission_v22.csv'),
    'V16_XGB': ('oof_v16.csv', 'submission_v16.csv'),
    'V13_XGB': ('oof_v13.csv', 'submission_v13.csv'),
    'V15_XGB': ('oof_v15.csv', 'submission_v15.csv'),
    'V12_XGB': ('oof_v12.csv', 'submission_v12.csv'),
    'V11_XGB': ('oof_v11.csv', 'submission_v11.csv'),
    'V10_XGB': ('oof_v10.csv', 'submission_v10.csv'),
    
    # FT-Transformer variants
    'V44_FTT': ('oof_v44_ftt.csv', 'submission_v44_ftt.csv'),
    'V27_FTT': ('oof_v27_ftt.csv', 'submission_v27_ftt.csv'),
    
    # ResNet
    'V45_ResNet': ('oof_v45_resnet.csv', 'submission_v45_resnet.csv'),
    
    # LightGBM variants
    'V46_LGB': ('oof_v46_lgb.csv', 'submission_v46_lgb.csv'),
    'V33_LGB': ('oof_v33_lgbm.csv', 'submission_v33_lgbm.csv'),
    
    # Stage 3 models (with Golden Features)
    'S3_XGB': ('oof_stage3_xgb.csv', 'submission_stage3_xgb.csv'),
    'S3_FTT': ('oof_stage3_ftt.csv', 'submission_stage3_ftt.csv'),
    'S3_LGB': ('oof_stage3_lgb.csv', 'submission_stage3_lgb.csv'),
    'S3_ResNet': ('oof_stage3_resnet.csv', 'submission_stage3_resnet.csv'),
    
    # Additional FTT seeds (for diversity)
    'S3_FTT_42': ('oof_stage3_ftt_seed42.csv', 'submission_stage3_ftt.csv'),
    'S3_FTT_1003': ('oof_stage3_ftt_seed1003.csv', 'submission_stage3_ftt.csv'),
    'S3_FTT_2024': ('oof_stage3_ftt_seed2024.csv', 'submission_stage3_ftt.csv'),
}

oof_preds = {}
test_preds = {}
loaded_models = []

for name, (oof_file, sub_file) in all_models.items():
    try:
        oof_full = oof_path + oof_file
        sub_full = sub_path + sub_file
        
        if os.path.exists(oof_full):
            oof_df = pd.read_csv(oof_full)
            sub_df = pd.read_csv(sub_full)
        else:
            continue  # Skip if not found
        
        oof_col = 'oof_pred' if 'oof_pred' in oof_df.columns else 'exam_score'
        sub_col = 'exam_score' if 'exam_score' in sub_df.columns else sub_df.columns[-1]
        
        # Verify correct size
        if len(oof_df) != n_train:
            print(f"⚠️ {name}: Wrong OOF size ({len(oof_df)} vs {n_train})")
            continue
        if len(sub_df) != n_test:
            print(f"⚠️ {name}: Wrong test size ({len(sub_df)} vs {n_test})")
            continue
        
        oof_preds[name] = oof_df[oof_col].values
        test_preds[name] = sub_df[sub_col].values
        
        oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"✅ {name}: OOF={oof_rmse:.5f}")
        loaded_models.append(name)
        
    except Exception as e:
        print(f"❌ {name}: {e}")

print(f"\nLoaded {len(loaded_models)} models")

# ============================================================================
# 3. RIDGE STACKING
# ============================================================================

print(f"\n{'='*80}")
print(f"RIDGE STACKING ({len(loaded_models)} models)")
print("="*80)

X_stack = np.column_stack([oof_preds[name] for name in loaded_models])
X_test_stack = np.column_stack([test_preds[name] for name in loaded_models])

# RidgeCV with broad alpha range
alphas = np.logspace(-4, 5, 100)
ridge = RidgeCV(alphas=alphas, cv=10, scoring='neg_root_mean_squared_error')
ridge.fit(X_stack, y_train)

print(f"\nRidge best alpha: {ridge.alpha_:.4f}")

# Count positive/negative weights
pos_weights = sum(1 for w in ridge.coef_ if w > 0.01)
neg_weights = sum(1 for w in ridge.coef_ if w < -0.01)
zero_weights = len(ridge.coef_) - pos_weights - neg_weights

print(f"\nWeight distribution: {pos_weights} positive, {neg_weights} negative, {zero_weights} near-zero")

print(f"\nTop 10 Model Weights:")
weight_dict = {name: ridge.coef_[i] for i, name in enumerate(loaded_models)}
sorted_weights = sorted(weight_dict.items(), key=lambda x: abs(x[1]), reverse=True)
for name, weight in sorted_weights[:10]:
    status = "✅" if weight > 0.01 else "❌" if weight < -0.01 else "⚠️"
    print(f"  {status} {name}: {weight:.4f}")

# OOF predictions
oof_stacked = ridge.predict(X_stack)
oof_stacked = np.clip(oof_stacked, 19.6, 100.0)

# Test predictions
test_stacked = ridge.predict(X_test_stack)
test_stacked = np.clip(test_stacked, 19.6, 100.0)

stack_rmse = np.sqrt(mean_squared_error(y_train, oof_stacked))

print(f"\n{'='*80}")
print(f"V52 MAX OOF STACK RMSE: {stack_rmse:.5f}")
print("="*80)

# ============================================================================
# 4. SCIPY GRADIENT-BASED OPTIMIZATION
# ============================================================================

print(f"\n{'='*80}")
print("SCIPY GRADIENT-BASED WEIGHT OPTIMIZATION")
print("="*80)

from scipy.optimize import minimize

def rmse_objective(weights, X, y):
    """RMSE objective function for optimization."""
    weights = np.array(weights)
    preds = X @ weights
    preds = np.clip(preds, 19.6, 100.0)
    return np.sqrt(mean_squared_error(y, preds))

# Initial weights (equal)
n_models = len(loaded_models)
initial_weights = np.ones(n_models) / n_models

# Bounds: allow small negative weights like Ridge
bounds = [(-0.3, 1.0) for _ in range(n_models)]

# Constraint: weights sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# Optimize using L-BFGS-B
result = minimize(
    rmse_objective,
    initial_weights,
    args=(X_stack, y_train),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'disp': False}
)

scipy_weights = result.x
scipy_oof = np.clip(X_stack @ scipy_weights, 19.6, 100.0)
scipy_test = np.clip(X_test_stack @ scipy_weights, 19.6, 100.0)
scipy_rmse = np.sqrt(mean_squared_error(y_train, scipy_oof))

print(f"Scipy SLSQP converged: {result.success}")
print(f"Scipy OOF RMSE: {scipy_rmse:.5f}")

# Show top weights
scipy_weight_dict = {name: scipy_weights[i] for i, name in enumerate(loaded_models)}
sorted_scipy = sorted(scipy_weight_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
print("\nTop 5 Scipy weights:")
for name, w in sorted_scipy:
    print(f"  {name}: {w:.4f}")

# ============================================================================
# 5. BAYESIAN MODEL AVERAGING (Simple approximation)
# ============================================================================

print(f"\n{'='*80}")
print("BAYESIAN MODEL AVERAGING")
print("="*80)

# Calculate model evidence (inversely proportional to OOF error)
model_errors = []
for name in loaded_models:
    err = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
    model_errors.append(err)

model_errors = np.array(model_errors)

# BMA weights: exp(-error^2 / 2*sigma^2) normalized
# Using temperature scaling
temperatures = [0.001, 0.01, 0.1, 1.0]
best_bma_rmse = float('inf')
best_bma_temp = None
best_bma_oof = None
best_bma_test = None

for temp in temperatures:
    # Softmax-like weighting
    log_weights = -model_errors / temp
    log_weights = log_weights - np.max(log_weights)  # Numerical stability
    bma_weights = np.exp(log_weights)
    bma_weights = bma_weights / bma_weights.sum()
    
    # Calculate BMA predictions
    bma_oof = np.zeros(n_train)
    bma_test = np.zeros(n_test)
    for i, name in enumerate(loaded_models):
        bma_oof += bma_weights[i] * oof_preds[name]
        bma_test += bma_weights[i] * test_preds[name]
    
    bma_oof = np.clip(bma_oof, 19.6, 100.0)
    bma_test = np.clip(bma_test, 19.6, 100.0)
    bma_rmse = np.sqrt(mean_squared_error(y_train, bma_oof))
    
    print(f"  Temperature {temp}: RMSE = {bma_rmse:.5f}")
    
    if bma_rmse < best_bma_rmse:
        best_bma_rmse = bma_rmse
        best_bma_temp = temp
        best_bma_oof = bma_oof
        best_bma_test = bma_test

print(f"\nBest BMA: temp={best_bma_temp}, RMSE={best_bma_rmse:.5f}")

# ============================================================================
# 6. FINAL COMPARISON
# ============================================================================

print(f"\n{'='*80}")
print("FINAL COMPARISON")
print("="*80)

# Simple average baseline
simple_avg_oof = np.clip(np.mean(X_stack, axis=1), 19.6, 100.0)
simple_avg_rmse = np.sqrt(mean_squared_error(y_train, simple_avg_oof))

print(f"\n| Method | OOF RMSE | vs Ridge |")
print(f"|--------|----------|----------|")
print(f"| Simple Average | {simple_avg_rmse:.5f} | {stack_rmse - simple_avg_rmse:+.5f} |")
print(f"| **Ridge (30 models)** | **{stack_rmse:.5f}** | baseline |")
print(f"| Scipy SLSQP | {scipy_rmse:.5f} | {stack_rmse - scipy_rmse:+.5f} |")
print(f"| Bayesian MA | {best_bma_rmse:.5f} | {stack_rmse - best_bma_rmse:+.5f} |")
print(f"| V51 (previous best) | 8.58486 | - |")

# Find best method
results = {
    'Ridge': (oof_stacked, test_stacked, stack_rmse),
    'Scipy': (scipy_oof, scipy_test, scipy_rmse),
    'BMA': (best_bma_oof, best_bma_test, best_bma_rmse)
}

best_method = min(results.keys(), key=lambda k: results[k][2])
best_oof, best_test, best_rmse = results[best_method]

print(f"\n🏆 BEST METHOD: {best_method} (RMSE: {best_rmse:.5f})")
improvement = 8.58486 - best_rmse
print(f"Improvement vs V51: {improvement:+.5f}")

# ============================================================================
# 7. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

print(f"Using: {best_method}")

submission = pd.read_csv(test_path, usecols=['id'])
submission['exam_score'] = best_test
submission.to_csv("submission_v52.csv", index=False)

oof_out = pd.DataFrame({'id': train_df['id'], 'oof_pred': best_oof})
oof_out.to_csv("oof_v52.csv", index=False)

print(f"Saved: submission_v52.csv, oof_v52.csv")
print(f"\nV52 Best ({best_method}) OOF RMSE: {best_rmse:.5f}")