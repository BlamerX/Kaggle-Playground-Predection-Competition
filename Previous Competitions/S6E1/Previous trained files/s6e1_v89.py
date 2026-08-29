"""
S6E1 V89 - Best Ensemble (Ridge + Scipy + BMA)
===============================================
Combines all best single models using multiple stacking strategies.
Uses latest best models: V77, V87, V73, V79, V61, V75, etc.

Strategies:
1. RidgeCV (learns optimal weights)
2. Scipy SLSQP (gradient-based optimization)
3. Bayesian Model Averaging (temperature-scaled)
"""

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("="*80)
print("S6E1 V89 - Best Ensemble (Ridge + Scipy + BMA)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_path = '/kaggle/input/playground-series-s6e1/train.csv'
    test_path = '/kaggle/input/playground-series-s6e1/test.csv'
    oof_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/'
    sub_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/'
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
# 2. AUTO-DISCOVER ALL SINGLE MODEL OOF FILES
# ============================================================================

import glob

print(f"\n{'='*80}")
print("AUTO-DISCOVERING ALL OOF FILES")
print("="*80)

# Exclusion list - stacked models (not single models)
EXCLUDE_PATTERNS = [
    'stack', 'blend', 'ensemble', 'meta', 'final',
    'v40', 'v41', 'v42', 'v43', 'v47', 'v51', 'v52', 'v53', 'v54',  # Previous stacks
    'v48', 'v49',  # Diversity models (KNN, SVR not for single use)
]

oof_preds = {}
test_preds = {}
loaded_models = []

# Get all OOF files
oof_files = glob.glob(oof_path + "oof_*.csv")
print(f"Found {len(oof_files)} OOF files")

for oof_full in sorted(oof_files):
    try:
        oof_filename = os.path.basename(oof_full)
        
        # Skip excluded patterns
        skip = False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in oof_filename.lower():
                skip = True
                break
        if skip:
            continue
        
        # Derive submission filename
        sub_filename = oof_filename.replace('oof_', 'submission_')
        sub_full = sub_path + sub_filename
        
        # Try alternate naming patterns
        if not os.path.exists(sub_full):
            # Try without version suffix variations
            base = oof_filename.replace('oof_', '').replace('.csv', '')
            alt_patterns = [
                f"submission_{base}.csv",
                f"submission{base}.csv",
            ]
            for alt in alt_patterns:
                if os.path.exists(sub_path + alt):
                    sub_full = sub_path + alt
                    break
        
        if not os.path.exists(sub_full):
            print(f"⚠️ {oof_filename}: No matching submission")
            continue
        
        oof_df = pd.read_csv(oof_full)
        sub_df = pd.read_csv(sub_full)
        
        oof_col = 'oof_pred' if 'oof_pred' in oof_df.columns else 'exam_score'
        sub_col = 'exam_score' if 'exam_score' in sub_df.columns else sub_df.columns[-1]
        
        # Verify correct size
        if len(oof_df) != n_train:
            print(f"⚠️ {oof_filename}: Wrong OOF size ({len(oof_df)})")
            continue
        if len(sub_df) != n_test:
            print(f"⚠️ {oof_filename}: Wrong test size ({len(sub_df)})")
            continue
        
        # Create model name from filename
        model_name = oof_filename.replace('oof_', '').replace('.csv', '').upper()
        
        oof_preds[model_name] = oof_df[oof_col].values
        test_preds[model_name] = sub_df[sub_col].values
        
        oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds[model_name]))
        print(f"✅ {model_name}: OOF={oof_rmse:.5f}")
        loaded_models.append(model_name)
        
    except Exception as e:
        print(f"❌ {oof_full}: {e}")

print(f"\n🎯 Loaded {len(loaded_models)} single models for stacking")

if len(loaded_models) < 3:
    print("❌ ERROR: Need at least 3 models for stacking!")
    exit(1)

# ============================================================================
# 4. RIDGE STACKING
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

print(f"\nModel Weights (sorted by importance):")
weight_dict = {name: ridge.coef_[i] for i, name in enumerate(loaded_models)}
sorted_weights = sorted(weight_dict.items(), key=lambda x: abs(x[1]), reverse=True)
for name, weight in sorted_weights:
    status = "✅" if weight > 0.01 else "❌" if weight < -0.01 else "⚠️"
    print(f"  {status} {name}: {weight:.4f}")

# OOF and test predictions
oof_ridge = np.clip(ridge.predict(X_stack), 0, 100)
test_ridge = np.clip(ridge.predict(X_test_stack), 0, 100)
ridge_rmse = np.sqrt(mean_squared_error(y_train, oof_ridge))

print(f"\nRidge OOF RMSE: {ridge_rmse:.5f}")

# ============================================================================
# 5. SCIPY OPTIMIZATION
# ============================================================================

print(f"\n{'='*80}")
print("SCIPY SLSQP OPTIMIZATION")
print("="*80)

def rmse_objective(weights, X, y):
    """RMSE objective function."""
    preds = np.clip(X @ weights, 0, 100)
    return np.sqrt(mean_squared_error(y, preds))

n_models = len(loaded_models)
initial_weights = np.ones(n_models) / n_models
bounds = [(-0.3, 1.0) for _ in range(n_models)]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(
    rmse_objective,
    initial_weights,
    args=(X_stack, y_train),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

scipy_weights = result.x
oof_scipy = np.clip(X_stack @ scipy_weights, 0, 100)
test_scipy = np.clip(X_test_stack @ scipy_weights, 0, 100)
scipy_rmse = np.sqrt(mean_squared_error(y_train, oof_scipy))

print(f"Scipy converged: {result.success}")
print(f"Scipy OOF RMSE: {scipy_rmse:.5f}")

# ============================================================================
# 6. BAYESIAN MODEL AVERAGING
# ============================================================================

print(f"\n{'='*80}")
print("BAYESIAN MODEL AVERAGING")
print("="*80)

model_errors = np.array([np.sqrt(mean_squared_error(y_train, oof_preds[name])) for name in loaded_models])

best_bma_rmse = float('inf')
best_bma_temp = None
best_bma_oof = None
best_bma_test = None

for temp in [0.001, 0.01, 0.1, 1.0]:
    log_weights = -model_errors / temp
    log_weights = log_weights - np.max(log_weights)
    bma_weights = np.exp(log_weights)
    bma_weights = bma_weights / bma_weights.sum()
    
    bma_oof = np.zeros(n_train)
    bma_test = np.zeros(n_test)
    for i, name in enumerate(loaded_models):
        bma_oof += bma_weights[i] * oof_preds[name]
        bma_test += bma_weights[i] * test_preds[name]
    
    bma_oof = np.clip(bma_oof, 0, 100)
    bma_test = np.clip(bma_test, 0, 100)
    bma_rmse = np.sqrt(mean_squared_error(y_train, bma_oof))
    
    print(f"  temp={temp}: RMSE = {bma_rmse:.5f}")
    
    if bma_rmse < best_bma_rmse:
        best_bma_rmse = bma_rmse
        best_bma_temp = temp
        best_bma_oof = bma_oof
        best_bma_test = bma_test

print(f"\nBest BMA: temp={best_bma_temp}, RMSE={best_bma_rmse:.5f}")

# ============================================================================
# 7. SIMPLE WEIGHTED AVERAGE (by inverse RMSE)
# ============================================================================

print(f"\n{'='*80}")
print("WEIGHTED AVERAGE (Inverse RMSE)")
print("="*80)

inv_rmse_weights = 1 / model_errors
inv_rmse_weights = inv_rmse_weights / inv_rmse_weights.sum()

oof_weighted = np.zeros(n_train)
test_weighted = np.zeros(n_test)
for i, name in enumerate(loaded_models):
    oof_weighted += inv_rmse_weights[i] * oof_preds[name]
    test_weighted += inv_rmse_weights[i] * test_preds[name]

oof_weighted = np.clip(oof_weighted, 0, 100)
test_weighted = np.clip(test_weighted, 0, 100)
weighted_rmse = np.sqrt(mean_squared_error(y_train, oof_weighted))

print(f"Weighted Avg OOF RMSE: {weighted_rmse:.5f}")

# ============================================================================
# 8. FINAL COMPARISON
# ============================================================================

print(f"\n{'='*80}")
print("FINAL COMPARISON")
print("="*80)

simple_avg_oof = np.clip(np.mean(X_stack, axis=1), 0, 100)
simple_avg_rmse = np.sqrt(mean_squared_error(y_train, simple_avg_oof))

print(f"\n| Method | OOF RMSE | vs V77 (8.56347) |")
print(f"|--------|----------|------------------|")
print(f"| Simple Average | {simple_avg_rmse:.5f} | {8.56347 - simple_avg_rmse:+.5f} |")
print(f"| Weighted Avg | {weighted_rmse:.5f} | {8.56347 - weighted_rmse:+.5f} |")
print(f"| **Ridge** | **{ridge_rmse:.5f}** | **{8.56347 - ridge_rmse:+.5f}** |")
print(f"| Scipy SLSQP | {scipy_rmse:.5f} | {8.56347 - scipy_rmse:+.5f} |")
print(f"| Bayesian MA | {best_bma_rmse:.5f} | {8.56347 - best_bma_rmse:+.5f} |")

# Find best method
results = {
    'Ridge': (oof_ridge, test_ridge, ridge_rmse),
    'Scipy': (oof_scipy, test_scipy, scipy_rmse),
    'BMA': (best_bma_oof, best_bma_test, best_bma_rmse),
    'Weighted': (oof_weighted, test_weighted, weighted_rmse)
}

best_method = min(results.keys(), key=lambda k: results[k][2])
best_oof, best_test, best_rmse = results[best_method]

print(f"\n🏆 BEST METHOD: {best_method} (OOF RMSE: {best_rmse:.5f})")

# ============================================================================
# 9. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

print(f"Using: {best_method}")

submission = pd.DataFrame({'id': test_df['id'], 'exam_score': best_test})
submission.to_csv("submission_v89.csv", index=False)

oof_out = pd.DataFrame({'id': train_df['id'], 'exam_score': best_oof})
oof_out.to_csv("oof_v89.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_v89.csv")
print(f"  oof_v89.csv")
print(f"\n🏆 V89 Best ({best_method}) OOF RMSE: {best_rmse:.5f}")
print("="*80)
