"""
S6E1 V133 - Hill Climbing Ensemble Optimization
================================================
Goal: Find optimal blend weights using scipy.optimize
      instead of Ridge regression

V128 (current best): Ridge stack → 8.54649 LB
V133: Hill Climbing optimization

Models to blend:
- V123 (CatBoost) - 8.54676 LB
- V125 (TabM) - 8.54765 LB
- V127 (FTT) - 8.54783 LB
- V124 (XGBoost) - 8.54794 LB
- V126 (LightGBM) - 8.54899 LB
- V110 (CatBoost) - 8.54708 LB
- V122 (Ensemble) - 8.54693 LB
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("S6E1 V133 - Hill Climbing Ensemble Optimization")
print("=" * 80)

ON_KAGGLE = os.path.exists('/kaggle/input/')
print(f"Environment: {'KAGGLE' if ON_KAGGLE else 'LOCAL'}")

# ============================================================
# LOAD OOF PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("Loading OOF Predictions")
print("=" * 60)

if ON_KAGGLE:
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
else:
    base_path = 'Previous trained files/'
    train = pd.read_csv('Dataset/train.csv')
    test = pd.read_csv('Dataset/test.csv')

y_true = train['exam_score'].values

# Load OOF predictions
models = {
    'V123_CatBoost': 'oof_v123.csv',
    'V125_TabM': 'oof_v125.csv',
    'V127_FTT': 'oof_v127.csv',
    'V124_XGBoost': 'oof_v124.csv',
    'V126_LightGBM': 'oof_v126.csv',
    'V110_CatBoost': 'oof_v110.csv',
    'V122_Ensemble': 'oof_v122.csv'
}

oof_preds = {}
test_preds = {}

for name, oof_file in models.items():
    oof_df = pd.read_csv(base_path + f'OOF/{oof_file}')
    sub_df = pd.read_csv(base_path + f'Submissions/submission_{oof_file.replace("oof_", "").replace(".csv", ".csv")}')
    
    oof_df = oof_df.sort_values('id').reset_index(drop=True)
    sub_df = sub_df.sort_values('id').reset_index(drop=True)
    
    oof_col = 'exam_score' if 'exam_score' in oof_df.columns else 'oof_pred'
    oof_preds[name] = oof_df[oof_col].values
    test_preds[name] = sub_df['exam_score'].values
    
    oof_rmse = np.sqrt(mean_squared_error(y_true, oof_preds[name]))
    print(f"  {name:<20} OOF RMSE: {oof_rmse:.5f}")

# Stack OOF predictions
X_oof = np.column_stack(list(oof_preds.values()))
X_test = np.column_stack(list(test_preds.values()))

print(f"\nOOF matrix shape: {X_oof.shape}")
print(f"Test matrix shape: {X_test.shape}")

# ============================================================
# HILL CLIMBING OPTIMIZATION
# ============================================================
print("\n" + "=" * 60)
print("Hill Climbing Optimization")
print("=" * 60)

def objective(weights):
    """RMSE objective to minimize"""
    weights = np.array(weights)
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    pred = X_oof @ weights
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    return rmse

# Method 1: Nelder-Mead (from S4E4 winner)
print("\nMethod 1: Nelder-Mead Optimization")
initial_weights = np.ones(len(models)) / len(models)  # Equal weights
bounds = [(0, 1) for _ in range(len(models))]

result_nm = minimize(
    objective,
    initial_weights,
    method='Nelder-Mead',
    options={'maxiter': 10000, 'xatol': 1e-8}
)

weights_nm = result_nm.x / result_nm.x.sum()
oof_nm = X_oof @ weights_nm
test_nm = X_test @ weights_nm
rmse_nm = np.sqrt(mean_squared_error(y_true, oof_nm))

print(f"  OOF RMSE: {rmse_nm:.5f}")
print("  Weights:")
for name, w in zip(models.keys(), weights_nm):
    print(f"    {name:<20} {w:.4f}")

# Method 2: Differential Evolution (global optimizer)
print("\nMethod 2: Differential Evolution")

result_de = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=1000,
    atol=1e-8,
    tol=1e-8
)

weights_de = result_de.x / result_de.x.sum()
oof_de = X_oof @ weights_de
test_de = X_test @ weights_de
rmse_de = np.sqrt(mean_squared_error(y_true, oof_de))

print(f"  OOF RMSE: {rmse_de:.5f}")
print("  Weights:")
for name, w in zip(models.keys(), weights_de):
    print(f"    {name:<20} {w:.4f}")

# Method 3: Simple Ridge (V128 baseline)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_oof, y_true)
weights_ridge = ridge.coef_
weights_ridge = np.maximum(weights_ridge, 0)  # Non-negative
weights_ridge = weights_ridge / weights_ridge.sum()

oof_ridge = X_oof @ weights_ridge
test_ridge = X_test @ weights_ridge
rmse_ridge = np.sqrt(mean_squared_error(y_true, oof_ridge))

print(f"\nMethod 3: Ridge (V128 baseline)")
print(f"  OOF RMSE: {rmse_ridge:.5f}")
print("  Weights:")
for name, w in zip(models.keys(), weights_ridge):
    print(f"    {name:<20} {w:.4f}")

# ============================================================
# SELECT BEST METHOD
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

results = {
    'Nelder-Mead': (rmse_nm, weights_nm, test_nm),
    'Differential Evolution': (rmse_de, weights_de, test_de),
    'Ridge (V128)': (rmse_ridge, weights_ridge, test_ridge)
}

best_method = min(results.items(), key=lambda x: x[1][0])
print(f"\n{'Method':<25} {'OOF RMSE':<12} {'vs V128':<12}")
print("-" * 50)
for method, (rmse, _, _) in sorted(results.items(), key=lambda x: x[1][0]):
    delta = rmse - rmse_ridge
    marker = "🏆 BEST" if method == best_method[0] else ""
    print(f"{method:<25} {rmse:.5f}      {delta:+.5f}     {marker}")

# ============================================================
# SAVE BEST RESULT
# ============================================================
print("\n" + "=" * 60)
print(f"SAVING V133 (Best: {best_method[0]})")
print("=" * 60)

best_rmse, best_weights, best_test = best_method[1]

# Save submission
sub_df = pd.DataFrame({
    'id': test['id'],
    'exam_score': best_test
})
sub_df.to_csv('submission_v133.csv', index=False)
print("✅ Saved: submission_v133.csv")

# Save OOF
oof_df = pd.DataFrame({
    'id': train['id'],
    'exam_score': X_oof @ best_weights
})
oof_df.to_csv('oof_v133.csv', index=False)
print("✅ Saved: oof_v133.csv")

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"\n{'Model':<12} {'OOF RMSE':<12} {'LB Score':<12} {'Method':<20}")
print("-" * 56)
print(f"{'V128':<12} {rmse_ridge:.5f}      {'8.54649':<12} {'Ridge Stack':<20}")
print(f"{'V133':<12} {best_rmse:.5f}      {'???':<12} {best_method[0]:<20}")

improvement = rmse_ridge - best_rmse
print(f"\nImprovement: {improvement:+.5f} OOF RMSE")

if improvement > 0:
    print(f"✅ V133 improved over V128!")
    print(f"   Expected LB improvement: {improvement * 0.8:+.5f} (conservative)")
else:
    print(f"⚠️ V133 did not improve over V128")
    print(f"   Hill Climbing found similar weights to Ridge")

print("\n" + "=" * 60)
print("🎯 Ready for submission!")
print("=" * 60)
