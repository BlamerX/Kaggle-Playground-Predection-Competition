"""
S6E1 V122 - Extended HillClimber Ensemble (7 Models)
=====================================================
Adding more diverse models for lower error correlation:
  - V110: CatBoost DART 5-seed (8.54708 LB) - BEST
  - V101: XGBoost Multi-KD (8.54860 LB)
  - V105: TabM Multi-KD (8.54963 LB)
  - V70: FTT (8.56168 LB) - Different architecture
  - V67: LightGBM (8.57986 LB) - Different algorithm
  - V77: CatBoost Base (8.56133 LB) - Reference
  - V73: XGBoost Base (8.56137 LB) - Reference

Goal: Lower error correlation = better ensemble
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
start_time = time.time()

print("="*80)
print("S6E1 V122 - Extended HillClimber Ensemble (7 Models)")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"

TARGET = "exam_score"
y = train_df[TARGET].values

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    oof_vals = oof[col].values
    sub_vals = sub['exam_score'].values
    rmse = np.sqrt(mean_squared_error(y, oof_vals))
    print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
    return oof_vals, sub_vals

print("\nLoading OOF files...")
models = {}
# Best singles
models['V110'] = load_oof("V110 (CatBoost DART Best)", "oof_v110.csv", "submission_v110.csv")
models['V101'] = load_oof("V101 (XGBoost Multi-KD)", "oof_v101.csv", "submission_v101.csv")
models['V105'] = load_oof("V105 (TabM Multi-KD)", "oof_v105.csv", "submission_v105.csv")
# Diverse models
models['V70'] = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
models['V67'] = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
models['V77'] = load_oof("V77 (CatBoost Base)", "oof_v77.csv", "submission_v77.csv")
models['V73'] = load_oof("V73 (XGBoost Base)", "oof_v73.csv", "submission_v73.csv")

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("ERROR CORRELATION ANALYSIS")
print("="*80)

error_df = pd.DataFrame({name: y - m[0] for name, m in models.items()})
error_corr = error_df.corr()

print("\nError Correlations (lower = better for ensemble):")
print(error_corr.round(4))

# Find lowest correlation pairs
corr_pairs = []
for i, n1 in enumerate(error_corr.columns):
    for j, n2 in enumerate(error_corr.columns):
        if i < j:
            corr_pairs.append((n1, n2, error_corr.loc[n1, n2]))
corr_pairs.sort(key=lambda x: x[2])

print("\nLowest Error Correlation Pairs (best for ensemble):")
for n1, n2, corr in corr_pairs[:5]:
    print(f"  {n1} + {n2}: {corr:.4f}")

# ============================================================================
# 3. HILLCLIMBER OPTIMIZATION
# ============================================================================

print(f"\n{'='*80}")
print("HILLCLIMBER OPTIMIZATION")
print("="*80)

model_names = list(models.keys())
n_models = len(model_names)
oof_matrix = np.column_stack([models[name][0] for name in model_names])
test_matrix = np.column_stack([models[name][1] for name in model_names])

def evaluate_weights(weights):
    weights = np.array(weights) / np.sum(weights)
    pred = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y, pred))

# Start with equal weights
best_weights = np.ones(n_models) / n_models
best_rmse = evaluate_weights(best_weights)
print(f"Equal weights RMSE: {best_rmse:.5f}")

# HillClimber with finer steps
n_iterations = 1000
step_sizes = [0.05, 0.02, 0.01, 0.005]  # Decreasing step sizes
improvement_count = 0

for step_size in step_sizes:
    print(f"\nStep size: {step_size}")
    for iteration in range(n_iterations // len(step_sizes)):
        for i in range(n_models):
            for direction in [-1, 1]:
                new_weights = best_weights.copy()
                new_weights[i] += direction * step_size
                new_weights = np.maximum(new_weights, 0)
                
                new_rmse = evaluate_weights(new_weights)
                if new_rmse < best_rmse:
                    best_weights = new_weights
                    best_rmse = new_rmse
                    improvement_count += 1
    print(f"  Current RMSE: {best_rmse:.5f}")

# Normalize final weights
best_weights = best_weights / np.sum(best_weights)

print(f"\nOptimization complete! {improvement_count} improvements found.")

# ============================================================================
# 4. RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("OPTIMAL WEIGHTS")
print("="*80)

for name, weight in zip(model_names, best_weights):
    if weight > 0.001:
        print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")
    else:
        print(f"  {name}: 0.0% (excluded)")

# Final predictions
final_oof = oof_matrix @ best_weights
final_test = test_matrix @ best_weights
final_rmse = np.sqrt(mean_squared_error(y, final_oof))

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"""
V121 (5 models): 8.55803 OOF
V122 (7 models): {final_rmse:.5f} OOF

V110 (Best Single): 8.55927 OOF → 8.54708 LB

V122 vs V110: {8.55927 - final_rmse:+.5f} improvement
V122 vs V121: {8.55803 - final_rmse:+.5f} improvement
""")

# ============================================================================
# 5. SAVE
# ============================================================================

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v122.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v122.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"Files saved: submission_v122.csv, oof_v122.csv")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
