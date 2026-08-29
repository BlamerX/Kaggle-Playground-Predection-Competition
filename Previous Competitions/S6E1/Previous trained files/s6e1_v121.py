"""
S6E1 V121 - HillClimber Ensemble (5 Models)
============================================
Models:
  - V110: CatBoost DART 5-seed (8.54708 LB) - BEST
  - V111: CatBoost DART + Ridge (8.54725 LB)
  - V112: CatBoost DART + Binned (8.54724 LB)
  - V101: XGBoost Multi-KD (8.54860 LB)
  - V105: TabM Multi-KD (8.54963 LB)

Method: HillClimber optimization to find optimal blend weights
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
print("S6E1 V121 - HillClimber Ensemble (5 Models)")
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
models['V110'] = load_oof("V110 (CatBoost DART 5-seed)", "oof_v110.csv", "submission_v110.csv")
models['V111'] = load_oof("V111 (CatBoost DART + Ridge)", "oof_v111.csv", "submission_v111.csv")
models['V112'] = load_oof("V112 (CatBoost DART + Binned)", "oof_v112.csv", "submission_v112.csv")
models['V101'] = load_oof("V101 (XGBoost Multi-KD)", "oof_v101.csv", "submission_v101.csv")
models['V105'] = load_oof("V105 (TabM Multi-KD)", "oof_v105.csv", "submission_v105.csv")

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("CORRELATION ANALYSIS")
print("="*80)

# Create OOF DataFrame
oof_df = pd.DataFrame({name: m[0] for name, m in models.items()})
corr_matrix = oof_df.corr()

print("\nOOF Prediction Correlations:")
print(corr_matrix.round(4))

# Calculate error correlations (more important for ensembling)
error_df = pd.DataFrame({name: y - m[0] for name, m in models.items()})
error_corr = error_df.corr()

print("\nError Correlations (lower = better for ensemble):")
print(error_corr.round(4))

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
    """Calculate RMSE for given weights"""
    weights = np.array(weights) / np.sum(weights)  # Normalize
    pred = oof_matrix @ weights
    return np.sqrt(mean_squared_error(y, pred))

# Start with equal weights
best_weights = np.ones(n_models) / n_models
best_rmse = evaluate_weights(best_weights)
print(f"Equal weights RMSE: {best_rmse:.5f}")

# HillClimber iterations
n_iterations = 500
step_size = 0.02
improvement_count = 0

print(f"\nRunning {n_iterations} iterations...")
for iteration in range(n_iterations):
    # Try adjusting each weight
    for i in range(n_models):
        for direction in [-1, 1]:
            new_weights = best_weights.copy()
            new_weights[i] += direction * step_size
            new_weights = np.maximum(new_weights, 0)  # Keep weights >= 0
            
            new_rmse = evaluate_weights(new_weights)
            if new_rmse < best_rmse:
                best_weights = new_weights
                best_rmse = new_rmse
                improvement_count += 1
    
    if iteration % 100 == 99:
        print(f"  Iter {iteration+1}: RMSE = {best_rmse:.5f}")

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
    print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")

# Final predictions
final_oof = oof_matrix @ best_weights
final_test = test_matrix @ best_weights
final_rmse = np.sqrt(mean_squared_error(y, final_oof))

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"""
| Model | OOF RMSE | Weight |
|-------|----------|--------|
| V110 (Best Single) | 8.55927 | {best_weights[0]:.1%} |
| V111 | 8.55988 | {best_weights[1]:.1%} |
| V112 | 8.55999 | {best_weights[2]:.1%} |
| V101 | 8.55902 | {best_weights[3]:.1%} |
| V105 | 8.56176 | {best_weights[4]:.1%} |
| **V121 Ensemble** | **{final_rmse:.5f}** | 100% |

Best Single (V110): 8.55927 OOF → 8.54708 LB
V121 Ensemble: {final_rmse:.5f} OOF → ? LB

OOF Improvement: {8.55927 - final_rmse:+.5f}
""")

# ============================================================================
# 5. SAVE
# ============================================================================

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v121.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v121.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"Files saved: submission_v121.csv, oof_v121.csv")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
