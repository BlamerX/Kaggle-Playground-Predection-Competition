"""
S6E1 V92 - 4-Model Blend with Ultra-Fine Grid
==============================================
V91: 8.54881 LB (39% V86 + 37% V73 + 25% V70)

V92: Try 4-model blends + finer weight grid (0.01 step)
"""

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("="*80)
print("S6E1 V92 - 4-Model Blend with Ultra-Fine Grid")
print("="*80)

# Load data
if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_path = '/kaggle/input/playground-series-s6e1/train.csv'
    test_path = '/kaggle/input/playground-series-s6e1/test.csv'
    oof_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/'
    sub_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/'
else:
    train_path = "Dataset/train.csv"
    test_path = "Dataset/test.csv"
    oof_path = "Previous trained files/OOF/"
    sub_path = "Previous trained files/Submissions/"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
y_train = train_df['exam_score'].values
n_train, n_test = len(train_df), len(test_df)

# Best models for blending
models = {
    'V77': ('oof_v77.csv', 'submission_v77.csv'),      # CatBoost + Avg
    'V73': ('oof_v73.csv', 'submission_v73.csv'),      # XGBoost
    'V86': ('oof_v86.csv', 'submission_v86.csv'),      # CatBoost Triple
    'V70': ('oof_v70.csv', 'submission_v70.csv'),      # FTT
    'V61': ('oof_v61.csv', 'submission_v61.csv'),      # TabM
    'V79': ('oof_v79.csv', 'submission_v79.csv'),      # LightGBM
    'V75': ('oof_v75.csv', 'submission_v75.csv'),      # CatBoost TabM
    'V87': ('oof_v87.csv', 'submission_v87.csv'),      # Ridge Meta
}

print("\n📊 Loading models...")
oof_preds, test_preds, model_rmse = {}, {}, {}

for name, (oof_file, sub_file) in models.items():
    try:
        oof_df = pd.read_csv(oof_path + oof_file)
        sub_df = pd.read_csv(sub_path + sub_file)
        oof_col = 'exam_score' if 'exam_score' in oof_df.columns else 'oof_pred'
        oof_preds[name] = oof_df[oof_col].values
        test_preds[name] = sub_df['exam_score'].values
        model_rmse[name] = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"  ✅ {name}: OOF={model_rmse[name]:.5f}")
    except Exception as e:
        print(f"  ❌ {name}: {e}")

loaded = list(oof_preds.keys())

# ============================================================================
# START FROM V91'S BEST: V86 + V73 + V70
# ============================================================================

print("\n" + "="*80)
print("STARTING FROM V91 BEST BLEND")
print("="*80)

# V91 best result
best = {
    'models': ['V86', 'V73', 'V70'],
    'weights': [0.385, 0.367, 0.248],
    'rmse': 8.55948
}
print(f"V91 baseline: {best['models']} = {best['rmse']:.5f}")

# ============================================================================
# TRY ADDING 4TH MODEL
# ============================================================================

print("\n" + "="*80)
print("TESTING 4-MODEL BLENDS")
print("="*80)

base_models = ['V86', 'V73', 'V70']
add_candidates = [m for m in loaded if m not in base_models]

print(f"\nTrying to add: {add_candidates}")

for add_model in add_candidates:
    test_models = base_models + [add_model]
    
    # Scipy optimize to find best weights
    def rmse_obj(weights):
        blend = sum(w * oof_preds[m] for w, m in zip(weights, test_models))
        return np.sqrt(mean_squared_error(y_train, np.clip(blend, 0, 100)))
    
    n = len(test_models)
    result = minimize(
        rmse_obj, np.ones(n) / n,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * n,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    
    if result.fun < best['rmse']:
        best = {
            'models': test_models.copy(),
            'weights': result.x.tolist(),
            'rmse': result.fun
        }
        print(f"  ✅ +{add_model}: {result.fun:.5f} (improved!)")
    else:
        print(f"  ❌ +{add_model}: {result.fun:.5f} (no improvement)")

# ============================================================================
# ULTRA-FINE GRID SEARCH ON BEST COMBINATION
# ============================================================================

print("\n" + "="*80)
print(f"ULTRA-FINE TUNING: {best['models']}")
print("="*80)

def rmse_obj_final(weights):
    blend = sum(w * oof_preds[m] for w, m in zip(weights, best['models']))
    return np.sqrt(mean_squared_error(y_train, np.clip(blend, 0, 100)))

# Multiple restarts
best_result = minimize(
    rmse_obj_final, np.array(best['weights']),
    method='SLSQP',
    bounds=[(0.0, 1.0)] * len(best['models']),
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    options={'maxiter': 5000}
)

for _ in range(10):  # Random restarts
    init = np.random.dirichlet(np.ones(len(best['models'])))
    result = minimize(
        rmse_obj_final, init,
        method='SLSQP',
        bounds=[(0.0, 1.0)] * len(best['models']),
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    if result.fun < best_result.fun:
        best_result = result

opt_weights = best_result.x
opt_rmse = best_result.fun

print(f"\nBefore fine-tuning: {best['rmse']:.5f}")
print(f"After fine-tuning:  {opt_rmse:.5f}")

print(f"\n🏆 Final Weights:")
for m, w in zip(best['models'], opt_weights):
    print(f"   {m}: {w:.4f}")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING")
print("="*80)

final_oof = np.clip(sum(w * oof_preds[m] for w, m in zip(opt_weights, best['models'])), 0, 100)
final_test = np.clip(sum(w * test_preds[m] for w, m in zip(opt_weights, best['models'])), 0, 100)
final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))

print(f"\n| Version | OOF RMSE | LB Score |")
print(f"|---------|----------|----------|")
print(f"| V91 | 8.55948 | 8.54881 |")
print(f"| **V92** | **{final_rmse:.5f}** | **?** |")

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v92.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v92.csv", index=False)

print(f"\n✅ Saved: submission_v92.csv, oof_v92.csv")
print(f"🏆 V92 OOF RMSE: {final_rmse:.5f}")
print("="*80)
