"""
S6E1 V91 - Improved Weighted Blend (V90 achieved 8.54886!)
=============================================================
V90: 77% V77 + 23% V70 = 8.54886 LB (NEW BEST!)
V91: Try 3-model blends with finer weight grid

Goal: Beat 8.54886 with better blend
"""

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("="*80)
print("S6E1 V91 - Improved Weighted Blend")
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

# Diverse models for blending
models = {
    'V77': ('oof_v77.csv', 'submission_v77.csv'),      # 8.55149 LB - CatBoost
    'V73': ('oof_v73.csv', 'submission_v73.csv'),      # 8.56137 LB - XGBoost
    'V79': ('oof_v79.csv', 'submission_v79.csv'),      # 8.55752 LB - LightGBM
    'V61': ('oof_v61.csv', 'submission_v61.csv'),      # 8.56152 LB - TabM
    'V70': ('oof_v70.csv', 'submission_v70.csv'),      # 8.56168 LB - FTT
    'V86': ('oof_v86.csv', 'submission_v86.csv'),      # 8.55155 LB - CatBoost Triple
    'V75': ('oof_v75.csv', 'submission_v75.csv'),      # 8.55821 LB - CatBoost TabM
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
# GRID SEARCH: 2-model and 3-model blends
# ============================================================================

print("\n" + "="*80)
print("GRID SEARCH FOR BEST BLEND")
print("="*80)

best = {'models': ['V77'], 'weights': [1.0], 'rmse': model_rmse.get('V77', 9.0)}

# 2-model blends (fine grid)
print("\n🔍 Testing 2-model blends...")
for m1 in loaded:
    for m2 in loaded:
        if m1 >= m2:
            continue
        for w1 in np.arange(0.50, 0.95, 0.01):
            w2 = 1 - w1
            blend = w1 * oof_preds[m1] + w2 * oof_preds[m2]
            rmse = np.sqrt(mean_squared_error(y_train, np.clip(blend, 0, 100)))
            if rmse < best['rmse']:
                best = {'models': [m1, m2], 'weights': [w1, w2], 'rmse': rmse}

print(f"   Best 2-model: {best['models']} @ {[f'{w:.2f}' for w in best['weights']]} = {best['rmse']:.5f}")

# 3-model blends (fine grid)
print("\n🔍 Testing 3-model blends...")
for m1 in loaded:
    for m2 in loaded:
        if m1 >= m2:
            continue
        for m3 in loaded:
            if m2 >= m3:
                continue
            for w1 in np.arange(0.40, 0.80, 0.02):
                for w2 in np.arange(0.10, 0.40, 0.02):
                    w3 = 1 - w1 - w2
                    if w3 < 0.05 or w3 > 0.35:
                        continue
                    blend = w1 * oof_preds[m1] + w2 * oof_preds[m2] + w3 * oof_preds[m3]
                    rmse = np.sqrt(mean_squared_error(y_train, np.clip(blend, 0, 100)))
                    if rmse < best['rmse']:
                        best = {'models': [m1, m2, m3], 'weights': [w1, w2, w3], 'rmse': rmse}

print(f"   Best 3-model: {best['models']} @ {[f'{w:.2f}' for w in best['weights']]} = {best['rmse']:.5f}")

# ============================================================================
# SCIPY OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("SCIPY OPTIMIZATION")
print("="*80)

def rmse_obj(weights, models_list):
    blend = sum(w * oof_preds[m] for w, m in zip(weights, models_list))
    return np.sqrt(mean_squared_error(y_train, np.clip(blend, 0, 100)))

n = len(best['models'])
result = minimize(
    rmse_obj, np.array(best['weights']),
    args=(best['models'],),
    method='SLSQP',
    bounds=[(0.0, 1.0)] * n,
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
)

opt_weights = result.x
opt_rmse = rmse_obj(opt_weights, best['models'])

print(f"\nGrid search: {best['rmse']:.5f}")
print(f"Optimized:   {opt_rmse:.5f}")
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

print(f"\n| V90 (previous best) | 8.56020 OOF | 8.54886 LB |")
print(f"| **V91** | **{final_rmse:.5f} OOF** | **? LB** |")

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v91.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v91.csv", index=False)

print(f"\n✅ Saved: submission_v91.csv, oof_v91.csv")
print(f"🏆 V91 OOF RMSE: {final_rmse:.5f}")
print("="*80)
