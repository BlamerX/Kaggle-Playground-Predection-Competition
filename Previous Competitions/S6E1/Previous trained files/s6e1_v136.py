"""
S6E1 V136 - Clean Ensemble with Power Averaging
===============================================
Based on V135 finding: Genetic Programming/Autoencoder features caused overfitting (OOF -0.00069, LB +0.00048).
However, the *weights* assigned to specific models (V122, V101, V110) were very strong.

Strategy:
1. Load Top 6 Diversified Models (Selection based on V135 weights):
   - V122 (HillClimb 7-model): 29% weight in V135
   - V101 (Single Ensemble): 27% weight in V135
   - V128 (Ridge Stack): 17% weight in V135
   - V110 (CatBoost DART): 14% weight in V135
   - V124 (XGBoost KD): 8% weight in V135
   - V67  (LightGBM): Diversity

2. Optimization:
   - Use Nelder-Mead (Hill Climbing) to optimize weights directly for RMSE.
   - Use Power Averaging: pred = (w1*p1^p + w2*p2^p)^(1/p)
   - Test p=[1, 2, 4, 8] to see if emphasizing higher/lower predictions helps.

3. Goal: Beating 8.54649 LB (Current Best V128)
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

print("=" * 80)
print("S6E1 V136 - Clean Ensemble + Power Averaging")
print("=" * 80)

start_time = time.time()

# ============================================================
# 1. LOADING DATA & PREDICTIONS
# ============================================================
if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    BASE_DIR = '/kaggle/input/playground-series-s6e1/'
    OOF_DIR = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/'
    SUB_DIR = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/'
else:
    BASE_DIR = 'Dataset/'
    OOF_DIR = 'Previous trained files/OOF/'
    SUB_DIR = 'Previous trained files/Submissions/'

train = pd.read_csv(BASE_DIR + 'train.csv')
test = pd.read_csv(BASE_DIR + 'test.csv')
y_true = train['exam_score'].values

def load_preds(name, oof_name, sub_name):
    try:
        oof = pd.read_csv(f"{OOF_DIR}{oof_name}")
        sub = pd.read_csv(f"{SUB_DIR}{sub_name}")
        
        # Sort by ID
        if 'id' in oof.columns:
            oof = oof.sort_values('id').reset_index(drop=True)
        if 'id' in sub.columns:
            sub = sub.sort_values('id').reset_index(drop=True)
            
        # Extract prediction column
        oof_col = [c for c in oof.columns if c != 'id'][0]
        sub_col = [c for c in sub.columns if c != 'id'][0]
        
        return oof[oof_col].values, sub[sub_col].values
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None

# Load the "Magnificent Seven" - Our Cleanest, Strongest Assets
model_files = {
    'V122_HC': ('oof_v122.csv', 'submission_v122.csv'),      # Best Ensemble (HillClimb)
    'V101_Single': ('oof_v101.csv', 'submission_v101.csv'),  # Best Single-File Ensemble
    'V128_Ridge': ('oof_v128.csv', 'submission_v128.csv'),   # Best LB (Ridge)
    'V110_CatDART': ('oof_v110.csv', 'submission_v110.csv'), # Best Pure Single
    'V124_XGBKD': ('oof_v124.csv', 'submission_v124.csv'),   # Fast XGB KD
    'V67_LGB': ('oof_v67.csv', 'submission_v67.csv')         # Diversity LGB
}

oofs = {}
subs = {}

print("\nLoading Models:")
for name, (oof_f, sub_f) in model_files.items():
    oof, sub = load_preds(name, oof_f, sub_f)
    if oof is not None:
        oofs[name] = oof
        subs[name] = sub
        loss = np.sqrt(mean_squared_error(y_true, oof))
        print(f"  {name:<15}: OOF RMSE = {loss:.5f}")

model_names = list(oofs.keys())
X_oof = np.column_stack([oofs[n] for n in model_names])
X_sub = np.column_stack([subs[n] for n in model_names])

# ============================================================
# 2. POWER AVERAGING OPTIMIZATION
# ============================================================
print("\n[2] Optimizing Power Averaging Weights...")

def get_power_mean(X, weights, p):
    # Ensure weights sum to 1 and are positive
    w = np.array(weights)
    if np.sum(w) == 0: return np.zeros(len(X))
    w = w / np.sum(w)
    
    # Power mean: (sum(w * x^p))^(1/p)
    # Clip X to be positive for fractional powers
    X_safe = np.maximum(X, 1e-6)
    
    weighted_pow_sum = np.dot(X_safe**p, w)
    return weighted_pow_sum**(1/p)

def objective(weights, p):
    pred = get_power_mean(X_oof, weights, p)
    return np.sqrt(mean_squared_error(y_true, pred))

# Test different powers
powers = [1.0, 1.5, 2.0, 3.0, 4.0]
results = []

for p in powers:
    init_w = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    res = minimize(objective, init_w, args=(p,), method='SLSQP', bounds=bounds, constraints=cons)
    
    best_rmse = res.fun
    print(f"  Power p={p:<3}: OOF RMSE = {best_rmse:.6f}")
    results.append((best_rmse, p, res.x))

# Select best power
results.sort(key=lambda x: x[0])
best_rmse, best_p, best_w = results[0]

print(f"\n🏆 Best Configuration: Power p={best_p}")
print(f"   Best OOF RMSE: {best_rmse:.6f}")
print("   Weights:")
for name, w in zip(model_names, best_w):
    print(f"     {name:<15}: {w:.4f}")

# ============================================================
# 3. GENERATE SUBMISSION
# ============================================================
final_oof = get_power_mean(X_oof, best_w, best_p)
final_sub = get_power_mean(X_sub, best_w, best_p)

# Sanity Check OOF
final_rmse = np.sqrt(mean_squared_error(y_true, final_oof))
print(f"\nFinal Calculated OOF: {final_rmse:.6f}")

# Save
pd.DataFrame({'id': train['id'], 'exam_score': final_oof}).to_csv("oof_v136.csv", index=False)
pd.DataFrame({'id': test['id'], 'exam_score': final_sub}).to_csv("submission_v136.csv", index=False)

print(f"\nSaved v136 files. Total time: {(time.time()-start_time)/60:.1f} min")
print("="*80)
