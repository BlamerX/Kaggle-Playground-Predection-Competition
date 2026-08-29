"""
S6E1 V137 - Regularized Clean Ensemble
======================================
User Insight: V128 (Ridge Stack) beats V135 (Hill Climb) on LB despite worse OOF.
Conclusion: Hill Climbing overfits to OOF. Ridge Regularization is the key to LB generalization.

Strategy:
1. TAKE the "Magnifient Six" from V136 (V122, V101, V110, V128, V124, V67).
2. BUT use RidgeCV Stacking instead of Hill Climbing.
   - Ridge penalizes large weights, preventing overfitting to specific models.
   - This combines the *improved model selection* of V136 with the *proven regularization* of V128.

Goal: Beat 8.54649 LB with a regularized blend of our best assets.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("S6E1 V137 - Regularized Clean Ensemble (Ridge)")
print("=" * 80)

# ============================================================
# 1. LOADING DATA
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
            
        oof_col = [c for c in oof.columns if c != 'id'][0]
        sub_col = [c for c in sub.columns if c != 'id'][0]
        
        return oof[oof_col].values, sub[sub_col].values
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None

# Load the "Magnificent Six" (Same as V136)
model_files = {
    'V122_HC': ('oof_v122.csv', 'submission_v122.csv'),      # Best Ensemble (HillClimb)
    'V101_Single': ('oof_v101.csv', 'submission_v101.csv'),  # Best Single-File Ensemble
    'V128_Ridge': ('oof_v128.csv', 'submission_v128.csv'),   # Best LB (Ridge)
    'V110_CatDART': ('oof_v110.csv', 'submission_v110.csv'), # Best Pure Single
    'V124_XGBKD': ('oof_v124.csv', 'submission_v124.csv'),   # Fast XGB KD
    'V67_LGB': ('oof_v67.csv', 'submission_v67.csv')         # Diversity LGB
}

oofs = []
subs = []
model_names = []

print("\nLoading Models:")
for name, (oof_f, sub_f) in model_files.items():
    oof, sub = load_preds(name, oof_f, sub_f)
    if oof is not None:
        oofs.append(oof)
        subs.append(sub)
        model_names.append(name)
        print(f"  {name:<15}: OOF RMSE = {np.sqrt(mean_squared_error(y_true, oof)):.5f}")

X_oof = np.column_stack(oofs)
X_sub = np.column_stack(subs)

# ============================================================
# 2. RIDGE REGULARIZATION (The Fix for Overfitting)
# ============================================================
print("\n[2] Training RidgeCV Meta-Learner...")

# Use broader alpha range to find optimal regularization
alphas = np.logspace(-6, 6, 200) 
meta_model = RidgeCV(alphas=alphas, fit_intercept=True) # Intercept handles bias
meta_model.fit(X_oof, y_true)

print(f"  Best Alpha: {meta_model.alpha_:.6f}")
print("  Coefficients:")
for name, coef in zip(model_names, meta_model.coef_):
    print(f"    {name:<15}: {coef:.4f}")
print(f"    Intercept      : {meta_model.intercept_:.4f}")

# ============================================================
# 3. PREDICTION & SAVING
# ============================================================
final_oof = meta_model.predict(X_oof)
final_sub = meta_model.predict(X_sub)

# Clip predictions to valid range (Ridge is unconstrained)
final_oof = np.clip(final_oof, 0, 100)
final_sub = np.clip(final_sub, 0, 100)

final_rmse = np.sqrt(mean_squared_error(y_true, final_oof))
print(f"\nFinal Regularized OOF: {final_rmse:.6f}")

pd.DataFrame({'id': train['id'], 'exam_score': final_oof}).to_csv("oof_v137.csv", index=False)
pd.DataFrame({'id': test['id'], 'exam_score': final_sub}).to_csv("submission_v137.csv", index=False)

print(f"\nSaved v137 files (Regularized).")
print("="*80)
