"""
S6E1 V138 - The Redemption Stack (Pure Ridge on Upgraded Base)
==============================================================
Analysis of V137 Failure:
V137 (LB 8.54681) failed to beat V128 (LB 8.54649) because it included V122 (HillClimber).
V122's OOF (8.55763) is BIASED because it was optimized on the OOF data itself.
Ridge saw this "fake" high accuracy and gave it 85% weight, inheriting the overfitting.

Strategy for V138:
1. Revert to V128 strategy: Pure RidgeCV Stacking.
2. Upgrade the Inputs (Base Models):
   - V128 used: [V123, V124, V125, V126, V127]
   - V138 uses:
     1. V110 (CatBoost DART): 8.54708 LB (Replaces V123 - stronger)
     2. V101 (Single Ensemble): 8.54860 LB (New addition - diversity)
     3. V125 (TabM): Strong Neural diversity
     4. V127 (FTT): Strong Transformer diversity
     5. V67 (LGB): 8.57986 LB (Replaces V126 - stronger)
     6. V124 (XGB): Strong XGB diversity

Hypothesis: 
Since Ridge(Stronger Models) > Ridge(Weaker Models), V138 should mathematically beat V128.
"""

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("S6E1 V138 - The Redemption Stack")
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

# Input Models (Pure OOFs only - NO Ensembles/HillClimbers!)
model_files = {
    'V110_CatDART': ('oof_v110.csv', 'submission_v110.csv'), # Best Pure Single
    'V101_Single':  ('oof_v101.csv', 'submission_v101.csv'), # Diversity Tree
    'V125_TabM':    ('oof_v125.csv', 'submission_v125.csv'), # Neural
    'V127_FTT':     ('oof_v127.csv', 'submission_v127.csv'), # Transformer
    'V67_LGB':      ('oof_v67.csv', 'submission_v67.csv'),   # Best LGB
    'V124_XGB':     ('oof_v124.csv', 'submission_v124.csv')   # Fast XGB
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
        rmse = np.sqrt(mean_squared_error(y_true, oof))
        print(f"  {name:<15}: OOF RMSE = {rmse:.5f}")

X_oof = np.column_stack(oofs)
X_sub = np.column_stack(subs)

# ============================================================
# 2. RIDGE STACKING (Like V128)
# ============================================================
print("\n[2] Training RidgeCV...")

# Use broader alpha range to find optimal regularization
alphas = np.logspace(-6, 6, 200) 
meta_model = RidgeCV(alphas=alphas, fit_intercept=True)
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

# Clip to valid range
final_oof = np.clip(final_oof, 0, 100)
final_sub = np.clip(final_sub, 0, 100)

final_rmse = np.sqrt(mean_squared_error(y_true, final_oof))
print(f"\nFinal V138 OOF: {final_rmse:.6f}")

# Sanity Check vs V128 (8.55846)
if final_rmse < 8.55846:
    print("✅ V138 beats V128 on OOF!")
else:
    print("⚠️ V138 OOF is higher than V128. Check correlations.")

pd.DataFrame({'id': train['id'], 'exam_score': final_oof}).to_csv("oof_v138.csv", index=False)
pd.DataFrame({'id': test['id'], 'exam_score': final_sub}).to_csv("submission_v138.csv", index=False)

print(f"\nSaved v138 files.")
print("="*80)
