"""
S6E1 V146 - 3-Way Public Blend
==============================
Combines our best internal stack (V144a Ridge) with TWO strong public submissions.

Components:
1. V144a (Ridge on 9 Diverse Models):
   - Our Best (5): V110, V101, V105, V67, V70
   - Public NNs (4): DeepTables, ResNet, Trompt, LNN
   - Weights: Learned via Ridge on OOF (handles negative weights correctly)

2. Public Submission 1 (LB 8.54350):
   - File: Public Submissions/submission.csv
   - Strongest individual public file

3. Public Submission 2 (LB 8.54362):
   - File: Public Submissions/Public submission.csv
   - Previous baseline

Strategy: Weighted Average
Since we lack OOFs for Public 1 & 2, we use fixed weights based on trust/LB.
"""

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
start_time = time.time()

print("="*80)
print("S6E1 V146 - 3-Way Public Blend")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("  Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    public_path = '/kaggle/input/oof-and-submission/Season6episode1/Public Submissions/'
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"
    public_path = "Public Submissions/"

TARGET = "exam_score"
y = train_df[TARGET].values

# ============================================================================
# 2. GENERATE V144a (RIDGE STACK BASE)
# ============================================================================

print("\n[STEP 1] Generating V144a Base (Ridge on 9 Models)")
print("-"*40)

def load_for_stack(name, path, is_public=False):
    try:
        if is_public:
            if "lnn" in name:
                oof = pd.read_csv(path + "oof_lnn.csv.csv")
                sub = pd.read_csv(path + "sub_lnn.csv.csv")
                oof_vals = oof[oof.columns[0]].values
            else:
                oof = pd.read_csv(path + f"oof_{name}.csv")
                sub = pd.read_csv(path + f"sub_{name}.csv")
                col = 'oof_pred' if 'oof_pred' in oof.columns else 'exam_score'
                oof_vals = oof[col].values
            sub_vals = sub['exam_score'].values
        else:
            oof = pd.read_csv(base_path + f"OOF/oof_{name}.csv")
            sub = pd.read_csv(base_path + f"Submissions/submission_{name}.csv")
            col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
            oof_vals = oof[col].values
            sub_vals = sub['exam_score'].values
        return oof_vals, sub_vals
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None, None

models = {}
# Our Models
for m in ["v110", "v101", "v105", "v67", "v70"]:
    o, s = load_for_stack(m, base_path)
    if o is not None: models[m] = (o, s)

# Public NNs
for m in ["deeptables", "resnet", "trompt", "lnn"]:
    o, s = load_for_stack(m, public_path, is_public=True)
    if o is not None: models[m] = (o, s)

print(f"  Loaded {len(models)} base models for Ridge stacking")

# Ridge Stack
oof_stack = np.column_stack([m[0] for m in models.values()])
sub_stack = np.column_stack([m[1] for m in models.values()])

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge.fit(oof_stack, y)
v144a_test = np.clip(ridge.predict(sub_stack), 0, 100)
v144a_oof = np.clip(ridge.predict(oof_stack), 0, 100)
v144a_rmse = np.sqrt(mean_squared_error(y, v144a_oof))

print(f"  ✓ V144a Ridge RMSE: {v144a_rmse:.5f}")

# ============================================================================
# 3. LOAD PUBLIC SUBMISSIONS
# ============================================================================

print("\n[STEP 2] Loading Public Submissions")
print("-"*40)

# Public 1 (LB 8.54350)
try:
    pub1 = pd.read_csv(public_path + "submission.csv")
    pub1_test = pub1['exam_score'].values
    print("  ✓ Public 1 (8.54350) loaded")
except:
    print("  ✗ Public 1 (submission.csv) NOT FOUND - Using V144a")
    pub1_test = v144a_test

# Public 2 (LB 8.54362)
try:
    pub2 = pd.read_csv(public_path + "Public submission.csv")
    pub2_test = pub2['exam_score'].values
    print("  ✓ Public 2 (8.54362) loaded")
except:
    print("  ✗ Public 2 (Public submission.csv) NOT FOUND - Using V144a")
    pub2_test = v144a_test

# ============================================================================
# 4. CREATE BLENDS
# ============================================================================

print("\n[STEP 3] Creating Blends")
print("-"*40)

# Optimized Blends: 25% BASE FIXED (User Request)
# Remaining 75% split between Public 1 (8.54350) and Public 2 (8.54362)

# V146a: Standard Favor P1 (45/30)
# 25% Base + 45% P1 + 30% P2
v146a = 0.25 * v144a_test + 0.45 * pub1_test + 0.30 * pub2_test

# V146b: Aggressive Favor P1 (50/25)
# 25% Base + 50% P1 + 25% P2
v146b = 0.25 * v144a_test + 0.50 * pub1_test + 0.25 * pub2_test

# V146c: Balanced Publics (37.5/37.5)
# 25% Base + 37.5% P1 + 37.5% P2
v146c = 0.25 * v144a_test + 0.375 * pub1_test + 0.375 * pub2_test

# V146d: Super Aggressive P1 (55/20)
# 25% Base + 55% P1 + 20% P2
v146d = 0.25 * v144a_test + 0.55 * pub1_test + 0.20 * pub2_test

# ============================================================================
# 5. SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

def save_sub(pred, name, note):
    df = pd.DataFrame({'id': test_df['id'], 'exam_score': pred})
    df.to_csv(f"{name}.csv", index=False)
    print(f"  ✓ {name}.csv: {note}")

save_sub(v146a, "submission_v146a", "25% Base + 45% P1 + 30% P2")
save_sub(v146b, "submission_v146b", "25% Base + 50% P1 + 25% P2")
save_sub(v146c, "submission_v146c", "25% Base + 37.5% P1 + 37.5% P2")
save_sub(v146d, "submission_v146d", "25% Base + 55% P1 + 20% P2")

print(f"\n  Total time: {time.time() - start_time:.1f}s")
print("="*80)
