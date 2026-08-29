"""
S6E1 V143 - Ridge-Only on 24 Models + Public Blend
====================================================
V142 showed: Ridge on 24 models = 8.54738 OOF (best!)
Tree-based meta-learners OVERFIT.

Strategy: 
- Use ONLY Ridge (no CatBoost/XGBoost/LightGBM)
- Blend with Public submission (LB 8.54363)

This is simpler than V142 but should avoid overfitting.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
start_time = time.time()

print("="*80)
print("S6E1 V143 - Ridge-Only on 24 Models + Public Blend")
print("="*80)
print("Goal: Simple Ridge (no tree-based overfit) + Public blend")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[STEP 1] DATA LOADING")
print("-"*40)

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("  Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    public_sub_path = '/kaggle/input/oof-and-submission/Season6episode1/Public submission.csv'
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"
    public_sub_path = "Public submission.csv"

TARGET = "exam_score"
y = train_df[TARGET].values

print(f"  Train: {len(train_df):,} rows")
print(f"  Test: {len(test_df):,} rows")

# ============================================================================
# 2. LOAD 24 HIGH-QUALITY OOFs (same as V142, no public-derived models)
# ============================================================================

print("\n[STEP 2] LOADING HIGH-QUALITY OOFs")
print("-"*40)

RMSE_THRESHOLD = 8.60

def load_oof(name, oof_file, sub_file):
    try:
        oof = pd.read_csv(base_path + f"OOF/{oof_file}")
        sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
        col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
        oof_vals = oof[col].values
        sub_vals = sub['exam_score'].values
        rmse = np.sqrt(mean_squared_error(y, oof_vals))
        if rmse > RMSE_THRESHOLD:
            return None, None, rmse, False
        return oof_vals, sub_vals, rmse, True
    except Exception as e:
        print(f"    Error loading {name}: {e}")
        return None, None, None, False

oof_dict = {}
sub_dict = {}

# Same 24 models as V142 (no models that used Public submission in training)
model_files = [
    # CatBoost
    ("v110", "oof_v110.csv", "submission_v110.csv"),
    ("v108", "oof_v108.csv", "submission_v108.csv"),
    ("v109", "oof_v109.csv", "submission_v109.csv"),
    ("v123", "oof_v123.csv", "submission_v123.csv"),
    ("v88", "oof_v88.csv", "submission_v88.csv"),
    ("v77", "oof_v77.csv", "submission_v77.csv"),
    # XGBoost
    ("v101", "oof_v101.csv", "submission_v101.csv"),
    ("v100", "oof_v100.csv", "submission_v100.csv"),
    ("v102", "oof_v102.csv", "submission_v102.csv"),
    ("v124", "oof_v124.csv", "submission_v124.csv"),
    ("v73", "oof_v73.csv", "submission_v73.csv"),
    ("v99", "oof_v99.csv", "submission_v99.csv"),
    # TabM
    ("v61", "oof_v61.csv", "submission_v61.csv"),
    ("v105", "oof_v105.csv", "submission_v105.csv"),
    ("v125", "oof_v125.csv", "submission_v125.csv"),
    # LightGBM
    ("v67", "oof_v67.csv", "submission_v67.csv"),
    ("v126", "oof_v126.csv", "submission_v126.csv"),
    # FTT
    ("v70", "oof_v70.csv", "submission_v70.csv"),
    ("v127", "oof_v127.csv", "submission_v127.csv"),
    # Other strong
    ("v106", "oof_v106.csv", "submission_v106.csv"),
    ("v107", "oof_v107.csv", "submission_v107.csv"),
    ("v111", "oof_v111.csv", "submission_v111.csv"),
    ("v112", "oof_v112.csv", "submission_v112.csv"),
    ("v113", "oof_v113.csv", "submission_v113.csv"),
]

for name, oof_f, sub_f in model_files:
    oof, sub, rmse, valid = load_oof(name, oof_f, sub_f)
    if valid:
        oof_dict[name] = oof
        sub_dict[name] = sub
        print(f"  ✓ {name}: {rmse:.5f}")

print(f"\n  Loaded {len(oof_dict)} models")

# ============================================================================
# 3. RIDGE STACKING (SIMPLE, NO TREE-BASED)
# ============================================================================

print("\n[STEP 3] RIDGE STACKING")
print("-"*40)

model_names = list(oof_dict.keys())
oof_stack = np.column_stack([oof_dict[m] for m in model_names])
test_stack = np.column_stack([sub_dict[m] for m in model_names])

print(f"  OOF stack shape: {oof_stack.shape}")

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

ridge_oof = np.zeros(len(train_df))
ridge_test = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), 1):
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    ridge.fit(oof_stack[tr_idx], y[tr_idx])
    ridge_oof[val_idx] = np.clip(ridge.predict(oof_stack[val_idx]), 0, 100)
    ridge_test.append(np.clip(ridge.predict(test_stack), 0, 100))

v143a_test = np.mean(ridge_test, axis=0)
v143a_oof = ridge_oof
v143a_rmse = np.sqrt(mean_squared_error(y, v143a_oof))

print(f"  V143a Ridge OOF: {v143a_rmse:.5f}")

# Get Ridge weights
ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge_final.fit(oof_stack, y)
print(f"\n  Ridge weights (top 10):")
weights = list(zip(model_names, ridge_final.coef_))
for name, w in sorted(weights, key=lambda x: -abs(x[1]))[:10]:
    print(f"    {name:10s}: {w:+.4f}")

# ============================================================================
# 4. BLEND WITH PUBLIC
# ============================================================================

print("\n[STEP 4] BLEND WITH PUBLIC")
print("-"*40)

try:
    public_sub = pd.read_csv(public_sub_path)
    public_test = public_sub['exam_score'].values
    print(f"  ✓ Public submission loaded (LB 8.54363)")
    
    # V143b: 30% Ridge + 70% Public (same as winning V141b_37 ratio)
    v143b_test = 0.3 * v143a_test + 0.7 * public_test
    
    # V143c: 25% Ridge + 75% Public
    v143c_test = 0.25 * v143a_test + 0.75 * public_test
    
    # V143d: 35% Ridge + 65% Public
    v143d_test = 0.35 * v143a_test + 0.65 * public_test
    
    has_public = True
except Exception as e:
    print(f"  ❌ Public not found: {e}")
    has_public = False

# ============================================================================
# 5. RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"""
| Version | Description | OOF RMSE |
|---------|-------------|----------|
| V141a   | Ridge on 14 models | 8.55716 |
| V142    | Ridge on 24 models | 8.54738 |
| **V143a**| **Ridge on 24 models (this)** | **{v143a_rmse:.5f}** |

V143a should match V142's Ridge (8.54738) since same data.

Blends to submit:
- V143b: 30% V143a + 70% Public
- V143c: 25% V143a + 75% Public
- V143d: 35% V143a + 65% Public

Compare to V141b_37 (LB 8.54336) which used 30% V141a + 70% Public.
V143 uses better base (24 models vs 14) so should be better!
""")

# ============================================================================
# 6. SAVE
# ============================================================================

print("="*80)
print("SAVING FILES")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': v143a_test}).to_csv("submission_v143a.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': v143a_oof}).to_csv("oof_v143a.csv", index=False)
print(f"  ✓ submission_v143a.csv (pure Ridge)")
print(f"  ✓ oof_v143a.csv")

if has_public:
    pd.DataFrame({'id': test_df['id'], 'exam_score': v143b_test}).to_csv("submission_v143b.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': v143c_test}).to_csv("submission_v143c.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': v143d_test}).to_csv("submission_v143d.csv", index=False)
    print(f"  ✓ submission_v143b.csv (30% V143a + 70% Public)")
    print(f"  ✓ submission_v143c.csv (25% V143a + 75% Public)")
    print(f"  ✓ submission_v143d.csv (35% V143a + 65% Public)")

total_time = time.time() - start_time
print(f"\n  Total time: {total_time:.1f} seconds")
print("="*80)
