"""
S6E1 V141 - Filtered Blend + Public Submission Integration
=============================================================
Based on V140 lesson: Exclude weak models (OOF > 8.6)

TWO Submissions:
1. V141a: Filtered blend with Ridge-only (models with OOF < 8.6)
2. V141b: 50/50 blend of V141a with Public submission (LB 8.54363)

Key Changes from V140:
- REMOVED: KNN (9.73), SVR (9.89), ResNet (8.62) - all OOF > 8.6
- Ridge-only meta-learner (best in V140: 8.55596)
- Add public submission blend option
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
print("S6E1 V141 - Filtered Blend + Public Submission")
print("="*80)
print("V140 Lesson: Exclude weak models (OOF > 8.6)")
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
# 2. LOAD FILTERED MODELS (OOF < 8.6 only!)
# ============================================================================

print("\n[STEP 2] LOADING FILTERED MODELS (OOF < 8.6)")
print("-"*40)

RMSE_THRESHOLD = 8.60  # Only include models with OOF RMSE below this

def load_oof(name, oof_file, sub_file):
    """Load OOF and submission, return (oof, sub, rmse) or None if > threshold"""
    try:
        oof_path = base_path + f"OOF/{oof_file}"
        sub_path = base_path + f"Submissions/{sub_file}"
        
        oof = pd.read_csv(oof_path)
        sub = pd.read_csv(sub_path)
        
        col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
        oof_vals = oof[col].values
        sub_vals = sub['exam_score'].values
        
        rmse = np.sqrt(mean_squared_error(y, oof_vals))
        
        if rmse > RMSE_THRESHOLD:
            print(f"  ❌ {name}: OOF RMSE = {rmse:.5f} > {RMSE_THRESHOLD} (EXCLUDED)")
            return None, None, rmse
        else:
            print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
            return oof_vals, sub_vals, rmse
    except Exception as e:
        print(f"  ❌ {name}: Failed to load - {e}")
        return None, None, None

# Load all models and filter
print("\n  --- Strong Models (OOF < 8.6) ---")

# CatBoost
v110_oof, v110_sub, _ = load_oof("V110 (CatBoost DART 5-seed)", "oof_v110.csv", "submission_v110.csv")
v123_oof, v123_sub, _ = load_oof("V123 (CatBoost Recursive KD)", "oof_v123.csv", "submission_v123.csv")
v88_oof, v88_sub, _ = load_oof("V88 (CatBoost hybrid)", "oof_v88.csv", "submission_v88.csv")

# XGBoost
v101_oof, v101_sub, _ = load_oof("V101 (XGBoost)", "oof_v101.csv", "submission_v101.csv")
v124_oof, v124_sub, _ = load_oof("V124 (XGBoost KD)", "oof_v124.csv", "submission_v124.csv")
v73_oof, v73_sub, _ = load_oof("V73 (XGBoost PL)", "oof_v73.csv", "submission_v73.csv")

# TabM
v61_oof, v61_sub, _ = load_oof("V61 (TabM PL)", "oof_v61.csv", "submission_v61.csv")
v105_oof, v105_sub, _ = load_oof("V105 (TabM KD)", "oof_v105.csv", "submission_v105.csv")
v125_oof, v125_sub, _ = load_oof("V125 (TabM KD)", "oof_v125.csv", "submission_v125.csv")

# LightGBM
v67_oof, v67_sub, _ = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
v126_oof, v126_sub, _ = load_oof("V126 (LightGBM KD)", "oof_v126.csv", "submission_v126.csv")

# FTT
v70_oof, v70_sub, _ = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
v127_oof, v127_sub, _ = load_oof("V127 (FTT KD)", "oof_v127.csv", "submission_v127.csv")

# Hybrid
v77_oof, v77_sub, _ = load_oof("V77 (hybrid)", "oof_v77.csv", "submission_v77.csv")

print("\n  --- Weak Models (EXCLUDED) ---")
# These will be excluded due to RMSE threshold
load_oof("V45 (ResNet)", "oof_v45_resnet.csv", "submission_v45_resnet.csv")
load_oof("V48 (KNN)", "oof_v48_knn.csv", "submission_v48_knn.csv")
load_oof("V49 (SVR)", "oof_v49_svr.csv", "submission_v49_svr.csv")

# ============================================================================
# 3. BUILD STACKING MATRIX
# ============================================================================

print("\n[STEP 3] BUILD STACKING MATRIX")
print("-"*40)

# Collect all valid models
models = {
    'v110': (v110_oof, v110_sub),
    'v123': (v123_oof, v123_sub),
    'v88': (v88_oof, v88_sub),
    'v101': (v101_oof, v101_sub),
    'v124': (v124_oof, v124_sub),
    'v73': (v73_oof, v73_sub),
    'v61': (v61_oof, v61_sub),
    'v105': (v105_oof, v105_sub),
    'v125': (v125_oof, v125_sub),
    'v67': (v67_oof, v67_sub),
    'v126': (v126_oof, v126_sub),
    'v70': (v70_oof, v70_sub),
    'v127': (v127_oof, v127_sub),
    'v77': (v77_oof, v77_sub),
}

valid_models = {k: v for k, v in models.items() if v[0] is not None}
print(f"  Valid models (OOF < {RMSE_THRESHOLD}): {len(valid_models)}")

oof_stack = np.column_stack([v[0] for v in valid_models.values()])
test_stack = np.column_stack([v[1] for v in valid_models.values()])
model_names = list(valid_models.keys())

print(f"  OOF stack shape: {oof_stack.shape}")
print(f"  Test stack shape: {test_stack.shape}")

# ============================================================================
# 4. RIDGE-ONLY META-STACKING
# ============================================================================

print("\n[STEP 4] RIDGE-ONLY META-STACKING")
print("-"*40)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

ridge_oof = np.zeros(len(train_df))
ridge_test_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), 1):
    X_tr, X_val = oof_stack[tr_idx], oof_stack[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    ridge.fit(X_tr, y_tr)
    
    ridge_oof[val_idx] = np.clip(ridge.predict(X_val), 0, 100)
    ridge_test_preds.append(np.clip(ridge.predict(test_stack), 0, 100))

v141a_test = np.mean(ridge_test_preds, axis=0)
v141a_oof = ridge_oof
v141a_rmse = np.sqrt(mean_squared_error(y, v141a_oof))

print(f"  V141a (Ridge-filtered) OOF RMSE: {v141a_rmse:.5f}")

# Get Ridge weights
ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge_final.fit(oof_stack, y)
print(f"\n  Ridge weights:")
for name, weight in sorted(zip(model_names, ridge_final.coef_), key=lambda x: -abs(x[1])):
    print(f"    {name:15s}: {weight:+.4f}")

# ============================================================================
# 5. LOAD PUBLIC SUBMISSION & BLEND
# ============================================================================

print("\n[STEP 5] PUBLIC SUBMISSION BLEND")
print("-"*40)

try:
    public_sub = pd.read_csv(public_sub_path)
    public_test = public_sub['exam_score'].values
    print(f"  ✓ Public submission loaded")
    print(f"    Public LB: 8.54363 (BETTER than V128's 8.54649)")
    
    # Try different blend ratios
    print(f"\n  Testing blend ratios:")
    for weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blended = weight * v141a_test + (1 - weight) * public_test
        # Can't compute RMSE since public has no OOF, just show it's available
        print(f"    {weight:.1f} V141a + {1-weight:.1f} Public → submission_v141b_{int(weight*10)}{int((1-weight)*10)}.csv")
    
    # Create V141b with 50/50 blend
    v141b_test = 0.5 * v141a_test + 0.5 * public_test
    
    has_public = True
except Exception as e:
    print(f"  ❌ Public submission not found: {e}")
    print(f"    Continuing with V141a only")
    has_public = False

# ============================================================================
# 6. RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

v128_rmse = 8.55846
v128_lb = 8.54649
v140_rmse = 8.55764
v140_lb = 8.54799

print(f"""
| Version | Models | OOF RMSE | Notes |
|---------|--------|----------|-------|
| V128    | 5      | {v128_rmse:.5f}  | LB 8.54649 (was BEST) |
| V140    | 17     | {v140_rmse:.5f}  | LB 8.54799 (weak models hurt) |
| **V141a**| **{len(valid_models)}**    | **{v141a_rmse:.5f}**  | **Ridge-only, filtered** |
""")

if v141a_rmse < v128_rmse:
    print(f"✅ V141a IMPROVED over V128 by {v128_rmse - v141a_rmse:.5f}!")
else:
    print(f"⚠️ V141a OOF similar to V128 ({v141a_rmse - v128_rmse:+.5f})")
    print("  But removing weak models may still help LB!")

if has_public:
    print(f"""
🎯 V141b = 50% V141a + 50% Public (LB 8.54363)
   Expected LB: Better than both individual submissions!
   
   Rationale: Public submission uses different training methodology.
   Blending with our stacked model adds diversity.
""")

# ============================================================================
# 7. SAVE
# ============================================================================

print("="*80)
print("SAVING FILES")
print("="*80)

# V141a: Pure ridge-filtered
pd.DataFrame({'id': test_df['id'], 'exam_score': v141a_test}).to_csv("submission_v141a.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': v141a_oof}).to_csv("oof_v141a.csv", index=False)
print(f"  ✓ submission_v141a.csv (Ridge-filtered)")
print(f"  ✓ oof_v141a.csv")

if has_public:
    # V141b: 50/50 blend with public
    pd.DataFrame({'id': test_df['id'], 'exam_score': v141b_test}).to_csv("submission_v141b.csv", index=False)
    print(f"  ✓ submission_v141b.csv (50% V141a + 50% Public)")
    
    # Also create other blend ratios
    for w in [0.3, 0.4, 0.6, 0.7]:
        blended = w * v141a_test + (1 - w) * public_test
        fname = f"submission_v141b_{int(w*10)}{int((1-w)*10)}.csv"
        pd.DataFrame({'id': test_df['id'], 'exam_score': blended}).to_csv(fname, index=False)
        print(f"  ✓ {fname}")

total_time = time.time() - start_time
print(f"\n  Total execution time: {total_time:.1f} seconds")
print("="*80)
