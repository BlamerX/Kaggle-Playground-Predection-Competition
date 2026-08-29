"""
================================================================================
S6E1 V40 - Ridge Stacking Ensemble with Hill Climbing
================================================================================
Combines top OOF predictions using Ridge regression + Hill Climbing optimization.

Primary Models (5):
  1. V28 - TabM (3-seed) - OOF: 8.597
  2. Stage3 XGB - XGBoost (5-seed Hybrid) - OOF: 8.601
  3. Stage3 FTT - FT-Transformer (3-seed) - OOF: 8.605
  4. Stage3 LGB - LightGBM (5-seed CPU) - OOF: 8.623
  5. Stage3 ResNet - Tabular ResNet (5-seed) - OOF: 8.621

Standby Models (added if correlation helps):
  6. V23 - XGBoost (different FE) - OOF: 8.607
  7. V27 - FT-Transformer (pytabkit) - OOF: 8.630
  8. Stage3 Tobit - Tobit XGBoost - OOF: 8.661
================================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

print("="*80)
print("S6E1 V40 - Ridge Stacking Ensemble with Hill Climbing")
print("="*80)

# Path configuration (Kaggle vs Local)
if os.path.exists("/kaggle/input/playground-series-s6e1/train.csv"):
    print("[LOG] Detected Kaggle Environment")
    TRAIN_PATH = "/kaggle/input/playground-series-s6e1/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s6e1/test.csv"
    # OOF files would need to be uploaded as dataset on Kaggle
    OOF_BASE = "/kaggle/input/s6e1-oof-predictions/"
    STAGE3_OOF = "/kaggle/input/s6e1-stage3-oof/"
elif os.path.exists("Previous trained files/OOF/oof_v28.csv"):
    print("[LOG] Detected Local Environment")
    TRAIN_PATH = "Dataset/train.csv"
    TEST_PATH = "Dataset/test.csv"
    OOF_BASE = "Previous trained files/OOF/"
    STAGE3_OOF = "Stage 3/OOF/"
    STAGE3_SUB = "Stage 3/Submission/"
    SUB_BASE = "Previous trained files/Submissions/"
else:
    print("[ERROR] Cannot find OOF files!")
    raise FileNotFoundError("OOF files not found")

# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("\n[LOG] Loading train data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "exam_score"
y_train = train_df[TARGET].values
n_train = len(y_train)
n_test = len(test_df)

print(f"Train samples: {n_train}")
print(f"Test samples: {n_test}")

# ============================================================================
# 3. DEFINE OOF SOURCES
# ============================================================================

# Primary models (5 core - MUST include)
PRIMARY_MODELS = {
    "V28_TabM": {
        "oof": OOF_BASE + "oof_v28.csv",
        "sub": SUB_BASE + "submission_v28.csv" if os.path.exists(SUB_BASE + "submission_v28.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.597
    },
    "S3_XGB": {
        "oof": STAGE3_OOF + "oof_stage3_xgb.csv",
        "sub": STAGE3_SUB + "submission_stage3_xgb.csv" if os.path.exists(STAGE3_SUB + "submission_stage3_xgb.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.601
    },
    "S3_FTT": {
        "oof": STAGE3_OOF + "oof_stage3_ftt.csv",
        "sub": STAGE3_SUB + "submission_stage3_ftt.csv" if os.path.exists(STAGE3_SUB + "submission_stage3_ftt.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.605
    },
    "S3_LGB": {
        "oof": STAGE3_OOF + "oof_stage3_lgb.csv",
        "sub": STAGE3_SUB + "submission_stage3_lgb.csv" if os.path.exists(STAGE3_SUB + "submission_stage3_lgb.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.623
    },
    "S3_ResNet": {
        "oof": STAGE3_OOF + "oof_stage3_resnet.csv",
        "sub": STAGE3_SUB + "submission_stage3_resnet.csv" if os.path.exists(STAGE3_SUB + "submission_stage3_resnet.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.621
    },
}

# Standby models (add if they help)
STANDBY_MODELS = {
    "V23_XGB": {
        "oof": OOF_BASE + "oof_v23.csv",
        "sub": SUB_BASE + "submission_v23.csv" if os.path.exists(SUB_BASE + "submission_v23.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.607
    },
    "V27_FTT": {
        "oof": OOF_BASE + "oof_v27_ftt.csv",
        "sub": SUB_BASE + "submission_v27_ftt.csv" if os.path.exists(SUB_BASE + "submission_v27_ftt.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.630
    },
    "S3_Tobit": {
        "oof": STAGE3_OOF + "oof_stage3_tobit.csv",
        "sub": STAGE3_SUB + "submission_stage3_tobit.csv" if os.path.exists(STAGE3_SUB + "submission_stage3_tobit.csv") else None,
        "oof_col": "oof_pred",
        "sub_col": "exam_score",
        "expected_oof": 8.661
    },
}

# ============================================================================
# 4. LOAD OOF PREDICTIONS
# ============================================================================

def load_oof(config, name):
    """Load OOF predictions from file."""
    try:
        df = pd.read_csv(config["oof"])
        col = config["oof_col"] if config["oof_col"] in df.columns else "exam_score"
        if col not in df.columns:
            col = df.columns[-1]  # Last column
        oof = df[col].values
        rmse = np.sqrt(mean_squared_error(y_train, oof))
        print(f"  ✓ {name}: OOF RMSE = {rmse:.5f} (expected: {config['expected_oof']:.3f})")
        return oof
    except Exception as e:
        print(f"  ✗ {name}: FAILED - {e}")
        return None

def load_sub(config, name):
    """Load test predictions from submission file."""
    try:
        if config["sub"] is None or not os.path.exists(config["sub"]):
            return None
        df = pd.read_csv(config["sub"])
        col = config["sub_col"] if config["sub_col"] in df.columns else "exam_score"
        return df[col].values
    except Exception as e:
        print(f"  ✗ {name} submission: FAILED - {e}")
        return None

print("\n" + "="*80)
print("Loading PRIMARY OOF Predictions...")
print("="*80)

oof_dict = {}
sub_dict = {}

for name, config in PRIMARY_MODELS.items():
    oof = load_oof(config, name)
    if oof is not None:
        oof_dict[name] = oof
        sub = load_sub(config, name)
        if sub is not None:
            sub_dict[name] = sub

print(f"\n[LOG] Loaded {len(oof_dict)} PRIMARY OOFs")

print("\n" + "="*80)
print("Loading STANDBY OOF Predictions...")
print("="*80)

standby_oof = {}
standby_sub = {}

for name, config in STANDBY_MODELS.items():
    oof = load_oof(config, name)
    if oof is not None:
        standby_oof[name] = oof
        sub = load_sub(config, name)
        if sub is not None:
            standby_sub[name] = sub

print(f"\n[LOG] Loaded {len(standby_oof)} STANDBY OOFs")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("OOF Correlation Matrix (Primary Models)")
print("="*80)

oof_names = list(oof_dict.keys())
oof_matrix = np.column_stack([oof_dict[name] for name in oof_names])
corr_matrix = np.corrcoef(oof_matrix.T)

print(f"\n{'':15}", end="")
for name in oof_names:
    print(f"{name[:8]:>10}", end="")
print()

for i, name in enumerate(oof_names):
    print(f"{name[:15]:15}", end="")
    for j in range(len(oof_names)):
        print(f"{corr_matrix[i,j]:10.4f}", end="")
    print()

# ============================================================================
# 6. RIDGE STACKING (CV)
# ============================================================================

print("\n" + "="*80)
print("Ridge Stacking (5-Fold CV)")
print("="*80)

kf = KFold(n_splits=5, shuffle=True, random_state=1003)
oof_stack = np.zeros(n_train)

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge.fit(oof_matrix, y_train)

print(f"\nRidge optimal alpha: {ridge.alpha_}")
print(f"Ridge coefficients:")
for name, coef in zip(oof_names, ridge.coef_):
    print(f"  {name}: {coef:.4f} ({coef/sum(ridge.coef_)*100:.1f}%)")

# CV prediction
for fold, (train_idx, val_idx) in enumerate(kf.split(oof_matrix), 1):
    ridge_fold = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge_fold.fit(oof_matrix[train_idx], y_train[train_idx])
    oof_stack[val_idx] = ridge_fold.predict(oof_matrix[val_idx])
    rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_stack[val_idx]))
    print(f"  Fold {fold}: RMSE = {rmse:.5f}")

ridge_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_stack))
print(f"\n[RIDGE] OOF RMSE: {ridge_oof_rmse:.5f}")

# ============================================================================
# 7. HILL CLIMBING OPTIMIZATION
# ============================================================================

print("\n" + "="*80)
print("Hill Climbing Weight Optimization")
print("="*80)

def objective(weights):
    """Minimize RMSE with given weights."""
    weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize
    blend = np.sum(oof_matrix * weights, axis=1)
    return np.sqrt(mean_squared_error(y_train, blend))

# Start with Ridge coefficients
initial_weights = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))
print(f"Initial weights (from Ridge): {initial_weights}")

# Optimize
result = minimize(
    objective, 
    initial_weights, 
    method='Nelder-Mead',
    options={'maxiter': 1000, 'xatol': 1e-8}
)

optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
print(f"Optimized weights: {optimal_weights}")

# Calculate OOF with optimal weights
oof_hill = np.sum(oof_matrix * optimal_weights, axis=1)
hill_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_hill))

print(f"\n[HILL CLIMBING] OOF RMSE: {hill_oof_rmse:.5f}")
print(f"[IMPROVEMENT] Ridge → Hill: {ridge_oof_rmse - hill_oof_rmse:.5f}")

# ============================================================================
# 8. CHOOSE BEST METHOD
# ============================================================================

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

# Simple average baseline
simple_avg = np.mean(oof_matrix, axis=1)
simple_rmse = np.sqrt(mean_squared_error(y_train, simple_avg))

print(f"| Method          | OOF RMSE  |")
print(f"|-----------------|-----------|")
print(f"| Simple Average  | {simple_rmse:.5f}  |")
print(f"| Ridge Stack     | {ridge_oof_rmse:.5f}  |")
print(f"| Hill Climbing   | {hill_oof_rmse:.5f}  |")

# Choose best
if hill_oof_rmse < ridge_oof_rmse:
    best_method = "Hill Climbing"
    best_weights = optimal_weights
    best_oof_rmse = hill_oof_rmse
else:
    best_method = "Ridge Stack"
    best_weights = np.abs(ridge.coef_) / np.sum(np.abs(ridge.coef_))
    best_oof_rmse = ridge_oof_rmse

print(f"\n[BEST] {best_method} with OOF RMSE: {best_oof_rmse:.5f}")

# ============================================================================
# 9. GENERATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("Generating Submission...")
print("="*80)

# Check if we have all submission files
missing_subs = [name for name in oof_names if name not in sub_dict]
if missing_subs:
    print(f"[WARNING] Missing submission files: {missing_subs}")
    print("[LOG] Cannot generate submission without all test predictions.")
else:
    # Stack test predictions
    test_matrix = np.column_stack([sub_dict[name] for name in oof_names])
    
    # Apply best weights
    test_blend = np.sum(test_matrix * best_weights, axis=1)
    
    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "exam_score": test_blend
    })
    
    # Save
    submission.to_csv("submission_v40.csv", index=False)
    print(f"✓ Saved submission_v40.csv")
    print(f"  Predictions range: [{test_blend.min():.2f}, {test_blend.max():.2f}]")

# Save OOF
oof_df = pd.DataFrame({
    "id": train_df["id"],
    "oof_pred": oof_hill if best_method == "Hill Climbing" else oof_stack
})
oof_df.to_csv("oof_v40.csv", index=False)
print(f"✓ Saved oof_v40.csv")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("V40 RIDGE STACKING COMPLETE")
print("="*80)

print(f"\nFinal Weights ({best_method}):")
for name, weight in zip(oof_names, best_weights):
    print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")

print(f"\nExpected Performance:")
print(f"  OOF RMSE: {best_oof_rmse:.5f}")
print(f"  Expected LB (OOF - 0.035): ~{best_oof_rmse - 0.035:.5f}")
print(f"  V33 Benchmark: OOF 8.590, LB 8.555")

print("\n" + "="*80)
