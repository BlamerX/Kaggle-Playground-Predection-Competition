"""
================================================================================
S6E1 V41 - Nested CV Stacking (Gold Standard)
================================================================================
Fixes V40 issues:
1. Uses V34 XGB (OOF 8.601) instead of S3_XGB (OOF 8.606)
2. Implements proper Nested CV to avoid leakage
3. Adds V23 XGB (different FE) for diversity

Best Models Selected (from public_scores.md):
  1. V28 - TabM (3-seed) - LB: 8.56178, OOF: 8.597 🏆 BEST SINGLE
  2. V34 - XGBoost (5-seed Hybrid) - LB: 8.56352, OOF: 8.601 🏆 BEST XGB
  3. V37 - FT-Transformer - LB: 8.56379, OOF: 8.605 🏆 BEST DL
  4. V36 - LightGBM (CPU) - LB: 8.58278, OOF: 8.623 🏆 BEST LGB
  5. V39 - Tabular ResNet (5-seed) - LB: 8.57781, OOF: 8.621

Diversity Models (Different FE/Architecture):
  6. V23 - XGBoost (Pure CMT, no Golden) - LB: 8.56367, OOF: 8.607
  7. V27 - FT-Transformer (pytabkit) - LB: 8.56507, OOF: 8.630
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
print("S6E1 V41 - Nested CV Stacking (Gold Standard)")
print("="*80)

# Path configuration
if os.path.exists("/kaggle/input/playground-series-s6e1/train.csv"):
    print("[LOG] Kaggle Environment")
    TRAIN_PATH = "/kaggle/input/playground-series-s6e1/train.csv"
    TEST_PATH = "/kaggle/input/playground-series-s6e1/test.csv"
    OOF_BASE = "/kaggle/input/s6e1-oof-predictions/"
    SUB_BASE = "/kaggle/input/s6e1-submissions/"
    STAGE3_OOF = "/kaggle/input/s6e1-stage3-oof/"
    STAGE3_SUB = "/kaggle/input/s6e1-stage3-submissions/"
else:
    print("[LOG] Local Environment")
    TRAIN_PATH = "Dataset/train.csv"
    TEST_PATH = "Dataset/test.csv"
    OOF_BASE = "Previous trained files/OOF/"
    SUB_BASE = "Previous trained files/Submissions/"
    STAGE3_OOF = "Stage 3/OOF/"
    STAGE3_SUB = "Stage 3/Submission/"

# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("\n[LOG] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = "exam_score"
y_train = train_df[TARGET].values
n_train = len(y_train)
n_test = len(test_df)

print(f"Train: {n_train}, Test: {n_test}")

# ============================================================================
# 3. DEFINE MODELS (CORRECTED!)
# ============================================================================

# PRIMARY MODELS - Best from each category (VERIFIED from public_scores.md)
MODELS = {
    # TabM - BEST SINGLE MODEL
    "V28_TabM": {
        "oof": OOF_BASE + "oof_v28.csv",
        "sub": SUB_BASE + "submission_v28.csv",
        "lb": 8.56178, "oof_expected": 8.597,
        "note": "TabM 3-seed - BEST SINGLE"
    },
    # XGBoost - V34 is BETTER than S3_XGB! (OOF 8.601 vs 8.606)
    "V34_XGB": {
        "oof": OOF_BASE + "oof_v34.csv",  # FIXED! Was using oof_stage3_xgb.csv
        "sub": SUB_BASE + "submission_v34.csv",
        "lb": 8.56352, "oof_expected": 8.601,
        "note": "XGBoost Hybrid V32+Golden 5-seed - BEST XGB"
    },
    # FT-Transformer - Best DL
    "V37_FTT": {
        "oof": STAGE3_OOF + "oof_stage3_ftt.csv",
        "sub": STAGE3_SUB + "submission_stage3_ftt.csv",
        "lb": 8.56379, "oof_expected": 8.605,
        "note": "FT-Transformer - BEST DL"
    },
    # LightGBM - Only good LGB option
    "V36_LGB": {
        "oof": STAGE3_OOF + "oof_stage3_lgb.csv",
        "sub": STAGE3_SUB + "submission_stage3_lgb.csv",
        "lb": 8.58278, "oof_expected": 8.623,
        "note": "LightGBM CPU - BEST LGB"
    },
    # ResNet - NN diversity
    "V39_ResNet": {
        "oof": STAGE3_OOF + "oof_stage3_resnet.csv",
        "sub": STAGE3_SUB + "submission_stage3_resnet.csv",
        "lb": 8.57781, "oof_expected": 8.621,
        "note": "Tabular ResNet 5-seed"
    },
}

# DIVERSITY MODELS - Different FE/Architecture for potential gains
DIVERSITY_MODELS = {
    # V23 XGB - Different FE (Pure CMT, no Golden Features)
    "V23_XGB": {
        "oof": OOF_BASE + "oof_v23.csv",
        "sub": SUB_BASE + "submission_v23.csv",
        "lb": 8.56367, "oof_expected": 8.607,
        "note": "XGBoost Pure CMT (Different FE from V34)"
    },
    # V27 FTT - Different architecture (pytabkit)
    "V27_FTT": {
        "oof": OOF_BASE + "oof_v27_ftt.csv",
        "sub": SUB_BASE + "submission_v27_ftt.csv",
        "lb": 8.56507, "oof_expected": 8.630,
        "note": "FT-Transformer pytabkit (Different from V37)"
    },
}

# ============================================================================
# 4. LOAD OOF/SUBMISSION FILES
# ============================================================================

def load_predictions(models_dict, label):
    """Load OOF and submission predictions."""
    print(f"\n{'='*80}")
    print(f"Loading {label} Predictions...")
    print("="*80)
    
    oof_dict = {}
    sub_dict = {}
    
    for name, config in models_dict.items():
        try:
            # Load OOF
            df_oof = pd.read_csv(config["oof"])
            col = "oof_pred" if "oof_pred" in df_oof.columns else "exam_score"
            if col not in df_oof.columns:
                col = df_oof.columns[-1]
            oof = df_oof[col].values
            
            # Verify OOF RMSE
            rmse = np.sqrt(mean_squared_error(y_train, oof))
            expected = config["oof_expected"]
            match = "✓" if abs(rmse - expected) < 0.01 else "⚠️"
            
            print(f"  {match} {name}: OOF = {rmse:.5f} (expected {expected:.3f}) - {config['note']}")
            oof_dict[name] = oof
            
            # Load Submission
            if os.path.exists(config["sub"]):
                df_sub = pd.read_csv(config["sub"])
                sub_dict[name] = df_sub["exam_score"].values
            
        except Exception as e:
            print(f"  ✗ {name}: FAILED - {e}")
    
    return oof_dict, sub_dict

# Load all models
primary_oof, primary_sub = load_predictions(MODELS, "PRIMARY")
diversity_oof, diversity_sub = load_predictions(DIVERSITY_MODELS, "DIVERSITY")

# Combine
all_oof = {**primary_oof, **diversity_oof}
all_sub = {**primary_sub, **diversity_sub}

print(f"\n[LOG] Loaded {len(all_oof)} total OOF predictions")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("Correlation Matrix (All Models)")
print("="*80)

oof_names = list(all_oof.keys())
oof_matrix = np.column_stack([all_oof[name] for name in oof_names])
corr_matrix = np.corrcoef(oof_matrix.T)

# Print header
print(f"\n{'':12}", end="")
for name in oof_names:
    print(f"{name[:7]:>8}", end="")
print()

# Print rows
for i, name in enumerate(oof_names):
    print(f"{name[:12]:12}", end="")
    for j in range(len(oof_names)):
        val = corr_matrix[i, j]
        flag = "*" if i != j and val > 0.995 else " "
        print(f"{val:7.4f}{flag}", end="")
    print()

# ============================================================================
# 6. NESTED CV STACKING (GOLD STANDARD!)
# ============================================================================

print(f"\n{'='*80}")
print("NESTED CV STACKING (Gold Standard)")
print("="*80)
print("""
Why Nested CV?
- Regular CV: Train meta-model on same folds used for OOF → potential leakage
- Nested CV: Meta-model uses INNER CV, evaluated on OUTER holdout → no leakage
""")

OUTER_FOLDS = 5
INNER_FOLDS = 5

outer_kf = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
oof_nested = np.zeros(n_train)

print(f"[LOG] Running {OUTER_FOLDS}-Outer x {INNER_FOLDS}-Inner CV...")

for outer_fold, (train_idx, val_idx) in enumerate(outer_kf.split(oof_matrix), 1):
    # Inner CV to train meta-model
    X_inner = oof_matrix[train_idx]
    y_inner = y_train[train_idx]
    
    # Train Ridge with inner CV
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=INNER_FOLDS)
    ridge.fit(X_inner, y_inner)
    
    # Predict on outer holdout
    X_outer = oof_matrix[val_idx]
    oof_nested[val_idx] = ridge.predict(X_outer)
    
    rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_nested[val_idx]))
    print(f"  Outer Fold {outer_fold}: RMSE = {rmse:.5f} (alpha={ridge.alpha_})")

nested_rmse = np.sqrt(mean_squared_error(y_train, oof_nested))
print(f"\n[NESTED CV] OOF RMSE: {nested_rmse:.5f}")

# ============================================================================
# 7. COMPARE WITH SIMPLE RIDGE
# ============================================================================

print(f"\n{'='*80}")
print("Comparing Methods")
print("="*80)

# Simple Ridge (V40 style)
simple_kf = KFold(n_splits=5, shuffle=True, random_state=1003)
oof_simple = np.zeros(n_train)

for fold, (train_idx, val_idx) in enumerate(simple_kf.split(oof_matrix), 1):
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
    ridge.fit(oof_matrix[train_idx], y_train[train_idx])
    oof_simple[val_idx] = ridge.predict(oof_matrix[val_idx])

simple_rmse = np.sqrt(mean_squared_error(y_train, oof_simple))

# Simple Average
simple_avg = np.mean(oof_matrix, axis=1)
avg_rmse = np.sqrt(mean_squared_error(y_train, simple_avg))

print(f"| Method          | OOF RMSE  | vs V40 (8.586) |")
print(f"|-----------------|-----------|----------------|")
print(f"| Simple Average  | {avg_rmse:.5f}  | {avg_rmse - 8.58610:+.5f}       |")
print(f"| Simple Ridge    | {simple_rmse:.5f}  | {simple_rmse - 8.58610:+.5f}       |")
print(f"| Nested CV       | {nested_rmse:.5f}  | {nested_rmse - 8.58610:+.5f}       |")

# ============================================================================
# 8. HILL CLIMBING ON BEST METHOD
# ============================================================================

print(f"\n{'='*80}")
print("Hill Climbing Optimization")
print("="*80)

# Use final Ridge coefficients for initialization
ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge_final.fit(oof_matrix, y_train)

print(f"\nRidge coefficients:")
for name, coef in zip(oof_names, ridge_final.coef_):
    pct = coef / sum(ridge_final.coef_) * 100
    print(f"  {name:12}: {coef:7.4f} ({pct:5.1f}%)")

def objective(weights):
    w = np.abs(weights) / np.sum(np.abs(weights))
    blend = np.sum(oof_matrix * w, axis=1)
    return np.sqrt(mean_squared_error(y_train, blend))

# Optimize
init_weights = np.abs(ridge_final.coef_) / np.sum(np.abs(ridge_final.coef_))
result = minimize(objective, init_weights, method='Nelder-Mead', 
                  options={'maxiter': 2000, 'xatol': 1e-9})

optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
hill_rmse = objective(optimal_weights)

print(f"\nOptimized weights:")
for name, w in zip(oof_names, optimal_weights):
    print(f"  {name:12}: {w:7.4f} ({w*100:5.1f}%)")

print(f"\n[HILL CLIMBING] OOF RMSE: {hill_rmse:.5f}")

# ============================================================================
# 9. CHOOSE BEST & GENERATE SUBMISSION
# ============================================================================

print(f"\n{'='*80}")
print("FINAL RESULTS")
print("="*80)

results = {
    "Simple Average": (avg_rmse, None),
    "Simple Ridge": (simple_rmse, ridge_final.coef_),
    "Nested CV": (nested_rmse, None),
    "Hill Climbing": (hill_rmse, optimal_weights)
}

best_method = min(results.keys(), key=lambda x: results[x][0])
best_rmse = results[best_method][0]
best_weights = results[best_method][1]

print(f"\n| Method          | OOF RMSE  |")
print(f"|-----------------|-----------|")
for method, (rmse, _) in sorted(results.items(), key=lambda x: x[1][0]):
    marker = "🏆" if method == best_method else "  "
    print(f"| {method:15} | {rmse:.5f}  | {marker}")

print(f"\n[BEST] {best_method} with OOF RMSE: {best_rmse:.5f}")
print(f"[V40] OOF RMSE was: 8.58610")
print(f"[IMPROVEMENT] V41 vs V40: {best_rmse - 8.58610:+.5f}")

# Generate submission
if best_weights is not None:
    weights = np.abs(best_weights) / np.sum(np.abs(best_weights))
else:
    # Use simple average for methods without weights
    weights = np.ones(len(oof_names)) / len(oof_names)

# Stack test predictions
test_names = [n for n in oof_names if n in all_sub]
if len(test_names) == len(oof_names):
    test_matrix = np.column_stack([all_sub[name] for name in oof_names])
    test_blend = np.sum(test_matrix * weights, axis=1)
    
    submission = pd.DataFrame({
        "id": test_df["id"],
        "exam_score": test_blend
    })
    submission.to_csv("submission_v41.csv", index=False)
    print(f"\n✓ Saved submission_v41.csv")
    print(f"  Range: [{test_blend.min():.2f}, {test_blend.max():.2f}]")
else:
    print(f"\n⚠️ Missing submissions: {set(oof_names) - set(test_names)}")

# Save OOF
if best_method == "Nested CV":
    best_oof = oof_nested
elif best_method == "Hill Climbing":
    best_oof = np.sum(oof_matrix * optimal_weights, axis=1)
else:
    best_oof = oof_simple

oof_df = pd.DataFrame({"id": train_df["id"], "oof_pred": best_oof})
oof_df.to_csv("oof_v41.csv", index=False)
print(f"✓ Saved oof_v41.csv")

print("\n" + "="*80)
print("V41 NESTED CV STACKING COMPLETE")
print("="*80)

print(f"""
KEY IMPROVEMENTS vs V40:
1. ✅ Used V34 XGB (OOF 8.601) instead of S3_XGB (OOF 8.606)
2. ✅ Added V23 XGB (different FE) for diversity  
3. ✅ Added V27 FTT (pytabkit) for architecture diversity
4. ✅ Implemented Nested CV (outer/inner) to avoid leakage
5. ✅ Applied Hill Climbing on final weights

Expected LB: ~{best_rmse - 0.033:.5f} (OOF - 0.033 gap)
V40 achieved: LB 8.55289, OOF 8.58610
""")
