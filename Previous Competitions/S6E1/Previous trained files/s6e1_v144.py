"""
S6E1 V144 - Diversity Blend with Public Neural Network Models
==============================================================
Blend our best models with public NN models (DeepTables, LNN, ResNet, Trompt)
for maximum diversity.
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
print("S6E1 V144 - Diversity Blend with Public Neural Network Models")
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
    public_path = '/kaggle/input/oof-and-submission/Season6episode1/Public Submissions/'
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"
    public_path = "Public Submissions/"

TARGET = "exam_score"
y = train_df[TARGET].values
n_train = len(train_df)
n_test = len(test_df)

print(f"  Train: {n_train:,} rows")
print(f"  Test: {n_test:,} rows")

# ============================================================================
# 2. LOAD OUR BEST MODELS
# ============================================================================

print("\n[STEP 2] LOADING OUR BEST MODELS")
print("-"*40)

def load_our_oof(name, oof_file, sub_file):
    """Load our OOF files from Previous trained files"""
    try:
        oof = pd.read_csv(base_path + f"OOF/{oof_file}")
        sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
        col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
        oof_vals = oof[col].values
        sub_vals = sub['exam_score'].values
        rmse = np.sqrt(mean_squared_error(y, oof_vals))
        print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
        return oof_vals, sub_vals, rmse
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None, None, None

our_models = {}

# Best single models
models_to_load = [
    ("v110_catboost", "oof_v110.csv", "submission_v110.csv"),
    ("v101_xgboost", "oof_v101.csv", "submission_v101.csv"),
    ("v105_tabm", "oof_v105.csv", "submission_v105.csv"),
    ("v67_lgb", "oof_v67.csv", "submission_v67.csv"),
    ("v70_ftt", "oof_v70.csv", "submission_v70.csv"),
]

for name, oof_f, sub_f in models_to_load:
    oof, sub, rmse = load_our_oof(name, oof_f, sub_f)
    if oof is not None:
        our_models[name] = {"oof": oof, "sub": sub, "rmse": rmse}

# ============================================================================
# 3. LOAD PUBLIC NEURAL NETWORK MODELS
# ============================================================================

print("\n[STEP 3] LOADING PUBLIC NEURAL NETWORK MODELS")
print("-"*40)

public_models = {}  

# DeepTables
try:
    dt_oof = pd.read_csv(public_path + "oof_deeptables.csv")
    dt_sub = pd.read_csv(public_path + "sub_deeptables.csv")
    dt_oof_col = 'oof_pred' if 'oof_pred' in dt_oof.columns else 'exam_score'
    dt_oof_vals = dt_oof[dt_oof_col].values
    dt_sub_vals = dt_sub['exam_score'].values
    dt_rmse = np.sqrt(mean_squared_error(y, dt_oof_vals))
    public_models["deeptables"] = {"oof": dt_oof_vals, "sub": dt_sub_vals, "rmse": dt_rmse}
    print(f"  ✓ DeepTables: OOF RMSE = {dt_rmse:.5f}")
except Exception as e:
    print(f"  ✗ DeepTables: {e}")

# ResNet
try:
    resnet_oof = pd.read_csv(public_path + "oof_resnet.csv")
    resnet_sub = pd.read_csv(public_path + "sub_resnet.csv")
    resnet_oof_col = 'exam_score' if 'exam_score' in resnet_oof.columns else 'oof_pred'
    resnet_oof_vals = resnet_oof[resnet_oof_col].values
    resnet_sub_vals = resnet_sub['exam_score'].values
    resnet_rmse = np.sqrt(mean_squared_error(y, resnet_oof_vals))
    public_models["resnet"] = {"oof": resnet_oof_vals, "sub": resnet_sub_vals, "rmse": resnet_rmse}
    print(f"  ✓ ResNet: OOF RMSE = {resnet_rmse:.5f}")
except Exception as e:
    print(f"  ✗ ResNet: {e}")

# Trompt
try:
    trompt_oof = pd.read_csv(public_path + "oof_trompt.csv")
    trompt_sub = pd.read_csv(public_path + "sub_trompt.csv")
    trompt_oof_col = 'oof_pred' if 'oof_pred' in trompt_oof.columns else 'exam_score'
    trompt_oof_vals = trompt_oof[trompt_oof_col].values
    trompt_sub_vals = trompt_sub['exam_score'].values
    trompt_rmse = np.sqrt(mean_squared_error(y, trompt_oof_vals))
    public_models["trompt"] = {"oof": trompt_oof_vals, "sub": trompt_sub_vals, "rmse": trompt_rmse}
    print(f"  ✓ Trompt: OOF RMSE = {trompt_rmse:.5f}")
except Exception as e:
    print(f"  ✗ Trompt: {e}")

# LNN (needs special handling - no id column)
try:
    lnn_oof = pd.read_csv(public_path + "oof_lnn.csv.csv")  # Note: double .csv
    lnn_sub = pd.read_csv(public_path + "sub_lnn.csv.csv")
    # LNN OOF doesn't have id, just values
    lnn_oof_col = lnn_oof.columns[0]  # First column is the prediction
    lnn_oof_vals = lnn_oof[lnn_oof_col].values
    lnn_sub_vals = lnn_sub['exam_score'].values
    lnn_rmse = np.sqrt(mean_squared_error(y, lnn_oof_vals))
    public_models["lnn"] = {"oof": lnn_oof_vals, "sub": lnn_sub_vals, "rmse": lnn_rmse}
    print(f"  ✓ LNN: OOF RMSE = {lnn_rmse:.5f}")
except Exception as e:
    print(f"  ✗ LNN: {e}")

print(f"\n  Loaded {len(our_models)} of our models + {len(public_models)} public models")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

print("\n[STEP 4] CORRELATION ANALYSIS")
print("-"*40)

all_models = {**our_models, **public_models}
model_names = list(all_models.keys())
n_models = len(model_names)

# Build correlation matrix
oof_matrix = np.column_stack([all_models[m]["oof"] for m in model_names])
corr_matrix = np.corrcoef(oof_matrix.T)

print(f"\n  Model Correlations (lower = more diverse):")
for i, name in enumerate(model_names):
    avg_corr = (np.sum(corr_matrix[i]) - 1) / (n_models - 1)  # Exclude self
    print(f"    {name:20s}: avg corr = {avg_corr:.4f}")

# ============================================================================
# 5. RIDGE STACKING
# ============================================================================

print("\n[STEP 5] RIDGE STACKING")
print("-"*40)

oof_stack = oof_matrix
test_stack = np.column_stack([all_models[m]["sub"] for m in model_names])

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

ridge_oof = np.zeros(n_train)
ridge_test_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), 1):
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    ridge.fit(oof_stack[tr_idx], y[tr_idx])
    ridge_oof[val_idx] = np.clip(ridge.predict(oof_stack[val_idx]), 0, 100)
    ridge_test_preds.append(np.clip(ridge.predict(test_stack), 0, 100))

v144a_test = np.mean(ridge_test_preds, axis=0)
v144a_rmse = np.sqrt(mean_squared_error(y, ridge_oof))

print(f"  V144a Ridge OOF: {v144a_rmse:.5f}")

# Get Ridge weights
ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge_final.fit(oof_stack, y)
print(f"\n  Ridge weights:")
for name, w in sorted(zip(model_names, ridge_final.coef_), key=lambda x: -abs(x[1])):
    print(f"    {name:20s}: {w:+.4f}")

# ============================================================================
# 6. BLEND WITH PUBLIC SUBMISSION
# ============================================================================

print("\n[STEP 6] BLEND WITH PUBLIC SUBMISSION (LB 8.54363)")
print("-"*40)

try:
    public_sub = pd.read_csv(public_path + "Public submission.csv")
    public_test = public_sub['exam_score'].values
    print(f"  ✓ Public submission loaded")
    
    # Different blend ratios
    v144b_test = 0.30 * v144a_test + 0.70 * public_test  # 30/70
    v144c_test = 0.25 * v144a_test + 0.75 * public_test  # 25/75
    v144d_test = 0.35 * v144a_test + 0.65 * public_test  # 35/65
    v144e_test = 0.40 * v144a_test + 0.60 * public_test  # 40/60
    
    has_public = True
except Exception as e:
    print(f"  ✗ Public not found: {e}")
    has_public = False

# ============================================================================
# 7. RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"""
  V144a Ridge Stacking ({len(all_models)} models):
    - Our models: {list(our_models.keys())}
    - Public NN: {list(public_models.keys())}
    - OOF RMSE: {v144a_rmse:.5f}

  Comparison:
    | Version | Models | OOF RMSE | Notes |
    |---------|--------|----------|-------|
    | V141a   | 14 (our only) | 8.55716 | Ridge stack |
    | V143a   | 24 (our only) | 8.54739 | Ridge stack |
    | **V144a**| **{len(all_models)}** | **{v144a_rmse:.5f}** | **+ Public NN diversity** |
""")

# ============================================================================
# 8. SAVE
# ============================================================================

print("="*80)
print("SAVING FILES")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': v144a_test}).to_csv("submission_v144a.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': ridge_oof}).to_csv("oof_v144a.csv", index=False)
print(f"  ✓ submission_v144a.csv (pure Ridge)")
print(f"  ✓ oof_v144a.csv")

if has_public:
    pd.DataFrame({'id': test_df['id'], 'exam_score': v144b_test}).to_csv("submission_v144b.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': v144c_test}).to_csv("submission_v144c.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': v144d_test}).to_csv("submission_v144d.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': v144e_test}).to_csv("submission_v144e.csv", index=False)
    print(f"  ✓ submission_v144b.csv (30% V144a + 70% Public)")
    print(f"  ✓ submission_v144c.csv (25% V144a + 75% Public)")
    print(f"  ✓ submission_v144d.csv (35% V144a + 65% Public)")
    print(f"  ✓ submission_v144e.csv (40% V144a + 60% Public)")

total_time = time.time() - start_time
print(f"\n  Total time: {total_time:.1f} seconds")
print("="*80)
