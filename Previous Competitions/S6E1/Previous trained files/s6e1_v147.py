"""
S6E1 V147 - FINAL SUPER BLEND
=============================
Replicates V144c Success (8.54287) with an EVEN STRONGER Base.

Strategy:
1. Base: Ridge Stack of ~14 Internal Models (Best of Best) + 4 Public NNs.
   - Core: V110, V101, V105, V67, V70
   - Bonus: V139 (Cat), V46 (L LGB), V44 (FTT), V77, V122 (TabM)
   - Public NNs: DeepTables, ResNet, Trompt, LNN
   
2. Target: 'submission.csv' (LB 8.54350) - The confirmed best single public file.

3. Blend: Focus on the magic ratio (25% Base / 75% Public) that scored 8.54287.

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
print("S6E1 V147 - FINAL SUPER BLEND")
print("="*80)

# ============================================================================
# 1. SETUP & PATHS
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("  Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    public_path = '/kaggle/input/oof-and-submission/Season6episode1/Public Submissions/'
    path_oof = base_path + "OOF/"
    path_sub = base_path + "Submissions/"
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"
    public_path = "Public Submissions/"
    path_oof = base_path + "OOF/"
    path_sub = base_path + "Submissions/"

TARGET = "exam_score"
y = train_df[TARGET].values

# ============================================================================
# 2. GENERATE SUPER RIDGE BASE
# ============================================================================

print("\n[STEP 1] Generating Super Ridge Base")
print("-"*40)

def load_model_data(name, oof_name, sub_name, is_public=False):
    try:
        if is_public:
            if "lnn" in name:
                oof = pd.read_csv(public_path + "oof_lnn.csv.csv")
                sub = pd.read_csv(public_path + "sub_lnn.csv.csv")
                oof_vals = oof[oof.columns[0]].values
            elif "deeptables" in name:
                oof = pd.read_csv(public_path + "oof_deeptables.csv")
                sub = pd.read_csv(public_path + "sub_deeptables.csv")
                col = 'oof_pred' if 'oof_pred' in oof.columns else 'exam_score'
                oof_vals = oof[col].values
            elif "resnet" in name:
                oof = pd.read_csv(public_path + "oof_resnet.csv")
                sub = pd.read_csv(public_path + "sub_resnet.csv")
                oof_vals = oof['exam_score'].values
            elif "trompt" in name:
                oof = pd.read_csv(public_path + "oof_trompt.csv")
                sub = pd.read_csv(public_path + "sub_trompt.csv")
                oof_vals = oof['oof_pred'].values
            else:
                return None, None
            sub_vals = sub['exam_score'].values
        else:
            oof = pd.read_csv(path_oof + oof_name)
            sub = pd.read_csv(path_sub + sub_name)
            col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
            oof_vals = oof[col].values
            sub_vals = sub['exam_score'].values
            
        rmse = np.sqrt(mean_squared_error(y, oof_vals))
        print(f"  ✓ {name:15s}: RMSE={rmse:.5f}")
        return oof_vals, sub_vals
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None, None

# EXPANDED Model List (Verified OOFs)
models_to_load = [
    # Core V144 set
    ("v110", "oof_v110.csv", "submission_v110.csv", False),
    ("v101", "oof_v101.csv", "submission_v101.csv", False),
    ("v105", "oof_v105.csv", "submission_v105.csv", False),
    ("v67",  "oof_v67.csv",  "submission_v67.csv",  False),
    ("v70",  "oof_v70.csv",  "submission_v70.csv",  False),
    
    # Bonus Strong Models
    ("v139", "oof_v139.csv", "submission_v139.csv", False), # CatBoost
    ("v46",  "oof_v46_lgb.csv", "submission_v46_lgb.csv", False), # LGB
    ("v44",  "oof_v44_ftt.csv", "submission_v44_ftt.csv", False), # FTT
    ("v77",  "oof_v77.csv",  "submission_v77.csv",  False),
    ("v122", "oof_v122.csv", "submission_v122.csv", False), # TabM Recursive

    # Public NNs
    ("deeptables", "", "", True),
    ("resnet",     "", "", True),
    ("trompt",     "", "", True),
    ("lnn",        "", "", True)
]

stored_models = {}
for name, oof_f, sub_f, is_pub in models_to_load:
    o, s = load_model_data(name, oof_f, sub_f, is_pub)
    if o is not None: stored_models[name] = (o, s)

print(f"  Loaded {len(stored_models)} base models.")

# Ridge Stacking
X_train = np.column_stack([m[0] for m in stored_models.values()])
X_test = np.column_stack([m[1] for m in stored_models.values()])

ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge.fit(X_train, y)
v147_base_test = np.clip(ridge.predict(X_test), 0, 100)
v147_base_rmse = np.sqrt(mean_squared_error(y, np.clip(ridge.predict(X_train), 0, 100)))

print(f"  ✓ V147 Super Ridge RMSE: {v147_base_rmse:.5f}")

# ============================================================================
# 3. LOAD BEST PUBLIC SUBMISSION
# ============================================================================

print("\n[STEP 2] Loading Best Public Submission")
print("-"*40)

try:
    # Specifically using 'submission.csv' (8.54350) as requested
    pub_df = pd.read_csv(public_path + "submission.csv")
    pub_target = pub_df['exam_score'].values
    print("  ✓ Loaded 'submission.csv' (LB 8.54350)")
except Exception as e:
    print(f"  ✗ FATAL: Target submission.csv not found! {e}")
    exit(1)

# ============================================================================
# 4. COMPUTE FINAL BLENDS
# ============================================================================

print("\n[STEP 3] Computing Final Blends")
print("-"*40)

# V147a: Exact V144c Ratio (25/75) 
v147a = 0.25 * v147_base_test + 0.75 * pub_target

# V147b: Slightly More Base (28/72) - Safety against overfitting public
v147b = 0.28 * v147_base_test + 0.72 * pub_target

# V147c: Slightly Less Base (22/78) - Aggressive Public Pushing
v147c = 0.22 * v147_base_test + 0.78 * pub_target

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

save_sub(v147a, "submission_v147a", "25% Base + 75% Public (8.54350)")
save_sub(v147b, "submission_v147b", "28% Base + 72% Public (8.54350)")
save_sub(v147c, "submission_v147c", "22% Base + 78% Public (8.54350)")

print(f"\n  Total time: {time.time() - start_time:.1f}s")
print("="*80)
