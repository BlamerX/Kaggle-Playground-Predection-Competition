# !pip install pytabkit -q 
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
try:
    from pytabkit import RealMLP_TD_Classifier
except ImportError:
    print("❌ Pytabkit not found! initializing Dummy or Failing...")
    # Fallback or Error - User must install it
    pass

import time
import os
import random

warnings.filterwarnings('ignore')

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V64"
    DESCRIPTION = "Distillation_from_V62_Champion"

    # RealMLP_TD_Classifier Params (Same as V59/V51)
    # Optimized for 0.95397+ Performance
    PARAM_GRID = {
        'device': DEVICE,
        'random_state': 42,
        'verbosity': 1,     
        'n_epochs': 100,
        'batch_size': 256,
        'n_ens': 1, 
        'use_early_stopping': True,
        'early_stopping_additive_patience': 20,
        'early_stopping_multiplicative_patience': 1,
        'act': "mish",
        'embedding_size': 8,
        'first_layer_lr_factor': 0.5962121993798933,
        'hidden_sizes': "rectangular",
        'hidden_width': 384,
        'lr': 0.04,
        'ls_eps': 0.011498317194338772,
        'ls_eps_sched': "coslog4",
        'max_one_hot_cat_size': 18,
        'n_hidden_layers': 4,
        'p_drop': 0.07301419697186451,
        'p_drop_sched': "flat_cos",
        'plr_hidden_1': 16,
        'plr_hidden_2': 8,
        'plr_lr_factor': 0.1151437622270563,
        'plr_sigma': 2.3316811282666916,
        'scale_lr_factor': 2.244801835541429,
        'sq_mom': 1.0 - 0.011834054955582318,
        'wd': 0.02369230879235962,
    }

    SEEDS = [42] # Single Seed Distillation first (can expand to multi later)
    N_FOLDS = 5
    
    # Pseudo-Labeling Config (Hard Labels)
    PL_THRESHOLD_HIGH = 0.99
    PL_THRESHOLD_LOW = 0.01

    # Paths (Kaggle vs Local)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    # Teacher Path (V62 Submission)
    TEACHER_PATH_KAGGLE = '/kaggle/input/oof-and-submission/S6E2/Previous Trained Files/Submission/submission_v62.csv'
    TEACHER_PATH_LOCAL = 'Previous Trained Files/Submission/submission_v62.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# FEATURE ENGINEERING
# ==================================================================================
def add_engineered_features(df, original, base_features):
    df_temp = df.copy()
    for col in base_features:
        if col in original.columns:
            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
            df_temp = df_temp.merge(stats, on=col, how='left')
            fill_values = {
                f"orig_{col}_mean": original['Heart Disease'].mean(),
                f"orig_{col}_median": original['Heart Disease'].median(),
                f"orig_{col}_std": 0, f"orig_{col}_skew": 0, f"orig_{col}_count": 0
            }
            df_temp = df_temp.fillna(value=fill_values)
    return df_temp

def add_tier1_features(df):
    df = df.copy()
    if 'EKG results' in df.columns:
        df['EKG_Binary'] = ((df['EKG results'] == 0) | (df['EKG results'] == 1)).astype(int)
    if 'Slope of ST' in df.columns and 'ST depression' in df.columns:
        df['ST_Slope_Interaction'] = df['Slope of ST'] * df['ST depression']
    if 'Chest pain type' in df.columns:
        df['Chest_Pain_Binary'] = (df['Chest pain type'] == 4).astype(int)
    return df

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()

    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "Dataset/train.csv")
    test = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "Dataset/test.csv")
    original = pd.read_csv(CFG.ORIG_PATH if os.path.exists(CFG.ORIG_PATH) else "Dataset/Heart_Disease_Prediction.csv")
    
    # 2. Teacher Loading
    teacher_path = CFG.TEACHER_PATH_LOCAL
    if os.path.exists(CFG.TEACHER_PATH_KAGGLE):
        teacher_path = CFG.TEACHER_PATH_KAGGLE
    elif not os.path.exists(teacher_path):
         if os.path.exists("submission_v62.csv"): teacher_path = "submission_v62.csv"
         else: 
             print(f"❌ Teacher Not Found!"); return
             
    print(f"Loading Teacher: {teacher_path}")
    sub = pd.read_csv(teacher_path)
    
    # 3. Pseudo-Label Generation (Hard)
    # Identify Confident Predictions
    high_conf = sub[sub['Heart Disease'] > CFG.PL_THRESHOLD_HIGH].copy()
    low_conf = sub[sub['Heart Disease'] < CFG.PL_THRESHOLD_LOW].copy()
    
    high_conf['Heart Disease'] = 'Presence' 
    low_conf['Heart Disease'] = 'Absence'
    
    # Merge Features
    high_conf = test[test['id'].isin(high_conf['id'])].merge(high_conf[['id', 'Heart Disease']], on='id')
    low_conf = test[test['id'].isin(low_conf['id'])].merge(low_conf[['id', 'Heart Disease']], on='id')
    
    pl_data = pd.concat([high_conf, low_conf])
    print(f"Added {len(pl_data)} Pseudo-Labeled samples ({len(high_conf)} Pos, {len(low_conf)} Neg)")
    
    if len(pl_data) > 0:
        # Note: We do NOT concat here for CV. We concat inside the fold loop to avoid leakage into Validation.
        # But wait, V59 concatenation logic:
        # V59 Concats inside the loop. Correct.
        pass
    else:
        print("⚠️ No samples met the threshold constraint!")

    # 4. Preprocessing
    le = LabelEncoder()
    train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])
    
    # Encoder for PLs
    # 'Presence' -> 1, 'Absence' -> 0 (Verify map)
    # Train: Presence=1, Absence=0.
    # Let's verify standard LE behavior or force it.
    # Presence is usually 1.
    if len(pl_data) > 0:
        pl_data['Heart Disease'] = pl_data['Heart Disease'].map({'Presence': 1, 'Absence': 0})

    print("Injecting original dataset features...")
    base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']]
    train = add_engineered_features(train, original, base_features)
    test = add_engineered_features(test, original, base_features)
    if len(pl_data) > 0:
        pl_data = add_engineered_features(pl_data, original, base_features)
    
    print("Injecting Tier 1 Features...")
    train = add_tier1_features(train)
    test = add_tier1_features(test)
    if len(pl_data) > 0:
        pl_data = add_tier1_features(pl_data)

    X = train.drop(['id', 'Heart Disease'], axis=1)
    y = train['Heart Disease']
    X_test = test.drop(['id'], axis=1)
    
    if len(pl_data) > 0:
        X_pl = pl_data.drop(['id', 'Heart Disease'], axis=1)
        y_pl = pl_data['Heart Disease']

    print("Converting all features to categorical type...")
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')
        if len(pl_data) > 0:
             X_pl[col] = X_pl[col].astype(str).astype('category')

    # 5. Multi-Seed CV Training (Fixed Folds for valid OOF)
    # Identify Original Indices
    # train_orig = train[~train['id'].isin(test['id'])] # Wait, train definition above is standard.
    # Simply use X, y as train.
    
    train_orig = train
    X_orig = X
    y_orig = y
    
    final_oof_preds = np.zeros(len(train_orig))
    final_test_preds = np.zeros(len(test))
    
    # FIXED FOLDS (Seed 42)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42) 
    fold_indices = list(skf.split(X_orig, y_orig))
    
    print(f"\nStarting Training ({len(CFG.SEEDS)} Seeds)...")

    for seed_idx, seed in enumerate(CFG.SEEDS):
        print(f"\n⚡ STARTING SEED {seed} ({seed_idx + 1}/{len(CFG.SEEDS)}) ⚡")
        
        CFG.PARAM_GRID['random_state'] = seed
        
        seed_oof = np.zeros(len(train_orig))
        seed_test = np.zeros(len(test))
        
        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            X_tr_orig = X_orig.iloc[train_idx]
            y_tr_orig = y_orig.iloc[train_idx]
            X_val = X_orig.iloc[val_idx]
            y_val = y_orig.iloc[val_idx]
            
            # Augment with PLs
            if len(pl_data) > 0:
                X_tr = pd.concat([X_tr_orig, X_pl])
                y_tr = pd.concat([y_tr_orig, y_pl])
            else:
                X_tr, y_tr = X_tr_orig, y_tr_orig

            model = RealMLP_TD_Classifier(**CFG.PARAM_GRID)
            model.fit(X_tr, y_tr.values, X_val, y_val.values)

            val_probs = model.predict_proba(X_val)[:, 1]
            fold_test_probs = model.predict_proba(X_test)[:, 1]

            seed_oof[val_idx] = val_probs
            seed_test += fold_test_probs / CFG.N_FOLDS
            
            print(f"  Fold {fold+1} AUC: {roc_auc_score(y_val, val_probs):.5f}")
            
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
                
        final_oof_preds += seed_oof / len(CFG.SEEDS)
        final_test_preds += seed_test / len(CFG.SEEDS)
        
        print(f"Seed {seed} Done.")

    # 6. Eval & Save
    overall_score = roc_auc_score(y_orig, final_oof_preds)
    print(f"\n{'=' * 40}")
    print(f"Overall OOF ROC-AUC: {overall_score:.5f}")
    print(f"{'=' * 40}")

    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)

    pd.DataFrame({'id': train_orig['id'], 'Heart Disease_prob': final_oof_preds}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test['id'], 'Heart Disease': final_test_preds}).to_csv(CFG.SUBMISSION_PATH, index=False)

    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")

if __name__ == "__main__":
    main()
