# !pip install pytabkit -q 
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pytabkit import RealMLP_TD_Classifier
import time
import os

warnings.filterwarnings('ignore')

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V58"
    DESCRIPTION = "RealMLP_Distillation_from_V53_Recreated"

    # RealMLP_TD_Classifier Params (Same as V51/V48)
    PARAM_GRID = {
        'device': DEVICE,
        'random_state': 42,
        'verbosity': 2,
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

    SEED = 42
    N_FOLDS = 5
    
    # Distillation / Pseudo-Labeling Config
    PL_THRESHOLD_HIGH = 0.99
    PL_THRESHOLD_LOW = 0.01

    # Paths (Kaggle)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'

    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# BLEND RECREATION (V53 Logic)
# ==================================================================================
def recreate_v53_blend():
    print("Recreating V53 Blend from components...")
    
    # V53 Weights
    weights = {
        'submission_v48.csv': 0.4774,
        'submission_v49.csv': 0.4000,
        'submission_v51.csv': 0.0989,
        'submission_v52.csv': 0.0238
    }
    
    preds = []
    total_weight = 0
    ids = None
    
    for filename, w in weights.items():
        path = f"/kaggle/input/oof-and-submission/S6E2/Previous Trained Files/Submission/{filename}"
        if not os.path.exists(path):
            print(f"❌ Missing component: {path}")
            return None
        
        df = pd.read_csv(path)
        col = [c for c in df.columns if 'Heart' in c or 'pred' in c][0]
        
        if ids is None: ids = df['id']
        
        preds.append(df[col].values * w)
        total_weight += w
        print(f"  Loaded {filename} (w={w})")
        
    final_pred = np.sum(preds, axis=0) / total_weight
    return pd.DataFrame({'id': ids, 'Heart Disease': final_pred})

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
    
    # 2. Recreate V53 Blend & Add Pseudo-Labels
    sub = recreate_v53_blend()
    
    if sub is not None:
        print(f"V53 Recreated. Range: {sub['Heart Disease'].min():.4f} - {sub['Heart Disease'].max():.4f}")
        
        # Identify Confident Predictions
        high_conf = sub[sub['Heart Disease'] > CFG.PL_THRESHOLD_HIGH].copy()
        low_conf = sub[sub['Heart Disease'] < CFG.PL_THRESHOLD_LOW].copy()
        
        high_conf['Heart Disease'] = 'Presence' 
        low_conf['Heart Disease'] = 'Absence'
        
        # Merge Features
        # Note: 'id' in sub corresponds to 'id' in test
        high_conf = test[test['id'].isin(high_conf['id'])].merge(high_conf[['id', 'Heart Disease']], on='id')
        low_conf = test[test['id'].isin(low_conf['id'])].merge(low_conf[['id', 'Heart Disease']], on='id')
        
        pl_data = pd.concat([high_conf, low_conf])
        print(f"Added {len(pl_data)} Pseudo-Labeled samples ({len(high_conf)} Pos, {len(low_conf)} Neg)")
        
        if len(pl_data) > 0:
            train = pd.concat([train, pl_data], axis=0).reset_index(drop=True)
            print(f"New Train Shape: {train.shape}")
        else:
            print("⚠️ No samples met the threshold constraint!")
    else:
        print(f"❌ Could not recreate V53! Proceeding without PL.")

    # 3. Preprocessing
    le = LabelEncoder()
    train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

    print("Injecting original dataset features...")
    base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']]
    train = add_engineered_features(train, original, base_features)
    test = add_engineered_features(test, original, base_features)
    
    print("Injecting Tier 1 Features...")
    train = add_tier1_features(train)
    test = add_tier1_features(test)

    X = train.drop(['id', 'Heart Disease'], axis=1)
    y = train['Heart Disease']
    X_test = test.drop(['id'], axis=1)

    print("Converting all features to categorical type...")
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')

    # 4. CV Training (Validating on Original Train Only)
    # Identify Original Indices (ID check is safer)
    train_orig = train[~train['id'].isin(test['id'])]
    y_orig = train_orig['Heart Disease']
    X_orig = X.iloc[:len(train_orig)]
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds_orig = np.zeros(len(train_orig))
    test_preds = np.zeros(len(test))
    fold_scores = []
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV (Train on Train+PL, Validate on Original Train)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_orig, y_orig)):
        print(f"\n--- Starting Fold {fold + 1} ---")
        
        # Original Train/Val for this fold
        X_tr_orig = X_orig.iloc[train_idx]
        y_tr_orig = y_orig.iloc[train_idx]
        X_val = X_orig.iloc[val_idx]
        y_val = y_orig.iloc[val_idx]
        
        # Add ALL PL data to the Training Set of this fold
        if 'pl_data' in locals() and len(pl_data) > 0:
            X_pl = X.iloc[len(train_orig):] # PL part is properly appended at end
            y_pl = y.iloc[len(train_orig):]
            
            X_tr = pd.concat([X_tr_orig, X_pl])
            y_tr = pd.concat([y_tr_orig, y_pl])
        else:
            X_tr, y_tr = X_tr_orig, y_tr_orig

        # Train
        model = RealMLP_TD_Classifier(**CFG.PARAM_GRID)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        # Predict
        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]

        oof_preds_orig[val_idx] = val_probs
        test_preds += fold_test_probs / CFG.N_FOLDS

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"Fold {fold + 1} ROC-AUC Score: {score:.5f}")

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 5. Eval & Save
    overall_score = roc_auc_score(y_orig, oof_preds_orig)
    print(f"\n{'=' * 40}")
    print(f"Overall OOF ROC-AUC: {overall_score:.5f}")
    print(f"Mean Fold Score: {np.mean(fold_scores):.5f}")
    print(f"{'=' * 40}")

    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)

    pd.DataFrame({'id': train_orig['id'], 'Heart Disease_prob': oof_preds_orig}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds}).to_csv(CFG.SUBMISSION_PATH, index=False)

    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")

if __name__ == "__main__":
    main()
