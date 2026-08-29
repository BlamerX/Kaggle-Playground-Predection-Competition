
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import os
import gc
import sys
import subprocess

warnings.filterwarnings('ignore')

# Auto-Install Interpret if missing
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    EBM_AVAILABLE = True
except ImportError:
    print("⚠️ 'interpret' library not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "interpret","-q"])
        from interpret.glassbox import ExplainableBoostingClassifier
        EBM_AVAILABLE = True
        print("✅ 'interpret' installed successfully!")
    except Exception as e:
        EBM_AVAILABLE = False
        print(f"❌ Failed to install 'interpret': {e}")
        print("Please install it manually: !pip install interpret")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V36"
    DESCRIPTION = "EBM_Baseline"
    
    # EBM Params
    # EBM is a GAM, so it is naturally additive.
    # Interactions: EBM automatically detects interactions (default: 10).
    EBM_PARAMS = {
        'max_bins': 256,
        'max_interaction_bins': 32,
        'interactions': 20,         # Increase interactions for more capacity
        'outer_bags': 8,            # 8 outer bags for stability
        'inner_bags': 0,            # 0 inner bags for speed
        'learning_rate': 0.01,
        'validation_size': 0.15,
        'early_stopping_rounds': 50,
        'n_jobs': -1,
        'random_state': 42
    }
    
    SEED = 42
    N_FOLDS = 5 # EBM is slower, stick to 5 folds
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    
    if not EBM_AVAILABLE:
        print("⚠️ SKIPPING: EBM library missing.")
        return

    start_time = time.time()
    
    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "train.csv")
    test = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "test.csv")
    try:
        orig = pd.read_csv(CFG.ORIG_PATH if os.path.exists(CFG.ORIG_PATH) else "Heart_Disease_Prediction.csv")
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    # Target Mapping
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
        # Handle Unmapped (if any)
        train = train.dropna(subset=['Heart Disease'])
        train['Heart Disease'] = train['Heart Disease'].astype(int)
        
    if len(orig) > 0 and orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})
        orig = orig.dropna(subset=['Heart Disease'])
        orig['Heart Disease'] = orig['Heart Disease'].astype(int)

    # 2. Basic Feature Engineering (Sticking to Raw mainly)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']
    
    # EBM handles categorical/continuous automatically, but treating some low-cardinality nums as cats helps
    # For EBM, specifically, we don't need OHE. It likes raw.
    
    FEATURES = NUMS + CATS
    
    # 3. Validation Loop
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []
    
    # Augment
    X_aug = pd.concat([train, orig], axis=0).reset_index(drop=True) if len(orig) > 0 else train.copy()
    y_aug = X_aug['Heart Disease']
    X_aug = X_aug[FEATURES]

    # Main train is just train for CV splitting correctness
    X_train_full = train[FEATURES]
    y_train_full = train['Heart Disease']
    X_test_full = test[FEATURES]
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with EBM...")
    
    for i, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
        
        # Split Standard Train
        X_tr = X_train_full.iloc[train_index]
        y_tr = y_train_full.iloc[train_index]
        X_val = X_train_full.iloc[val_index]
        y_val = y_train_full.iloc[val_index]
        
        # Add Original Data to Train
        if len(orig) > 0:
            X_orig_feat = orig[FEATURES]
            y_orig_feat = orig['Heart Disease']
            X_tr = pd.concat([X_tr, X_orig_feat], axis=0)
            y_tr = pd.concat([y_tr, y_orig_feat], axis=0)
            
        # Shuffle
        X_tr = X_tr.sample(frac=1, random_state=42).reset_index(drop=True)
        y_tr = y_tr.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # EBM Training
        model = ExplainableBoostingClassifier(**CFG.EBM_PARAMS)
        model.fit(X_tr, y_tr)
        
        # Predict
        val_p = model.predict_proba(X_val)[:, 1]
        oof[val_index] = val_p
        
        roc_auc_fold = roc_auc_score(y_val, val_p)
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f}")
        
        # Test Prediction
        pred += model.predict_proba(X_test_full)[:, 1] / CFG.N_FOLDS
        
        del model
        gc.collect()

    overall_score = roc_auc_score(y_train_full, oof)
    print(f"\nOverall EBM CV AUC: {overall_score:.5f}")
    
    # Save Files
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': y_train_full.values, 'pred': oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
