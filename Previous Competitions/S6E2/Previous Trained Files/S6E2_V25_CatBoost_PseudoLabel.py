
import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V25"
    DESCRIPTION = "CatBoost_PseudoLabel"
    
    SEED = 42
    N_FOLDS = 10
    INNER_FOLDS = 5
    
    # Pseudo-Labeling Config
    PL_THRESHOLD_HIGH = 0.99
    PL_THRESHOLD_LOW = 0.01
    
    # Base V17 Params
    CAT_PARAMS = {
        'iterations': 5000,
        'learning_rate': 0.02,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'early_stopping_rounds': 500,
        'task_type': 'GPU',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'allow_writing_files': False
    }
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

def train_model(X, y, X_test, cat_features=[], model_params=CFG.CAT_PARAMS, desc="Stage 1"):
    print(f"\nTraining {desc}...")
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        train_pool = cb.Pool(X_train, y_train, cat_features=cat_features)
        val_pool = cb.Pool(X_val, y_val, cat_features=cat_features)
        
        model = cb.CatBoostClassifier(**model_params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof[val_idx] = val_preds
        test_pred += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS
        
    score = roc_auc_score(y, oof)
    print(f"{desc} CV AUC: {score:.5f}")
    return oof, test_pred

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Self-Training (Pseudo-Labeling) to boost V17.")
    print(f"      1. Train V17 Base.")
    print(f"      2. Select confident Test Preds (> {CFG.PL_THRESHOLD_HIGH} or < {CFG.PL_THRESHOLD_LOW}).")
    print(f"      3. Retrain V17 with Train + Pseudo-Labeled Test.")
    print(f"================================================================================")
    
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH
    
    if not os.path.exists(train_path):
        train_path = "train.csv"
        test_path = "test.csv"
        orig_path = "Heart_Disease_Prediction.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    try:
        orig = pd.read_csv(orig_path)
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. Feature Engineering (Standard V17 Deotte)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).astype(str) # CatBoost handles strings well
    
    # Num to Cat
    for col in NUMS:
        for df in [train, test, orig]:
            df[f'CAT_{col}'] = df[col].astype(str)

    ALL_CATS = CATS + [f'FREQ_{c}' for c in NUMS] + [f'CAT_{c}' for c in NUMS]
    
    # 3. Stage 1: Base Model Training
    # Data Prep: Train + Orig
    X_train_full = pd.concat([train, orig], axis=0).reset_index(drop=True)
    y_train_full = X_train_full['Heart Disease'].values
    X_train_full = X_train_full.drop(columns=['Heart Disease', 'id'], errors='ignore')
    X_test = test.drop(columns=['id'], errors='ignore')
    
    # Ensure types for CatBoost
    for c in ALL_CATS:
        X_train_full[c] = X_train_full[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    # Train Stage 1
    # Note: We need OOF for Train Only to check CV, but we train on Train+Orig?
    # V17 logic: Train on Train (Augmented with Orig inside folds) or Train on Train+Orig?
    # V17 used "Augment with Original" inside the fold loop.
    # To keep it simple for Pseudo-Labeling, we'll concatenation upfront but use StratifiedKFold on 'Train' part.
    # Actually, let's stick to V17's exact loop structure for consistency, but simplified.
    # The helper function above does a simple KFold on the passed X.
    # Let's pass X_train (original train) and augment inside the helper?
    # No, let's just pass X_train_full (Train + Orig) and accept that OOF is on mixed data?
    # Better: Use the V17 loop logic exactly.
    
    # Let's Refine Logic:
    # 1. Train V17 exactly as defined in V17 script (Train w/ Inner Augment).
    # 2. Get Test Preds.
    
    print("\n--- STAGE 1: Generating Initial Predictions ---")
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    test_pred_stage1 = np.zeros(len(test))
    
    # We will simulate V17 training
    # Prepare X, y form Train
    X = train.drop(columns=['Heart Disease', 'id'])
    y = train['Heart Disease'].values
    
    # Prepare Orig
    X_orig = orig.drop(columns=['Heart Disease', 'id'], errors='ignore')
    y_orig = orig['Heart Disease'].values
    
    # Preprocess strings
    for c in ALL_CATS:
        X[c] = X[c].astype(str)
        X_orig[c] = X_orig[c].astype(str)
        X_test[c] = X_test[c].astype(str)
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # Augment
        X_tr_aug = pd.concat([X_tr, X_orig], axis=0)
        y_tr_aug = np.concatenate([y_tr, y_orig], axis=0)
        
        train_pool = cb.Pool(X_tr_aug, y_tr_aug, cat_features=ALL_CATS)
        val_pool = cb.Pool(X_val, y_val, cat_features=ALL_CATS)
        
        model = cb.CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        
        test_pred_stage1 += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS
        if fold == 0: print("Stage 1 Fold 1 done.")

    # 4. Filter Pseudo-Labels
    high_conf_mask = (test_pred_stage1 > CFG.PL_THRESHOLD_HIGH) | (test_pred_stage1 < CFG.PL_THRESHOLD_LOW)
    X_pl = X_test[high_conf_mask].copy()
    # Hard labels usually better? Or Soft? Let's use Hard Labels for CatBoost
    y_pl = (test_pred_stage1[high_conf_mask] > 0.5).astype(int)
    
    print(f"\n--- PSEUDO-LABELING ---")
    print(f"Total Test Samples: {len(test)}")
    print(f"Confident Samples: {len(X_pl)} ({len(X_pl)/len(test):.1%})")
    print(f"Thresholds: < {CFG.PL_THRESHOLD_LOW} or > {CFG.PL_THRESHOLD_HIGH}")
    
    if len(X_pl) == 0:
        print("No comfortable samples found! Exiting.")
        return

    # 5. Stage 2: Retraining
    print("\n--- STAGE 2: Retraining with Pseudo-Labels ---")
    
    # We add PL to the Orig augmentation pool? Or to the Train pool?
    # Best practice: Add PL to Train data, but NOT to Validation set.
    # So we augment X_tr inside the fold with X_pl.
    
    oof_stage2 = np.zeros(len(X))
    test_pred_stage2 = np.zeros(len(test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # Augment: Train + Orig + PL
        X_tr_aug = pd.concat([X_tr, X_orig, X_pl], axis=0)
        y_tr_aug = np.concatenate([y_tr, y_orig, y_pl], axis=0)
        
        # Shuffle
        p = np.random.permutation(len(X_tr_aug))
        X_tr_aug = X_tr_aug.iloc[p]
        y_tr_aug = y_tr_aug[p]
        
        train_pool = cb.Pool(X_tr_aug, y_tr_aug, cat_features=ALL_CATS)
        val_pool = cb.Pool(X_val, y_val, cat_features=ALL_CATS)
        
        model = cb.CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_stage2[val_idx] = val_preds
        test_pred_stage2 += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

    # Results
    final_score = roc_auc_score(y, oof_stage2)
    print(f"\nStage 2 CV AUC: {final_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_pred_stage2})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof_stage2})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
