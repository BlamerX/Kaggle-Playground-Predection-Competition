
import os
import gc
import sys
import subprocess
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Install pytabkit if missing
try:
    from pytabkit import TabM_D_Classifier
    print("✅ PyTabKit loaded successfully!")
except ImportError:
    print("📦 Installing PyTabKit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Classifier
    print("✅ PyTabKit installed & loaded!")

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V23"
    DESCRIPTION = "TabM_Deotte_Hybrid"
    
    SEED = 42
    N_FOLDS = 5 # Standard for NNs
    INNER_FOLDS = 5 # For TE
    
    # TabM Hyperparameters (Adapted from s6e1_v61.py + Classification)
    # Using 'tabm-mini-normal' as a robust baseline
    TABM_PARAMS = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbosity': 1,
        'arch_type': 'tabm-mini-normal',
        'tabm_k': 32, # Batch Ensemble Size
        'num_emb_type': 'pwl', # Piecewise Linear Embeddings for Numericals
        'd_embedding': 24, 
        'batch_size': 512, 
        'lr': 1e-3, 
        'n_epochs': 50, # Classification might need more epochs
        'dropout': 0.2,
        'd_block': 256, 
        'n_blocks': 3,
        'patience': 10,
        'weight_decay': 1e-3,
        'random_state': 42,
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.SEED)

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train TabM (Tabular Deep Learning) using Hybrid Features.")
    print(f"      1. Deotte Features (Freq + TE) as Numericals (Strong Signal).")
    print(f"      2. Raw Categoricals (Ordinal) for TabM Embeddings.")
    print(f"================================================================================")
    
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH
    
    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
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

    # Map Target
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. Feature Engineering Setup (EXACTLY MATCHING DEOTTE / V17)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    NEW_NUMS = []
    NEW_CATS = []
    
    # Important: For TabM, we want Raw Cats to remain as Strings initially for OrdinalEncoding
    # But for Deotte TE, we need copies.
    
    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency Encoding
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # Numerical as Categorical (For TE generation)
    # We will GENERATE TE features, but we will NOT replace the original columns.
    # We add TE features as *New Numericals*.
    
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS # We will calc TE for both Raw Cats and Num-Cats
    STATS = ['mean']

    # 3. Preparation for Inner Loop TE
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []
    
    X_orig = orig.copy()
    y_orig = orig['Heart Disease'].copy()
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")
    
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        
        # Outer Split
        X_train = train.iloc[train_index].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease'].values
        
        # Augment with Original
        X_train_aug = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train_aug = np.concatenate([y_train, y_orig], axis=0) # Numpy concat for labels
        
        X_val = train.iloc[val_index].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease'].values

        X_test_fold = test.copy()

        # Inner CV for TE (On Augmented Data)
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        
        # We need to add TE columns to X_train_aug, X_val, X_test_fold
        # Initialize TE columns
        te_feature_names = []
        for col in TE_COLUMNS:
            for s in STATS:
                te_feature_names.append(f"TE1_{col}_{s}")
        
        # Create holders
        for df in [X_train_aug, X_val, X_test_fold]:
            for c in te_feature_names:
                df[c] = 0.0

        # Perform Inner TE Calculation on X_train_aug
        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train_aug)):
            
            X_tr2 = X_train_aug.iloc[train_index2]
            X_val2 = X_train_aug.iloc[val_index2]
            
            # Calc TE
            for col in TE_COLUMNS:
                # Calculate mean on inner train
                tmp = X_tr2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                
                # Update inner val
                merged = X_val2[[col]].merge(tmp, on=col, how='left')[tmp.columns]
                for c in tmp.columns:
                    X_train_aug.loc[val_index2, c] = merged[c].values

        # Outer TE for Val and Test (using full X_train_aug)
        for col in TE_COLUMNS:
            tmp = X_train_aug.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            
            # Update Val
            merged_val = X_val[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns:
                X_val[c] = merged_val[c].values
                
            # Update Test
            merged_test = X_test_fold[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns:
                X_test_fold[c] = merged_test[c].values

        # 4. Final Feature Selection for TabM
        # Inputs:
        # A. Raw Categoricals (CATS) -> Ordinal Encoded -> cat_col_names
        # B. Numericals (NUMS + NEW_NUMS + TE columns) -> Standard Scaled -> num_cols
        
        ALL_NUMS = NUMS + NEW_NUMS + te_feature_names
        ALL_CATS = CATS 
        
        # Preprocessing: Ordinal Encoding for CATS
        # TabM handles raw strings if we pass cat_col_names? No, pytabkit implies numpy arrays usually?
        # Let's verify s6e1_v61.py. It used OrdinalEncoder.
        # "encoder.fit(train_eng[CATS])"
        # "cats_encoded = pd.DataFrame(encoder.transform(df_eng[CATS])..."
        # So we MUST Ordinal Encode.
        
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        # Note: Fit on ALL available data for consistent encoding
        # (Train + Val + Test + Orig) to avoid -1s as much as possible?
        # Strict way: Fit on Train_Aug.
        encoder.fit(X_train_aug[ALL_CATS].astype(str))
        
        X_tr_cat = encoder.transform(X_train_aug[ALL_CATS].astype(str))
        X_val_cat = encoder.transform(X_val[ALL_CATS].astype(str))
        X_test_cat = encoder.transform(X_test_fold[ALL_CATS].astype(str))
        
        # Preprocessing: Scaling for NUMS
        # FillNa before scaling
        X_train_aug[ALL_NUMS] = X_train_aug[ALL_NUMS].fillna(0).astype('float32')
        X_val[ALL_NUMS] = X_val[ALL_NUMS].fillna(0).astype('float32')
        X_test_fold[ALL_NUMS] = X_test_fold[ALL_NUMS].fillna(0).astype('float32')
        
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_train_aug[ALL_NUMS])
        X_val_num = scaler.transform(X_val[ALL_NUMS])
        X_test_num = scaler.transform(X_test_fold[ALL_NUMS])
        
        # Concatenate for TabM
        # Order: Nums then Cats usually? Or pass explicit col names.
        # PyTabKit's `fit` takes `cat_col_names`.
        # If we pass a DataFrame, it figures it out.
        # Let's reconstruct DataFrames.
        
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=ALL_NUMS + ALL_CATS)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=ALL_NUMS + ALL_CATS)
        X_test_final = pd.DataFrame(np.hstack([X_test_num, X_test_cat]), columns=ALL_NUMS + ALL_CATS)
        
        # Fix types: Cats must be integers? Or does TabM handle floats if we say they are cats?
        # Safe bet: Ints for cats.
        for c in ALL_CATS:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_test_final[c] = X_test_final[c].astype(int)

        # Train TabM
        model = TabM_D_Classifier(**CFG.TABM_PARAMS)
        
        # PyTabKit fit signature: X, y, X_val=None, y_val=None, cat_col_names=None
        model.fit(
            X_tr_final, y_train_aug, 
            X_val=X_val_final, y_val=y_val, 
            cat_col_names=ALL_CATS
        )
        
        # Predict
        val_probs = model.predict_proba(X_val_final)[:, 1]
        oof[val_index] = val_probs
        
        test_probs = model.predict_proba(X_test_final)[:, 1]
        pred += test_probs / CFG.N_FOLDS
        
        roc_auc_fold = roc_auc_score(y_val, val_probs)
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f}")
        
        del model, X_tr_final, X_val_final, X_test_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Overall & Save
    overall_score = roc_auc_score(train['Heart Disease'], oof)
    print(f"\nOverall CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
