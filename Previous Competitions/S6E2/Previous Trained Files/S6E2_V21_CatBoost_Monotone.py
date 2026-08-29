
# 1. Import RAPIDS (Must be first)
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✅ cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("⚠️ cuDF not found. Falling back to standard pandas.")
    pass
except Exception as e:
    print(f"⚠️ cuDF failed to load: {e}")
    print("Falling back to standard pandas.")
    pass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
import os
import gc

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V21"
    DESCRIPTION = "CatBoost_Monotonic_Constraints"
    
    # Base V17 Params
    CAT_PARAMS = {
        'iterations': 50000,
        'learning_rate': 0.0025,
        'depth': 3,
        'subsample': 0.8,
        'random_seed': 42,
        'early_stopping_rounds': 1000,
        'eval_metric': 'AUC',
        'task_type': 'CPU', # GPU failed with constraints. Fallback to CPU is mandatory.
        # 'grow_policy': 'Depthwise', # Not needed for CPU
        'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False
    }
    
    SEED = 42
    N_FOLDS = 5 # CPU is slower, use 5 folds
    INNER_FOLDS = 5
    MONOTONIC_THRESHOLD = 0.15 
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

def get_monotonic_constraints(train_df, target_col, threshold=0.15):
    """
    Calculates Spearman correlation and returns a dictionary of monotonic constraints.
    1: Increasing constraint (Positive correlation)
    -1: Decreasing constraint (Negative correlation)
    0: No constraint
    """
    print(f"\nAnalyzing Monotonic Relationships (Threshold: |{threshold}|)...")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    corrs = train_df[numeric_cols + [target_col]].corr(method='spearman')[target_col].drop(target_col)
    
    constraints = {}
    print(f"{'Feature':<30} | {'Corr':<8} | {'Constraint'}")
    print("-" * 55)
    
    for feat, corr in corrs.items():
        constraint = 0
        if corr > threshold:
            constraint = 1
        elif corr < -threshold:
            constraint = -1
            
        if constraint != 0:
            constraints[feat] = constraint
            print(f"{feat:<30} | {corr:.4f}   | {constraint}")
    
    return constraints

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Enforce domain logic (Monotonic Constraints) to regularize V17.")
    print(f"      Features with high correlation will be forced to be monotonic.")
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
    else:
        print(f"Loading from Kaggle: {train_path}")

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
    NUM_AS_CAT = []
    TO_REMOVE = []

    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency Encoding
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # Numerical as Categorical
    for col in NUMS:    
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str).astype('category')

    FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT
    STATS = ['mean']
    TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS
    TO_REMOVE += NUM_AS_CAT + CATS + NEW_CATS

    # 3. Determine Constraints (Pre-Loop)
    constraints_dict = get_monotonic_constraints(train, 'Heart Disease', CFG.MONOTONIC_THRESHOLD)
    
    # CRITICAL FIX: Remove constraints for features that will be DROPPED
    # The Deotte strategy drops RAW CATEGORICALS (CATS).
    # CatBoost will error if we define a constraint for a missing feature.
    cols_to_drop = set(TO_REMOVE)
    final_constraints = {k: v for k, v in constraints_dict.items() if k not in cols_to_drop}
    
    print(f"\nFiltered Constraints (Excluding Dropped Features):")
    print("-" * 55)
    for feat, const in final_constraints.items():
        print(f"{feat:<30} | {const}")
    
    CFG.CAT_PARAMS['monotone_constraints'] = final_constraints

    # 4. Validation Loop
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []
    
    X_orig = orig[FEATURES+['Heart Disease']].copy()
    y_orig = orig['Heart Disease'].copy()
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")
    
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        
        # Outer Split
        X_train = train.loc[train_index, FEATURES+['Heart Disease']].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease'] # Series
        
        # Augment
        X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()
        
        X_val = train.loc[val_index, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease'] # Series

        X_test_fold = test[FEATURES].reset_index(drop=True).copy()

        # Inner CV for TE
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        
        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
            
            X_train2 = X_train.loc[train_index2, FEATURES + ['Heart Disease']].copy()
            X_val2   = X_train.loc[val_index2, FEATURES].copy()
            
            # --- TE Feature Set 1 ---
            for col in TE_COLUMNS:
                tmp = X_train2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                
                # Merge to Inner Validation Chunk
                X_val2 = X_val2.merge(tmp, on=col, how="left") 
            
                # Assign back to Main X_train
                for c in tmp.columns:
                    X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")

        # Outer TE (Val & Test)
        for col in TE_COLUMNS:
            tmp = X_train.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            
            X_val = X_val.merge(tmp, on=col, how="left")
            X_test_fold = X_test_fold.merge(tmp, on=col, how="left")
    
        # Final Prep
        current_cols = X_train.columns.tolist()
        drop_cols_train = [c for c in TO_REMOVE if c in current_cols]
        X_train.drop(columns=drop_cols_train, inplace=True)
        
        drop_cols_val = [c for c in TO_REMOVE if c in X_val.columns]
        X_val.drop(columns=drop_cols_val, inplace=True)
        
        drop_cols_test = [c for c in TO_REMOVE if c in X_test_fold.columns]
        X_test_fold.drop(columns=drop_cols_test, inplace=True)

        if 'Heart Disease' in X_train.columns:
            X_train = X_train.drop(['Heart Disease'], axis=1)
        
        # Train
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        model = CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False,
            use_best_model=True
        )
        
        # Predict
        val_p = model.predict_proba(X_val)[:,1]
        oof[val_index] = val_p

        roc_auc_fold = roc_auc_score(y_val, val_p)
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f}")

        pred += model.predict_proba(X_test_fold)[:,1] / CFG.N_FOLDS
        
        # Cleanup
        del X_train, X_val, X_test_fold, model, train_pool, val_pool
        gc.collect()

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
