
import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V26"
    DESCRIPTION = "LGBM_DART_Deotte"
    
    SEED = 42
    N_FOLDS = 10
    INNER_FOLDS = 5
    
    # LightGBM DART Params
    LGBM_PARAMS = {
        'boosting_type': 'dart', # <--- The Key Change
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05, # DART often likes higher LR than GBDT, or longer training
        'n_estimators': 3000, 
        'max_depth': 6,
        'num_leaves': 31,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'drop_rate': 0.1,         # Dropout rate
        'skip_drop': 0.5,         # Probability to skip dropout
        'xgboost_dart_mode': True,
        'uniform_drop': False,
        'random_state': 42,
        'n_jobs': -1,
        'device': 'gpu',          # GPU acceleration
        'verbose': -1
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

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train LightGBM with DART (Dropout) for high diversity.")
    print(f"      DART drops trees during training, acting like a Neural Network regularization.")
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

    # 2. Feature Engineering (Deotte Recipe)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0)
    
    # Num to Cat for TE
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS
    STATS = ['mean']

    # 3. CV Loop
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    
    # Prepare Data
    X_orig = orig.drop(columns=['Heart Disease', 'id'], errors='ignore')
    y_orig = orig['Heart Disease'].values
    
    # Handle Categoricals for LGBM (Set as 'category' dtype)
    # Actually, for TE augmentation, we'll produce floats.
    # The raw CATS should be cast to category.
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")
    
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        
        # Outer Split
        X_train = train.iloc[train_index].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease'].values
        
        # Augment
        X_train_aug = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train_aug = np.concatenate([y_train, y_orig], axis=0)
        y_train_aug_series = pd.Series(y_train_aug, index=X_train_aug.index) # Helper for groupby
        
        X_val = train.iloc[val_index].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease'].values
        X_test_fold = test.copy()

        # Inner TE Calculation
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feature_names = [f"TE1_{col}_{s}" for col in TE_COLUMNS for s in STATS]
        
        for df in [X_train_aug, X_val, X_test_fold]:
            for c in te_feature_names:
                df[c] = 0.0

        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train_aug)):
            X_tr2 = X_train_aug.iloc[train_index2]
            X_val2 = X_train_aug.iloc[val_index2]
            y_tr2 = y_train_aug_series.iloc[train_index2]
            
            # We temporarily join Y to X_tr2 for easy groupby
            # But wait, X_tr2 is a copy? Yes.
            X_tr2_y = X_tr2.copy()
            X_tr2_y['target'] = y_tr2
            
            for col in TE_COLUMNS:
                tmp = X_tr2_y.groupby(col)['target'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                merged = X_val2[[col]].merge(tmp, on=col, how='left')[tmp.columns]
                for c in tmp.columns:
                    X_train_aug.loc[val_index2, c] = merged[c].values

        # Outer TE
        X_aug_y = X_train_aug.copy()
        X_aug_y['target'] = y_train_aug
        
        for col in TE_COLUMNS:
            tmp = X_aug_y.groupby(col)['target'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            
            merged_val = X_val[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns: X_val[c] = merged_val[c].values
            
            merged_test = X_test_fold[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns: X_test_fold[c] = merged_test[c].values

        # Prepare for LGBM
        # Drop non-feature cols
        drop_cols = ['Heart Disease', 'id'] + [c for c in X_train_aug.columns if c.startswith('CAT_')] # Drop the string helper cols
        # Wait, we need Raw Cats for LGBM? Yes. Lgbm handles 'category'.
        # We only created CAT_cols for TE logic primarily, but we kept raw CATS.
        
        features = [c for c in X_train_aug.columns if c not in ['Heart Disease', 'id', 'target'] and not c.startswith('CAT_')]
        
        # Cast categories
        for c in CATS:
            X_train_aug[c] = X_train_aug[c].astype('category')
            X_val[c] = X_val[c].astype('category')
            X_test_fold[c] = X_test_fold[c].astype('category')

        # Train
        # Note: DART does not support Early Stopping easily with `callbacks`.
        # We just run for fixed estimators usually, or use a very large window.
        # However, standard lgb.train supports it.
        
        train_data = lgb.Dataset(X_train_aug[features], label=y_train_aug, categorical_feature=CATS)
        val_data = lgb.Dataset(X_val[features], label=y_val, categorical_feature=CATS, reference=train_data)
        
        callbacks = [
            lgb.log_evaluation(period=200),
            lgb.early_stopping(stopping_rounds=300) 
        ]
        
        model = lgb.train(
            CFG.LGBM_PARAMS,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        val_preds = model.predict(X_val[features], num_iteration=model.best_iteration)
        oof[val_index] = val_preds
        pred += model.predict(X_test_fold[features], num_iteration=model.best_iteration) / CFG.N_FOLDS
        
        score = roc_auc_score(y_val, val_preds)
        print(f"Fold {i+1} AUC: {score:.5f}")
        
        del model, X_train_aug, X_val, X_test_fold
        gc.collect()

    # Overall
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
