import numpy as np
import pandas as pd
import re
import gc
import warnings
import os
import time
from sklearn.preprocessing import KBinsDiscretizer, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V39"
    DESCRIPTION = "CatBoost_Ordered_GlobalStats"
    
    N_FOLDS = 5
    SEEDS = [42]
    TARGET = 'Heart Disease'
    ID = 'id'
    
    # Paths
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    
    # CatBoost Params
    CB_PARAMS = {
        'iterations': 8000,
        'learning_rate': 0.015,
        'depth': 5,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.5,
        'boosting_type': 'Ordered',  # Key hyperparameter
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 200,
        'task_type': 'GPU',
        'random_seed': 42,
        'verbose': 500
    }
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

def normalize_cols(df):
    """Normalize column names"""
    df = df.copy()
    df.columns = [re.sub(r"[^\w\s]", "", c.strip().lower()).replace(" ", "_") for c in df.columns]
    return df

def apply_feature_engineering(df, stats_mean, stats_count, global_mean, num_cols, cat_cols, is_train=False):
    """
    Apply feature engineering:
    1. Global Target Statistics (Leakage features handled by Ordered Boosting)
    2. Frequency Encoding
    3. Uniform KBins Discretization
    4. Robust Scaling
    """
    out = df.copy()
    
    norm = lambda x: re.sub(r"[^\w\s]", "", x.strip().lower()).replace(" ", "_")

    # 1. Global Target Statistics
    for col in num_cols + cat_cols:
        col_norm = norm(col)
        out[f'mean_{col_norm}'] = out[col_norm].map(stats_mean.get(col, {})).fillna(global_mean)
        out[f'count_{col_norm}'] = out[col_norm].map(stats_count.get(col, {})).fillna(0)
    
    # 2. Frequency Encoding
    for col in num_cols + cat_cols:
        col_norm = norm(col)
        if is_train:
            freq = out[col_norm].value_counts(normalize=True).to_dict()
            if not hasattr(apply_feature_engineering, 'freqs'):
                apply_feature_engineering.freqs = {}
            apply_feature_engineering.freqs[col] = freq
        else:
            freq = getattr(apply_feature_engineering, 'freqs', {}).get(col, {})
            
        out[f'freq_{col_norm}'] = out[col_norm].map(freq).fillna(0)
        
    # 3. Uniform KBins Discretization
    bin_targets = [norm(c) for c in num_cols]
    
    if is_train:
        kbd = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal')
        apply_feature_engineering.kbd = kbd
        try:
            out[[f'bin_{c}' for c in bin_targets]] = kbd.fit_transform(out[bin_targets]).astype(int)
        except Exception as e:
            print(f"Error in KBinsDiscretizer: {e}")
    else:
        out[[f'bin_{c}' for c in bin_targets]] = apply_feature_engineering.kbd.transform(out[bin_targets]).astype(int)
        
    # 4. Robust Scaling
    if is_train:
        rs = RobustScaler()
        apply_feature_engineering.rs = rs
        out[bin_targets] = rs.fit_transform(out[bin_targets])
    else:
        out[bin_targets] = apply_feature_engineering.rs.transform(out[bin_targets])
        
    return out

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()
    
    # Load Data
    # Load Data
    train_raw = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "Dataset/train.csv")
    test_raw = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "Dataset/test.csv")

    print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")
    
    # Normalize Columns
    train = normalize_cols(train_raw)
    test = normalize_cols(test_raw)
    
    # Columns
    target_col = re.sub(r"[^\w\s]", "", CFG.TARGET.strip().lower()).replace(" ", "_")
    cat_cols = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
                'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
    num_cols = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
    
    # Map Target
    if train[target_col].dtype == 'object':
         train[target_col] = train[target_col].map({'Presence': 1, 'Absence': 0})
    
    # Global Statistics (Leakage)
    print("Computing global statistics...")
    global_mean = train[target_col].mean()
    stats_mean = {}
    stats_count = {}
    
    norm = lambda x: re.sub(r"[^\w\s]", "", x.strip().lower()).replace(" ", "_")

    for col in num_cols + cat_cols:
        col_norm = norm(col)
        stats_mean[col] = train.groupby(col_norm)[target_col].mean().to_dict()
        stats_count[col] = train.groupby(col_norm)[target_col].count().to_dict()
        
    # Feature Engineering
    print("Applying feature engineering...")
    apply_feature_engineering.freqs = {}
    
    train_fe = apply_feature_engineering(train, stats_mean, stats_count, global_mean, num_cols, cat_cols, is_train=True)
    test_fe = apply_feature_engineering(test, stats_mean, stats_count, global_mean, num_cols, cat_cols, is_train=False)
    
    # Categorical Features
    cat_cols_norm = [norm(c) for c in cat_cols]
    num_cols_norm = [norm(c) for c in num_cols]
    ordinal_cols = [f'bin_{c}' for c in num_cols_norm] + cat_cols_norm
    
    for df in [train_fe, test_fe]:
        for c in ordinal_cols:
            df[c] = df[c].astype(str).astype('category')
        
    id_col = CFG.ID.lower() 
    features = [c for c in train_fe.columns if c not in [id_col, target_col]]
    
    print(f"Features: {len(features)}")
    
    X = train_fe[features]
    y = train_fe[target_col]
    X_test = test_fe[features]
    
    # Arrays
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    # Training
    for seed in CFG.SEEDS:
        print(f"\nRunning Seed: {seed}")
        skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
        
        val_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            params = CFG.CB_PARAMS.copy()
            params['random_seed'] = seed
            
            model = CatBoostClassifier(**params)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                cat_features=ordinal_cols,
                use_best_model=True
            )
            
            val_pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_pred 
            test_preds += model.predict_proba(X_test)[:, 1] / (len(CFG.SEEDS) * CFG.N_FOLDS)
            
            score = roc_auc_score(y_val, val_pred)
            val_aucs.append(score)
            print(f"Fold {fold+1} AUC: {score:.5f}")
            
        print(f"Seed {seed} Mean AUC: {np.mean(val_aucs):.5f}")

    # Results
    oof_score = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {oof_score:.5f}")
    
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    
    # Save
    pd.DataFrame({'id': train[id_col], 'target': y, 'pred': oof_preds}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test[id_col], 'Heart Disease': test_preds}).to_csv(CFG.SUBMISSION_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
