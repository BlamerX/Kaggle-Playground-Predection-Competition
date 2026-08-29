
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
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import re
import os
import gc

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V45"
    DESCRIPTION = "LightGBM_V12Plus"
    
    # V12 EXACT recipe (LB 0.95378) + 2 improvements:
    #   1. Original data augmentation (proven +0.00003 for CatBoost V17 vs V11)
    #   2. Frequency encoding (proven +0.00020 from V42 greedy growth)
    #   3. 15-fold instead of 5 (more stable ensemble)
    
    PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,          # V12 EXACT (matches V11 XGB's 0.084)
        'num_leaves': 4,                # Stumps (2^2)
        'max_depth': 2,                 # Explicit Stumps
        'min_child_samples': 20,
        'subsample': 0.7,              # V12 EXACT
        'colsample_bytree': 0.6,       # V12 EXACT
        'reg_alpha': 0.1,              # V12 EXACT
        'reg_lambda': 4.0,             # V12 EXACT (high regularization)
        'n_estimators': 5000,          # More rounds (V12 used 3000)
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
    }
    
    EARLY_STOPPING = 300     # V12 used 200, give a bit more patience
    
    SEED = 42
    N_FOLDS = 15             # V12 used 5, more folds = larger ensemble
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# PREPROCESSING — V12 OHE + StandardScaler (EXACT), then add FREQ
# ==================================================================================
def _fix_cols(df, keep=("id", "Heart Disease")):
    keep = set(keep)
    new_cols = []
    for c in df.columns:
        if c in keep:
            new_cols.append(c)
        else:
            s = str(c).strip()
            s = re.sub(r"\s+", "_", s)
            new_cols.append(s)
    df = df.copy()
    df.columns = new_cols
    return df

def preprocess(train, test, orig):
    """V12 exact preprocessing + original data augmentation + FREQ encoding."""
    print("Applying V12 Preprocessing (OHE + Scaling) + FREQ encoding...")
    
    # 0. Map target
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if len(orig) > 0 and orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    
    # 1. Frequency Encoding BEFORE OHE (on raw numeric columns)
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 
            'Number of vessels fluro', 'EKG results']
    
    freq_cols = []
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        freq_cols.append(f'FREQ_{col}')
    print(f"  Added {len(freq_cols)} FREQ features")
    
    # 2. Augment train with original data BEFORE OHE
    y_train = train['Heart Disease'].copy()
    y_orig = orig['Heart Disease'].copy()
    y = pd.concat([y_train, y_orig], axis=0, ignore_index=True)
    
    train_nontarget = train.drop(columns=['Heart Disease'])
    orig_nontarget = orig.drop(columns=['Heart Disease'])
    
    # Mark original data for tracking
    train_aug = pd.concat([train_nontarget, orig_nontarget], axis=0, ignore_index=True)
    
    # 3. Concat all for OHE (train_aug + test)
    n_train = len(train_aug)
    full = pd.concat([train_aug, test], axis=0, ignore_index=True)
    
    # 4. One-Hot Encoding (V12 exact)
    cat_cols = full.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'id']  # Don't OHE 'id'
    full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)
    
    # 5. Split back
    train_encoded = full_encoded.iloc[:n_train].copy()
    test_encoded  = full_encoded.iloc[n_train:].copy()
    
    # 6. Fix column names
    train_encoded = _fix_cols(train_encoded, keep=["id"])
    test_encoded = _fix_cols(test_encoded, keep=["id"])
    
    # 7. StandardScaler (V12 exact — scale cols with >2 unique values)
    scaler = StandardScaler()
    num_cols = [
        c for c in train_encoded.columns 
        if c != "id" 
        and np.issubdtype(train_encoded[c].dtype, np.number) 
        and train_encoded[c].nunique() > 2
    ]
    
    print(f"  Scaling {len(num_cols)} numerical features")
    train_encoded[num_cols] = scaler.fit_transform(train_encoded[num_cols])
    test_encoded[num_cols]  = scaler.transform(test_encoded[num_cols])
    
    features = [c for c in train_encoded.columns if c != 'id']
    print(f"  Total features: {len(features)} (OHE + FREQ)")
    
    return train_encoded, test_encoded, y, features, len(train)

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Recipe: V12 Stumps (LB 0.95378) + Original Data + FREQ + 15-fold")
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

    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original shape: {orig.shape}")

    # 2. Preprocess — V12 exact + improvements
    X, X_test, y, features, n_synth = preprocess(train, test, orig)
    
    # 3. Cross Validation — StratifiedKFold (V12 exact)
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    scores = []
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold StratifiedKFold CV...")
    print(f"  Train: {len(X)} rows ({n_synth} synthetic + {len(X) - n_synth} original)")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr = X.iloc[train_idx][features]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx][features]
        y_val = y.iloc[val_idx]
        
        # LGBMClassifier (V12 exact API)
        model = lgb.LGBMClassifier(**CFG.PARAMS)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=CFG.EARLY_STOPPING, verbose=False)]
        )
        
        val_p = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_p
        test_preds += model.predict_proba(X_test[features])[:, 1] / CFG.N_FOLDS
        
        score = roc_auc_score(y_val, val_p)
        scores.append(score)
        print(f"  Fold {fold+1} AUC: {score:.5f} (best_iter: {model.best_iteration_})")
        
        del X_tr, X_val, model
        gc.collect()

    # Overall & Save
    overall_score = roc_auc_score(y, oof_preds)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\n{'='*60}")
    print(f"Overall OOF AUC: {overall_score:.5f}")
    print(f"Mean Fold AUC: {mean_score:.5f} ± {std_score:.5f}")
    print(f"{'='*60}")
    print(f"\nComparison:")
    print(f"  V12 (Stumps, 5-fold, synth only):     CV 0.95558 / LB 0.95378")
    print(f"  V45 (Stumps+Orig+FREQ, {CFG.N_FOLDS}-fold):  CV {overall_score:.5f}")
    
    # Save using original train IDs only (not augmented)
    sub = pd.DataFrame({'id': X_test['id'].values, 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': X['id'].values, 'target': y.values, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
