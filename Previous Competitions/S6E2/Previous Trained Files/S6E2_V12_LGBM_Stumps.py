import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import re
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V12"
    DESCRIPTION = "LGBM_Stumps_OHE"
    
    # ------------------------------------------------------------------------------
    # STRATEGY: V11 Recipe applied to LightGBM
    # 1. Preprocessing: OHE + StandardScaling (Exact V11 match)
    # 2. Architecture: Stumps (max_depth=2 => num_leaves=4)
    # 3. Data: Synthetic Only (No Original Augmentation)
    # ------------------------------------------------------------------------------
    PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,           # Similar to V11 XGB (0.084)
        'num_leaves': 4,                 # 2^2 = 4 (Stumps)
        'max_depth': 2,                  # Explicit Stumps
        'min_child_samples': 20,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.1,
        'reg_lambda': 4.0,               # High Regularization (Matches V11)
        'n_estimators': 3000,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1
    }
    
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "Heart_Disease" # Will be renamed during OHE
    
    # PATHS
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# PREPROCESSING (Exact V11 Replica)
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

def preprocess_stumps(train, test):
    print("Applying V11 'Stumps' Preprocessing (OHE + Scaling)...")
    
    # 1. Target Separation
    y = train["Heart Disease"]
    train_nontarget = train.drop(columns=["Heart Disease"])
    
    # 2. Concat for Encoding
    full = pd.concat([train_nontarget, test], axis=0, ignore_index=True)
    
    # 3. One-Hot Encoding
    cat_cols = full.select_dtypes(include=['object', 'category']).columns.tolist()
    full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)
    
    # 4. Split back
    train_encoded = full_encoded.iloc[:len(train)].copy()
    test_encoded  = full_encoded.iloc[len(train):].copy()
    
    # 5. Fix Columns
    train_encoded = _fix_cols(train_encoded, keep=["id"])
    test_encoded = _fix_cols(test_encoded, keep=["id"])
    
    # 6. Scaling (StandardScaler)
    scaler = StandardScaler()
    num_cols = [
        c for c in train_encoded.columns 
        if c != "id" 
        and np.issubdtype(train_encoded[c].dtype, np.number) 
        and train_encoded[c].nunique() > 2
    ]
    
    print(f"Scaling {len(num_cols)} numerical features...")
    train_encoded[num_cols] = scaler.fit_transform(train_encoded[num_cols])
    test_encoded[num_cols]  = scaler.transform(test_encoded[num_cols])
    
    return train_encoded, test_encoded, y

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()
    
    # Load Raw
    if os.path.exists("/kaggle/input"):
        base_path = "/kaggle/input/playground-series-s6e2"
    else:
        base_path = "."
        
    train_raw = pd.read_csv(os.path.join(base_path, CFG.TRAIN_PATH))
    test_raw = pd.read_csv(os.path.join(base_path, CFG.TEST_PATH))
    
    # Preprocess
    X, X_test, y_map = preprocess_stumps(train_raw, test_raw)
    
    # Target Mapping
    y = y_map.map({'Presence': 1, 'Absence': 0})
    
    # Features
    features = [c for c in X.columns if c != 'id']
    
    # ------------------------------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------------------------------
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr = X.iloc[train_idx][features]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx][features]
        y_val = y.iloc[val_idx]
        
        # Model
        model = lgb.LGBMClassifier(**CFG.PARAMS)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test[features])[:, 1] / CFG.N_FOLDS
        
        score = roc_auc_score(y_val, val_preds)
        scores.append(score)
        print(f"Fold {fold+1} | AUC: {score:.5f}")
        
    # Overall
    mean_score = np.mean(scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.5f}")
    
    # Save
    sub = pd.DataFrame({'id': X_test['id'], 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': X['id'], 'target': y, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    # ------------------------------------------------------------------------------
    # LOGGING (V1 Format)
    # ------------------------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved:")
    print(f"  {CFG.SUBMISSION_PATH}")
    print(f"  {CFG.OOF_PATH} (for ensemble use)")
    print(f"\nTotal time: {elapsed:.1f} minutes")

    print("\n" + "="*80)
    print(f"{CFG.VERSION} SUMMARY")
    print("="*80)
    print(f"\n| Version | Model | Features | CV AUC |")
    print(f"|---------|-------|----------|--------|")
    print(f"| **{CFG.VERSION}** | **LGBM** | **OHE+Stumps** | **{mean_score:.5f}** |")
    print(f"\n✅ {CFG.VERSION} Stumps ready for submission!")
    print("="*80)

if __name__ == "__main__":
    main()
