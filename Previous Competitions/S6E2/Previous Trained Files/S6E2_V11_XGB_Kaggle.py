import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import re
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V11"
    DESCRIPTION = "XGB_Kaggle_Clone_Exact"
    
    # ------------------------------------------------------------------------------
    # PUBLIC NOTEBOOK REPLICATION
    # Source: xgb-lgb-options-w-optuna-tuning-disease (LB 0.95376)
    # Key Factors:
    # 1. One-Hot Encoding + StandardScaler (vs our Raw/LabelEnc)
    # 2. Max Depth = 2 (Stumps)
    # 3. Synth Data Only (No Original Augmentation)
    # ------------------------------------------------------------------------------
    PARAMS = {
        'learning_rate': 0.08438590925890956,
        'max_depth': 2,
        'min_child_weight': 0.011965414914744715,
        'subsample': 0.7061263810064461,
        'colsample_bytree': 0.5985469566596684,
        'gamma': 1.3490094227070655,
        'reg_alpha': 0.11833977171171162,
        'reg_lambda': 4.064070444500402,
        'seed': 42,
        'nthread': -1,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'tree_method': 'hist'
    }
    
    # EXACT BEST ITERATION FROM NOTEBOOK
    N_ESTIMATORS = 2730 
    
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "Heart_Disease" # Note underscore from fixing cols
    
    # PATHS
    TRAIN_PATH = "train.csv"
    TEST_PATH = "test.csv"
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# PUBLIC NOTEBOOK 'EXACT' PREPROCESSING
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

def preprocess_exact(train, test):
    print("Applying Exact Public Notebook Preprocessing...")
    
    # 1. Target Separation using Raw Names first
    y = train["Heart Disease"]
    train_nontarget = train.drop(columns=["Heart Disease"])
    
    # 2. Concat for Encoding
    full = pd.concat([train_nontarget, test], axis=0, ignore_index=True)
    
    # 3. One-Hot Encoding (pd.get_dummies)
    cat_cols = full.select_dtypes(include=['object', 'category']).columns.tolist()
    full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)
    
    # 4. Split back
    train_encoded = full_encoded.iloc[:len(train)].copy()
    test_encoded  = full_encoded.iloc[len(train):].copy()
    
    # 5. Fix Columns (Spaces to Underscores) -> THIS CHANGES TARGET NAME TO Heart_Disease
    # But wait, target was dropped. We need to re-attach it correctly or handle it separately.
    # The public notebook attaches y back to train_encoded BEFORE scaling (buggy warning).
    # We will fix cols on features first.
    
    train_encoded = _fix_cols(train_encoded, keep=["id"])
    test_encoded = _fix_cols(test_encoded, keep=["id"])
    
    # 6. Scaling (StandardScaler) on Numerical Columns
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
    
    # Load Raw
    if os.path.exists("/kaggle/input"):
        base_path = "/kaggle/input/playground-series-s6e2"
    else:
        base_path = "."
        
    train_raw = pd.read_csv(os.path.join(base_path, CFG.TRAIN_PATH))
    test_raw = pd.read_csv(os.path.join(base_path, CFG.TEST_PATH))
    
    # Preprocess EXACTLY like public notebook
    X, X_test, y_map = preprocess_exact(train_raw, test_raw)
    
    # Target Mapping (Public notebook used: Presence=1, Absence=0)
    # We can use the mapped user code:
    y = y_map.map({'Presence': 1, 'Absence': 0})
    
    # Convert to DMatrix for consistency if needed, but sklearn API is cleaner
    # We use Sklearn API to match our V1 style, but pass same params
    
    # ------------------------------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------------------------------
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    features = [c for c in X.columns if c != 'id'] # ID is preserved in X
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr = X.iloc[train_idx][features]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx][features]
        y_val = y.iloc[val_idx]
        
        # NOTE: No Original Data Augmentation (To match Public Notebook)
        
        # Model
        model = xgb.XGBClassifier(
            n_estimators=CFG.N_ESTIMATORS, 
            **CFG.PARAMS
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test[features])[:, 1] / CFG.N_FOLDS
        
        score = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} | AUC: {score:.5f}")
        
    # Overall
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.5f}")
    
    # Save
    sub = pd.DataFrame({'id': X_test['id'], 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': X['id'], 'Heart Disease': y, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    print(f"Saved {CFG.VERSION}. Mean CV: {overall_auc:.5f}")

if __name__ == "__main__":
    main()
