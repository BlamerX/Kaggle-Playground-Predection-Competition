
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import os
import re
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V14"
    DESCRIPTION = "Sklearn_Stumps_OHE"
    
    # ------------------------------------------------------------------------------
    # STRATEGY: Sklearn GradientBoosting Stumps
    # 1. Implementation: sklearn.ensemble.GradientBoostingClassifier 
    #    (Different logic than XGB/LGBM)
    # 2. Depth: 2 (Stumps)
    # 3. Preprocessing: OHE + Scaling (Manual)
    # ------------------------------------------------------------------------------
    PARAMS = {
        'n_estimators': 3000,
        'learning_rate': 0.05,            # Slower learning for Sklearn
        'max_depth': 2,                   # Stumps
        'subsample': 0.7,
        'min_samples_leaf': 20,
        'validation_fraction': 0.1,
        'n_iter_no_change': 100,
        'random_state': 42,
        'verbose': 0
    }
    
    SEED = 42
    N_FOLDS = 5
    
    # PATHS (Kaggle Standard)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    
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

def preprocess_v11(train, test):
    print("Applying V11 Preprocessing (OHE + Scaling) for Sklearn...")
    y = train["Heart Disease"]
    train_nontarget = train.drop(columns=["Heart Disease"])
    
    full = pd.concat([train_nontarget, test], axis=0, ignore_index=True)
    
    cat_cols = full.select_dtypes(include=['object', 'category']).columns.tolist()
    full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)
    
    train_encoded = full_encoded.iloc[:len(train)].copy()
    test_encoded  = full_encoded.iloc[len(train):].copy()
    
    train_encoded = _fix_cols(train_encoded, keep=["id"])
    test_encoded = _fix_cols(test_encoded, keep=["id"])
    
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
    
    # 1. Load Data
    if os.path.exists(CFG.TRAIN_PATH):
        print(f"Loading from Kaggle: {CFG.TRAIN_PATH}")
        train = pd.read_csv(CFG.TRAIN_PATH)
        test = pd.read_csv(CFG.TEST_PATH)
    else:
        print("Loading from Local (Fallback)...")
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
    
    # 2. Preprocess
    X, X_test, y_map = preprocess_v11(train, test)
    y = y_map.map({'Presence': 1, 'Absence': 0})
    
    features = [c for c in X.columns if c != 'id']
    
    # 3. Cross Validation
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx][features], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx][features], y.iloc[val_idx]
        
        # Sklearn Gradient Boosting
        model = GradientBoostingClassifier(**CFG.PARAMS)
        model.fit(X_tr, y_tr)
        
        # Note: Sklearn doesn't support 'eval_set' early stopping easily without warm_start loop
        # But 'n_iter_no_change' parameter works similarly in fit() if validation_fraction is set.
        # However, here we just fit on train (internal validation used for stopping if configured)
    
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
    # LOGGING
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
    print(f"| **{CFG.VERSION}** | **Sklearn** | **Stumps (Depth=2)** | **{mean_score:.5f}** |")
    print(f"\n✅ {CFG.VERSION} Stumps ready for submission!")
    print("="*80)

if __name__ == "__main__":
    main()
