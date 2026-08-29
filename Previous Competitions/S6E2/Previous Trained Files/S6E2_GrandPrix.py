# !pip install gplearn -q

import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import os
import re
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "GrandPrix"
    DESCRIPTION = "Genetic_Programming_Features"
    
    # ------------------------------------------------------------------------------
    # STRATEGY: Genetic Programming (Grand-Prix)
    # 1. Use gplearn to evolve new non-linear features
    # 2. Add top 10 evolved features to the dataset
    # 3. Train XGBoost on Extended Dataset
    # ------------------------------------------------------------------------------
    GP_PARAMS = {
        'population_size': 1000,
        'generations': 10,  # Keep it fast
        'tournament_size': 20,
        'stopping_criteria': 0.0,
        'const_range': (-1.0, 1.0),
        'p_crossover': 0.7,
        'p_subtree_mutation': 0.1,
        'p_hoist_mutation': 0.05,
        'p_point_mutation': 0.1,
        'max_samples': 0.9,
        'verbose': 1,
        'parsimony_coefficient': 0.001,
        'random_state': 42,
        'n_jobs': -1,
        'feature_names': None # Will be filled dynamically
    }
    
    XGB_PARAMS = {
        'max_depth': 2,       # Stumps work best
        'learning_rate': 0.05,
        'n_estimators': 2000,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'eval_metric': 'auc',
        'n_jobs': -1,
        'random_state': 42
    }
    
    SEED = 42
    N_FOLDS = 5
    
    # PATHS (Kaggle Standard)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# PREPROCESSING
# ==================================================================================
def preprocess_gp(train, test):
    print("Applying GP Preprocessing...")
    y = train["Heart Disease"].map({'Presence': 1, 'Absence': 0})
    train = train.drop(columns=["id", "Heart Disease"])
    test = test.drop(columns=["id"])
    
    # Simple Numeric for GP
    feature_cols = train.columns.tolist()
    CFG.GP_PARAMS['feature_names'] = feature_cols
    
    # Handle Cats roughly for GP (Label Enc is fine for math evolution)
    for c in train.select_dtypes(include=['object']):
        train[c] = train[c].astype('category').cat.codes
        test[c] = test[c].astype('category').cat.codes
        
    return train, test, y, feature_cols

def evolve_features(X_train, y_train, X_test, feature_names):
    print("\n🧬 Evolving Features with Genetic Programming...")
    gp = SymbolicTransformer(**CFG.GP_PARAMS)
    
    # Train GP
    gp.fit(X_train, y_train)
    
    # Transform
    new_features_train = gp.transform(X_train)
    new_features_test = gp.transform(X_test)
    
    print(f"Generated {new_features_train.shape[1]} new features.")
    
    # Create DF
    gp_feat_names = [f"GP_{i}" for i in range(new_features_train.shape[1])]
    
    df_gp_train = pd.DataFrame(new_features_train, columns=gp_feat_names)
    df_gp_test = pd.DataFrame(new_features_test, columns=gp_feat_names)
    
    # Concatenate
    X_train_ext = pd.concat([X_train.reset_index(drop=True), df_gp_train], axis=1)
    X_test_ext = pd.concat([X_test.reset_index(drop=True), df_gp_test], axis=1)
    
    return X_train_ext, X_test_ext

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
        train_raw = pd.read_csv(CFG.TRAIN_PATH)
        test_raw = pd.read_csv(CFG.TEST_PATH)
    else:
        print("Loading from Local (Fallback)...")
        train_raw = pd.read_csv("train.csv")
        test_raw = pd.read_csv("test.csv")
    
    # 2. Preprocess & Evolve
    X, X_test, y, feat_names = preprocess_gp(train_raw, test_raw)
    
    try:
        X_ext, X_test_ext = evolve_features(X, y, X_test, feat_names)
    except NameError:
        print("gplearn not installed. Skipping GP evolution.")
        X_ext, X_test_ext = X, X_test
    except Exception as e:
        print(f"GP Evolution failed: {e}. Using raw features.")
        X_ext, X_test_ext = X, X_test
        
    # 3. XGBoost Cross Validation
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X_ext))
    test_preds = np.zeros(len(X_test_ext))
    
    scores = []
    
    print(f"\nTraining XGB on Extended Features ({X_ext.shape[1]} cols)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_ext, y)):
        X_tr, y_tr = X_ext.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_ext.iloc[val_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**CFG.XGB_PARAMS)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test_ext)[:, 1] / CFG.N_FOLDS
        
        score = roc_auc_score(y_val, val_preds)
        scores.append(score)
        print(f"Fold {fold+1} | AUC: {score:.5f}")
        
    # Overall
    mean_score = np.mean(scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.5f}")
    
    # Save
    sub = pd.DataFrame({'id': test_raw['id'], 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train_raw['id'], 'target': y, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    # ------------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved:")
    print(f"  {CFG.SUBMISSION_PATH}")
    print(f"  {CFG.OOF_PATH}")
    print(f"\nTotal time: {elapsed:.1f} minutes")

    print("\n" + "="*80)
    print(f"{CFG.VERSION} SUMMARY")
    print("="*80)
    print(f"\n| Version | Model | Features | CV AUC |")
    print(f"|---------|-------|----------|--------|")
    print(f"| **{CFG.VERSION}** | **GP+XGB** | **Raw+Symbolic** | **{mean_score:.5f}** |")
    print(f"\n✅ {CFG.VERSION} GP ready for submission!")
    print("="*80)

if __name__ == "__main__":
    main()
