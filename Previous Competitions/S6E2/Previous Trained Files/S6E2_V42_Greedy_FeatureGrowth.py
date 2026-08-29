
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
    VERSION = "V42"
    DESCRIPTION = "CatBoost_Greedy_FeatureGrowth"
    
    # Base: V17 CatBoost Deotte Params (Champion LB 0.95385)
    CAT_PARAMS = {
        'iterations': 50000,
        'learning_rate': 0.0025,
        'depth': 3,
        'subsample': 0.8,
        'random_seed': 42,
        'early_stopping_rounds': 1000,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False
    }
    
    SEED = 42
    N_FOLDS = 5         # 5-fold for speed during greedy search
    INNER_FOLDS = 15    # Inner TE folds (same as V17)
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"


def evaluate_features(train, test, orig, feature_set, te_columns, to_remove, fold_indices):
    """
    Evaluate a specific feature set using V17 Deotte pipeline.
    Returns OOF AUC score.
    """
    STATS = ['mean']
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    
    X_orig = orig[feature_set + ['Heart Disease']].copy()
    y_orig = orig['Heart Disease'].copy()
    
    for i, (train_index, val_index) in enumerate(fold_indices):
        
        X_train = train.loc[train_index, feature_set + ['Heart Disease']].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease']
        
        # Augment with Original Data
        X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()
        
        X_val = train.loc[val_index, feature_set].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease']
        X_test = test[feature_set].reset_index(drop=True).copy()
        
        # Inner CV for Target Encoding
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        
        # Only TE columns that exist in the current feature set
        active_te = [c for c in te_columns if c in feature_set]
        
        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
            X_train2 = X_train.loc[train_index2, feature_set + ['Heart Disease']].copy()
            X_val2 = X_train.loc[val_index2, feature_set].copy()
            
            for col in active_te:
                tmp = X_train2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_val2 = X_val2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")
        
        # Outer TE (Val & Test)
        for col in active_te:
            tmp = X_train.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_test = X_test.merge(tmp, on=col, how="left")
        
        # Drop Categoricals
        active_remove = [c for c in to_remove if c in X_train.columns]
        X_train.drop(columns=active_remove, inplace=True)
        active_remove_val = [c for c in to_remove if c in X_val.columns]
        X_val.drop(columns=active_remove_val, inplace=True)
        active_remove_test = [c for c in to_remove if c in X_test.columns]
        X_test.drop(columns=active_remove_test, inplace=True)
        
        if 'Heart Disease' in X_train.columns:
            X_train = X_train.drop(['Heart Disease'], axis=1)
        
        # Train CatBoost
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        model = CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=False, use_best_model=True)
        
        val_p = model.predict_proba(X_val)[:, 1]
        oof[val_index] = val_p
        fold_scores.append(roc_auc_score(y_val, val_p))
        pred += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS
        
        del X_train, X_val, X_test, model, train_pool, val_pool
        gc.collect()
    
    overall = roc_auc_score(train['Heart Disease'], oof)
    return overall, fold_scores, oof, pred


def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH
    
    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
        train_path = "Dataset/train.csv"
        test_path = "Dataset/test.csv"
        orig_path = "Dataset/Heart_Disease_Prediction.csv"
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
    
    # 2. Build Feature Pool
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']
    
    # Generate ALL candidate features
    print("Building feature pool...")
    
    # Frequency Encoding
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
    
    # Numerical as Categorical (for TE)
    for col in NUMS:
        for df in [train, test, orig]:
            df[f'CAT_{col}'] = df[col].astype(str).astype('category')
    
    # Discussion features (tested individually in V41, but maybe useful in greedy context)
    for df in [train, test, orig]:
        df['EKG_binary'] = (df['EKG results'] == 2).astype(int)
        df['ST_Slope'] = df['ST depression'] * df['Slope of ST']
        df['Chest_asymptomatic'] = (df['Chest pain type'] == 4).astype(int)
    
    # Define feature groups for greedy addition
    NUM_AS_CAT = [f'CAT_{col}' for col in NUMS]
    FREQ_FEATURES = [f'FREQ_{col}' for col in NUMS]
    INTERACTION_FEATURES = ['EKG_binary', 'ST_Slope', 'Chest_asymptomatic']
    
    TE_COLUMNS = NUM_AS_CAT + CATS  # Columns that get Target Encoded
    TO_REMOVE = NUM_AS_CAT + CATS   # Dropped after TE
    
    # ==================================================================================
    # GREEDY FEATURE GROWTH
    # Start with raw NUMS → add feature groups one-by-one → keep if CV improves
    # ==================================================================================
    
    # Fix folds for fair comparison
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    fold_indices = list(kf.split(train))
    
    # Candidate feature GROUPS to try adding (order: most likely to help first)
    candidate_groups = [
        ("CATS (Categoricals)",       CATS),
        ("NUM_AS_CAT (for TE)",       NUM_AS_CAT),
        ("FREQ (Frequency Encoding)", FREQ_FEATURES),
        ("EKG_binary",                ['EKG_binary']),
        ("ST_Slope",                  ['ST_Slope']),
        ("Chest_asymptomatic",        ['Chest_asymptomatic']),
    ]
    
    # Start with raw numerical features
    current_features = NUMS.copy()
    
    print(f"\n{'='*70}")
    print(f"  GREEDY FEATURE GROWTH")
    print(f"  Starting with {len(current_features)} raw numerical features")
    print(f"  Testing {len(candidate_groups)} candidate groups")
    print(f"{'='*70}")
    
    # Evaluate baseline (raw NUMS only)
    print(f"\n--- Baseline: Raw NUMS only ({len(current_features)} features) ---")
    baseline_score, baseline_folds, best_oof, best_pred = evaluate_features(
        train, test, orig, current_features, TE_COLUMNS, TO_REMOVE, fold_indices
    )
    print(f"  Baseline OOF AUC: {baseline_score:.5f}")
    
    best_score = baseline_score
    growth_log = [("Baseline (Raw NUMS)", len(current_features), baseline_score, 0.0, "START")]
    
    # Greedy: try each candidate group
    for group_name, group_features in candidate_groups:
        print(f"\n--- Testing: +{group_name} ---")
        
        trial_features = current_features + [f for f in group_features if f not in current_features]
        
        score, folds, oof, pred = evaluate_features(
            train, test, orig, trial_features, TE_COLUMNS, TO_REMOVE, fold_indices
        )
        delta = score - best_score
        
        if delta > 0:
            decision = "✅ KEEP"
            current_features = trial_features
            best_score = score
            best_oof = oof
            best_pred = pred
        else:
            decision = "❌ SKIP"
        
        print(f"  OOF AUC: {score:.5f} | Delta: {delta:+.5f} | {decision}")
        growth_log.append((group_name, len(trial_features), score, delta, decision))
    
    # ==================================================================================
    # RESULTS SUMMARY
    # ==================================================================================
    print(f"\n{'='*70}")
    print(f"  GREEDY FEATURE GROWTH RESULTS")
    print(f"{'='*70}")
    print(f"{'Step':<35} {'N_Feat':>6} {'OOF AUC':>10} {'Delta':>10} {'Decision':>10}")
    print(f"{'-'*71}")
    
    for step_name, n_feat, score, delta, decision in growth_log:
        sign = "+" if delta >= 0 else ""
        print(f"{step_name:<35} {n_feat:>6} {score:>10.5f} {sign}{delta:>9.5f} {decision:>10}")
    
    print(f"\n✅ Final Feature Set ({len(current_features)} features):")
    print(f"   {current_features}")
    print(f"   Final OOF AUC: {best_score:.5f}")
    
    # Save
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': best_pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': best_oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
