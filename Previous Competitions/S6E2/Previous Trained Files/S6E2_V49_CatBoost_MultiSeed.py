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
from scipy.stats import rankdata

# Suppress warnings
warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V49"
    DESCRIPTION = "CatBoost_MultiSeed_Ensemble"
    
    N_FOLDS = 5
    SEEDS = [42, 123, 456, 789, 2026]  # 5 seeds for diversity
    TARGET = 'Heart Disease'
    ID = 'id'
    
    # Paths (Kaggle)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    
    # CatBoost Params (V39 Ordered — exact match)
    CB_PARAMS = {
        'iterations': 8000,
        'learning_rate': 0.015,
        'depth': 5,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.5,
        'boosting_type': 'Ordered',
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'eval_metric': 'AUC',
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 200,
        'task_type': 'GPU',
        'verbose': 500
    }
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# FEATURE ENGINEERING (Same as V39)
# ==================================================================================
def normalize_cols(df):
    df = df.copy()
    df.columns = [re.sub(r"[^\w\s]", "", c.strip().lower()).replace(" ", "_") for c in df.columns]
    return df

def apply_feature_engineering(df, stats_mean, stats_count, global_mean, num_cols, cat_cols, is_train=False):
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

# ==================================================================================
# TRAIN SINGLE SEED
# ==================================================================================
def train_single_seed(seed, X, y, X_test, ordinal_cols, n_folds=5):
    """Train CatBoost with a specific seed, returns OOF and test predictions."""
    
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Seed {seed} — Fold {fold + 1}/{n_folds}")
        
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        params = CFG.CB_PARAMS.copy()
        params['random_seed'] = seed
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            cat_features=ordinal_cols,
            use_best_model=True
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_pred
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        
        score = roc_auc_score(y_val, val_pred)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")
        
        del model; gc.collect()
    
    overall = roc_auc_score(y, oof_preds)
    print(f"\n  Seed {seed} — OOF AUC: {overall:.5f} (Mean: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    
    return oof_preds, test_preds, overall

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Training CatBoost Ordered with {len(CFG.SEEDS)} seeds for diversity")
    print(f"Seeds: {CFG.SEEDS}")
    start_time = time.time()
    
    # Load Data
    train_raw = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "Dataset/train.csv")
    test_raw = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "Dataset/test.csv")
    print(f"Train: {train_raw.shape}, Test: {test_raw.shape}")
    
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
    
    # Global Statistics
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
    
    # Train each seed
    all_oof = {}
    all_test = {}
    all_scores = {}
    
    for seed in CFG.SEEDS:
        oof_preds, test_preds, score = train_single_seed(seed, X, y, X_test, ordinal_cols, CFG.N_FOLDS)
        all_oof[seed] = oof_preds
        all_test[seed] = test_preds
        all_scores[seed] = score
    
    # ====================================================================
    # ENSEMBLE STRATEGIES
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"MULTI-SEED ENSEMBLE RESULTS")
    print(f"{'='*70}")
    
    seeds = list(all_oof.keys())
    
    # Print single seed scores
    print(f"\n  {'Seed':<10} {'OOF AUC':<12}")
    print(f"  {'-'*22}")
    for seed in seeds:
        print(f"  {seed:<10} {all_scores[seed]:.5f}")
    
    # A. Simple average
    avg_oof = np.mean([all_oof[s] for s in seeds], axis=0)
    avg_test = np.mean([all_test[s] for s in seeds], axis=0)
    avg_score = roc_auc_score(y, avg_oof)
    print(f"\n  A. Simple Average ({len(seeds)} seeds): {avg_score:.5f}")
    
    # B. Rank average
    rank_oof = np.mean([rankdata(all_oof[s]) / len(y) for s in seeds], axis=0)
    rank_test = np.mean([rankdata(all_test[s]) / len(X_test) for s in seeds], axis=0)
    rank_score = roc_auc_score(y, rank_oof)
    print(f"  B. Rank Average ({len(seeds)} seeds): {rank_score:.5f}")
    
    # C. AUC-weighted average
    scores_arr = np.array([all_scores[s] for s in seeds])
    weights = np.exp((scores_arr - scores_arr.min()) * 10000)
    weights = weights / weights.sum()
    
    wauc_oof = sum(all_oof[s] * w for s, w in zip(seeds, weights))
    wauc_test = sum(all_test[s] * w for s, w in zip(seeds, weights))
    wauc_score = roc_auc_score(y, wauc_oof)
    print(f"  C. AUC-Weighted Average: {wauc_score:.5f}")
    for s, w in zip(seeds, weights):
        print(f"     Seed {s}: weight={w:.4f}")
    
    # Pick best
    methods = {
        'A. Simple Avg': (avg_score, avg_oof, avg_test),
        'B. Rank Avg': (rank_score, rank_oof, rank_test),
        'C. AUC-Weighted': (wauc_score, wauc_oof, wauc_test),
    }
    
    best_method = max(methods, key=lambda k: methods[k][0])
    best_score, best_oof, best_test = methods[best_method]
    
    print(f"\n  Best: {best_method} → {best_score:.5f}")
    
    # Save outputs
    print(f"\n{'='*70}")
    print(f"SAVING FILES")
    print(f"{'='*70}")
    
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    
    # Save best blend
    pd.DataFrame({'id': train[id_col], 'target': y, 'pred': best_oof}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test[id_col], 'Heart Disease': best_test}).to_csv(CFG.SUBMISSION_PATH, index=False)
    print(f"  {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    
    # Save individual seed outputs
    for seed in CFG.SEEDS:
        pd.DataFrame({
            'id': train[id_col], 'target': y, 'pred': all_oof[seed]
        }).to_csv(f"oof_v49_seed{seed}.csv", index=False)
        
        pd.DataFrame({
            'id': test[id_col], 'Heart Disease': all_test[seed]
        }).to_csv(f"submission_v49_seed{seed}.csv", index=False)
    print(f"  Individual seed files: oof_v49_seed*.csv, submission_v49_seed*.csv")
    
    # Final summary
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Seeds trained: {len(CFG.SEEDS)}")
    print(f"  Best single seed: {max(all_scores.items(), key=lambda x: x[1])}")
    print(f"  Best ensemble ({best_method}): {best_score:.5f}")
    print(f"  Ensemble gain over best seed: {best_score - max(all_scores.values()):+.5f}")
    print(f"  V39 (seed=42) reference: 0.95577 OOF / 0.95390 LB")
    print(f"  Total time: {elapsed:.1f} min")
    print(f"\n✅ Submit {CFG.SUBMISSION_PATH} to Kaggle!")

if __name__ == "__main__":
    main()
