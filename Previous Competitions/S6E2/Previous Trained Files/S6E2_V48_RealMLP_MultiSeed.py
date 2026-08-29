# !pip install pytabkit -q 
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pytabkit import RealMLP_TD_Classifier
from scipy.stats import rankdata
import time
import os

warnings.filterwarnings('ignore')

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V48"
    DESCRIPTION = "RealMLP_MultiSeed_Ensemble"

    # 5 Seeds for diversity — each produces a different NN initialization
    SEEDS = [42, 123, 456, 789, 2026]
    
    # RealMLP_TD_Classifier Params (Reference: V40 exact match)
    BASE_PARAMS = {
        'device': DEVICE,
        'verbosity': 2,
        'n_epochs': 100,
        'batch_size': 256,
        'n_ens': 8,
        'use_early_stopping': True,
        'early_stopping_additive_patience': 20,
        'early_stopping_multiplicative_patience': 1,
        'act': "mish",
        'embedding_size': 8,
        'first_layer_lr_factor': 0.5962121993798933,
        'hidden_sizes': "rectangular",
        'hidden_width': 384,
        'lr': 0.04,
        'ls_eps': 0.011498317194338772,
        'ls_eps_sched': "coslog4",
        'max_one_hot_cat_size': 18,
        'n_hidden_layers': 4,
        'p_drop': 0.07301419697186451,
        'p_drop_sched': "flat_cos",
        'plr_hidden_1': 16,
        'plr_hidden_2': 8,
        'plr_lr_factor': 0.1151437622270563,
        'plr_sigma': 2.3316811282666916,
        'scale_lr_factor': 2.244801835541429,
        'sq_mom': 1.0 - 0.011834054955582318,
        'wd': 0.02369230879235962,
    }

    N_FOLDS = 5

    # Paths (Kaggle)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'

    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# FEATURE ENGINEERING: ORIGINAL DATA INJECTION (Same as V40)
# ==================================================================================
def add_engineered_features(df, original, base_features):
    """
    Injects statistics from the original dataset for overlap features.
    Reference: the-best-solo-model-so-far-realmlp-lb-0-95397
    """
    df_temp = df.copy()

    for col in base_features:
        if col in original.columns:
            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]
            df_temp = df_temp.merge(stats, on=col, how='left')

            fill_values = {
                f"orig_{col}_mean": original['Heart Disease'].mean(),
                f"orig_{col}_median": original['Heart Disease'].median(),
                f"orig_{col}_std": 0,
                f"orig_{col}_skew": 0,
                f"orig_{col}_count": 0
            }
            df_temp = df_temp.fillna(value=fill_values)

    return df_temp

# ==================================================================================
# TRAIN SINGLE SEED
# ==================================================================================
def train_single_seed(seed, X, y, X_test, n_folds=5):
    """Train RealMLP with a specific seed, returns OOF and test predictions."""
    
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Seed {seed} — Fold {fold + 1}/{n_folds}")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create model with this seed
        params = CFG.BASE_PARAMS.copy()
        params['random_state'] = seed
        
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]
        
        oof_preds[val_idx] = val_probs
        test_preds += fold_test_probs / n_folds
        
        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"  Fold {fold + 1} AUC: {score:.5f}")
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    
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
    print(f"Training RealMLP with {len(CFG.SEEDS)} seeds for diversity")
    print(f"Seeds: {CFG.SEEDS}")
    start_time = time.time()

    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "Dataset/train.csv")
    test = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "Dataset/test.csv")
    original = pd.read_csv(CFG.ORIG_PATH if os.path.exists(CFG.ORIG_PATH) else "Dataset/Heart_Disease_Prediction.csv")

    print(f"Train: {train.shape}, Test: {test.shape}, Original: {original.shape}")

    # 2. Encode Target
    le = LabelEncoder()
    train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

    # 3. Feature Engineering (same as V40)
    print("Injecting original dataset features...")
    base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']]
    train = add_engineered_features(train, original, base_features)
    test = add_engineered_features(test, original, base_features)

    X = train.drop(['id', 'Heart Disease'], axis=1)
    y = train['Heart Disease']
    X_test = test.drop(['id'], axis=1)

    # 4. Convert to categorical (Reference exact match)
    print("Converting all features to categorical type...")
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')

    print(f"Total features: {len(X.columns)}")

    # 5. Train each seed
    all_oof = {}
    all_test = {}
    all_scores = {}
    
    for seed in CFG.SEEDS:
        oof_preds, test_preds, score = train_single_seed(seed, X, y, X_test, CFG.N_FOLDS)
        all_oof[seed] = oof_preds
        all_test[seed] = test_preds
        all_scores[seed] = score

    # 6. Ensemble Strategies
    print(f"\n{'='*70}")
    print(f"MULTI-SEED ENSEMBLE RESULTS")
    print(f"{'='*70}")
    
    # Print single seed scores
    print(f"\n  {'Seed':<10} {'OOF AUC':<12}")
    print(f"  {'-'*22}")
    for seed in CFG.SEEDS:
        print(f"  {seed:<10} {all_scores[seed]:.5f}")
    
    seeds = list(all_oof.keys())
    
    # A. Simple average (equal weight)
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
    
    # Pick best method
    methods = {
        'A. Simple Avg': (avg_score, avg_oof, avg_test),
        'B. Rank Avg': (rank_score, rank_oof, rank_test),
        'C. AUC-Weighted': (wauc_score, wauc_oof, wauc_test),
    }
    
    best_method = max(methods, key=lambda k: methods[k][0])
    best_score, best_oof, best_test = methods[best_method]
    
    print(f"\n  Best: {best_method} → {best_score:.5f}")
    
    # 7. Save outputs
    print(f"\n{'='*70}")
    print(f"SAVING FILES")
    print(f"{'='*70}")
    
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    
    # Save best blend
    pd.DataFrame({'id': train['id'], 'Heart Disease_prob': best_oof}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test['id'], 'Heart Disease': best_test}).to_csv(CFG.SUBMISSION_PATH, index=False)
    print(f"  {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    
    # Save individual seed outputs for later analysis
    for seed in CFG.SEEDS:
        pd.DataFrame({
            'id': train['id'], 'Heart Disease_prob': all_oof[seed]
        }).to_csv(f"oof_v48_seed{seed}.csv", index=False)
        
        pd.DataFrame({
            'id': test['id'], 'Heart Disease': all_test[seed]
        }).to_csv(f"submission_v48_seed{seed}.csv", index=False)
    print(f"  Individual seed files: oof_v48_seed*.csv, submission_v48_seed*.csv")
    
    # 8. Final summary
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Seeds trained: {len(CFG.SEEDS)}")
    print(f"  Best single seed: {max(all_scores.items(), key=lambda x: x[1])}")
    print(f"  Best ensemble ({best_method}): {best_score:.5f}")
    print(f"  Ensemble gain over best seed: {best_score - max(all_scores.values()):+.5f}")
    print(f"  V40 (seed=42) reference: 0.95541 OOF / 0.95394 LB")
    print(f"  Total time: {elapsed:.1f} min")
    print(f"\n✅ Submit {CFG.SUBMISSION_PATH} to Kaggle!")

if __name__ == "__main__":
    main()
