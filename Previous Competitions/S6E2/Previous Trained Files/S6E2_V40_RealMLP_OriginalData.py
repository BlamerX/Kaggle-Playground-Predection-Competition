# !pip install pytabkit -q 
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pytabkit import RealMLP_TD_Classifier
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
    VERSION = "V40"
    DESCRIPTION = "RealMLP_TD_OriginalDataFeatures"

    # RealMLP_TD_Classifier Params (Reference Exact Match)
    PARAM_GRID = {
        'device': DEVICE,
        'random_state': 42,
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

    SEED = 42
    N_FOLDS = 5

    # Paths
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'

    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# FEATURE ENGINEERING: ORIGINAL DATA INJECTION
# ==================================================================================
def add_engineered_features(df, original, base_features):
    """
    Injects statistics from the original dataset for overlap features.
    Reference: the-best-solo-model-so-far-realmlp-lb-0-95397
    """
    df_temp = df.copy()

    for col in base_features:
        if col in original.columns:
            # Calculate stats from original data
            stats = original.groupby(col)['Heart Disease'].agg(['mean', 'median', 'std', 'skew', 'count']).reset_index()
            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'skew', 'count']]

            # Merge stats into current df
            df_temp = df_temp.merge(stats, on=col, how='left')

            # Fill NaNs for values not present in original data
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
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()

    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "Dataset/train.csv")
    test = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "Dataset/test.csv")
    original = pd.read_csv(CFG.ORIG_PATH if os.path.exists(CFG.ORIG_PATH) else "Dataset/Heart_Disease_Prediction.csv")

    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original shape: {original.shape}")

    # 2. Encode Target
    le = LabelEncoder()
    train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    original['Heart Disease'] = le.fit_transform(original['Heart Disease'])

    # 3. Feature Engineering
    print("Injecting original dataset features...")
    base_features = [col for col in train.columns if col not in ['Heart Disease', 'id']]

    train = add_engineered_features(train, original, base_features)
    test = add_engineered_features(test, original, base_features)

    X = train.drop(['id', 'Heart Disease'], axis=1)
    y = train['Heart Disease']
    X_test = test.drop(['id'], axis=1)

    # 4. Convert all features to categorical (Reference: exact match)
    print("Converting all features to categorical type...")
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')

    print(f"Total features after engineering: {len(X.columns)}")

    # 5. Cross-Validation with RealMLP_TD_Classifier
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []

    print(f"\nStarting {CFG.N_FOLDS}-Fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Starting Fold {fold + 1} ---")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Model Setup & Training
        model = RealMLP_TD_Classifier(**CFG.PARAM_GRID)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        # Inference
        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]

        oof_preds[val_idx] = val_probs
        test_preds += fold_test_probs / CFG.N_FOLDS

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"Fold {fold + 1} ROC-AUC Score: {score:.5f}")

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 6. Evaluation & Save
    overall_score = roc_auc_score(y, oof_preds)

    print(f"\n{'=' * 40}")
    print(f"Overall OOF ROC-AUC: {overall_score:.5f}")
    print(f"Mean Fold Score: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"{'=' * 40}")

    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)

    pd.DataFrame({'id': train['id'], 'Heart Disease_prob': oof_preds}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds}).to_csv(CFG.SUBMISSION_PATH, index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
