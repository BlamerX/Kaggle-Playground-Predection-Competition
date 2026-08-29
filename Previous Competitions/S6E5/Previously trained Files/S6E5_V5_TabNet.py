"""
S6E5 V5 - TabNet Baseline (GPU)
================================================================================
Strategy: TabNet with V1's proven FE pipeline + Original Data

TabNet — Attention-based tabular DL (Arik & Pfister, 2021):
- Sparse attention for automatic feature selection per decision step
- Sequential multi-step reasoning (3-5 steps)
- Learned feature interactions through attention masks
- Handles class imbalance via weights parameter

FE Pipeline (identical to V1 — 14 raw -> 38 features):
- 2 ratio features (LapNumber/RaceProgress, TyreLife/LapNumber)
- 13 floor-categorization: {num_col}_cat_ (floor + factorize) -> int codes
- 7 count encodings: _{cat_col}_count
- 2 discretized bins: RaceProgress_200q, LapTime_7q
- 2 interaction categories: Race_Compound_, Race_Year_
- 2 TE features on interaction combos (per-fold)
- Original F1 data concatenated per-fold

TabNet Configuration (Kaggle-proven):
  - n_d=32, n_a=32: decision + attention width (larger than paper defaults)
  - n_steps=5: 5-step sequential reasoning
  - gamma=1.5: feature reuse relaxation
  - lambda_sparse=1e-4: sparsity regularization
  - mask_type='entmax': sharper attention than sparsemax
  - Adam optimizer, lr=0.02
  - StepLR scheduler (step_size=10, gamma=0.9)
  - batch_size=8192, virtual_batch_size=1024
  - max_epochs=100, patience=15

Categorical Handling:
  - Label-encode all categoricals to int -> treat as numeric float32
  - NO cat_idxs/cat_dims (simpler, avoids embedding OOB issues)
  - This is the most common Kaggle approach for TabNet

Golden Rules: SKF(10, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import sys
import subprocess
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

# Auto-install pytorch-tabnet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("pytorch-tabnet loaded successfully!")
except ImportError:
    print("Installing pytorch-tabnet...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "pytorch-tabnet", "-q"]
    )
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("pytorch-tabnet installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v5"
    EXP_ID = "S6E5_V5_TabNet"
    DEVICE = DEVICE

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    TE = True  # Target Encoding on interaction categories

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. TABNET PARAMETERS (Kaggle-proven config)
# =============================================================================
TABNET_PARAMS = {
    # Architecture
    'n_d': 32,                # Decision prediction width
    'n_a': 32,                # Attention width (paper: n_d = n_a)
    'n_steps': 5,            # Sequential decision steps
    'gamma': 1.5,            # Feature reuse relaxation
    'n_independent': 2,      # Independent GLU layers per step
    'n_shared': 2,           # Shared GLU layers across steps
    'lambda_sparse': 1e-4,   # Sparsity on attention masks
    'mask_type': 'entmax',   # Sharper than sparsemax

    # Optimizer
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2),

    # Scheduler
    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
    'scheduler_params': dict(step_size=10, gamma=0.9),

    # Misc
    'seed': CFG.RANDOM_SEED,
    'device_name': CFG.DEVICE,
    'verbose': 0,
}

# =============================================================================
# 5. FEATURE ENGINEERING (identical to V1)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> 38 features (same as V1).

    Features created:
    - 2 ratio features: _LapNumber_/_RaceProgress, _TyreLife_/_LapNumber
    - 13 floor-categorization: {num_col}_cat_ (floor + factorize)
    - 7 count encodings: _{cat_col}_count
    - 2 discretized: RaceProgress_200_quantile_bin_, LapTime (s)_7_quantile_bin_
    - 2 interaction categories: Race_Compound_, Race_Year_

    Returns:
        df, new_cat_cols, new_num_cols, combo_names
    """
    # ------------------------------------------------------------------
    # 1. ARITHMETIC INTERACTION (2 ratio features)
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (
        df['LapNumber'] / (df['RaceProgress'] + 1e-6)
    ).astype('float32')
    df['_TyreLife_/_LapNumber'] = (
        df['TyreLife'] / df['LapNumber'].clip(lower=1)
    ).astype('float32')

    # ------------------------------------------------------------------
    # 2. CATEGORIZE NUMERICALS (floor + factorize) -> 13 _cat_ features
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes

    # ------------------------------------------------------------------
    # 3. COUNT ENCODING (original cats + Year_cat_ + PitStop_cat_) -> 7 features
    # ------------------------------------------------------------------
    count_cols = cat_cols + ['Year_cat_', 'PitStop_cat_']
    for col in count_cols:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # ------------------------------------------------------------------
    # 4. DISCRETIZE NUMERICALS (KBinsDiscretizer) -> 2 _bin_ features
    # ------------------------------------------------------------------
    bin_config = {
        'RaceProgress': [200],
        'LapTime (s)': [7],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy=strategy,
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES -> 2 combo features
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
    ]
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes

    # Identify new feature types
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Drop Normalized_TyreLife from original (intentionally removed from competition)
    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    # Store IDs and separate target
    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    orig     = orig.drop(columns=[CFG.TARGET], axis=1, errors='ignore')

    X      = train.drop(columns=['id', CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    X_test = test.drop(columns=['id'], axis=1)

    del train, test

    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"   Base cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"   Base num_cols: {len(num_cols)} -> {num_cols}")

    # Target distribution
    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (14 raw -> 38 features)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    # Update column lists
    all_cat_cols = cat_cols + new_cat_cols
    all_num_cols = num_cols + new_num_cols

    print(f"   New cat_cols: {len(new_cat_cols)}")
    print(f"   New num_cols: {len(new_num_cols)}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"\n   Total cat_cols: {len(all_cat_cols)}")
    print(f"   Total num_cols: {len(all_num_cols)}")
    print(f"   Total features: {len(all_cat_cols) + len(all_num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [2.5/5] PRE-DTYPE CONVERSION — Label encode strings -> int, all -> float32
    # =========================================================================
    print(f"\n[2.5/5] Pre-converting dtypes for TabNet...")

    # Label-encode base string categoricals to int (combined fit for consistency)
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([
            X[col].astype(str),
            X_test[col].astype(str),
            orig[col].astype(str),
        ], axis=0)
        le.fit(combined)
        X[col]      = le.transform(X[col].astype(str)).astype('int32')
        X_test[col] = le.transform(X_test[col].astype(str)).astype('int32')
        orig[col]   = le.transform(orig[col].astype(str)).astype('int32')

    # Convert everything to float32 for TabNet (no cat_idxs/cat_dims needed)
    feature_cols = X.columns.tolist()
    for col in feature_cols:
        X[col]      = X[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')
        orig[col]   = orig[col].astype('float32')

    print(f"   Done: {len(feature_cols)} features -> float32")

    # =========================================================================
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> TabNet)
    # =========================================================================
    print(f"\n[3/5] Training TabNet ({CFG.N_FOLDS}-Fold CV, orig concat)...")

    skf = StratifiedKFold(
        n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    t0 = time.time()

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} ...")
        print(f"{'#' * 16}")

        # ---- Per-fold: concat competition train + original ----
        X_tr    = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat(
            [y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0
        ).reset_index(drop=True)
        X_val   = X.iloc[val_idx].copy()
        y_val   = y.iloc[val_idx]
        X_tst   = X_test.copy()

        print(f"   Train (comp+orig): {X_tr.shape} | "
              f"Val: {X_val.shape} | Test: {X_tst.shape}")

        # ---- TARGET ENCODING on interaction categories (per-fold) ----
        if CFG.TE:
            te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_']
            te_names = [f"_{col}TE" for col in te_cols]

            TE = TargetEncoder(
                cv=CFG.N_FOLDS, smooth='auto',
                shuffle=True, random_state=CFG.RANDOM_SEED)

            tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
            val_enc = TE.transform(X_val[te_cols])
            tst_enc = TE.transform(X_tst[te_cols])

            X_tr[te_names]  = tr_enc.astype('float32')
            X_val[te_names] = val_enc.astype('float32')
            X_tst[te_names] = tst_enc.astype('float32')

            print(f"   TE cols: {te_cols} -> {te_names}")

        if fold == 1:
            print(f"   len(FEATURES): {len(X_tr.columns.tolist())}")

        # ---- Prepare numpy arrays for TabNet ----
        X_tr_np  = X_tr.values
        X_val_np = X_val.values
        X_tst_np = X_tst.values
        y_tr_np  = y_tr.values
        y_val_np = y_val.values

        # ---- Compute class weights for imbalance ----
        neg = (y_tr_np == 0).sum()
        pos = (y_tr_np == 1).sum()
        class_weights = {0: len(y_tr_np) / (2 * neg), 1: len(y_tr_np) / (2 * pos)}

        # ---- TRAIN TABNET ----
        model = TabNetClassifier(**TABNET_PARAMS)

        model.fit(
            X_tr_np, y_tr_np,
            eval_set=[(X_val_np, y_val_np)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=100,
            patience=15,
            batch_size=8192,
            virtual_batch_size=1024,
            drop_last=False,
            num_workers=0,
            weights=class_weights,
        )

        val_preds     = model.predict_proba(X_val_np)[:, 1]
        fold_test_preds = model.predict_proba(X_tst_np)[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"BestEpoch: {model.best_epoch} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val, model, TE
        del X_tr_np, X_val_np, X_tst_np, y_tr_np, y_val_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Overall OOF AUC ----
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: test_preds,
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    oof_df = pd.DataFrame({
        'id': train_id,
        'pred': oof_preds,
    })
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V5 RESULTS - TabNet ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(feature_cols)} "
          f"({len(all_cat_cols)} cat + {len(all_num_cols)} num + {len(te_names)} TE)")
    print(f"TabNet: n_d={TABNET_PARAMS['n_d']}, n_a={TABNET_PARAMS['n_a']}, "
          f"n_steps={TABNET_PARAMS['n_steps']}, gamma={TABNET_PARAMS['gamma']}")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
