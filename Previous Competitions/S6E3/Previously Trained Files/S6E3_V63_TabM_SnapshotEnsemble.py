"""
S6E3 V63 - TabM + Snapshot Ensemble
================================================================================
Strategy: Save checkpoints during training with cyclic LR, average predictions

Key Idea:
  1. Train with cyclic learning rate (cosine annealing)
  2. Save model snapshot at each cycle minimum (when LR is lowest)
  3. Average predictions from all snapshots
  4. "Free" ensemble from single training run

Reference: 
  - V21 TabM (OOF: 0.91898, LB: 0.91682)
  - Paper: "Snapshot Ensembles: Train 1, get M for free" (ICLR)

Implementation:
  - Use CosineAnnealingWarmRestarts scheduler
  - Save snapshot at end of each cycle
  - Average all snapshot predictions

Diversity: ⭐⭐⭐⭐ (Free ensemble from single training)

Rules:
  - NO ENSEMBLING / BLENDING / STACKING with other models
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import random
import os
import copy
from itertools import combinations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# TabM-style Model
# ═══════════════════════════════════════════════════════════════════════════════

class TabMModel(nn.Module):
    """TabM-style feedforward network"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    VERSION_NAME = "v63"
    EXP_ID = "S6E3_V63_TabM_SnapshotEnsemble"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    INNER_FOLDS = 5
    RANDOM_SEED = 42

# Snapshot Ensemble Parameters
SE_PARAMS = {
    'hidden_dims': [256, 128, 64],
    'dropout': 0.1,
    'n_cycles': 5,              # Number of snapshot cycles
    'cycle_epochs': 20,         # Epochs per cycle
    'batch_size': 1024,
    'lr': 0.01,                 # Initial learning rate
    'min_lr': 1e-6,             # Minimum learning rate
    'weight_decay': 1e-5,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def feature_engineering(train, test, orig, target):
    """V16 Feature Engineering Pipeline"""
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []

    # 1. Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
        
    # 2. Arithmetic Interactions
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # 3. Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # 4. ORIG_proba mapping
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[target].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # 5. Distribution Features
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')
    def zscore_against(values, reference):
        mu, sigma = np.mean(reference), np.std(reference)
        return (np.zeros(len(values), dtype='float32') if sigma == 0 
                else ((values - mu) / sigma).astype('float32'))
    
    orig_churner_tc = orig.loc[orig[target] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[target] == 0, 'TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService')['MonthlyCharges'].mean()
    
    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nonchurner_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_churner_tc)
        df['zscore_churn_gap_TC'] = (np.abs(zscore_against(tc, orig_churner_tc)) - 
                                     np.abs(zscore_against(tc, orig_nonchurner_tc))).astype('float32')
        df['zscore_nonchurner_TC'] = zscore_against(tc, orig_nonchurner_tc)
        df['pctrank_churn_gap_TC'] = (pctrank_against(tc, orig_churner_tc) - 
                                      pctrank_against(tc, orig_nonchurner_tc)).astype('float32')
        df['resid_IS_MC'] = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc_mean).fillna(0)).astype('float32')
        
        vals = np.zeros(len(df), dtype='float32')
        for cat_val in orig['InternetService'].unique():
            mask = df['InternetService'] == cat_val
            ref = orig.loc[orig['InternetService'] == cat_val, 'TotalCharges'].values
            if len(ref) > 0 and mask.sum() > 0:
                vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
        df['cond_pctrank_IS_TC'] = vals
        
        vals = np.zeros(len(df), dtype='float32')
        for cat_val in orig['Contract'].unique():
            mask = df['Contract'] == cat_val
            ref = orig.loc[orig['Contract'] == cat_val, 'TotalCharges'].values
            if len(ref) > 0 and mask.sum() > 0:
                vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
        df['cond_pctrank_C_TC'] = vals
    
    NEW_NUMS += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]
    
    # Quantile distance features
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
            
    NEW_NUMS += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_ch_q25', 'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_ch_q75', 'dist_To_nc_q75', 'qdist_gap_To_q75'
    ]

    # 6. Digit Features
    DIGIT_FEATURES = [
        'tenure_first_digit', 'tenure_last_digit', 'tenure_second_digit',
        'tenure_mod10', 'tenure_mod12', 'tenure_num_digits',
        'tenure_is_multiple_10', 'tenure_rounded_10', 'tenure_dev_from_round10',
        'mc_first_digit', 'mc_last_digit', 'mc_second_digit',
        'mc_mod10', 'mc_mod100', 'mc_num_digits', 
        'mc_is_multiple_10', 'mc_is_multiple_50',
        'mc_rounded_10', 'mc_fractional', 'mc_dev_from_round10',
        'tc_first_digit', 'tc_last_digit', 'tc_second_digit',
        'tc_mod10', 'tc_mod100', 'tc_num_digits',
        'tc_is_multiple_10', 'tc_is_multiple_100',
        'tc_rounded_100', 'tc_fractional', 'tc_dev_from_round100',
        'tenure_years', 'tenure_months_in_year', 'mc_per_digit', 'tc_per_digit'
    ]

    for df in [train, test]:
        t_str = df['tenure'].astype(str)
        df['tenure_first_digit'] = t_str.str[0].astype(int)
        df['tenure_last_digit'] = t_str.str[-1].astype(int)
        df['tenure_second_digit'] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10'] = df['tenure'] % 10
        df['tenure_mod12'] = df['tenure'] % 12
        df['tenure_num_digits'] = t_str.str.len()
        df['tenure_is_multiple_10'] = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10'] = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])
        
        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit'] = mc_str.str[0].astype(int)
        df['mc_last_digit'] = mc_str.str[-1].astype(int)
        df['mc_second_digit'] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10'] = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100'] = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits'] = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10'] = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50'] = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10'] = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional'] = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])
        
        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit'] = tc_str.str[0].astype(int)
        df['tc_last_digit'] = tc_str.str[-1].astype(int)
        df['tc_second_digit'] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10'] = np.floor(df['TotalCharges']) % 10
        df['tc_mod100'] = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits'] = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100'] = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional'] = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])
        
        df['tenure_years'] = df['tenure'] // 12
        df['tenure_months_in_year'] = df['tenure'] % 12
        df['mc_per_digit'] = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit'] = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)

        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')

    NEW_NUMS += DIGIT_FEATURES

    # 7. N-gram Features
    BIGRAM_COLS = []
    TRIGRAM_COLS = []
    
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
        BIGRAM_COLS.append(col_name)
    
    TOP4 = TOP_CATS_FOR_NGRAM[:4] 
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str))
        TRIGRAM_COLS.append(col_name)
    
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS
    
    return train, test, NUMS, NEW_NUMS, CATS, NGRAM_COLS


if __name__ == "__main__":
    seed_everything(CFG.RANDOM_SEED)
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print(f"Strategy: TabM + Snapshot Ensemble")
    print(f"Snapshot cycles: {SE_PARAMS['n_cycles']} x {SE_PARAMS['cycle_epochs']} epochs")
    print(f"Device: {DEVICE}")
    
    # [1/5] Loading data
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)
        
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    print(f"Train : {train.shape}")
    print(f"Test  : {test.shape}")
    print(f"Orig  : {orig.shape}")
    
    # [2/5] Feature Engineering
    print("\n[2/5] Feature Engineering (V16 pipeline)...")
    y_all = train[CFG.TARGET].values
    train, test, NUMS, NEW_NUMS, CATS, NGRAM_COLS = feature_engineering(train, test, orig, CFG.TARGET)
    print(f"  Base numericals: {len(NUMS)}")
    print(f"  Engineered numericals: {len(NEW_NUMS)}")
    print(f"  Categoricals: {len(CATS)}")
    print(f"  N-gram columns: {len(NGRAM_COLS)}")
    
    # [3/5] Prepare features
    print(f"\n[3/5] Preparing features...")
    
    # Label encode categoricals
    for col in CATS + NGRAM_COLS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col]).astype('float32')
        test[col] = le.transform(test[col]).astype('float32')
    
    ALL_FEATURES = NUMS + NEW_NUMS + CATS + NGRAM_COLS
    print(f"  Total features: {len(ALL_FEATURES)}")
    
    # Fill NaNs
    for col in NUMS + NEW_NUMS:
        train[col] = train[col].fillna(0).astype('float32')
        test[col] = test[col].fillna(0).astype('float32')
    
    # Scale
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[ALL_FEATURES].values)
    test_scaled = scaler.transform(test[ALL_FEATURES].values)
    
    # [4/5] Training with Snapshot Ensemble
    print(f"\n[4/5] Training TabM with Snapshot Ensemble ({CFG.N_FOLDS}-Fold CV)...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf.split(train_scaled, y_all)):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr = train_scaled[train_idx]
        y_tr = y_all[train_idx]
        X_val = train_scaled[val_idx]
        y_val = y_all[val_idx]
        X_te = test_scaled
        
        # Tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
        
        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t), 
            batch_size=SE_PARAMS['batch_size'], 
            shuffle=True
        )
        
        # Initialize model
        model = TabMModel(
            input_dim=len(ALL_FEATURES),
            hidden_dims=SE_PARAMS['hidden_dims'],
            dropout=SE_PARAMS['dropout'],
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=SE_PARAMS['lr'],
            weight_decay=SE_PARAMS['weight_decay']
        )
        
        # Cosine annealing with warm restarts for snapshot ensemble
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=SE_PARAMS['cycle_epochs'],  # Cycle length
            T_mult=2,  # Double cycle length after each restart
            eta_min=SE_PARAMS['min_lr']
        )
        
        criterion = nn.BCELoss()
        
        # Store snapshots
        snapshots = []
        best_val_auc = 0
        cycle_count = 0
        
        total_epochs = SE_PARAMS['n_cycles'] * SE_PARAMS['cycle_epochs']
        
        for epoch in range(total_epochs):
            # Training
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_pred = val_output.detach().cpu().numpy()
            val_auc = roc_auc_score(y_val, val_pred)
            
            # Save snapshot at end of each cycle (when LR is at minimum)
            if (epoch + 1) % SE_PARAMS['cycle_epochs'] == 0:
                snapshots.append(copy.deepcopy(model.state_dict()))
                cycle_count += 1
                print(f"  Epoch {epoch+1}: AUC={val_auc:.5f} | Saved snapshot {cycle_count}")
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
        
        # Ensemble predictions from all snapshots
        val_preds_list = []
        test_preds_list = []
        
        for snapshot in snapshots:
            model.load_state_dict(snapshot)
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                test_output = model(X_te_t)
                val_preds_list.append(val_output.detach().cpu().numpy())
                test_preds_list.append(test_output.detach().cpu().numpy())
        
        # Average all snapshots
        val_pred_avg = np.mean(val_preds_list, axis=0)
        test_pred_avg = np.mean(test_preds_list, axis=0)
        
        oof[val_idx] = val_pred_avg
        pred += test_pred_avg / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_pred_avg)
        fold_scores.append(fold_auc)
        
        # Also report best single model
        best_single_auc = max(roc_auc_score(y_val, vp) for vp in val_preds_list)
        print(f"   Fold {i+1} AUC (Ensemble): {fold_auc:.5f} | Best single: {best_single_auc:.5f} | Δ: {fold_auc - best_single_auc:+.5f}")
        print(f"   Time: {(time.time()-t0)/60:.1f} min")
        
        del model, snapshots, X_tr_t, X_val_t, X_te_t, train_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Results
    overall_auc = roc_auc_score(y_all, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\n{'='*80}")
    print(f"V63 RESULTS — TabM + Snapshot Ensemble")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V21 Reference (TabM):  0.91898 (OOF) / LB 0.91682")
    print(f"Delta vs V21:   {overall_auc - 0.91898:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    
    verdict = "IMPROVED" if overall_auc > 0.91898 else "MARGINAL" if overall_auc > 0.91850 else "SAME"
    print(f"Verdict: {verdict}")
    
    # Save
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
