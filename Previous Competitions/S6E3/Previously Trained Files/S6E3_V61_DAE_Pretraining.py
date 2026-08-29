"""
S6E3 V61 - Denoising AutoEncoder Pre-training
================================================================================
Strategy: Self-supervised pre-training on train+test, then fine-tune classifier

Key Idea:
  1. Pre-train DAE on ALL data (train + test) using swap noise
  2. This learns useful representations from test data without labels
  3. Fine-tune encoder for classification on labeled train data

Reference: 
  - Michael Jahrer 1st place Porto Seguro (DAE + NN)
  - V21 TabM (OOF: 0.91898, LB: 0.91682)
  - Paper: "Self Supervised Pre-training for Large Scale Tabular Data"

Architecture:
  Encoder: input_dim -> 512 -> 256 -> 128 -> 64 (bottleneck)
  Decoder: 64 -> 128 -> 256 -> 512 -> input_dim
  
Training:
  - Pre-train: 50 epochs on train+test with swap noise
  - Fine-tune: 10-fold CV classifier using encoder features

Diversity: ⭐⭐⭐⭐⭐ (Pre-trained on test data - unique signal)

Rules:
  - NO ENSEMBLING / BLENDING / STACKING
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import random
import os
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# Denoising AutoEncoder
# ═══════════════════════════════════════════════════════════════════════════════

class DenoisingAutoEncoder(nn.Module):
    """DAE with swap noise for tabular data"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], bottleneck=64, dropout=0.1):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims + [bottleneck]:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        # Remove last dropout for bottleneck
        self.encoder = nn.Sequential(*encoder_layers[:-1])
        
        # Decoder
        decoder_layers = []
        prev_dim = bottleneck
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class Classifier(nn.Module):
    """Classifier on top of encoder"""
    def __init__(self, encoder, bottleneck=64, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features).squeeze(-1)


def apply_swap_noise(X, noise_ratio=0.15):
    """Apply swap noise: replace values with random values from same column"""
    X_noisy = X.copy()
    n_samples, n_features = X.shape
    n_noisy = int(n_samples * noise_ratio)
    
    for col in range(n_features):
        noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
        swap_values = np.random.choice(X[:, col], n_noisy, replace=True)
        X_noisy[noisy_idx, col] = swap_values
    
    return X_noisy


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    VERSION_NAME = "v61"
    EXP_ID = "S6E3_V61_DAE_Pretraining"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    INNER_FOLDS = 5
    RANDOM_SEED = 42

# DAE Parameters
DAE_PARAMS = {
    'hidden_dims': [512, 256, 128],
    'bottleneck': 64,
    'dropout': 0.1,
    'noise_ratio': 0.15,
    'pretrain_epochs': 50,
    'finetune_epochs': 30,
    'batch_size': 1024,
    'lr': 0.001,
    'patience': 10,
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


def feature_engineering(train, test, orig):
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
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # 5. Distribution & Quantile Features
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')
    def zscore_against(values, reference):
        mu, sigma = np.mean(reference), np.std(reference)
        return (np.zeros(len(values), dtype='float32') if sigma == 0 
                else ((values - mu) / sigma).astype('float32'))
    
    orig_churner_tc = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService')['MonthlyCharges'].mean()
    
    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nonchurner_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_churner_tc)
        df['pctrank_orig_TC'] = pctrank_against(tc, orig_tc)
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
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]
    
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
            
    NEW_NUMS += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75'
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
    print(f"Architecture: Denoising AutoEncoder Pre-training")
    print(f"Bottleneck: {DAE_PARAMS['bottleneck']}, Noise ratio: {DAE_PARAMS['noise_ratio']}")
    print(f"Pretrain epochs: {DAE_PARAMS['pretrain_epochs']}, Finetune epochs: {DAE_PARAMS['finetune_epochs']}")
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
    train, test, NUMS, NEW_NUMS, CATS, NGRAM_COLS = feature_engineering(train, test, orig)
    print(f"  Base numericals: {len(NUMS)}")
    print(f"  Engineered numericals: {len(NEW_NUMS)}")
    print(f"  Categoricals: {len(CATS)}")
    print(f"  N-gram columns: {len(NGRAM_COLS)}")
    
    # [3/5] Prepare features
    print("\n[3/5] Preparing features...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    y_all = train[CFG.TARGET].values
    
    # Label encode categoricals
    for col in CATS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col]).astype('float32')
        test[col] = le.transform(test[col]).astype('float32')
    
    # N-gram TE for train
    ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
    train[CFG.TARGET] = y_all
    for c in ng_te_feat_names:
        train[c] = 0.5
        test[c] = 0.5
    
    for j, (in_tr, in_va) in enumerate(skf_inner.split(train, y_all)):
        train2 = train.iloc[in_tr]
        for col in NGRAM_COLS:
            ng_te = train2.groupby(col)[CFG.TARGET].mean()
            train.loc[train.index[in_va], f"TE_ng_{col}"] = train.iloc[in_va][col].map(ng_te).fillna(0.5).values
    
    for col in NGRAM_COLS:
        ng_te = train.groupby(col)[CFG.TARGET].mean()
        test[f"TE_ng_{col}"] = test[col].map(ng_te).fillna(0.5).values
    train.drop(columns=[CFG.TARGET], inplace=True)
    
    # Prepare feature matrix
    ALL_NUMS = NUMS + NEW_NUMS + ng_te_feat_names
    ALL_FEATURES = ALL_NUMS + CATS
    
    print(f"  Total features: {len(ALL_FEATURES)}")
    
    # Fill NaNs
    for col in ALL_NUMS:
        train[col] = train[col].fillna(0).astype('float32')
        test[col] = test[col].fillna(0).astype('float32')
    
    # Scale features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[ALL_FEATURES].values)
    test_scaled = scaler.transform(test[ALL_FEATURES].values)
    
    # Combine train+test for DAE pre-training
    all_data = np.vstack([train_scaled, test_scaled])
    print(f"  Combined data for DAE: {all_data.shape}")
    
    # [4/5] Pre-train DAE
    print(f"\n[4/5] Pre-training DAE on train+test ({DAE_PARAMS['pretrain_epochs']} epochs)...")
    
    dae = DenoisingAutoEncoder(
        input_dim=len(ALL_FEATURES),
        hidden_dims=DAE_PARAMS['hidden_dims'],
        bottleneck=DAE_PARAMS['bottleneck'],
        dropout=DAE_PARAMS['dropout'],
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(dae.parameters(), lr=DAE_PARAMS['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DAE_PARAMS['pretrain_epochs'])
    
    all_tensor = torch.tensor(all_data, dtype=torch.float32)
    pretrain_loader = DataLoader(TensorDataset(all_tensor), batch_size=DAE_PARAMS['batch_size'], shuffle=True)
    
    t0_pretrain = time.time()
    for epoch in range(DAE_PARAMS['pretrain_epochs']):
        dae.train()
        total_loss = 0
        for (batch,) in pretrain_loader:
            batch = batch.to(DEVICE)
            
            # Apply swap noise
            batch_np = batch.cpu().numpy()
            batch_noisy = apply_swap_noise(batch_np, DAE_PARAMS['noise_ratio'])
            batch_noisy = torch.tensor(batch_noisy, dtype=torch.float32).to(DEVICE)
            
            # Forward pass
            output = dae(batch_noisy)
            loss = F.mse_loss(output, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(pretrain_loader)
            print(f"      Epoch {epoch+1}/{DAE_PARAMS['pretrain_epochs']}: Loss={avg_loss:.4f}")
    
    print(f"      DAE pre-training done: {(time.time()-t0_pretrain)/60:.1f} min")
    
    # [5/5] Fine-tune classifier with 10-fold CV
    print(f"\n[5/5] Fine-tuning classifier ({CFG.N_FOLDS}-Fold CV)...")
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    
    # Get encoder features for train and test
    dae.eval()
    with torch.no_grad():
        train_features = dae.encode(torch.tensor(train_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        test_features = dae.encode(torch.tensor(test_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    
    print(f"  Encoder features shape: {train_features.shape}")
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, y_all)):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr = train_features[train_idx]
        y_tr = y_all[train_idx]
        X_val = train_features[val_idx]
        y_val = y_all[val_idx]
        X_te = test_features
        
        # Convert to tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        
        # Simple classifier
        classifier = nn.Sequential(
            nn.Linear(DAE_PARAMS['bottleneck'], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=DAE_PARAMS['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DAE_PARAMS['finetune_epochs'])
        criterion = nn.BCELoss()
        
        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=DAE_PARAMS['batch_size'], shuffle=True)
        
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(DAE_PARAMS['finetune_epochs']):
            classifier.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred_batch = classifier(xb).squeeze(-1)
                loss = criterion(pred_batch, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validation
            classifier.eval()
            with torch.no_grad():
                val_pred = classifier(X_val_t).squeeze(-1).cpu().numpy()
                val_auc = roc_auc_score(y_val, val_pred)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= DAE_PARAMS['patience']:
                    break
        
        # Load best model and predict
        classifier.load_state_dict(best_state)
        classifier.eval()
        with torch.no_grad():
            val_probs = classifier(X_val_t).squeeze(-1).cpu().numpy()
            test_probs = classifier(X_te_t).squeeze(-1).cpu().numpy()
        
        oof[val_idx] = val_probs
        pred += test_probs / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)
        print(f"   Fold {i+1} AUC : {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del classifier, X_tr_t, X_val_t, X_te_t
        gc.collect()
        torch.cuda.empty_cache()

    # Results
    overall_auc = roc_auc_score(y_all, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\n{'='*80}")
    print(f"V61 RESULTS — DAE Pre-training + V16 Features")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V21 Reference (TabM):  0.91898 (OOF) / LB 0.91682")
    print(f"V16b Reference (XGB):  0.91925 (OOF) / LB 0.91680")
    print(f"Delta vs V21:   {overall_auc - 0.91898:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    
    verdict = "IMPROVED" if overall_auc > 0.91898 else "MARGINAL" if overall_auc > 0.91850 else "SAME"
    print(f"Verdict: {verdict}")
    
    # Save outputs
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
