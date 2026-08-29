"""
S6E3 V24 - FT-Transformer (Feature Tokenizer + Transformer) with V16 Features
================================================================================
Architecture: Pure PyTorch FT-Transformer (Gorishniy et al., 2021)
  - Feature Tokenizer: each numeric → linear projection, each cat → embedding lookup
  - [CLS] token prepended to feature sequence
  - L=3 Transformer encoder layers (pre-norm, MHSA + FFN, GELU)
  - CLS token → Linear classifier (sigmoid)
  - AdamW optimizer, cosine LR schedule, early stopping on val AUC

V16 Features:
  - Numerics (120): V7 core + ORIG_proba + dist/quantile + 35 digit features + TE features
  - Categoricals (16 raw CATS): OrdinalEncoded → embedding lookup

Self-attention across all feature tokens captures GLOBAL feature interactions
that neither tree models nor MLP-based models (RealMLP, TabM) can express.
Third distinct NN architecture → maximum ensemble diversity.

Hyperparams (from FTT paper + practical tuning):
  d_token = 64, n_heads = 4, n_layers = 3, ffn_factor = 4/3
  dropout_attn = 0.1, dropout_ffn = 0.1
  lr = 1e-4, wd = 1e-5, batch_size = 512, n_epochs = 100, patience = 10

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / STACKING / BLENDING (single FTT model)
"""

import os
import gc
import sys
import random
import warnings
import time
import math
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

class CFG:
    VERSION       = "v24"
    EXP_ID        = "S6E3_V24_FTT_V16Features"
    TRAIN_PATH    = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH     = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    TARGET        = 'Churn'

    SEED          = 42
    N_FOLDS       = 10
    INNER_FOLDS   = 5

    # FTT hyperparameters
    D_TOKEN       = 64       # token dimension per feature
    N_HEADS       = 4        # attention heads (D_TOKEN must be divisible)
    N_LAYERS      = 3        # transformer encoder layers
    FFN_FACTOR    = 4 / 3    # FFN hidden dim = D_TOKEN × FFN_FACTOR
    DROPOUT       = 0.1
    LR            = 1e-4
    WD            = 1e-5
    BATCH_SIZE    = 512
    N_EPOCHS      = 100
    PATIENCE      = 10       # early stopping on val AUC (no improvement)

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

# ─── FT-Transformer Model ─────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Numeric features: x_i → W_i @ x_i + b_i (per-feature linear projection to d_token)
    Categorical features: cat_i → Embedding(n_cats_i, d_token)
    """
    def __init__(self, n_num, cat_cardinalities, d_token):
        super().__init__()
        self.n_num = n_num
        self.d_token = d_token

        # Numeric: weight (n_num, d_token) and bias (n_num, d_token)
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.Tensor(n_num, d_token))
            self.num_bias   = nn.Parameter(torch.Tensor(n_num, d_token))
            nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))
            nn.init.zeros_(self.num_bias)

        # Categorical: one embedding per feature
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(n + 1, d_token) for n in cat_cardinalities
        ]) if cat_cardinalities else None

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x_num, x_cat):
        tokens = []
        if self.n_num > 0 and x_num is not None:
            # (B, n_num) → (B, n_num, d_token)
            num_tok = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tok)
        if self.cat_embeds is not None and x_cat is not None:
            cat_tok = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)], dim=1)
            tokens.append(cat_tok)

        all_tokens = torch.cat(tokens, dim=1)  # (B, n_features, d_token)

        # Prepend CLS
        cls = self.cls_token.expand(all_tokens.size(0), -1, -1)
        return torch.cat([cls, all_tokens], dim=1)  # (B, n_features+1, d_token)


class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, ffn_d, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, ffn_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_d, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformerModel(nn.Module):
    def __init__(self, n_num, cat_cardinalities, d_token, n_heads, n_layers, ffn_factor, dropout):
        super().__init__()
        ffn_d = max(1, int(d_token * ffn_factor))
        self.tokenizer   = FeatureTokenizer(n_num, cat_cardinalities, d_token)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_token, n_heads, ffn_d, dropout)
            for _ in range(n_layers)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.head      = nn.Linear(d_token, 1)

    def forward(self, x_num, x_cat):
        tokens = self.tokenizer(x_num, x_cat)   # (B, n+1, d)
        tokens = self.transformer(tokens)        # (B, n+1, d)
        cls    = self.head_norm(tokens[:, 0])    # CLS token (B, d)
        return self.head(cls).squeeze(-1)        # (B,)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype='float32') if sigma == 0
            else ((values - mu) / sigma).astype('float32'))

def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=DEVICE)

def train_ftt(model, X_num_tr, X_cat_tr, y_tr,
              X_num_va, X_cat_va, y_va, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.N_EPOCHS)
    crit = nn.BCEWithLogitsLoss()

    best_auc, patience_cnt, best_state = 0.0, 0, None
    n = len(y_tr)

    for epoch in range(cfg.N_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        loss_sum = 0.0
        for i in range(0, n, cfg.BATCH_SIZE):
            idx = perm[i:i + cfg.BATCH_SIZE]
            xn = X_num_tr[idx] if X_num_tr is not None else None
            xc = X_cat_tr[idx] if X_cat_tr is not None else None
            yb = y_tr[idx]
            opt.zero_grad()
            loss = crit(model(xn, xc), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
        scheduler.step()

        # Val AUC — batched to avoid OOM on large val sets
        model.eval()
        probs_list = []
        n_va = X_num_va.size(0) if X_num_va is not None else X_cat_va.size(0)
        with torch.no_grad():
            for i in range(0, n_va, 2048):
                xn = X_num_va[i:i+2048] if X_num_va is not None else None
                xc = X_cat_va[i:i+2048] if X_cat_va is not None else None
                probs_list.append(torch.sigmoid(model(xn, xc)).cpu().numpy())
        probs = np.concatenate(probs_list)
        auc = roc_auc_score(y_va.cpu().numpy(), probs)

        if auc > best_auc + 1e-6:
            best_auc     = auc
            patience_cnt = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return best_auc

def predict_ftt(model, X_num, X_cat, batch_size=2048):
    model.eval()
    preds = []
    n = X_num.size(0) if X_num is not None else X_cat.size(0)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xn = X_num[i:i+batch_size] if X_num is not None else None
            xc = X_cat[i:i+batch_size] if X_cat is not None else None
            preds.append(torch.sigmoid(model(xn, xc)).cpu().numpy())
    return np.concatenate(preds)


def main():
    seed_everything(CFG.SEED)
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print("=" * 80)
    print("\nV24 Architecture:")
    print(f"  FT-Transformer: d_token={CFG.D_TOKEN}, n_heads={CFG.N_HEADS}, n_layers={CFG.N_LAYERS}")
    print(f"  lr={CFG.LR}, batch={CFG.BATCH_SIZE}, patience={CFG.PATIENCE}")
    print("  V16 Features: digit (35) + N-gram TE (19) + V7 core")
    print("  10-Fold CV, seed=42")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIGINAL_PATH)

    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET]  = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)

    train_ids = train['id'].copy()
    test_ids  = test['id'].copy()
    print(f"  Train:{train.shape}  Test:{test.shape}  Orig:{orig.shape}")

    # ── Feature Engineering (V16 pipeline) ───────────────────────────────────
    print("\n[2/5] Feature Engineering (V16 pipeline)...")
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS    = ['tenure', 'MonthlyCharges', 'TotalCharges']
    NEW_NUMS = []

    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')

    for df in [train, test, orig]:
        df['charges_deviation']      = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    SVC = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
           'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet']  = (df['InternetService'] != 'No').astype('float32')
        df['has_phone']     = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']

    for col in CATS + NUMS:
        tmp   = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test  = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)

    orig_ch_tc = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nc_tc = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc    = orig['TotalCharges'].values
    orig_is_mc = orig.groupby('InternetService')['MonthlyCharges'].mean()

    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nc_tc)
        df['pctrank_churner_TC']    = pctrank_against(tc, orig_ch_tc)
        df['pctrank_orig_TC']       = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC']   = (np.abs(zscore_against(tc, orig_ch_tc)) -
                                       np.abs(zscore_against(tc, orig_nc_tc))).astype('float32')
        df['zscore_nonchurner_TC']  = zscore_against(tc, orig_nc_tc)
        df['pctrank_churn_gap_TC']  = (pctrank_against(tc, orig_ch_tc) -
                                       pctrank_against(tc, orig_nc_tc)).astype('float32')
        df['resid_IS_MC']           = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc).fillna(0)).astype('float32')
        for cat_col, out_col in [('InternetService','cond_pctrank_IS_TC'), ('Contract','cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cv in orig[cat_col].unique():
                mask = df[cat_col] == cv
                ref  = orig.loc[orig[cat_col] == cv, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[out_col] = vals

    NEW_NUMS += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}']   = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}']   = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
    NEW_NUMS += [
        'qdist_gap_To_q50','dist_To_ch_q50','dist_To_nc_q50',
        'dist_To_nc_q25','qdist_gap_To_q25',
        'dist_To_nc_q75','dist_To_ch_q75','qdist_gap_To_q75'
    ]

    # ── Digit Features ────────────────────────────────────────────────────────
    print("\n[3/5] V16 Digit Features...")
    DIGIT_FEATURES = [
        'tenure_first_digit','tenure_last_digit','tenure_second_digit',
        'tenure_mod10','tenure_mod12','tenure_num_digits',
        'tenure_is_multiple_10','tenure_rounded_10','tenure_dev_from_round10',
        'mc_first_digit','mc_last_digit','mc_second_digit',
        'mc_mod10','mc_mod100','mc_num_digits',
        'mc_is_multiple_10','mc_is_multiple_50',
        'mc_rounded_10','mc_fractional','mc_dev_from_round10',
        'tc_first_digit','tc_last_digit','tc_second_digit',
        'tc_mod10','tc_mod100','tc_num_digits',
        'tc_is_multiple_10','tc_is_multiple_100',
        'tc_rounded_100','tc_fractional','tc_dev_from_round100',
        'tenure_years','tenure_months_in_year','mc_per_digit','tc_per_digit'
    ]
    for df in [train, test]:
        t_str  = df['tenure'].astype(str)
        df['tenure_first_digit']      = t_str.str[0].astype(int)
        df['tenure_last_digit']       = t_str.str[-1].astype(int)
        df['tenure_second_digit']     = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10']            = df['tenure'] % 10
        df['tenure_mod12']            = df['tenure'] % 12
        df['tenure_num_digits']       = t_str.str.len()
        df['tenure_is_multiple_10']   = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10']       = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])
        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit']      = mc_str.str[0].astype(int)
        df['mc_last_digit']       = mc_str.str[-1].astype(int)
        df['mc_second_digit']     = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10']            = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100']           = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits']       = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10']   = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50']   = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10']       = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional']       = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])
        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit']       = tc_str.str[0].astype(int)
        df['tc_last_digit']        = tc_str.str[-1].astype(int)
        df['tc_second_digit']      = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10']             = np.floor(df['TotalCharges']) % 10
        df['tc_mod100']            = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits']        = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10']    = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100']   = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100']       = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional']        = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])
        df['tenure_years']         = df['tenure'] // 12
        df['tenure_months_in_year']= df['tenure'] % 12
        df['mc_per_digit']         = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit']         = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)
        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')
    NEW_NUMS += DIGIT_FEATURES
    print(f"  ✅ {len(DIGIT_FEATURES)} digit features added")

    # ── N-gram Columns ────────────────────────────────────────────────────────
    print("\n[4/5] N-gram Categorical Features (V16 Top-6)...")
    BIGRAM_COLS, TRIGRAM_COLS = [], []
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
    print(f"  ✅ {len(NGRAM_COLS)} n-gram columns")

    NUM_AS_CAT = []
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)
    TE_COLUMNS  = NUM_AS_CAT + CATS

    # ── Training (10-Fold CV) ─────────────────────────────────────────────────
    print(f"\n[5/5] Training FT-Transformer ({CFG.N_FOLDS}-Fold CV)...")
    skf       = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof         = np.zeros(len(train))
    pred        = np.zeros(len(test))
    fold_scores = []
    y_all       = train[CFG.TARGET].values

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")
        seed_everything(CFG.SEED + fold_i)

        X_tr  = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr  = y_all[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te  = test.copy()

        # Inner K-Fold TE (mean only)
        te_feat_names = [f"TE1_{col}_mean" for col in TE_COLUMNS]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.0

        X_tr[CFG.TARGET] = y_tr
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in TE_COLUMNS:
                tmp    = X_tr2.groupby(col)[CFG.TARGET].mean().rename(f"TE1_{col}_mean")
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how='left')
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = merged[f"TE1_{col}_mean"].values

        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col)[CFG.TARGET].mean().rename(f"TE1_{col}_mean")
            X_val[f"TE1_{col}_mean"] = X_val[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
            X_te[f"TE1_{col}_mean"]  = X_te[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(0.5).astype('float32')

        # N-gram TE
        ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
        X_tr[CFG.TARGET] = y_tr
        for col in NGRAM_COLS:
            ng_te = X_tr.groupby(col)[CFG.TARGET].mean()
            ng_n  = f"TE_ng_{col}"
            X_tr[ng_n]  = X_tr[col].map(ng_te).fillna(0.5).astype('float32')
            X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype('float32')
            X_te[ng_n]  = X_te[col].map(ng_te).fillna(0.5).astype('float32')
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # Build numeric and categorical arrays
        NUM_FEATURES = NUMS + NEW_NUMS + te_feat_names + ng_te_feat_names
        CAT_FEATURES = CATS

        if fold_i == 0:
            print(f"  Numeric features: {len(NUM_FEATURES)}")
            print(f"  Categorical features: {len(CAT_FEATURES)}")
            print(f"  Total tokens per sample (incl. CLS): {len(NUM_FEATURES) + len(CAT_FEATURES) + 1}")

        # Scale numerics
        scaler = StandardScaler()
        X_num_tr  = scaler.fit_transform(X_tr[NUM_FEATURES].fillna(0).astype('float32')).astype('float32')
        X_num_val = scaler.transform(X_val[NUM_FEATURES].fillna(0).astype('float32')).astype('float32')
        X_num_te  = scaler.transform(X_te[NUM_FEATURES].fillna(0).astype('float32')).astype('float32')

        # Ordinal encode categoricals
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(X_tr[CAT_FEATURES].astype(str))
        X_cat_tr  = enc.transform(X_tr[CAT_FEATURES].astype(str)).astype(int)
        X_cat_val = enc.transform(X_val[CAT_FEATURES].astype(str)).astype(int)
        X_cat_te  = enc.transform(X_te[CAT_FEATURES].astype(str)).astype(int)
        # Shift -1 (unknown) to 0, shift valid values by 1
        X_cat_tr  = np.clip(X_cat_tr + 1, 0, None)
        X_cat_val = np.clip(X_cat_val + 1, 0, None)
        X_cat_te  = np.clip(X_cat_te + 1, 0, None)
        cat_cardinalities = [len(enc.categories_[i]) for i in range(len(CAT_FEATURES))]

        # To GPU tensors
        tn  = lambda a: to_tensor(a, torch.float32)
        ti  = lambda a: to_tensor(a, torch.long)
        T_num_tr, T_cat_tr, T_y_tr = tn(X_num_tr), ti(X_cat_tr), to_tensor(y_tr.astype('float32'))
        T_num_va, T_cat_va, T_y_va = tn(X_num_val), ti(X_cat_val), to_tensor(y_val.astype('float32'))
        T_num_te, T_cat_te         = tn(X_num_te), ti(X_cat_te)

        # Build & train FTT
        model = FTTransformerModel(
            n_num=len(NUM_FEATURES),
            cat_cardinalities=cat_cardinalities,
            d_token=CFG.D_TOKEN,
            n_heads=CFG.N_HEADS,
            n_layers=CFG.N_LAYERS,
            ffn_factor=CFG.FFN_FACTOR,
            dropout=CFG.DROPOUT
        ).to(DEVICE)

        best_val_auc = train_ftt(model, T_num_tr, T_cat_tr, T_y_tr,
                                  T_num_va, T_cat_va, T_y_va, CFG)

        oof[val_idx]  = predict_ftt(model, T_num_va, T_cat_va)
        fold_auc      = roc_auc_score(y_val, oof[val_idx])
        fold_scores.append(fold_auc)

        pred += predict_ftt(model, T_num_te, T_cat_te) / CFG.N_FOLDS

        # V21 per-fold reference
        V21_FOLDS = [0.91945, 0.91820, 0.92080, 0.91848, 0.91825,
                     0.91940, 0.92104, 0.91948, 0.91852, 0.91685]
        v21_ref = V21_FOLDS[fold_i] if fold_i < len(V21_FOLDS) else None
        delta   = f"{fold_auc - v21_ref:+.5f}" if v21_ref else "N/A"
        print(f"   Fold {fold_i+1} AUC: {fold_auc:.5f} (V21 TabM: {v21_ref:.5f} | Δ={delta}) | {(time.time()-t0)/60:.1f} min")

        del model, T_num_tr, T_cat_tr, T_y_tr, T_num_va, T_cat_va, T_num_te, T_cat_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Results ───────────────────────────────────────────────────────────────
    overall_auc = roc_auc_score(y_all, oof)
    mean_score  = np.mean(fold_scores)
    std_score   = np.std(fold_scores)
    V21_OOF     = 0.91898

    print(f"\n{'='*80}")
    print(f"V24 RESULTS — FT-Transformer with V16 Features")
    print(f"{'='*80}")
    print(f"Overall CV AUC  : {overall_auc:.5f}  (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V21 TabM OOF    : {V21_OOF:.5f}  (LB 0.91682 — best NN reference)")
    print(f"Delta vs V21    : {overall_auc - V21_OOF:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")

    verdict = ("🏆 BEATS V21"  if overall_auc > V21_OOF + 0.00020 else
               "✅ COMPETITIVE" if abs(overall_auc - V21_OOF) < 0.00050 else
               "⚠️ WORSE NN"  if overall_auc < V21_OOF - 0.00050 else
               "= SIMILAR")
    print(f"Verdict: {verdict}")

    # Save always (NN diversity is the value, not OOF rank)
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(
        f"oof_{CFG.VERSION}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(
        f"sub_{CFG.VERSION}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION}.csv  and  sub_{CFG.VERSION}.csv")
    print(f"Total time: {(time.time()-t0_all)/60:.1f} min")
    print("=" * 80)

if __name__ == "__main__":
    main()
