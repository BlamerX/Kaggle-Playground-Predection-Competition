"""
S6E1 V135 - Recreation of S5E10 1st Place Solution (Genetic Programming + Stacking)
====================================================================================
Based on: https://www.kaggle.com/competitions/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming
Diagram: Genetic Programming + Autoencoder + Keras Ensemble -> CatBoost Ensemble -> Hill Climbing

Architecture:
1. BASE MODELS (Pre-computed OOFs):
   - TabM (V125), FTT (V127) -> Acts as "Keras/NN Ensemble"
   - XGB (V124), CatB (V123), LGB (V126)
   - V128/V133 (Current Best Ensemble) -> Acts as "Baseline"

2. FEATURE ENGINEERING:
   - Genetic Programming (GP): Generate non-linear interactions
   - Autoencoder (AE): dimensionality reduction / feature extraction

3. META-LEARNER ("The Red Box"):
   - Model: CatBoost
   - Input: [Original_Features, GP_Features, AE_Features, NN_Ensemble_Preds]
   - Target: exam_score

4. FINAL OPTIMIZATION:
   - Hill Climbing blending Red Box output with other strong models.
"""

import pandas as pd
import numpy as np
import os
import time
import random
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Disable warnings
warnings.filterwarnings('ignore')

# Seed everything
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

print("=" * 80)
print("S6E1 V135 - S5E10 1st Place Strategy Recreation")
print("=" * 80)

# ============================================================
# 1. LOADING DATA & OOFs
# ============================================================
print("\n[1] Loading Data & Pre-computed OOFs...")

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    BASE_DIR = '/kaggle/input/playground-series-s6e1/'
    OOF_DIR = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/'
    SUB_DIR = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/'
else:
    print("Environment: LOCAL")
    BASE_DIR = 'Dataset/'
    OOF_DIR = 'Previous trained files/OOF/'
    SUB_DIR = 'Previous trained files/Submissions/'

train = pd.read_csv(BASE_DIR + 'train.csv')
test = pd.read_csv(BASE_DIR + 'test.csv')
orig = pd.read_csv(BASE_DIR + 'Exam_Score_Prediction.csv' if os.path.exists(BASE_DIR + 'Exam_Score_Prediction.csv') else '/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

y = train['exam_score'].values
initial_features = [c for c in train.columns if c not in ['id', 'exam_score']]

# Load OOFs to serve as "Ensemble Baseline" proxies
# Mapping to diagram:
# "Keras Ensemble" -> Average of V125 (TabM) + V127 (FTT)
# "Baseline" -> V128 or V133

def load_preds(name, oof_name, sub_name):
    try:
        oof = pd.read_csv(f"{OOF_DIR}{oof_name}")
        sub = pd.read_csv(f"{SUB_DIR}{sub_name}")
        
        # Sort by ID to ensure alignment
        if 'id' in oof.columns:
            oof = oof.sort_values('id').reset_index(drop=True)
        if 'id' in sub.columns:
            sub = sub.sort_values('id').reset_index(drop=True)
            
        oof_col = [c for c in oof.columns if c != 'id'][0]
        sub_col = [c for c in sub.columns if c != 'id'][0]
        
        return oof[oof_col].values, sub[sub_col].values
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None

# Load NN models for "Keras Ensemble" proxy
v125_oof, v125_pred = load_preds("V125_TabM", "oof_v125.csv", "submission_v125.csv")
v127_oof, v127_pred = load_preds("V127_FTT", "oof_v127.csv", "submission_v127.csv")

# Create NN Ensemble Proxy
if v125_oof is not None and v127_oof is not None:
    nn_baseline_oof = (v125_oof + v127_oof) / 2
    nn_baseline_pred = (v125_pred + v127_pred) / 2
    print("  Created 'NN Ensemble' proxy from V125+V127")
else:
    print("  WARNING: Could not load NN OOFs. Using placeholder.")
    nn_baseline_oof = np.zeros(len(train))
    nn_baseline_pred = np.zeros(len(test))

# Load V128 as general baseline for residuals
v128_oof, v128_pred = load_preds("V128_Ridge", "oof_v128.csv", "submission_v128.csv")
if v128_oof is None:
    v128_oof = np.zeros(len(train)) # Fallback

# Load ADDITIONAL strong models for Stacking
v110_oof, v110_pred = load_preds("V110_CatDART", "oof_v110.csv", "submission_v110.csv") # Best Single
v122_oof, v122_pred = load_preds("V122_HC", "oof_v122.csv", "submission_v122.csv") # Best Ensemble
v67_oof, v67_pred = load_preds("V67_LGB", "oof_v67.csv", "submission_v67.csv") # Best LGB
v101_oof, v101_pred = load_preds("V101_Single", "oof_v101.csv", "submission_v101.csv") # Strong Single

# ============================================================
# 2. FEATURE ENGINEERING: GENETIC PROGRAMMING (Custom)
# ============================================================
print("\n[2] Feature Engineering: Genetic Programming...")

# Custom GP Generator to imply simple arithmetic features
# S5E10 winner used ~10-20 GP features. We will generate candidates and select best.

class SimpleGeneticGenerator:
    def __init__(self, n_features=20):
        self.n_features = n_features
        self.best_expressions = []
        
    def fit(self, X, y, residuals):
        # Generate random expressions using key numerical columns
        # FIX: Exclude target 'exam_score' and 'id' from feature candidates!
        nums = X.select_dtypes(include=[np.number]).columns.tolist()
        nums = [c for c in nums if c not in ['id', 'exam_score']]
        candidates = []
        
        # 1. Simple interactions
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                col1, col2 = nums[i], nums[j]
                
                # Multiplication
                expr_name = f"GP_mul_{col1}_{col2}"
                val = X[col1] * X[col2]
                corr = np.abs(np.corrcoef(val, residuals)[0,1])
                candidates.append((corr, expr_name, lambda x, c1=col1, c2=col2: x[c1] * x[c2]))
                
                # Division (safe)
                expr_name = f"GP_div_{col1}_{col2}"
                val = X[col1] / (X[col2] + 1e-5)
                corr = np.abs(np.corrcoef(val, residuals)[0,1])
                candidates.append((corr, expr_name, lambda x, c1=col1, c2=col2: x[c1] / (x[c2] + 1e-5)))
                
                # Difference
                expr_name = f"GP_sub_{col1}_{col2}"
                val = X[col1] - X[col2]
                corr = np.abs(np.corrcoef(val, residuals)[0,1])
                candidates.append((corr, expr_name, lambda x, c1=col1, c2=col2: x[c1] - x[c2]))

        # Select best k features correlated with RESIDUALS of the baseline
        # (This helps find features that explain what the model missed)
        candidates.sort(key=lambda x: x[0], reverse=True)
        self.best_expressions = candidates[:self.n_features]
        print(f"  Selected {len(self.best_expressions)} GP features based on residual correlation")
        for i, (corn, name, _) in enumerate(self.best_expressions[:5]):
            print(f"    #{i+1}: {name} (Corr: {corn:.4f})")
            
    def transform(self, X):
        res = pd.DataFrame(index=X.index)
        for _, name, func in self.best_expressions:
            res[name] = func(X)
        return res

# Calculate residuals from V128 baseline
residuals = y - v128_oof

# Generate GP features
gp_gen = SimpleGeneticGenerator(n_features=15)
gp_gen.fit(train, y, residuals)

X_train_gp = gp_gen.transform(train)
X_test_gp = gp_gen.transform(test)

# ============================================================
# 3. FEATURE ENGINEERING: AUTOENCODER
# ============================================================
print("\n[3] Feature Engineering: Autoencoder...")

# Standard scale features for AE
scaler = QuantileTransformer(output_distribution='normal')
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in ['id', 'exam_score']]

X_all_num = pd.concat([train[num_cols], test[num_cols]], axis=0)
X_all_scaled = scaler.fit_transform(X_all_num)

X_train_scaled = X_all_scaled[:len(train)]
X_test_scaled = X_all_scaled[len(train):]

# Define DAE
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Training AE on {device}...")

input_dim = X_train_scaled.shape[1]
model_ae = DenoisingAutoencoder(input_dim).to(device)
optimizer = optim.Adam(model_ae.parameters(), lr=0.005)
criterion = nn.MSELoss()

# Prepare loaders
train_tensor = torch.FloatTensor(X_train_scaled).to(device)
test_tensor = torch.FloatTensor(X_test_scaled).to(device)
loader = DataLoader(TensorDataset(train_tensor), batch_size=2048, shuffle=True)

# Train AE
for epoch in range(20): # Fast training
    for batch in loader:
        x = batch[0]
        # Noise injection
        noise = torch.randn_like(x) * 0.1
        x_noisy = x + noise
        
        encoded, decoded = model_ae(x_noisy)
        loss = criterion(decoded, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Extract Latent Features
with torch.no_grad():
    train_encoded, _ = model_ae(train_tensor)
    test_encoded, _ = model_ae(test_tensor)

X_train_ae = pd.DataFrame(train_encoded.cpu().numpy(), columns=[f"AE_{i}" for i in range(16)])
X_test_ae = pd.DataFrame(test_encoded.cpu().numpy(), columns=[f"AE_{i}" for i in range(16)])

print(f"  Generated {X_train_ae.shape[1]} AE features")

# ============================================================
# 4. PREPARE RED BOX DATASET
# ============================================================
print("\n[4] Preparing Meta-Learner Dataset (The Red Box)...")

# Base Features (Original) - use simple preprocessing
def simple_preprocess(df):
    d = df.copy()
    # Simple label encoding for cats
    for c in d.select_dtypes(include='object').columns:
        d[c] = d[c].astype('category').cat.codes
    return d[initial_features]

X_train_base = simple_preprocess(train).reset_index(drop=True)
X_test_base = simple_preprocess(test).reset_index(drop=True)

# Stack everything: [Base + GP + AE + NN_Preds]
X_train_meta = pd.concat([
    X_train_base, 
    X_train_gp.reset_index(drop=True), 
    X_train_ae.reset_index(drop=True)
], axis=1)

X_test_meta = pd.concat([
    X_test_base, 
    X_test_gp.reset_index(drop=True), 
    X_test_ae.reset_index(drop=True)
], axis=1)

# Add NN Baseline Prediction as a feature (Crucial step from diagram)
X_train_meta['nn_pred'] = nn_baseline_oof
X_test_meta['nn_pred'] = nn_baseline_pred

# Add other strong models to "Red Box" input (Stacking)
if v110_oof is not None:
    X_train_meta['v110_pred'] = v110_oof
    X_test_meta['v110_pred'] = v110_pred
if v67_oof is not None:
    X_train_meta['v67_pred'] = v67_oof
    X_test_meta['v67_pred'] = v67_pred
if v101_oof is not None:
    X_train_meta['v101_pred'] = v101_oof
    X_test_meta['v101_pred'] = v101_pred

print(f"  Final Meta-Feature Shape: {X_train_meta.shape}")

# ============================================================
# 5. TRAIN CATBOOST META-LEARNER
# ============================================================
print("\n[5] Training CatBoost Meta-Learner (Red Box)...")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
red_box_oof = np.zeros(len(train))
red_box_test = np.zeros(len(test))

params = {
    'iterations': 3000,
    'learning_rate': 0.01,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'verbose': 0,
    'early_stopping_rounds': 100,
    'task_type': 'GPU' if torch.cuda.is_available() else 'CPU'
}

for fold, (idx_tr, idx_val) in enumerate(kf.split(X_train_meta)):
    X_tr, y_tr = X_train_meta.iloc[idx_tr], y[idx_tr]
    X_val, y_val = X_train_meta.iloc[idx_val], y[idx_val]
    
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    
    red_box_oof[idx_val] = model.predict(X_val)
    red_box_test += model.predict(X_test_meta) / 10
    
    if fold % 2 == 0:
        print(f"  Fold {fold} RMSE: {model.get_best_score()['validation']['RMSE']:.5f}")

rmse_red = np.sqrt(mean_squared_error(y, red_box_oof))
print(f"  Red Box OOF RMSE: {rmse_red:.5f}")

# ============================================================
# 6. FINAL HILL CLIMBING
# ============================================================
print("\n[6] Final Hill Climbing Optimization...")

# Load other strong model OOFs for the final mix
models = {
    'Red_Box': red_box_oof,
    'V128_Ridge': v128_oof,
    'V123_CatB': load_preds("V123", "oof_v123.csv", "submission_v123.csv")[0],
    'V124_XGB': load_preds("V124", "oof_v124.csv", "submission_v124.csv")[0],
    'V110_CatDART': v110_oof,
    'V122_HC': v122_oof,
    'V101_Single': v101_oof
}

# Filter out None
models = {k: v for k, v in models.items() if v is not None}
X_stack = np.column_stack(list(models.values()))
model_names = list(models.keys())

def objective(weights):
    w = np.array(weights)
    pred = X_stack @ w
    # Constraint already handled by normalize in loop, but here simpler:
    return np.sqrt(mean_squared_error(y, pred))

# Constraint: sum(weights) = 1
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(len(models))]
init_w = np.ones(len(models)) / len(models)

res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=cons)
best_w = res.x

print("\nOptimal Weights:")
for name, w in zip(model_names, best_w):
    print(f"  {name:<15}: {w:.4f}")

# Final Prediction
red_test_pred = red_box_test
v128_test_pred = v128_pred
v123_test_pred = load_preds("V123", "oof_v123.csv", "submission_v123.csv")[1]
v124_test_pred = load_preds("V124", "oof_v124.csv", "submission_v124.csv")[1]

# Construct final test stack
test_preds_map = {
    'Red_Box': red_test_pred,
    'V128_Ridge': v128_test_pred,
    'V123_CatB': v123_test_pred,
    'V124_XGB': v124_test_pred,
    'V110_CatDART': v110_pred,
    'V122_HC': v122_pred,
    'V101_Single': v101_pred
}
test_stack_cols = [test_preds_map[name] for name in model_names]
X_test_stack = np.column_stack(test_stack_cols)

final_pred = X_test_stack @ best_w
final_oof_pred = X_stack @ best_w
final_rmse = np.sqrt(mean_squared_error(y, final_oof_pred))

print(f"\nFinal Hill Climb RMSE: {final_rmse:.5f}")

# ============================================================
# 7. SAVING
# ============================================================

pd.DataFrame({'id': test['id'], 'exam_score': final_pred}).to_csv("submission_v135.csv", index=False)
pd.DataFrame({'id': train['id'], 'exam_score': final_oof_pred}).to_csv("oof_v135.csv", index=False)

print(f"\nSaved v135 files. Total time: {(time.time()-start_time)/60:.1f} min")
print("="*80)
