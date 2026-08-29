# !pip install lightning -q

import os
# Please ensure 'lightning' is installed in your environment: pip install lightning

import gc
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

# --- Constants ---
SEED = 42
N_SPLITS = 5
BATCH_SIZE = 1024
EPOCHS = 40
LR = 2e-3
USE_GPU = torch.cuda.is_available()

# Set float32 matmul precision for speed
torch.set_float32_matmul_precision('medium')

print(f"Device: {'GPU' if USE_GPU else 'CPU'}")

# ============================================================================
# Feature Engineering (V16/V13 Exact - 45 Features)
# ============================================================================
def preprocess_v13(df):
    """V13 exact features - 45 total."""
    df_temp = df.copy()
    eps = 1e-5

    # Polynomials
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Log/sqrt transforms
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)
    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # Interactions
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    # Ratios
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    # Ordinal encoding
    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2}).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2}).fillna(1).astype(int)

    # Ordinal × numeric
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']

    # Ordinal × ordinal
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    # Binned 
    df_temp["age_bin_num"] = pd.cut(df_temp["age"], bins=[0,17,19,21,23,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["study_bin_num"] = pd.cut(df_temp["study_hours"], bins=[-1, 2, 4, 6, 8, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df_temp["sleep_bin_num"] = pd.cut(df_temp["sleep_hours"], bins=[-1,5,6,7,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["attendance_bin_num"] = pd.cut(df_temp["class_attendance"], bins=[-1,60,75,85,95,101], labels=[0,1,2,3,4]).astype(float)

    # Gaps
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()
    
    return df_temp

# ============================================================================
# PyTorch Components (ResNet Block)
# ============================================================================
class ResNetBlock(nn.Module):
    def __init__(self, indim, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, indim),
            nn.BatchNorm1d(indim),
            nn.Dropout(dropout)
        )
        self.activation = nn.ReLU() # Activation after addition

    def forward(self, x):
        return self.activation(x + self.block(x))

class StudentGradeNet(pl.LightningModule):
    def __init__(self, num_dim, cat_dims, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.save_hyperparameters()
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cats, min(50, (num_cats + 1) // 2)) 
            for num_cats in cat_dims
        ])
        
        total_emb_dim = sum(e.embedding_dim for e in self.embeddings)
        input_dim = num_dim + total_emb_dim
        
        # Initial projection
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual Blocks (Deep ResNet)
        self.res_blocks = nn.Sequential(
            ResNetBlock(hidden_dim, hidden_dim * 2, dropout),
            ResNetBlock(hidden_dim, hidden_dim * 2, dropout),
            ResNetBlock(hidden_dim, hidden_dim * 2, dropout)
        )
        
        # Output
        self.output = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x_num, x_cat):
        # Embeddings
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        
        # Concat with numeric
        x = torch.cat([x_num, x_emb], dim=1)
        
        # Main Network
        x = self.head(x)
        x = self.res_blocks(x)
        return self.output(x)

    def training_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        preds = self(x_num, x_cat).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num, x_cat, y = batch
        preds = self(x_num, x_cat).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=1e-4) # slightly higher decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    pl.seed_everything(SEED)
    
    print("Loading Data...")
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        original = pd.read_csv('Exam_Score_Prediction.csv')
    except FileNotFoundError:
        train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
        test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
        original = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

    TARGET = 'exam_score'
    
    # Feature Engineering (V16)
    print("Feature Engineering...")
    train_fe = preprocess_v13(train)
    test_fe = preprocess_v13(test)
    original_fe = preprocess_v13(original)
    
    # Identify col types
    numeric_cols = train_fe.select_dtypes(include=['int64', 'float64']).columns.drop(['id', TARGET], errors='ignore').tolist()
    cat_cols = train_fe.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Label Encode Categoricals
    print("Encoding Categoricals...")
    cat_dims = []
    
    # Fill Nulls First
    for df in [train_fe, test_fe, original_fe]:
        df[cat_cols] = df[cat_cols].fillna("MISSING")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on all unique values
        unique_vals = pd.concat([train_fe[col], test_fe[col], original_fe[col]]).unique()
        le.fit(unique_vals)
        
        train_fe[col] = le.transform(train_fe[col])
        test_fe[col] = le.transform(test_fe[col])
        original_fe[col] = le.transform(original_fe[col])
        
        # Add 1 for potential unknown handling if needed, though we covered all
        cat_dims.append(len(unique_vals))

    # 2. RankGauss (QuantileTransformer) for Numerics
    print("Applying RankGauss (QuantileTransformer)...")
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    
    # Fit on all data for robust distribution
    all_num = pd.concat([train_fe[numeric_cols], test_fe[numeric_cols], original_fe[numeric_cols]], axis=0)
    qt.fit(all_num)
    
    train_fe[numeric_cols] = qt.transform(train_fe[numeric_cols])
    test_fe[numeric_cols] = qt.transform(test_fe[numeric_cols])
    original_fe[numeric_cols] = qt.transform(original_fe[numeric_cols])
    
    # Prepare Arrays
    X_train_num = train_fe[numeric_cols].values.astype(np.float32)
    X_train_cat = train_fe[cat_cols].values.astype(np.int64)
    y_train = train[TARGET].values.astype(np.float32)

    X_test_num = test_fe[numeric_cols].values.astype(np.float32)
    X_test_cat = test_fe[cat_cols].values.astype(np.int64)
    
    X_orig_num = original_fe[numeric_cols].values.astype(np.float32)
    X_orig_cat = original_fe[cat_cols].values.astype(np.int64)
    y_orig = original[TARGET].values.astype(np.float32)

    # 5-Fold CV
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros((len(test), N_SPLITS))
    cv_scores = []
    
    print(f"Starting {N_SPLITS}-Fold Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_num, y_train)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        # Data Slicing
        # Valid: S6E1 Only
        X_val_num_f = X_train_num[val_idx]
        X_val_cat_f = X_train_cat[val_idx]
        y_val_f = y_train[val_idx]
        
        # Train: S6E1 Fold + Original
        X_tr_num_f = np.concatenate([X_train_num[train_idx], X_orig_num], axis=0)
        X_tr_cat_f = np.concatenate([X_train_cat[train_idx], X_orig_cat], axis=0)
        y_tr_f = np.concatenate([y_train[train_idx], y_orig], axis=0)
        
        # Datasets
        train_ds = TensorDataset(torch.tensor(X_tr_num_f), torch.tensor(X_tr_cat_f), torch.tensor(y_tr_f))
        val_ds = TensorDataset(torch.tensor(X_val_num_f), torch.tensor(X_val_cat_f), torch.tensor(y_val_f))
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 if USE_GPU else 0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Init Model
        model = StudentGradeNet(num_dim=len(numeric_cols), cat_dims=cat_dims)
        
        # Callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=6, mode="min", verbose=False)
        checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename=f"best_model_fold_{fold}_v18.1")
        
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu" if USE_GPU else "cpu",
            devices=1,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=False, # Disable for anti-timeout
            logger=False
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Predict
        best_path = checkpoint.best_model_path
        if best_path:
            model = StudentGradeNet.load_from_checkpoint(best_path)
            
        model.eval()
        model.to('cpu')
        
        with torch.no_grad():
            # OOF
            val_p = model(torch.tensor(X_val_num_f), torch.tensor(X_val_cat_f)).squeeze().numpy()
            # Test
            test_p = model(torch.tensor(X_test_num), torch.tensor(X_test_cat)).squeeze().numpy()
            
        val_p = np.clip(val_p, 0, 100)
        test_p = np.clip(test_p, 0, 100)
        
        oof_preds[val_idx] = val_p
        test_preds[:, fold] = test_p
        
        rmse = root_mean_squared_error(y_val_f, val_p)
        print(f"Fold {fold+1} Best RMSE: {rmse:.5f}")
        cv_scores.append(rmse)
        
        del model, trainer, train_ds, val_ds
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*30)
    print("RESULTS (V18.1 - ResNet + RankGauss)")
    print("="*30)
    mean_rmse = np.mean(cv_scores)
    print(f"Mean OOF RMSE: {mean_rmse:.5f}")
    
    oof_df = pd.DataFrame({'id': train['id'], 'exam_score': oof_preds})
    oof_df.to_csv('oof_v18_resnet.csv', index=False)
    
    submission = pd.DataFrame({'id': test['id'], 'exam_score': np.mean(test_preds, axis=1)})
    submission.to_csv('submission_v18_resnet.csv', index=False)
    print("Saved submission_v18_resnet.csv")

if __name__ == "__main__":
    main()