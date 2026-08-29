
# V19.2 TabM - Baseline features + HP tuning only
# V19.1 cos features HURT (-0.004). Reverting to baseline features.

import subprocess
import sys
import gc
import warnings
import numpy as np
import pandas as pd
import torch

try:
    from pytabkit import TabM_D_Regressor
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Regressor

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
TARGET = 'exam_score'

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ============================================================================
# EXACT Feature Engineering from Baseline Notebook
# ============================================================================
def add_engineered_features(df, base_features):
    """EXACT replication of notebook's feature engineering."""
    df_temp = df.copy()
    
    num_features = ['study_hours', 'class_attendance', 'sleep_hours']
    
    # Cyclic features - EXACT from baseline (cos features HURT)
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32') 

    # Log and Squared (EXACT from notebook)
    for col in num_features:
        if col in df_temp.columns:
            df_temp[f'log_{col}'] = np.log1p(df_temp[col])
            df_temp[f'{col}_sq'] = df_temp[col] ** 2

    # Feature Formula (EXACT from notebook)
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )
    
    # Convert base features to string (EXACT from notebook)
    for col in base_features:
        df_temp[col] = df_temp[col].astype(str)
        
    return df_temp

def main():
    print("Loading Data...")
    try:
        train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
        test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
        original = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    except FileNotFoundError:
        try:
            train = pd.read_csv('train.csv')
            test = pd.read_csv('test.csv')
            original = pd.read_csv('Exam_Score_Prediction.csv')
        except FileNotFoundError:
            train = pd.read_csv('Dataset/train.csv')
            test = pd.read_csv('Dataset/test.csv')
            original = pd.read_csv('Dataset/Exam_Score_Prediction.csv')

    # Base features (EXACT from notebook)
    base_features = [col for col in train.columns if col not in [TARGET, 'id']]
    
    print("Feature Engineering...")
    train_eng = add_engineered_features(train, base_features)
    
    CATS = base_features
    NUMS = [col for col in train_eng.columns if col not in CATS + [TARGET, 'id']]
    
    print(f"Categoricals: {len(CATS)}, Numerics: {len(NUMS)}")
    
    # Fit encoders on train only (EXACT from notebook)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).set_output(transform="pandas")
    scaler = StandardScaler().set_output(transform="pandas")
    
    encoder.fit(train_eng[CATS])
    scaler.fit(train_eng[NUMS])
    
    def preprocess(df):
        df_eng = add_engineered_features(df, base_features)
        cats_encoded = encoder.transform(df_eng[CATS])
        nums_scaled = scaler.transform(df_eng[NUMS])
        return pd.concat([nums_scaled, cats_encoded], axis=1)
    
    X = preprocess(train)
    y = train[TARGET].values
    X_test = preprocess(test)
    X_original = preprocess(original)
    y_original = original[TARGET].values
    
    # TabM parameters - V19 FINAL (baseline is optimal)
    param_grid_TabM = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'random_state': SEED,
        'verbosity': 0,
        'arch_type': 'tabm-mini-normal',  # REVERTED: normal hurts
        'tabm_k': 24,
        'num_emb_type': 'pwl',
        'd_embedding': 16, 
        'batch_size': 256, 
        'lr': 1e-3,
        'n_epochs': 100,  # REVERTED to baseline
        'dropout': 0.11,
        'd_block': 256, 
        'n_blocks': 5,
        'patience': 4,  # REVERTED to baseline
        'weight_decay': 1e-2,
    }
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)  # Also changed to 42
    
    oof_preds = np.zeros(len(X))
    test_preds_list = []
    
    print(f"\nStarting {N_SPLITS}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        X_train_fold = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train_fold = y[train_idx]
        y_val = y[val_idx]
        
        # Mix Original Data (EXACT from notebook)
        X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
        y_train_combined = np.concatenate([y_train_fold, y_original], axis=0)
        
        model = TabM_D_Regressor(**param_grid_TabM)
        
        model.fit(
            X_train_combined, 
            y_train_combined, 
            X_val, 
            y_val, 
            cat_col_names=CATS
        )
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.5f}")
        
        test_p = model.predict(X_test)
        test_preds_list.append(test_p)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    
    print("-----------------------")
    print(f"Overall OOF RMSE: {oof_rmse:.5f}")
    
    # Save outputs
    oof_df = pd.DataFrame({'id': train['id'], TARGET: oof_preds})
    oof_df.to_csv('oof_v19_tabm.csv', index=False)
    
    submission = pd.DataFrame({'id': test['id'], TARGET: np.mean(test_preds_list, axis=0)})
    submission.to_csv('submission_v19_tabm.csv', index=False)
    print("Saved submission_v19_tabm.csv (V19.2)")

if __name__ == "__main__":
    main()