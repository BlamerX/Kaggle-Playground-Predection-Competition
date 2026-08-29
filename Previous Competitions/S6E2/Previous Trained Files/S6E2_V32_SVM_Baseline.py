
import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V32"
    DESCRIPTION = "SVM_Nystroem_Baseline"
    
    SEED = 42
    N_FOLDS = 5
    INNER_FOLDS = 5
    
    # Nystroem SVM Hyperparameters
    N_COMPONENTS = 2000 # Number of kernel components (higher = better approx, slower)
    GAMMA = 0.05        # Kernel coefficient (tune this!)
    ALPHA = 0.001       # L2 regularization strength
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

# ==================================================================================
# MAIN
# ==================================================================================

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train SVM (Support Vector Machine) Scalable Baseline.")
    print(f"      Uses Nystroem Kernel Approximation + SGDClassifier(loss='hinge').")
    print(f"================================================================================")
    
    start_time = time.time()
    
    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "train.csv")
    test = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "test.csv")
    try:
        orig = pd.read_csv(CFG.ORIG_PATH if os.path.exists(CFG.ORIG_PATH) else "Heart_Disease_Prediction.csv")
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if len(orig) > 0 and orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. Feature Engineering (Deotte Recipe)
    # SVM benefits heavily from Non-Linear features, so FE is crucial.
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']
    
    # Frequency
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat] if len(orig) > 0 else pd.Series(), test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            if len(df) > 0:
                df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0)
    
    # Num to Cat for TE
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            if len(df) > 0:
                df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS

    kf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds_accum = np.zeros(len(test))
    
    print(f"\nStarting SVM {CFG.N_FOLDS}-Fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(train, train['Heart Disease'])):
        
        # 1. Split Data
        X_tr = train.iloc[train_idx].copy()
        y_tr = train.iloc[train_idx]['Heart Disease'].values
        X_val = train.iloc[val_idx].copy()
        y_val_targets = train.iloc[val_idx]['Heart Disease'].values
        X_te = test.copy()
        
        # Augment
        if len(orig) > 0:
            X_tr_aug = pd.concat([X_tr, orig], axis=0).reset_index(drop=True)
            y_tr_aug = X_tr_aug['Heart Disease'].values
        else:
            X_tr_aug = X_tr.copy()
            y_tr_aug = y_tr
        
        # 2. Inner TE
        kf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feats = [f"TE_{c}_mean" for c in TE_COLUMNS]
        
        for df in [X_tr_aug, X_val, X_te]:
            for c in te_feats: df[c] = 0.0
            
        for i_tr, i_val in kf_inner.split(X_tr_aug, y_tr_aug):
            X_fold_tr = X_tr_aug.iloc[i_tr]
            X_fold_val = X_tr_aug.iloc[i_val]
            for col in TE_COLUMNS:
                mean_enc = X_fold_tr.groupby(col)['Heart Disease'].mean()
                X_tr_aug.loc[X_tr_aug.index[i_val], f"TE_{col}_mean"] = X_fold_val[col].map(mean_enc).fillna(X_fold_tr['Heart Disease'].mean())

        for col in TE_COLUMNS:
            mean_enc = X_tr_aug.groupby(col)['Heart Disease'].mean()
            global_mean = X_tr_aug['Heart Disease'].mean()
            X_val[f"TE_{col}_mean"] = X_val[col].map(mean_enc).fillna(global_mean)
            X_te[f"TE_{col}_mean"] = X_te[col].map(mean_enc).fillna(global_mean)

        # 3. Scale Features (Crucial for SVM/Nystroem)
        drop_cols = ['id', 'Heart Disease'] + TE_COLUMNS
        feat_cols = [c for c in X_tr_aug.columns if c not in drop_cols]
        
        X_tr_final = X_tr_aug[feat_cols].astype(float).values
        y_tr_final = y_tr_aug
        X_val_final = X_val[feat_cols].astype(float).values
        y_val_final = y_val_targets
        X_te_final = X_te[feat_cols].astype(float).values
        
        # Using QuantileTransformer for SVM often works best to "Gaussianize" inputs
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        X_tr_final = qt.fit_transform(X_tr_final)
        X_val_final = qt.transform(X_val_final)
        X_te_final = qt.transform(X_te_final)
        
        # 4. Define & Train Model Pipeline
        # Nystroem -> SGDClassifier (Hinge Loss) -> CalibratedClassifierCV (Sigmoid)
        
        base_pipeline = Pipeline([
            ('nystroem', Nystroem(
                kernel='rbf', 
                gamma=CFG.GAMMA, 
                n_components=CFG.N_COMPONENTS, 
                random_state=CFG.SEED,
                n_jobs=-1
            )),
            ('sgd', SGDClassifier(
                loss='hinge', 
                alpha=CFG.ALPHA, 
                max_iter=1000, 
                tol=1e-3, 
                random_state=CFG.SEED,
                n_jobs=-1,
                early_stopping=True,
                validation_fraction=0.1
            ))
        ])
        
        # Calibration is needed because SGD(hinge) does not produce probabilities
        calibrated_model = CalibratedClassifierCV(
            estimator=base_pipeline, 
            method='sigmoid', 
            cv=3 # Internal CV for calibration
        )
        
        calibrated_model.fit(X_tr_final, y_tr_final)
        
        val_preds = calibrated_model.predict_proba(X_val_final)[:, 1]
        auc = roc_auc_score(y_val_final, val_preds)
        print(f"  Fold {fold+1} SVM AUC: {auc:.5f}")
        
        oof_preds[val_idx] = val_preds
        test_preds_accum += calibrated_model.predict_proba(X_te_final)[:, 1] / CFG.N_FOLDS

    overall_score = roc_auc_score(train['Heart Disease'], oof_preds)
    print(f"\nOverall SVM CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds_accum})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
