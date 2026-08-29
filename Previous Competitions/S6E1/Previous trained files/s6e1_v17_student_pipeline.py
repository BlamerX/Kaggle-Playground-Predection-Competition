import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler

# --- Constants ---
SEED = 42
N_SPLITS = 5
TARGET = 'exam_score'
USE_GPU = True   # Set to True for Kaggle T4
DEBUG = False    # Set to True for fast local testing (1% data)
TUNE_MODE = False  # Set to False to use hardcoded best params, True to re-tune (7+ hours)

# --- GPU Configurations ---
if USE_GPU:
    print("🚀 GPU Training Enabled")
    xgb_params_base = {'tree_method': 'hist', 'device': 'cuda', 'n_jobs': -1}
    lgb_params_base = {'device': 'gpu', 'n_jobs': -1, 'verbose': -1}
    cat_params_base = {'task_type': 'GPU', 'verbose': 0, 'allow_writing_files': False}
else:
    print("🐢 CPU Training Mode")
    xgb_params_base = {'tree_method': 'hist', 'n_jobs': -1}
    lgb_params_base = {'n_jobs': -1, 'verbose': -1}
    cat_params_base = {'verbose': 0, 'allow_writing_files': False}

# --- Data Loading ---
print("Loading data...")
try:
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    original = pd.read_csv('Exam_Score_Prediction.csv')
except FileNotFoundError:
    # Kaggle Paths
    train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    original = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Original shape: {original.shape}")

# --- Data Preprocessing & Alignment ---
if 'student_id' in original.columns:
    original = original.rename(columns={'student_id': 'id'})
if 'attendance_percentage' in original.columns:
    original = original.rename(columns={'attendance_percentage': 'class_attendance'})
if 'study_hours_per_day' in original.columns:
    original = original.rename(columns={'study_hours_per_day': 'study_hours'})

common_cols = [c for c in train.columns if c in original.columns]
train_final = pd.concat([train, original[common_cols]], axis=0).reset_index(drop=True)

if DEBUG:
    print("⚠️ DEBUG MODE: Using 1% of data")
    train_final = train_final.sample(frac=0.01, random_state=SEED).reset_index(drop=True)

print(f"Final Train shape: {train_final.shape}")

# Drop ID
train_final = train_final.drop(columns=['id'])
test_ids = test['id']
test_final = test.drop(columns=['id'])

# Separate Target
X = train_final.drop(columns=[TARGET])
y = train_final[TARGET]
X_test = test_final

# --- Pipeline Definition ---
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)

# --- Hyperparameter Tuning (Optuna) ---

def objective(trial, model_name):
    # Preprocess Data First (to save time inside CV) - ideally inside pipeline but OK for specific tuning
    # For correctness we keep Pipeline inside loop OR assume no leakage from Imputer/Scaler (minimal)
    # To be safe and fast, we use Pipeline inside CV loop as before.
    
    if model_name == 'XGBoost':
        # V14 Best: n_est=32k, max_leaves=9, lr=0.025, subsample=0.87, colsample=0.94
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 2000, 10000), # Cap at 10k for speed, 32k is huge
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_leaves': trial.suggest_int('max_leaves', 0, 50), # 0 is unlimited
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 30.0), # V14 was 22.2
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 100), # V14 was small
            **xgb_params_base
        }
        # Note: 'max_leaves' works well when max_depth is high or grow_policy is lossguide
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED, **params)
        
    elif model_name == 'LightGBM':
        # V14 Best: n_est=7860, num_leaves=5, lr=0.14
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 2000, 9000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 100), # Centered around 5 but allow more
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0), # V14 was 5.6
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
            **lgb_params_base
        }
        model = lgb.LGBMRegressor(random_state=SEED, **params)
        
    elif model_name == 'CatBoost':
        # V14 Best: n_est=8192, lr=0.188, early_stopping=11
        params = {
            'iterations': trial.suggest_int('iterations', 2000, 9000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_seed': SEED,
            **cat_params_base
        }
        model = cb.CatBoostRegressor(**params)
        
    elif model_name == 'RandomForest':
        # V14 Best: n_est=1252, max_features=0.26
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
            'max_depth': trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1,
            'random_state': SEED
        }
        model = RandomForestRegressor(**params)
        
    elif model_name == 'Ridge':
        alpha = trial.suggest_float('alpha', 0.01, 100.0, log=True)
        model = Ridge(alpha=alpha, random_state=SEED)

    # Pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', model)])
    
    # 3-Fold for Tuning Speed (5-Fold for Final)
    kf_tune = KFold(n_splits=3, shuffle=True, random_state=SEED)
    
    cv_scores = []
    
    # Manual Loop to ensure clipping and proper metrics (root_mean_squared_error is sklearn > 1.4)
    for train_idx, val_idx in kf_tune.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        val_pred = clf.predict(X_val)
        val_pred = np.clip(val_pred, 0, 100)
        
        score = root_mean_squared_error(y_val, val_pred)
        cv_scores.append(score)
        
    return np.mean(cv_scores)

if TUNE_MODE:
    print("\n🔮 Starting Hyperparameter Tuning (Optuna)...")
    print("🎯 Using Targeted Search Ranges from V14 AutoML Insights")
    study_results = {}
    
    # Tuning List: XGB, LGBM, CatBoost, RF, Ridge
    models_to_tune = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'Ridge']
    
    for name in models_to_tune:
        print(f"\n⚡ Tuning {name}...")
        
        # Reduced trials for RandomForest as it's slow
        n_trials = 10 if name == 'RandomForest' else 30 
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
        study.optimize(lambda trial: objective(trial, name), n_trials=n_trials)
        
        print(f"  Best {name} params: {study.best_params}")
        print(f"  Best {name} RMSE: {study.best_value:.5f}")
        study_results[name] = study.best_params
    
    print("\nSaving Best Params...")
    import json
    with open('v17_best_params.json', 'w') as f:
        json.dump(study_results, f, indent=4)

else:
    # =========================================================================
    # FINAL TRAINING with HARDCODED Best Params (from v17_log.txt)
    # =========================================================================
    print("\n" + "="*50)
    print("🚀 FINAL TRAINING with V17 Best Optuna Params")
    print("="*50)
    
    # Best params from V17 Optuna tuning (7 hours run)
    best_xgb_params = {
        'n_estimators': 8859, 
        'learning_rate': 0.026140079563340732, 
        'max_depth': 4, 
        'max_leaves': 0, 
        'subsample': 0.8385686851279809, 
        'colsample_bytree': 0.8827503742849342, 
        'reg_alpha': 0.015325553183452379, 
        'reg_lambda': 16.478885133859453, 
        'min_child_weight': 36
    }
    
    best_lgb_params = {
        'n_estimators': 7746, 
        'learning_rate': 0.014681402803987357, 
        'num_leaves': 46, 
        'max_depth': 5, 
        'subsample': 0.9913019543592999, 
        'colsample_bytree': 0.6789906952929146, 
        'reg_alpha': 2.6242227784438223, 
        'reg_lambda': 0.08210063306974046
    }
    
    best_cat_params = {
        'iterations': 6942, 
        'learning_rate': 0.03896319829750166, 
        'depth': 5, 
        'l2_leaf_reg': 0.04455066178195098, 
        'random_strength': 1.1349835828662909, 
        'bagging_temperature': 0.2126303629303074
    }
    
    final_models = {
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED, **best_xgb_params, **xgb_params_base),
        'LightGBM': lgb.LGBMRegressor(random_state=SEED, **best_lgb_params, **lgb_params_base),
        'CatBoost': cb.CatBoostRegressor(random_seed=SEED, **best_cat_params, **cat_params_base),
    }
    
    kf_final = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    final_oof = {name: np.zeros(len(X)) for name in final_models}
    final_test_preds = {name: np.zeros(len(X_test)) for name in final_models}
    final_results = {}
    
    for name, model in final_models.items():
        print(f"\n📊 Final CV: {name}")
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf_final.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            clf.fit(X_tr, y_tr)
            val_pred = clf.predict(X_val)
            val_pred = np.clip(val_pred, 0, 100)
            final_oof[name][val_idx] = val_pred
            
            score = root_mean_squared_error(y_val, val_pred)
            cv_scores.append(score)
            print(f"  Fold {fold+1} RMSE: {score:.5f}")
            
            test_fold_pred = clf.predict(X_test)
            test_fold_pred = np.clip(test_fold_pred, 0, 100)
            final_test_preds[name] += test_fold_pred / N_SPLITS
        
        avg_score = np.mean(cv_scores)
        final_results[name] = avg_score
        print(f"  {name} Final OOF RMSE: {avg_score:.5f}")
    
    # Sort and Find Best
    sorted_final = sorted(final_results.items(), key=lambda x: x[1])
    best_model_name = sorted_final[0][0]
    best_model_score = sorted_final[0][1]
    
    print("\n" + "="*50)
    print("📈 FINAL RESULTS")
    print("="*50)
    for name, score in sorted_final:
        marker = " 🏆" if name == best_model_name else ""
        print(f"{name}: {score:.5f}{marker}")
    
    # Save Best Model Submission
    print(f"\n✅ Using {best_model_name} for submission (RMSE: {best_model_score:.5f})")
    submission = pd.DataFrame({'id': test_ids, TARGET: final_test_preds[best_model_name]})
    submission.to_csv('submission_v17.csv', index=False)
    print("Saved submission_v17.csv")
    
    # Save OOF
    oof_df = pd.DataFrame({'id': train['id'], TARGET: final_oof[best_model_name]})
    oof_df.to_csv('oof_v17.csv', index=False)
    print("Saved oof_v17.csv")
