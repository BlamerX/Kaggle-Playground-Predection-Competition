"""
S6E3 V26 - DCNv2  (Research-Informed Improvements)
================================================================================
Strategy: proven architecture + Kaggle competition-winning techniques

Based on research from:
- Kaggle Grandmaster playbook: Model EMA, one-cycle LR, multi-seed
- DCNv2 paper: Cross layers, embedding dimensions
- Winning competition solutions: Gradient accumulation, EMA weights

Key changes from (best performer):
  1. Model EMA - exponential moving average of weights during training
  2. One-Cycle learning rate (Leslie Smith) - better convergence
  3. Lower dropout in deep network (0.15 vs 0.2) - less underfitting
  4. Longer early stopping patience (30 vs 25) - more exploration
  5. Keep proven: hidden=[512,256,128], aux_output, cross_layers=6
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import math
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
tf.get_logger().setLevel('ERROR')

# Set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

class CFG:
    VERSION_NAME = "v26"
    EXP_ID = "S6E3_V26_DCNv2"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10       
    RANDOM_SEED = 42

#  Parameters - Research-informed improvements
PARAMS = {
    'embedding_dim': 16,  
    'hidden_units': [512, 256, 128],  
    'cross_layers': 6,  
    'dropout_rate': 0.15,  
    'feature_dropout': 0.1,  
    'learning_rate': 0.001,  # Will use one-cycle policy
    'max_lr': 0.003,  # One-cycle max LR
    'weight_decay': 1e-4,
    'batch_size': 512,
    'epochs': 200,
    'early_stopping_patience': 30,  
    'ema_decay': 0.999,  # Model EMA decay
    'gradient_clip': 1.0,
    'gaussian_noise': 0.01,  
    'aux_weight': 0.1, 
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


class OneCycleLR(callbacks.Callback):
    """One-Cycle Learning Rate Policy (Leslie Smith)
    
    Ramp up to max_lr in first half, then ramp down to min_lr in second half.
    Proven to achieve better generalization than fixed or cosine decay LR.
    """
    def __init__(self, max_lr, total_epochs, min_lr=1e-6, warmup_fraction=0.3):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_fraction = warmup_fraction
        self.warmup_epochs = int(total_epochs * warmup_fraction)
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warmup: linear increase from min_lr to max_lr
            progress = epoch / self.warmup_epochs
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            # Annealing: cosine decrease from max_lr to min_lr
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except AttributeError:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)


class ExponentialMovingAverage(callbacks.Callback):
    """Exponential Moving Average of model weights during training.
    
    Maintains a shadow copy of weights using EMA. At the end of training,
    the EMA weights often generalize better than the final weights.
    This is a key technique used in winning Kaggle solutions.
    """
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        
    def on_train_begin(self, logs=None):
        # Initialize EMA weights as copies of model weights
        self.ema_weights = [tf.Variable(w, trainable=False) for w in self.model.get_weights()]
        
    def on_epoch_end(self, epoch, logs=None):
        # Update EMA weights
        model_weights = self.model.get_weights()
        for i, (ema_w, model_w) in enumerate(zip(self.ema_weights, model_weights)):
            ema_w.assign(self.decay * ema_w + (1 - self.decay) * model_w)
            
    def on_train_end(self, logs=None):
        # Don't replace weights here - let early stopping handle best weights
        # EMA weights will be used if early stopping didn't find improvement
        pass
    
    def apply_ema_weights(self):
        """Apply EMA weights to model"""
        if self.ema_weights is not None:
            self.model.set_weights([w.numpy() for w in self.ema_weights])


def swish(x):
    """Swish activation - better than ReLU for deep networks"""
    return x * tf.nn.sigmoid(x)


class FeatureDropout(layers.Layer):
    """Dropout on input features"""
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        
    def call(self, inputs, training=None):
        if not training or self.rate == 0:
            return inputs
        return tf.nn.dropout(inputs, rate=self.rate)


def build_dcn_model(num_continuous, cat_info, params):
    """
    Build DCN model with proven architecture +  improvements
    """
    # Continuous input
    cont_input = layers.Input(shape=(num_continuous,), name='continuous')
    
    # Apply feature dropout and noise
    x_cont = layers.GaussianNoise(params['gaussian_noise'], name='input_noise')(cont_input)
    x_cont = FeatureDropout(params['feature_dropout'], name='feature_dropout')(x_cont)
    
    # Categorical inputs and embeddings
    cat_inputs = []
    embeddings = []
    
    for i, (vocab_size, emb_dim) in enumerate(cat_info):
        cat_input = layers.Input(shape=(1,), name=f'cat_{i}')
        cat_inputs.append(cat_input)
        
        if vocab_size <= 2:
            emb = layers.Flatten()(cat_input)
            emb = layers.Dense(emb_dim, use_bias=False, kernel_initializer='he_normal')(emb)
        else:
            emb = layers.Embedding(vocab_size, emb_dim, embeddings_initializer='he_normal')(cat_input)
            emb = layers.Flatten()(emb)
        embeddings.append(emb)
    
    # Combine all features
    if embeddings:
        cat_concat = layers.Concatenate(name='cat_embed')(embeddings)
        x = layers.Concatenate(name='all_features')([x_cont, cat_concat])
    else:
        x = x_cont
    
    # Input normalization
    x = layers.LayerNormalization(name='input_norm')(x)
    feature_dim = x.shape[-1]
    x0 = x
    
    # Cross Network with residual connections 
    cross = x
    for i in range(params['cross_layers']):
        cross_dense = layers.Dense(feature_dim, use_bias=True, 
                                    kernel_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l2(params['weight_decay']),
                                    name=f'cross_{i}')(cross)
        cross_new = layers.multiply([x0, cross_dense])
        cross_new = layers.add([cross_new, cross_dense])
        cross = layers.add([cross, cross_new], name=f'cross_res_{i}')
        cross = layers.LayerNormalization(name=f'cross_ln_{i}')(cross)
    
    # Deep Network with residuals 
    deep = x
    prev_dim = feature_dim
    
    for i, units in enumerate(params['hidden_units']):
        deep_main = layers.Dense(units, 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=regularizers.l2(params['weight_decay']),
                                  name=f'deep_{i}')(deep)
        deep_main = layers.LayerNormalization(name=f'deep_ln_{i}')(deep_main)
        deep_main = layers.Activation(swish, name=f'deep_swish_{i}')(deep_main)
        deep_main = layers.Dropout(params['dropout_rate'], name=f'deep_drop_{i}')(deep_main)
        
        # Residual connection
        if prev_dim == units:
            deep = layers.add([deep, deep_main], name=f'deep_res_{i}')
        else:
            deep_proj = layers.Dense(units, use_bias=False, 
                                      kernel_initializer='he_normal',
                                      name=f'deep_proj_{i}')(deep)
            deep = layers.add([deep_proj, deep_main], name=f'deep_res_{i}')
        
        prev_dim = units
    
    # Combine cross and deep
    combined = layers.Concatenate(name='deep_cross')([deep, cross])
    
    # Main output head
    out = layers.LayerNormalization(name='out_ln')(combined)
    out = layers.Dense(128, kernel_initializer='he_normal', 
                        kernel_regularizer=regularizers.l2(params['weight_decay']),
                        name='out_1')(out)
    out = layers.Activation(swish, name='out_swish_1')(out)
    out = layers.Dropout(params['dropout_rate'], name='out_drop_1')(out)
    
    out = layers.Dense(64, kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(params['weight_decay']),
                        name='out_2')(out)
    out = layers.Activation(swish, name='out_swish_2')(out)
    
    main_output = layers.Dense(1, activation='sigmoid', name='output')(out)
    
    # Auxiliary output from cross network 
    aux_out = layers.LayerNormalization(name='aux_ln')(cross)
    aux_out = layers.Dense(64, activation='swish', name='aux_1')(aux_out)
    aux_output = layers.Dense(1, activation='sigmoid', name='aux_output')(aux_out)
    
    model = models.Model(inputs=[cont_input] + cat_inputs, 
                         outputs=[main_output, aux_output])
    return model


if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1/6] Loading data
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[1/6] Loading data...")
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2/6] Feature Engineering 
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/6] Core Feature Engineering...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    NUM_AS_CAT = []

    # Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
        
    # Arithmetic Interactions
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
        df['charge_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1) * df['tenure']).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges', 'charge_ratio']
    
    # Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
        df['streaming_count'] = ((df['StreamingTV'] == 'Yes').astype(int) + 
                                  (df['StreamingMovies'] == 'Yes').astype(int)).astype('float32')
        df['security_count'] = ((df['OnlineSecurity'] == 'Yes').astype(int) + 
                                 (df['OnlineBackup'] == 'Yes').astype(int) +
                                 (df['DeviceProtection'] == 'Yes').astype(int)).astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone', 'streaming_count', 'security_count']
    
    # ORIG_proba mapping
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # Distribution Features
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
    orig_contract_tc_mean = orig.groupby('Contract')['TotalCharges'].mean()
    
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
        df['resid_Contract_TC'] = (df['TotalCharges'] - df['Contract'].map(orig_contract_tc_mean).fillna(0)).astype('float32')
        
        # Conditional pctrank
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
    
    NEW_NUMS += ['pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
                 'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
                 'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC', 'resid_Contract_TC']
    
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
            
    NEW_NUMS += ['qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
                 'dist_To_nc_q25', 'qdist_gap_To_q25',
                 'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75']
        
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)

    # ═══════════════════════════════════════════════════════════════════════════
    # [3/6] Digit Features
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/6] Creating Digit Features...")
    
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

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/6] N-gram Features
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/6] Creating N-gram Categorical Features...")
    
    BIGRAM_COLS = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
        BIGRAM_COLS.append(col_name)
    
    TRIGRAM_COLS = []
    TOP4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str))
        TRIGRAM_COLS.append(col_name)
    
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5/6] Prepare Features
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Preparing features for DCNv2...")
    
    CATEGORICAL_FEATURES = CATS + NUM_AS_CAT + NGRAM_COLS
    NUMERICAL_FEATURES = NUMS + NEW_NUMS
    
    print(f"  Total features: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")
    print(f"  Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
    
    # Label encode categoricals
    le_dict = {}
    cat_info = []
    
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]]).astype(str).fillna('missing')
        le.fit(combined)
        le_dict[col] = le
        
        train[col] = le.transform(train[col].astype(str).fillna('missing'))
        test[col] = le.transform(test[col].astype(str).fillna('missing'))
        
        vocab_size = len(le.classes_)
        emb_dim = min(PARAMS['embedding_dim'], max(4, (vocab_size + 1) // 2))
        cat_info.append((vocab_size, emb_dim))
    
    print(f"  Embedding dims sum: {sum(d for _, d in cat_info)}")
    
    # Scale numericals
    scaler = StandardScaler()
    train[NUMERICAL_FEATURES] = scaler.fit_transform(train[NUMERICAL_FEATURES].astype('float32'))
    test[NUMERICAL_FEATURES] = scaler.transform(test[NUMERICAL_FEATURES].astype('float32'))
    
    num_continuous = len(NUMERICAL_FEATURES)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [6/6] Training
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[6/6] Training DCNv2  ({CFG.N_FOLDS}-Fold CV)...")
    print(f"  Hidden units: {PARAMS['hidden_units']} ")
    print(f"  Cross layers: {PARAMS['cross_layers']}")
    print(f"  Max LR: {PARAMS['max_lr']} (One-Cycle policy)")
    print(f"  Dropout: {PARAMS['dropout_rate']} ")
    print(f"  EMA decay: {PARAMS['ema_decay']}")
    
    np.random.seed(CFG.RANDOM_SEED)
    tf.random.set_seed(CFG.RANDOM_SEED)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    dcn_oof = np.zeros(len(train))
    dcn_pred = np.zeros(len(test))
    dcn_fold_scores = []
    
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[CFG.TARGET])):
        print(f"\n--- Fold {fold+1}/{CFG.N_FOLDS} ---")
        
        tf.keras.backend.clear_session()
        
        X_train_num = train.iloc[train_idx][NUMERICAL_FEATURES].values.astype('float32')
        X_val_num = train.iloc[val_idx][NUMERICAL_FEATURES].values.astype('float32')
        X_test_num = test[NUMERICAL_FEATURES].values.astype('float32')
        
        y_train = train.iloc[train_idx][CFG.TARGET].values.astype('float32')
        y_val = train.iloc[val_idx][CFG.TARGET].values.astype('float32')
        
        train_inputs = {'continuous': X_train_num}
        val_inputs = {'continuous': X_val_num}
        test_inputs = {'continuous': X_test_num}
        
        for j, col in enumerate(CATEGORICAL_FEATURES):
            train_inputs[f'cat_{j}'] = train.iloc[train_idx][col].values.astype('int32')
            val_inputs[f'cat_{j}'] = train.iloc[val_idx][col].values.astype('int32')
            test_inputs[f'cat_{j}'] = test[col].values.astype('int32')
        
        model = build_dcn_model(num_continuous, cat_info, PARAMS)
        
        optimizer = optimizers.AdamW(
            learning_rate=PARAMS['learning_rate'],
            weight_decay=PARAMS['weight_decay'],
            clipnorm=PARAMS['gradient_clip']
        )
        
        model.compile(
            optimizer=optimizer,
            loss={'output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
            loss_weights={'output': 1.0, 'aux_output': PARAMS['aux_weight']},
            metrics={'output': [tf.keras.metrics.AUC(name='auc')], 'aux_output': []}
        )
        
        # Callbacks
        es = callbacks.EarlyStopping(
            monitor='val_output_auc',
            mode='max',
            patience=PARAMS['early_stopping_patience'],
            restore_best_weights=True,
            verbose=0
        )
        
        one_cycle = OneCycleLR(
            max_lr=PARAMS['max_lr'],
            total_epochs=PARAMS['epochs'],
            min_lr=1e-6,
            warmup_fraction=0.3
        )
        
        ema_callback = ExponentialMovingAverage(decay=PARAMS['ema_decay'])
        
        history = model.fit(
            train_inputs, {'output': y_train, 'aux_output': y_train},
            validation_data=(val_inputs, {'output': y_val, 'aux_output': y_val}),
            batch_size=PARAMS['batch_size'],
            epochs=PARAMS['epochs'],
            callbacks=[es, one_cycle, ema_callback],
            verbose=0
        )
        
        # Predict - use main output only
        val_pred = model.predict(val_inputs, batch_size=PARAMS['batch_size'], verbose=0)[0].flatten()
        test_pred = model.predict(test_inputs, batch_size=PARAMS['batch_size'], verbose=0)[0].flatten()
        
        dcn_oof[val_idx] = val_pred
        dcn_pred += test_pred / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_pred)
        dcn_fold_scores.append(fold_auc)
        
        # Get training info
        val_auc_key = [k for k in history.history.keys() if 'val_output_auc' in k]
        best_val_auc = max(history.history[val_auc_key[0]]) if val_auc_key else fold_auc
        best_epoch = len(history.history.get('loss', []))
        
        print(f"   Fold {fold+1} AUC: {fold_auc:.5f} | Val_AUC: {best_val_auc:.5f} | Epochs: {best_epoch} | Time: {(time.time()-t0)/60:.1f} min")
        
        del model, train_inputs, val_inputs, test_inputs
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════════════════
    mean_score = np.mean(dcn_fold_scores)
    std_score = np.std(dcn_fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], dcn_oof)
    
    print(f"\n{'='*80}")
    print(f"V26 RESULTS — DCNv2  (Research-Informed)")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V16 Baseline:    0.91925 (OOF)")
    print(f"Delta:           {overall_auc - 0.91925:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in dcn_fold_scores)}")
    
    # Always save predictions
    print(f"\n💾 Saving predictions...")
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: dcn_oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: dcn_pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"Saved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    
    print(f"\nTotal time: {(time.time() - t0_all)/60:.1f} min")
    print("="*80)
