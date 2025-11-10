"""
Preprocessing module for Road Accident Risk Prediction
Handles feature encoding, scaling, and data preparation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from config import TEST_SIZE, SELECTED_SCALING_METHOD, RANDOM_STATE

def encode_features(df):
    """
    Encode categorical and boolean features
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Encoded dataframe
    """
    df_encoded = df.copy()
    
    # Boolean to integer
    for col in df_encoded.select_dtypes(include='bool').columns:
        df_encoded[col] = df_encoded[col].astype(int)
    
    # Categorical to integer using OrdinalEncoder
    categorical_cols = df_encoded.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder()
        df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])
    
    return df_encoded

def analyze_scaling_needs(features):
    """
    Analyze if scaling is needed based on feature ranges
    
    Args:
        features (pd.DataFrame): Feature dataframe
        
    Returns:
        dict: Scaling analysis results
    """
    print("\n" + "="*50)
    print("SCALING ANALYSIS")
    print("="*50)
    
    # Summary statistics
    print("Summary Statistics:")
    print(features.describe())
    
    # Feature ranges
    range_df = features.max() - features.min()
    print(f"\nFeature Ranges:")
    print(range_df.sort_values(ascending=False))
    
    # Check for scale differences
    high_range_features = range_df[range_df > range_df.mean()].index.tolist()
    print(f"\nFeatures with significantly higher ranges: {high_range_features}")
    
    # Decision on scaling
    if range_df.max() / range_df.min() > 10:
        scaling_needed = True
        print("\n✅ Feature scaling is likely necessary (large scale differences detected).")
    else:
        scaling_needed = False
        print("\n❌ Feature scaling might not be strictly necessary (features on similar scales).")
    
    # Correlation check
    corr_matrix = features.corr()
    
    return {
        'features_range': range_df,
        'high_range_features': high_range_features,
        'scaling_needed': scaling_needed,
        'correlation_matrix': corr_matrix,
        'scale_ratio': range_df.max() / range_df.min() if range_df.min() > 0 else float('inf')
    }

def get_scaler(method='standard'):
    """
    Get scaler based on method
    
    Args:
        method (str): Scaling method name
        
    Returns:
        scaler: Scikit-learn scaler object
    """
    if method == 'min_max':
        return MinMaxScaler()
    elif method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    elif method == 'power':
        return PowerTransformer(method='yeo-johnson')
    else:
        return None

def prepare_data_for_training(train_df, test_df, target_col='accident_risk', scaling_method='standard'):
    """
    Complete data preparation pipeline
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        target_col (str): Target column name
        scaling_method (str): Scaling method
        
    Returns:
        dict: Prepared data and scaler
    """
    print("🔧 Preparing data for training...")
    
    # Extract IDs
    train_ids = train_df['id'] if 'id' in train_df.columns else None
    test_ids = test_df['id'] if 'id' in test_df.columns else None
    
    # Separate features and target
    X_train = train_df.drop(columns=['id', target_col], errors='ignore')
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=['id'], errors='ignore')
    
    # Ensure numeric columns only
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    # Analyze scaling needs
    scaling_analysis = analyze_scaling_needs(X_train)
    
    # Encode features
    X_train_encoded = encode_features(X_train)
    X_test_encoded = encode_features(X_test)
    
    # Ensure all columns are numeric after encoding
    X_train_encoded = X_train_encoded.select_dtypes(include=[np.number])
    X_test_encoded = X_test_encoded.select_dtypes(include=[np.number])
    
    # Train-test split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_encoded, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Get and apply scaler
    scaler = get_scaler(scaling_method)
    
    if scaler is not None:
        print(f"Applying {scaling_method} scaling...")
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val_split)
        X_full_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
    else:
        X_train_scaled = X_train_split
        X_val_scaled = X_val_split
        X_full_scaled = X_train_encoded
        X_test_scaled = X_test_encoded
    
    return {
        'train_ids': train_ids,
        'test_ids': test_ids,
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'y_train': y_train_split,
        'y_val': y_val_split,
        'X_full': X_full_scaled,
        'y_full': y_train,
        'X_test': X_test_scaled,
        'scaler': scaler,
        'scaling_analysis': scaling_analysis,
        'feature_names': X_train_encoded.columns.tolist()
    }

def get_preprocessing_summary(prepared_data):
    """
    Get summary of preprocessing steps
    
    Args:
        prepared_data (dict): Output from prepare_data_for_training
        
    Returns:
        dict: Preprocessing summary
    """
    return {
        'train_shape': prepared_data['X_train'].shape,
        'validation_shape': prepared_data['X_val'].shape,
        'full_train_shape': prepared_data['X_full'].shape,
        'test_shape': prepared_data['X_test'].shape,
        'feature_names': prepared_data['feature_names'],
        'num_features': len(prepared_data['feature_names']),
        'scaling_method': prepared_data['scaler'].__class__.__name__ if prepared_data['scaler'] else 'None',
        'scaling_analysis': prepared_data['scaling_analysis']
    }