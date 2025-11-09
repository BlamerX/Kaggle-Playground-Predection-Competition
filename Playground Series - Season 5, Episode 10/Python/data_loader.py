"""
Data loader module for Road Accident Risk Prediction
Handles loading and initial exploration of datasets
"""

import pandas as pd
from config import TRAIN_PATH, TEST_PATH, SUBMISSION_PATH

def load_datasets():
    """
    Load train, test, and submission datasets
    
    Returns:
        tuple: (train_df, test_df, submission_df)
    """
    try:
        print("Loading datasets...")
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        submission_df = pd.read_csv(SUBMISSION_PATH)
        
        print(f"✅ Train shape: {train_df.shape}")
        print(f"✅ Test shape: {test_df.shape}")
        print(f"✅ Submission shape: {submission_df.shape}")
        
        return train_df, test_df, submission_df
    
    except FileNotFoundError as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None, None

def explore_data_basic(df):
    """
    Basic data exploration
    
    Args:
        df (pd.DataFrame): DataFrame to explore
        
    Returns:
        dict: Basic statistics
    """
    if df is None:
        return {}
    
    print("\n" + "="*50)
    print("BASIC DATA EXPLORATION")
    print("="*50)
    
    # Shape info
    print(f"Dataset shape: {df.shape}")
    
    # Column info
    print(f"Columns: {list(df.columns)}")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Missing values
    print("\nMissing Values:")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found!")
    
    # Unique values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nUnique values in categorical columns:")
        for col in categorical_cols:
            print(f"{col}: {df[col].unique()}")
    
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': missing_values.to_dict(),
        'categorical_cols': list(categorical_cols)
    }

def get_data_summary(train_df, test_df):
    """
    Get comprehensive data summary for both train and test sets
    
    Args:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        
    Returns:
        dict: Summary information
    """
    if train_df is None or test_df is None:
        return {}
    
    summary = {
        'train': explore_data_basic(train_df),
        'test': explore_data_basic(test_df)
    }
    
    # Compare feature sets
    train_cols = set(train_df.columns) - {'id', 'accident_risk'}
    test_cols = set(test_df.columns) - {'id'}
    
    print("\n" + "="*50)
    print("FEATURE SET COMPARISON")
    print("="*50)
    print(f"Train features (excluding id, target): {len(train_cols)}")
    print(f"Test features (excluding id): {len(test_cols)}")
    print(f"Common features: {len(train_cols & test_cols)}")
    
    if train_cols != test_cols:
        print(f"Features only in train: {train_cols - test_cols}")
        print(f"Features only in test: {test_cols - train_cols}")
    
    return summary