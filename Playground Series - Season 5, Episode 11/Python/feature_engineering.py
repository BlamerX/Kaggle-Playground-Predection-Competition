"""
Feature Engineering Module
Handles feature transformation, creation, and selection for loan prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from data_loader import DataLoader
from config import *
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Class for feature engineering and transformation"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.selected_features = []
        self.importance_scores = {}
        
    def load_data(self):
        """Load data for feature engineering"""
        if not self.data_loader.load_data():
            return False, None, None, None
        
        X_train, X_test, y_train = self.data_loader.get_train_test_split()
        return True, X_train, X_test, y_train
    
    def handle_missing_values(self, X_train, X_test):
        """Advanced missing value handling"""
        print("Handling missing values...")
        
        # For numerical features: use median for train, same median for test
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if X_train[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)
        
        # For categorical features: use mode for train, mode from train for test
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if X_train[col].isnull().sum() > 0 or X_test[col].isnull().sum() > 0:
                mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
                X_train[col].fillna(mode_val, inplace=True)
                X_test[col].fillna(mode_val, inplace=True)
        
        return X_train, X_test
    
    def encode_categorical_features(self, X_train, X_test):
        """Encode categorical features"""
        print("Encoding categorical features...")
        
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        # Apply label encoding to categorical features
        for col in categorical_cols:
            # Fit on training data only
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            
            # Transform test data, handling unseen categories
            test_values = X_test[col].astype(str)
            # Map unseen categories to a default value
            test_mapped = []
            for val in test_values:
                if val in le.classes_:
                    test_mapped.append(le.transform([val])[0])
                else:
                    # Assign to the most frequent class in training
                    test_mapped.append(0)
            X_test[col] = test_mapped
            
            self.label_encoders[col] = le
        
        return X_train, X_test
    
    def create_new_features(self, X_train, X_test):
        """Create new features from existing ones"""
        print("Creating new features...")
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        # Create ratio features for numerical columns (if we have at least 2 numerical features)
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:5]):  # Limit to first 5 to avoid explosion
                for col2 in numerical_cols[i+1:6]:
                    if col1 != col2:
                        # Avoid division by zero
                        ratio_name = f"{col1}_div_{col2}"
                        X_train[ratio_name] = np.where(X_train[col2] != 0, 
                                                     X_train[col1] / X_train[col2], 0)
                        X_test[ratio_name] = np.where(X_test[col2] != 0, 
                                                    X_test[col1] / X_test[col2], 0)
                        
                        # Log ratio (add small constant to avoid log(0))
                        log_ratio_name = f"log_{col1}_div_{col2}"
                        X_train[log_ratio_name] = np.log1p(np.where(X_train[col2] != 0, 
                                                                  X_train[col1] / X_train[col2], 0))
                        X_test[log_ratio_name] = np.log1p(np.where(X_test[col2] != 0, 
                                                                 X_test[col1] / X_test[col2], 0))
        
        # Create polynomial features for important numerical features
        if len(numerical_cols) > 0:
            for col in numerical_cols[:3]:  # Limit to first 3 features
                X_train[f"{col}_squared"] = X_train[col] ** 2
                X_test[f"{col}_squared"] = X_test[col] ** 2
                
                X_train[f"{col}_sqrt"] = np.sqrt(np.abs(X_train[col]))
                X_test[f"{col}_sqrt"] = np.sqrt(np.abs(X_test[col]))
        
        # Create interaction features (if we have enough features)
        if len(numerical_cols) >= 2:
            X_train["sum_features"] = X_train[numerical_cols[:3]].sum(axis=1)
            X_test["sum_features"] = X_test[numerical_cols[:3]].sum(axis=1)
            
            X_train["mean_features"] = X_train[numerical_cols[:3]].mean(axis=1)
            X_test["mean_features"] = X_test[numerical_cols[:3]].mean(axis=1)
            
            X_train["std_features"] = X_train[numerical_cols[:3]].std(axis=1)
            X_test["std_features"] = X_test[numerical_cols[:3]].std(axis=1)
        
        return X_train, X_test
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        print("Scaling features...")
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit scaler on training data only
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        return X_train_scaled, X_test_scaled
    
    def feature_selection(self, X_train, y_train, k=20):
        """Perform feature selection"""
        print(f"Performing feature selection (selecting top {k} features)...")
        
        # Get feature importance using Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        
        # Get feature importance scores
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 20 most important features:")
        print(importance_df.head(20))
        
        # Select top k features
        self.selected_features = importance_df.head(k)['feature'].tolist()
        
        # Filter datasets to selected features
        X_train_selected = X_train[self.selected_features]
        
        return X_train_selected, importance_df
    
    def reduce_dimensionality(self, X_train, X_test, n_components=10):
        """Reduce dimensionality using PCA"""
        print(f"Reducing dimensionality to {n_components} components...")
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Create feature names for PCA components
        pca_feature_names = [f"PC{i+1}" for i in range(n_components)]
        
        # Convert back to DataFrames
        X_train_pca = pd.DataFrame(X_train_pca, columns=pca_feature_names, index=X_train.index)
        X_test_pca = pd.DataFrame(X_test_pca, columns=pca_feature_names, index=X_test.index)
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_train_pca, X_test_pca, pca
    
    def full_feature_engineering_pipeline(self, target_k=None, use_pca=False):
        """Run the complete feature engineering pipeline"""
        print("="*50)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        # Load data
        success, X_train, X_test, y_train = self.load_data()
        if not success:
            print("Failed to load data.")
            return None, None, None, None
        
        print(f"Initial shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 1: Handle missing values
        X_train, X_test = self.handle_missing_values(X_train, X_test)
        
        # Step 2: Encode categorical features
        X_train, X_test = self.encode_categorical_features(X_train, X_test)
        
        print(f"After encoding - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 3: Create new features
        X_train, X_test = self.create_new_features(X_train, X_test)
        
        print(f"After feature creation - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 4: Feature scaling
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 5: Feature selection or PCA
        if use_pca and target_k:
            X_train_final, X_test_final, pca = self.reduce_dimensionality(
                X_train_scaled, X_test_scaled, target_k)
        elif target_k:
            X_train_selected, importance_df = self.feature_selection(X_train_scaled, y_train, target_k)
            # Apply same selection to test data
            X_test_selected = X_test_scaled[self.selected_features]
            X_train_final, X_test_final = X_train_selected, X_test_selected
        else:
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
        
        print(f"Final shapes - Train: {X_train_final.shape}, Test: {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train, self

def run_feature_engineering(target_features=20, use_pca=False):
    """Convenience function to run feature engineering"""
    fe = FeatureEngineer()
    X_train, X_test, y_train, feature_engineer = fe.full_feature_engineering_pipeline(
        target_k=target_features, use_pca=use_pca)
    return X_train, X_test, y_train, feature_engineer

if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering...")
    X_train, X_test, y_train, fe = run_feature_engineering()
    
    if X_train is not None:
        print(f"\nFinal training shape: {X_train.shape}")
        print(f"Final test shape: {X_test.shape}")
        print(f"Target shape: {y_train.shape}")
        print(f"\nFeature columns: {list(X_train.columns)}")
    else:
        print("Feature engineering failed.")