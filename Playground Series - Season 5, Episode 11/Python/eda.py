"""
Exploratory Data Analysis Module
Provides comprehensive data analysis and visualization capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader
from config import *
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ExploratoryDataAnalysis:
    """Class for performing exploratory data analysis"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.train_data = None
        self.test_data = None
        
    def load_data(self):
        """Load data for EDA"""
        if not self.data_loader.load_data():
            return False
        
        self.train_data = self.data_loader.train_data
        self.test_data = self.data_loader.test_data
        return True
    
    def basic_statistics(self):
        """Generate basic statistics for the dataset"""
        if self.train_data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        print(f"\nDataset Shape:")
        print(f"Training: {self.train_data.shape}")
        print(f"Test: {self.test_data.shape}")
        
        print(f"\nTarget Variable Distribution:")
        if FEATURE_ENGINEERING_CONFIG['target_column'] in self.train_data.columns:
            target_col = FEATURE_ENGINEERING_CONFIG['target_column']
            print(self.train_data[target_col].value_counts().sort_index())
            print(f"Target proportions:\n{self.train_data[target_col].value_counts(normalize=True).sort_index()}")
        
        print(f"\nMissing Values in Training Data:")
        missing_train = self.train_data.isnull().sum()
        missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
        if len(missing_train) > 0:
            print(missing_train)
        else:
            print("No missing values found!")
            
        print(f"\nMissing Values in Test Data:")
        missing_test = self.test_data.isnull().sum()
        missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
        if len(missing_test) > 0:
            print(missing_test)
        else:
            print("No missing values found!")
        
        print(f"\nData Types:")
        print(self.train_data.dtypes.value_counts())
        
        # Numerical columns summary
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\nNumerical Columns Summary:")
            print(self.train_data[numerical_cols].describe())
        
        # Categorical columns summary
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical Columns Summary:")
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                print(f"\n{col}:")
                print(self.train_data[col].value_counts().head())
    
    def correlation_analysis(self, save_plots=True):
        """Perform correlation analysis on numerical features"""
        if self.train_data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis.")
            return
        
        print("="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Calculate correlation matrix
        corr_matrix = self.train_data[numerical_cols].corr()
        
        # Print highly correlated pairs
        print(f"\nHighly Correlated Features (|correlation| > 0.8):")
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            for col1, col2, corr in high_corr_pairs:
                print(f"{col1} - {col2}: {corr:.3f}")
        else:
            print("No highly correlated feature pairs found.")
        
        # Correlation with target
        target_col = FEATURE_ENGINEERING_CONFIG['target_column']
        if target_col in numerical_cols:
            print(f"\nCorrelation with {target_col}:")
            target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
            print(target_corr.drop(target_col).head(10))
        
        # Create heatmap
        if save_plots:
            plt.figure(figsize=(12, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nCorrelation heatmap saved to {OUTPUT_PATH / 'correlation_heatmap.png'}")
    
    def target_analysis(self, save_plots=True):
        """Analyze target variable distribution and relationships"""
        if self.train_data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        target_col = FEATURE_ENGINEERING_CONFIG['target_column']
        if target_col not in self.train_data.columns:
            print(f"Target column '{target_col}' not found in data.")
            return
        
        print("="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        
        # Target distribution
        target_counts = self.train_data[target_col].value_counts().sort_index()
        print(f"\nTarget Variable Distribution:")
        print(target_counts)
        print(f"Target proportions:\n{target_counts / len(self.train_data)}")
        
        if save_plots:
            # Target distribution plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar plot
            target_counts.plot(kind='bar', ax=ax1)
            ax1.set_title('Target Distribution')
            ax1.set_xlabel('Classes')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=0)
            
            # Pie plot
            target_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
            ax2.set_title('Target Distribution (Percentage)')
            ax2.set_ylabel('')
            
            plt.tight_layout()
            plt.savefig(OUTPUT_PATH / 'target_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Target distribution plot saved to {OUTPUT_PATH / 'target_distribution.png'}")
    
    def feature_analysis(self, save_plots=True):
        """Analyze individual features"""
        if self.train_data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        target_col = FEATURE_ENGINEERING_CONFIG['target_column']
        
        # Analyze numerical features
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        if len(numerical_cols) > 0:
            print("="*50)
            print("NUMERICAL FEATURES ANALYSIS")
            print("="*50)
            
            # Distribution plots for numerical features
            if save_plots and len(numerical_cols) <= 10:  # Limit plots
                n_cols = min(3, len(numerical_cols))
                n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(numerical_cols):
                    if i < len(axes):
                        self.train_data[col].hist(bins=30, ax=axes[i], alpha=0.7)
                        axes[i].set_title(f'Distribution of {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frequency')
                
                # Hide unused subplots
                for i in range(len(numerical_cols), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(OUTPUT_PATH / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Numerical distributions plot saved to {OUTPUT_PATH / 'numerical_distributions.png'}")
        
        # Analyze categorical features
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            print(f"\n{'='*50}")
            print("CATEGORICAL FEATURES ANALYSIS")
            print("="*50)
            
            for col in categorical_cols[:5]:  # Analyze first 5 categorical features
                print(f"\n{col}:")
                value_counts = self.train_data[col].value_counts()
                print(f"Unique values: {len(value_counts)}")
                print("Top 10 values:")
                print(value_counts.head(10))
                
                # Show target distribution by this feature
                if target_col in self.train_data.columns and len(value_counts) <= 20:
                    target_by_feature = pd.crosstab(self.train_data[col], self.train_data[target_col], normalize='index')
                    print(f"Target distribution by {col}:")
                    print(target_by_feature.round(3))
    
    def missing_value_analysis(self, save_plots=True):
        """Analyze missing values in detail"""
        if self.train_data is None:
            print("Data not loaded. Call load_data() first.")
            return
        
        print("="*50)
        print("MISSING VALUE ANALYSIS")
        print("="*50)
        
        # Calculate missing value percentages
        train_missing = (self.train_data.isnull().sum() / len(self.train_data) * 100).round(2)
        test_missing = (self.test_data.isnull().sum() / len(self.test_data) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Train_Missing_%': train_missing,
            'Test_Missing_%': test_missing
        })
        
        missing_df = missing_df[missing_df['Train_Missing_%'] > 0].sort_values('Train_Missing_%', ascending=False)
        
        if len(missing_df) > 0:
            print("Missing value percentages:")
            print(missing_df)
            
            if save_plots:
                plt.figure(figsize=(12, 6))
                missing_df.plot(kind='bar', ax=plt.gca())
                plt.title('Missing Value Percentages by Feature')
                plt.xlabel('Features')
                plt.ylabel('Missing Percentage')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(OUTPUT_PATH / 'missing_values.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Missing values plot saved to {OUTPUT_PATH / 'missing_values.png'}")
        else:
            print("No missing values found in the dataset!")
    
    def run_full_eda(self, save_plots=True):
        """Run complete EDA analysis"""
        print("Starting Exploratory Data Analysis...")
        print("="*80)
        
        if not self.load_data():
            print("Failed to load data.")
            return
        
        # Run all analyses
        self.basic_statistics()
        self.correlation_analysis(save_plots)
        self.target_analysis(save_plots)
        self.feature_analysis(save_plots)
        self.missing_value_analysis(save_plots)
        
        print(f"\n{'='*80}")
        print("EDA completed! Check the 'output' folder for generated plots.")
        print(f"{'='*80}")

def run_eda():
    """Convenience function to run EDA"""
    eda = ExploratoryDataAnalysis()
    eda.run_full_eda()

if __name__ == "__main__":
    # Run EDA
    run_eda()