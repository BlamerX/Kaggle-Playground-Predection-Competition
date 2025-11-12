"""
Exploratory Data Analysis (EDA) module for Road Accident Risk Prediction
Handles visualization and statistical analysis of the dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def univariate_analysis(df, target_col='accident_risk'):
    """
    Perform univariate analysis on numerical features
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        
    Returns:
        dict: Analysis results
    """
    print("\n" + "="*50)
    print("UNIVARIATE ANALYSIS")
    print("="*50)
    
    # Get numerical columns (excluding id and target)
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    print(f"Found {len(numerical_cols)} numerical features for analysis")
    
    # Basic statistics for each numerical column
    analysis_results = {}
    
    for col in numerical_cols:
        print(f"\nAnalyzing {col}...")
        
        # Basic statistics
        col_stats = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        
        analysis_results[col] = col_stats
        
        # Print summary
        print(f"  Mean: {col_stats['mean']:.4f}")
        print(f"  Std: {col_stats['std']:.4f}")
        print(f"  Range: [{col_stats['min']:.4f}, {col_stats['max']:.4f}]")
        print(f"  Skewness: {col_stats['skewness']:.4f}")
        
        # Create interactive plots
        try:
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=(f"Distribution of {col}", f"Box Plot of {col}"))
            
            # Histogram
            fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=30), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(y=df[col], name=col), row=1, col=2)
            
            fig.update_layout(title_text=f'Univariate Analysis of {col}', showlegend=False)
            
            # Save plot (optional - uncomment to save)
            # fig.write_html(f"univariate_{col}.html")
            fig.show()
            
        except Exception as e:
            print(f"  Warning: Could not create interactive plot for {col}: {e}")
            # Create matplotlib fallback
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(df[col], bins=30, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.boxplot(df[col])
            plt.title(f'Box Plot of {col}')
            plt.ylabel(col)
            
            plt.tight_layout()
            plt.show()
    
    return analysis_results

def correlation_analysis(df, target_col='accident_risk'):
    """
    Perform correlation analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Select only numeric columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_df = df[numerical_cols]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.7:  # Threshold for high correlation
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_matrix.iloc[i, j]
                ))
    
    print(f"\nHighly correlated feature pairs (|correlation| > 0.7):")
    if high_corr_pairs:
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("  No highly correlated feature pairs found.")
    
    # Show correlations with target
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        print(f"\nTop correlations with {target_col}:")
        for feature, corr in target_corr.items():
            if feature != target_col:
                print(f"  {feature}: {corr:.3f}")
    
    # Create interactive heatmap
    try:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Heatmap of Numerical Features',
            width=800, height=800
        )
        
        # Save plot (optional - uncomment to save)
        # fig.write_html("correlation_heatmap.html")
        fig.show()
        
    except Exception as e:
        print(f"Warning: Could not create interactive heatmap: {e}")
        # Create matplotlib fallback
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    return corr_matrix

def target_distribution_analysis(df, target_col='accident_risk'):
    """
    Analyze target variable distribution
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        
    Returns:
        dict: Target analysis results
    """
    print("\n" + "="*50)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("="*50)
    
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataframe")
        return {}
    
    target = df[target_col]
    
    # Basic statistics
    target_stats = {
        'mean': target.mean(),
        'std': target.std(),
        'min': target.min(),
        'max': target.max(),
        'median': target.median(),
        'q25': target.quantile(0.25),
        'q75': target.quantile(0.75),
        'skewness': target.skew(),
        'kurtosis': target.kurtosis(),
        'unique_values': target.nunique(),
        'zero_count': (target == 0).sum(),
        'zero_percentage': (target == 0).mean() * 100
    }
    
    print(f"Target Statistics:")
    print(f"  Mean: {target_stats['mean']:.4f}")
    print(f"  Std: {target_stats['std']:.4f}")
    print(f"  Range: [{target_stats['min']:.4f}, {target_stats['max']:.4f}]")
    print(f"  Skewness: {target_stats['skewness']:.4f}")
    print(f"  Kurtosis: {target_stats['kurtosis']:.4f}")
    print(f"  Zero values: {target_stats['zero_count']} ({target_stats['zero_percentage']:.2f}%)")
    
    # Create visualization
    try:
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Target Distribution', 'Box Plot', 
                                         'Q-Q Plot', 'Cumulative Distribution'))
        
        # Histogram
        fig.add_trace(go.Histogram(x=target, name='Target', nbinsx=50), row=1, col=1)
        
        # Box plot
        fig.add_trace(go.Box(y=target, name='Target'), row=1, col=2)
        
        # Q-Q plot (simple version)
        sorted_target = np.sort(target)
        theoretical_quantiles = np.linspace(0, 1, len(sorted_target))
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_target, 
                               mode='markers', name='Q-Q'), row=2, col=1)
        
        # CDF
        y_values = np.arange(1, len(sorted_target) + 1) / len(sorted_target)
        fig.add_trace(go.Scatter(x=sorted_target, y=y_values, mode='lines', 
                               name='CDF'), row=2, col=2)
        
        fig.update_layout(title_text=f'Distribution Analysis of {target_col}', 
                         showlegend=False, height=600)
        
        # Save plot (optional - uncomment to save)
        # fig.write_html("target_distribution.html")
        fig.show()
        
    except Exception as e:
        print(f"Warning: Could not create interactive target plots: {e}")
        # Create matplotlib fallback
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(target, bins=50, alpha=0.7)
        plt.title('Target Distribution')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.boxplot(target)
        plt.title('Target Box Plot')
        plt.ylabel(target_col)
        
        plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(target, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.subplot(2, 2, 4)
        plt.hist(target, bins=50, cumulative=True, density=True, alpha=0.7)
        plt.title('Cumulative Distribution')
        plt.xlabel(target_col)
        plt.ylabel('Cumulative Probability')
        
        plt.tight_layout()
        plt.show()
    
    return target_stats

def perform_eda(df, target_col='accident_risk', save_plots=False):
    """
    Perform complete EDA analysis
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        save_plots (bool): Whether to save plots to files
        
    Returns:
        dict: Complete EDA results
    """
    print("🔍 Starting Exploratory Data Analysis...")
    
    eda_results = {}
    
    # 1. Target distribution analysis
    print("\n1. Analyzing target variable distribution...")
    eda_results['target_analysis'] = target_distribution_analysis(df, target_col)
    
    # 2. Univariate analysis
    print("\n2. Performing univariate analysis...")
    eda_results['univariate_analysis'] = univariate_analysis(df, target_col)
    
    # 3. Correlation analysis
    print("\n3. Performing correlation analysis...")
    eda_results['correlation_matrix'] = correlation_analysis(df, target_col)
    
    print("\n✅ EDA completed successfully!")
    
    return eda_results