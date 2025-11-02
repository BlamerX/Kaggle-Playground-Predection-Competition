"""
Submission module for Road Accident Risk Prediction
Handles final predictions and submission file generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import SUBMISSION_FILE

def generate_predictions(model, X_test, test_ids, clip_predictions=True, min_val=0, max_val=1):
    """
    Generate predictions on test data
    
    Args:
        model: Trained model
        X_test (np.array): Test features
        test_ids (pd.Series): Test IDs
        clip_predictions (bool): Whether to clip predictions to valid range
        min_val (float): Minimum prediction value
        max_val (float): Maximum prediction value
        
    Returns:
        np.array: Predictions
    """
    print("🔮 Generating predictions using the best model...")
    
    try:
        # Make predictions
        predictions = model.predict(X_test)
        
        # Clip predictions if requested
        if clip_predictions:
            original_min = predictions.min()
            original_max = predictions.max()
            predictions = np.clip(predictions, min_val, max_val)
            
            print(f"✅ Predictions generated")
            print(f"   Original range: [{original_min:.6f}, {original_max:.6f}]")
            print(f"   Clipped range: [{predictions.min():.6f}, {predictions.max():.6f}]")
            print(f"   Clipped values: {np.sum((predictions != predictions.clip(min_val, max_val)))}")
        else:
            print(f"✅ Predictions generated without clipping")
            print(f"   Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None

def create_submission_file(predictions, test_ids, submission_file=SUBMISSION_FILE):
    """
    Create submission DataFrame and save to CSV
    
    Args:
        predictions (np.array): Model predictions
        test_ids (pd.Series): Test IDs
        submission_file (str): Output file path
        
    Returns:
        pd.DataFrame: Submission dataframe
    """
    print("📝 Creating submission file...")
    
    try:
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': test_ids,
            'accident_risk': predictions
        })
        
        # Sort by ID to ensure proper order
        submission = submission.sort_values('id').reset_index(drop=True)
        
        # Save to CSV
        submission.to_csv(submission_file, index=False)
        
        print(f"✅ Submission file '{submission_file}' saved successfully!")
        print(f"   Shape: {submission.shape}")
        print(f"   Columns: {list(submission.columns)}")
        
        return submission
        
    except Exception as e:
        print(f"❌ Error creating submission file: {e}")
        return None

def validate_submission(submission_df):
    """
    Validate submission file format and content
    
    Args:
        submission_df (pd.DataFrame): Submission dataframe
        
    Returns:
        dict: Validation results
    """
    print("\n🔍 Validating submission file...")
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Check if dataframe is empty
        if submission_df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Submission file is empty")
            return validation_results
        
        # Check required columns
        required_cols = ['id', 'accident_risk']
        missing_cols = [col for col in required_cols if col not in submission_df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'id' in submission_df.columns:
            if not pd.api.types.is_numeric_dtype(submission_df['id']):
                validation_results['errors'].append("ID column must be numeric")
                validation_results['is_valid'] = False
        
        if 'accident_risk' in submission_df.columns:
            if not pd.api.types.is_numeric_dtype(submission_df['accident_risk']):
                validation_results['errors'].append("accident_risk column must be numeric")
                validation_results['is_valid'] = False
        
        # Check for missing values
        if submission_df.isnull().any().any():
            validation_results['errors'].append("Found missing values in submission")
            validation_results['is_valid'] = False
        
        # Check prediction range
        if 'accident_risk' in submission_df.columns:
            pred_min = submission_df['accident_risk'].min()
            pred_max = submission_df['accident_risk'].max()
            
            validation_results['statistics']['prediction_range'] = (pred_min, pred_max)
            
            if pred_min < 0 or pred_max > 1:
                validation_results['warnings'].append(
                    f"Predictions outside [0,1] range: [{pred_min:.6f}, {pred_max:.6f}]"
                )
            
            # Check for extreme values
            if pred_min < -10 or pred_max > 10:
                validation_results['errors'].append(
                    f"Extreme prediction values detected: [{pred_min:.6f}, {pred_max:.6f}]"
                )
                validation_results['is_valid'] = False
        
        # Check for duplicate IDs
        if 'id' in submission_df.columns:
            if submission_df['id'].duplicated().any():
                validation_results['errors'].append("Found duplicate IDs")
                validation_results['is_valid'] = False
        
        # Summary statistics
        validation_results['statistics'].update({
            'num_rows': len(submission_df),
            'num_cols': len(submission_df.columns),
            'unique_ids': submission_df['id'].nunique() if 'id' in submission_df.columns else 0,
            'mean_prediction': submission_df['accident_risk'].mean() if 'accident_risk' in submission_df.columns else None,
            'std_prediction': submission_df['accident_risk'].std() if 'accident_risk' in submission_df.columns else None
        })
        
        # Print validation results
        if validation_results['is_valid']:
            print("✅ Submission file is valid!")
        else:
            print("❌ Submission file has errors!")
        
        if validation_results['errors']:
            print("\nErrors:")
            for error in validation_results['errors']:
                print(f"  • {error}")
        
        if validation_results['warnings']:
            print("\nWarnings:")
            for warning in validation_results['warnings']:
                print(f"  • {warning}")
        
        print(f"\nStatistics:")
        print(f"  Rows: {validation_results['statistics']['num_rows']}")
        print(f"  Columns: {validation_results['statistics']['num_cols']}")
        if validation_results['statistics']['mean_prediction'] is not None:
            print(f"  Mean prediction: {validation_results['statistics']['mean_prediction']:.6f}")
            print(f"  Std prediction: {validation_results['statistics']['std_prediction']:.6f}")
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Validation error: {str(e)}")
        print(f"❌ Error during validation: {e}")
    
    return validation_results

def visualize_predictions(predictions, save_plot=False):
    """
    Create visualization of prediction distribution
    
    Args:
        predictions (np.array): Model predictions
        save_plot (bool): Whether to save plot to file
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    print("📊 Creating prediction distribution visualization...")
    
    try:
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Predicted Accident Risk')
        plt.xlabel('Accident Risk')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(predictions, vert=True)
        plt.title('Box Plot of Predictions')
        plt.ylabel('Accident Risk')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_preds = np.sort(predictions)
        y_values = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
        plt.plot(sorted_preds, y_values, linewidth=2)
        plt.title('Cumulative Distribution')
        plt.xlabel('Accident Risk')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""Prediction Statistics:
        
Mean: {predictions.mean():.6f}
Median: {np.median(predictions):.6f}
Std: {predictions.std():.6f}
Min: {predictions.min():.6f}
Max: {predictions.max():.6f}
Q25: {np.percentile(predictions, 25):.6f}
Q75: {np.percentile(predictions, 75):.6f}

Zeros: {np.sum(predictions == 0)}
Ones: {np.sum(predictions == 1)}
Outside [0,1]: {np.sum((predictions < 0) | (predictions > 1))}
"""
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
            print("📊 Plot saved as 'prediction_distribution.png'")
        
        plt.show()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return None

def create_submission_pipeline(trained_model, test_data, test_ids, 
                              clip_predictions=True, validate=True, visualize=True):
    """
    Complete submission pipeline
    
    Args:
        trained_model: Trained model
        test_data (dict): Test data (from preprocessing)
        test_ids (pd.Series): Test IDs
        clip_predictions (bool): Whether to clip predictions
        validate (bool): Whether to validate submission
        visualize (bool): Whether to create visualizations
        
    Returns:
        dict: Pipeline results
    """
    print("🚀 Starting submission pipeline...")
    print("="*60)
    
    pipeline_results = {
        'predictions': None,
        'submission_df': None,
        'validation_results': None,
        'visualization': None
    }
    
    try:
        # Generate predictions
        predictions = generate_predictions(
            trained_model, 
            test_data['X_test'], 
            test_ids, 
            clip_predictions=clip_predictions
        )
        
        if predictions is None:
            print("❌ Failed to generate predictions")
            return pipeline_results
        
        pipeline_results['predictions'] = predictions
        
        # Create submission file
        submission_df = create_submission_file(predictions, test_ids)
        
        if submission_df is None:
            print("❌ Failed to create submission file")
            return pipeline_results
        
        pipeline_results['submission_df'] = submission_df
        
        # Validate submission
        if validate:
            validation_results = validate_submission(submission_df)
            pipeline_results['validation_results'] = validation_results
        
        # Create visualization
        if visualize:
            visualization = visualize_predictions(predictions, save_plot=False)
            pipeline_results['visualization'] = visualization
        
        print("\n✅ Submission pipeline completed successfully!")
        print(f"📊 Predictions summary:")
        print(f"   Count: {len(predictions)}")
        print(f"   Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"   Mean: {predictions.mean():.6f}")
        print(f"   Std: {predictions.std():.6f}")
        
    except Exception as e:
        print(f"❌ Error in submission pipeline: {e}")
    
    return pipeline_results