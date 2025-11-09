"""
Simple test version of the road accident prediction pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data():
    """Load the datasets"""
    print("Loading datasets...")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """Preprocess the data"""
    print("Preprocessing data...")
    
    # Get target and features
    X_train = train_df.drop(['id', 'accident_risk'], axis=1)
    y_train = train_df['accident_risk']
    X_test = test_df.drop(['id'], axis=1)
    test_ids = test_df['id']
    
    # Encode categorical variables
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        # Fit on combined data to ensure consistent encoding
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    
    # Ensure numeric columns only
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, test_ids

def train_model(X_train, y_train):
    """Train the model"""
    print("Training model...")
    
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train Random Forest (simple and robust)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)
    
    # Validate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {mse:.6f}")
    
    # Retrain on full data
    model.fit(X_train, y_train)
    
    return model

def generate_submission(model, X_test, test_ids):
    """Generate submission file"""
    print("Generating predictions...")
    
    predictions = model.predict(X_test)
    
    # Clip predictions to [0,1] range
    predictions = np.clip(predictions, 0, 1)
    
    submission = pd.DataFrame({
        'id': test_ids,
        'accident_risk': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    print(f"Predictions summary:")
    print(f"  Count: {len(predictions)}")
    print(f"  Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"  Mean: {predictions.mean():.6f}")
    
    return submission

def main():
    """Main function"""
    print("=== ROAD ACCIDENT RISK PREDICTION ===")
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Preprocess
        X_train, y_train, X_test, test_ids = preprocess_data(train_df, test_df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Generate submission
        submission = generate_submission(model, X_test, test_ids)
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()