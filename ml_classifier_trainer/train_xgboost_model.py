# train_xgboost_model.py
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os

# Configuration
MODEL_DIR = "models"
MODEL_FILE = "social_support_xgboost_model.pkl"
DUMMY_DATA_SIZE = 5000

def generate_dummy_data():
    """Generate realistic dummy data for UAE social support"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'income': np.clip(np.random.normal(15000, 6000, DUMMY_DATA_SIZE), 2000, 50000),
        'loans': np.clip(np.random.normal(10000, 4000, DUMMY_DATA_SIZE), 0, 30000),
        'dependents': np.random.randint(0, 7, DUMMY_DATA_SIZE),
        'employment_status': np.random.choice([0, 1], DUMMY_DATA_SIZE, p=[0.2, 0.8]),
        'existing_benefits': np.random.choice([0, 1], DUMMY_DATA_SIZE, p=[0.7, 0.3])
    }
    
    # Generate target based on UAE-like rules
    conditions = (
        (data['income'] < 18000) & 
        (data['loans'] > 5000) & 
        (data['dependents'] >= 1) & 
        (data['employment_status'] == 1) & 
        (data['existing_benefits'] == 0)
    )
    data['eligible'] = np.where(conditions, 1, 0)
    return pd.DataFrame(data)

def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Generate and prepare data
    df = generate_dummy_data()
    X = df.drop('eligible', axis=1)
    y = df['eligible']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=True)
    
    # Evaluating model  
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nModel trained - Train accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")
    
    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILE))
    print(f"Model saved to {os.path.join(MODEL_DIR, MODEL_FILE)}")
    
    # Save feature list
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "model_features.pkl"))

if __name__ == "__main__":
    train_and_save_model()
