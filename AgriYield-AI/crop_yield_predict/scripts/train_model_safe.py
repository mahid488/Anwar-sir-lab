#!/usr/bin/env python
"""
Safe model training script with better error handling and disk space awareness.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from pathlib import Path

def train_and_save_model():
    try:
        # Load data
        data_path = Path(__file__).parent.parent / 'data' / 'Crop_recommendation.csv'
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        print(f"Data loaded: {len(data)} rows")
        
        # Encode labels
        le = LabelEncoder()
        data['label'] = le.fit_transform(data['label'])
        print(f"Encoded {len(le.classes_)} crop types")
        
        # Create synthetic yield column
        np.random.seed(42)
        data['Yield'] = (data['rainfall'] * 0.01) + (data['N'] * 0.02) + \
            (data['P'] * 0.015) + (data['K'] * 0.01) + np.random.normal(0, 0.5, len(data))
        print(f"Generated yield column")
        
        # Prepare features and target
        X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']]
        y = data['Yield']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training RandomForestRegressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        
        # Save with joblib (more reliable for large objects)
        model_dir = Path(__file__).parent.parent / 'model'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / 'crop_model.joblib'
        encoder_path = model_dir / 'label_encoder.joblib'
        
        print(f"Saving model to {model_path}")
        joblib.dump(model, model_path, compress=3)
        
        print(f"Saving encoder to {encoder_path}")
        joblib.dump(le, encoder_path)
        
        print("✓ Model and encoder saved successfully!")
        
        # Also save as pickle for compatibility (if space allows)
        try:
            import pickle
            model_pkl = model_dir / 'crop_model.pkl'
            encoder_pkl = model_dir / 'label_encoder.pkl'
            with open(model_pkl, 'wb') as f:
                pickle.dump(model, f)
            with open(encoder_pkl, 'wb') as f:
                pickle.dump(le, f)
            print("✓ Also saved as pickle format")
        except Exception as e:
            print(f"⚠ Could not save pickle format: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = train_and_save_model()
    sys.exit(0 if success else 1)
