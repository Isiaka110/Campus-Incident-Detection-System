# model/train.py - Final Robust Version
import pandas as pd
import joblib 
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- PATH FIX START (Ensures imports work from any sub-folder) ---
# Define the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project root to the system path so Python can find 'utils'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- PATH FIX END ---

# Now the import should work
from utils.preprocessing import load_data, feature_engineering 

# Define the base directories for file operations
MODEL_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data' 

def train_model(df_path):
    """
    Loads data, engineers features, trains the RandomForest model, 
    and saves the model and encoder (FR4).
    """
    print("Starting data load and preprocessing...")
    
    df = load_data(str(df_path)) 
    df_processed, feature_cols, target_col, le_incident = feature_engineering(df)
    
    # Check if there's enough data and features
    if len(df_processed) < 2 or not feature_cols or len(df_processed[target_col].unique()) < 2:
        print("Error: Not enough unique data or classes to train the model.")
        return None, None
        
    X = df_processed[feature_cols]
    y = df_processed[target_col]

    # Split data (80% train, 20% test)
    # FIX: Removed 'stratify=y' to prevent the small sample size ValueError.
    # 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training RandomForest model with {len(X_train)} samples...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nModel Evaluation (Test Set):\n")
    
    # --- CRITICAL FIX for ValueError: Number of classes mismatch ---
    # We must explicitly pass ALL possible labels to classification_report 
    # using the encoder's knowledge to handle classes missing from the small test set.
    all_known_labels = le_incident.transform(le_incident.classes_)
    
    print(classification_report(
        y_test, 
        y_pred, 
        labels=all_known_labels,  # Use all known integer labels
        target_names=le_incident.classes_, # Use all known class names
        zero_division=0 # Handle zero predictions gracefully
    ))
    # --- END CRITICAL FIX ---

    # Define save paths using MODEL_DIR 
    model_save_path = MODEL_DIR / 'random_forest_model.joblib'
    encoder_save_path = MODEL_DIR / 'incident_type_encoder.joblib'
    
    # Save the model and encoder
    joblib.dump(model, model_save_path)
    joblib.dump(le_incident, encoder_save_path)
    print(f"\nModel and encoder successfully saved to: {MODEL_DIR}")
    
    return model, le_incident
    
if __name__ == '__main__':
    # Define the default data path relative to the script
    default_data_path = DATA_DIR / 'aau_security_logs.csv'
    
    if not default_data_path.exists():
        print(f"Error: Data file not found at {default_data_path}. Please ensure 'aau_security_logs.csv' is in the 'data/' directory.")
    else:
        # Pass the Path object to the training function
        train_model(default_data_path)