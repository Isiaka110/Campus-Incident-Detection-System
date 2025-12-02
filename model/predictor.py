# model/predictor.py - Final Fixed Version
import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path

from utils.preprocessing import feature_engineering, inverse_transform_incident_type

# Setup paths for loading the model and encoder
MODEL_PATH = Path('model/random_forest_model.joblib')
ENCODER_PATH = Path('model/incident_type_encoder.joblib')

# Load the trained model and encoder (assuming they've been trained and saved)
try:
    MODEL = joblib.load(MODEL_PATH)
    ENCODER = joblib.load(ENCODER_PATH)
    IS_MODEL_LOADED = True
except FileNotFoundError:
    print(f"Warning: Model or Encoder not found at {MODEL_PATH}. Please run model/train.py first.")
    MODEL = None
    ENCODER = None
    IS_MODEL_LOADED = False


def create_prediction_df(original_df, le_incident):
    """
    Creates a prediction DataFrame by sampling unique combinations from the historical data
    and applying feature engineering to the sampled scenarios.
    
    The key is to select raw features from the original_df *before* engineering.
    """
    
    # 1. First, apply feature engineering to the *entire* raw data just to extract the 
    #    list of training features (training_feature_cols) for later use.
    _, training_feature_cols, _, _ = feature_engineering(original_df.copy())
    
    # 2. Identify the scenario-defining features (these must be present in original_df)
    # Note: We must include the features used to derive Hour, Day_of_Week, etc.
    pred_raw_cols_for_sampling = ['Location', 'Electricity Status', 'Semester Period', 'Academic Context', 'Hour'] 
    
    # CRITICAL FIX: To get 'Hour' into the raw data for sampling, we must perform the 
    # simplest feature creation step first (Date/Time derived features). 
    # The simplest way is to manually run the necessary initial steps from utils/preprocessing.py load_data/feature_engineering.
    
    # Manually derive 'Hour' in the original_df temporarily for sampling purposes:
    # Ensure Date/Time are in a processable state (this mimics the cleaning in load_data)
    df_temp = original_df.copy()
    
    # Copy of load_data initial cleaning/conversion for sampling:
    df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
    df_temp['Time'] = pd.to_datetime(df_temp['Time'].astype(str), errors='coerce', infer_datetime_format=True).dt.time
    df_temp.dropna(subset=['Date', 'Time'], inplace=True) 

    # Copy of feature_engineering initial derivation for sampling:
    df_temp['Hour'] = df_temp['Time'].apply(lambda x: x.hour if x is not np.nan else 0) 
    
    # 3. Select unique combinations from this temporarily processed (but not OHE'd) data
    # The sampling features are now guaranteed to exist in df_temp
    unique_combinations = df_temp[pred_raw_cols_for_sampling].drop_duplicates().sample(
        n=min(50, len(df_temp)), 
        replace=True, 
        random_state=42
    )

    # 4. Create prediction base
    prediction_df = unique_combinations.copy()
    
    # 5. Re-introduce the mandatory columns needed by the full feature_engineering()
    prediction_df['Date'] = pd.to_datetime('2025-01-01') # Dummy Future date
    prediction_df['Time'] = prediction_df['Hour'].apply(lambda x: pd.to_datetime(f"{int(x)}:00", format='%H:%M').time())
    
    # We must ensure all columns used in feature_engineering exist, even if as placeholders
    # 'Incident Type' and 'Severity' are needed by feature_engineering for encoding, 
    # even though they are not used for prediction scenario definition. Use the mode of the original data.
    df_temp_modes = original_df[['Incident Type', 'Severity']].mode().iloc[0]
    prediction_df['Incident Type'] = df_temp_modes['Incident Type']
    prediction_df['Severity'] = df_temp_modes['Severity']
    prediction_df['Description'] = 'Prediction scenario' # Placeholder
    
    # 6. Re-process the prediction set to get the same one-hot features as training
    df_processed, feature_cols, _, _ = feature_engineering(prediction_df)
    
    # 7. Ensure prediction features match training features (crucial step for the ML model)
    X_pred = df_processed.reindex(columns=training_feature_cols, fill_value=0)
    
    return prediction_df, X_pred, feature_cols


def predict_risks(df_logs):
    """
    Performs predictions using the loaded RandomForest model (FR5, FR6).
    """
    if not IS_MODEL_LOADED:
        return pd.DataFrame(), {}
    
    # CRITICAL FIX: Check if input data is empty
    if df_logs.empty or len(df_logs) < 2:
        empty_cols = ['Location', 'Hour', 'Predicted_Incident', 'Risk_Score', 'Risk_Level']
        print("Warning: Insufficient or empty data for prediction. Displaying empty dashboard.")
        return pd.DataFrame(columns=empty_cols), {}

    # 1. Create a future/prediction set
    df_prediction_base, X_pred, _ = create_prediction_df(df_logs, ENCODER)

    # 2. Predict the encoded incident type
    y_pred_encoded = MODEL.predict(X_pred)
    df_prediction_base['Predicted_Incident_Encoded'] = y_pred_encoded
    
    # 3. Predict probability (Risk Score 0-1)
    y_pred_proba = MODEL.predict_proba(X_pred)
    # The risk score is the probability of the *predicted* class
    predicted_proba = np.choose(y_pred_encoded, y_pred_proba.T) 
    df_prediction_base['Risk_Score'] = predicted_proba # FR5
    
    # 4. Inverse transform to get incident name
    df_prediction_base['Predicted_Incident'] = inverse_transform_incident_type(ENCODER, y_pred_encoded)

    # 5. Classify Risk Level (FR6)
    def classify_risk(score):
        if score >= 0.75: return 'High'
        elif score >= 0.4: return 'Medium'
        else: return 'Low'
        
    df_prediction_base['Risk_Level'] = df_prediction_base['Risk_Score'].apply(classify_risk)

    # 6. Summarize high-risk information (R3, R4)
    high_risk_df = df_prediction_base[df_prediction_base['Risk_Level'] == 'High']
    
    high_risk_info = {
        "Top High-Risk Location": high_risk_df['Location'].mode().tolist()[0] if not high_risk_df.empty else 'N/A',
        # Ensure we handle the 'Hour' conversion gracefully, it should be an int from the temporary data 
        "Top High-Risk Time": high_risk_df['Hour'].mode().tolist()[0] if not high_risk_df.empty else 'N/A',
        "Most Probable Incident Type": high_risk_df['Predicted_Incident'].mode().tolist()[0] if not high_risk_df.empty else 'N/A'
    }

    # 7. Clean up and select final columns for display (R3)
    final_cols = ['Location', 'Hour', 'Predicted_Incident', 'Risk_Score', 'Risk_Level']
    df_predictions_final = df_prediction_base[final_cols].sort_values('Risk_Score', ascending=False)
    
    # 8. Rename Hour for clarity (FR9)
    # The Hour column comes from prediction_df and is an integer.
    df_predictions_final['Hour'] = df_predictions_final['Hour'].astype(int).astype(str) + ':00 - ' + (df_predictions_final['Hour'].astype(int) + 1).astype(str) + ':00'

    return df_predictions_final, high_risk_info