# utils/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np 

def load_data(file_path):
    """
    Loads CSV data, cleans column names, validates required fields, 
    and handles robust type conversion for Date and Time.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Data file not found at {file_path}")

    # --- FIX: Standardize column names by stripping whitespace ---
    df.columns = df.columns.str.strip()
    
    # Required columns based on PRD Section 7.0
    required_cols = ['Date', 'Time', 'Location', 'Incident Type', 'Severity', 'Description', 
                     'Electricity Status', 'Semester Period', 'Academic Context']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Raise an error if essential columns are missing
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_cols)}. "
                         f"Please ensure your header matches the PRD.")

    # --- Data Type Conversion and Cleaning ---
    
    # 1. Date Conversion (Highly resilient)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') 

    # 2. Time Conversion (CRITICAL FIX: More flexible parsing)
    # Convert 'Time' column to string first to handle NaNs, then use infer_datetime_format=True
    # to let Pandas intelligently guess the time format (HH:MM or H:MM etc.) before extracting time.
    df['Time'] = pd.to_datetime(df['Time'].astype(str), errors='coerce', infer_datetime_format=True).dt.time

    # Drop rows where essential date/time conversion failed (critical data loss)
    df.dropna(subset=['Date', 'Time'], inplace=True) 
    
    if df.empty:
        raise ValueError("Data frame is empty after cleaning dates/times. Double-check all rows for valid date/time entries.")

    return df


def feature_engineering(df):
    """
    Creates derived features from Date and Time columns (Derived Features).
    Encodes categorical features for ML model.
    """
    df = df.copy()

    # Derived Features (Time-based)
    # Note: df['Time'] is a datetime.time object, need to ensure we can get .hour
    df['Hour'] = df['Time'].apply(lambda x: x.hour if x is not np.nan else 0) 
    df['Day_of_Week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['Month'] = df['Date'].dt.month
    df['Day_of_Year'] = df['Date'].dt.dayofyear

    # Derived Feature (Day Period)
    def get_day_period(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 21: return 'Evening'
        else: return 'Night'
    df['Day_Period'] = df['Hour'].apply(get_day_period)

    # Derived Feature (Incident Frequency per Location)
    # This is an important feature for predicting high-risk locations
    location_freq = df['Location'].value_counts().to_dict()
    df['Location_Freq'] = df['Location'].map(location_freq)

    # Categorical Feature Encoding (FR2)
    # Target Encoding: Incident Type (what we predict)
    le_incident = LabelEncoder()
    df['Incident Type'] = df['Incident Type'].astype(str) # Ensure it's a string before encoding
    df['Incident_Type_Encoded'] = le_incident.fit_transform(df['Incident Type'])
    
    # Feature Encoding: Other Categorical Columns (One-Hot Encoding)
    # This converts text categories into numerical features needed by RandomForest
    df = pd.get_dummies(df, columns=['Location', 'Severity', 'Electricity Status', 
                                     'Semester Period', 'Academic Context', 'Day_Period'], 
                        drop_first=True) # drop_first=True helps avoid multicollinearity

    # Define features and target (Select only the numerical columns for the ML model)
    feature_cols = [col for col in df.columns if 'Location_' in col or 
                    'Severity_' in col or 
                    'Electricity Status_' in col or 
                    'Semester Period_' in col or 
                    'Academic Context_' in col or 
                    'Day_Period_' in col or 
                    col in ['Hour', 'Day_of_Week', 'Month', 'Day_of_Year', 'Location_Freq']]
    
    target_col = 'Incident_Type_Encoded'
    
    return df, feature_cols, target_col, le_incident

def inverse_transform_incident_type(le, encoded_values):
    """Maps encoded predictions back to original incident types."""
    return le.inverse_transform(encoded_values)