# app/app.py - Robust Streamlit Application
import streamlit as st
import pandas as pd
import io
import os
from pathlib import Path
import plotly.express as px # <--- NEW IMPORT

# Use the safe loading logic from the training script
from utils.preprocessing import load_data # Removed feature_engineering import from here
from utils.visualization import create_bar_chart # Keeping this if you still use it
from utils.report_exporter import create_pdf_report
from model.predictor import predict_risks, IS_MODEL_LOADED
from news.news_fetch import fetch_aau_news
from model.train import train_model # Import for retraining option
import numpy as np # <--- NEW IMPORT

# --- Configuration ---
st.set_page_config(
    page_title="AAU Campus Security Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
DEFAULT_DATA_PATH = Path('data/aau_security_logs.csv')

# --- Helper Functions ---

# New Plotly Heatmap function
def create_plotly_heatmap(df):
    """
    Generates a Plotly Heatmap for Incident Count aggregated by Location and Hour (FR7).
    """
    # 1. CRITICAL: Ensure 'Hour' exists in the filtered DataFrame (df)
    if 'Time' in df.columns and 'Hour' not in df.columns:
        # Convert time objects to hour integers for correct aggregation
        df['Hour'] = df['Time'].apply(lambda x: x.hour if x is not np.nan else 0)
    
    if df.empty or 'Location' not in df.columns or 'Hour' not in df.columns:
        return None 
        
    # 2. Aggregate the data
    heatmap_data = df.groupby(['Location', 'Hour']).size().reset_index(name='Incident_Count')
    
    # Prepare 'Hour' for display/sorting
    heatmap_data['Hour'] = heatmap_data['Hour'].astype(int).astype(str) + ':00' 

    # 3. Create the Plotly Heatmap
    fig = px.density_heatmap(
        heatmap_data, 
        x='Hour', 
        y='Location', 
        z='Incident_Count',
        title='Incident Count by Location and Time of Day (FR7)',
        color_continuous_scale="Plasma" 
    )
    
    # Sort the x-axis (Hour) numerically
    # The list comprehension ensures correct integer sorting
    hour_categories = sorted(heatmap_data['Hour'].unique(), key=lambda x: int(x.split(':')[0]))
    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': hour_categories}
    )
    return fig


@st.cache_data(show_spinner="Loading and cleaning data...")
def initial_data_load(data_file_path):
    """Loads and cleans data once, handles critical ValueError from load_data."""
    try:
        # load_data handles both string path and file-like object (io.BytesIO)
        df = load_data(str(data_file_path)) 
        print("Initial data loaded successfully.")
        return df
    except ValueError as e:
        st.error(f"FATAL DATA ERROR: Could not process incident logs. Reason: {e}. Check your Date/Time formats.")
        return pd.DataFrame()
    except FileNotFoundError as e:
        st.warning(f"Default data file not found at {data_file_path}.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

def handle_file_upload(uploaded_file):
    """Loads and stores the uploaded CSV data (R1)."""
    if uploaded_file is not None:
        try:
            # Pass the file-like object directly to load_data
            df = load_data(uploaded_file)
            st.session_state['df_logs'] = df
            st.success(f"Log file loaded successfully with {len(df)} records.")
            # Clear any old prediction/training status
            st.session_state['model_trained'] = False
        except Exception as e:
            st.error(f"Error loading file: Ensure the CSV format matches the required fields. Error: {e}")

def train_and_update_model(df):
    """Triggers the model training and updates session state."""
    # Temporarily save the data locally for the model script to access
    temp_path = 'data/temp_logs.csv'
    df.to_csv(temp_path, index=False)
    
    with st.spinner("Training RandomForest Model... This may take a moment."):
        model, encoder = train_model(temp_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if model is not None:
        st.session_state['model_trained'] = True
        st.success("Model trained and saved successfully! Ready for prediction.")
    else:
        st.session_state['model_trained'] = False
        st.error("Model training failed. Check data quality/size (need more than 6 samples and 2 classes).")


# --- Session State Initialization (FR3) ---
if 'df_logs' not in st.session_state:
    st.session_state['df_logs'] = initial_data_load(DEFAULT_DATA_PATH)

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = IS_MODEL_LOADED


# --- Sidebar: Data Upload and Model Training ---
with st.sidebar:
    st.title("⚙️ System Control")
    
    # 1. Data Upload (R1)
    st.markdown("### 1. Upload Logs (R1)")
    uploaded_file = st.file_uploader("Choose a CSV file (e.g., aau_security_logs.csv)", type="csv", on_change=lambda: handle_file_upload(uploaded_file))

    # --- Load default if session state is empty and file exists ---
    if st.session_state['df_logs'].empty and DEFAULT_DATA_PATH.exists():
        st.info(f"Using default sample data from {DEFAULT_DATA_PATH}.")
    
    if not st.session_state['df_logs'].empty:
        st.markdown("---")
        # 2. Model Training (FR4)
        st.markdown("### 2. Model Training (FR4)")
        if not st.session_state['model_trained']:
            st.warning("Model needs to be trained or reloaded.")
            if st.button("Train Model Now", use_container_width=True):
                train_and_update_model(st.session_state['df_logs'])
        else:
            st.success("Model is currently trained and loaded.")
            if st.button("Retrain Model", use_container_width=True):
                train_and_update_model(st.session_state['df_logs'])
    
    st.markdown("---")
    # 3. News API Key (Optional)
    st.markdown("### 3. News API Key (Optional)")
    api_key = st.text_input("Enter News API Key (e.g., from NewsAPI)", type="password")


# --- Main Application Layout ---
st.title("Predictive Campus Security Risk Analysis System 🛡️")
st.markdown("A low-cost, AAU-friendly model using historical logs and Machine Learning.")
st.markdown("---")

if st.session_state['df_logs'].empty:
    st.info("Please upload a CSV file with security logs in the sidebar to begin analysis.")
    # If model is not loaded, show general warning
    if not st.session_state['model_trained']:
        st.warning("Prediction functionality is disabled because no model is trained/loaded.")

else:
    # ----------------------------------------------------
    # TAB 1: Historical Trend Analysis (R2, FR8)
    # ----------------------------------------------------
    tab_trends, tab_prediction, tab_news = st.tabs(["📊 Historical Trends", "🚨 Risk Prediction", "📰 Ekpoma News"])

    with tab_trends:
        st.header("Historical Incident Trends (R2)")
        df_display = st.session_state['df_logs'].copy()
        
        # Filtering (FR8)
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_location = st.selectbox("Filter by Location", ['All'] + sorted(df_display['Location'].unique().tolist()))
        with col2:
            selected_incident = st.selectbox("Filter by Incident Type", ['All'] + sorted(df_display['Incident Type'].unique().tolist()))
        with col3:
            # Ensure the Date column is not None before accessing dt.year
            # Use .dt.year.dropna().unique() to safely get years
            available_years = df_display['Date'].dt.year.dropna().unique().astype(int).tolist()
            selected_year = st.selectbox("Filter by Year", ['All'] + sorted(available_years))

        # Apply filters
        if selected_location != 'All':
            df_display = df_display[df_display['Location'] == selected_location]
        if selected_incident != 'All':
            df_display = df_display[df_display['Incident Type'] == selected_incident]
        if selected_year != 'All':
            df_display = df_display[df_display['Date'].dt.year == selected_year]
        
        st.subheader("Data Preview")
        st.dataframe(df_display.head(10), use_container_width=True)
        
        if not df_display.empty:
            
            # --- Visualizations (FR7) ---
            st.subheader("Key Visualizations")
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.markdown("#### Incidents by Location (Bar Chart)")
                # Assume create_bar_chart is now Plotly or works with st.image
                bar_chart_b64 = create_bar_chart(df_display, 'Location', 'Top Incident Locations')
                if bar_chart_b64:
                    st.image(f"data:image/png;base64,{bar_chart_b64}")

            with vis_col2:
                st.markdown("#### Incident Frequency: Location vs. Hour (Heatmap)")
                # CRITICAL FIX: Call the new Plotly function
                heatmap_fig = create_plotly_heatmap(df_display.copy()) 
                
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True) # <--- FIXED DISPLAY METHOD
                else:
                    st.warning("Heatmap data is insufficient or required columns are missing.")
            
        else:
            st.warning("No data matches the selected filters.")


    # ----------------------------------------------------
    # TAB 2: Risk Prediction (R3, R4, R6)
    # ----------------------------------------------------
    with tab_prediction:
        st.header("Future Risk Predictions (R3, R4)")
        
        if st.session_state['model_trained']:
            df_predictions, high_risk_info = predict_risks(st.session_state['df_logs'])
            
            if not df_predictions.empty:
                st.subheader("Overall Risk Summary")
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                
                sum_col1.metric("Top High-Risk Location", high_risk_info.get('Top High-Risk Location', 'N/A'))
                sum_col2.metric("Top High-Risk Time (Hour)", high_risk_info.get('Top High-Risk Time', 'N/A'))
                sum_col3.metric("Most Probable Incident Type", high_risk_info.get('Most Probable Incident Type', 'N/A'))
                
                st.subheader("Detailed Risk Forecast")
                st.dataframe(df_predictions[df_predictions['Risk_Level'] != 'Low'], use_container_width=True, height=300)
                
                st.markdown("---")
                
                # Report Export (R6)
                pdf_output = create_pdf_report(df_predictions, high_risk_info, None) # None for charts
                st.download_button(
                    label="Export Risk Report (PDF)",
                    data=pdf_output,
                    file_name="AAU_Risk_Report.pdf",
                    mime="application/pdf"
                )
                
            else:
                st.error("Could not generate predictions. The model may be trained, but data size is insufficient (need at least 2 samples).")
                
        else:
            st.warning("Model is not yet trained or loaded. Please train the model in the sidebar.")


    # ----------------------------------------------------
    # TAB 3: Supplementary News (R5, FR12)
    # ----------------------------------------------------
    with tab_news:
        st.header("External News Context (R5)")
        st.info("This section fetches external news articles for supplementary context around Ekpoma/AAU.")
        
        # api_key is unused in the final fetch_aau_news function but kept for compatibility
        news_articles = fetch_aau_news(api_key) 
        
        if news_articles:
            for article in news_articles:
                st.markdown(f"**📰 {article['title']}**")
                st.caption(f"{article['description']}")
                st.markdown(f"[Read Full Article]({article['url']})")
                st.markdown("---")
        else:
            st.warning("Could not fetch or load news articles. Check your data connection or news source configuration.")


# --- Instructions for Running ---
if __name__ == '__main__':
    st.caption("System ready. Run with: `python -m streamlit run app/app.py`")