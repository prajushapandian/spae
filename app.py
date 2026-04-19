import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
st.set_page_config(page_title="SPAE - Predictive Maintenance", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    # Load the model and the specific list of feature names used during training
    model = joblib.load('random_forest_model.pkl')
    try:
        features = joblib.load('feature_columns.pkl')
    except:
        # If the file is missing, we list the standard FD001 features + rollmeans
        # IMPORTANT: This list must match your training notebook exactly!
        st.error("feature_columns.pkl not found. Re-run your notebook to save it.")
        features = None 
    return model, features

model, trained_features = load_assets()

# --- APP HEADER ---
st.title("🛡️ SPAE: Smart Predictive Analytics Engine")
st.subheader("Early Detection & Maintenance Forecasting for Industrial Turbofans")
st.divider()

# --- SIDEBAR: UPLOAD & INPUT ---
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload FD001 Test Data", type=['csv', 'txt'])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    
    # NASA C-MAPSS standard column mapping
    columns = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
    df.columns = columns[:len(df.columns)]
    
    # Select Engine
    engine_list = df['engine_id'].unique()
    selected_engine = st.sidebar.selectbox("Select Engine Unit", engine_list)
    
    # Process Engine Data
    engine_df = df[df['engine_id'] == selected_engine].copy()
    
    # --- FEATURE ENGINEERING ---
    # We must create the rollmean columns because the model expects them
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    for col in sensor_cols:
        engine_df[f'{col}_rollmean'] = engine_df[col].rolling(window=10).mean()
    
    # Handle NaNs created by rolling window (vital for the first 10 cycles)
    engine_df = engine_df.ffill().bfill()
    
    # Get only the very latest cycle for prediction
    latest_state = engine_df.iloc[-1:].copy()
    
    # --- FEATURE ALIGNMENT (The Fix for your ValueError) ---
    if trained_features is not None:
        # Force the dataframe to have ONLY the trained features, in the CORRECT order
        X_input = latest_state.reindex(columns=trained_features, fill_value=0)
    else:
        # Emergency Fallback if pkl is missing: Drop non-features
        X_input = latest_state.drop(columns=['engine_id', 'cycle', 'RUL'], errors='ignore')

    # --- PREDICTION ---
    try:
        prediction = model.predict(X_input)[0]
        current_cycle = int(latest_state['cycle'].values[0])

        # --- DASHBOARD LAYOUT ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Predicted RUL", value=f"{int(prediction)} Cycles")
        
        with col2:
            if prediction > 50:
                st.success("**Status:** HEALTHY")
            elif prediction > 20:
                st.warning("**Status:** MAINTENANCE DUE")
            else:
                st.error("**Status:** CRITICAL RISK")

        with col3:
            st.metric(label="Total Cycles Run", value=current_cycle)

        st.divider()

        # --- VISUALIZATIONS ---
        tab1, tab2 = st.tabs(["📈 Sensor Analytics", "🔍 Feature Importance"])

        with tab1:
            st.write(f"### Historical Sensor Trends for Engine {selected_engine}")
            target_sensor = st.selectbox("Select Sensor to Visualize", sensor_cols, index=10) # Default sensor 11
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(engine_df['cycle'], engine_df[target_sensor], label="Raw Sensor", alpha=0.5)
            ax.plot(engine_df['cycle'], engine_df[f'{target_sensor}_rollmean'], label="Rolling Mean (10)", color='red')
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Reading")
            ax.legend()
            st.pyplot(fig)

        with tab2:
            st.write("### Model Decision Factors")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X_input.columns).sort_values(ascending=False).head(10)
            
            fig_imp, ax_imp = plt.subplots()
            feat_imp.plot(kind='barh', ax=ax_imp, color='skyblue')
            ax_imp.invert_yaxis()
            st.pyplot(fig_imp)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Expected Features:", trained_features)
        st.write("Provided Features:", list(X_input.columns))

else:
    st.info("👋 Welcome to SPAE. Please upload a NASA C-MAPSS data file from the sidebar to begin analysis.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=100)