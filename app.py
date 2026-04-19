import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SPAE - Predictive Maintenance", layout="wide")

# --- 2. THEME & VISIBILITY ENFORCER (Fixes the white-box issue) ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Force Metric Text Visibility */
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 1.1rem;
    }
    
    /* Professional Card Styling for Metrics */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('random_forest_model.pkl')
        features = joblib.load('feature_columns.pkl')
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, trained_features = load_assets()

# --- 4. APP HEADER ---
st.title("🛡️ SPAE: Smart Predictive Analytics Engine")
st.markdown("### *Foretelling industrial health through high-fidelity sensor telemetry*")
st.caption("Named from the Scots 'spae' (to prophesy), this framework estimates Remaining Useful Life (RUL).")
st.divider()

# --- 5. SIDEBAR ---
st.sidebar.header("📁 Data Integration")
uploaded_file = st.sidebar.file_uploader("Upload FD001/FD002 Test Data", type=['csv', 'txt'])

if uploaded_file and model is not None:
    # Load raw data
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    
    # Standard NASA C-MAPSS Mapping
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
    df.columns = cols[:len(df.columns)]
    
    # Engine Selection
    engine_list = df['engine_id'].unique()
    selected_engine = st.sidebar.selectbox("Select Engine Unit", engine_list)
    
    # Filter for selected engine
    engine_df = df[df['engine_id'] == selected_engine].copy()
    
    # --- 6. FEATURE ENGINEERING ---
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    for col in sensor_cols:
        # Generate rolling means to match the "spaeing" (prediction) logic
        engine_df[f'{col}_rollmean'] = engine_df[col].rolling(window=10).mean()
    
    # Fill gaps from rolling window start
    engine_df = engine_df.ffill().bfill()
    
    # Get the latest state
    latest_state = engine_df.iloc[-1:].copy()
    
    # Alignment with Training Features
    if trained_features:
        X_input = latest_state.reindex(columns=trained_features, fill_value=0)
    else:
        X_input = latest_state.drop(columns=['engine_id', 'cycle'], errors='ignore')

    # --- 7. INFERENCE ---
    prediction = model.predict(X_input)[0]
    current_cycle = int(latest_state['cycle'].values[0])

    # --- 8. DASHBOARD DISPLAY ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Prophesied RUL", value=f"{int(prediction)} Cycles")
    
    with col2:
        # Logic for health status
        if prediction > 50:
            st.success("✅ **STATUS: HEALTHY**")
        elif prediction > 20:
            st.warning("⚠️ **STATUS: MAINTENANCE DUE**")
        else:
            st.error("🚨 **STATUS: CRITICAL RISK**")

    with col3:
        st.metric(label="Total Cycles Completed", value=current_cycle)

    st.divider()

    # --- 9. ANALYTICS TABS ---
    tab1, tab2 = st.tabs(["📊 Sensor Diagnostics", "🧠 Feature Explainability"])

    with tab1:
        st.subheader("Temporal Sensor Trends")
        target = st.selectbox("Select Sensor", sensor_cols, index=10) # Default to sensor 11
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=engine_df, x='cycle', y=target, label="Raw Pulse", alpha=0.4, ax=ax)
        sns.lineplot(data=engine_df, x='cycle', y=f'{target}_rollmean', label="Smoothed Trend", color='red', ax=ax)
        ax.set_title(f"Degradation Profile: {target}")
        st.pyplot(fig)

    with tab2:
        st.subheader("Importance Scores (Model Interpretability)")
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X_input.columns).sort_values(ascending=False).head(10)
            
            fig_imp, ax_imp = plt.subplots()
            feat_imp.plot(kind='barh', color='#3b82f6', ax=ax_imp)
            ax_imp.invert_yaxis()
            st.pyplot(fig_imp)

else:
    # Welcome Screen
    st.info("👋 Ready to spae (foretell) engine health? Please upload a NASA C-MAPSS data file to begin.")
    st.markdown("""
    ### System Workflow for Journal Documentation:
    1. **Data Acquisition:** Ingestion of multivariate sensor telemetry.
    2. **Temporal Transformation:** Calculation of rolling statistical means.
    3. **Feature Alignment:** Synchronization of input vectors with the Random Forest architecture.
    4. **Inference:** Generation of RUL estimation.
    """)