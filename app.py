import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SPAE - Predictive Maintenance", layout="wide")

# --- 2. THEME & BOX UNIFORMITY (The specific changes are here) ---
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }

    /* Force all metric containers to be the same height and style */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        min-height: 150px; /* Forces uniform size */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Target the first metric box (Predicted RUL) - Light Blue background */
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] {
        background-color: #e0f2fe !important;
        border: 1px solid #7dd3fc !important;
    }

    /* Target the third metric box (Total Cycles) - Light Grey/Purple background */
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] {
        background-color: #f3f4f6 !important;
        border: 1px solid #d1d5db !important;
    }

    /* Force text to be dark for visibility */
    [data-testid="stMetricValue"] > div {
        color: #1e293b !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] > div > p {
        color: #475569 !important;
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
    except:
        return None, None

model, trained_features = load_assets()

# --- 4. APP HEADER ---
st.title("🛡️ SPAE: Smart Predictive Analytics Engine")
st.markdown("### *Forecasting Industrial Asset Longevity*")
st.divider()

# --- 5. SIDEBAR ---
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload FD001/FD002 Data", type=['csv', 'txt'])

if uploaded_file and model is not None:
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    cols = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
    df.columns = cols[:len(df.columns)]
    
    selected_engine = st.sidebar.selectbox("Select Engine Unit", df['engine_id'].unique())
    engine_df = df[df['engine_id'] == selected_engine].copy()
    
    # Feature Engineering
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    for col in sensor_cols:
        engine_df[f'{col}_rollmean'] = engine_df[col].rolling(window=10).mean()
    
    engine_df = engine_df.ffill().bfill()
    latest_state = engine_df.iloc[-1:].copy()
    
    if trained_features:
        X_input = latest_state.reindex(columns=trained_features, fill_value=0)
    else:
        X_input = latest_state.drop(columns=['engine_id', 'cycle'], errors='ignore')

    # --- 6. PREDICTION ---
    prediction = model.predict(X_input)[0]
    current_cycle = int(latest_state['cycle'].values[0])

    # --- 7. DISPLAY BOXES ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Prophesied RUL", value=f"{int(prediction)} Cycles")
    
    with col2:
        # Note: success/warning/error boxes have their own internal heights
        if prediction > 50:
            st.success("✅ **HEALTHY**")
        elif prediction > 20:
            st.warning("⚠️ **DUE**")
        else:
            st.error("🚨 **CRITICAL**")

    with col3:
        st.metric(label="Total Cycles Run", value=current_cycle)

    st.divider()
    
    # (Visualizations tabs follow here...)
    st.info("Visual Analytics active for Engine " + str(selected_engine))

else:
    st.info("Upload data to begin.")