import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SPAE - Predictive Maintenance", layout="wide")

# --- 2. HIGH-CONTRAST THEME ENFORCER (Fixes the white-on-white issue) ---
st.markdown("""
    <style>
    /* Force the main background to a light grey */
    .stApp {
        background-color: #f1f5f9 !important;
    }
    
    /* Force ALL text in the metrics to be Dark Blue/Black */
    [data-testid="stMetricValue"] > div {
        color: #0f172a !important; 
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] > div > p {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Create a visible border and white background for the metric boxes */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        padding: 15px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Fix sidebar visibility */
    .stSidebar {
        background-color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        # If you are using FD002, ensure these files are the FD002 versions
        model = joblib.load('random_forest_model.pkl')
        features = joblib.load('feature_columns.pkl')
        return model, features
    except Exception as e:
        return None, None

model, trained_features = load_assets()

# --- 4. APP HEADER ---
st.title("🛡️ SPAE: Smart Predictive Analytics Engine")
st.markdown("### *Prophesying Engine Health via Neural Telemetry*")
st.caption("Derived from the Scots 'spae' (to foretell).")
st.divider()

# --- 5. SIDEBAR ---
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload C-MAPSS Data (FD001/FD002)", type=['csv', 'txt'])

if uploaded_file and model is not None:
    # Load raw data
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    
    # NASA C-MAPSS Mapping
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
        engine_df[f'{col}_rollmean'] = engine_df[col].rolling(window=10).mean()
    
    engine_df = engine_df.ffill().bfill()
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
    # These containers will now have dark text on white backgrounds
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(label="Prophesied RUL", value=f"{int(prediction)} Cycles")
    
    with c2:
        if prediction > 50:
            st.success("✅ **STATUS: HEALTHY**")
        elif prediction > 20:
            st.warning("⚠️ **STATUS: MAINTENANCE DUE**")
        else:
            st.error("🚨 **STATUS: CRITICAL RISK**")

    with c3:
        st.metric(label="Cycles Completed", value=current_cycle)

    st.divider()

    # --- 9. VISUALS ---
    t1, t2 = st.tabs(["📈 Sensor Trends", "🧠 Model Explainability"])

    with t1:
        target = st.selectbox("Select Sensor", sensor_cols, index=10)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.lineplot(data=engine_df, x='cycle', y=target, label="Raw Pulse", alpha=0.3, ax=ax)
        sns.lineplot(data=engine_df, x='cycle', y=f'{target}_rollmean', label="Smoothed Trend", color='red', ax=ax)
        st.pyplot(fig)

    with t2:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X_input.columns).sort_values(ascending=False).head(10)
            fig_imp, ax_imp = plt.subplots()
            feat_imp.plot(kind='barh', color='#2563eb', ax=ax_imp)
            ax_imp.invert_yaxis()
            st.pyplot(fig_imp)
else:
    st.info("👋 Welcome to SPAE. Upload a dataset to begin the prophecy.")