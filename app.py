import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
st.set_page_config(page_title="SPAE Engine", layout="wide")

# --- CUSTOM CSS (PREMIUM LOOK) ---
st.markdown("""
<style>
body {
    background-color: #0f172a;
    color: white;
}
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}
.title {
    font-size: 42px;
    font-weight: 700;
    color: #e2e8f0;
}
.subtitle {
    font-size: 18px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    model = joblib.load('random_forest_model.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

model, trained_features = load_assets()

# --- HEADER ---
st.markdown('<div class="title">🛡️ SPAE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Predictive Analytics Engine</div>', unsafe_allow_html=True)

st.caption("“Spae” — predict the future")

st.divider()

# --- SIDEBAR ---
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload NASA Turbofan Data", type=['csv', 'txt'])

if uploaded_file:

    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    columns = ['engine_id', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
    df.columns = columns[:len(df.columns)]

    engine_list = df['engine_id'].unique()
    selected_engine = st.sidebar.selectbox("Select Engine", engine_list)

    engine_df = df[df['engine_id'] == selected_engine].copy()

    # --- FEATURE ENGINEERING ---
    sensor_cols = [c for c in df.columns if 'sensor' in c]
    for col in sensor_cols:
        engine_df[f'{col}_rollmean'] = engine_df[col].rolling(window=10).mean()

    engine_df = engine_df.ffill().bfill()

    latest_state = engine_df.iloc[-1:].copy()

    X_input = latest_state.reindex(columns=trained_features, fill_value=0)

    # --- PREDICTION ---
    prediction = model.predict(X_input)[0]
    current_cycle = int(latest_state['cycle'].values[0])

    # --- STATUS LOGIC ---
    if prediction > 50:
        status = "HEALTHY"
        color = "#22c55e"
        risk = 20
    elif prediction > 20:
        status = "MAINTENANCE SOON"
        color = "#f59e0b"
        risk = 60
    else:
        status = "CRITICAL"
        color = "#ef4444"
        risk = 90

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="metric-card">
        <h3>Predicted RUL</h3>
        <h1 style="color:{color};">{int(prediction)} cycles</h1>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <h3>System Status</h3>
        <h2 style="color:{color};">{status}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <h3>Lifecycle Progress</h3>
        <h2>{current_cycle} cycles</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- RISK BAR ---
    st.write("### ⚠️ Failure Risk Estimation")
    st.progress(risk)

    st.divider()

    # --- TABS ---
    tab1, tab2 = st.tabs(["📈 Sensor Intelligence", "🧠 Model Insights"])

    # --- SENSOR PLOT ---
    with tab1:
        st.subheader("Sensor Behavior Over Time")

        target_sensor = st.selectbox("Choose Sensor", sensor_cols, index=10)

        fig, ax = plt.subplots(figsize=(10, 4))

        sns.lineplot(x=engine_df['cycle'], y=engine_df[target_sensor], ax=ax, label="Raw", alpha=0.4)
        sns.lineplot(x=engine_df['cycle'], y=engine_df[f'{target_sensor}_rollmean'], ax=ax, label="Smoothed", linewidth=2)

        ax.set_title(f"{target_sensor} Trend")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Sensor Value")

        st.pyplot(fig)

    # --- FEATURE IMPORTANCE ---
    with tab2:
        st.subheader("Top Drivers Behind Prediction")

        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X_input.columns).sort_values(ascending=False).head(10)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_imp)

        ax_imp.set_title("Feature Importance")
        st.pyplot(fig_imp)

else:
    st.markdown("""
    ### 👋 Welcome to SPAE
    
    **SPAE (Smart Predictive Analytics Engine)** is designed to *foretell engine failures before they happen*.
    
    Upload a NASA turbofan dataset to begin:
    
    - 🔍 Analyze sensor behavior  
    - 📊 Predict Remaining Useful Life  
    - ⚠️ Detect failure risks early  
    """)