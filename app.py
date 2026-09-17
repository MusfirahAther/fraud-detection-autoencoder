import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# ---------- Page setup ----------
st.set_page_config(page_title="Fraud Detection", layout="wide", page_icon="🛡️")

# ---------- Full dark theme styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(180deg, #0a0e14 0%, #0d1117 100%);
    color: #e6e6e6;
}

[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-weight: 800;
}

h1 { color: #ffffff !important; font-weight: 800; letter-spacing: -0.5px; }
h2, h3 { color: #f0f0f0 !important; font-weight: 700; }
.stCaption, [data-testid="stCaptionContainer"] { color: #8b949e !important; }

[data-testid="stTabs"] button {
    color: #8b949e;
    font-weight: 600;
    font-size: 16px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff4b4b !important;
    border-bottom: 3px solid #ff4b4b !important;
}

input, textarea {
    background-color: #161b22 !important;
    color: #e6e6e6 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}
label, [data-testid="stWidgetLabel"] p {
    color: #8b949e !important;
    font-weight: 600 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff4b4b 0%, #d92b2b 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 700;
    font-size: 15px;
    box-shadow: 0 4px 14px rgba(255, 75, 75, 0.3);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
    transform: translateY(-1px);
}

[data-testid="stFileUploader"] {
    background-color: #161b22;
    border: 1px dashed #30363d;
    border-radius: 10px;
    padding: 10px;
}

.metric-box {
    background: linear-gradient(145deg, #161b22, #1c2129);
    padding: 22px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #30363d;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.metric-box h2 {
    margin: 5px 0 0 0;
    font-size: 28px;
    color: #ffffff !important;
}

.result-normal {
    background: linear-gradient(135deg, #0d2818, #12351f);
    border: 2px solid #2ecc71;
    color: #2ecc71;
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    box-shadow: 0 0 30px rgba(46, 204, 113, 0.15);
}
.result-fraud {
    background: linear-gradient(135deg, #2e0d0d, #3d1212);
    border: 2px solid #e74c3c;
    color: #e74c3c;
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    box-shadow: 0 0 30px rgba(231, 76, 60, 0.2);
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 22px !important;
    overflow: visible !important;
    white-space: nowrap !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 10px;
}

hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ---------- Load model and scalers once ----------
@st.cache_resource
def load_all():
    input_dim = 30
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(20, activation='relu')(input_layer)
    encoder = Dense(14, activation='relu')(encoder)
    bottleneck = Dense(10, activation='relu')(encoder)
    decoder = Dense(14, activation='relu')(bottleneck)
    decoder = Dense(20, activation='relu')(decoder)
    output_layer = Dense(input_dim, activation='linear')(decoder)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.load_weights("model/autoencoder.weights.h5")

    with open("model/scaler_amount.pkl", "rb") as f:
        scaler_amount = pickle.load(f)
    with open("model/scaler_time.pkl", "rb") as f:
        scaler_time = pickle.load(f)
    with open("model/threshold.txt", "r") as f:
        threshold = float(f.read().strip())
    return model, scaler_amount, scaler_time, threshold

model, scaler_amount, scaler_time, threshold = load_all()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.write("This tool uses an **Autoencoder** trained only on normal transactions. "
             "It flags a transaction as fraud when the model struggles to reconstruct it.")
    st.markdown("---")
    st.markdown("#### Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", "0.72")
    col2.metric("Recall", "0.77")
    col3.metric("F1-score", "0.74")
    st.markdown("---")
    st.markdown(f"**Threshold:** `{threshold:.4f}`")

st.markdown("# Credit Card Fraud Detection")
st.caption("Autoencoder-based anomaly detection")
st.markdown("")

# ---------- Helper functions ----------
def score_transaction(v_values, amount, time):
    amount_scaled = scaler_amount.transform([[amount]])[0][0]
    time_scaled = scaler_time.transform([[time]])[0][0]
    features = np.array(v_values + [amount_scaled, time_scaled]).reshape(1, -1)
    reconstruction = model.predict(features, verbose=0)
    error = np.mean(np.power(features - reconstruction, 2))
    return error

def score_dataframe(df):
    df = df.copy()
    df["Amount_scaled"] = scaler_amount.transform(df[["Amount"]])
    df["Time_scaled"] = scaler_time.transform(df[["Time"]])
    v_cols = [f"V{i}" for i in range(1, 29)]
    feature_cols = v_cols + ["Amount_scaled", "Time_scaled"]
    X = df[feature_cols].values
    reconstructions = model.predict(X, verbose=0)
    errors = np.mean(np.power(X - reconstructions, 2), axis=1)
    df["Reconstruction_Error"] = errors
    df["Flagged"] = np.where(errors > threshold, "Fraud", "Normal")
    return df

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["Manual Entry", "Upload CSV"])

with tab1:
    st.markdown("### Enter Transaction Details")
    col1, col2 = st.columns(2)
    amount = col1.number_input("Amount", min_value=0.0, value=100.0)
    time = col2.number_input("Time (seconds)", min_value=0.0, value=50000.0)
    v_text = st.text_area("Paste 28 comma-separated V1-V28 values",
                           placeholder="e.g. -1.35, -0.07, 2.53, ... (28 values total)")

    if st.button("Check Transaction"):
        try:
            v_values = [float(x.strip()) for x in v_text.split(",")]
            if len(v_values) != 28:
                st.error(f"Expected 28 values, got {len(v_values)}. Please check your input.")
            else:
                error = score_transaction(v_values, amount, time)
                is_fraud = error > threshold

                st.markdown("")
                if is_fraud:
                    st.markdown('<div class="result-fraud">⚠️ FLAGGED AS FRAUD</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-normal">✓ NORMAL</div>', unsafe_allow_html=True)

                st.markdown("")
                col1, col2 = st.columns(2)
                col1.markdown(f'<div class="metric-box">Reconstruction Error<h2>{error:.4f}</h2></div>', unsafe_allow_html=True)
                col2.markdown(f'<div class="metric-box">Threshold<h2>{threshold:.4f}</h2></div>', unsafe_allow_html=True)

                st.markdown("")
                chart_df = pd.DataFrame({"Value": [error, threshold]}, index=["Reconstruction Error", "Threshold"])
                st.bar_chart(chart_df, color="#ff4b4b")
        except ValueError:
            st.error("Could not parse the V values. Make sure they're comma-separated numbers.")

with tab2:
    st.markdown("### Upload a CSV File")
    st.write("CSV must contain columns: `Time`, `V1` to `V28`, `Amount`")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            result_df = score_dataframe(df)
            fraud_count = (result_df["Flagged"] == "Fraud").sum()

            st.markdown("")
            col1, col2 = st.columns(2)
            col1.markdown(f'<div class="metric-box">Total Transactions<h2>{len(result_df)}</h2></div>', unsafe_allow_html=True)
            col2.markdown(f'<div class="metric-box">Flagged as Fraud<h2>{fraud_count}</h2></div>', unsafe_allow_html=True)

            st.markdown("")
            st.dataframe(result_df[["Time", "Amount", "Reconstruction_Error", "Flagged"]], use_container_width=True)