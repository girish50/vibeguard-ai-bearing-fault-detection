import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="VibeGuard AI", layout="centered")

# -------------------------------
# STYLE (CLEAN IEEE LOOK)
# -------------------------------
st.markdown("""
<style>
.title {
    font-size: 34px;
    font-weight: 600;
}
.subtitle {
    color: #888;
    margin-bottom: 20px;
}
.section {
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">VibeGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Based Predictive Maintenance for Bearing Fault Detection</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# SYSTEM FLOW (IMPORTANT)
# -------------------------------
st.markdown("### 🧠 System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.markdown("**📡 Sensor**")
col1.caption("Vibration Input")

col2.markdown("**⚙️ Processing**")
col2.caption("Feature Extraction")

col3.markdown("**🤖 ML Model**")
col3.caption("Anomaly + Classification")

col4.markdown("**📊 Output**")
col4.caption("Health + Fault")

st.markdown("---")

# -------------------------------
# LOAD MODELS
# -------------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
iso = joblib.load("models/isolation_model.pkl")

# -------------------------------
# FILE INPUT
# -------------------------------
st.markdown("### 📂 Input Signal")

uploaded_file = st.file_uploader("Upload vibration signal file (any format)")

# -------------------------------
# FEATURE FUNCTION
# -------------------------------
def extract_features(signal):
    time_feat = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sqrt(np.mean(signal**2)),
        kurtosis(signal),
        skew(signal)
    ]

    fft = np.fft.fft(signal)
    mag = np.abs(fft)

    freq_feat = [
        np.mean(mag),
        np.max(mag),
        np.std(mag)
    ]

    return time_feat + freq_feat

# -------------------------------
# PROCESS
# -------------------------------
if uploaded_file:

    try:
        data = pd.read_csv(uploaded_file, sep='\t', header=None)
        signal = data[0].values

        st.success("File loaded successfully")

        # -------------------------------
        # HARDWARE CONTEXT
        # -------------------------------
        st.markdown("### 🛠️ System Context")
        st.info("""
        In real-world deployment, vibration sensors are mounted on rotating machinery.
        The sensor captures vibration signals continuously, which are processed to detect anomalies and predict bearing faults.
        """)

        # -------------------------------
        # SIGNAL GRAPH
        # -------------------------------
        st.markdown("### 📊 Signal Visualization")
        st.line_chart(signal[:500])

        # -------------------------------
        # MODEL PIPELINE
        # -------------------------------
        features = extract_features(signal)

        columns = [
            "mean","std","max","min","rms",
            "kurtosis","skew",
            "freq_mean","freq_max","freq_std"
        ]

        df = pd.DataFrame([features], columns=columns)

        anomaly_score = iso.decision_function(df)[0]
        df["anomaly_score"] = anomaly_score

        scaled = scaler.transform(df)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        health = int((1 - prob) * 100)

        # -------------------------------
        # STATUS
        # -------------------------------
        if prob < 0.4:
            status = "NORMAL"
        elif prob < 0.7:
            status = "WARNING"
        else:
            status = "FAULT"

        # -------------------------------
        # RESULT
        # -------------------------------
        st.markdown("---")
        st.markdown("### 📌 System Assessment")

        col1, col2, col3 = st.columns(3)

        col1.metric("Condition", status)
        col2.metric("Health Score", f"{health}/100")
        col3.metric("Anomaly Score", round(anomaly_score, 4))

        st.markdown("### 📈 Health Indicator")
        st.progress(health / 100)

        # -------------------------------
        # INTERPRETATION
        # -------------------------------
        st.markdown("### 📘 Interpretation")

        if status == "NORMAL":
            st.success("System operating under normal conditions.")
        elif status == "WARNING":
            st.warning("Early degradation detected. Preventive maintenance recommended.")
        else:
            st.error("Fault condition detected. Immediate maintenance required.")

        # -------------------------------
        # MODEL INSIGHT
        # -------------------------------
        st.markdown("### 🔍 Model Insight")

        st.write(f"""
        - Prediction: **{status}**
        - Probability of failure: **{round(prob,3)}**
        - Anomaly score indicates deviation from normal signal pattern.
        - Health score reflects overall system condition.
        """)

        # -------------------------------
        # FAULT REGION
        # -------------------------------
        if status != "NORMAL":

            st.markdown("### ⚠️ Fault Region Analysis")

            mean = np.mean(signal)
            std = np.std(signal)

            threshold_high = mean + 2 * std
            threshold_low = mean - 2 * std

            abnormal = np.where(
                (signal > threshold_high) | (signal < threshold_low)
            )[0]

            if len(abnormal) > 0:
                idx = int(abnormal[-1])

                st.write(f"Fault likely near index: {idx}")

                start = max(0, idx - 200)
                st.line_chart(signal[start:idx])

            else:
                st.info("Pattern-based degradation detected without sharp spikes.")

    except:
        st.error("Error reading file. Ensure correct format.")