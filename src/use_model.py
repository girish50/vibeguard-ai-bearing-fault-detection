# ================================
# FINAL: ADVANCED USE MODEL (FINAL CLEAN)
# ================================

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -------------------------------
# LOAD MODELS
# -------------------------------
model = joblib.load("../models/best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
iso = joblib.load("../models/isolation_model.pkl")

print("✅ Models loaded!")

# -------------------------------
# FILE SELECT
# -------------------------------
Tk().withdraw()

file_path = askopenfilename(
    initialdir="../dataset",
    title="Select dataset file",
    filetypes=[("Text Files", "*.*")]
)

if not file_path:
    print("❌ No file selected")
    exit()

print(f"\n📂 Selected file: {file_path}")

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv(file_path, sep='\t', header=None)
signal = data[0].values

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(signal):
    time_feat = [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sqrt(np.mean(signal**2)),  # RMS
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

features = extract_features(signal)

# -------------------------------
# FEATURE NAMES (MUST MATCH TRAINING)
# -------------------------------
feature_names = [
    "mean","std","max","min","rms",
    "kurtosis","skew",
    "freq_mean","freq_max","freq_std"
]

# -------------------------------
# CREATE DATAFRAME
# -------------------------------
df_model = pd.DataFrame([features], columns=feature_names)

# -------------------------------
# ANOMALY SCORE
# -------------------------------
anomaly_score = iso.decision_function(df_model)[0]
df_model["anomaly_score"] = anomaly_score

# -------------------------------
# SCALE + PREDICT
# -------------------------------
sample_scaled = scaler.transform(df_model)

pred = model.predict(sample_scaled)[0]

if hasattr(model, "predict_proba"):
    prob = model.predict_proba(sample_scaled)[0][1]
else:
    prob = 0.5

# -------------------------------
# HEALTH SCORE
# -------------------------------
health_score = int((1 - prob) * 100)

# -------------------------------
# RESULT (DEFINE FIRST)
# -------------------------------
if prob < 0.4:
    status = "🟢 NORMAL"
elif prob < 0.7:
    status = "🟡 WARNING"
else:
    status = "🔴 FAULT"

# -------------------------------
# EARLY WARNING (IMPROVED)
# -------------------------------
early_warning = (status == "🟢 NORMAL" and anomaly_score < -0.15 and prob < 0.3)

# -------------------------------
# OUTPUT
# -------------------------------
print("\n📊 Prediction:", status)
print("🤖 Model Decision:", "FAULT" if pred == 1 else "NORMAL")
print("📈 Probability:", round(prob, 3))
print("🧠 Health Score:", health_score)
print("⚠️ Anomaly Score:", round(anomaly_score, 4))

if early_warning:
    print("🚨 EARLY WARNING DETECTED (Degradation starting)")
else:
    print("✅ System Stable")

# -------------------------------
# FAULT / ANOMALY LOCATION
# -------------------------------
if status != "🟢 NORMAL" or early_warning:

    print("\n🔍 Signal Analysis (Abnormal Region Detection)")

    mean = np.mean(signal)
    std = np.std(signal)

    threshold_high = mean + 2 * std
    threshold_low = mean - 2 * std

    abnormal_indices = np.where(
        (signal > threshold_high) | (signal < threshold_low)
    )[0]

    if len(abnormal_indices) > 0:

        print("\n📍 First abnormal index:", abnormal_indices[0])
        print("📍 Last abnormal index:", abnormal_indices[-1])
        print(f"🔢 Total abnormal points: {len(abnormal_indices)}")

        # show signal near last anomaly
        start = max(0, abnormal_indices[-1] - 20)
        end = abnormal_indices[-1]

        print("\n📉 Signal near fault region:")
        print(signal[start:end])

        print("\n🚨 Fault region identified near end of signal")

    else:
        print("\n⚠️ No strong spikes found, but model detected pattern-based degradation")

# -------------------------------
# DONE
# -------------------------------
print("\n🎉 Analysis Completed Successfully!")