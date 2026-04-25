# ================================
# STEP 5: Install Required Libraries (RUN IN TERMINAL)
# pip install pandas numpy matplotlib
# ================================

# ================================
# STEP 5: Early Fault Detection
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# STEP 5.1: Load Dataset
# -------------------------------
df = pd.read_csv("features_dataset.csv")

# -------------------------------
# STEP 5.2: Recompute Health Score (reuse logic)
# -------------------------------
from sklearn.ensemble import IsolationForest

X = df.drop(columns=["file_name"])

model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

scores = model.decision_function(X)

# Normalize → Health Score
health_score = (scores - np.min(scores)) / (np.max(scores) - np.min(scores)) * 100
df["health_score"] = health_score

# -------------------------------
# STEP 5.3: Detect Early Fault Point
# -------------------------------
threshold = 60   # you can tune this

detection_point = None

for i, score in enumerate(df["health_score"]):
    if score < threshold:
        detection_point = i
        break

# -------------------------------
# STEP 5.4: Define Failure Point
# -------------------------------
failure_point = len(df) - 1

# -------------------------------
# STEP 5.5: Compute Early Detection
# -------------------------------
early_gap = failure_point - detection_point

print("Detection Point (T'):", detection_point)
print("Failure Point (T):", failure_point)
print("Early Detection Gap (Δ):", early_gap)

# -------------------------------
# STEP 5.6: Visualization
# -------------------------------
plt.figure()
plt.plot(df["health_score"], label="Health Score")

# Mark detection point
plt.axvline(x=detection_point, color='red', linestyle='--', label="Detection Point")

# Mark failure point
plt.axvline(x=failure_point, color='black', linestyle='--', label="Failure Point")

plt.title("Early Fault Detection")
plt.xlabel("Time")
plt.ylabel("Health Score")
plt.legend()
plt.show()

print("✅ Early Detection Completed")
print("🎉 STEP 5 COMPLETED")