# ================================
# STEP 4: Install Required Libraries (RUN IN TERMINAL)
# pip install scikit-learn matplotlib pandas numpy
# ================================

# ================================
# STEP 4: Anomaly Detection
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# -------------------------------
# STEP 4.1: Load Dataset
# -------------------------------
df = pd.read_csv("features_dataset.csv")

print("Dataset loaded:", df.shape)

# -------------------------------
# STEP 4.2: Prepare Features
# -------------------------------
X = df.drop(columns=["file_name"])

# -------------------------------
# STEP 4.3: Train Isolation Forest
# -------------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X)

# -------------------------------
# STEP 4.4: Get Anomaly Score
# -------------------------------
scores = model.decision_function(X)

df["anomaly_score"] = scores

# -------------------------------
# STEP 4.5: Convert to Health Score
# -------------------------------
min_s = np.min(scores)
max_s = np.max(scores)

health_score = (scores - min_s) / (max_s - min_s) * 100

df["health_score"] = health_score

# -------------------------------
# STEP 4.6: Plot Health Score
# -------------------------------
plt.figure()
plt.plot(df["health_score"])
plt.title("Health Score over Time")
plt.xlabel("Time")
plt.ylabel("Health Score")
plt.show()

print("✅ Anomaly Detection Done")
print("🎉 STEP 4 COMPLETED")