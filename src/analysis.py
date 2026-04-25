# ================================
# STEP 7: Install Required Libraries (RUN IN TERMINAL)
# pip install scikit-learn matplotlib pandas numpy xgboost
# ================================

# ================================
# STEP 7: Feature Importance + Ablation Study
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score

# -------------------------------
# STEP 7.1: Load Dataset
# -------------------------------
df = pd.read_csv("features_dataset.csv")

# -------------------------------
# STEP 7.2: Recreate Anomaly Score
# -------------------------------
X_base = df.drop(columns=["file_name"])

iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso.fit(X_base)

scores = iso.decision_function(X_base)
df["anomaly_score"] = scores

# -------------------------------
# STEP 7.3: Create Labels
# -------------------------------
n = len(df)
labels = np.zeros(n)
labels[int(0.7 * n):] = 1
df["label"] = labels

# -------------------------------
# STEP 7.4: Prepare Data
# -------------------------------
X_without = df.drop(columns=["file_name", "label", "anomaly_score"])
X_with    = df.drop(columns=["file_name", "label"])

y = df["label"]

# -------------------------------
# STEP 7.5: Time-Based Split
# -------------------------------
split = int(0.8 * len(df))

Xw_train, Xw_test = X_without[:split], X_without[split:]
Xh_train, Xh_test = X_with[:split], X_with[split:]

y_train, y_test = y[:split], y[split:]

# -------------------------------
# STEP 7.6: Train Models
# -------------------------------
rf_without = RandomForestClassifier()
rf_with    = RandomForestClassifier()

rf_without.fit(Xw_train, y_train)
rf_with.fit(Xh_train, y_train)

# -------------------------------
# STEP 7.7: Evaluate (Ablation Study)
# -------------------------------
pred_without = rf_without.predict(Xw_test)
pred_with    = rf_with.predict(Xh_test)

acc_without = accuracy_score(y_test, pred_without)
acc_with    = accuracy_score(y_test, pred_with)

print("\n📊 ABLATION STUDY RESULTS")
print("Without anomaly score:", acc_without)
print("With anomaly score   :", acc_with)

# -------------------------------
# STEP 7.8: Feature Importance
# -------------------------------
importances = rf_with.feature_importances_
feature_names = X_with.columns

# Sort features
indices = np.argsort(importances)[::-1]

print("\n🔥 Feature Importance Ranking:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# -------------------------------
# STEP 7.9: Plot Feature Importance
# -------------------------------
plt.figure()
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

print("✅ STEP 7 COMPLETED")