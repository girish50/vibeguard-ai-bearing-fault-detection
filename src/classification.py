# ================================
# FINAL: TUNED CLASSIFICATION PIPELINE
# ================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# -------------------------------
# CREATE FOLDERS
# -------------------------------
os.makedirs("../models", exist_ok=True)
os.makedirs("../results/plots", exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/features_dataset.csv")

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
X_base = df.drop(columns=["file_name"])

iso = IsolationForest(n_estimators=150, contamination=0.08, random_state=42)
iso.fit(X_base)

joblib.dump(iso, "../models/isolation_model.pkl")
print("💾 Isolation model saved!")

df["anomaly_score"] = iso.decision_function(X_base)

# -------------------------------
# LABEL CREATION (IMPROVED)
# -------------------------------
n = len(df)

labels = np.zeros(n)

# 🔥 improved labeling (fault near end)
labels[int(0.4 * n):] = 1

df["label"] = labels

# -------------------------------
# PREPARE DATA
# -------------------------------
X = df.drop(columns=["file_name", "label"])
y = df["label"]

# -------------------------------
# TIME-BASED SPLIT
# -------------------------------
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "../models/scaler.pkl")
print("💾 Scaler saved!")

# ================================
# 🔥 MODEL TUNING
# ================================

models = {}

# -------------------------------
# Logistic Regression (Tuned)
# -------------------------------
log_grid = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs"]
}

log_model = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    log_grid,
    cv=3
)
log_model.fit(X_train, y_train)
models["Logistic"] = log_model.best_estimator_

# -------------------------------
# SVM (Tuned)
# -------------------------------
svm_grid = {
    "C": [0.5, 1, 2],
    "kernel": ["rbf"]
}

svm_model = GridSearchCV(
    SVC(probability=True, class_weight="balanced"),
    svm_grid,
    cv=3
)
svm_model.fit(X_train, y_train)
models["SVM"] = svm_model.best_estimator_

# -------------------------------
# Random Forest (Tuned)
# -------------------------------
rf_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

rf_model = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    rf_grid,
    cv=3
)
rf_model.fit(X_train, y_train)
models["RandomForest"] = rf_model.best_estimator_

# ================================
# EVALUATION
# ================================

results = []

for name, model in models.items():

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append([name, acc])

    print(f"\n{name} Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = f"../results/plots/{name}_cm.png"
    plt.savefig(path)
    plt.close()

    # Save model
    joblib.dump(model, f"../models/{name}.pkl")

# -------------------------------
# MODEL COMPARISON
# -------------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

print("\n📊 Model Comparison:")
print(results_df)

# -------------------------------
# BEST MODEL
# -------------------------------
best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]

best_model = models[best_model_name]

joblib.dump(best_model, "../models/best_model.pkl")

print(f"\n🏆 Best Model: {best_model_name}")
print("💾 Best model saved!")

print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")