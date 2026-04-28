# ================================
# FINAL: COMPLETE CLASSIFICATION PIPELINE
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
os.makedirs("models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

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

joblib.dump(iso, "models/isolation_model.pkl")
print("💾 Isolation model saved!")

df["anomaly_score"] = iso.decision_function(X_base)

# -------------------------------
# LABEL CREATION (IMPORTANT)
# -------------------------------
# -------------------------------
# LABEL CREATION (DATA-DRIVEN)
# -------------------------------
# Lower anomaly_score = more anomalous (fault-like)
threshold = np.percentile(df["anomaly_score"], 30)
df["label"] = (df["anomaly_score"] < threshold).astype(int)

# Check distribution
print("\nLabel distribution (overall):")
print(df["label"].value_counts())
# -------------------------------
# PREPARE DATA
# -------------------------------
X = df.drop(columns=["file_name", "label", "anomaly_score"])
y = df["label"]

# -------------------------------
# TIME-AWARE SPLIT (CORRECT)
# -------------------------------
split = int(0.8 * len(df))

X_train = X.iloc[:split]
y_train = y.iloc[:split]

X_test = X.iloc[split:]
y_test = y.iloc[split:]

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain distribution:")
print(y_train.value_counts())

print("\nTest distribution:")
print(y_test.value_counts())
# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
print("💾 Scaler saved!")

# ================================
# MODEL TRAINING (TIME-SERIES AWARE)
# ================================

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)

models = {}

# -------------------------------
# Logistic Regression
# -------------------------------
log_model = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    {"C": [0.1, 1, 10]},
    cv=tscv
)
log_model.fit(X_train, y_train)
models["Logistic"] = log_model.best_estimator_

# -------------------------------
# SVM
# -------------------------------
svm_model = GridSearchCV(
    SVC(probability=True, class_weight="balanced"),
    {"C": [0.5, 1, 2], "kernel": ["rbf"]},
    cv=tscv
)
svm_model.fit(X_train, y_train)
models["SVM"] = svm_model.best_estimator_

# -------------------------------
# Random Forest
# -------------------------------
rf_model = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    cv=tscv
)
rf_model.fit(X_train, y_train)
models["RandomForest"] = rf_model.best_estimator_
# ================================
# EVALUATION (FINAL CORRECT)
# ================================

results = []

for name, model in models.items():

    print("\n==============================")
    print(f"Model: {name}")

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # -------------------------------
    # SAFE CLASS EXTRACTION (IMPORTANT)
    # -------------------------------
    # get only class labels (exclude avg + accuracy)
    class_labels = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

    # pick FAULT class (usually '1')
    target_class = '1' if '1' in class_labels else class_labels[-1]

    # Save results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report[target_class]["precision"],
        "Recall": report[target_class]["recall"],
        "F1-Score": report[target_class]["f1-score"]
    })

    # -------------------------------
    # SAVE CONFUSION MATRIX PLOT
    # -------------------------------
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"results/plots/{name}_cm.png")
    plt.close()

    # Save model
    joblib.dump(model, f"models/{name}.pkl")

# ================================
# MODEL COMPARISON
# ================================
results_df = pd.DataFrame(results)

print("\n📊 Model Comparison:")
print(results_df)

# Save CSV
results_df.to_csv("results/model_comparison.csv", index=False)

# -------------------------------
# COMPARISON GRAPH
# -------------------------------
plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)

plt.savefig("results/plots/model_comparison.png")
plt.close()

# ================================
# BEST MODEL
# ================================
best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]

best_model = models[best_model_name]

joblib.dump(best_model, "models/best_model.pkl")

print(f"\n🏆 Best Model: {best_model_name}")
print("💾 Best model saved!")

print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")