# ================================
# STEP 0: Start Message
# ================================
print("🚀 Feature Extraction Started...")

# ================================
# STEP 1: Import Libraries
# ================================
import numpy as np
import os
import pandas as pd
from scipy.stats import kurtosis, skew

# ================================
# STEP 2: Set Dataset Path
# ================================
folder_path = "../dataset"   # relative path

# ================================
# STEP 3: Check Dataset
# ================================
print("\n📂 Checking dataset path...")
print("Path exists:", os.path.exists(folder_path))

files = sorted(os.listdir(folder_path))
print("Total files found:", len(files))
print("Sample files:", files[:5])

# ================================
# STEP 4: Feature Functions
# ================================
def time_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sqrt(np.mean(signal**2)),   # RMS
        kurtosis(signal),
        skew(signal)
    ]

def freq_features(signal):
    fft = np.fft.fft(signal)
    mag = np.abs(fft)

    return [
        np.mean(mag),
        np.max(mag),
        np.std(mag)
    ]

# ================================
# STEP 5: Process Files
# ================================
X = []
file_names = []
error_files = []

print("\n⚙️ Processing files...")

for i, file in enumerate(files):
    file_path = os.path.join(folder_path, file)

    # 🔥 Skip folders
    if not os.path.isfile(file_path):
        continue

    try:
        data = np.loadtxt(file_path)
        data = data.flatten()

        

        if len(data.shape) > 1:
            data = data[:, 0]
        
        # Skip empty files
        if len(data) == 0:
            error_files.append(file)
            continue

        t_feat = time_features(data)
        f_feat = freq_features(data)

        features = t_feat + f_feat

        X.append(features)
        file_names.append(file)

        # 🔥 Progress print
        if i % 500 == 0:
            print(f"Processed {i} files...")

    except Exception as e:
        error_files.append(file)

# ================================
# STEP 6: Convert to DataFrame
# ================================
columns = [
    "mean", "std", "max", "min", "rms",
    "kurtosis", "skew",
    "freq_mean", "freq_max", "freq_std"
]

df = pd.DataFrame(X, columns=columns)
df["file_name"] = file_names

print("\n✅ Dataset shape:", df.shape)

# ================================
# STEP 7: Save Dataset
# ================================
os.makedirs("../data", exist_ok=True)

save_path = "../data/features_dataset.csv"
df.to_csv(save_path, index=False)

print(f"💾 Feature dataset saved at: {save_path}")

# ================================
# STEP 8: Error Report
# ================================
print("\n⚠️ Files skipped due to errors:", len(error_files))

if len(error_files) > 0:
    print("Example error files:", error_files[:5])

print("\n🎉 Feature Extraction Completed Successfully!")