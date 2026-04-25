
# ================================
# STEP 3: Install Required Libraries (RUN IN TERMINAL)
# pip install matplotlib
# ================================


# ================================
# STEP 3: Visualization of Features
# ================================

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# STEP 3.1: Load Feature Dataset
# -------------------------------
df = pd.read_csv("features_dataset.csv")

print("Dataset loaded:", df.shape)

# -------------------------------
# STEP 3.2: Plot RMS (VERY IMPORTANT)
# -------------------------------
plt.figure()
#plt.plot(df[4])   # RMS is 5th feature
plt.plot(df['4'])
plt.title("RMS over Time (Degradation Trend)")
plt.xlabel("Time (File Index)")
plt.ylabel("RMS Value")
plt.show()

# -------------------------------
# STEP 3.3: Plot Mean
# -------------------------------
plt.figure()
#plt.plot(df[0])
plt.plot(df['0'])
plt.title("Mean over Time")
plt.xlabel("Time")
plt.ylabel("Mean")
plt.show()

# -------------------------------
# STEP 3.4: Plot Frequency Feature
# -------------------------------
plt.figure()
#plt.plot(df[7])   # frequency mean
plt.plot(df['7'])
plt.title("Frequency Feature over Time")
plt.xlabel("Time")
plt.ylabel("Frequency Magnitude")
plt.show()