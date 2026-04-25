
# 🔧 VibeGuard AI – Smart Bearing Health Monitoring System

## 📄 Paper Title

**VibeGuard AI: A Hybrid Machine Learning Framework for Predictive Maintenance and Bearing Fault Detection with Health Monitoring**


## 🚀 Overview

VibeGuard AI is an intelligent predictive maintenance system designed to detect bearing faults using hybrid machine learning techniques.
The system analyzes vibration signal data to identify anomalies, predict failures, and provide early warnings before breakdown occurs.


## 🎯 Key Features



## 🧠 Technologies Used



## � Project Structure


project/

│

├── app.py                  # Streamlit application

├── models/

├── .gitignore              # Ignored files

## 🔗 Live demo

Try the live interactive demo here:

[Live demo placeholder] - https://your-live-demo-url.example

Replace the URL above with your deployed Streamlit/Flask/Streamlit Cloud link.

---

## ✨ What's new in this README

- A concise project elevator pitch so visitors immediately understand what this repo does.
- A prominent "Live demo" link you can replace with your deployment URL.
- Quick start commands to run the project locally (Streamlit instructions included).
- Where to find trained models, sample data and results inside the repo.

---

## Quick Start (Run locally)

1. Create and activate a Python virtual environment (recommended):

```powershell
# create venv
python -m venv .venv
# activate (PowerShell)
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app (if `app.py` or `src/visualization.py` exposes a Streamlit UI):

```powershell
streamlit run app.py
# or if the entrypoint is different: streamlit run src/visualization.py
```

Open the displayed URL (usually http://localhost:8501) in your browser.

---

## Project highlights

- Fault classification (Normal / Warning / Fault)
- Early warning detection and health score (0–100)
- Hybrid approach: Isolation Forest for anomaly detection + classical classifiers
- Pre-trained models included in `models/` and `src/models/`

## Where to look next

- Data: `dataset/` (raw vibration files) and `data/features_dataset.csv` (extracted features)
- Models: `models/` and `src/models/` (pickled scikit-learn models)
- Results & plots: `results/` and `src/results/plots/`
- Scripts: `src/` (feature extraction, training, detection, visualization)

---

## Contributing

Feel free to open issues or pull requests. A few ideas:

- Add a deployment script and CI workflow (GitHub Actions) to deploy the Streamlit app.
- Add end-to-end tests or a small demo dataset for quick verification.
- Provide a Dockerfile for reproducible deployment.

---

## License & Contact

This repository currently doesn't specify a license file. Add a `LICENSE` if you want to make usage terms explicit.

Maintainer: girish50

---

If you want, I can also:

- Add a ready-to-copy GitHub Actions workflow to auto-deploy the Streamlit app to Streamlit Cloud or an Azure Web App.
- Replace the placeholder live demo link with your actual deployment URL.

│   ├── scaler.pkl          # Feature scaler

│   ├── isolation_model.pkl # Anomaly detection model

│

├── src/                    # Training and processing scripts

├── requirements.txt        # Dependencies

├── README.md               # Project documentation

├── .gitignore              # Ignored files
