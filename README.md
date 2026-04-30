Self healing Network by 
Shambhavi Pahade 24BYB1091
Dhrubajyoti Paul 24BYB1054

#  Self-Healing Network using Predictive Failure Analysis

Predict network failures before they happen, explain why, and automatically trigger recovery actions — all in one pipeline.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/self-healing-network.git
cd self-healing-network/self_healing_network

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib

# 3. Run the full pipeline
python main.py

# 4. (Optional) Launch the dashboard
streamlit run app.py
```

---

## What It Does

| Step | Description |
|---|---|
| Data | Merges 3 network datasets (NSL-KDD, CICIDS, Server Health) into 15,000 rows |
| Models | Trains Random Forest, XGBoost, SVM, Logistic Regression |
| Reasoning | Explains failures using feature importance (top contributing factors) |
| Healing | Triggers automated recovery actions based on root cause |

---

## ⚙️ Self-Healing Logic

```
High latency      →  Reroute traffic
High packet loss  →  Restart node
Low bandwidth     →  Switch to backup server
High CPU/Memory   →  Migrate workload
SYN flood         →  Activate DDoS protection
```

---

## 🖥️ Dashboard Preview

The Streamlit dashboard lets you:
- Select any model and see **only that model's** results
- Adjust network metrics with sliders for live prediction
- View feature importance, confusion matrix, and healing simulation

---

##  Project Structure

```
self_healing_network/
├── data/                  # Auto-generated datasets
├── src/
│   ├── preprocessing.py   # Data loading & merging
│   ├── train.py           # Model training
│   ├── evaluate.py        # Metrics & plots
│   └── self_healing.py    # Reasoning & recovery
├── main.py                # Run the full pipeline
├── app.py                 # Streamlit dashboard
└── generate_data.py       # Dataset generator
```

---

##  Tech Stack

`pandas` · `numpy` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `Streamlit`
