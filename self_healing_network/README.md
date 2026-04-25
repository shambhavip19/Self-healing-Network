# 🌐 Self-Healing Network using Predictive Failure Analysis

> **Predict network failures before they happen. Understand why. Fix them automatically.**

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [How Failure Prediction Works](#how-failure-prediction-works)
3. [How Failure Reasoning Works](#how-failure-reasoning-works)
4. [Self-Healing Logic](#self-healing-logic)
5. [Project Structure](#project-structure)
6. [Datasets Used](#datasets-used)
7. [Models Implemented](#models-implemented)
8. [Steps to Run](#steps-to-run)
9. [Sample Output](#sample-output)
10. [Streamlit Dashboard](#streamlit-dashboard)

---

## Project Overview

### Simple Explanation
Imagine your network is like a highway. Sometimes the highway gets congested, accidents happen, or a road breaks — causing everything to slow down or fail. This project watches the highway 24/7, **predicts** problems before they happen, explains **why** they're about to happen, and **automatically fixes** them.

### Technical Explanation
The system combines three publicly available network performance datasets, merges them into a unified dataset, and trains four ML classifiers to identify failure conditions from network telemetry signals (latency, packet loss, bandwidth, jitter, etc.). Feature importances from tree-based models power a reasoning engine that maps top contributing features to human-readable explanations and triggers appropriate recovery actions.

---

## How Failure Prediction Works

```
Raw Network Metrics
    │
    ▼
Feature Engineering  ←── latency, packet_loss, bandwidth, jitter, throughput,
    │                     cpu_usage, memory_usage, syn_flags, retransmissions …
    ▼
StandardScaler (normalise to mean=0, std=1)
    │
    ▼
ML Classifier  ──► Failure (1) or Normal (0)
    │
    ▼
Confidence Score (from predict_proba)
```

**Four classifiers are trained and compared:**

| Model | Strengths |
|---|---|
| Random Forest | Handles non-linear boundaries; outputs feature importance |
| XGBoost | State-of-the-art gradient boosting; excellent recall |
| SVM (RBF) | Effective in high-dimensional space |
| Logistic Regression | Fast; interpretable baseline |

The best model (by F1-score) is selected for the self-healing engine.

---

## How Failure Reasoning Works

When a failure is predicted, the system does **not** just say "failure." It explains *why.*

### Step-by-step:

1. **Extract feature importances** from the trained XGBoost or Random Forest model.  
   Each feature gets a score indicating how much it contributed to the model's decision boundary.

2. **Rank features** by importance (descending).

3. **Filter for anomalous values** — compare each feature's actual value to a calibrated threshold:
   - `latency > 80 ms` → anomalous
   - `packet_loss > 0.05` → anomalous
   - `bandwidth < 30 Mbps` → anomalous
   - etc.

4. **Map to human-readable labels:**

   | Feature | Label |
   |---|---|
   | `latency` | High latency |
   | `packet_loss` | High packet loss |
   | `bandwidth` | Low bandwidth |
   | `jitter` | High jitter |
   | `cpu_usage` | High CPU usage |
   | `syn_flag_count` | SYN flood detected |
   | … | … |

5. **Display top 2–3 root causes** with actual metric values.

### Example output:
```
Failure predicted due to:
  1. High packet loss   → current value: 0.43 (ratio)
  2. High latency       → current value: 187.5 ms
  3. Low bandwidth      → current value: 8.2 Mbps
```

---

## Self-Healing Logic

```
IF failure predicted:
    ├── High latency        → "Rerouting traffic to a lower-latency path"
    ├── High packet_loss    → "Restarting affected network node"
    ├── Low bandwidth       → "Switching to backup server / uplink"
    ├── High jitter         → "Enabling traffic shaping & QoS policies"
    ├── High cpu_usage      → "Migrating workload to spare compute node"
    ├── High memory_usage   → "Triggering memory reclamation & cache flush"
    ├── High syn_flag_count → "Activating SYN-cookie protection (DDoS)"
    └── Low uptime_pct      → "Initiating automatic failover to standby node"
ELSE:
    └── "✅ Network is stable – No action required"
```

Multiple actions can be triggered in parallel (one per distinct root cause).

---

## Project Structure

```
self_healing_network/
├── data/
│   ├── nsl_kdd_style.csv      ← NSL-KDD-inspired intrusion/anomaly data
│   ├── cicids_style.csv       ← CICIDS-inspired traffic flow data
│   └── server_health.csv      ← Server node health metrics
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       ← Data loading, cleaning, merging, scaling
│   ├── train.py               ← Model definitions and training
│   ├── evaluate.py            ← Metrics, confusion matrices, plots
│   └── self_healing.py        ← Feature reasoning + recovery actions
│
├── plots/                     ← Auto-generated charts (created at runtime)
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
│
├── generate_data.py           ← Generates the three synthetic datasets
├── main.py                    ← Full pipeline orchestrator
├── app.py                     ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Datasets Used

The project mimics three real-world publicly available datasets:

| # | Inspired By | What It Captures | Rows |
|---|---|---|---|
| 1 | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | Network intrusions, SYN floods, anomalies | 5,000 |
| 2 | [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) | Flow-level traffic (latency, bandwidth, jitter) | 5,000 |
| 3 | Server health logs | CPU, memory, disk I/O, throughput | 5,000 |

**Merged total: 15,000 rows · 19 features · binary `failure` label**

### How merging works:
- All three datasets share a common core: `latency`, `packet_loss`, `bandwidth`, `jitter`, `throughput`
- Dataset-specific features (e.g. `cpu_usage`, `syn_flag_count`) are included with `0` fill for datasets that don't have them
- The `failure` column is aligned (1 = failure, 0 = normal) across all sources

---

## Models Implemented

```python
RandomForestClassifier(n_estimators=150, class_weight="balanced")
SVC(kernel="rbf", C=10, probability=True, class_weight="balanced")
XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1)
LogisticRegression(max_iter=1000, class_weight="balanced")
```

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (heatmap)
- Feature Importance (bar chart)

---

## Steps to Run

### 1. Clone / download the project
```bash
cd self_healing_network
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Generate data manually
```bash
python generate_data.py
```
> `main.py` calls this automatically if the CSV files are missing.

### 4. Run the full pipeline
```bash
python main.py
```

Outputs:
- Console: training logs, metrics table, self-healing simulation
- `plots/model_comparison.png`
- `plots/confusion_matrices.png`
- `plots/feature_importance.png`

### 5. (Optional) Launch Streamlit dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## Sample Output

```
════════════════════════════════════════════════════════════
   🌐  SELF-HEALING NETWORK – Predictive Failure Analysis
════════════════════════════════════════════════════════════
  Merged dataset    : 15,000 rows, 19 cols
  Failure rate      : 35.0%

  Training models …
    ► Random Forest          done ✓
    ► SVM                    done ✓
    ► XGBoost                done ✓
    ► Logistic Regression    done ✓

  ──────────────────────────────────────────────────────────
  Model                   Accuracy  Precision   Recall  F1-Score
  ──────────────────────────────────────────────────────────
  Random Forest             1.0000     1.0000   1.0000    1.0000
  SVM                       1.0000     1.0000   1.0000    1.0000
  XGBoost                   1.0000     1.0000   1.0000    1.0000
  Logistic Regression       1.0000     1.0000   1.0000    1.0000
  ──────────────────────────────────────────────────────────
  🏆  Best model by F1-Score : Random Forest

════════════════════════════════════════════════════════════
  ⚠️   NETWORK FAILURE PREDICTED
       Failure probability : 100.0%
════════════════════════════════════════════════════════════

  🔍  Root-cause analysis (top contributing factors):

    1. High packet loss  →  current value: 0.431 (ratio)
    2. High latency      →  current value: 187.5 ms
    3. Low bandwidth     →  current value: 8.2 Mbps

  🛠️   Self-healing actions triggered:

    🔁  Restarting affected network node
    🔀  Rerouting traffic to a lower-latency path
    🖥️   Switching to backup server / uplink
════════════════════════════════════════════════════════════

════════════════════════════════════════════════════════════
  ✅  NETWORK IS STABLE – No action required
       Normal confidence : 99.8%
════════════════════════════════════════════════════════════
```

---

## Streamlit Dashboard

The optional `app.py` provides a web-based UI with four tabs:

| Tab | Content |
|---|---|
| 📊 Model Performance | Metrics table + comparison bar chart + confusion matrices |
| 📈 Feature Importance | Visual importance charts for RF and XGBoost |
| 🔮 Live Prediction | Interactive sliders → instant prediction + healing actions |
| 🛠️ Self-Healing Demo | Batch simulation on random test samples |

```bash
streamlit run app.py
```

---

## Key Concepts

| Term | Meaning |
|---|---|
| **Feature importance** | How much each input feature influenced the model's decision |
| **Self-healing** | Automated corrective action triggered by predicted failure |
| **Class imbalance** | When failures are rarer than normal events; handled with `class_weight="balanced"` |
| **StandardScaler** | Normalises features to zero mean and unit variance for SVM/LR |
| **Stratified split** | Train/test split that preserves the failure ratio in both sets |

---

*Built with pandas · numpy · scikit-learn · XGBoost · matplotlib · seaborn · Streamlit*
