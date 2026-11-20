# DIGITAL PAYMENT FRAUD DETECTION USING DATA MINING AND VISUALIZATION

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-white?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58%2B-white?style=flat&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)](https://opensource.org/licenses/MIT)
[![Design](https://img.shields.io/badge/Theme-SpaceX%20Minimalist-white?style=flat)](#)

An enterprise-grade, SpaceX-inspired financial surveillance and machine learning platform engineered to detect fraudulent transactions in high-volume digital payment networks using advanced data mining, balance discrepancy telemetry, and supervised/unsupervised machine learning ensembles.

---

## 🚀 Mission Overview

Digital payment networks process millions of transactions every hour. Traditional static threshold rules (e.g. flagging single transactions exceeding $200,000) fail catastrophically: in the PaySim benchmark dataset of **6,362,620 transactions**, rule-based filters caught a mere **16 out of 8,213** verified fraud attacks—a **99.81% failure rate**.

This platform delivers an autonomous threat intelligence layer that:
1. Profiles and analyzes the entire 6.36M PaySim mobile money dataset.
2. Formulates domain-driven balance delta discrepancies (`errorBalanceOrig`, `errorBalanceDest`) and mule wallet indicators.
3. Overcomes extreme class imbalance (1 fraud per 774 genuine transactions) using strictly non-leaking SMOTE oversampling.
4. Benchmarks **Logistic Regression**, **Decision Trees**, **Random Forests**, and **Isolation Forest Anomaly Detection**.
5. Surfaces interactive SpaceX-themed telemetry, real-time transaction prediction, and human-readable indicator explainability.

---

## 📊 Dataset & Mining Insights

The system uses the financial research dataset **PaySim**, simulating mobile financial transfers based on real-world logs from a multinational cellular financial service provider.

- **Total Operations:** 6,362,620
- **Confirmed Fraudulent Exfiltrations:** 8,213 (0.1291%)
- **Verified Genuine Transactions:** 6,354,407 (99.8709%)
- **Rule Flagged Fraud:** 16 (0.19% rule catch rate)

### Core Discoveries
1. **Transaction Type Vulnerability:** Fraud occurs **exclusively** within `TRANSFER` (4,097 cases) and `CASH_OUT` (4,116 cases). Zero fraudulent records exist in `PAYMENT`, `CASH_IN`, or `DEBIT`.
2. **Account Liquidation Signature:** Attackers drain 100% of the origin account balance to exactly $0.00 in over 96% of successful fraudulent transfers.
3. **Mule Wallet Inconsistencies:** Money is routed to recipient accounts with $0 prior history, with recipient balances failing to show legal ledger reconciliations.

---

## 🧠 Machine Learning Benchmark

All supervised models are trained on SMOTE-balanced training partitions while the test partition preserves the real-world imbalanced distribution with **zero data leakage**.

| Algorithm | Paradigm | Accuracy | Precision | Recall (Catch Rate) | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | Supervised Ensemble | **99.95%** | **99.82%** | **99.57%** | **0.9970** | **0.9990** |
| **Decision Tree** | Supervised Tree | 99.94% | 99.63% | 99.63% | 0.9963 | 0.9980 |
| **Logistic Regression** | Linear (Balanced) | 96.93% | 72.73% | 98.05% | 0.8351 | 0.9956 |
| **Isolation Forest** | Unsupervised Anomaly | 88.80% | 28.42% | 27.08% | 0.2773 | 0.7442 |

> **Note on Metric Prioritization:** In severe class imbalance, plain **Accuracy** is completely misleading (a naive classifier predicting 100% genuine achieves 99.87% accuracy while intercepting $0 of stolen funds). We prioritize **Recall** and **F1-Score** to maximize threat interception while controlling false alarms.

---

## 🎨 SpaceX UI/UX Design System

The application strictly enforces SpaceX's minimalist industrial visual identity:
- **Pure Black Foundation (`#000000`):** True black canvas with no extraneous visual noise.
- **High-Contrast Monochrome Typography:** White display headers with wide tracking (`letter-spacing: 0.12em`).
- **Minimal Industrial Framing:** Thin grey architectural borders (`1px solid #1E1E1E`), no colorful cards or rounded shadows.
- **Ghost-Outlined Pill Buttons:** Clean bordered pills that invert on hover.
- **Dynamic Dark Plotly Telemetry:** Pure black charts with subtle gridlines and crisp high-contrast data series.

---

## 🗂️ Project Directory Structure

```
digital_payment_fraud_detection/
├── .gitignore                    # Excludes 500MB raw dataset & caches
├── render.yaml                   # Infrastructure-as-code for Render
├── requirements.txt              # Production dependency specifications
├── README.md                     # Comprehensive technical documentation
├── app.py                        # Streamlit SpaceX Mission Control App
├── .streamlit/
│   └── config.toml               # Streamlit theme & headless server config
├── data/
│   ├── dataset_summary.json      # Full 6.36M transaction macro metrics
│   └── paysim_sample.csv         # Stratified 100K transaction dataset (100% frauds)
├── models/
│   ├── best_model.pkl            # Champion Random Forest model
│   ├── decision_tree.pkl         # Decision Tree model
│   ├── random_forest.pkl         # Random Forest ensemble
│   ├── logistic_regression.pkl   # Logistic Regression model
│   ├── isolation_forest.pkl      # Isolation Forest Anomaly Detector
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── feature_columns.json      # Ordered feature definitions
│   └── model_metrics.json        # Precomputed benchmark evaluations & ROC
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data ingestion, validation & feature delta math
│   ├── models.py                 # Multi-algorithm training & evaluation
│   ├── predict.py                # Live inference & explainable risk indicators
│   ├── visualizations.py         # SpaceX-themed Plotly charts
│   └── utils.py                  # Formatters, CSS injector & test scenarios
└── tests/
    └── test_fraud_detector.py    # Automated pytest validation suite
```

---

## ⚙️ Quickstart & Local Execution

### 1. Clone the repository
```bash
git clone https://github.com/gt1432/digital-payment-fraud-detection.git
cd digital-payment-fraud-detection
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run automated test suite
```bash
python -m pytest tests/
```

### 5. Launch SpaceX Mission Control Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment on Render

This project is fully optimized for continuous deployment on [Render](https://render.com) Web Services.

1. Create a new **Web Service** on Render connected to your GitHub repository.
2. Select runtime: **Python 3**.
3. Set **Build Command**:
   ```bash
   pip install -r requirements.txt
   ```
4. Set **Start Command**:
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Deploy service. The application will start and serve traffic on the assigned Render URL.

---

## 🔒 Security & Data Integrity

- No API keys, credentials, or secrets are required or stored in this codebase.
- Raw 500MB CSV files are excluded from git tracking via `.gitignore` to prevent repository bloat and ensure fast deployments.
- All inferences are explicitly distinguished from legal determinations of confirmed fraud.

---

## 📜 License
MIT License. Open for educational, academic, and research applications.
