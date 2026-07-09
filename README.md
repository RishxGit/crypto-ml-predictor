# Crypto Price Movement Predictor — ML Classification Pipeline
### Predicts whether a cryptocurrency will appreciate or decline over a 1-year horizon

🚀 **[Live Demo](https://crypto-ml-predictor.streamlit.app/)**

---

## Overview

A production-grade binary classification pipeline that predicts whether a cryptocurrency will go **UP (1)** or **DOWN (0)** over a rolling 1-year timeline. The system navigates one of the hardest problems in quantitative finance — making confident, low-noise directional calls on an asset class with extreme volatility and structural class imbalance.

This is **Part 3** of an ongoing crypto intelligence series built on the same dataset:

| Part | Project | Tech |
|---|---|---|
| 1 | SQL Market Intelligence | MySQL |
| 2 | Python EDA + Feature Engineering | Pandas · NumPy · Matplotlib · Seaborn |
| 3 | ML Price Movement Predictor (this) | scikit-learn · XGBoost · Streamlit |

---

## The Core Challenge — Class Imbalance

The dataset exhibits severe structural bias:
- **86% Class 0 (Losers)** — coins that declined over the year
- **14% Class 1 (Winners)** — coins that appreciated over the year

Standard classifiers collapse under this imbalance, achieving high accuracy by simply predicting everything as 0 — the **Accuracy Paradox**. This project required customized sample routing, imbalance-aware algorithms, and a shifted decision threshold to produce genuinely useful signals.

---

## Target Engineering

```
Target (y) = 1 if price_change_percentage_1y > 0
             0 otherwise
```

The `price_change_percentage_1y` column was used to create the target, then immediately removed from the feature matrix to prevent data leakage.

---

## Feature Matrix — 12 Optimized Features

All raw dollar-scale features were eliminated to prevent distortion from absolute price differences between assets like Bitcoin ($92k) and low-priced utility tokens.

| Feature | Category | Why It Matters |
|---|---|---|
| `volume_to_marketcap` | Engineered | Liquidity velocity — how actively traded relative to size |
| `market_cap_rank` | Market Size | Relative position in the market hierarchy |
| `market_cap` | Market Size | Total capitalization |
| `total_volume` | Market Size | 24h trading activity |
| `circulating_supply` | Supply | Active coins in market |
| `supply_utilization` | Supply | Circulating / total supply ratio — inflation risk proxy |
| `ath_change_percentage` | Momentum | Distance from all-time high — recovery potential |
| `atl_change_percentage` | Momentum | Distance from all-time low — structural health |
| `price_change_percentage_1h` | Delta | Short-term directional signal |
| `price_change_percentage_24h` | Delta | Daily momentum |
| `price_change_percentage_7d` | Delta | Weekly trend |
| `price_change_percentage_30d` | Delta | Medium-term momentum — strongest single predictor |

---

## Iterative Model Development

**Train/Test Split:** 80/20 (800 training examples, 200 held-out test examples)
**Validation:** 5-Fold Stratified Cross-Validation throughout

### Model 1 — Baseline Logistic Regression
Pipeline: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`
- Cross-validation accuracy: 0.865
- Class 1 Precision: 0.50 | Class 1 Recall: **3.7%**
- **Finding:** Classic Accuracy Paradox — blindly predicted 0 for almost everything. High accuracy, zero analytical value.

### Model 2 — Balanced Logistic Regression
Added `class_weight='balanced'`
- Global Accuracy: 0.69 | Class 1 Precision: 0.23 | Class 1 Recall: **56.0%**
- **Finding:** Fixed the bias problem but generated a 77% false alarm rate on buy signals — completely unusable for any real screening application.

### Model 3 — Random Forest Classifier
Pipeline: `SimpleImputer` → `RandomForestClassifier(class_weight='balanced_subsample')`
- Cross-validation F1: 0.1895
- Class 1 Precision: 0.50 | Class 1 Recall: **4.0%**
- **Finding:** Severely overfitted on structural data cuts. Standard tree ensembles struggle with heavy class imbalance even with subsample weights.

### Model 4 — Baseline XGBoost
Pipeline: `SimpleImputer` → `XGBClassifier(scale_pos_weight=imbalance_ratio)`
- Global Accuracy: 0.86 | Precision: 0.50 | Recall: **33.0%** | F1: 0.40
- **Finding:** First real breakthrough. Gradient boosting's sequential error-correction stabilized the framework significantly.

### Model 5 — Optimized XGBoost + Threshold Adjustment (Final)
Pipeline: `GridSearchCV` tuning → Decision threshold shifted from 0.50 → **0.70**
- **Global Accuracy: 88.0%**
- **Class 1 Precision: 60.0%**
- **Class 1 Recall: 33.0%**
- **Class 1 F1: 43.0%**

---

## Final Architecture — Conservative High-Conviction Filter

The 70% confidence threshold transforms the model from a noisy classifier into a **selective screening tool**. By requiring high conviction before flagging a coin as bullish, the system filters out 67% of market noise to deliver clean, high-confidence upward signals.

**The tradeoff is explicit and intentional:**
- High precision (60%) over high recall (33%)
- Miss some winners to avoid flooding output with false positives
- In volatile crypto markets, false buy signals are more costly than missed opportunities

### Top Predictive Signals (Feature Importance)
1. **price_change_percentage_30d** — Medium-term momentum is the strongest consolidation signal
2. **atl_change_percentage** — Structural health relative to historical valuation floor
3. **supply_utilization + volume_to_marketcap** — Relative ratios carry far more signal than absolute inventory sizes

---

## Tech Stack

```
Python 3.11
scikit-learn    — Pipeline, SimpleImputer, StandardScaler, GridSearchCV
XGBoost         — Gradient boosted classification
pandas + numpy  — Data manipulation
joblib          — Model serialization
Streamlit       — Production web interface
Matplotlib      — Feature importance visualization
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/RishxGit/crypto-ml-predictor
cd crypto-ml-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Loading the Production Model

```python
import joblib
import pandas as pd

# Load
loaded_model = joblib.load('crypto_xgb_model.joblib')
best_pipeline = loaded_model.best_estimator_

# Engineer the custom ratio
df_new['volume_to_marketcap'] = df_new['total_volume'] / df_new['market_cap']

X_new = df_new[[
    'market_cap_rank', 'market_cap', 'total_volume', 'circulating_supply',
    'ath_change_percentage', 'atl_change_percentage', 'price_change_percentage_24h',
    'price_change_percentage_1h', 'price_change_percentage_7d', 'price_change_percentage_30d',
    'supply_utilization', 'volume_to_marketcap'
]]

# Apply 70% confidence threshold
probabilities = best_pipeline.predict_proba(X_new)[:, 1]
predictions = (probabilities >= 0.70).astype(int)
```

---

## Repository Structure

```
crypto-ml-predictor/
├── app.py                        — Streamlit production interface
├── crypto_xgb_model.joblib       — Serialized production pipeline
├── crypto_predictor.ipynb        — Full training + evaluation notebook
├── crypto_top1000_dataset.csv    — Source dataset (CoinGecko API)
├── requirements.txt              — Dependencies
└── README.md
```

---

## Author
[GitHub](https://github.com/RishxGit) 

---

*Dataset: CoinGecko API via Kaggle — December 2024 snapshot. Not financial advice. Built for educational and portfolio purposes only.*
EOF
