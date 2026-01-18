# Quant Trading System

A complete end-to-end **quantitative trading research and backtesting system** built in Python.  
This project demonstrates a realistic quant workflow including data engineering, feature engineering, regime detection using HMM, rule-based strategy implementation, machine learning enhancement, and performance evaluation.

---

## 📌 Project Overview

This project implements a full quantitative trading pipeline:

1. **Data Engineering**
   - Intraday OHLCV data loading and cleaning
   - Timestamp alignment and deduplication

2. **Feature Engineering**
   - Returns and volatility features
   - EMA-based momentum features
   - Volume normalization (Z-score)
   - High–low range
   - Sentiment pressure proxy
   - Futures basis proxy
   - Options Greek proxies (Delta, Gamma, Vega, Theta)
   - Time-based features

3. **Market Regime Detection**
   - Hidden Markov Model (HMM) with 3 regimes:
     - Uptrend
     - Sideways
     - Downtrend
   - Regime used as a **strategy filter**

4. **Trading Strategy**
   - EMA(5/15) crossover strategy
   - Regime-aware:
     - Long only in Uptrend
     - Short only in Downtrend
     - No trades in Sideways

5. **Machine Learning Enhancement**
   - Trade quality prediction using:
     - Gradient Boosting (XGBoost-style)
     - LSTM (sequence of last 10 candles)
   - ML models used as **execution filters**
   - Comparison between:
     - Baseline strategy
     - ML-filtered strategy (GB)
     - ML-filtered strategy (LSTM)

6. **Backtesting & Evaluation**
   - Equity curves
   - Drawdown
   - Sharpe, Sortino, Calmar ratios
   - Win rate, profit factor, trade count
   - Full comparison saved to CSV

---

## ⚙️ Installation Instructions

### 1. Clone the repository
git clone https://github.com/lakshaysharma2004/quant-trading-system.git
cd quant-trading-system

### 2. Create and activate virtual environment (recommended)
python -m venv quant_env
quant_env\Scripts\activate   # Windows

### 3. Install dependencies
pip install -r requirements.txt

▶️ How to Run

Run the notebooks in order:

01_data_acquisition.ipynb
02_data_cleaning.ipynb
03_feature_engineering.ipynb
04_regime_detection.ipynb
05_strategy_backtest.ipynb
06_ml_models.ipynb

Each notebook builds on the previous one.

🗂 Project Structure Explanation
quant-trading-system/
│
├── data/
│   ├── spot_5min_features.csv
│   └── spot_5min_features_with_regime.csv
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_regime_detection.ipynb
│   ├── 05_strategy_backtest.ipynb
│   └── 06_ml_models.ipynb
│
├── src/
│   └── metrics.py         # Sharpe, Sortino, Drawdown, Calmar etc.
│
├── models/
│   ├── hmm_model.pkl
│   ├── trade_filter_gb.pkl
│   └── trade_filter_lstm.keras
│
├── plots/
│   ├── regime_overlay.png
│   └── equity_baseline_vs_ml.png
│
├── results/
│   ├── baseline_trades.csv
│   ├── gb_trades.csv
│   ├── lstm_trades.csv
│   └── performance_comparison.csv
│
├── requirements.txt
└── README.md

📊 Key Results Summary

The baseline EMA strategy performs poorly on the limited dataset.

Regime filtering reduces bad trades and drawdown.

Machine learning filters (GB & LSTM):

Reduce drawdowns

Reduce trade frequency

Improve risk-adjusted behavior

Due to limited historical depth and small number of trades:

ML results are exploratory

System design and pipeline quality are the main focus

The architecture is fully general and can be extended to longer historical datasets or live APIs.

👤 Author

Lakshay Sharma
B.Tech CSE (AIML)
Focus: Quantitative Systems, Data Science, Machine Learning