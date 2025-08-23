# stock-predictions-neural-networks


Project Title:

Against the Complexity-Performance Paradigm: A Systematic Comparison of Neural Network Stock Price Predictions with MDA-Driven Feature Selection

# Against the Complexity–Performance Paradigm: A Systematic Comparison of Neural Network Stock Price Predictions with MDA-Driven Feature Selection

**Student:** Abdul Rashid Omeni  
**Supervisor:** Prof. Frank Wang

## Overview
This repository contains the project corpus for the dissertation comparing multiple neural architectures for short-horizon stock price prediction (AAPL, 2015–2025). The pipeline enforces leakage-safe processing, **MDA-driven feature selection**, consistent time-based splits, and architecture-aware pruning with retraining.

### Architectures compared
- **Simple multivariate LSTM** (non-hybrid, strongest overall baseline)
- **TCN→LSTM** (hybrid; improved after pruning)
- **LSTM→Transformer** (hybrid; modest pruning gains)

### Key findings (holdout)
- Increasing architectural complexity **does not guarantee** better accuracy.
- The **simple LSTM** remained the most reliable non-hybrid baseline.
- **TCN→LSTM** benefited from pruning (~10–12% RMSE reduction in latest run).
- MDA consistently reduced features to ~7–8 (e.g., `Close`, `candle_ratio`, `volatility_63`, `OBV`, `EMA_12`, `MACD_line`, `tweet_volume`) **without loss**.

## Reproducibility summary
- **Split:** 85% development (train/val), 15% holdout (chronological).
- **Scaling:** MinMax on train; applied to val/holdout.
- **Window:** 60 timesteps; target `Close`.
- **Random seeds:** `numpy`, `tensorflow` set to 42.
- **MDA:** permutation importance on development data only.
- **Pruning:** width-reduction while preserving topology; safe partial weight copy where compatible; full retraining.

## How to run (local)
```bash
# 1) Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install
pip install --upgrade pip
pip install -r Code/requirements.txt

# 3) (Re)generate data
python Code/scripts/download_data.py --ticker AAPL --start 2015-01-01 --end 2025-05-30

# 4) Launch Streamlit app
streamlit run app.py
