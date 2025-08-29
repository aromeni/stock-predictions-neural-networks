# Challenging the Complexity-Performance Paradigm: Neural Network Architectures for Stock Price Prediction



## Abstract

This research systematically evaluates **nine neural network architectures** for financial time series prediction, revealing a fundamental complexity paradox: architectural sophistication inversely correlates with predictive performance. Through rigorous experimental comparison using Apple Inc. (AAPL) stock data (2015-2025) and Mean Decrease in Accuracy (MDA) feature selection, the study demonstrates that simple LSTM networks achieve superior performance (R² = 98.35%) compared to sophisticated alternatives including Transformers (83.25%) and hybrid architectures.

## Key Findings

### 1. Complexity Paradox Demonstrated
- **Simple LSTM + MDA**: 98.35% R², RMSE: $4.20
- **Transformer (properly implemented)**: 83.25% R², RMSE: $13.22
- **TCN-LSTM hybrid**: 61.98% R², RMSE: $20.10

### 2. Universal Feature Selection
Identical 8-feature set optimized performance across all architectures:
```
Close, Open, candle_ratio, volatility_63, OBV, tweet_volume, EMA_12, MACD_line
```

### 3. Computational Efficiency
- LSTM+MDA: 18.3 min training, 2.4ms inference
- Transformer: 127.8 min training, 8.9ms inference
- **8-12x better cost-effectiveness** for simple architectures



## Architecture Performance Rankings

| Rank | Architecture | R² (%) | RMSE ($) | Training Time | Inference (ms) |
|------|--------------|--------|----------|---------------|----------------|
| 1 | LSTM + MDA | **98.35** | 4.20 | 18.3 min | 2.4 |
| 2 | GRU + MDA | 97.81 | 4.82 | 14.2 min | 1.8 |
| 3 | LSTM-Transformer | 97.35 | 5.31 | 89.4 min | 6.7 |
| 4 | Simple LSTM | 94.59 | 5.75 | 12.8 min | 2.1 |
| 5 | TCN | 94.22 | 5.89 | 42.1 min | 3.2 |
| 6 | Complex LSTM | 92.65 | 8.54 | 95.2 min | 3.8 |
| 7 | CNN-LSTM | 87.35 | 11.59 | 31.7 min | 4.1 |
| 8 | Transformer | 83.25 | 13.22 | 127.8 min | 8.9 |
| 9 | TCN-LSTM | 61.98 | 20.10 | 156.4 min | 11.2 |



## Repository Structure
STOCK-PREDICTIONS-NEURAL-NETWORKS/
├── __pycache__/
├── .ipynb_checkpoints/
├── .venv/
├── approaches/
│   ├── __pycache__/
│   ├── approach_1.py          # Simple LSTM (Univariate)
│   ├── approach_2.py          # Complex LSTM (All Features)
│   ├── approach_3.py          # LSTM + MDA (Champion Model)
│   ├── approach_4.py          # GRU + MDA
│   ├── approach_5.py          # Temporal Convolutional Network (TCN)
│   ├── approach_6.py          # Transformer with Causal Attention
│   ├── approach_7.py          # CNN-LSTM Hybrid
│   ├── approach_8.py          # TCN-LSTM Hybrid
│   └── approach_9.py          # LSTM-Transformer Hybrid
├── data/
│   ├── raw/
│   │   ├── AAPL_2015_2025.csv
│   │   └── price_date.csv
│   ├── processed/
│   │   ├── daily_tweet_sentiments.csv
│   │  
│   
├── papers/
│   ├── literature_review/
│   │   ├── transformer_papers.pdf
│   │   ├── lstm_finance.pdf
│   │   └── feature_selection.pdf
│   └── references.bib
├── results-data/
│   ├── performance_metrics for approach
│   ├── architecture-aware pruning and compression results
│   ├── r-sqaure distribution chart
│   ├── rmse distribution chart
│   
├── sentiment_data_generator/
    ├── simulated_sentiments_advanced.ipynb│ 
│   
├── meeting_minutes.pdf
├── app.py
├── index.html
 ── user_guide.md
 ── requirements.tx
 ── utils.py
 ── README.md


## Quick Start

### Installation
```bash
git clone https://github.com/aromeni/stock-predictions-neural-networks.git
cd  stock-predictions-neural-networks
pip install -r requirements.txt
run streamlit run app.py  
goto http://localhost:8501/, select Select Prediction Approach fromthe drop down and click "Run Forecast" for each approach. All complete running for performance metrics to display.

```

### Reproduce Key Results
```bash
# Run champion LSTM+MDA model
# Compare all 9 architectures
# Generate performance visualizations
python run 

```

### MDA Feature Selection
```python
from src.features.mda_selector import MDAFeatureSelector

selector = MDAFeatureSelector(n_features=8)
X_selected = selector.fit_transform(X_train, y_train)
print(selector.selected_features_)
# ['Close', 'Open', 'candle_ratio', 'volatility_63', 'OBV', 'tweet_volume', 'EMA_12', 'MACD_line']
```

## Experimental Framework

### Data Specifications
- **Primary Asset**: AAPL (2015-2025)
- **Features**: 21 technical indicators + sentiment data
- **Sentiment Events**: 108,388 synthetic financial tweets
- **Train/Test Split**: 85% / 15% (time-series split)
- **Validation**: Walk-forward analysis across 6 temporal periods

### Model Architectures
1. **Simple LSTM** - Univariate baseline
2. **Complex LSTM** - 21 features, no selection
3. **LSTM + MDA** - 8 MDA-selected features
4. **GRU + MDA** - Gated recurrent alternative
5. **TCN** - Temporal convolutional networks
6. **Transformer** - Causal attention with financial adaptations
7. **CNN-LSTM** - Convolutional + recurrent hybrid
8. **TCN-LSTM** - Multi-scale temporal hybrid
9. **LSTM-Transformer** - Sequential + attention hybrid

### Evaluation Metrics
- **Primary**: R², RMSE, MAE, MAPE
- **Secondary**: Directional accuracy, Sharpe ratio
- **Efficiency**: Training time, inference latency, memory usage
- **Statistical**: Diebold-Mariano significance tests

## Key Contributions

### 1. Theoretical
- **Complexity Paradox Theory**: Formal framework explaining inverse complexity-performance relationship in financial domains
- **Domain-Specific Architecture Requirements**: Evidence that financial time series favor sequential over attention-based processing
- **Universal Feature Selection**: Architecture-agnostic optimal feature identification

### 2. Methodological
- **Systematic Comparison Framework**: Controlled experimental protocol for architectural evaluation
- **Architecture-Aware Pruning**: Model-specific compression strategies
- **Statistical Validation Protocol**: Rigorous significance testing for financial ML

### 3. Practical
- **Performance Benchmarks**: Prediction (98.35% R²)
- **Resource Optimization Guidelines**: Evidence-based efficiency recommendations


## Statistical Significance

All performance differences validated through:
- **Diebold-Mariano tests** (α = 0.05)
- **Bootstrap confidence intervals** (10,000 iterations)
- **Cross-temporal validation** (6 market periods)
- **Multiple random seed verification** (42 initializations)

Key statistical results:
- LSTM+MDA vs. Transformer: p < 0.001 (extremely significant)
- Complexity-performance correlation: r = -0.81, p < 0.001
- Feature universality: correlation = 0.96, p < 0.001

## Reproducibility

Complete reproducibility ensured through:
- **Fixed random seeds**: 42 across all experiments
- **Identical preprocessing**: Standardized data pipeline
- **Version pinning**: Exact dependency specifications


## Citation

```bibtex
@mastersthesis{omeni2025complexity,
  title={Challenging the Complexity-Performance Paradigm: A Systematic Comparison of Neural Network Architectures for Stock Price Prediction with MDA-Driven Feature Selection},
  author={Omeni, Abdul Rashid},
  year={2025},
  school={University of Kent},
  type={MSc in Artificial Intelligence},
  supervisor={Professor Wang, Frank}
}
```

## Results Replication

### Hardware Requirements
- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU, GPU (optional)
- **Training time**: ~2-3 hours for all architectures

### Expected Performance
The champion LSTM+MDA model should achieve:
- R² ≥ 98.0% on holdout set
- RMSE ≤ $4.50 (AAPL price range $150-250)
- Training convergence within 15 epochs
- Inference latency < 3ms per prediction

## Limitations and Future Work

### Current Limitations
- **Single Asset Scope**: AAPL-specific findings require multi-asset validation
- **Synthetic Sentiment**: Real social media integration needed for production
- **Temporal Coverage**: 2015-2025 period may not capture all market regimes
- **Transaction Costs**: Not incorporated in performance evaluation

### Future Research Directions
- **Cross-Asset Validation**: Extend to bonds, commodities, currencies
- **Real-Time Deployment**: Production trading system implementation
- **Alternative Data Integration**: ESG metrics, satellite imagery, patent filings
- **Risk-Adjusted Metrics**: Sharpe ratio, maximum drawdown, tail risk measures





## License

This project is licensed under the MIT License.

## Acknowledgments

- **Supervisor**: Prof. Frank Wang, University of Kent
- **Data Sources**: Yahoo Finance API, Synthetic sentiment generation framework
- **Computational Resources**: University of Kent HPC cluster
- **Community**: Open-source machine learning ecosystem

## Contact

**Abdul Rashid Omeni**  
MSc Artificial Intelligence  
University of Kent  
Email: [aoo60@kent.ac.uk]  
 
---

**Disclaimer**: This research is for academic purposes only. Not financial advice. Past performance does not guarantee future results.