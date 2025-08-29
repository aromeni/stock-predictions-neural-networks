# Systematic Comparison of Deep Learning Architectures for Financial Time Series Prediction

## Project Title
**Against the Complexity–Performance Paradigm:  
A Comprehensive Multi-Model Analysis of Neural Network Architectures with MDA-Driven Feature Selection**

**Student:** Abdul Rashid Omeni  
**Supervisor:** Prof. Frank Wang  

---

## Executive Summary
This study systematically compares **nine deep learning architectures** for predicting **AAPL stock price (2015–2025)**.  
Counterintuitive findings reveal that **architectural complexity does not guarantee superior performance** in financial forecasting.  
Through rigorous experimentation with identical datasets, preprocessing, and evaluation metrics, the research shows that **simple, feature-engineered models outperform more advanced architectures**.  

---

## Key Research Questions
- **Architecture Performance:** Which architecture performs best for financial time series prediction?  
- **Feature Engineering Impact:** How does feature selection influence model outcomes?  
- **Complexity–Performance Relationship:** Does increasing model sophistication improve accuracy?  
- **Computational Efficiency:** What are the trade-offs between complexity and resource usage?  

---

## Methodology Overview
- **Dataset:** AAPL stock data (2015–2025) + 108,388 synthetic sentiment events  
- **Feature Engineering:** Technical indicators, price data, volume metrics, sentiment scores  
- **Feature Selection:** Mean Decrease in Accuracy (MDA)  
- **Architectures Tested (9):** LSTM, GRU, TCN, Transformer, CNN–LSTM, TCN–LSTM, LSTM–Transformer, plus baselines  
- **Evaluation Metrics:** $R^2$, RMSE, MAE, MAPE, Directional Accuracy  
- **Validation:** Time-series split (85% train, 15% holdout)  

---

## Results Summary

### Performance Rankings
| Rank | Approach | Architecture        | $R^2$   | RMSE ($) | Features Used |
|------|----------|---------------------|---------|----------|---------------|
| 🥇 1 | 3        | LSTM + MDA          | **98.35%** | 4.20     | 8 (MDA-selected) |
| 🥈 2 | 4        | GRU + MDA           | 97.81%  | 4.82     | 8 |
| 🥉 3 | 9        | LSTM–Transformer    | 97.35%  | 5.31     | 8 |
| 4th  | 1        | Simple LSTM         | 94.59%  | 5.75     | Close only |
| 5th  | 5        | TCN                 | 94.22%  | 5.89     | 8 |
| 6th  | 2        | Complex LSTM        | 92.65%  | 8.54     | 21 (unselected) |
| 7th  | 7        | CNN–LSTM            | 87.35%  | 11.59    | 8 |
| 8th  | 6        | Transformer         | 83.25%  | 13.22    | 8 |
| 9th  | 8        | TCN–LSTM            | 61.98%  | 20.10    | 8 |

---

## Critical Findings

### 1. The Complexity Paradox
- Simple **LSTM + MDA** achieved the highest performance ($R^2 = 98.35\%$).  
- More complex models **performed worse**, with Transformers and TCN–LSTMs failing catastrophically.  
- Over-parameterisation harms financial prediction accuracy.  

### 2. Universal Feature Selection
- The **same 8 features** consistently emerged as optimal across all architectures:  
  `Close, Open, candle_ratio, volatility_63, OBV, tweet_volume, EMA_12, MACD_line`  
- This demonstrates that **domain knowledge > architecture sophistication**.  

### 3. Computational Efficiency
- **LSTM:** fastest (11 epochs, minimal parameters, best performance-to-resource ratio).  
- **GRU:** slightly slower but still efficient.  
- **Transformer/TCN-LSTM:** highest resource consumption, weakest performance.  

### 4. Pruning Benefits
- Pruning improved 4/9 architectures.  
- Complex models consistently benefited from simplification.  

---

## Statistical Significance
- Champion Model (**LSTM+MDA**): $R^2 = 98.35\%$, RMSE ≈ \$4.20 on \$150–250 prices (~2–3% error).  
- **Directional Accuracy:** >85% correct trend prediction.  
- Robust performance across multiple market conditions (2015–2025).  

---

## Scientific Contributions
1. **Paradigm Shift in Financial ML:** Proves that simplicity outperforms complexity in forecasting.  
2. **Universal Feature Selection:** First evidence of **architecture-agnostic optimal features** in finance.  
3. **Resource Efficiency Framework:** Guidelines for balancing accuracy with computational cost.  
4. **Domain-Specific Architecture Insights:** LSTM > GRU > others; attention and convolution approaches fail in finance.  

---

## Implications

**For Practitioners:**  
- Focus on feature engineering over architectural complexity.  
- Use **LSTM + MDA** as the gold standard.  
- Avoid over-engineering — simplicity wins.  

**For Researchers:**  
- Re-examine complexity assumptions in financial ML.  
- Validate domain-specific choices empirically.  
- Incorporate computational efficiency in evaluations.  

**For the Field:**  
- New benchmark: $R^2 = 98.35\%$.  
- A reproducible methodology for architecture comparison.  
- Clear, evidence-based recommendations for finance.  

---

## Data Quality & Methodology
- **Synthetic Sentiment:** 108,388 realistic tweets (2015–2025), 8,372 market events.  
- **Experimental Controls:** identical preprocessing, same splits, consistent metrics.  
- **Reproducibility:** fixed random seeds; full pipeline archived.  

---

## Limitations & Future Work
**Current Limitations:**  
- Single asset (AAPL).  
- Synthetic sentiment; real-world integration needed.  
- US-only; international markets not explored.  
- Transaction costs not modelled.  

**Future Directions:**  
- Multi-asset portfolio optimisation with LSTM+MDA.  
- Real-time sentiment via APIs.  
- Risk-adjusted metrics beyond $R^2$ and RMSE.  
- Regime detection for adaptive model selection.  

---

## Conclusion
This study establishes a **new paradigm** for financial machine learning:  
- **Feature engineering and simplicity outperform architectural sophistication.**  
- **LSTM + MDA** sets a new gold standard for stock price prediction ($R^2 = 98.35\%$).  

The implications are clear: practitioners, researchers, and the broader field should prioritise **domain knowledge, feature selection, and computational efficiency** over unnecessary complexity.  

---

## Project Details
- **Study Period:** 2015–2025  
- **Models Tested:** 9  
- **Features Considered:** 21  
- **Champion Model:** LSTM + MDA ($R^2 = 98.35\%$)  
- **Dataset Size:** 108,388 sentiment observations  

---

## License
MIT License © 2025 Abdul Rashid Omeni
