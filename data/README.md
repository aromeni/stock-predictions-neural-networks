# Data

## Sources
- **Prices:** Downloaded via `yfinance` (Yahoo data) in `Code/scripts/download_data.py`.
- **Indicators:** computed locally (pandas-ta).
- **Sentiment (optional):** `synthetic_financial_tweets_labeled.AAPL.csv` if provided.  
  If absent, the pipeline sets `tweet_volume` placeholder to 0.0.

## Regeneration
```bash
python Code/scripts/download_data.py --ticker AAPL --start 2015-01-01 --end 2025-05-30
