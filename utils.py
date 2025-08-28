# ============================
#  Imports & Configuration
# ============================


import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific warning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names.*")




import os
import yfinance as yf
import streamlit as st
import pandas as pd
import pandas_ta as ta  # technical‑analysis indicators (Bollinger, RSI ..)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from packaging import version
import sklearn
import tensorflow as tf




# ==========================================
#  Data Acquisition & Feature Engineering
# ----------------------------------------

def load_stock_data(ticker, start_date, end_date, cache_file):
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, skiprows=[1], parse_dates=['Date'])
      
        print(f"Loaded from cache: {cache_file}")
    else:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df.to_csv(cache_file, index=False)
        print(f"Downloaded and saved to cache: {cache_file}")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)
    return df


# --------------------------------------
#  Visualize Feature Distributions
# --------------------------------------

def plot_feature_distribution(df, column='Close'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[column], bins=30, edgecolor='black', alpha=0.75)
    ax.set_title(f'Distribution of {column} Price')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.grid(True)
    st.pyplot(fig)



#------------------------------------------------------------
# Plot Close Price History
#------------------------------------------------------------

def plot_close_price_history(df, ticker=''):
    # Create explicit figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], linewidth=1.5)
    ax.set_title(f"Historical Close Price of {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True)
    
    # Streamlit display with cleanup
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)  # Explicitly close
    st.info("Close Price History plotted successfully.")


#------------------------------------------------------------
#  Compute Institutional-Grade Technical Features
#------------------------------------------------------------


def compute_features(df, dropna: bool = True):
    """Generates institutional-grade technical features with robust error handling"""

    st.write("## Computing Technical Indicators")


    initial_shape = df.shape
    st.write(f"Initial data shape: {initial_shape}")
    # Create working copy
    out = df.copy()
    
    # Index Handling 
    if not isinstance(out.index, pd.DatetimeIndex):
        date_cols = ['Date', 'date', 'datetime', 'time']
        found = False
        for col in date_cols:
            if col in out.columns:
                out = out.set_index(col)
                found = True
                break
        if not found:
            # Create dummy datetime index if none exists
            out.index = pd.date_range(start='2000-01-01', periods=len(out), freq='D')
    
    # Ensure datetime index
    out.index = pd.to_datetime(out.index)
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Price Transformations 
    out['log_close'] = np.log(out['Close'])
    out['log_ret'] = out['log_close'].diff()
    
    # Safe candle_ratio calculation
    hl_diff = out['High'] - out['Low']
    hl_diff[hl_diff == 0] = np.nan  # Handle zero division
    out['candle_ratio'] = (out['Close'] - out['Open']) / hl_diff
    
    # Volatility Features 
    windows = [10, 20, 63]
    for w in windows:
        out[f'volatility_{w}'] = out['log_ret'].rolling(w, min_periods=1).std() * np.sqrt(252)
    
    # Technical Indicators (with safe calculation)
    try:
        out['ATR_14'] = ta.atr(out['High'], out['Low'], out['Close'], length=14)
        out['RSI_14'] = ta.rsi(out['Close'], length=14)
        out['CCI_20'] = ta.cci(out['High'], out['Low'], out['Close'], length=20)
        out['OBV'] = ta.obv(out['Close'], out['Volume'])
    except Exception as e:
        print(f"Indicator calculation error: {e}")
    
    # VWAP with robust fallback
    try:
        out['VWAP'] = ta.vwap(
            high=out['High'], 
            low=out['Low'], 
            close=out['Close'], 
            volume=out['Volume']
        )
    except:
        # Volume-weighted fallback
        cum_vol = out['Volume'].rolling(20, min_periods=1).sum()
        cum_val = (out['Close'] * out['Volume']).rolling(20, min_periods=1).sum()
        out['VWAP'] = np.where(cum_vol > 0, cum_val / cum_vol, out['Close'])
    
    # Trend Features 
    out['EMA_12'] = ta.ema(out['Close'], length=12)
    out['EMA_26'] = ta.ema(out['Close'], length=26)
    out['MACD_line'] = out['EMA_12'] - out['EMA_26']
    
    # Cleanup 
    #  Replace infinities
    # out = out.replace([np.inf, -np.inf], np.nan)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    
    #  Forward fill then backfill
    # out = out.ffill().bfill()
    if dropna:
        out = out.dropna()
    else:
        out = out.ffill()
    
    #  Drop intermediate columns
    out.drop(['log_close'], axis=1, inplace=True, errors='ignore')
    
    #  Drop remaining NA rows but preserve initial data
    return out.iloc[max(windows):]  # Skip initial window period




#------------------------------------------------------------
#  Prepare Daily Sentiment Features
#------------------------------------------------------------
def prepare_daily_sentiment_features(file_path):
    """
    Loads and prepares daily sentiment features from a labeled tweet CSV
    with clear Streamlit display for before/after inspection.
    """

    st.write("### Raw Sentiment Data (Before Processing)")
    try:
        sent_df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading sentiment CSV: {e}")
        return None

    st.dataframe(sent_df.head(10))

    # Ensure 'Date' column parsing
    if 'Date' not in sent_df.columns:
        st.error("The sentiment CSV must contain a 'Date' column.")
        return None

    # Compute daily sentiment statistics
    daily_sent = (
        sent_df
          .assign(Date=lambda df: pd.to_datetime(df['Date'], dayfirst=True, errors='coerce'))
          .dropna(subset=['Date'])
          .resample('D', on='Date')
          ['final_label']
          .agg(['mean', 'std', 'count'])
          .rename(columns={
              'mean':  'sentiment_mean',
              'std':   'sentiment_std',
              'count': 'tweet_volume'
          })
    )

    st.write("### Daily Sentiment Data (After Processing)")
    st.dataframe(daily_sent.head(5))

    return daily_sent


#------------------------------------------------------------
def align_final_dataframe():
    try:
        price_df = (
            pd.read_csv("data/price_data.csv", parse_dates=['Date'])
              .set_index('Date')
              .sort_index()
        )
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        return None

    try:
        sent_df = (
            pd.read_csv("data/daily_tweet_sentiment.csv", parse_dates=['Date'])
              .set_index('Date')
              .sort_index()
        )
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return None

    start = max(price_df.index.min(), sent_df.index.min())
    end = min(price_df.index.max(), sent_df.index.max())

    price_df = price_df.loc[start:end]
    sent_df = sent_df.loc[start:end]
    sent_df = sent_df.reindex(price_df.index, method='ffill')

    full_df = price_df.join(sent_df, how='inner').dropna()

    st.success(f"Final DataFrame aligned:\n**Rows - X:** {full_df.shape[0]} | **Columns - y:** {full_df.shape[1]}")
    st.write("### Preview of Final Aligned DataFrame")
    st.dataframe(full_df.head(5))
    full_df.to_csv("full_Dataframe.csv", index=True)
    st.info("Saved as `full_Dataframe.csv`.")

    return full_df

#***********************************************
def initial_test_split(
    csv_path: str = "full_Dataframe.csv",
    test_frac: float = 0.15
):
    """
    Initial split: reserve test data that will NEVER be touched during feature selection.
    
    Returns:
    --------
    dev_df : pd.DataFrame
        Development data (85%) for MDA feature selection and train/val split
    test_df : pd.DataFrame  
        Holdout test data (15%) - never used until final evaluation
    """
    # Load data
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None, None
    
    # Time-based split: test data comes from the END (most recent)
    n = len(df)
    test_start = int(n * (1 - test_frac))
    
    dev_df = df.iloc[:test_start].copy()    # First 85% for development
    test_df = df.iloc[test_start:].copy()   # Last 15% for final testing
    
    st.success(f"""
    Initial data split complete:
    - **Development data**: {dev_df.index[0]} to {dev_df.index[-1]} ({len(dev_df)} samples, {len(dev_df)/len(df)*100:.1f}%)
    - **Test data (HOLDOUT)**: {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} samples, {len(test_df)/len(df)*100:.1f}%)
    """)
    
    return dev_df, test_df

#-------------------------------------------------------------------


def split_train_test_scaled(df, train_frac=0.80, target_col="Close"):
    df = df[[target_col]]
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[[target_col]])
    test_scaled = scaler.transform(test_df[[target_col]])
    return train_scaled, test_scaled, scaler, train_df, test_df
#-------------------------------------------------------------------

def select_informative_features(
    dev_df: pd.DataFrame,
    horizon: int = 1,
    n_splits: int = 5,
    embargo: int = 5,
    seed: int = 0,
    min_mda_threshold: float = 0.0  # Only keep features with MDA > this value
) -> pd.Series:
    """
    Run MDA feature selection on development data only.
    
    Parameters:
    -----------
    dev_df : pd.DataFrame
        Development data (85% of original data)
    min_mda_threshold : float
        Minimum MDA score to keep a feature (default: 0.0 = only positive)
    
    Returns:
    --------
    informative_features : pd.Series
        MDA scores for features that meet the threshold (MDA > min_mda_threshold)
    """
    
    # Build X, y from development data only
    X, y = build_X_y_for_mda(dev_df, horizon=horizon)
    
    if X is None or y is None:
        st.error("Failed to build X, y for MDA")
        return None
    
    st.info(f"Running MDA feature selection on {X.shape[0]} development samples with {X.shape[1]} features...")
    
    # Compute MDA importance
    mda_scores = compute_mda_importance(
        X, y, 
        n_splits=n_splits, 
        embargo=embargo, 
        seed=seed
    )
    
    # Filter to keep only informative features
    informative_mask = mda_scores > min_mda_threshold
    informative_features = mda_scores[informative_mask].sort_values(ascending=False)
    
    n_total = len(mda_scores)
    n_informative = len(informative_features)
    n_negative = (mda_scores <= min_mda_threshold).sum()
    
    st.success(f"""
    MDA Feature Selection Results:
    - **Total features analyzed**: {n_total}
    - **Informative features** (MDA > {min_mda_threshold}): {n_informative} ({n_informative/n_total*100:.1f}%)
    - **Non-informative features**: {n_negative} ({n_negative/n_total*100:.1f}%)
    - **Best feature**: {informative_features.index[0]} (MDA = {informative_features.iloc[0]:.4f})
    """)
    
    return informative_features

#------------------------------------------------------------------------------

def split_train_val_with_selected_features(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    informative_features: pd.Series,
    train_frac: float = 0.82,  # 82% of dev data = ~70% of total
    val_frac: float = 0.18     # 18% of dev data = ~15% of total  
):
    """
    Split development data into train/val using only the selected informative features.
    Also prepare test data with the same features.
    
    Parameters:
    -----------
    dev_df : pd.DataFrame
        Development data (85% of original)
    test_df : pd.DataFrame  
        Holdout test data (15% of original)
    informative_features : pd.Series
        Selected features with their MDA scores
    train_frac : float
        Fraction of dev_df to use for training
    val_frac : float
        Fraction of dev_df to use for validation
    """
    
    # Validate inputs
    if train_frac + val_frac > 1.0:
        raise ValueError(f"train_frac ({train_frac}) + val_frac ({val_frac}) must be <= 1.0")
    
    if informative_features is None or len(informative_features) == 0:
        st.error("No informative features provided")
        return None, None, None, None, None, None, None, None
    
    # Get feature names that exist in both datasets
    feature_cols = informative_features.index.tolist()
    missing_in_dev = [col for col in feature_cols if col not in dev_df.columns]
    missing_in_test = [col for col in feature_cols if col not in test_df.columns]
    
    if missing_in_dev:
        st.warning(f"Features missing in dev data: {missing_in_dev}")
    if missing_in_test:
        st.warning(f"Features missing in test data: {missing_in_test}")
    
    # Keep only features that exist in both datasets
    available_features = [col for col in feature_cols 
                         if col in dev_df.columns and col in test_df.columns]
    
    if not available_features:
        st.error("No features available in both dev and test datasets")
        return None, None, None, None, None, None, None, None
    
    st.info(f"Using {len(available_features)} features for train/val/test split")
    
    # Select features from datasets
    dev_selected = dev_df[available_features].copy()
    test_selected = test_df[available_features].copy()
    
    # Split development data into train/val
    n_dev = len(dev_selected)
    train_end = int(n_dev * train_frac)
    
    train_df = dev_selected.iloc[:train_end].copy()
    val_df = dev_selected.iloc[train_end:].copy()
    
    # Fit scaler ONLY on training data
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)  
    test_scaled = scaler.transform(test_selected)  # Scale test data with train scaler
    
    # Log split information
    total_samples = len(dev_df) + len(test_df)
    st.success(f"""
    Final data split with {len(available_features)} informative features:
    
    **Training**: {train_df.index[0]} to {train_df.index[-1]}
    - Samples: {len(train_df)} ({len(train_df)/total_samples*100:.1f}% of total data)
    
    **Validation**: {val_df.index[0]} to {val_df.index[-1]}  
    - Samples: {len(val_df)} ({len(val_df)/total_samples*100:.1f}% of total data)
    
    **Test (Holdout)**: {test_selected.index[0]} to {test_selected.index[-1]}
    - Samples: {len(test_selected)} ({len(test_selected)/total_samples*100:.1f}% of total data)
    
    **Features used**: {len(available_features)} informative features
    **Scaler**: Fitted on training data only
    """)
    
    return (train_scaled, val_scaled, test_scaled, scaler, 
            available_features, train_df, val_df, test_selected)




# Complete workflow function
def complete_ml_pipeline(
    csv_path: str = "full_Dataframe.csv",
    test_frac: float = 0.15,
    horizon: int = 1,
    min_mda_threshold: float = 0.0,
    n_splits: int = 5,
    embargo: int = 5,
    seed: int = 0
):
    """
    Complete ML pipeline with proper holdout and feature selection.
    
    Workflow:
    1. Reserve 15% test data (never touched)
    2. Run MDA on remaining 85% 
    3. Select only positive/informative features
    4. Split remaining data into train/val
    5. Return scaled data ready for modeling
    """
    
    st.header("Complete ML Pipeline")
    
    # Step 1: Initial split
    st.subheader(" Reserve Test Data")
    dev_df, test_df = initial_test_split(csv_path, test_frac)
    if dev_df is None:
        return None
    
    # Step 2: Feature selection on dev data only
    st.subheader(" MDA Feature Selection")
    informative_features = select_informative_features(
        dev_df, horizon, n_splits, embargo, seed, min_mda_threshold
    )
    if informative_features is None:
        return None
    
    # Step 3: Split with selected features
    st.subheader(" Train/Val Split with Selected Features")
    results = split_train_val_with_selected_features(
        dev_df, test_df, informative_features
    )
    
    if results[0] is None:
        return None
    
    # Step 4: Show feature importance plot
    st.subheader(" Selected Feature Importance")
    plot_feature_importance(informative_features, n_top=min(15, len(informative_features)))
    
    return {
        'train_scaled': results[0],
        'val_scaled': results[1], 
        'test_scaled': results[2],
        'scaler': results[3],
        'selected_features': results[4],
        'train_df': results[5],
        'val_df': results[6], 
        'test_df': results[7],
        'feature_importance': informative_features
    }

#************************************************
##------------------------------------------------------------
# Helper: version‑aware Bagging wrapper                
##---------------------------------------------------------

def _bagged_tree(
    *,
    seed: int = 0,
    min_w_leaf: float = 0.0,
    n_estimators: int = 500,
    max_samples: float = 1.0,
) -> BaggingClassifier:
    """Bag of entropy trees with one feature per split."""
    base = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=min_w_leaf,
        random_state=seed,
    )
    param = (
        "estimator"
        if version.parse(sklearn.__version__) >= version.parse("1.2")
        else "base_estimator"
    )
    return BaggingClassifier(
        **{param: base},
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=1.0,
        oob_score=True,
        n_jobs=-1,
        random_state=seed,
    )

#------------------------------------------------------
#  permutation importance under purged walk‑forward CV               #
#-----------------------------------------------------

def compute_mda_importance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    embargo: int = 5,  # days to skip after each train fold
    seed: int = 0,
) -> pd.Series:
    """Mean Decrease in Accuracy with a purged CV split."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=embargo)
    clf = _bagged_tree(seed=seed)
    base_score = cross_val_score(
        clf, X, y, cv=splitter, scoring="accuracy", n_jobs=-1
    ).mean()
    rng = np.random.default_rng(seed)
    scores = []
    for col in X.columns:
        Xp = X.copy()
        Xp[col] = rng.permutation(Xp[col].values)
        s = cross_val_score(
            clf, Xp, y, cv=splitter, scoring="accuracy", n_jobs=-1
        ).mean()
        scores.append(base_score - s)
    return pd.Series(scores, index=X.columns, name="MDA")




#------------------------------------------------------------
#  Build X and y for MDA Feature Importance
#------------------------------------------------------------

def build_X_y_for_mda(full_df, horizon=1):
    """
    Builds X and y for MDA feature importance:
    - y: binary direction of future log return (1 = up, 0 = down)
    - X: all features available at day t (EXCLUDING log_ret to prevent leakage)
    """

    # Check required column
    if "log_ret" not in full_df.columns:
        st.error("'log_ret' column is required in full_df for building y.")
        return None, None

    #  Target: binary direction (1 = up, 0 = down)
    y = (full_df["log_ret"].shift(-horizon) > 0).astype(int)

    # Features: everything we know at day t EXCEPT log_ret (to prevent data leakage)
    feature_columns = [col for col in full_df.columns if col != "log_ret"]
    X = full_df[feature_columns].copy()
    
    # Optional: Might add lagged returns as features (these are legitimate predictors)
    # X['log_ret_lag1'] = full_df['log_ret'].shift(1)  # Previous day return
    # X['log_ret_lag2'] = full_df['log_ret'].shift(2)  # Two days ago return

    # Align lengths (drop the last `horizon` rows that now have NaN in y)
    X, y = X.iloc[:-horizon], y.iloc[:-horizon]
    
    # Check what features we're actually using
    st.info(f"""
    **Features for MDA:**
    - Total features: {X.shape[1]}
    - Excluded 'log_ret' to prevent data leakage
    - Sample features: {list(X.columns[:10])}...
    """)

    st.success(f"Built X and y for MDA:\n**X shape:** {X.shape} | **y shape:** {y.shape}")

    return X, y

#------------------------------------------------------------



#------------------------------------------------------------
#  Feature Importance Visualization
#------------------------------------------------------------
def plot_feature_importance(imp: pd.Series, n_top: int = 15) -> None:
    """
    Streamlit-friendly horizontal bar plot of the n_top most important features.

    Parameters
    ----------
    imp   : pd.Series
        Importance scores (e.g., MDA). Higher = more important.
    n_top : int
        How many features to show.
    """
    top = (
        imp.nlargest(n_top)   # get top n features
           .sort_values()     # sort for clean barh plot
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    top.plot(kind="barh", alpha=0.8, ax=ax)
    ax.set_title(f"Top {n_top} Feature Importance Scores (MDA – accuracy drop)")
    ax.set_xlabel("Mean Decrease in Accuracy")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig)

#------------------------------------------------------------



#------------------------------------------------------------
#  Prepare Sequences for CNN-LSTM
#------------------------------------------------------------


def create_training_features(full_df, selected_features):
    """
    Creates training features DataFrame, ensuring 'Close' is included and positioned first.
    More robust version with better error handling.
    """
    # Ensure 'Close' is in selected features
    if 'Close' not in selected_features:
        st.warning("'Close' not in selected features - adding it automatically")
        selected_features = ['Close'] + [f for f in selected_features if f != 'Close']
    
    # Create features DataFrame
    try:
        training_features = full_df[selected_features].copy()
    except KeyError as e:
        missing = [col for col in selected_features if col not in full_df.columns]
        st.error(f"Missing columns in full_df: {', '.join(missing)}")
        raise ValueError(f"Required features missing: {missing}") from e
        
    # Double-check Close exists
    if 'Close' not in training_features.columns:
        st.error("Critical error: 'Close' feature missing after selection")
        raise ValueError("'Close' feature required but not found")
    
    # Ensure Close is first column
    if training_features.columns[0] != 'Close':
        st.info("Reorganizing features with 'Close' as first column")
        cols = ['Close'] + [col for col in training_features.columns if col != 'Close']
        training_features = training_features[cols]
    
    return training_features






def train_test_split_timeseries(data, test_frac=0.15):
    """
    Time-based train/test split for time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (should be time-indexed)
    test_frac : float
        Fraction of data for testing (from the end)
    
    Returns:
    --------
    train_data : pd.DataFrame
        Training data (earlier time periods)
    test_data : pd.DataFrame
        Test data (later time periods)
    """
    
    n = len(data)
    split_idx = int(n * (1 - test_frac))
    
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    st.success(f"""
    **Time-based Train/Test Split:**
    - Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} samples)
    - Test: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} samples)
    """)
    
    return train_data, test_data


#-----------------------------------------------------------------------

def make_sequences(arr, window=60, target_index=0, horizon=1, stride=1):
    arr = np.asarray(arr)
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    X, y = [], []
    L = arr.shape[0]
    last = L - (window + horizon - 1)
    if last <= 0:
        raise ValueError(f"Not enough samples: {L} for window={window}, horizon={horizon}")
    for i in range(0, last, stride):
        X.append(arr[i:i+window, :])
        y.append(arr[i+window+horizon-1, target_index])
    return np.array(X), np.array(y)

def make_sequences_from_df(X_df: pd.DataFrame, y: pd.Series, look_back: int = 60):
    """
    Convert tabular (time x features) into (samples, look_back, features) and aligned y.
    No shuffling.
    """
    X_arr = X_df.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)
    n, f = X_arr.shape
    if n <= look_back:
        raise ValueError("Not enough rows to build sequences.")

    X_seq = np.stack([X_arr[i - look_back:i] for i in range(look_back, n)])
    y_seq = y_arr[look_back:]
    feature_names = list(X_df.columns)  # preserve exact order
    return X_seq, y_seq, feature_names


#-------------------------------------------------------------
#  Make Predictions 
#--------------------------------------------------------------

def make_predictions(model, scaler, train_df, test_df, FEATURES, LOOK_BACK=60, CTX_FRAC_TRAIN=0.20):
    """Robust predictions with scikit-learn compatibility"""
    try:
        # 1. Convert features to tuple for consistent typing
        if not isinstance(FEATURES, tuple):
            FEATURES = tuple(FEATURES)
        
        # 2. Validate Close position
        if 'Close' not in FEATURES:
            raise ValueError("Target feature 'Close' missing in feature list")
        CLOSE_IDX = FEATURES.index('Close')
        
        # 3. Create context window
        ctx_len = int(len(train_df) * CTX_FRAC_TRAIN)
        context = train_df.tail(ctx_len)
        full_test = pd.concat([context, test_df], axis=0).sort_index()
        
        # 4. Extract features in consistent order
        # Convert to numpy array to bypass sklearn's feature name validation
        X_full = full_test[list(FEATURES)].values
        
        # 5. Transform using scaler
        scaled_full = scaler.transform(X_full)
        
        # 6. Create sequences
        X_pred = np.stack([
            scaled_full[i - LOOK_BACK:i]
            for i in range(LOOK_BACK, len(scaled_full))
        ])
        
        # 7. Get true values
        y_true = full_test['Close'].values[LOOK_BACK:]
        
        # 8. Make predictions
        y_pred_scaled = model.predict(X_pred, verbose=0).flatten()
        
        # 9. Inverse scaling
        scaled_for_inv = scaled_full[LOOK_BACK:].copy()
        scaled_for_inv[:, CLOSE_IDX] = y_pred_scaled
        y_pred = scaler.inverse_transform(scaled_for_inv)[:, CLOSE_IDX]
        
        # Return aligned prediction dates
        aligned_dates = full_test.index[LOOK_BACK:]
        
        return y_true, y_pred, aligned_dates, full_test
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error(f"Feature type: {type(FEATURES)}")
        st.error(f"First feature type: {type(FEATURES[0]) if FEATURES else 'N/A'}")
        st.error(f"Scaler features: {getattr(scaler, 'feature_names_in_', 'Not available')}")
        raise
  
     
# ------------------------------------------------------------
#  Compute  Evaluation Metrics
#------------------------------------------------------------


def evaluate_model(y_true, y_pred):
    """
    - Computes and displays key evaluation metrics with consistent ordering.
    - Returns a DataFrame with metric names and values.
    """
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE with zero-division protection
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')

    st.write("### Evaluation Metrics")

    # CONSISTENT METRIC ORDERING - taking care of the extra rows
    metrics = {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),        # Use R² consistently
        "MAPE": round(mape, 2) if mape != float('inf') else "N/A"
    }

    # # Display as table with proper indexing
    # metrics_df = pd.DataFrame([
    #     {"Metric": "Samples Predicted", "Value": len(y_true)},
    #     {"Metric": "MSE", "Value": metrics["MSE"]},
    #     {"Metric": "RMSE", "Value": metrics["RMSE"]},
    #     {"Metric": "MAE", "Value": metrics["MAE"]},
    #     {"Metric": "R²", "Value": metrics["R2"]},
    #     {"Metric": "MAPE (%)", "Value": metrics["MAPE"]}
    # ])
    
    st.table(metrics)
    
    # Return a dictionary for easier access in other functions
    return metrics

#------------------------------------------------------------




import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# More robust metric calculation
def safe_mape(y_true, y_pred):
    """Calculate MAPE safely handling edge cases"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero using a small epsilon
    epsilon = np.finfo(np.float64).eps
    safe_denominator = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    # Calculate percentage errors
    percentage_errors = np.abs((y_true - y_pred) / safe_denominator) * 100.0
    
    # Handle extreme outliers by capping at 1000%
    capped_errors = np.minimum(percentage_errors, 1000.0)
    
    return float(np.mean(capped_errors))

#------------------------------------------------------------

def prune_retrain_model(
    model_builder,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    keep_frac: float = 0.85,  # More conservative default
    n_repeats: int = 5,
    random_state: int | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0,
    test_size: float = 0.2,
    original_target_scaler=None,
    target_is_scaled: bool = True,
    mandatory_features: set = None  # New: features to always keep
) -> tuple[dict[str, float], list[str], tf.keras.callbacks.History, dict[str, float], tf.keras.Model]:
    
    if mandatory_features is None:
        mandatory_features = {"Close"}  # Always keep Close price
    
    # Set random seeds for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
        tf.keras.utils.set_random_seed(random_state)

    feature_names = [str(f) for f in feature_names]
    if X.ndim != 3:
        raise ValueError(f"Expected X to be (samples, timesteps, features), got shape {X.shape}")
    if len(feature_names) != X.shape[-1]:
        raise ValueError(f"feature_names length ({len(feature_names)}) != X.shape[-1] ({X.shape[-1]}).")

    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)

    # Proper time-series split without shuffling
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, shuffle=False
    )

    # Train warm-up model
    warm_epochs = max(5, epochs // 8)
    raw_model = model_builder(X_train.shape[1:])
    
    # Simplified training with early stopping
    raw_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=warm_epochs,
        batch_size=batch_size,
        verbose=max(0, verbose - 1),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True, monitor="val_loss"
            ),
        ]
    )

    # Calculate permutation importance
    importances = permutation_importance_seq(
        model=raw_model, X=X_val, y=y_val, feature_names=feature_names,
        n_repeats=max(3, n_repeats), random_state=random_state
    )

    # Feature selection with mandatory protection
    ranked = [(f, imp) for f, imp in importances.items() if imp >= 0]
    if not ranked:
        raise ValueError("No features with non-negative importance.")
    
    ranked.sort(key=lambda t: t[1], reverse=True)
    n_keep = max(1, int(len(ranked) * keep_frac))
    final_features = [str(f) for (f, _) in ranked[:n_keep]]
    
    # Always include mandatory features
    mandatory_to_keep = set(mandatory_features) & set(feature_names)
    final_features = list(set(final_features) | mandatory_to_keep)
    
    # Ensure proper ordering
    feat_idx = [feature_names.index(f) for f in final_features]

    # Prepare pruned datasets
    X_train_p = X_train[:, :, feat_idx]
    X_val_p = X_val[:, :, feat_idx]
    X_test_p = X_test[:, :, feat_idx]

    # Train pruned model
    pruned_model = model_builder(X_train_p.shape[1:])
    history = pruned_model.fit(
        X_train_p, y_train,
        validation_data=(X_val_p, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=max(7, epochs // 10), 
                restore_best_weights=True, 
                monitor="val_loss"
            ),
        ],
    )

    # Predictions with proper inverse scaling
    test_preds = pruned_model.predict(X_test_p, verbose=0).ravel()

    # Compute metrics with proper inverse scaling
    y_eval = y_test
    y_pred_eval = test_preds
    
    if target_is_scaled and original_target_scaler is not None:
        y_eval = original_target_scaler.inverse_transform(y_eval.reshape(-1, 1)).ravel()
        y_pred_eval = original_target_scaler.inverse_transform(y_pred_eval.reshape(-1, 1)).ravel()

    # Safe metric calculation
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred_eval)))
    mae = float(mean_absolute_error(y_eval, y_pred_eval))
    r2 = float(r2_score(y_eval, y_pred_eval))
    mape = safe_mape(y_eval, y_pred_eval)  # Using the safe function

    metrics = {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE": mape}
    
    return importances, final_features, history, metrics, pruned_model


 #-------------------------------------------------------------------------------


def plot_model_prediction(dates, y_true, y_pred):
    if len(dates) != len(y_true) or len(y_true) != len(y_pred):
        st.error(" Mismatch in lengths of prediction arrays.")
        st.write(f"dates: {len(dates)}, y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, label="True", linewidth=1.2)
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.2)
    ax.set_title("Close-Price Prediction\n(Context = Last 20% of Train)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    st.pyplot(fig)

##################################################################


def permutation_importance_seq(model, X, y, feature_names, n_repeats=5, random_state=None):
    rng = np.random.default_rng(random_state)

    base = model.predict(X, verbose=0).flatten()
    base_mse = mean_squared_error(y, base)

    importances = {}
    for j, name in enumerate(feature_names):
        diffs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            # shuffle *within each sample window* along the time axis
            for s in range(Xp.shape[0]):
                rng.shuffle(Xp[s, :, j])
            pred = model.predict(Xp, verbose=0).flatten()
            diffs.append(mean_squared_error(y, pred) - base_mse)
        importances[name] = float(np.mean(diffs))
    return dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))


#------------------------------------------------------------
#  Plot Permutation Importance Bar Chart
#------------------------------------------------------------

def plot_permutation_importance_barh(importances: dict, title: str = "Permutation Feature Importance"):
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]

    plt.figure(figsize=(8, 5))
    plt.barh(features[::-1], scores[::-1])
    plt.xlabel("Permutation Importance (\u0394 MSE)", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)


#------------------------------------------------------------
#  Display Training Results for Pruned Features
#------------------------------------------------------------
def display_pruned_training_results(history, metrics):
    st.write("### Model Evaluation on Pruned Features")
    st.write(f"**RMSE:** {metrics['RMSE']:.6f}")
    st.write(f"**MAE:** {metrics['MAE']:.6f}")
    st.write(f"**R² Score:** {metrics['R²']:.6f}")

    plt.figure(figsize=(6, 3))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)


#**********************************************


def get_optimal_pruning_config():
    """
    Architecture-aware defaults for LSTM price regression.
    """
    return {
        # Always keep these (can't prune them)
        'lock_features': ['Close'],

        # Encourage *some* pruning, but not too much
        'importance_threshold': 0.0,   # keep any non-harmful features
        'min_features': 5,             # LSTM benefits from richer context
        'max_keep_frac': 0.60,         # cap at 60% of current features

        # Optional group coverage: try to keep at least one from each bucket
        'feature_groups': {
            'price':     ['Close','Open','High','Low','VWAP'],
            'trend':     ['EMA_12','EMA_26','MACD_line'],
            'volatility':['ATR_14','volatility_10','volatility_20','volatility_63'],
            'volume':    ['Volume','OBV'],
            'sentiment': ['sentiment_mean','tweet_volume','sentiment_std'],
        },
        'min_group_coverage': 3,       # aim to cover 3 distinct groups

        # Permutation / training
        'n_repeats': 5,
        'random_state': 42,
        'epochs': 40,
        'batch_size': 64,
        'patience': 10,
        'verbose': 1,
        'test_size': 0.2,

        # Model selection guardrails
        'min_improvement': 0.02,
        'max_degradation': 0.05,
    }

#######################################################################



def intelligent_feature_selection(importances, current_features, config):
    # ---- config & guards ----------------------------------------------------
    t              = max(float(config.get('importance_threshold', 0.0)), 1e-12)
    min_features   = int(config.get('min_features', 3))
    max_keep_frac  = float(config.get('max_keep_frac', 0.8))
    lock           = [f for f in config.get('lock_features', []) if f in current_features]
    groups         = config.get('feature_groups', {})
    target_groups  = int(config.get('min_group_coverage', 0))

    #  rank features by importance (desc), keep only those present 
    ranked = [(f, imp) for f, imp in
              sorted(importances.items(), key=lambda x: x[1], reverse=True)
              if f in current_features]

    #  strictly-positive (thresholded) keep list -> *feature names only* --
    keep_feats = [f for f, imp in ranked if imp > t]

    #  initial cap by fraction (but never below min_features) 
    max_keep = max(min_features, int(max(1, round(len(current_features) * max_keep_frac))))
    keep_feats = keep_feats[:max_keep]

    #  inject locked features at the front (preserve order) 
    for f in lock:
        if f not in keep_feats:
            keep_feats.insert(0, f)

    #  optional: group coverage heuristic 
    if target_groups and groups:
        covered = {g for f in keep_feats for g, members in groups.items() if f in members}
        if len(covered) < target_groups:
            for g, members in groups.items():
                if g in covered:
                    continue
                cand = next((m for m in members if m in current_features and m not in keep_feats), None)
                if cand:
                    keep_feats.append(cand)
                    covered.add(g)
                if len(covered) >= target_groups:
                    break

    #  guarantee minimum count 
    if len(keep_feats) < min_features:
        for f, _ in ranked:
            if f not in keep_feats:
                keep_feats.append(f)
            if len(keep_feats) >= min_features:
                break

    #  final cap with priority for locked features 
    if len(keep_feats) > max_keep:
        # keep all locked, then fill with highest-ranked non-locked
        allowed = [f for f in keep_feats if f in lock]
        for f, _ in ranked:
            if f in keep_feats and f not in allowed:
                allowed.append(f)
            if len(allowed) >= max_keep:
                break
        keep_feats = allowed

    #  deduplicate while preserving order 
    seen = set()
    final_features = [f for f in keep_feats if not (f in seen or seen.add(f))]

    return final_features



#####################################################################


def should_use_pruned_model(original_metrics, pruned_metrics, config):
    """
    Decide whether to use pruned model based on performance comparison.
    """
    if not pruned_metrics or 'RMSE' not in pruned_metrics:
        return False, "No pruned metrics available"
    
    # Extract RMSE for comparison (lower is better)
    try:
        # Handle different metric formats
        if isinstance(original_metrics, dict) and 'RMSE' in original_metrics:
            original_rmse = original_metrics['RMSE']
        elif hasattr(original_metrics, 'loc') and 'RMSE' in original_metrics.index:
            original_rmse = original_metrics.loc['RMSE', 'Value']
        else:
            return False, "Cannot extract original RMSE"
            
        pruned_rmse = pruned_metrics['RMSE']
        
        # Calculate relative change
        improvement = (original_rmse - pruned_rmse) / original_rmse
        
        if improvement >= config['min_improvement']:
            return True, f"Pruned model is {improvement:.1%} better (RMSE: {pruned_rmse:.4f} vs {original_rmse:.4f})"
        elif improvement >= -config['max_degradation']:
            return True, f"Pruned model has acceptable performance (RMSE degradation: {-improvement:.1%})"
        else:
            return False, f"Pruned model performance too low (RMSE degradation: {-improvement:.1%})"
            
    except Exception as e:
        return False, f"Error comparing metrics: {str(e)}"

# UPDATing PRUNING PIPELINE WITH BEST PRACTICES

#####################################################################


def run_intelligent_pruning_pipeline(
    model,
    X_train, X_val, y_train, y_val,
    training_features,
    train_data, test_data,
    original_metrics,
    model_builder=None,  # <-- added
):
    """
    Architecture-aware pruning and retraining.
    Returns: importances (dict), final_features (list), pruned_model (tf.keras.Model), val_metrics (dict)
    """
    st.header(" Intelligent Model Pruning (Architecture-Aware)")

    #  Config + current feature list
    config = get_optimal_pruning_config()
    current_features = list(training_features.columns)

    #  Permutation importance on validation split
    importances = permutation_importance_seq(
        model=model,
        X=X_val,
        y=y_val,
        feature_names=current_features,
        n_repeats=config['n_repeats'],
        random_state=config['random_state']
    )

    #  Select final features
    final_features = intelligent_feature_selection(importances, current_features, config)
    feat_idx = [current_features.index(f) for f in final_features]

    #  Slice features (3D tensors)
    X_train_p = X_train[:, :, feat_idx]
    X_val_p   = X_val[:, :, feat_idx]

    #  Choose a builder (use provided; fallback is simple LSTM)
    if model_builder is None:
        def model_builder(input_shape):
            m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            return m

    # 6) Train pruned model
    history = None  # Initialize history
    if len(final_features) < len(current_features):
        with st.spinner("Training pruned model..."):
            pruned_model = model_builder((X_train_p.shape[1], X_train_p.shape[2]))
            history = pruned_model.fit(
                X_train_p, y_train,
                validation_data=(X_val_p, y_val),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                verbose=config['verbose'],
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    patience=config['patience'], restore_best_weights=True)]
            )

        # 7) Validation diagnostics
        val_pred = pruned_model.predict(X_val_p, verbose=0).flatten()
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae  = mean_absolute_error(y_val, val_pred)
        val_r2   = r2_score(y_val, val_pred)

        # Include both R² and R2 for downstream compatibility
        val_metrics = {
            "RMSE": float(val_rmse),
            "MAE":  float(val_mae),
            "R²":   float(val_r2),
            "R2":   float(val_r2),
        }

        return importances, final_features, pruned_model, history
    else:
        st.info("No feature were pruned - using original model")
        return importances, final_features, pruned_model, None




#------------------------------------------------------------
#  Standardize Metrics Format
#------------------------------------------------------------
def fix_metrics_dataframe(metrics_df): 
    """
    Fix metrics DataFrame by adding proper labels and converting to dictionary.
    """
    if not isinstance(metrics_df, pd.DataFrame):
        return metrics_df
    
    # Common metric names in order (adjust based on  evaluate_model function)
    metric_names = ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Directional_Accuracy']
    
    # Create a copy and add proper index
    fixed_df = metrics_df.copy()
    
    # for the 6 metrics, use the standard names
    if len(fixed_df) == 6:
        fixed_df.index = metric_names[:len(fixed_df)]
    elif len(fixed_df) == 5:
        fixed_df.index = ['RMSE', 'MAE', 'R²', 'MAPE', 'Directional_Accuracy']
    elif len(fixed_df) == 4:
        fixed_df.index = ['RMSE', 'MAE', 'R²', 'MAPE']
    else:
        # Fallback: generic names
        fixed_df.index = [f'Metric_{i}' for i in range(len(fixed_df))]
    
    return fixed_df

def metrics_to_dict(metrics):
    """
    Convert metrics to a standardized dictionary format.
    """
    if isinstance(metrics, dict):
        return metrics
    
    if isinstance(metrics, pd.DataFrame):
        # Fix the DataFrame first
        fixed_df = fix_metrics_dataframe(metrics)
        # Convert to dictionary
        if 'Value' in fixed_df.columns:
            return fixed_df['Value'].to_dict()
        else:
            return fixed_df.iloc[:, 0].to_dict()
    
    if isinstance(metrics, pd.Series):
        return metrics.to_dict()
    
    return metrics

#------------------------------------------------------------


def create_comprehensive_comparison_dashboard(
    original_predictions, pruned_predictions, original_metrics, pruned_metrics,
    original_history, pruning_history, feature_evolution, importances,
    aligned_dates, ticker
):
    """
    Creates a comprehensive dashboard comparing original vs pruned model performance.
    """
    
    st.header("Complete Model Evolution Dashboard")
    
    # ==========================================================================
    #  EXECUTIVE SUMMARY CARDS
    # ==========================================================================
    st.subheader(" Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Features Reduction", 
            f"{len(feature_evolution['final_features'])}",
            f"-{feature_evolution['pruned_count']}"
        )
    
    with col2:
        if pruned_metrics and 'RMSE' in pruned_metrics:
            rmse_change = ((pruned_metrics['RMSE'] - original_metrics['RMSE']) / original_metrics['RMSE']) * 100
            st.metric(
                "RMSE Change", 
                f"{pruned_metrics['RMSE']:.4f}",
                f"{rmse_change:+.1f}%"
            )
        else:
            st.metric("RMSE Change", "N/A", "N/A")
    
    with col3:
        if pruned_metrics and 'R²' in pruned_metrics:
            r2_change = pruned_metrics['R²'] - original_metrics['R²'] if 'R²' in original_metrics else 0
            st.metric(
                "R² Score", 
                f"{pruned_metrics['R²']:.4f}",
                f"{r2_change:+.4f}"
            )
        else:
            st.metric("R² Score", "N/A", "N/A")
    
    with col4:
        complexity_reduction = (feature_evolution['pruned_count'] / len(feature_evolution['mda_selected'])) * 100
        st.metric(
            "Complexity Reduction", 
            f"{complexity_reduction:.1f}%",
            "Simpler Model"
        )
    
    # ===========================================
    #  FEATURE EVOLUTION VISUALIZATION
    # ==========================================
    st.subheader("Architecture After Pruning")
    
    # Feature evolution flow chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create feature evolution visualization
        if importances:
            imp_df = pd.DataFrame([
                {
                    'Feature': feat, 
                    'Importance': imp, 
                    'Status': 'Final Selected' if feat in feature_evolution['final_features'] else 'Removed',
                    'Stage': 'MDA Selected'
                } 
                for feat, imp in importances.items()
            ]).sort_values('Importance', ascending=True)
            
            # Color mapping
            colors = {'Final Selected': 'green', 'Removed': 'red'}
            
            bars = ax.barh(imp_df['Feature'], imp_df['Importance'], 
                          color=[colors[status] for status in imp_df['Status']], 
                          alpha=0.7)
            
            # Add importance threshold line
            ax.axvline(x=0.0001, color='orange', linestyle='--', 
                      linewidth=2, label='Importance Threshold')
            
            ax.set_xlabel('Permutation Importance Score')
            ax.set_title(f'{ticker} - Feature Selection: Permutation Importance Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, imp_df['Importance']):
                width = bar.get_width()
                ax.text(width + max(imp_df['Importance']) * 0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{value:.4f}', 
                       ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("### Evolution Summary")
        st.write(f"**Original Dataset:** All features")
        st.write(f"**MDA Selection:** {len(feature_evolution['mda_selected'])} features")
        st.write(f"**Final Selection:** {len(feature_evolution['final_features'])} features")
        st.write(f"**Total Reduction:** {len(feature_evolution['mda_selected']) - len(feature_evolution['final_features'])} features")
        
        st.write("###  Final Selected Features")
        for i, feat in enumerate(feature_evolution['final_features'], 1):
            importance = importances.get(feat, 0) if importances else 0
            st.write(f"{i}. **{feat}** ({importance:.4f})")
    
    # ==========================================================================
    # 3. TRAINING HISTORY COMPARISON
    # ==========================================================================
    st.subheader(" Training History Evolution")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original model training history
    if original_history:
        original_df = pd.DataFrame(original_history.history)
        
        ax1.plot(original_df['loss'], label='Training Loss', color='blue', alpha=0.8)
        ax1.plot(original_df['val_loss'], label='Validation Loss', color='red', alpha=0.8)
        ax1.set_title('Original Model - Training History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training efficiency metrics
        final_train_loss = original_df['loss'].iloc[-1]
        final_val_loss = original_df['val_loss'].iloc[-1]
        epochs_trained = len(original_df)
        
        ax2.bar(['Training Loss', 'Validation Loss'], [final_train_loss, final_val_loss], 
               color=['blue', 'red'], alpha=0.7)
        ax2.set_title(f'Original Model - Final Losses\n({epochs_trained} epochs)')
        ax2.set_ylabel('Final Loss Value')
        ax2.grid(True, alpha=0.3)
    
    # Pruned model training history
    if pruning_history:
        pruning_df = pd.DataFrame(pruning_history.history)
        
        ax3.plot(pruning_df['loss'], label='Training Loss', color='green', alpha=0.8)
        ax3.plot(pruning_df['val_loss'], label='Validation Loss', color='orange', alpha=0.8)
        ax3.set_title('Pruned Model - Training History')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Training efficiency metrics
        final_train_loss_pruned = pruning_df['loss'].iloc[-1]
        final_val_loss_pruned = pruning_df['val_loss'].iloc[-1]
        epochs_trained_pruned = len(pruning_df)
        
        ax4.bar(['Training Loss', 'Validation Loss'], [final_train_loss_pruned, final_val_loss_pruned], 
               color=['green', 'orange'], alpha=0.7)
        ax4.set_title(f'Pruned Model - Final Losses\n({epochs_trained_pruned} epochs)')
        ax4.set_ylabel('Final Loss Value')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ==========================================================================
    #  PREDICTION COMPARISON PLOTS
    # ==========================================================================
    st.subheader(" Prediction Comparison Analysis")
    
    # Main prediction comparison plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    y_true_orig = original_predictions['y_true']
    y_pred_orig = original_predictions['y_pred']
    y_true_pruned = pruned_predictions['y_true']
    y_pred_pruned = pruned_predictions['y_pred']
    
    # Plot 1: Time series comparison
    ax1.plot(aligned_dates, y_true_orig, label='True Price', color='black', alpha=0.8, linewidth=2)
    ax1.plot(aligned_dates, y_pred_orig, label='Original Model', color='blue', alpha=0.7, linestyle='--')
    ax1.plot(aligned_dates, y_pred_pruned, label='Pruned Model', color='red', alpha=0.7, linestyle=':')
    ax1.set_title(f'{ticker} - Prediction Comparison: Original vs Pruned Model')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction accuracy scatter plots
    ax2.scatter(y_true_orig, y_pred_orig, alpha=0.6, color='blue', label='Original Model')
    ax2.scatter(y_true_pruned, y_pred_pruned, alpha=0.6, color='red', label='Pruned Model')
    
    # Perfect prediction line
    min_val = min(min(y_true_orig), min(y_true_pruned))
    max_val = max(max(y_true_orig), max(y_true_pruned))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
    
    ax2.set_xlabel('True Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Prediction Accuracy: True vs Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction errors comparison
    error_orig = y_pred_orig - y_true_orig
    error_pruned = y_pred_pruned - y_true_pruned
    
    ax3.plot(aligned_dates, error_orig, label='Original Model Error', color='blue', alpha=0.7)
    ax3.plot(aligned_dates, error_pruned, label='Pruned Model Error', color='red', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(aligned_dates, error_orig, alpha=0.3, color='blue')
    ax3.fill_between(aligned_dates, error_pruned, alpha=0.3, color='red')
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Prediction Error ($)')
    ax3.set_title('Prediction Errors Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ==========================================================================
    #  DETAILED METRICS COMPARISON
    # ==========================================================================

    st.subheader(" Detailed Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("###  Original Model Metrics")
        if isinstance(original_metrics, dict):
            orig_df = pd.DataFrame([original_metrics]).T
            orig_df.columns = ['Value']
        else:
            orig_df = original_metrics
        st.dataframe(orig_df.style.format("{:.6f}"))
    
    with col2:
        st.write("### Pruned Model Metrics")
        if pruned_metrics:
            pruned_df = pd.DataFrame([pruned_metrics]).T
            pruned_df.columns = ['Value']
            st.dataframe(pruned_df.style.format("{:.6f}"))
        else:
            st.write("No pruned model metrics available")
    
    # Metrics comparison visualization
    if pruned_metrics:
        st.write("### Performance Comparison Chart")
        
        # Prepare comparison data
        metrics_comparison = {}
        for metric in ['RMSE', 'MAE', 'R²', 'MAPE']:
            if metric in original_metrics and metric in pruned_metrics:
                metrics_comparison[metric] = {
                    'Original': original_metrics[metric],
                    'Pruned': pruned_metrics[metric]
                }
        
        if metrics_comparison:
            comparison_df = pd.DataFrame(metrics_comparison)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.plot(kind='bar', ax=ax, alpha=0.8)
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('Metric Value')
            ax.set_xlabel('Metrics')
            ax.legend(title='Model Type')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
    
    # ===================================
    # 6. STATISTICAL ANALYSIS
    # ===================================
    st.subheader(" Statistical Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("###  Error Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(error_orig, bins=30, alpha=0.7, label='Original', color='blue', density=True)
        ax.hist(error_pruned, bins=30, alpha=0.7, label='Pruned', color='red', density=True)
        ax.set_xlabel('Prediction Error ($)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.write("###  Error Statistics")
        
        error_stats = pd.DataFrame({
            'Original Model': [
                np.mean(error_orig),
                np.std(error_orig),
                np.median(error_orig),
                np.percentile(np.abs(error_orig), 95)
            ],
            'Pruned Model': [
                np.mean(error_pruned),
                np.std(error_pruned),
                np.median(error_pruned),
                np.percentile(np.abs(error_pruned), 95)
            ]
        }, index=['Mean Error', 'Std Error', 'Median Error', '95th Percentile |Error|'])
        
        st.dataframe(error_stats.style.format("{:.4f}"))
    
    with col3:
        st.write("###  Model Insights")
        
        # Calculate improvement metrics
        try:
           rmse_improvement = ((original_metrics['RMSE'] - pruned_metrics['RMSE']) / original_metrics['RMSE'] * 100) if pruned_metrics else 0
        except: rmse_improvement = 0  # Default fallback
        complexity_reduction = (feature_evolution['pruned_count'] / len(feature_evolution['mda_selected']) * 100)
        
        st.write(f"**RMSE Change:** {rmse_improvement:+.1f}%")
        st.write(f"**Complexity Reduction:** {complexity_reduction:.1f}%")
        st.write(f"**Feature Efficiency:** {len(feature_evolution['final_features'])} features")
        
        if rmse_improvement > 0:
            st.success(" Pruned model is more accurate!")
        elif rmse_improvement > -5:
            st.info(" Acceptable accuracy trade-off for simplicity")
        else:
            st.warning(" Significant accuracy reduction")
    
    return True

#******************************


#  predict_after_pruning with proper Close handling
def predict_after_pruning(model, train_df, test_df, FEATURES, look_back=60, ctx_frac=0.2):
    """
    FIXED VERSION: Properly handles Close column and R² calculation
    """
    
    # CRITICAL FIX: Ensure Close is in features
    if 'Close' not in FEATURES:
        st.error(" 'Close' column missing from FEATURES!")
        st.write(f"Available features: {FEATURES}")
        return None, None, None, None
    
    CLOSE_IDX = FEATURES.index('Close')
    st.write(f"Close column found at index: {CLOSE_IDX}")

    # Verify data shapes
    st.write(f" Data shapes - Train: {train_df.shape}, Test: {test_df.shape}")
    st.write(f" Features being used: {FEATURES}")

    # Create context and combine
    context = train_df.tail(int(len(train_df) * ctx_frac))
    combined = pd.concat([context, test_df])[FEATURES]
    
    st.write(f" Combined data shape: {combined.shape}")

    # CRITICAL FIX: Fit scaler on ALL required features
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURES])  # Now includes Close
    
    scaled_full = scaler.transform(combined)
    st.write(f" Scaled data shape: {scaled_full.shape}")

    # Create sequences
    X_pred = np.stack([
        scaled_full[i - look_back:i]
        for i in range(look_back, len(scaled_full))
    ])
    
    st.write(f" Prediction sequences shape: {X_pred.shape}")
    
    # CRITICAL FIX: Use correct target values
    y_true = combined['Close'].values[look_back:]
    st.write(f" True values shape: {y_true.shape}")
    st.write(f" True values range: {y_true.min():.2f} to {y_true.max():.2f}")

    # Model prediction
    y_pred_scaled = model.predict(X_pred, verbose=0).flatten()
    st.write(f" Scaled predictions shape: {y_pred_scaled.shape}")
    st.write(f" Scaled predictions range: {y_pred_scaled.min():.4f} to {y_pred_scaled.max():.4f}")

    # CRITICAL FIX: Proper inverse scaling
    scaled_for_inv = scaled_full[look_back:].copy()
    scaled_for_inv[:, CLOSE_IDX] = y_pred_scaled
    y_pred = scaler.inverse_transform(scaled_for_inv)[:, CLOSE_IDX]
    
    st.write(f" Final predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")

    # VERIFIED METRICS CALCULATION
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Direct calculation with validation
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Validate R² is reasonable
    if r2 > 1.0 or r2 < -10.0:
        st.error(f" IMPOSSIBLE R² VALUE: {r2:.6f}")
        st.write(" Debug info:")
        st.write(f"  - MSE: {mse:.6f}")
        st.write(f"  - Variance of y_true: {np.var(y_true):.6f}")
        st.write(f"  - Mean of y_true: {np.mean(y_true):.6f}")
        st.write(f"  - Mean of y_pred: {np.mean(y_pred):.6f}")
    else:
        st.success(f" R² value is reasonable: {r2:.6f}")
    
    # MAPE with zero-division protection
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = float('inf')
        st.warning(" All true values are zero - MAPE is infinite")

    metrics = {
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 6), 
        "MAE": round(mae, 6),
        "R²": round(r2, 6),  # Use R² consistently
        "MAPE": round(mape, 6) if mape != float('inf') else None
    }
    
    st.write(" **Final Metrics:**")
    for key, value in metrics.items():
        st.write(f"  - {key}: {value}")

    return y_true, y_pred, combined, metrics

#---------------------------------------------------------------


def enhanced_pruning_with_comprehensive_visualization(
    model, X_train, X_val, y_train, y_val, training_features,
    train_data, test_data, original_metrics, original_history,
    aligned_dates, ticker, y_true_orig, y_pred_orig, 
    pruning_function=None,
    pruning_kwargs=None, 
    scaler=None
):
    """
    Generic pruning pipeline template with comprehensive visualization
    - Returns standardized results dictionary
    """
    # Initialize with safe defaults
    results = {
        'final_model': model,
        'final_features': list(training_features.columns),
        'importances': {},
        'final_predictions': {'y_true': y_true_orig, 'y_pred': y_pred_orig},
        'final_metrics': original_metrics,
        'pruning_success': False
    }
    pruning_history = None

    # Run architecture-specific pruning if provided
    if pruning_function:
        try:
            # Execute architecture-specific pruning
            pruned_model = pruning_function(
                original_model=model,
                X_val=X_val,
                y_val=y_val,
                **pruning_kwargs
            )
            
            # Generic retraining
            st.header("Retraining Pruned Architecture")
            pruning_history = pruned_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=10)
                ],
                verbose=0
            )
            
            # Generic evaluation
            st.info(" Evaluating pruned model performance...")
            y_true_pruned, y_pred_pruned, _, _ = make_predictions(
                model=pruned_model,
                scaler=scaler,
                train_df=train_data,
                test_df=test_data,
                FEATURES=results['final_features'],
                LOOK_BACK=60
            )
            metrics_pruned = evaluate_model(y_true_pruned, y_pred_pruned)
            metrics_pruned = normalise_metrics_keys(metrics_pruned)

            # Update results
            results.update({
                'final_model': pruned_model,
                'final_metrics': evaluate_model(y_true_pruned, y_pred_pruned),
                'final_predictions': {'y_true': y_true_pruned, 'y_pred': y_pred_pruned},
                'pruning_success': True
            })
            
            # Capture importances if available
            if hasattr(pruned_model, 'pruning_importances'):
                results['importances'] = pruned_model.pruning_importances
                
        except Exception as e:
            st.error(f"Pruning failed: {str(e)}")
            st.exception(e)
    
    # Create dashboard if pruning succeeded
    if results['pruning_success']:
        # PROPERLY STRUCTURED FEATURE EVOLUTION DICTIONARY
        feature_evolution = {
            'mda_selected': list(training_features.columns),
            'final_features': results['final_features'],
            'pruned_count': len(training_features.columns) - len(results['final_features'])
        }
        
        create_comprehensive_comparison_dashboard(
            original_predictions={'y_true': y_true_orig, 'y_pred': y_pred_orig},
            pruned_predictions=results['final_predictions'],
            original_metrics=original_metrics,
            pruned_metrics=results['final_metrics'],
            original_history=original_history,
            pruning_history=pruning_history,
            feature_evolution=feature_evolution,  
            importances=results['importances'],
            aligned_dates=aligned_dates,
            ticker=ticker
        )
    
    return results


#*************************************************************************



def ensure_flat_array(arr):
    """Convert any array-like to 1D numpy array"""
    arr = np.array(arr)
    if arr.ndim > 1:
        return arr.squeeze()  # Remove single-dimensional entries
    return arr.flatten()      # Make 1D if not already






#------------------------------------------------------------
# Validate Feature DataFrame
#------------------------------------------------------------
def plot_features_distribution(df, max_plots=20):
    """
    Visualizes distributions of all numeric features using simple histograms
    """
    # Select only numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric features found for distribution plots")
        return
    
    st.subheader("Feature Distributions")
    
    # Limit number of plots to avoid overload
    if len(numeric_cols) > max_plots:
        st.info(f"Showing first {max_plots} features out of {len(numeric_cols)}")
        numeric_cols = numeric_cols[:max_plots]
    
    # Create grid layout
    cols_per_row = 3
    rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
    
    # Plot settings
    bins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
    color = st.color_picker("Histogram color", "#4C72B0", key="hist_color")
    
    # Create plots in a grid
    for i, feature in enumerate(numeric_cols):
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        
        with st.container():
            if col_idx == 0:
                # Create new row container
                cols = st.columns(cols_per_row)
            
            with cols[col_idx]:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                
                # Simple histogram
                ax.hist(
                    df[feature].dropna(),
                    bins=bins,
                    color=color,
                    edgecolor='white',
                    alpha=0.85
                )
                
                # Basic formatting
                ax.set_title(f"{feature}", fontsize=12)
                ax.set_xlabel("Value", fontsize=9)
                ax.set_ylabel("Frequency", fontsize=9)
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
                plt.close(fig)

# Usage example in my approach file:
# engineered_df = compute_features(df)
# plot_features_distribution(engineered_df)


# ========= Architecture-aware importance (LSTM) =========

def default_block_size(look_back: int, frac: float = 0.20, min_block: int = 4) -> int:
    """Choose a time-occlusion block for RNNs as a fraction of look_back."""
    return max(min_block, int(round(look_back * frac)))

def time_window_occlusion_importance_lstm(
    model, X, y, feature_names, block: int, position: str = "recent"
) -> dict:
    """
    LSTM-specific occlusion: zero out a contiguous time block for each feature
    (by default the most recent timesteps), then measure ΔMSE vs baseline.
    """
    baseline_preds = model.predict(X, verbose=0).flatten()
    baseline_mse   = mean_squared_error(y, baseline_preds)

    n, T, F = X.shape
    importances = {}
    start = (T - block) if position == "recent" else 0
    end   = start + block

    for j, name in enumerate(feature_names):
        X_occ = X.copy()
        X_occ[:, start:end, j] = 0.0           # occlude recent memory slice
        mse = mean_squared_error(y, model.predict(X_occ, verbose=0).flatten())
        importances[name] = float(mse - baseline_mse)

    return importances

def architecture_aware_importance(
    *,
    model,
    X,
    y,
    feature_names,
    architecture_type: str,
    look_back: int,
    n_repeats: int = 5,
    random_state: int = 42,
    block_frac: float = 0.20,
    w_perm: float = 0.4,
    w_occ: float = 0.6,
) -> dict:
    """
    Combine global permutation importance with LSTM-aware time-window occlusion.
    Returns a dict with 'perm', 'occ', and 'combined' (final score per feature).
    """
    # 1) global permutation
    perm = permutation_importance_seq(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
        n_repeats=n_repeats,
        random_state=random_state
    )

    # 2) architecture-specific occlusion
    if architecture_type.upper() == "LSTM":
        block = default_block_size(look_back, frac=block_frac)
        occ = time_window_occlusion_importance_lstm(
            model=model, X=X, y=y,
            feature_names=feature_names,
            block=block, position="recent"
        )
    else:
        # fallback: just permutation (kept for future architectures)
        occ = {k: 0.0 for k in feature_names}

    # 3) simple [0,1] normalization per method, then weighted sum
    def _normalize(d):
        vals = np.array(list(d.values()), dtype=float)
        lo, hi = vals.min(), vals.max()
        rng = hi - lo if hi != lo else 1.0
        return {k: float((v - lo) / rng) for k, v in d.items()}

    perm_n = _normalize(perm)
    occ_n  = _normalize(occ)
    combined = {
        f: w_perm * perm_n.get(f, 0.0) + w_occ * occ_n.get(f, 0.0)
        for f in feature_names
    }
    combined = dict(sorted(combined.items(), key=lambda kv: kv[1], reverse=True))
    return {"perm": perm, "occ": occ, "combined": combined}


#------------------------------------------------------------
# Normalise metric names for consistency in dashboards
#------------------------------------------------------------
def normalise_metrics_keys(d):
    """Canonicalise metric names for dashboards and summaries."""
    if d is None:
        return {}
    # If a pandas object sneaks in
    if hasattr(d, "to_dict"):
        d = d.to_dict()

    mapping = {
        "R²": "R2",
        "R^2": "R2",
        "MAPE (%)": "MAPE",
        "Samples Predicted": "n_samples",
        "Samples": "n_samples",
    }
    out = {}
    for k, v in d.items():
        out[mapping.get(k, k)] = float(v) if isinstance(v, (int, float)) else v
    return out


#------------------------------------------------------------

# ==============================
# Feature-Pruning Comparison Dashboard
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def _pct(a, b):
    """Percentage change b vs a (positive means improvement if lower-is-better)."""
    try:
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or a == 0:
            return None
        return 100.0 * (a - b) / a
    except Exception:
        return None

def _norm_keys(d):
    if d is None: return {}
    mapping = {"R²":"R2", "R^2":"R2", "MAPE (%)":"MAPE", "Samples Predicted":"n_samples"}
    out = {}
    for k,v in d.items():
        out[mapping.get(k,k)] = v
    return out

def create_feature_pruning_dashboard(
    *,
    original_predictions: dict,      # {'y_true': array, 'y_pred': array}
    pruned_predictions: dict,        # {'y_true': array, 'y_pred': array}
    original_metrics: dict,          # {'MSE','RMSE','MAE','R2','MAPE',...}
    pruned_metrics: dict,            # same keys as above (after pruning)
    original_history=None,           # keras History (optional)
    pruned_history=None,             # keras History (optional)
    aligned_dates=None,              # pd.DatetimeIndex or list (for prediction plots)
    all_features: list[str] = None,  # features available before pruning
    kept_features: list[str] = None, # final selected features
    importances: dict = None,        # {feature: permutation_importance}
    ticker: str = "AAPL",
    title_suffix: str = ""
):
    """
    Streamlit dashboard for Intelligent Feature Pruning (Approach 10).
    Mirrors the style of  previous 'Complete Model Evolution Dashboard'.
    """

    orig_m = _norm_keys(original_metrics)
    prun_m = _norm_keys(pruned_metrics)

    base_rmse = orig_m.get("RMSE")
    new_rmse  = prun_m.get("RMSE")
    rmse_delta_pct = _pct(base_rmse, new_rmse)

    base_r2 = orig_m.get("R2")
    new_r2  = prun_m.get("R2")
    r2_delta = None
    if base_r2 is not None and new_r2 is not None and np.isfinite(base_r2) and np.isfinite(new_r2):
        r2_delta = 100.0 * (new_r2 - base_r2)

    kept_features = kept_features or []
    all_features  = all_features  or kept_features
    removed = [f for f in all_features if f not in kept_features]

    # ---------------------------
    # Executive Summary
    # ---------------------------
    st.header("Complete Model Evolution Dashboard (Feature Pruning)")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Features Kept", f"{len(kept_features)}", delta=f"-{len(all_features)-len(kept_features)}")
    with c2:
        if rmse_delta_pct is None:
            st.metric("RMSE Change", "N/A")
        else:
            st.metric("RMSE Change", f"{new_rmse:.4f}", delta=f"{rmse_delta_pct:+.1f}%")
    with c3:
        if r2_delta is None or not np.isfinite(new_r2):
            st.metric("R² Score", "N/A")
        else:
            st.metric("R² Score", f"{new_r2:.3f}", delta=f"{r2_delta:+.1f}%")
    with c4:
        st.metric("Complexity Reduction", f"{_pct(len(all_features), len(kept_features)) or 0:.1f}%", "Simpler Feature Set")

    st.caption(f"{ticker} {title_suffix}".strip())

    # ---------------------------
    # Feature Importance: kept vs removed
    # ---------------------------
    st.subheader("Feature Set After Pruning")
    if importances:
        imp_df = (pd.DataFrame({"Feature": list(importances.keys()),
                                "Importance": list(importances.values())})
                  .sort_values("Importance", ascending=False))
        imp_df["Status"] = imp_df["Feature"].isin(kept_features).map({True:"Kept", False:"Removed"})

        fig, ax = plt.subplots(figsize=(10,6))
        kept = imp_df[imp_df["Status"]=="Kept"]
        rem  = imp_df[imp_df["Status"]=="Removed"]
        ax.barh(kept["Feature"], kept["Importance"], alpha=0.8, label="Kept")
        ax.barh(rem["Feature"],  rem["Importance"],  alpha=0.5, label="Removed")
        ax.invert_yaxis()
        ax.set_xlabel("Permutation Importance")
        ax.set_title(f"{ticker} – Feature Importance (Kept vs Removed)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Importance scores unavailable for plotting.")

    # ---------------------------
    # Training History Evolution
    # ---------------------------
    st.subheader("Training History Evolution")
    colA, colB = st.columns(2)
    if original_history is not None and hasattr(original_history, "history"):
        with colA:
            df = pd.DataFrame(original_history.history)
            fig, ax = plt.subplots(figsize=(6,4))
            if "loss" in df:     df["loss"].plot(ax=ax, label="Training Loss")
            if "val_loss" in df: df["val_loss"].plot(ax=ax, label="Validation Loss")
            ax.set_title("Original Model – Training History")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    if pruned_history is not None and hasattr(pruned_history, "history"):
        with colB:
            df = pd.DataFrame(pruned_history.history)
            fig, ax = plt.subplots(figsize=(6,4))
            if "loss" in df:     df["loss"].plot(ax=ax, label="Training Loss", color="tab:green")
            if "val_loss" in df: df["val_loss"].plot(ax=ax, label="Validation Loss", color="tab:orange")
            ax.set_title("Pruned Model – Training History")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ---------------------------
    # Prediction Comparison
    # ---------------------------
    st.subheader("Prediction Comparison Analysis")
    y_true_o = original_predictions.get("y_true")
    y_pred_o = original_predictions.get("y_pred")
    y_true_p = pruned_predictions.get("y_true")
    y_pred_p = pruned_predictions.get("y_pred")

    if aligned_dates is not None and y_true_o is not None and y_pred_o is not None and y_pred_p is not None:
        # 1) timeline plot
        fig, ax = plt.subplots(figsize=(11,4))
        ax.plot(aligned_dates, y_true_o, color="k", lw=1.3, label="True Price")
        ax.plot(aligned_dates, y_pred_o, lw=1.0, ls="--", label="Original")
        ax.plot(aligned_dates, y_pred_p, lw=1.0, ls=":",  label="Pruned")
        ax.set_title(f"{ticker} – Original vs Pruned Predictions")
        ax.set_xlabel("Date"); ax.set_ylabel("Close Price ($)")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 2) scatter true vs pred
        fig, ax = plt.subplots(figsize=(11,4))
        ax.scatter(y_true_o, y_pred_o, s=10, alpha=0.4, label="Original")
        ax.scatter(y_true_p, y_pred_p, s=10, alpha=0.4, label="Pruned")
        lims = [min(y_true_o.min(), y_true_p.min()), max(y_true_o.max(), y_true_p.max())]
        ax.plot(lims, lims, "k--", lw=1, label="Perfect")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title("Prediction Accuracy: True vs Predicted")
        ax.set_xlabel("True Price ($)"); ax.set_ylabel("Predicted Price ($)")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 3) error-over-time
        fig, ax = plt.subplots(figsize=(11,3.5))
        ax.plot(aligned_dates, (y_pred_o - y_true_o), lw=0.9, label="Original Error")
        ax.plot(aligned_dates, (y_pred_p - y_true_p), lw=0.9, label="Pruned Error")
        ax.axhline(0, color="k", lw=0.8)
        ax.fill_between(aligned_dates, 0, (y_pred_p - y_true_p), alpha=0.15)
        ax.set_title("Prediction Errors Over Time"); ax.set_xlabel("Date"); ax.set_ylabel("Error ($)")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # ---------------------------
    # Metrics tables + bar chart
    # ---------------------------
    st.subheader("Model Metrics")
    def _as_df(m):
        rows = []
        for k in ["MSE","RMSE","MAE","R2","MAPE"]:
            if k in m:
                rows.append({"Metric":k, "Value":m[k]})
        return pd.DataFrame(rows)
    col1,col2 = st.columns(2)
    with col1:  st.dataframe(_as_df(orig_m), use_container_width=True)
    with col2:  st.dataframe(_as_df(prun_m), use_container_width=True)

    # small comparison bars
    keys = [k for k in ["RMSE","MAE","MAPE"] if k in orig_m and k in prun_m]
    if keys:
        fig, ax = plt.subplots(figsize=(6.5,4))
        x = np.arange(len(keys))
        ax.bar(x-0.18, [orig_m[k] for k in keys], width=0.36, label="Original")
        ax.bar(x+0.18, [prun_m[k] for k in keys], width=0.36, label="Pruned")
        ax.set_xticks(x); ax.set_xticklabels(keys)
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Metric Value")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # ---------------------------
    # Statistical analysis
    # ---------------------------
    st.subheader("Statistical Analysis")
    if y_true_o is not None and y_pred_o is not None and y_pred_p is not None:
        err_o = y_pred_o - y_true_o
        err_p = y_pred_p - y_true_p
        # histogram
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(err_o, bins=40, alpha=0.45, label="Original", density=True)
        ax.hist(err_p, bins=40, alpha=0.45, label="Pruned",   density=True)
        ax.set_title("Error Distribution Comparison"); ax.set_xlabel("Prediction Error ($)"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        def stats(e):
            return {
                "Mean Error": np.mean(e),
                "Std Error":  np.std(e),
                "Median Error": np.median(e),
                "95th Percentile |Error|": np.percentile(np.abs(e), 95)
            }
        st.dataframe(pd.DataFrame({
            "Original Model": stats(err_o),
            "Pruned Model":   stats(err_p)
        }), use_container_width=True)

    # ---------------------------
    # Final summary block
    # ---------------------------
    st.subheader("Complete Pipeline Summary (Feature Pruning)")
    lines = []
    lines.append("**Transformer with Intelligent Feature Pruning**")
    lines.append("")
    lines.append(f"- **Feature Journey:** All features: {len(all_features)} → **Kept:** {len(kept_features)}")
    if rmse_delta_pct is not None and np.isfinite(new_rmse):
        lines.append(f"- **RMSE:** {base_rmse:.4f} → **{new_rmse:.4f}** ({rmse_delta_pct:+.1f}%)")
    if r2_delta is not None and np.isfinite(new_r2):
        lines.append(f"- **R²:** {base_r2:.4f} → **{new_r2:.4f}** ({r2_delta:+.1f}%)")
    if kept_features:
        lines.append("")
        lines.append("**Final Selected (Kept) Features**")
        lines.append(", ".join(kept_features))
    st.success("\n".join(lines))




#  adding a utility function to normalize metrics format
def normalize_metrics(metrics):
    if isinstance(metrics, dict):
        return metrics
    elif isinstance(metrics, pd.DataFrame):
        return dict(zip(metrics["Metric"], metrics["Value"]))
    return {}