
# approach_2.py 

# ============================
#  Imports & Configuration
# ============================
import os
import yfinance as yf
import streamlit as st
import pandas as pd
import pandas_ta as ta  # technical‑analysis indicators (Bollinger, RSI ..)
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.ensemble import BaggingClassifier


from packaging import version
import keras_tuner as kt
from scipy.stats import norm



# Re-producibility
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)


from utils import (
    load_stock_data,
    plot_feature_distribution,
    plot_close_price_history,
    compute_features,
    align_final_dataframe,
    plot_permutation_importance_barh,
    make_sequences,
    plot_features_distribution,
    prepare_daily_sentiment_features,
    plot_model_prediction,
)


#------------------------------------------------------------
#  Split Train/Test and Scale
#------------------------------------------------------------


def split_train_test_scaled(csv_path="data/full_Dataframe.csv", train_frac=0.80):
    """
    Loads features from full_Dataframe.csv, splits into train/test sets, and applies MinMax scaling.

    Parameters:
    - feature_list: list of features to use (if None, use all columns)
    - csv_path: path to CSV file with full feature set
    - train_frac: float (default=0.80), training split ratio

    Returns:
    - train_scaled: np.ndarray
    - test_scaled: np.ndarray
    - scaler: fitted MinMaxScaler object
    - train_df: original (unscaled) training DataFrame
    - test_df: original (unscaled) test DataFrame
    """

    # Load and preprocess
    df = (
        pd.read_csv(csv_path, parse_dates=["Date"])
        .set_index("Date")
        .sort_index()
        .dropna()
    )

    # Train/Test split
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Scaling: Fit on train only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    return train_scaled, test_scaled, scaler, train_df, test_df


#------------------------------------------------------------
#  Build LSTM Model
#------------------------------------------------------------

def build_lstm_model(input_shape):
    """
    Builds and compiles an LSTM regression model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units=50, return_sequences=True),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dense(16),
        layers.Dense(1),  # regression output
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model

#------------------------------------------------------------
#  Display Model Summary in Streamlit
#------------------------------------------------------------

import io
import sys

def display_model_summary(model):
    """
    Captures and displays the model summary in Streamlit.
    """
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__  # reset
    summary_string = stream.getvalue()
    st.text(summary_string)

#-------------------------------------------------------------
#  Make Predictions with LSTM
#--------------------------------------------------------------

def make_predictions(model, scaler, train_df, test_df, LOOK_BACK=60, CTX_FRAC_TRAIN=0.20):
    """
    Make LSTM predictions using all features.
    
    Parameters:
    - model: trained LSTM model
    - scaler: fitted MinMaxScaler
    - train_df: original training DataFrame (unscaled)
    - test_df: original test DataFrame (unscaled)
    - LOOK_BACK: int, number of timesteps in LSTM window
    - CTX_FRAC_TRAIN: float, proportion of trailing train data to provide context

    Returns:
    - y_true: ground-truth Close prices
    - y_pred: predicted Close prices
    - aligned_dates: dates corresponding to predictions
    - full_test: concatenated context + test DataFrame
    """
    FEATURES = train_df.columns.tolist()
    CLOSE_IDX = FEATURES.index('Close')  # Position of 'Close' for inverse transform

    # Context from end of training to initialize LSTM state
    ctx_len = int(len(train_df) * CTX_FRAC_TRAIN)
    context = train_df.tail(ctx_len)
    full_test = pd.concat([context, test_df], axis=0).sort_index()

    # Scale all features using train-fitted scaler
    scaled_full = scaler.transform(full_test[FEATURES])

    # Build rolling windows for prediction
    X_pred = np.stack([
        scaled_full[i - LOOK_BACK:i]
        for i in range(LOOK_BACK, len(scaled_full))
    ])

    # Ground-truth Close prices
    y_true = full_test['Close'].values[LOOK_BACK:]

    # Predict using the LSTM model
    y_pred_scaled = model.predict(X_pred, verbose=0).flatten()

    # For inverse scaling: insert predicted Close back into scaled data
    scaled_for_inv = scaled_full[LOOK_BACK:].copy()
    scaled_for_inv[:, CLOSE_IDX] = y_pred_scaled

    # Inverse scale the entire row, then extract the corrected Close
    y_pred = scaler.inverse_transform(scaled_for_inv)[:, CLOSE_IDX]

    # Align with correct date index
    aligned_dates = full_test.index[LOOK_BACK:]

    return y_true, y_pred, aligned_dates, full_test



# ------------------------------------------------------------
#  Compute  Evaluation Metrics
#------------------------------------------------------------

def evaluate_model(y_true, y_pred):
    """
    Computes and displays key evaluation metrics for regression models.
    Returns a DataFrame with metric names and values.
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    st.write("### Evaluation Metrics")

    metrics = {
        "Samples Predicted": len(y_true),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    }

    # Display as table
    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.table(metrics_df)

    return metrics_df


#------------------------------------------------------------
#  Permutation Importance for Sequence Inputs
#------------------------------------------------------------

def permutation_importance_seq(model, X, y, feature_names, n_repeats=5, random_state=None):
    """
    Permutation importance for sequence (3-D) inputs.

    Parameters
    ----------
    model          : trained Keras/TensorFlow model
    X              : ndarray, shape (n_samples, look_back, n_features)
    y              : ndarray, shape (n_samples,)
    feature_names  : list[str] ordered exactly as columns in X
    n_repeats      : int, number of shuffles per feature
    random_state   : int | None, for reproducibility

    Returns
    -------
    dict {feature_name: importance_score}, sorted descending
        where importance = (permuted_MSE - baseline_MSE)
                          (positive ⇒ feature is useful)
    """
    rng = np.random.default_rng(random_state)

    # Baseline (unpermuted) error
    baseline_preds = model.predict(X, verbose=0).flatten()
    baseline_mse   = mean_squared_error(y, baseline_preds)

    importances = {}
    for idx, name in enumerate(feature_names):
        mse_diffs = []

        for _ in range(n_repeats):
            X_p = X.copy()                                  # copy to permute

            # Flatten the (samples, timesteps) slice, shuffle, reshape back
            slice_flat = X_p[:, :, idx].ravel()
            rng.shuffle(slice_flat)
            X_p[:, :, idx] = slice_flat.reshape(X_p[:, :, idx].shape)

            perm_preds = model.predict(X_p, verbose=0).flatten()
            perm_mse   = mean_squared_error(y, perm_preds)
            mse_diffs.append(perm_mse - baseline_mse)        # positive if performance worsens

        importances[name] = float(np.mean(mse_diffs))

    # Rank descending
    importances = dict(sorted(importances.items(),
                              key=lambda kv: kv[1], 
                              reverse=True))
    
  
    
    return importances

#---------------------------------------------------------
#  Prune and Retrain LSTM with MDA Feature Importance
# ---------------------------------------------------------

def prune_retrain_lstm(
    model_builder,           # callable → compiled model
    X_train, y_train,
    X_val,   y_val,
    feature_names,
    *,
    keep_frac   = 0.5,
    n_repeats   = 5,
    random_state=None,
    epochs      = 50,
    batch_size  = 32,
    verbose     = 1,
    test_frac   = 0.1,       # fraction of VAL reserved as final test
):
    """
    Fit baseline LSTM → permutation importance on VAL → prune → retrain →
    compare baseline vs. pruned on a small unseen TEST slice.
    """

    # --------------------- reproducibility ----------------------------
    rs = check_random_state(random_state)
    np.random.seed(rs.randint(0, 2**31 - 1))
    tf.random.set_seed(rs.randint(0, 2**31 - 1))

    # --------------------- split VAL → VAL + TEST ---------------------
    n_test = max(1, int(len(X_val) * test_frac))
    X_test,  y_test  = X_val[-n_test:], y_val[-n_test:]
    X_val_i, y_val_i = X_val[:-n_test], y_val[:-n_test]

    # --------------------- train baseline -----------------------------
    # es = tf.keras.callbacks.EarlyStopping(patience=10,
    #                                       restore_best_weights=True,
    #                                       monitor="val_loss")
    callbacks_list = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]
    
    baseline = model_builder(X_train.shape[1:])
    baseline.fit(X_train, y_train,
                 validation_data=(X_val_i, y_val_i),
                 epochs=epochs, batch_size=batch_size,
                 callbacks=callbacks_list, verbose=verbose)

    # --------------------- permutation importance ---------------------
    importances = permutation_importance_seq(
        model=baseline,
        X=X_val_i, y=y_val_i,
        feature_names=feature_names,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    # keep positive scores, sort desc
    pos = sorted([(f, imp) for f, imp in importances.items() if imp >= 0],
                 key=lambda x: x[1], reverse=True)
    if not pos:
        raise ValueError("All permutation scores are negative; nothing to keep.")

    k = max(1, int(len(pos) * keep_frac))
    kept = [f for f, _ in pos[:k]]
    idx  = [ {n:i for i,n in enumerate(feature_names)}[f] for f in kept ]

    # --------------------- prune datasets -----------------------------
    X_train_p = X_train[:, :, idx]
    X_val_p   = X_val_i[:, :, idx]
    X_test_p  = X_test[:, :, idx]

    # --------------------- retrain on pruned --------------------------
    pruned = model_builder(X_train_p.shape[1:])
    pruned.fit(X_train_p, y_train,
               validation_data=(X_val_p, y_val_i),
               epochs=epochs, batch_size=batch_size,
               callbacks=callbacks_list, verbose=verbose)

    # --------------------- helper for metrics -------------------------
    def _m(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {"MSE": mse,
                "RMSE": np.sqrt(mse),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2":  r2_score(y_true, y_pred)}

    baseline_metrics = _m(y_test, baseline.predict(X_test).flatten())
    pruned_metrics   = _m(y_test, pruned.predict(X_test_p).flatten())

    return (importances, kept,
            baseline_metrics, pruned_metrics,
            pruned)  # models can be saved from caller


#------------------------------------------------------------
#  Main Streamlit App
#------------------------------------------------------------

def run_close_indicators_sentiment_lstm(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    FIXED: Complete all-features LSTM pipeline 
    """    
    # =========================================================================
    # STEP 1: DATA LOADING AND VISUALIZATION
    # =========================================================================
    with st.spinner("Fetching and preparing data..."):
        # Load data using your approach_3 utility
        df = load_stock_data(ticker, str(start_date), str(end_date), f"{ticker}_{start_date}_to_{end_date}.csv")

        st.write("### Sample of Downloaded Data")
        st.dataframe(df.head(10))

    st.write("### Visualizing Feature Distributions")
    plot_feature_distribution(df)

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    with st.spinner("Computing technical features..."):
        df_features = compute_features(df)

        st.write("### Visualizing Price Trends")
        plot_close_price_history(df)

        st.write("### Feature-Engineered Data Sample")
        st.dataframe(df_features.head(5))

    # =========================================================================
    # STEP 3: SENTIMENT DATA INTEGRATION
    # =========================================================================
    with st.spinner("Loading and processing sentiment data..."):
        daily_sent = prepare_daily_sentiment_features("data/synthetic_financial_tweets_labeled.AAPL.csv")
        if daily_sent is None:
            st.stop()  # Stop execution if loading failed

    with st.spinner("Aligning price and sentiment data..."):
        full_df = align_final_dataframe()
        if full_df is None:
            st.stop() 

    # Display feature analysis
    st.write(f"### Plotting  Features Distribution:**")
    plot_features_distribution(full_df, max_plots=20)


    # =======================================================
    # STEP 4: DATA SPLITTING AND SCALING (ALL FEATURES)
    # =======================================================

    st.write("###  Using ALL Features (No MDA Selection)")
    st.info("This approach uses all 20+ features without any feature selection to test LSTM's ability to handle feature complexity.")
    
    # Display feature analysis
    st.write(f"**Total Features Available:** {len(full_df.columns)}")
    st.write(f"**Feature List:** {list(full_df.columns)}")



    # Split into train/test and scale ALL features
    st.write("### Splitting and Scaling Data")
    X_train_scaled, X_test_scaled, scaler, train_df, test_df = split_train_test_scaled()
    
    if X_train_scaled is None:
        st.stop()

    # ==========================================
    # STEP 5: LSTM SEQUENCE PREPARATION
    # ==========================================
    WINDOW = 60
    
    # Define features and get index of 'Close' column
    FEATURES = train_df.columns.tolist()
    CLOSE_IDX = FEATURES.index('Close')

    st.write(f"**Using {len(FEATURES)} features for LSTM training:**")
    st.write(FEATURES)

    # Generate LSTM sequences from scaled data
    X_train, y_train = make_sequences(X_train_scaled, window=WINDOW, target_index=CLOSE_IDX)
    X_test, y_test = make_sequences(X_test_scaled, window=WINDOW, target_index=CLOSE_IDX)

    # Display success confirmation in Streamlit
    st.success("✅ LSTM sequences generated successfully using all features.")

    # Show detailed shape information for verification
    st.write("### Sequence Data Shapes")
    st.write(f"**X_train**: {X_train.shape}  **y_train**: {y_train.shape}")
    st.write(f"**X_test**:  {X_test.shape}  **y_test**:  {y_test.shape}")

    # =========================================================================
    # STEP 6: MODEL BUILDING
    # =========================================================================
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model (same architecture as Approach 3 for fair comparison)
    model = build_lstm_model(input_shape)

    # Display summary inside your Streamlit app
    st.write("### LSTM Model Summary (All Features)")
    display_model_summary(model)

    # =================================
    # STEP 7: VALIDATION SPLIT
    # =================================
    VAL_FRAC = 0.10
    val_start = int(len(X_train) * (1 - VAL_FRAC))

    X_tr, X_val = X_train[:val_start], X_train[val_start:]
    y_tr, y_val = y_train[:val_start], y_train[val_start:]

    st.write(f"Validation split: {VAL_FRAC*100:.0f}% of training data")
    st.write(f"Shapes ➔ X_tr: {X_tr.shape}, y_tr: {y_tr.shape} | X_val: {X_val.shape}, y_val: {y_val.shape}")

    # =========================================================================
    # STEP 8: MODEL TRAINING WITH EARLY STOPPING
    # =========================================================================
    # es = callbacks.EarlyStopping(
    #     patience=7,
    #     restore_best_weights=True
    # )
    callbacks_list = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]

    with st.spinner("Training the LSTM model with ALL features..."):
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=2,
        )

    st.success("Model training completed.")

    # ==========================================
    # STEP 9: TRAINING HISTORY VISUALIZATION
    # ==========================================
    st.write("### Training History")
    
    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(figsize=(8, 4))
    history_df['loss'].plot(ax=ax, label='Training Loss')
    history_df['val_loss'].plot(ax=ax, label='Validation Loss')
    ax.set_title('Training and Validation Loss (All Features)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ========================================
    # STEP 10: PREDICTIONS AND EVALUATION
    # =========================================
    st.write("###  Predictions & Evaluation")
    
    # Prediction and inverse scaling
    with st.spinner("Making predictions..."):
        y_true, y_pred, aligned_dates, full_test = make_predictions(
            model=model,
            scaler=scaler,
            train_df=train_df,
            test_df=test_df,
        )
        
    # Plotting predictions
    plot_model_prediction(aligned_dates, y_true, y_pred)

    # Display true vs predicted values (first 10)
    results_df = pd.DataFrame({
        'True Close': y_true[:10],
        'Predicted Close': y_pred[:10],
        'Difference': y_pred[:10] - y_true[:10],
        'Error %': ((y_pred[:10] - y_true[:10]) / y_true[:10] * 100).round(2)
    })
    st.write("###  True vs Predicted Close Prices (First 10)")
    st.dataframe(results_df)

    # =================================
    # STEP 11: MODEL EVALUATION
    # =================================
    # Evaluate model performance
    metrics_df = evaluate_model(y_true, y_pred)

    # Display evaluation metrics
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Metrics as CSV",
        data=csv_metrics,
        file_name=f"{ticker}_all_features_lstm_metrics.csv",
        mime='text/csv'
    )

    # ============================================================