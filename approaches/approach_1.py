# approaches/approach_1.py


import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress specific warning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names.*")


import yfinance as yf
import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
from sklearn.preprocessing import MinMaxScaler
import io
import sys

np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)


from utils import (
    load_stock_data,
    plot_feature_distribution,
    plot_close_price_history,
    evaluate_model,
    plot_model_prediction
    )

#---------------------------------------------------
# Build Simple LSTM Model
#---------------------------------------------------
# Model for approach 1

def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units=50, return_sequences=True),
        layers.LSTM(units=50, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def display_model_summary(model):
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    summary_string = stream.getvalue()
    st.text(summary_string)

#--------------------------------------------------------------


def split_train_test_scaled_single(df, train_frac=0.80):
    """
    FIXED: Handle single-column (Close only) DataFrame properly AND preserve DatetimeIndex
    """
    # Use only Close column
    close_df = df[['Close']].copy()
    
    # Train/Test split
    split_idx = int(len(close_df) * train_frac)
    train_df = close_df.iloc[:split_idx].copy()
    test_df = close_df.iloc[split_idx:].copy()

    #  Scaling with .values to preserve DataFrame index
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)  
    test_scaled = scaler.transform(test_df.values)        

    return train_scaled, test_scaled, scaler, train_df, test_df


#----------------------------------------------------------------

#   Better sequence generation with error checking
def make_sequences_improved(arr, window):
    """
    Improved univariate sequence generation with better error handling
    """
    if len(arr) <= window:
        raise ValueError(f"Not enough samples ({len(arr)}) for window size {window}")
    
    # Ensure 2D array
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2 and arr.shape[1] > 1:
        # Take only first column if multiple columns provided
        arr = arr[:, :1]
    
    n_samples = arr.shape[0]
    
    # Generate sequences
    X = []
    y = []
    
    for i in range(window, n_samples):
        X.append(arr[i-window:i, 0])  # Look-back window
        y.append(arr[i, 0])           # Next value to predict
    
    X = np.array(X).reshape(len(X), window, 1)  # (samples, timesteps, features=1)
    y = np.array(y)
    
    st.success(f"Generated {X.shape[0]} sequences → X: {X.shape}, y: {y.shape}")
    return X, y

#-------------------------------------------------------------------------------

def make_predictions_improved(model, scaler, train_df, test_df, LOOK_BACK=60, CTX_FRAC_TRAIN=0.20):
    """
    FIXED: Proper date alignment for predictions
    """
    
    #  Ensure sufficient context (at least LOOK_BACK)
    ctx_len = max(LOOK_BACK, int(len(train_df) * CTX_FRAC_TRAIN))
    context = train_df.tail(ctx_len)
    
    #  Combine context with test data (preserve proper DatetimeIndex)
    full_test = pd.concat([context, test_df])
    
    #  Scale the combined data
    scaled_full = scaler.transform(full_test[['Close']])
    
    #  Create sequences for prediction
    X_test = []
    for i in range(LOOK_BACK, len(scaled_full)):
        X_test.append(scaled_full[i-LOOK_BACK:i, 0])
    X_test = np.array(X_test).reshape(-1, LOOK_BACK, 1)
    
    #  Make predictions (scaled)
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    #  Inverse transform predictions
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    
    #  Get correct test data alignment
    y_true = test_df['Close'].values
    test_dates = test_df.index  # This should be actual dates
    
    # Proper prediction alignment
    # The first prediction corresponds to the first test sample
    #  need the last len(test_df) predictions
    y_pred_aligned = y_pred[-len(test_df):]
    
    #  Return actual DatetimeIndex, not integer positions
    return y_true, y_pred_aligned, test_dates



#------------------------------------------------------------
#  Main function for Approach 1
#------------------------------------------------------------
def run_simple_close_lstm(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    IMPROVED: Main function with better error handling and data flow
    """
    
    st.header(" Approach 1: Simple Close Price LSTM")
    st.info("This approach uses ONLY the Close price to predict future Close prices - the ultimate simplicity test!")
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    with st.spinner("Fetching and preparing data..."):
        # Handle date format properly
        if isinstance(start_date, str):
            cache_file = f"data/{ticker}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
            df = load_stock_data(ticker, start_date, end_date, cache_file)
        else:
            cache_file = f"data/{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            df = load_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), cache_file)
        
        st.write("### Sample of Downloaded Data")
        st.dataframe(df.head(10))
        
        # Show data info
        st.write(f"**Total samples:** {len(df)}")
        st.write(f"**Date range:** {df.index[0]} to {df.index[-1]}")
        st.write(f"**Using only:** Close price column")

    # =====================================
    # STEP 2: VISUALIZATION
    # =====================================
    st.write("### Close Price Distribution")
    plot_feature_distribution(df, column='Close')

    st.write("### Historical Close Price")
    plot_close_price_history(df, ticker=ticker)

    # ==================================
    # STEP 3: DATA PREPARATION
    # ==================================
    st.write("###  Data Splitting and Scaling")
    
    # Use the improved splitting function
    X_train_scaled, X_test_scaled, scaler, train_df, test_df = split_train_test_scaled_single(df)
    
    if X_train_scaled is None:
        st.error(" Data preparation failed!")
        st.stop()

    st.write(f"**Train samples:** {len(X_train_scaled)}")
    st.write(f"**Test samples:** {len(X_test_scaled)}")

    # ====================================
    # STEP 4: SEQUENCE GENERATION
    # ====================================
    WINDOW = 60
    st.write(f"###  Generating LSTM Sequences (Window: {WINDOW})")

    # Use improved sequence generation
    X_train, y_train = make_sequences_improved(X_train_scaled, window=WINDOW)
    X_test, y_test = make_sequences_improved(X_test_scaled, window=WINDOW)

    st.write("###  Sequence Data Shapes")
    st.write(f"**X_train**: {X_train.shape} (samples, timesteps, features)")
    st.write(f"**y_train**: {y_train.shape}")
    st.write(f"**X_test**: {X_test.shape}")
    st.write(f"**y_test**: {y_test.shape}")

    # ================================
    # STEP 5: MODEL BUILDING
    # ================================
    input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 1)
    model = build_lstm_model(input_shape)

    st.write("### LSTM Model Architecture")
    display_model_summary(model)

    # ================================
    # STEP 6: TRAINING SETUP
    # ================================
    VAL_FRAC = 0.10
    val_start = int(len(X_train) * (1 - VAL_FRAC))
    X_tr, X_val = X_train[:val_start], X_train[val_start:]
    y_tr, y_val = y_train[:val_start], y_train[val_start:]
    
    st.write(f"**Validation split:** {VAL_FRAC*100:.0f}% of training data")
    st.write(f"**Training shapes:** X_tr: {X_tr.shape}, y_tr: {y_tr.shape}")
    st.write(f"**Validation shapes:** X_val: {X_val.shape}, y_val: {y_val.shape}")

    # ==================================
    # STEP 7: MODEL TRAINING
    # ==================================
     # Enhanced callbacks
    
    callbacks_list = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
    ]

    with st.spinner("Training the Simple Close-Price LSTM..."):
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=2,
        )
    
    st.success("✅ Model training completed!")

    # =========================================
    # STEP 8: TRAINING VISUALIZATION
    # =========================================
    st.write("###  Training History")
    
    history_df = pd.DataFrame(history.history)
    fig, ax = plt.subplots(figsize=(10, 4))
    history_df['loss'].plot(ax=ax, label='Training Loss', linewidth=2)
    history_df['val_loss'].plot(ax=ax, label='Validation Loss', linewidth=2)
    ax.set_title('Simple Close-Price LSTM: Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # =========================================
    # STEP 9: PREDICTIONS AND EVALUATION
    # =========================================
    st.write("###  Making Predictions")
    st.info("Using the last 20% of training data as context for LSTM warm-up")


    with st.spinner("Generating predictions..."):
        y_true, y_pred, test_dates = make_predictions_improved(
            model=model,
            scaler=scaler,
            train_df=train_df,
            test_df=test_df
        )

    # Plot predictions
    st.write("###  Prediction Results")
    plot_model_prediction(test_dates, y_true, y_pred)

    # Show sample predictions
    n_show = min(10, len(y_true))
    results_df = pd.DataFrame({
        'Date': test_dates[:n_show],
        'True Close': y_true[:n_show],
        'Predicted Close': y_pred[:n_show],
        'Error': (y_pred[:n_show] - y_true[:n_show]),
        'Error %': ((y_pred[:n_show] - y_true[:n_show]) / y_true[:n_show] * 100).round(2)
    })
    
    st.write(f"###  Sample Predictions (First {n_show})")
    st.dataframe(results_df)

    # =========================================
    # STEP 10: MODEL EVALUATION
    # =========================================
    st.write("###  Model Evaluation")
    
    metrics_result = evaluate_model(y_true, y_pred)

    # FIXED: Handle return format properly
    if isinstance(metrics_result, dict):
        metrics_df = pd.DataFrame([
            {"Metric": key, "Value": value} 
            for key, value in metrics_result.items()
        ])
    else:
        metrics_df = metrics_result  # FIXED INDENTATION

    # Download option
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Metrics as CSV",
        data=csv_metrics,
        file_name=f"{ticker}_simple_close_lstm_metrics.csv",
        mime='text/csv'
    )

    # ========================================
    # STEP 11: FINAL SUMMARY
    # ========================================
    st.header(" Approach 1 Complete Summary")
    
    final_rmse = metrics_df[metrics_df['Metric'] == 'RMSE']['Value'].iloc[0]
    final_r2 = metrics_df[metrics_df['Metric'] == 'R2']['Value'].iloc[0]
    
    st.success(f"""
    ** Simple Close-Price LSTM Results:**
    
     **Data:** {ticker} from {start_date} to {end_date}
     **Features Used:** Close price ONLY (ultimate simplicity)
     **Architecture:** Simple 2-layer LSTM
     **Training:** {len(history.history['loss'])} epochs with early stopping
    
    ** Final Performance:**
    - RMSE: {final_rmse:.4f}
    - R²: {final_r2:.4f}
    - Predictions: {len(y_pred)} test samples
    
    ** Key Insight:** This tests whether complex features are necessary or if price patterns alone are sufficient for prediction.
    """)

    return {
        'model': model,
        'metrics': {
            'RMSE': float(final_rmse),
            'R2': float(final_r2),
            'features_used': 1
        },
        'predictions': {'y_true': y_true, 'y_pred': y_pred, 'dates': test_dates},
        'training_history': history
    }
#------------------------------------------------------------






















































































































