# approaches/approach_7.py

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
from tensorflow.keras import layers, Model, regularizers, callbacks, models
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
    # Existing imports...
    load_stock_data,
    plot_feature_distribution,
    plot_close_price_history,
    compute_features,
    prepare_daily_sentiment_features,
    align_final_dataframe,
    plot_features_distribution,
    # build_cnn_lstm_model,
    # display_model_summary,
    complete_ml_pipeline,
    make_predictions,
    evaluate_model,
    plot_model_prediction,
    make_sequences,
    create_training_features,
    train_test_split_timeseries,
    get_optimal_pruning_config,
    intelligent_feature_selection,
    should_use_pruned_model,
    run_intelligent_pruning_pipeline,
    create_comprehensive_comparison_dashboard,
    enhanced_pruning_with_comprehensive_visualization,
    prune_retrain_model,
    permutation_importance_seq,
)



#   helpers scalar, slice, safe weights
def _scalar(x, typ, name):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            raise ValueError(f"{name} is empty")
        x = x[0]
    try:
        return typ(x)
    except Exception as e:
        raise TypeError(f"{name} must be {typ.__name__}, got {type(x)} with value {x}") from e

def _slice_to_overlap(src_arr, dst_arr):
    sl = tuple(slice(0, min(s, d)) for s, d in zip(src_arr.shape, dst_arr.shape))
    out = dst_arr.copy()
    out[sl] = src_arr[sl]
    return out

def _safe_copy_weights(src_layer, dst_layer):
    try:
        sw, dw = src_layer.get_weights(), dst_layer.get_weights()
        if not sw or not dw or len(sw) != len(dw):
            return
        new = []
        for s, d in zip(sw, dw):
            s = s.astype(d.dtype, copy=False)
            new.append(_slice_to_overlap(s, d))
        dst_layer.set_weights(new)
    except Exception:
        return

# --- architecture-aware pruner for CNN→LSTM ---
def prune_cnn_lstm_architecture_aware(
    original_model=None,
    X_val=None,
    y_val=None,
    prune_ratio=0.30,          # modest cut; keep it simple
    min_conv_filters=32,
    min_lstm_units=24,
    min_dense_units=16,
    kernel_size=3,
    dropout=0.20,
    **kwargs,                  # tolerate extras safely
):
    """
    Rebuild a smaller CNN→LSTM by reducing Conv1D filters, LSTM units, and Dense width.
    Preserves topology: 2×Conv1D → Dropout → 2×LSTM → Dense(32) → Dense(1) (with reduced widths).
    Returns a compiled pruned model.
    """
    if original_model is None:
        raise ValueError("original_model is required to infer input shape.")

    # Coerce knobs
    prune_ratio     = _scalar(prune_ratio, float, "prune_ratio")
    min_conv_filters= _scalar(min_conv_filters, int,  "min_conv_filters")
    min_lstm_units  = _scalar(min_lstm_units, int,    "min_lstm_units")
    min_dense_units = _scalar(min_dense_units, int,   "min_dense_units")
    kernel_size     = _scalar(kernel_size, int,       "kernel_size")
    dropout         = _scalar(dropout, float,         "dropout")
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in (0,1)")

    input_shape = original_model.input_shape[1:]

    # Infer baselines from the original model (fallbacks if not found)
    base_conv = []
    base_lstm = []
    base_dense = None
    for lyr in original_model.layers:
        if isinstance(lyr, tf.keras.layers.Conv1D):
            base_conv.append(int(lyr.filters))
        elif isinstance(lyr, tf.keras.layers.LSTM):
            base_lstm.append(int(lyr.units))
        elif isinstance(lyr, tf.keras.layers.Dense) and base_dense is None and lyr.units != 1:
            base_dense = int(lyr.units)
    if len(base_conv) < 2: base_conv = [64, 64]
    if len(base_lstm) < 2: base_lstm = [50, 50]
    if base_dense is None: base_dense = 32

    # Compute pruned widths
    pr_conv = max(min_conv_filters, int(base_conv[0] * (1.0 - prune_ratio)))
    pr_conv2= max(min_conv_filters, int(base_conv[1] * (1.0 - prune_ratio)))
    pr_lstm1= max(min_lstm_units,   int(base_lstm[0] * (1.0 - prune_ratio)))
    pr_lstm2= max(min_lstm_units,   int(base_lstm[1] * (1.0 - prune_ratio)))
    pr_dense= max(min_dense_units,  int(base_dense   * (1.0 - prune_ratio)))

    # Ensure kernel is odd for causal symmetry
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Build pruned CNN→LSTM
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(pr_conv,  kernel_size, activation='relu', padding='causal')(inp)
    x = layers.Conv1D(pr_conv2, kernel_size, activation='relu', padding='causal')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(pr_lstm1, return_sequences=True)(x)
    x = layers.LSTM(pr_lstm2)(x)
    x = layers.Dense(pr_dense, activation='relu')(x)
    out = layers.Dense(1)(x)
    pruned_model = models.Model(inp, out)

    # Optional: partial weight transfer for Conv1D & Dense (skip LSTM for simplicity)
    src_layers = [l for l in original_model.layers if isinstance(l, (tf.keras.layers.Conv1D, tf.keras.layers.Dense))]
    dst_layers = [l for l in pruned_model.layers  if isinstance(l, (tf.keras.layers.Conv1D, tf.keras.layers.Dense))]
    for s, d in zip(src_layers, dst_layers):
        if type(s) is type(d):
            _safe_copy_weights(s, d)

    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])

    # Metadata for your dashboard
    pruned_model.pruning_importances = {
        "conv1_filters": pr_conv,
        "conv2_filters": pr_conv2,
        "lstm1_units": pr_lstm1,
        "lstm2_units": pr_lstm2,
        "dense_units": pr_dense,
        "kernel_size": kernel_size,
        "dropout": dropout,
    }
    return pruned_model


#-------------------------------------------------------------
#  Buildinf CNN-LSTM Model
#-------------------------------------------------------------
from tensorflow.keras import layers

def build_cnn_lstm_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='causal')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.LSTM(50, return_sequences=True)(x)
    x = layers.LSTM(50)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


#------------------------------------------------------------
#  Displaying Model Summary in Streamlit
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


#------------------------------------------------------------
#  Main Streamlit App
#------------------------------------------------------------

def run_cnn_lstm_with_mda_pruning(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    Complete CNN-LSTM pipeline with MDA feature selection.
    """
    
    # =========================================
    # STEP 1: DATA LOADING AND VISUALIZATION
    # =========================================
    with st.spinner("Fetching and preparing data..."):
        df = load_stock_data(ticker, str(start_date), str(end_date), f"{ticker}_{start_date}_to_{end_date}.csv")

    st.write("### Sample of Downloaded Data")
    st.dataframe(df.head(10))

    st.write("### Visualizing Feature Distributions")
    plot_feature_distribution(df)

    st.write("### Visualizing Price Trends")
    plot_close_price_history(df)

    # =====================================
    # STEP 2: FEATURE ENGINEERING
    # =====================================
    with st.spinner("Computing technical features..."):
        df_features = compute_features(df)

    st.write("### Feature-Engineered Data Sample")
    st.dataframe(df_features.head(5))

    # ========================================
    # STEP 3: SENTIMENT DATA INTEGRATION
    # ========================================
    with st.spinner("Loading and processing sentiment data..."):
        daily_sent = prepare_daily_sentiment_features("data/synthetic_financial_tweets_labeled.AAPL.csv")
        if daily_sent is None:
            st.stop()

    with st.spinner("Aligning price and sentiment data..."):
        full_df = align_final_dataframe()
        if full_df is None:
            st.stop() 

             # Display feature distribution
    st.write(f"### Plotting  Features Distribution:**")
    plot_features_distribution(full_df, max_plots=20)

    # =========================================================================
    # STEP 4: MDA FEATURE SELECTION
    # =========================================================================
    st.header(" MDA Feature Selection")
    
    results = complete_ml_pipeline(
        csv_path="data/full_Dataframe.csv",
        test_frac=0.15,
        min_mda_threshold=0.0,
        horizon=1,
        seed=42
    )

    if not results:
        st.error("MDA Pipeline failed")
        return None
        
    selected_features = results['selected_features']
    feature_importance = results['feature_importance']

    #  ADDing  VALIDATION CHECK HERE 
    if 'Close' not in selected_features and 'Close' in full_df.columns:
        selected_features.append('Close')
        st.warning("Added 'Close' to features as prediction target")
    elif 'Close' not in full_df.columns:
        st.error("Critical error: 'Close' price data not found in dataframe")

        # Create immutable feature order
    FEATURE_ORDER = selected_features.copy()

    # Ensure Close is first
    if 'Close' in FEATURE_ORDER and FEATURE_ORDER[0] != 'Close':
        FEATURE_ORDER.remove('Close')
        FEATURE_ORDER = ['Close'] + FEATURE_ORDER
        st.info("Enforced feature order with 'Close' first")
    elif 'Close' not in FEATURE_ORDER:
        FEATURE_ORDER = ['Close'] + FEATURE_ORDER
        st.warning("Added 'Close' to feature order")

   
    
    st.success(f"MDA Pipeline completed! Selected {len(selected_features)} features.")
    
    # =========================================================================
    # STEP 5: PREPARE CNN-LSTM TRAINING DATA
    # =========================================================================
    st.header(" Preparing CNN-LSTM Training Data")
    
    with st.spinner("Creating training features..."):
        training_features = create_training_features(full_df, selected_features)

    with st.spinner("Splitting train/test data..."):
        train_data, test_data = train_test_split_timeseries(training_features, test_frac=0.15)

    with st.spinner("Scaling features..."):
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
    st.info(f"Scaled data shapes - Train: {train_scaled.shape}, Test: {test_scaled.shape}")

    with st.spinner("Creating sequences for CNN-LSTM..."):
        X_train, y_train = make_sequences(train_scaled, window=60, target_index=0)
        X_val, y_val = make_sequences(test_scaled, window=60, target_index=0)

    st.success(f"""
    **Sequences Created for CNN-LSTM:**
    - X_train: {X_train.shape} (samples, window, features)
    - y_train: {y_train.shape} (samples,)
    - X_val: {X_val.shape} (samples, window, features)  
    - y_val: {y_val.shape} (samples,)
    """)
    
    # =========================================================================
    # STEP 6: BUILD CNN-LSTM MODEL
    # =========================================================================
    st.header(" CNN-LSTM Model Architecture")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    st.info(f"CNN-LSTM Input Shape: {input_shape}")

    with st.spinner("Building CNN-LSTM model..."):
        model = build_cnn_lstm_model(input_shape)

    st.write("### CNN-LSTM Model Summary")
    display_model_summary(model)

    # ==================================
    # STEP 7: MODEL TRAINING
    # ==================================
    st.header(" Model Training")
    
     # Enhanced callbacks
    callbacks_list = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    with st.spinner("Training the CNN-LSTM model..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=2,
        )
    
    st.success("Model training completed.")

    # ==========================================
    # STEP 8: TRAINING HISTORY VISUALIZATION
    # ===========================================
    st.header(" Training History")
    
    history_df = pd.DataFrame(history.history)

    fig, ax = plt.subplots(figsize=(8, 4))
    history_df['loss'].plot(ax=ax, label='Training Loss')
    history_df['val_loss'].plot(ax=ax, label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # =======================================
    # STEP 9: PREDICTIONS AND EVALUATION 
    # ======================================
    st.header(" Predictions & Evaluation")
    
    with st.spinner("Making predictions..."):
        full_feature_list = list(training_features.columns)
        
        y_true, y_pred, aligned_dates, full_test = make_predictions(
            model=model,
            scaler=scaler,
            train_df=train_data,
            test_df=test_data,
            FEATURES=tuple(FEATURE_ORDER),
            LOOK_BACK=60
        )

    st.success(f"✅ Predictions completed! Generated {len(y_pred)} predictions.")
    
    # Display first 10 predictions
    results_df = pd.DataFrame({
        'True Close': y_true[:10],
        'Predicted Close': y_pred[:10],
        'Difference': y_pred[:10] - y_true[:10],
        'Error %': ((y_pred[:10] - y_true[:10]) / y_true[:10] * 100).round(2)
    })
    
    st.write("###  True vs Predicted Close Prices (First 10)")
    st.dataframe(results_df)
    
    # Plot predictions
    st.write("###  Prediction vs Actual Plot")
    plot_model_prediction(aligned_dates, y_true, y_pred)
    
    # Model evaluation metrics
    st.write("###  Model Evaluation Metrics")
    # metrics_df = evaluate_model(y_true, y_pred)
    metrics_result = evaluate_model(y_true, y_pred)
    
    # FIXED: Convert dict to DataFrame for CSV download
    if isinstance(metrics_result, dict):
        # Convert dictionary to DataFrame for display and download
        metrics_df = pd.DataFrame([
            {"Metric": key, "Value": value} 
            for key, value in metrics_result.items()
        ])
    else:
        # If it's already a DataFrame, use it as-is
     metrics_df = metrics_result
    
    
    # Download metrics
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Metrics as CSV",
        data=csv_metrics,
        file_name=f"{ticker}_cnn_lstm_metrics.csv",
        mime='text/csv'
    )
    
# =========================================================================
    # STEP 10: INTELLIGENT PRUNING WITH COMPREHENSIVE VISUALIZATION (FIXED)
    # =========================================================================
    st.header(" Intelligent Model Pruning & Analysis")
    
    # FIXED: Proper metrics conversion
    try:
        if isinstance(metrics_df, pd.DataFrame):
            # Add proper metric names to the DataFrame
            metric_names = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'Directional_Accuracy']
            metrics_with_names = metrics_df.copy()
            if len(metrics_with_names) <= len(metric_names):
                metrics_with_names.index = metric_names[:len(metrics_with_names)]
            
            # Convert to dictionary
            if 'Value' in metrics_with_names.columns:
                original_metrics_dict = metrics_with_names['Value'].to_dict()
            else:
                original_metrics_dict = metrics_with_names.iloc[:, 0].to_dict()
        else:
            original_metrics_dict = metrics_df
            
        # st.write("###  Debug: Original Metrics Format")
        # st.write(f"**Type:** {type(original_metrics_dict)}")
        # st.write(f"**Content:** {original_metrics_dict}")
        
    except Exception as e:
        st.error(f"Metrics conversion failed: {e}")
        st.write("Using fallback metrics format...")
        original_metrics_dict = {
            'RMSE': 0.1,  # Fallback values
            'MAE': 0.1,
            'R2': 0.8,
            'MAPE': 5.0
        }
    
    # Run enhanced pruning with comprehensive visualization
    try:
        pruning_results = enhanced_pruning_with_comprehensive_visualization(
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            training_features=training_features,
            train_data=train_data,
            test_data=test_data,
            original_metrics=original_metrics_dict,
            original_history=history,
            aligned_dates=aligned_dates,
            ticker=ticker,
            y_true_orig=y_true,
            y_pred_orig=y_pred,
            # Add architecture-aware specific parameters
            pruning_function=prune_cnn_lstm_architecture_aware,
            pruning_kwargs={'prune_ratio': 0.20,
                            'min_conv_filters': 32,
                           'min_lstm_units': 24,
                           'min_dense_units': 16,
                           'kernel_size': 3,
                           'dropout': 0.20}, 
            scaler=scaler  # Pass scaler for inverse transformation
        )

        
        # Extract final results
        final_model = pruning_results['final_model']
        final_features = pruning_results['final_features']
        final_predictions = pruning_results['final_predictions']
        final_metrics = pruning_results['final_metrics']
        
        pruning_success = True
        
    except Exception as e:
        st.error(f" Pruning pipeline failed: {str(e)}")
        st.info("Continuing with original model results...")
        
        # Fallback to original results
        final_model = model
        final_features = selected_features
        final_predictions = {'y_true': y_true, 'y_pred': y_pred}
        final_metrics = original_metrics_dict
        pruning_success = False
       
    #**************************************************************************
    #**************************************************************************


    # =========================================================================
    # STEP 11: FINAL SUMMARY (UPDATED)
    # =========================================================================
    st.header("  Complete Pipeline Summary")
        
    if pruning_success:
        st.success(f"""
        ** Complete CNN-LSTM Pipeline with Intelligent Pruning:**
            
        **Data:** {ticker} from {start_date} to {end_date}
        **Feature Journey:** 
        - Original: All features → MDA: {len(selected_features)} → Final: {len(final_features)}
        **Models:** Original CNN-LSTM + Intelligently Pruned CNN-LSTM
        **Training:** Comprehensive training with early stopping
        **Predictions:** {len(final_predictions['y_pred'])} optimized predictions
            
        ** Final Selected Features:** {', '.join(final_features)}
        """)
    else:
        st.warning(f"""
        ** CNN-LSTM Pipeline (Pruning Failed):**
            
        **Data:** {ticker} from {start_date} to {end_date}
        **Features:** {len(selected_features)} selected via MDA
        **Model:** CNN-LSTM with {input_shape} input shape
        **Training:** {len(history.history['loss'])} epochs (Early stopping)
        **Predictions:** {len(y_pred)} test predictions generated
            
        **Selected Features:** {', '.join(selected_features)}
        """)
        
        # Enhanced return with pruning results
    return {
        'original_model': model,
        'final_model': final_model,
        'training_history': history,
        'feature_evolution': {
            'all_features': list(full_df.columns),
            'mda_selected': selected_features,
            'final_features': final_features
        },
        'predictions': {
            'original': {'y_true': y_true, 'y_pred': y_pred},
            'final': final_predictions
        },
        'metrics': {
            'original': original_metrics_dict,
            'final': final_metrics
        },
        # 'importance_scores': pruning_results('importances', {}) if pruning_success else {},
        'scaler': scaler,
        'comprehensive_analysis': pruning_success,
        'pruning_success': pruning_success
    }
