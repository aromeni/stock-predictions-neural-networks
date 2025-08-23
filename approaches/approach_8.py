# approaches/approach_8.py

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
    prepare_daily_sentiment_features,
    align_final_dataframe,
    complete_ml_pipeline,
    make_predictions,
    evaluate_model,
    plot_model_prediction,
    make_sequences,
    create_training_features,
    train_test_split_timeseries,
    plot_features_distribution,
     enhanced_pruning_with_comprehensive_visualization,
)





#------------------------------------------------------------

def _slice_to_overlap(src_arr, dst_arr):
    """Copy the common top-left hyperrectangle from src into dst (no errors)."""
    sl = tuple(slice(0, min(s, d)) for s, d in zip(src_arr.shape, dst_arr.shape))
    out = dst_arr.copy()
    out[sl] = src_arr[sl]
    return out

def safe_copy_weights(src_layer, dst_layer):
    """
    Partially transfer weights from src_layer to dst_layer by overlapping shapes.
    Works for Conv1D and Dense. Silently skips if shapes/counts don’t align.
    """
    try:
        src_w = src_layer.get_weights()
        dst_w = dst_layer.get_weights()
        if not src_w or not dst_w or len(src_w) != len(dst_w):
            return
        new_weights = []
        for s_arr, d_arr in zip(src_w, dst_w):
            s_arr = s_arr.astype(d_arr.dtype, copy=False)
            new_weights.append(_slice_to_overlap(s_arr, d_arr))
        dst_layer.set_weights(new_weights)
    except Exception:
        return  # never let transfer break the run


def _scalar(x, typ, name):
    """
    Coerce lists/tuples/arrays/Series to a single scalar of type `typ`.
    If a sequence is passed, take the first element.
    """
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        if len(x) == 0:
            raise ValueError(f"{name} is empty")
        x = x[0]
    try:
        return typ(x)
    except Exception as e:
        raise TypeError(f"{name} must be {typ.__name__}, got {type(x)} with value {x}") from e

def _tuple_of_ints(v, name):
    """
    Ensure `v` is a tuple of ints (for dilations). Accepts list/tuple/ndarray/Series/scalar.
    For a scalar, returns a 1-tuple.
    """
    if isinstance(v, (np.ndarray, pd.Series)):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return tuple(int(int(x)) for x in v)
    return (int(v),)




#------------------------------------------------------------
#  Build TCN→LSTM Model
#------------------------------------------------------------
def build_tcn_lstm_model(input_shape,
                         filters=64,                 # 128 -> 64
                         kernel_size=5,              # 7  -> 5
                         stacks=2,                   # 3  -> 2
                         dilations=(1, 2, 4, 8),     # keep 4 rates per stack
                         lstm_units=64,              # 128 -> 64
                         dropout=0.20,               # stronger regularization
                         l2_reg=1e-4,                # stronger L2
                         lr=3e-4):                   # gentler LR
    inp = layers.Input(shape=input_shape)
    x = inp
    for _ in range(stacks):
        for d in dilations:
            y = layers.Conv1D(filters, kernel_size,
                              padding="causal",
                              dilation_rate=int(d),
                              activation="relu",
                              kernel_regularizer=regularizers.l2(l2_reg))(x)
            y = layers.SpatialDropout1D(dropout)(y)   # Spatial dropout works better for conv features
            if x.shape[-1] != y.shape[-1]:
                x = layers.Conv1D(filters, 1, padding="same")(x)
            x = layers.Add()([x, y])
            x = layers.LayerNormalization()(x)
    x = layers.LSTM(lstm_units, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
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


#------------------------------------------------------------
#  Pruning Function Specific to TCN→LSTM
#------------------------------------------------------------

def prune_tcn_lstm_architecture_aware(
    original_model=None,
    X_val=None,
    y_val=None,
    prune_ratio=0.25,          # 0.35 -> 0.25 (gentler)
    min_filters=32,            # 16  -> 32 (avoid collapsing capacity)
    min_lstm_units=32,         # 24  -> 32
    kernel_size=5,
    stacks=2,
    dilations=(1, 2, 4, 8),
    dropout=0.20,
    l2_reg=1e-4,
    **kwargs,
):
    if original_model is None:
        raise ValueError("original_model is required to infer input shape.")
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in (0, 1)")

    input_shape = original_model.input_shape[1:]

    # infer baselines (fallbacks if not present)
    base_filters = next((int(l.filters) for l in original_model.layers
                         if isinstance(l, tf.keras.layers.Conv1D)), 64)
    base_lstm_units = next((int(l.units) for l in original_model.layers
                            if isinstance(l, tf.keras.layers.LSTM)), 64)

    pruned_filters    = max(min_filters,    int(base_filters * (1.0 - prune_ratio)))
    pruned_lstm_units = max(min_lstm_units, int(base_lstm_units * (1.0 - prune_ratio)))

    if kernel_size % 2 == 0:
        kernel_size += 1

    inp = layers.Input(shape=input_shape)
    x = inp
    for _ in range(stacks):
        for d in dilations:
            y = layers.Conv1D(pruned_filters, kernel_size,
                              padding="causal",
                              dilation_rate=int(d),
                              activation="relu",
                              kernel_regularizer=regularizers.l2(l2_reg))(x)
            y = layers.SpatialDropout1D(dropout)(y)
            if x.shape[-1] != y.shape[-1]:
                x = layers.Conv1D(pruned_filters, 1, padding="same")(x)
            x = layers.Add()([x, y])
            x = layers.LayerNormalization()(x)

    x = layers.LSTM(pruned_lstm_units, return_sequences=False,
                    kernel_regularizer=regularizers.l2(l2_reg))(x)
    out = layers.Dense(1)(x)
    pruned_model = models.Model(inp, out)

    # partial weight transfer (Conv1D/Dense)
    CLASS_WHITELIST = (tf.keras.layers.Conv1D, tf.keras.layers.Dense)
    src_layers = [l for l in original_model.layers if isinstance(l, CLASS_WHITELIST)]
    dst_layers = [l for l in pruned_model.layers  if isinstance(l, CLASS_WHITELIST)]
    for s, d in zip(src_layers, dst_layers):
        safe_copy_weights(s, d)

    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="mse", metrics=["mae"])
    pruned_model.pruning_importances = {
        "conv_filters": pruned_filters,
        "lstm_units": pruned_lstm_units,
        "kernel_size": kernel_size,
        "stacks": int(stacks),
    }
    return pruned_model








#------------------------------------------------------------
#  Main Streamlit App
#------------------------------------------------------------

def run_tcn_lstm_with_mda_pruning(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    Complete TCN→LSTM-Transformer pipeline with MDA feature selection.
    """
    
    # =========================================================================
    # STEP 1: DATA LOADING AND VISUALIZATION
    # =========================================================================
    with st.spinner("Fetching and preparing data..."):
        df = load_stock_data(ticker, str(start_date), str(end_date), f"data/{ticker}_{start_date}_to_{end_date}.csv")

    st.write("### Sample of Downloaded Data")
    st.dataframe(df.head(10))

    st.write("### Visualizing Feature Distributions")
    plot_feature_distribution(df)

    st.write("### Visualizing Price Trends")
    plot_close_price_history(df)

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    with st.spinner("Computing technical features..."):
        df_features = compute_features(df)

    st.write("### Feature-Engineered Data Sample")
    st.dataframe(df_features.head(5))

    # =========================================================================
    # STEP 3: SENTIMENT DATA INTEGRATION
    # =========================================================================
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
    # STEP 5: PREPARE TCN→LSTM TRAINING DATA
    # =========================================================================
    st.header(" Preparing TCN→LSTM Training Data")
    
    with st.spinner("Creating training features..."):
        training_features = create_training_features(full_df, selected_features)

    with st.spinner("Splitting train/test data..."):
        train_data, test_data = train_test_split_timeseries(training_features, test_frac=0.15)

    with st.spinner("Scaling features..."):
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
    st.info(f"Scaled data shapes - Train: {train_scaled.shape}, Test: {test_scaled.shape}")

    with st.spinner("Creating sequences for TCN→LSTM..."):
        X_train, y_train = make_sequences(train_scaled, window=60, target_index=0)
        X_val, y_val = make_sequences(test_scaled, window=60, target_index=0)

    st.success(f"""
    **Sequences Created for TCN→LSTM:**
    - X_train: {X_train.shape} (samples, window, features)
    - y_train: {y_train.shape} (samples,)
    - X_val: {X_val.shape} (samples, window, features)  
    - y_val: {y_val.shape} (samples,)
    """)
    
    # =========================================================================
    # STEP 6: TCN→LSTM MODEL
    # =========================================================================
    st.header(" TCN→LSTM Model Architecture")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    st.info(f"TCN→LSTM Input Shape: {input_shape}")

    with st.spinner("Building TCN→LSTM model..."):
        model = build_tcn_lstm_model(
            input_shape=input_shape,
            filters=128,  # Increased for wider model)
            kernel_size=7,
            stacks=3,
            dilations=(1, 2, 4, 8, 16),
            lstm_units=128,  # Increased for wider model
            dropout=0.1,  # Slightly higher dropout for stability
            l2_reg=1e-5  # Regularization to prevent overfitting
        )

    st.write("### TCN→LSTM Model Summary")
    display_model_summary(model)

    # =========================================================================
    # STEP 7: MODEL TRAINING
    # =========================================================================
    st.header(" Model Training")
    
    # es = callbacks.EarlyStopping(
    #     patience=7,
    #     restore_best_weights=True
    # )


    # Enhanced callbacks
    callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    tf.keras.callbacks.ModelCheckpoint("best_tcnlstm.keras", monitor="val_loss", save_best_only=True)
]



    with st.spinner("Training the TCN→LSTM model..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=64,
            callbacks=callbacks_list,
            verbose=2,
        )
    
    st.success("Model training completed.")

    # =========================================================================
    # STEP 8: TRAINING HISTORY VISUALIZATION
    # =========================================================================
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
            FEATURES=tuple(FEATURE_ORDER),  # Immutable feature order. enforce stable order
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
    
    #  Convert dict to DataFrame for CSV download
    if isinstance(metrics_result, dict):
        # Convert dictionary to DataFrame for display and download
        metrics_df = pd.DataFrame([
            {"Metric": key, "Value": value} 
            for key, value in metrics_result.items()
        ])
    else:
        # If it's already a DataFrame, use it as-is
        metrics_df = metrics_result
    


    # st.dataframe(metrics_df)  # COMMENTED THIS LINE ---might
    
    # Download metrics
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Metrics as CSV",
        data=csv_metrics,
        file_name=f"{ticker}_tcn_lstm_metrics.csv",
        mime='text/csv'
    )
    
    # =========================================================================
    # STEP 10: INTELLIGENT PRUNING WITH COMPREHENSIVE VISUALIZATION 
    # =========================================================================
    st.header(" Intelligent Model Pruning & Analysis")
    
    #   metrics conversion
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
            #New: plusing in the architeccture-specific pruning function
            pruning_function=prune_tcn_lstm_architecture_aware,
            pruning_kwargs={
                "prune_ratio": 0.25, # Adjusted for wider pruning
                "min_filters": 32,      # Increased minimum filters
                "min_lstm_units": 32,    # Increased minimum LSTM units
                "kernel_size": 5, # Keeping kernel size odd for symmetry
                "stacks": 2, # Keeping stacks consistent
                "dilations": (1, 2, 4, 8),
                "dropout": 0.20,  # Keeping dropout low for stability 
                "l2_reg": 1e-4,  # Regularization to prevent overfitting       
            },
            scaler=scaler
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
    
    # ========================================
    # STEP 11: FINAL SUMMARY (UPDATED)
    # =================================================
    st.header(" Complete Pipeline Summary")
    
    if pruning_success:
        st.success(f"""
        ** Complete TCN→LSTM Pipeline with Intelligent Pruning:**
        
         **Data:** {ticker} from {start_date} to {end_date}
         **Feature Journey:** 
            - Original: All features → MDA: {len(selected_features)} → Final: {len(final_features)}
         **Models:** Original TCN→LSTM + Intelligently Pruned TCN→LSTM
         **Training:** Comprehensive training with early stopping
         **Predictions:** {len(final_predictions['y_pred'])} optimized predictions
        
        ** Final Selected Features:** {', '.join(final_features)}
        """)
    else:
        st.warning(f"""
        ** TCN→LSTM Pipeline (Pruning Failed):**
        
         **Data:** {ticker} from {start_date} to {end_date}
         **Features:** {len(selected_features)} selected via MDA
         **Model:** TCN→LSTM with {input_shape} input shape
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
        'importance_scores': pruning_results.get('importances', {}) if pruning_success else {},
        'scaler': scaler,
        'comprehensive_analysis': pruning_success,
        'pruning_success': pruning_success
    }








