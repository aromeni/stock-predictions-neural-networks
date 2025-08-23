# approaches/approach_9.py

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
#  Build LSTM-Transformer Model
#------------------------------------------------------------

def build_simple_lstm_transformer_model(input_shape):
    
    inputs = layers.Input(shape=input_shape)

    # First LSTM: return sequences for transformer
    x = layers.LSTM(units=50, return_sequences=True)(inputs)
    
    # Second LSTM: also return sequences for attention
    lstm_out = layers.LSTM(units=50, return_sequences=True)(x)
    
    # Just one attention layer - no complex FFN or multiple blocks
    attention_out = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=50//4,  # key_dim = lstm_units / num_heads
    )(lstm_out, lstm_out)
    
    # Simple residual connection (no layer norm to keep it minimal)
    x = lstm_out + attention_out
    
    # Take the last timestep (like your original LSTM)
    x = layers.Lambda(lambda t: t[:, -1, :])(x)  # last timestep, Keras-safe

    # x = x[:, -1, :]  # Shape: (batch_size, 50)
    
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)  # regression output
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    
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
def prune_simple_lstm_transformer_model(
    original_model=None,
    X_val=None,
    y_val=None,
    prune_ratio=0.15,    #prune_ratio ∈ {0.15, 0.25, 0.35} and pick best on validation.
    min_lstm_units=16,
    min_dense_units=8,
    num_heads=4
):
    """
    Architecture-aware pruning for the simple LSTM+MultiHeadAttention model.
    Strategy: rebuild a smaller model (no fragile weight slicing), then retrain.

    Args:
        original_model: baseline Keras model (unused except for input shape)
        X_val, y_val: present for a consistent signature with the wrapper
        prune_ratio (0..1): fraction to prune from LSTM and Dense widths
        min_lstm_units: floor to avoid collapsing the network
        min_dense_units: floor for the penultimate Dense layer
        num_heads: attention heads (keep constant)

    Returns:
        A freshly compiled, smaller Keras model with the same input shape.
    """
    # Infer input shape from the original model (batch, timesteps, features)
    if original_model is None:
        raise ValueError("original_model is required to infer input shape.")

    input_shape = original_model.input_shape[1:]  # (timesteps, features)

    # Baseline hyperparams from your builder
    base_lstm_units = 50
    base_dense_units = 32

    # Compute pruned sizes
    pruned_lstm_units = max(min_lstm_units, int(base_lstm_units * (1 - prune_ratio)))
    pruned_dense_units = max(min_dense_units, int(base_dense_units * (1 - prune_ratio)))

   

    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units=pruned_lstm_units, return_sequences=True)(inputs)
    lstm_out = layers.LSTM(units=pruned_lstm_units, return_sequences=True)(x)
    

        # infer baseline LSTM-Transformer width if possible; fall back to 50
    try:
        base_units = next(l.units for l in original_model.layers if isinstance(l, tf.keras.layers.LSTM))
    except StopIteration:
        base_units = 50

    BASE_KEY_DIM = max(1, base_units // num_heads)
    key_dim      = max(BASE_KEY_DIM, pruned_lstm_units // num_heads)

    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(lstm_out, lstm_out)

    x = lstm_out + attn_out
    x = x[:, -1, :]  # last timestep
    x = layers.Dense(pruned_dense_units, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    pruned_model = models.Model(inputs, outputs)
    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    # expose metadata for the dashboard
    pruned_model.pruning_importances = {
        "lstm_units": pruned_lstm_units,
        "dense_units": pruned_dense_units,
        "num_heads": num_heads,
        "key_dim": key_dim
    }

    return pruned_model






#------------------------------------------------------------
#  Main Streamlit App
#------------------------------------------------------------

def run_simple_lstm_transformer_with_mda_pruning(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    Complete LSTM-Transformer pipeline with MDA feature selection.
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
    
    st.success(f"MDA Pipeline completed! Selected {len(selected_features)} features.")
    
    # =========================================================================
    # STEP 5: PREPARE LSTM-Transformer TRAINING DATA
    # =========================================================================
    st.header(" Preparing LSTM-Transformer Training Data")
    
    with st.spinner("Creating training features..."):
        training_features = create_training_features(full_df, selected_features)

    with st.spinner("Splitting train/test data..."):
        train_data, test_data = train_test_split_timeseries(training_features, test_frac=0.15)

    with st.spinner("Scaling features..."):
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
    st.info(f"Scaled data shapes - Train: {train_scaled.shape}, Test: {test_scaled.shape}")

    with st.spinner("Creating sequences for LSTM-Transformer model..."):
        X_train, y_train = make_sequences(train_scaled, window=60, target_index=0)
        X_val, y_val = make_sequences(test_scaled, window=60, target_index=0)

    st.success(f"""
    **Sequences Created for LSTM-Transformer:**
    - X_train: {X_train.shape} (samples, window, features)
    - y_train: {y_train.shape} (samples,)
    - X_val: {X_val.shape} (samples, window, features)  
    - y_val: {y_val.shape} (samples,)
    """)
    
    # =========================================================================
    # STEP 6: LSTM-Transformer MODEL
    # =========================================================================
    st.header(" LSTM-Transformer Model Architecture")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    st.info(f"LSTM-Transformer Input Shape: {input_shape}")

    with st.spinner("Building LSTM-Transformer model..."):
        model = build_simple_lstm_transformer_model(input_shape)

    st.write("### LSTM-Transformer Model Summary")
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
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=15),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]



    with st.spinner("Training the LSTM-Transformer model..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
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
            FEATURES=full_feature_list,
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
    


    # st.dataframe(metrics_df)  # COMMENTED THIS LINE ---might
    
    # Download metrics
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=" Download Metrics as CSV",
        data=csv_metrics,
        file_name=f"{ticker}_lstm_transformer_metrics.csv",
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
            #New: plusing in the architeccture-specific pruning function
            pruning_function=prune_simple_lstm_transformer_model,
            pruning_kwargs={
                "prune_ratio": 0.25,
                "min_lstm_units": 24,   
                "min_dense_units": 16,
                "num_heads": 4  # Keeping the same number of heads          
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
        ** Complete LSTM-Transformer Pipeline with Intelligent Pruning:**
        
         **Data:** {ticker} from {start_date} to {end_date}
         **Feature Journey:** 
            - Original: All features → MDA: {len(selected_features)} → Final: {len(final_features)}
         **Models:** Original LSTM-Transformer + Intelligently Pruned LSTM-Transformer
         **Training:** Comprehensive training with early stopping
         **Predictions:** {len(final_predictions['y_pred'])} optimized predictions
        
        ** Final Selected Features:** {', '.join(final_features)}
        """)
    else:
        st.warning(f"""
        ** LSTM-Transformer Pipeline (Pruning Failed):**
        
         **Data:** {ticker} from {start_date} to {end_date}
         **Features:** {len(selected_features)} selected via MDA
         **Model:** LSTM-Transformer with {input_shape} input shape
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








