# approaches/approach_5.py

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
    plot_features_distribution,
    complete_ml_pipeline,
    make_predictions,
    evaluate_model,
    plot_model_prediction,
    make_sequences,
    create_training_features,
    train_test_split_timeseries,
    enhanced_pruning_with_comprehensive_visualization
)

# #-------------------------------------------------------------
# #  TCN Pruning Function
def prune_conv1d_layer(conv_layer, sensitivity):
    """Core 1D convolution pruning helper"""
    weights = conv_layer.get_weights()
    if not weights:
        return
    
    kernel = weights[0]
    time_importance = np.array([0.1, 0.8, 0.1])  # Center-focused importance
    if kernel.shape[0] > 3:
        time_importance = np.array([0.1, 0.2, 0.4, 0.2, 0.1])[:kernel.shape[0]]
    
    channel_norms = np.linalg.norm(kernel, axis=(0,1))
    importance_matrix = (
        np.abs(kernel) * 
        time_importance[:, None, None] * 
        channel_norms[None, None, :]
    )
    
    threshold = np.percentile(importance_matrix, sensitivity * 100)
    mask = importance_matrix > threshold
    weights[0] = kernel * mask
    conv_layer.set_weights(weights)

def temporal_tcn_pruning(original_model, X_val=None, y_val=None, sensitivity=0.25):
    """
    Architecture-aware pruning for Temporal Convolutional Networks
    Preserves critical temporal patterns while reducing complexity
    """
    pruned_model = tf.keras.models.clone_model(original_model)
    pruned_model.set_weights(original_model.get_weights())
    
    # Dilation-specific sensitivities (protect long-term patterns)
    dilation_sensitivities = {1:0.2, 2:0.25, 4:0.3, 8:0.35, 16:0.4, 32:0.45, 64:0.5}
    
    for layer in pruned_model.layers:
        if isinstance(layer, TCN):
            for block in layer.residual_blocks:
                # Get dilation rate
                dilation = block.conv1.dilation_rate[0]
                
                # Apply dilation-specific sensitivity
                block_sensitivity = dilation_sensitivities.get(dilation, sensitivity)
                prune_conv1d_layer(block.conv1, block_sensitivity)
                
                # More conservative for residual connection
                prune_conv1d_layer(block.conv2, block_sensitivity * 0.5)
    
    return pruned_model

# #-------------------------------------------------------------
# #  Build TCN Model
# #-------------------------------------------------------------


from tcn import TCN
def build_tcn_model(input_shape):
    model = tf.keras.Sequential([
        TCN(input_shape=input_shape,
            dilations=(1, 2, 4, 8, 16,32,64),
            nb_filters=64,
            kernel_size=3,
            dropout_rate=0.1),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# #------------------------------------------------------------
# #  Display Model Summary in Streamlit
# #------------------------------------------------------------

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
#  Main Streamlit App  for run_tcn_with_mda_pruning
#------------------------------------------------------------


def run_tcn_with_mda_pruning(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    Complete TCN pipeline with MDA feature selection.
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
     

    if 'Close' not in full_df.columns:
        st.error("'Close' column missing in final dataframe! Using fallback...")
        full_df['Close'] = full_df.get('close', full_df.get('Close', full_df.iloc[:, 0]))


     # Display feature distribution
    st.write(f"### Plotting  Features Distribution:**")
    plot_features_distribution(full_df, max_plots=20)


    # =========================================================================
    # STEP 4: MDA FEATURE SELECTION
    # =========================================================================
    st.header(" MDA Feature Selection")
    
    results = complete_ml_pipeline(
        csv_path="full_Dataframe.csv",
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
    # STEP 5: PREPARE TCN TRAINING DATA
    # =========================================================================
    st.header(" Preparing TCN Training Data")
    
    with st.spinner("Creating training features..."):
        training_features = create_training_features(full_df, selected_features)

    with st.spinner("Splitting train/test data..."):
        train_data, test_data = train_test_split_timeseries(training_features, test_frac=0.15)

    with st.spinner("Scaling features..."):
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
    st.info(f"Scaled data shapes - Train: {train_scaled.shape}, Test: {test_scaled.shape}")

    with st.spinner("Creating sequences for TCN..."):
        X_train, y_train = make_sequences(train_scaled, window=60, target_index=0)
        X_val, y_val = make_sequences(test_scaled, window=60, target_index=0)

    st.success(f"""
    **Sequences Created for TCN:**
    - X_train: {X_train.shape} (samples, window, features)
    - y_train: {y_train.shape} (samples,)
    - X_val: {X_val.shape} (samples, window, features)  
    - y_val: {y_val.shape} (samples,)
    """)
    
    # =========================================================================
    # STEP 6: TCN MODEL
    # =========================================================================
    st.header(" TCN Model Architecture")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    st.info(f"TCN Input Shape: {input_shape}")

    with st.spinner("Building TCN model..."):
        model = build_tcn_model(input_shape)

    st.write("### TCN Model Summary")
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
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

    with st.spinner("Training the TCN model..."):
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
        file_name=f"{ticker}_cnn_lstm_metrics.csv",
        mime='text/csv'
    )
    
    # =========================================================================
    # STEP 10: INTELLIGENT PRUNING WITH COMPREHENSIVE VISUALIZATION (FIXED)
    # =========================================================================
    st.header(" Intelligent Model Pruning & Analysis")
       # Define TCN-simple pruning function

    def temporal_tcn_pruning(original_model, X_val=None, y_val=None, sensitivity=0.25):
        """
        Robust TCN pruning compatible with different TCN implementations
        """
        pruned_model = tf.keras.models.clone_model(original_model)
        pruned_model.set_weights(original_model.get_weights())
    
        # Dilation-specific sensitivities (protect long-term patterns)
        dilation_sensitivities = {1:0.2, 2:0.25, 4:0.3, 8:0.35, 16:0.4, 32:0.45, 64:0.5}
    
        for layer in pruned_model.layers:
            if isinstance(layer, TCN):
            # Version-agnostic residual block access
             residual_blocks = getattr(layer, 'residual_blocks', [])
             if not residual_blocks:
                # Try alternative attribute names
                residual_blocks = getattr(layer, 'blocks', [])
                if not residual_blocks:
                    # Fallback to layer scanning
                    residual_blocks = [l for l in layer.layers 
                                       if 'residual' in l.name.lower() or 'block' in l.name.lower()]
            
            for block in residual_blocks:
                # Find dilated convolution layers
                conv_layers = [l for l in block.layers if isinstance(l, layers.Conv1D)]
                
                if not conv_layers:
                    continue
                    
                # Identify main temporal convolution (largest dilation)
                main_conv = max(conv_layers, key=lambda c: c.dilation_rate[0])
                dilation = main_conv.dilation_rate[0]
                
                # Apply dilation-specific sensitivity
                block_sensitivity = dilation_sensitivities.get(dilation, sensitivity)
                prune_conv1d_layer(main_conv, block_sensitivity)
                
                # Find residual connection (1x1 convolution)
                residual_conv = next((c for c in conv_layers 
                                      if c.kernel_size[0] == 1 and c != main_conv), None)
                if residual_conv:
                    prune_conv1d_layer(residual_conv, block_sensitivity * 0.5)
    
        return pruned_model
    


    # Configure pruning parameters
    pruning_kwargs = {'sensitivity': 0.25}    # Conservative default
                       
    # FIXED: Proper metrics conversion
    try:
        # Convert metrics DataFrame to dictionary format
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
            pruning_function=temporal_tcn_pruning,
            pruning_kwargs=pruning_kwargs,
            scaler=scaler,
                )
        # Show weight distribution before/after pruning
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Original weights
        original_weights = model.layers[1].get_weights()[0].flatten()
        ax1.hist(original_weights, bins=50, alpha=0.7)
        ax1.set_title('Original Weights')

        # Pruned weights
        pruned_weights = pruning_results['final_model'].layers[1].get_weights()[0].flatten()
        ax2.hist(pruned_weights, bins=50, alpha=0.7)
        ax2.set_title('Pruned Weights')
                
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
        ** Complete TCN Pipeline with Intelligent Pruning:**
        
         **Data:** {ticker} from {start_date} to {end_date}
         **Feature Journey:** 
            - Original: All features → MDA: {len(selected_features)} → Final: {len(final_features)}
         **Models:** Original TCN + Intelligently Pruned TCN
         **Training:** Comprehensive training with early stopping
         **Predictions:** {len(final_predictions['y_pred'])} optimized predictions
        
        ** Final Selected Features:** {', '.join(final_features)}
        """)
    else:
        st.warning(f"""
        ** TCN Pipeline (Pruning Failed):**
        
         **Data:** {ticker} from {start_date} to {end_date}
         **Features:** {len(selected_features)} selected via MDA
         **Model:** TCN with {input_shape} input shape
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




