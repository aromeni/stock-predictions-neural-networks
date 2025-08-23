# approaches/approach_6.py

# ============================
#  Imports & Configuration
# ============================
import os
import time
import yfinance as yf
import streamlit as st
import pandas as pd
import pandas_ta as ta  # technical‑analysis indicators (Bollinger, RSI ..)
import numpy as np
import matplotlib.pyplot as plt

import io
import sys
import tensorflow_model_optimization as tfmot

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models, Model  # FIXED: Added Model import
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

    get_optimal_pruning_config,
    intelligent_feature_selection,
    should_use_pruned_model,
    run_intelligent_pruning_pipeline,
    create_comprehensive_comparison_dashboard,
    enhanced_pruning_with_comprehensive_visualization,
    # prune_retrain_model,
    permutation_importance_seq,
)







from models import (
     build_transformer_model,
    # build_simplified_transformer
)

###############################################################


# ------- small helpers (consistent with your other approaches) -------
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

def _safe_copy_dense(src_layer, dst_layer):
    try:
        sw, dw = src_layer.get_weights(), dst_layer.get_weights()
        if not sw or not dw or len(sw) != len(dw):  # Dense: usually 2 tensors (kernel,bias)
            return
        new = []
        for s, d in zip(sw, dw):
            s = s.astype(d.dtype, copy=False)
            new.append(_slice_to_overlap(s, d))
        dst_layer.set_weights(new)
    except Exception:
        return

# ------- architecture-aware pruner for the Transformer -------
def prune_transformer_architecture_aware(
    original_model=None,
    X_val=None,
    y_val=None,
    prune_ratio=0.30,      # modest default
    min_d_model=32,
    min_heads=2,
    min_ff_dim=64,
    min_layers=1,
    dropout=0.10,
    **kwargs,              # tolerate extra keys
):
    """
    Rebuild a smaller Transformer by reducing d_model, num_heads, ff_dim, and num_layers.
    - Keeps input timesteps, positional embedding length unchanged.
    - Ensures d_model % num_heads == 0 by adjusting num_heads downward if needed.
    - Performs safe partial weight transfer for Dense layers only (projection/FFN/head).
    Returns: compiled pruned model.
    """
    if original_model is None:
        raise ValueError("original_model is required to infer input shape.")

    # ---- infer input shape from original ----
    input_shape = original_model.input_shape[1:]   # (timesteps, features)
    timesteps, in_feats = int(input_shape[0]), int(input_shape[1])

    # ---- coerce knobs ----
    prune_ratio = _scalar(prune_ratio, float, "prune_ratio")
    min_d_model = _scalar(min_d_model, int, "min_d_model")
    min_heads   = _scalar(min_heads,   int, "min_heads")
    min_ff_dim  = _scalar(min_ff_dim,  int, "min_ff_dim")
    min_layers  = _scalar(min_layers,  int, "min_layers")
    dropout     = _scalar(dropout, float, "dropout")
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in (0,1)")

    # ---- infer baselines from the original transformer ----
    # heuristics: read from first block names when possible; otherwise use your builder defaults
    base_d_model = None
    base_heads   = None
    base_ff_dim  = None
    base_layers  = 0

    for lyr in original_model.layers:
        if isinstance(lyr, tf.keras.layers.MultiHeadAttention) and base_heads is None:
            base_heads = int(lyr.num_heads)
            # Keras MHA has internal head_dim; derive d_model ~ num_heads * key_dim if can
            try:
                # `key_dim` is a constructor arg; not always exposed as attribute
                cfg = lyr.get_config()
                kd = int(cfg.get("key_dim", None) or cfg.get("key_dim", 0))
                if kd:
                    base_d_model = base_heads * kd
            except Exception:
                pass
            base_layers += 1
        elif isinstance(lyr, tf.keras.layers.Dense) and base_ff_dim is None:
            # First FFN expansion dense often has > d_model units
            units = int(getattr(lyr, "units", 0))
            if units > 0:
                base_ff_dim = max(units, base_ff_dim or 0)

    # sensible fallbacks if inference was incomplete
    if base_d_model is None: base_d_model = 64
    if base_heads   is None: base_heads   = 4
    if base_ff_dim  is None: base_ff_dim  = 256
    if base_layers  == 0:    base_layers  = 3

    # ---- compute pruned widths/depth ----
    pr_d_model = max(min_d_model, int(base_d_model * (1.0 - prune_ratio)))
    pr_ff_dim  = max(min_ff_dim,  int(base_ff_dim  * (1.0 - prune_ratio)))
    pr_layers  = max(min_layers,  int(round(base_layers * (1.0 - prune_ratio))))

    # ensure num_heads divides pruned d_model; never exceed original heads
    pr_heads   = min(base_heads, max(min_heads, base_heads))  # start from base
    # reduce heads until divisible
    while pr_heads > 1 and (pr_d_model % pr_heads != 0):
        pr_heads -= 1
    if pr_heads < 1: pr_heads = 1
    # if still not divisible (tiny d_model), force head=1
    if pr_d_model % pr_heads != 0:
        pr_heads = 1

    # ---- rebuild pruned transformer ----
    inputs = layers.Input(shape=input_shape)

    # positional encoding length = timesteps (unchanged)
    pos = tf.range(start=0, limit=timesteps, delta=1)
    pos_emb = layers.Embedding(input_dim=timesteps, output_dim=pr_d_model)(pos)

    # feature projection to pr_d_model
    x = layers.Dense(pr_d_model, name="proj_dense")(inputs)
    x = x + pos_emb

    key_dim = pr_d_model // pr_heads

    for i in range(pr_layers):
        attn = layers.MultiHeadAttention(
            num_heads=pr_heads, key_dim=key_dim, dropout=dropout,
            name=f"mha_pruned_{i}"
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_pruned_{i}")(x + attn)

        ffn = layers.Dense(pr_ff_dim, activation="gelu", name=f"ffn1_pruned_{i}")(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(pr_d_model, name=f"ffn2_pruned_{i}")(ffn)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_pruned_{i}")(x + ffn)

    x = layers.GlobalAveragePooling1D(name="gap_pruned")(x)
    x = layers.Dense(32, activation="relu", name="head_pruned")(x)
    outputs = layers.Dense(1, name="out_pruned")(x)
    pruned_model = models.Model(inputs, outputs, name="transformer_pruned")

    # ---- optional: partial transfer for Dense layers only (projection/FFN/head) ----
    src_denses = [l for l in original_model.layers if isinstance(l, tf.keras.layers.Dense)]
    dst_denses = [l for l in pruned_model.layers  if isinstance(l, tf.keras.layers.Dense)]
    for s, d in zip(src_denses, dst_denses):
        _safe_copy_dense(s, d)

    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="mse", metrics=["mae"]
    )

    # metadata for your dashboard
    pruned_model.pruning_importances = {
        "d_model": pr_d_model,
        "num_heads": pr_heads,
        "ff_dim": pr_ff_dim,
        "num_layers": pr_layers,
        "dropout": dropout
    }
    # (optional) parameter counts if you later want % reduction
    pruned_model.pruning_meta = {
        "orig_params": int(original_model.count_params()),
        "pruned_params": int(pruned_model.count_params())
    }
    return pruned_model


###############################################################


# #------------------------------------------------------------
# # Build Transformer Model
# #------------------------------------------------------------

# def build_transformer_model(input_shape, 
#                            d_model=64, 
#                            num_heads=4, 
#                            ff_dim=256, 
#                            num_layers=3, 
#                            dropout=0.1):
#     """
#     Professional Transformer for Financial Time Series
#     input_shape: (timesteps, features)
#     """
#     inputs = layers.Input(shape=input_shape)
    
#     # 1. Positional Encoding (Critical for time series)
#     if d_model % num_heads != 0:
#        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

#     positions = tf.range(start=0, limit=input_shape[0], delta=1)
#     position_embedding = layers.Embedding(
#         input_dim=input_shape[0], 
#         output_dim=d_model
#     )(positions)
    
#     # 2. Feature Embedding
#     x = layers.Dense(d_model)(inputs)  # Feature projection
#     x = x + position_embedding
    
#     # 3. Transformer Blocks
#     for i in range(num_layers):  # FIXED: Add index for naming
#         # Multi-head Attention
#         attn_output = layers.MultiHeadAttention(
#             num_heads=num_heads, 
#             key_dim=d_model//num_heads,
#             dropout=dropout,
#             name=f'multi_head_attention_{i}'  # FIXED: Add proper naming
#         )(x, x)
        
#         # Skip Connection 1
#         x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
#         # Feed Forward Network
#         ffn = layers.Dense(ff_dim, activation="gelu", name=f'feed_forward_1_{i}')(x)  # FIXED: Add naming
#         ffn = layers.Dropout(dropout)(ffn)
#         ffn = layers.Dense(d_model, name=f'feed_forward_2_{i}')(ffn)
        
#         # Skip Connection 2
#         x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
#     # 4. Temporal Pooling
#     x = layers.GlobalAveragePooling1D()(x)
    
#     # 5. Output Regression Head
#     x = layers.Dense(32, activation="relu")(x)
#     outputs = layers.Dense(1)(x)
    
#     model = Model(inputs, outputs)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#         loss="mse",
#         metrics=["mae"]
#     )
#     return model


#------------------------------------------------------------
#  Display Model Summary in Streamlit
#------------------------------------------------------------

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
#------------------------------------------------------------
#  Main Streamlit App
#------------------------------------------------------------
#------------------------------------------------------------


def run_transformer_with_mda_pruning(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """
    Complete Transformer pipeline with MDA feature selection.
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
    # STEP 5: PREPARE Transformer TRAINING DATA
    # =========================================================================
    st.header(" Preparing Transformer Training Data")
    
    with st.spinner("Creating training features..."):
        training_features = create_training_features(full_df, selected_features)

    with st.spinner("Splitting train/test data..."):
        train_data, test_data = train_test_split_timeseries(training_features, test_frac=0.15)

    with st.spinner("Scaling features..."):
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        
    st.info(f"Scaled data shapes - Train: {train_scaled.shape}, Test: {test_scaled.shape}")

    with st.spinner("Creating sequences for Transformer..."):
        X_train, y_train = make_sequences(train_scaled, window=60, target_index=0)
        X_val, y_val = make_sequences(test_scaled, window=60, target_index=0)

    st.success(f"""
    **Sequences Created for Transformer:**
    - X_train: {X_train.shape} (samples, window, features)
    - y_train: {y_train.shape} (samples,)
    - X_val: {X_val.shape} (samples, window, features)  
    - y_val: {y_val.shape} (samples,)
    """)
    
    # =========================================================================
    # STEP 6: Transformer MODEL
    # =========================================================================
    st.header(" Transformer Model Architecture")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    st.info(f"Transformer Input Shape: {input_shape}")

    with st.spinner("Building Transformer model..."):
        model = build_transformer_model(input_shape)

    st.write("### Transformer Model Summary")
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

    with st.spinner("Training the Transformer model..."):
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

    fig, ax = plt.subplots(figsize=(6, 4))
    history_df['loss'].plot(ax=ax, label='Training Loss')
    # history_df['val_loss'].plot(ax=ax, label='Validation Loss')
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
            pruning_function=prune_transformer_architecture_aware,
            pruning_kwargs={
                "prune_ratio": 0.20,
                "min_d_model": 32,   
                "min_heads": 2,
                "min_ff_dim": 64,
                "min_layers": 1,
                "dropout": 0.10                         
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
    

    # =========================================================================
    # STEP 11: FINAL SUMMARY
    # =========================================================================
    st.header(" Complete Pipeline Summary")

    if pruning_success:
        st.success(f"""
        **✅ Complete Transformer Pipeline with Intelligent Pruning:**
        
        - **Data:** {ticker} from {start_date} to {end_date}
        - **Feature Journey:** 
            - Original: {len(full_df.columns)} features 
            - MDA Selected: {len(selected_features)} features
        - **Model:** Transformer with intelligent pruning
        - **Training:** {len(history.history['loss'])} epochs + pruning fine-tuning
        - **Predictions:** {len(final_predictions['y_pred'])} optimized predictions
        
        ** Final Metrics Improvement:**
        - RMSE: {original_metrics_dict['RMSE']:.4f} → **{final_metrics['RMSE']:.4f}** ({(original_metrics_dict['RMSE']-final_metrics['RMSE'])/original_metrics_dict['RMSE']*100:.1f}% improvement)
        - R²: {original_metrics_dict['R2']:.4f} → **{final_metrics['R2']:.4f}** (↑{(final_metrics['R2']-original_metrics_dict['R2'])*100:.1f}%)
        """)
    else:
        st.warning(f"""
        ** Transformer Pipeline (Pruning Failed):**
        
        - **Data:** {ticker} from {start_date} to {end_date}
        - **Features:** {len(selected_features)} selected via MDA
        - **Model:** Transformer with {input_shape} input shape
        - **Training:** {len(history.history['loss'])} epochs (Early stopping)
        - **Predictions:** {len(y_pred)} test predictions generated
        
        ** Original Metrics:**
        - RMSE: {original_metrics_dict.get('RMSE', 'N/A'):.4f}
        - R²: {original_metrics_dict.get('R2', 'N/A'):.4f}
        """)

    # Display final feature set
    st.subheader(" Final Selected Features")
    st.info(f"Total features: {len(final_features)}")
    if len(final_features) > 0:
        st.write(", ".join(final_features[:min(10, len(final_features))]))
        if len(final_features) > 10:
            with st.expander("See all features"):
                st.write(", ".join(final_features))
    else:
        st.warning("No features selected!")

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
        'scaler': scaler,
        'pruning_success': pruning_success
    }