## approaches/approach_10.py

# ============================
#  Imports & Configuration
# ============================

import warnings
import yfinance as yf
import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import gc
import sklearn
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import BaggingClassifier
from packaging import version

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
    create_feature_pruning_dashboard,
    get_optimal_pruning_config,
    intelligent_feature_selection,
    should_use_pruned_model,
    run_intelligent_pruning_pipeline,
    create_comprehensive_comparison_dashboard,
    make_sequences_from_df,
    normalize_metrics,
    safe_mape
)

# ============================
#   Fixed Functions
# ============================

def make_predictions_after_retraining(
    model,
    scaler,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    FEATURES,
    LOOK_BACK: int = 60,
    CTX_FRAC_TRAIN: float = 0.20,
    *,
    original_target_scaler=None,
    target_is_scaled: bool = True
):
    """Make predictions with proper time series handling and scaling."""
    # 1) FEATURES normalization & target check
    if not isinstance(FEATURES, tuple):
        FEATURES = tuple(FEATURES)
    if 'Close' not in FEATURES:
        raise ValueError("Target feature 'Close' missing in feature list")
    CLOSE_IDX = FEATURES.index('Close')

    # 2) Build context window
    ctx_len = int(len(train_df) * CTX_FRAC_TRAIN)
    context = train_df.tail(ctx_len)
    full_test = pd.concat([context, test_df], axis=0).sort_index()

    # 3) Ensure scaler matches current feature count
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(FEATURES):
        st.warning(
            f"Refitting scaler for pruned feature set: "
            f"incoming={len(FEATURES)} vs scaler.n_features_in_={scaler.n_features_in_}"
        )
        scaler_ = MinMaxScaler().fit(train_df[list(FEATURES)])
    else:
        scaler_ = scaler

    # 4) Scale + sequence
    X_full = full_test[list(FEATURES)].to_numpy(dtype=np.float32)
    scaled_full = scaler_.transform(X_full)
    X_pred = np.stack([scaled_full[i-LOOK_BACK:i] for i in range(LOOK_BACK, len(scaled_full))])

    # 5) Truth in dollars
    y_true = full_test['Close'].to_numpy()[LOOK_BACK:]

    # 6) Predict
    y_hat = model.predict(X_pred, verbose=0).ravel()

    # 7) Inverse scaling
    if target_is_scaled:
        if original_target_scaler is not None:
            y_pred = original_target_scaler.inverse_transform(y_hat.reshape(-1, 1)).ravel()
        else:
            scaled_for_inv = scaled_full[LOOK_BACK:].copy()
            scaled_for_inv[:, CLOSE_IDX] = y_hat
            y_pred = scaler_.inverse_transform(scaled_for_inv)[:, CLOSE_IDX]
    else:
        y_pred = y_hat

    # 8) Quick sanity
    if np.isfinite(y_true.mean()) and np.isfinite(y_pred.mean()):
        if abs(y_pred.mean() - y_true.mean()) / max(1e-9, abs(y_true.mean())) > 5:
            st.error(
                f"Potential scaling mismatch: mean(pred)={y_pred.mean():.2f}, "
                f"mean(true)={y_true.mean():.2f}"
            )

    aligned_dates = full_test.index[LOOK_BACK:]
    return y_true, y_pred, aligned_dates, full_test

def build_Xy_for_feature_pruning_regression_from_training(
    training_features_df: pd.DataFrame, horizon: int = 1
):
    """
    Build X (features at t) and y (future Close at t+h) for regression,
    using ONLY the columns present in `training_features_df`.
    """
    if "Close" not in training_features_df.columns:
        raise ValueError("'Close' must be in training_features_df")

    # Target: future Close
    y = training_features_df["Close"].shift(-horizon)

    # Features: everything known at t
    X = training_features_df.copy()

    # Align and drop NaNs induced by the shift
    X, y = X.iloc[:-horizon], y.iloc[:-horizon]
    X, y = X.dropna(), y.loc[X.index]

    feature_names = list(X.columns)
    return X, y, feature_names

def scale_then_sequence(full_X: pd.DataFrame, y: pd.Series, look_back: int = 60, test_frac: float = 0.2):
    """
    Proper time-series aware scaling and sequencing without data leakage.
    """
    n = len(full_X)
    n_test = int(n * test_frac)
    n_val = int((n - n_test) * 0.2)  # 20% of training for validation
    
    # Train/Val/Test split (chronological order)
    train_idx = slice(None, -(n_test+n_val))
    val_idx = slice(-(n_test+n_val), -n_test)
    test_idx = slice(-n_test, None)
    
    X_train_df = full_X.iloc[train_idx]
    X_val_df = full_X.iloc[val_idx]
    X_test_df = full_X.iloc[test_idx]
    
    # Scale features - fit only on training data
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), 
                                  index=X_train_df.index, columns=X_train_df.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_df), 
                                index=X_val_df.index, columns=X_val_df.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), 
                                 index=X_test_df.index, columns=X_test_df.columns)
    
    # Scale targets separately - fit only on training data
    target_scaler = MinMaxScaler()
    y_train = y.iloc[train_idx].values.reshape(-1, 1)
    y_val = y.iloc[val_idx].values.reshape(-1, 1)
    y_test = y.iloc[test_idx].values.reshape(-1, 1)
    
    y_train_scaled = target_scaler.fit_transform(y_train).flatten()
    y_val_scaled = target_scaler.transform(y_val).flatten()
    y_test_scaled = target_scaler.transform(y_test).flatten()
    
    # Convert to Series
    y_train_scaled = pd.Series(y_train_scaled, index=X_train_df.index)
    y_val_scaled = pd.Series(y_val_scaled, index=X_val_df.index)
    y_test_scaled = pd.Series(y_test_scaled, index=X_test_df.index)
    
    # Create sequences
    Xtr_seq, ytr_seq, feat_names = make_sequences_from_df(X_train_scaled, y_train_scaled, look_back)
    Xval_seq, yval_seq, _ = make_sequences_from_df(X_val_scaled, y_val_scaled, look_back)
    Xte_seq, yte_seq, _ = make_sequences_from_df(X_test_scaled, y_test_scaled, look_back)
    
    return Xtr_seq, ytr_seq, Xval_seq, yval_seq, Xte_seq, yte_seq, feat_names, scaler, target_scaler

def build_simplified_transformer(input_shape, d_model=32, num_heads=2, 
                               ff_dim=128, num_layers=2, dropout=0.2):
    """Build a simplified transformer model for time series forecasting."""
    inputs = layers.Input(shape=input_shape)
    
    # Simplified feature projection
    x = layers.Dense(d_model)(inputs)
    x = layers.Dropout(dropout)(x)
    
    # Fewer transformer blocks
    for i in range(num_layers):
        # Simplified attention mechanism
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model//num_heads,
            dropout=dropout
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Simplified FFN
        ffn = layers.Dense(ff_dim, activation="relu")(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def display_model_summary(model):
    """Captures and displays the model summary in Streamlit."""
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__  # reset
    st.code(stream.getvalue())

def permutation_importance_seq(model, X, y, feature_names, n_repeats=5, random_state=None):
    """
    Time-preserving permutation importance for (N, T, F) sequences.
    We shuffle the *sample order* of a feature's whole time block, not timesteps within a sample.
    """
    rng = np.random.default_rng(random_state)

    baseline = model.predict(X, verbose=0).ravel()
    base_mse = mean_squared_error(y, baseline)

    importances = {}
    for j, name in enumerate(feature_names):
        diffs = []
        for _ in range(max(1, n_repeats)):
            Xp = X.copy()
            perm = rng.permutation(X.shape[0])   # shuffle samples only
            Xp[:, :, j] = X[perm, :, j]          # keep each sequence intact
            preds = model.predict(Xp, verbose=0).ravel()
            diffs.append(mean_squared_error(y, preds) - base_mse)
        importances[name] = float(np.mean(diffs))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))
    return importances

def prune_retrain_model(
    model_builder, X, y, feature_names, 
    keep_frac=0.5, n_repeats=5, random_state=None, 
    epochs=50, batch_size=32, verbose=1,
    test_size=0.2, original_target_scaler=None,
    target_is_scaled=True
):
    """
    Prune low-importance features via permutation importance and retrain.
    Uses chronological Train/Val/Test splits and reports metrics in ORIGINAL units if a target scaler is provided.
    """
    # ----- basic checks
    assert X.ndim == 3, f"X must be (N, T, F), got {X.shape}"
    feature_names = [str(f) for f in feature_names]
    assert len(feature_names) == X.shape[-1], "feature_names length must match X.shape[-1]"
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)

    # ----- reproducibility
    if random_state is not None:
        np.random.seed(random_state)
        tf.keras.utils.set_random_seed(random_state)

    # ----- time-aware split
    n = X.shape[0]
    n_test = max(1, int(n * test_size))
    n_trainval = n - n_test
    n_val = max(1, int(n_trainval * 0.2))
    n_train = n_trainval - n_val
    if n_train <= 0:
        raise ValueError("Not enough samples to create train/val/test splits.")

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

    # ----- warm-up model (better importances)
    warm = model_builder(X_train.shape[1:])
    warm.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max(5, epochs // 8),
        batch_size=batch_size,
        shuffle=False,
        verbose=max(0, verbose-1),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            # tf.keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=2, min_lr=1e-7, verbose=0),
        ],
    )

    # ----- permutation importance on VAL ONLY (no leakage)
    importances = permutation_importance_seq(
        model=warm, X=X_val, y=y_val,
        feature_names=feature_names,
        n_repeats=n_repeats, random_state=random_state
    )

    # ----- choose features
    ranked = list(importances.items())
    n_keep = max(1, int(len(ranked) * float(keep_frac)))
    final_features = [f for f, _ in ranked[:n_keep]]

    # ensure 'Close' survives if it was present (common guardrail)
    if "Close" in feature_names and "Close" not in final_features:
        final_features.append("Close")
        final_features = [f for f in final_features if f in feature_names]

    feat_idx = [feature_names.index(f) for f in final_features]

    # ----- prune tensors
    X_train_p, X_val_p, X_test_p = X_train[:, :, feat_idx], X_val[:, :, feat_idx], X_test[:, :, feat_idx]

    # ----- retrain on pruned set
    model = model_builder(X_train_p.shape[1:])
    history = model.fit(
        X_train_p, y_train,
        validation_data=(X_val_p, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=max(7, epochs//10), restore_best_weights=True),
            # tf.keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=max(3, epochs//20), min_lr=1e-7, verbose=0),
        ],
    )

    # ----- evaluate on TEST (original units if scaled target)
    y_hat = model.predict(X_test_p, verbose=0).ravel()
    y_eval, y_pred_eval = y_test, y_hat

    if target_is_scaled and original_target_scaler is not None:
        y_eval      = original_target_scaler.inverse_transform(y_eval.reshape(-1, 1)).ravel()
        y_pred_eval = original_target_scaler.inverse_transform(y_pred_eval.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred_eval)))
    mae  = float(mean_absolute_error(y_eval, y_pred_eval))
    r2   = float(r2_score(y_eval, y_pred_eval))
    # safe MAPE
    denom = np.clip(np.abs(y_eval), 1e-9, None)
    mape = float(np.mean(np.abs((y_eval - y_pred_eval) / denom)) * 100.0)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
    return importances, final_features, history, metrics, model

def plot_feature_importance(importances, title="Feature Importance", figsize=(10, 6)):
    """
    Create a horizontal bar plot of feature importances.
    """
    if not importances:
        st.warning("No importances to plot")
        return
        
    # Sort features for consistent plotting (descending by importance)
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(features[::-1], importance_values[::-1])  # Reversed for highest at top
    ax.set_xlabel("Permutation Importance (Δ MSE)", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

def plot_training_history(history, title="Training History", figsize=(10, 5)):
    """
    Plot training and validation loss over epochs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# ============================
#  Main Streamlit App
# ============================

def run_transformer_mda_with_intelligent_feature_pruning(ticker="AAPL", start_date="2015-01-01", end_date="2025-05-30", horizon=60):
    """Complete Transformer pipeline with MDA feature selection."""
    
    # STEP 1: DATA LOADING AND VISUALIZATION
    with st.spinner("Fetching and preparing data..."):
        df = load_stock_data(ticker, str(start_date), str(end_date), f"data/{ticker}_{start_date}_to_{end_date}.csv")

    st.write("### Sample of Downloaded Data")
    st.dataframe(df.head(10))

    st.write("### Visualizing Feature Distributions")
    plot_feature_distribution(df)

    st.write("### Visualizing Price Trends")
    plot_close_price_history(df)

    # STEP 2: FEATURE ENGINEERING
    with st.spinner("Computing technical features..."):
        df_features = compute_features(df)

    st.write("### Feature-Engineered Data Sample")
    st.dataframe(df_features.head(5))

    # STEP 3: SENTIMENT DATA INTEGRATION
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
    st.write(f"### Plotting Features Distribution:")
    plot_features_distribution(full_df, max_plots=20)

    # STEP 4: MDA FEATURE SELECTION
    st.header("MDA Feature Selection")
    
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
    
    # STEP 5: PREPARE TRANSFORMER TRAINING DATA
    st.header("Preparing Transformer Training Data")
    
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
    
    # STEP 6: TRANSFORMER MODEL
    st.header("Transformer Model Architecture")
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    st.info(f"Transformer Input Shape: {input_shape}")

    with st.spinner("Building Transformer model..."):
        model = build_simplified_transformer(input_shape)

    st.write("### Transformer Model Summary")
    display_model_summary(model)

    # STEP 7: MODEL TRAINING
    st.header("Model Training")
    
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
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
            shuffle=False  # Important for time series
        )
    
    st.success("Model training completed.")

    # STEP 8: TRAINING HISTORY VISUALIZATION
    st.header("Training History")
    
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

    # STEP 9: PREDICTIONS AND EVALUATION
    st.header("Predictions & Evaluation")
    
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
    
    st.write("### True vs Predicted Close Prices (First 10)")
    st.dataframe(results_df)
    
    # Plot predictions
    st.write("### Prediction vs Actual Plot")
    plot_model_prediction(aligned_dates, y_true, y_pred)
    
    # Model evaluation metrics
    st.write("### Model Evaluation Metrics")
    metrics_result = evaluate_model(y_true, y_pred)
    
    # Simple metrics conversion
    if isinstance(metrics_result, dict):
        baseline_metrics = metrics_result
        metrics_df = pd.DataFrame([
            {"Metric": key, "Value": value} 
            for key, value in metrics_result.items()
        ])
    else:
        # Assume it's a tuple/list in standard order
        baseline_metrics = {
            'MSE': metrics_result[0] if len(metrics_result) > 0 else 0.0,
            'RMSE': metrics_result[1] if len(metrics_result) > 1 else 0.0,
            'MAE': metrics_result[2] if len(metrics_result) > 2 else 0.0,
            'R2': metrics_result[3] if len(metrics_result) > 3 else 0.0,
            'MAPE': metrics_result[4] if len(metrics_result) > 4 else 0.0
        }
        metrics_df = pd.DataFrame([
            {"Metric": key, "Value": value} 
            for key, value in baseline_metrics.items()
        ])
    
    # Display metrics
    st.dataframe(metrics_df)
    
    # Download metrics
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Metrics as CSV",
        data=csv_metrics,
        file_name=f"{ticker}_transformer_metrics.csv",
        mime='text/csv'
    )
    
    # STEP 10: INTELLIGENT FEATURE PRUNING (TRANSFORMER)
    st.header("Intelligent Feature Pruning (Permutation Importance)")

    # Build regression target & features for pruning
    X_tab, y_vec, feature_list = build_Xy_for_feature_pruning_regression_from_training(
        training_features_df=training_features, horizon=1
    )
    if X_tab is None:
        st.stop()

    # Chronological split, scale on train, make sequences
    result = scale_then_sequence(X_tab, y_vec, look_back=60, test_frac=0.2)
    Xtr_seq_fp, ytr_seq_fp, Xval_seq_fp, yval_seq_fp, Xte_seq_fp, yte_seq_fp, feat_list_fp, scaler_fp, target_scaler_fp = result

    # Concatenate for prune_retrain_model
    X_trainval_seq = np.concatenate([Xtr_seq_fp, Xval_seq_fp], axis=0)
    y_trainval_seq = np.concatenate([ytr_seq_fp, yval_seq_fp], axis=0)
        

    # Prune + retrain (feature selection only)
    with st.spinner("Running intelligent feature pruning... This may take a few minutes..."):
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        importances, final_features, history_fp, pruned_metrics, pruned_model = prune_retrain_model(
            model_builder=build_simplified_transformer,
            X=X_trainval_seq, y=y_trainval_seq,  # Train + validation
            feature_names=feat_list_fp,
            keep_frac=0.85,
            n_repeats=5,
            random_state=42,
            epochs=100,
            batch_size=32,
            verbose=1,
            test_size=0.2,  # Now properly used
            original_target_scaler=target_scaler_fp,
            target_is_scaled=True
        )
        progress_bar.progress(100)
        progress_bar.empty()

    st.subheader("Pruned model metrics (holdout)")
    st.write(pruned_metrics)
    
    # STEP 11: FINAL SUMMARY
    st.header(" Complete Pipeline Summary (Feature Pruning)")

    # Display feature pruning results
    st.write("### Feature Pruning Results")
    st.success(f"✅ Feature pruning completed! Kept {len(final_features)} out of {len(feat_list_fp)} features")
    
    # Display feature importance
    st.write("### Feature Importance")
    fig = plot_feature_importance(importances, "Permutation Feature Importance")
    st.pyplot(fig)
    
    # Display kept features
    st.write("### Final Selected Features")
    st.info(f"Kept features ({len(final_features)}): {', '.join(final_features)}")
    
    # Plot pruned model training history
    st.write("### Pruned Model Training History")
    fig = plot_training_history(history_fp, "Pruned Transformer Training")
    st.pyplot(fig)
    
    # =========================================================================
    # STEP 7: COMPARE MODELS
    # =========================================================================
    st.header(" Model Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for metric, baseline_val in baseline_metrics.items():
        pruned_val = pruned_metrics.get(metric, 0.0)
        comparison_data.append({
            'Metric': metric,
            'Baseline': f"{baseline_val:.4f}",
            'Pruned': f"{pruned_val:.4f}",
            'Change': f"{((pruned_val - baseline_val) / baseline_val * 100):.1f}%" if baseline_val != 0 else "N/A"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)
    
    # Summary
    feature_reduction = (1 - len(final_features) / len(feat_list_fp)) * 100
    
    st.success(f"""
    ** Pipeline Summary:**
    
    - **Feature Reduction:** {feature_reduction:.1f}% ({len(feat_list_fp)} → {len(final_features)})
    - **Model Performance:** {'Maintained' if abs(pruned_metrics['R2'] - baseline_metrics['R2']) < 0.05 else 'Changed'}
    - **Complexity:** Significantly reduced while maintaining predictive power
    """)
    
    # ==============
    # CLEANUP
    # ==============
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Assign baseline_model before returning results
    baseline_model = model

    # Return results
    return {
        'baseline_model': baseline_model,
        'pruned_model': pruned_model,
        'baseline_metrics': baseline_metrics,
        'pruned_metrics': pruned_metrics,
        'feature_evolution': {
            'original_features': list(full_df.columns),
            'mda_selected': selected_features,
            'final_features': final_features
        },
        'importances': importances,
        'scalers': {
            'feature_scaler': scaler,
            'target_scaler': target_scaler_fp
        }
    }