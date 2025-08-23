# ============================
#  Imports & Configuration
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yfinance as yf
import pandas_ta as ta
import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from packaging import version
import keras_tuner as kt
from scipy.stats import norm

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)

# Import your custom pipeline function
        
from approaches.approach_3 import (
    load_stock_data, 
    plot_feature_distribution, 
    compute_features, 
    prepare_daily_sentiment_features,
    align_final_dataframe,
    build_X_y_for_mda, 
    compute_mda_importance, 
    plot_feature_importance,
    split_train_test_scaled_top_fifty_features,
    make_sequences,
    build_lstm_model,
    display_model_summary,
    make_predictions,
    evaluate_model, 
    plot_model_prediction,
    permutation_importance_seq,
    plot_permutation_importance_barh,
    prune_retrain_lstm,
    predict_after_pruning,
    display_pruned_training_results
    )  # Ensure these function are correctly implemented

# --------------------------------------------------------
#  Streamlit GUI Layout
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Price LSTM Prediction Dashboard", layout="wide")
st.sidebar.title("LSTM Stock Prediction")

ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2025, 5, 30))

approach = st.sidebar.selectbox(
    "Select Prediction Approach",
    [
        "Approach 1: Simple Close Price LSTM",
        "Approach 2: Close + Indicators + Sentiment",
        "Approach 3: MDA + Pruned LSTM",
        "Approach 4: MDA + Pruned LSTM + Attention"
    ]
)

horizon = st.sidebar.number_input("Prediction Horizon (Days Ahead)", min_value=1, max_value=30, value=1)
run_forecast = st.sidebar.button("Run Forecast")

st.title("Stock Price Prediction using LSTM")

if run_forecast:
    st.subheader(f"{approach} Running")

    with st.spinner("Fetching and preparing data..."):
        # Load data using your approach_3 utility
        df = load_stock_data(ticker, str(start_date), str(end_date), f"{ticker}_{start_date}_to_{end_date}.csv")

        st.write("### Sample of Downloaded Data")
        st.dataframe(df.head(10))

    st.write("### Visualizing Feature Distributions")
    plot_feature_distribution(df)

#----------------------------------------------------------------------------


    # ------------------------------------
    # Compute Institutional-Grade Features
    # ------------------------------------
    with st.spinner("Computing technical features..."):
        df_features = compute_features(df)

        st.write("### Feature-Engineered Data Sample")
        st.dataframe(df_features.head(5))

    #  Load Sentiment Data and Compute Features 
    with st.spinner("Loading and processing sentiment data..."):
        daily_sent = prepare_daily_sentiment_features("synthetic_financial_tweets_labeled.AAPL.csv")
        if daily_sent is None:
            st.stop()  # Stop execution if loading failed

#----------------------------------------------------------------------------   
    with st.spinner("Aligning price and sentiment data..."):
        full_df = align_final_dataframe()
        if full_df is None:
            st.stop() 
 #----------------------------------------------------------------------------
 
   
    with st.spinner("Building X and y for MDA feature importance..."):
        X_mda, y_mda = build_X_y_for_mda(full_df, horizon=1)
        if X_mda is None or y_mda is None:
           st.stop()
 #----------------------------------------------------------------------------
   
    with st.spinner("Computing MDA permutation importance..."):
        mda_imp = compute_mda_importance(
        X_mda, y_mda,
        n_splits=5,        # walk-forward folds
        embargo=5,         # 5-day embargo
        seed=42,
    ).sort_values(ascending=False)

    st.success("✅ MDA feature importance computed.")

    st.write("###  Features by MDA Importance")
    st.dataframe(mda_imp.head(X_mda.shape[1]))

    st.write("### MDA Feature Importance Plot")
    # Plot top MDA features
    mda_features = mda_imp.head(X_mda.shape[1])
    plot_feature_importance(mda_imp, n_top=len(mda_features))
    #all_features=mda_features.index.tolist()
    print("MDA Features:", mda_features)
  

#----------------------------------------------------------------------------

    X_train_scaled, X_test_scaled, scaler, selected_features, train_df, test_df = split_train_test_scaled_top_fifty_features(
    mda_imp,
    all_features=mda_features.index.tolist()
   )
    print("My Selected Features $$$$$$$$$$$$$$$$$$$$$$:", selected_features)
    if X_train_scaled is None:
      st.stop()
    
   

#----------------------------------------------------------------------------

     # Confirm your TARGET_COL for the current pipeline
    TARGET_COL = 'Close'
    WINDOW = 60

    # Get index of target within selected features
    try:
        target_index = selected_features.index(TARGET_COL)
    except ValueError:
        st.error(f"Target column '{TARGET_COL}' not found in selected features: {selected_features}")
        st.stop()

     # Build sequences
    X_train, y_train = make_sequences(X_train_scaled, WINDOW, target_index)
    X_test, y_test = make_sequences(X_test_scaled, WINDOW, target_index)

    # Display shape confirmation
    st.success(f"Sequence shapes built:\n"
           f"X_train ready for lstm: {X_train.shape}, y_train ready for lstm: {y_train.shape}\n"
           f"X_test ready for lstm: {X_test.shape}, y_test ready for lstm: {y_test.shape}")

#----------------------------------------------------------------------------
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build model
    model = build_lstm_model(input_shape)

    # Display summary inside your Streamlit app
    st.write("### LSTM Model Summary")
    display_model_summary(model)


    # ============================
    # Validation Split
    # ============================
    VAL_FRAC = 0.10
    val_start = int(len(X_train) * (1 - VAL_FRAC))

    X_tr, X_val = X_train[:val_start], X_train[val_start:]
    y_tr, y_val = y_train[:val_start], y_train[val_start:]

    st.write(f"Validation split: {VAL_FRAC*100:.0f}% of training data")
    st.write(f"Shapes ➔ X_tr: {X_tr.shape}, y_tr: {y_tr.shape} | X_val: {X_val.shape}, y_val: {y_val.shape}")

    # ============================
    # Model Training with EarlyStopping
    # ============================
    es = callbacks.EarlyStopping(
       patience=7,
       restore_best_weights=True
      )

    with st.spinner("Training the LSTM model..."):
        history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[es],
        verbose=2,
    )

        st.success("Model training completed.")

    # ============================
    # Plot Training and Validation Loss
    # ============================
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

    # Prediction and inverse scaling
    with st.spinner("Making predictions..."):
        y_true, y_pred, aligned_dates, full_test = make_predictions(
            model=model,
            scaler=scaler,
            train_df=train_df,
            test_df=test_df,
            FEATURES=selected_features,
            LOOK_BACK=WINDOW
        )

    plot_model_prediction(aligned_dates, y_true, y_pred)
  

    # Display true vs predicted values (first 10)
    results_df = pd.DataFrame({
        'True Close': y_true[:10],
        'Predicted Close': y_pred[:10]
    })
    st.write("### True vs Predicted Close Prices (First 10)")
    st.dataframe(results_df)


# -----------------------------------------------------

    # Evaluate model performance
    metrics_df = evaluate_model(y_true, y_pred)
   
 # -----------------------------------------------------


    # Display evaluation metrics
    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Metrics as CSV",
        data=csv_metrics,
        file_name="evaluation_metrics.csv",
        mime='text/csv'
    )

 # -----------------------------------------------------


    st.write("### 📈 Prediction vs Actual Plot")
    plot_model_prediction(full_test, y_true, y_pred)

# -----------------------------------------------------
    st.write("### Permutation Importance on Test Set")
#------------------------------------------------------


    with st.spinner("Computing permutation importance on test sequences..."):
        perm_importance = permutation_importance_seq(
            model=model,
            X=X_test,               # 3D: (samples, window, features)  # ✔️ test sequences
            y=y_test,               # 1D: true Close prices # ✔️ true test targets
            feature_names=selected_features,
            n_repeats=5,
            random_state=42
        )
# Why test set? It reflects generalisation: you're measuring how sensitive the
#  model’s test performance is to each feature being randomly permuted.
#This lets you visualise and rank which features truly influence predictions.
    
    
    
    # Convert to Series and sort
    perm_series = pd.Series(perm_importance).sort_values(ascending=False)

    st.success("Permutation importance computed.")
    st.write("### Top 10 Feature Importances (Test Set)")
    st.dataframe(perm_series.head(10))


#         # Here `perm_importance` should be  result from permutation_importance_seq
#     plot_permutation_importance_barh(
#         perm_importance,
#         title="Permutation Importance on Test Set (LSTM)"
#         )

#  #------------------------------------------------------
#     st.write("### Pruning and Retraining the LSTM Model")
#     st.write("This step will prune the model based on feature importance and retrain it.")
#     st.write("### 🔍 Selected Features for Pruning")
#     st.write(f"{len(selected_features)} features selected for pruning:")
#     st.write(selected_features)
#     # Ensure selected_features is not empty
#     if not selected_features:   
#         st.error("No features selected for pruning. Please check the previous steps.")
#         st.stop()  

    
#     with st.spinner("Running pruning and retraining..."):
#         importances, final_features, history, eval_metrics, pruned_model = prune_retrain_lstm(
#             model_builder=build_lstm_model,
#             X=X_train,
#             y=y_train,
#             feature_names=selected_features,
#             keep_frac=0.5,
#             n_repeats=5,
#             random_state=42,
#             epochs=50,
#             batch_size=32,
#             verbose=1        )

#     # Display retained features
#     st.success("✅ Pruning and retraining completed.")
#     st.write("### 🔍 Retained Predictive Features (MDA > 0)")
#     st.write(f"{len(final_features)} features kept:")
#     st.write(final_features)

   


# # ============================

#     display_pruned_training_results(history, eval_metrics)



# # ============================

#     y_true, y_pred, full_test, metrics = predict_after_pruning(
#         model=pruned_model,
#         train_df=train_df,
#         test_df=test_df,
#         FEATURES=final_features,   # list from pruning
#         look_back=60,
#         ctx_frac=0.2
#     )


    # === Prune and retrain model ===
    with st.spinner("Running pruning and retraining..."):
        importances, final_features, history, eval_metrics, pruned_model = prune_retrain_lstm(
            model_builder=build_lstm_model,
            X=X_train,
            y=y_train,
            feature_names=selected_features,
            keep_frac=0.5,
            n_repeats=5,
            random_state=42,
            epochs=50,
            batch_size=32,
            verbose=1
        )

    st.success("✅ Pruning and retraining completed.")
    st.write("### 🔍 Retained Predictive Features (MDA > 0)")
    st.write(final_features)

    # === Plot importance after pruning ===
    st.write("### 🔬 Permutation Importance (After Pruning)")
    plot_permutation_importance_barh(
        importances,
        title="Permutation Importance After Pruning (Retrained LSTM)"
    )

    # === Plot training curve and metrics ===
    display_pruned_training_results(history, eval_metrics)

    # === Predict with pruned model ===
    y_true, y_pred, full_test, metrics = predict_after_pruning(
        model=pruned_model,
        train_df=train_df,
        test_df=test_df,
        FEATURES=final_features,
        look_back=60,
        ctx_frac=0.2
    )

    # === Show prediction and metrics ===
    plot_model_prediction(full_test.index[60:], y_true, y_pred)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.write("### 📊 Evaluation Metrics After Pruning")
    st.table(metrics_df)





        
        
        
        
        
        
        










































    
#     # Placeholder for next stages: Preprocessing, Feature Engineering, Sequencing, Model Building
#     st.info("Continue building the pipeline here with clear separation of stages.")

    
    