# --------------------------------------------------------
#  Streamlit GUI Layout
# ---------------------------------------------------------
import streamlit as st
from datetime import datetime
import pandas as pd

# Set up page
st.set_page_config(page_title="Neural Network Stock Price Prediction Dashboard", layout="wide")
st.sidebar.title("Neural Network Stock Prediction")

# Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2025, 5, 30))

approach = st.sidebar.selectbox(
    "Select Prediction Approach",
    [
        "Approach 1: Simple Close Price LSTM",
        "Approach 2: Close + Indicators + Sentiment",
        "Approach 3: MDA + Pruned LSTM",
        "Approach 4: MDA + Pruned GRU",
        "Approach 5: MDA + Pruned TCN",
        "Approach 6: MDA + Pruned Transformer",
        "Approach 7: MDA + Pruned CNN-LSTM",
        "Approach 8: MDA + Pruned TCN-LSTM",
        "Approach 9: MDA + Pruned LSTM-Transformer"
    ]
)

horizon = st.sidebar.number_input("Prediction Horizon (Days Ahead)", min_value=1, max_value=30, value=1)
run_forecast = st.sidebar.button("Run Forecast")

# Main Title
st.title("A comparative Stock Price Prediction using NN")

# Handler
if run_forecast:
    st.subheader(f"{approach} Running...")

    if approach == "Approach 1: Simple Close Price LSTM":
        with st.spinner("Running Approach 1..."):
            from approaches.approach_1 import run_simple_close_lstm
            run_simple_close_lstm(ticker, start_date, end_date, horizon)

    elif approach == "Approach 2: Close + Indicators + Sentiment":
        with st.spinner("Running Approach 2..."):
            from approaches.approach_2 import run_close_indicators_sentiment_lstm
            run_close_indicators_sentiment_lstm(ticker, start_date, end_date, horizon)

    elif approach == "Approach 3: MDA + Pruned LSTM":
        with st.spinner("Running Approach 3..."):
            from approaches.approach_3 import run_simple_lstm_with_mda_pruning
            run_simple_lstm_with_mda_pruning(ticker, start_date, end_date, horizon)
    elif approach == "Approach 4: MDA + Pruned GRU":
        with st.spinner("Running Approach 4..."):
            from approaches.approach_4 import run_simple_gru_with_mda_pruning
            run_simple_gru_with_mda_pruning(ticker, start_date, end_date, horizon)
    elif approach == "Approach 5: MDA + Pruned TCN":
        with st.spinner("Running Approach 5..."):
            from approaches.approach_5 import run_tcn_with_mda_pruning
            run_tcn_with_mda_pruning(ticker, start_date, end_date, horizon)
    elif approach == "Approach 6: MDA + Pruned Transformer":
        with st.spinner("Running Approach 6..."):
            from approaches.approach_6 import run_transformer_with_mda_pruning
            run_transformer_with_mda_pruning(ticker, start_date, end_date, horizon)
    elif approach == "Approach 7: MDA + Pruned CNN-LSTM":
        with st.spinner("Running Approach 7..."):
            from approaches.approach_7 import run_cnn_lstm_with_mda_pruning
            run_cnn_lstm_with_mda_pruning(ticker, start_date, end_date, horizon)
    elif approach == "Approach 8: MDA + Pruned TCN-LSTM":
        with st.spinner("Running Approach 8..."):
            from approaches.approach_8 import run_tcn_lstm_with_mda_pruning
            run_tcn_lstm_with_mda_pruning(ticker, start_date, end_date, horizon)
    elif approach == "Approach 9: MDA + Pruned LSTM-Transformer":
        with st.spinner("Running Approach 9..."):
            from approaches.approach_9 import  run_simple_lstm_transformer_with_mda_pruning
            run_simple_lstm_transformer_with_mda_pruning(ticker, start_date, end_date,horizon)
    st.success("Model Training and Forecast complete.")

