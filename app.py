"""
app.py
Streamlit Entry point for the ML Stock Prediction App.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import your custom modules
from data import fetch_stock_data, get_company_info
from features import add_features
from model import train_and_predict
from visualizer import plot_predictions, plot_feature_importance

# App Configuration
st.set_page_config(page_title="Fusemachine Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom UI Styling
st.markdown("""
    <style>
    .metric-card { background-color: #1e222b; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .stDataFrame { background-color: #1e222b; }
    </style>
""", unsafe_allow_html=True)

#controlling sidebar
st.sidebar.header("Model Configuration")
ticker = st.sidebar.text_input("Enter Symbol of Company)", value="FUSE").upper().strip()

period = st.sidebar.selectbox("Historical Lookback Window", ["1y", "2y", "5y", "10y"], index=1)
model_type = st.sidebar.selectbox("ML Engine Core", ["Random Forest", "Gradient Boosting", "Linear Regression"], index=0)
forecast_days = st.sidebar.slider("Forecast Horizon (Trading Days)", min_value=5, max_value=60, value=30, step=5)
test_size = st.sidebar.slider("Validation Train/Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

run_pipeline = st.sidebar.button("Click Here to Check")

#main interface display
st.title("Stock Predict from Yahoo Finance by Ashim")
st.markdown("---")

if ticker:
    try:
        # Step 1: Data Pipeline Fetching
        with st.spinner("Fetching financial telemetry from data layer..."):
            info = get_company_info(ticker)
            df_raw = fetch_stock_data(ticker, period=period)
            df_feat = add_features(df_raw)
            
        # Display Company Summary Profile Card
        st.subheader(f" {info.get('name', ticker)}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sector", info.get("sector", "N/A"))
        with col2:
            m_cap = info.get("market_cap")
            st.metric("Market Cap", f"${m_cap:,.0f}" if m_cap else "N/A")
        with col3:
            st.metric("P/E Ratio", f"{info.get('pe_ratio'):.2f}" if info.get('pe_ratio') else "N/A")
        with col4:
            st.metric("Currency Base", info.get("currency", "USD"))

        # Step 2: Core ML Trigger Pipeline
        if run_pipeline:
            with st.spinner("Training ML Engine and performing iterative walk-forward predictions..."):
                results = train_and_predict(
                    df_feat, 
                    model_type=model_type, 
                    forecast_days=forecast_days, 
                    test_size=test_size
                )
            
            st.markdown("Forecasting & Historical Performance Validation")
            # Render Core Interactive Chart
            fig_main = plot_predictions(results, ticker)
            st.plotly_chart(fig_main, use_container_width=True)
            
            # Display Validation Errors (Simulating MLOps Metrics Evaluators)
            st.markdown("Model Metrics Summary")
            m_col1, m_col2 = st.columns(2)
            
            with m_col1:
                st.subheader("In-Sample Performance (Train)")
                st.json(results["metrics_train"])
                
            with m_col2:
                st.subheader("Out-of-Sample Performance (Test)")
                st.json(results["metrics_test"])
                
            st.markdown("---")
            b_col1, b_col2 = st.columns([1, 1])
            
            with b_col1:
                st.subheader("Feature Impact Attribution")
                if results["feature_importances"] is not None:
                    fig_fi = plot_feature_importance(results["feature_importances"])
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Linear Regression uses fixed weights rather than tree split features.")
                    
            with b_col2:
                st.subheader(f"Out-Of-Sample {forecast_days}-Day Predictions Table")
                forecast_df = pd.DataFrame({
                    "Date": results["future_dates"].strftime("%Y-%m-%d"),
                    "Projected Close (USD)": results["future_pred"].round(2)
                }).set_index("Date")
                st.dataframe(forecast_df, use_container_width=True, height=300)
                
            
    except Exception as e:
        st.error(f"Execution Error Intercepted: {e}")