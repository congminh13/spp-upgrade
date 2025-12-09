# pages/Evaluation.py
import streamlit as st
from app import menu
from Config import Config
from models.CombinedGraph import CombinedGraph as CG
from datetime import datetime, timedelta

st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.APP_ICON,
    layout="wide"
)
menu()

st.title('Evaluation & Conclusion')

st.subheader('***Model Comparison:***')

st.write("Popular Stock Symbols:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Period", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Select Training Ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

# st.subheader("Dự đoán tương lai (Keras Model)")
# start_date = st.date_input("Chọn ngày bắt đầu", value=datetime(2025, 5, 23))
# end_date = st.date_input("Chọn ngày kết thúc", value=datetime(2025, 6, 23))

# # Validate date range
# if start_date >= end_date:
#     st.error("Ngày kết thúc phải sau ngày bắt đầu.")
# else:
cg = CG(stock_symbol, period, train_ratio)
cg.visualize()
st.subheader('***Evaluation:***')
st.markdown(
    """
    1. Comparing the three models (basic linear regression, improved linear regression, and Keras LSTM), we see that the Keras model captures time trends better due to its LSTM structure.
    2. Both the improved model and Keras model have higher accuracy than the basic model, with Keras often having lower error in complex cases.
    3. The use of technical indicators (RSI, MACD) and temporal data significantly improved the performance of both the improved linear regression and Keras models.
    4. The Keras model can predict future stock prices with greater flexibility due to its ability to learn non-linear patterns.
    """
)

st.subheader('***Conclusion:***')
st.markdown(
    """
    ***Using Linear Regression is feasible when combined with technical indicators, but the Keras LSTM model provides more robust prediction capabilities in complex situations. The combination of technical data and deep learning models is key to improving stock price prediction accuracy.***
    """
)
st.markdown(
    """
    ***Note:*** The Keras model can be further optimized by adjusting the network architecture, adding new features, or using techniques like cross-validation.
    """
)