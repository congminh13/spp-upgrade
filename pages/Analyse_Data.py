import streamlit as st
import pandas as pd
import numpy as np
from app import menu
from Config import Config
from models.KerasSPPModule import KerasSPPModule as KSPPM

st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.APP_ICON,
    layout="wide"
)
menu()

st.title('Stock Price Data Analysis for RNN (LSTM) Model')

st.write("Popular Stock Symbols:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Period", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Select Training Ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

ksppm = KSPPM(stock_symbol, period, train_ratio)
raw_data = ksppm.fetch_data()

if raw_data is not None:
    with st.expander("Raw Stock Price Data"):
        st.write("**Raw Data:**")
        st.dataframe(raw_data)

    raw_data['RSI'] = ksppm.calculate_rsi(raw_data)
    raw_data['MACD'], raw_data['Signal Line'] = ksppm.calculate_macd(raw_data)

    raw_data['RSI'] = raw_data['RSI'].fillna(method='ffill').fillna(method='bfill')
    raw_data['MACD'] = raw_data['MACD'].fillna(method='ffill').fillna(method='bfill')
    raw_data['Signal Line'] = raw_data['Signal Line'].fillna(method='ffill').fillna(method='bfill')
    raw_data.dropna(inplace=True)

    with st.expander("Processed Data with Technical Indicators"):
        st.write("**Processed Data (including RSI and MACD):**")
        st.dataframe(raw_data[['Close', 'RSI', 'MACD', 'Signal Line']])

    st.subheader("Technical Indicators for LSTM Model")

    st.markdown("### RSI (Relative Strength Index)")
    st.markdown("RSI measures the relative strength of stock prices, helping to identify overbought or oversold zones, providing important features for LSTM.")
    st.latex(r"""
    \text{RSI} = 100 - \frac{100}{1 + RS}, \quad \text{where } RS = \frac{\text{Average Gain}}{\text{Average Loss}}
    """)
    st.code(
        """
        @staticmethod
        def calculate_rsi(data, period=14):
            delta = data['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        """,
        language="python"
    )

    st.markdown("### MACD (Moving Average Convergence Divergence)")
    st.markdown("MACD analyzes price trends through the relationship between two moving averages, supporting LSTM in predicting price fluctuations.")
    st.latex(r"""
    \text{MACD} = \text{EMA}_{\text{short-term}} - \text{EMA}_{\text{long-term}}, \quad \text{Signal Line} = \text{EMA}_{\text{signal period}}(\text{MACD})
    """)
    st.code(
        """
        @staticmethod
        def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
            short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
            long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
            macd = short_ema - long_ema
            signal_line = macd.ewm(span=signal_window, adjust=False).mean()
            return macd, signal_line
        """,
        language="python"
    )

    st.subheader("Data Analysis for RNN (LSTM)")
    st.markdown(
        """
        - **Close Price Data**: Reflects stock price trends, which is the main target for LSTM prediction.
        - **RSI**: Provides information on price momentum, helping LSTM detect short-term volatility patterns.
        - **MACD**: Adds information on long-term trends and reversal points, increasing LSTM's predictive capability.
        - **Preprocessing**: Data is normalized and split into training/testing sets (ratio {train_ratio:.2f}), ready for the LSTM model.
        - **Conclusion**: Data has been enriched with RSI and MACD, suitable for training LSTM models to predict stock prices more accurately.
        """.format(train_ratio=train_ratio)
    )
else:
    st.error("Unable to fetch data. Please check the stock symbol or time period.")