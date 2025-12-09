import os
import streamlit as st
from Config import Config

def menu():
    st.sidebar.page_link('app.py', label='Home')
    st.sidebar.page_link('pages/Analyse_Data.py', label='Data Analysis')
    st.sidebar.page_link('pages/Enhance_Model.py', label='Build RNN Model')
    st.sidebar.page_link('pages/Evaluation.py', label='Evaluation & Conclusion')

if __name__ == '__main__':
    st.set_page_config(
        page_title=Config.APP_NAME,
        page_icon=Config.APP_ICON,
        layout="wide"
    )

    st.title(f'{Config.APP_NAME}')
    st.write('Version:‎ ‎ ‎ ‎', Config.APP_VERSION)
    st.markdown("**Author:** Tran Cong Minh")

    st.divider()
  
    st.subheader('Idea:')
    st.markdown(
        """
        <p style="font-size:18px; text-align:justify;">
        Analyze the effectiveness and feasibility of Recurrent Neural Network (RNN) models, specifically LSTM, in predicting stock prices, while comparing with linear regression models to evaluate performance.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.subheader('Implementation:')
    st.markdown(
        """
        <ul style="font-size:18px; text-align:justify; list-style-type:disc; line-height:32px;">
            <li>Select specific time periods and stock symbols to collect stock price data.</li>
            <li>Data Collection: Use Yahoo Finance API (yfinance) to download stock price data for the selected period.</li>
            <li>Data Preprocessing: Remove missing data, calculate technical indicators (RSI, MACD), normalize data, and split into training and testing sets.</li>
            <li>Model Building: Use LSTM model (a type of RNN) to predict stock prices, while implementing basic and improved linear regression models for comparison.</li>
            <li>Model Evaluation: Compare the performance of LSTM with linear regression models through metrics such as MSE, MAE, R² and prediction visualization charts.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    menu()