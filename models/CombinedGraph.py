import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from .BasicSPPModule import BasicSPPModule
from .ImprovedSPPModule import ImprovedSPPModule
from .KerasSPPModule import KerasSPPModule

class CombinedGraph:
    def __init__(self, stock_symbol, period, train_ratio, start_date=None, end_date=None):
        self.stock_symbol = stock_symbol
        self.period = period
        self.train_ratio = train_ratio
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.basic_module = BasicSPPModule(stock_symbol, period, train_ratio)
        self.improved_module = ImprovedSPPModule(stock_symbol, period, train_ratio)
        self.keras_module = KerasSPPModule(stock_symbol, period, train_ratio)

    def fetch_and_prepare_data(self):
        self.data = yf.download(self.stock_symbol, period=self.period, progress=False)
        if self.data.empty:
            raise ValueError("No data fetched. Check stock symbol or period.")
        
        self.data['Days'] = np.arange(len(self.data))
        self.data.dropna(inplace=True)
        
        self.basic_module.data = self.data.copy()
        self.improved_module.data = self.data.copy()
        self.keras_module.data = self.data.copy()

        self.data['RSI'] = self.improved_module.calculate_rsi(self.data)
        self.data['MACD'], self.data['Signal Line'] = self.improved_module.calculate_macd(self.data)
        self.data = self.data.dropna()

        X_basic = self.data[['Days']]
        X_improved = self.data[['Days', 'RSI', 'MACD']]
        y = self.data['Close']

        train_days = int(len(self.data) * self.train_ratio)
        X_basic_train = X_basic[:train_days]
        X_basic_test = X_basic[train_days:]
        X_improved_train = X_improved[:train_days]
        X_improved_test = X_improved[train_days:]
        y_train = y[:train_days]
        y_test = y[train_days:]

        self.basic_module.X_train = X_basic_train
        self.basic_module.X_test = X_basic_test
        self.basic_module.y_train = y_train
        self.basic_module.y_test = y_test

        self.improved_module.X_train = X_improved_train
        self.improved_module.X_test = X_improved_test
        self.improved_module.y_train = y_train
        self.improved_module.y_test = y_test

        self.keras_module.fetch_data()
        self.keras_module.prepare_data()

    def visualize(self):
        self.fetch_and_prepare_data()
        
        self.basic_module.train_model()
        self.basic_module.predict()
        
        self.improved_module.train_model()
        self.improved_module.predict()
        
        self.keras_module.train_model()
        self.keras_module.predict()

        plt.figure(figsize=(14, 6))
        
        plt.plot(self.data['Days'], self.data['Close'], label="Actual Prices", color="blue", marker="o")
        plt.plot(self.basic_module.X_train['Days'], self.basic_module.model.predict(self.basic_module.X_train), label="Basic Model (Train)", color="orange", linestyle="--")
        plt.plot(self.basic_module.X_test['Days'], self.basic_module.y_pred, label="Basic Model (Predict)", color="red", linestyle="--")
        plt.plot(self.improved_module.X_train['Days'], self.improved_module.model.predict(self.improved_module.X_train), label="Improved Model (Train)", color="green", linestyle="--")
        plt.plot(self.improved_module.X_test['Days'], self.improved_module.y_pred, label="Improved Model (Predict)", color="purple", linestyle="--")
        plt.plot(self.data['Days'][-len(self.keras_module.y_test):], self.keras_module.y_pred, label="Keras Model (Predict)", color="cyan", linestyle="--")
        
        plt.axvline(x=self.basic_module.X_test['Days'].iloc[0], color="grey", linestyle="--", label="Train-Test Split")
        
        plt.xlabel("Days")
        plt.ylabel("Stock Closing Price")
        plt.title(f"Comparison of Stock Price Predictions ({self.stock_symbol})")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        plt.close()

        basic_metrics = self.basic_module.evaluate()
        improved_metrics = self.improved_module.evaluate()
        keras_metrics = self.keras_module.evaluate()

        st.write("### So sánh hiệu suất mô hình")
        st.write("**Chỉ số mô hình cơ bản (Basic Linear Regression):**")
        st.write(f"- Mean Squared Error (MSE): {basic_metrics['MSE']:.2f}")
        st.write(f"- Mean Absolute Error (MAE): {basic_metrics['MAE']:.2f}")
        st.write(f"- R-squared (R²): {basic_metrics['R2']:.2f}")
        st.write("**Chỉ số mô hình cải tiến (Improved Linear Regression):**")
        st.write(f"- Mean Squared Error (MSE): {improved_metrics['MSE']:.2f}")
        st.write(f"- Mean Absolute Error (MAE): {improved_metrics['MAE']:.2f}")
        st.write(f"- R-squared (R²): {improved_metrics['R2']:.2f}")
        st.write("**Chỉ số mô hình Keras (LSTM):**")
        st.write(f"- Mean Squared Error (MSE): {keras_metrics['MSE']:.2f}")
        st.write(f"- Mean Absolute Error (MAE): {keras_metrics['MAE']:.2f}")
        st.write(f"- R-squared (R²): {keras_metrics['R2']:.2f}")

        # # Future predictions with Keras model
        # if self.start_date and self.end_date:
        #     st.subheader("Dự đoán giá cổ phiếu trong tương lai (Keras Model)")
        #     future_pred = self.keras_module.predict_future(self.start_date, self.end_date)
        #     if future_pred is not None:
        #         st.dataframe(future_pred)
        #         plt.figure(figsize=(14, 6))
        #         plt.plot(future_pred['Date'], future_pred['Predicted Close'], label="Keras Future Predictions", color="cyan", linestyle="--")
        #         plt.xlabel("Date")
        #         plt.ylabel("Predicted Stock Closing Price")
        #         plt.title(f"Future Stock Price Predictions for {self.stock_symbol} (Keras Model)")
        #         plt.legend()
        #         plt.grid()
        #         st.pyplot(plt)
        #         plt.close()