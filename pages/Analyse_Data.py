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

st.title('Phân tích dữ liệu giá cổ phiếu cho mô hình RNN (LSTM)')

st.write("Các mã cổ phiếu phổ biến:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Nhập mã cổ phiếu", "AAPL")
period = st.selectbox("Chọn khoảng thời gian", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Chọn tỷ lệ huấn luyện", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

ksppm = KSPPM(stock_symbol, period, train_ratio)
raw_data = ksppm.fetch_data()

if raw_data is not None:
    with st.expander("Dữ liệu giá cổ phiếu gốc"):
        st.write("**Dữ liệu gốc (Raw Data):**")
        st.dataframe(raw_data)

    raw_data['RSI'] = ksppm.calculate_rsi(raw_data)
    raw_data['MACD'], raw_data['Signal Line'] = ksppm.calculate_macd(raw_data)

    raw_data['RSI'] = raw_data['RSI'].fillna(method='ffill').fillna(method='bfill')
    raw_data['MACD'] = raw_data['MACD'].fillna(method='ffill').fillna(method='bfill')
    raw_data['Signal Line'] = raw_data['Signal Line'].fillna(method='ffill').fillna(method='bfill')
    raw_data.dropna(inplace=True)

    with st.expander("Dữ liệu sau khi xử lý và thêm các chỉ số kỹ thuật"):
        st.write("**Dữ liệu đã xử lý (bao gồm RSI và MACD):**")
        st.dataframe(raw_data[['Close', 'RSI', 'MACD', 'Signal Line']])

    st.subheader("Chỉ báo kỹ thuật cho mô hình LSTM")

    st.markdown("### RSI (Relative Strength Index)")
    st.markdown("RSI đo lường sức mạnh tương đối của giá cổ phiếu, giúp xác định các vùng quá mua hoặc quá bán, cung cấp đặc trưng quan trọng cho LSTM.")
    st.latex(r"""
    \text{RSI} = 100 - \frac{100}{1 + RS}, \quad \text{với } RS = \frac{\text{Tăng trung bình}}{\text{Giảm trung bình}}
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
    st.markdown("MACD phân tích xu hướng giá thông qua mối quan hệ giữa hai đường trung bình động, hỗ trợ LSTM dự đoán biến động giá.")
    st.latex(r"""
    \text{MACD} = \text{EMA}_{\text{ngắn hạn}} - \text{EMA}_{\text{dài hạn}}, \quad \text{Đường tín hiệu} = \text{EMA}_{\text{kỳ tín hiệu}}(\text{MACD})
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

    st.subheader("Phân tích dữ liệu cho RNN (LSTM)")
    st.markdown(
        """
        - **Dữ liệu giá đóng cửa**: Phản ánh xu hướng giá cổ phiếu, là mục tiêu chính để LSTM dự đoán.
        - **RSI**: Cung cấp thông tin về động lượng giá, giúp LSTM phát hiện các mẫu biến động ngắn hạn.
        - **MACD**: Bổ sung thông tin về xu hướng dài hạn và điểm đảo chiều, tăng khả năng dự đoán của LSTM.
        - **Tiền xử lý**: Dữ liệu được chuẩn hóa và chia thành tập huấn luyện/kiểm tra (tỷ lệ {train_ratio:.2f}), sẵn sàng cho mô hình LSTM.
        - **Kết luận**: Dữ liệu đã được làm giàu với RSI và MACD, phù hợp để huấn luyện mô hình LSTM nhằm dự đoán giá cổ phiếu chính xác hơn.
        """.format(train_ratio=train_ratio)
    )
else:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra mã cổ phiếu hoặc khoảng thời gian.")