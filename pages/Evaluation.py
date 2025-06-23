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

st.title('Đánh giá & Kết luận')

st.subheader('***So sánh các mô hình:***')

st.write("Các mã cổ phiếu phổ biến:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Nhập mã cổ phiếu", "AAPL")
period = st.selectbox("Chọn khoảng thời gian", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Chọn tỷ lệ huấn luyện", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

# st.subheader("Dự đoán tương lai (Keras Model)")
# start_date = st.date_input("Chọn ngày bắt đầu", value=datetime(2025, 5, 23))
# end_date = st.date_input("Chọn ngày kết thúc", value=datetime(2025, 6, 23))

# # Validate date range
# if start_date >= end_date:
#     st.error("Ngày kết thúc phải sau ngày bắt đầu.")
# else:
cg = CG(stock_symbol, period, train_ratio)
cg.visualize()
st.subheader('***Đánh giá:***')
st.markdown(
    """
    1. Từ việc so sánh giữa ba mô hình (hồi quy tuyến tính cơ bản, hồi quy tuyến tính cải tiến, và mô hình Keras LSTM), ta thấy mô hình Keras có khả năng nắm bắt xu hướng thời gian tốt hơn nhờ sử dụng cấu trúc LSTM.
    2. Mô hình cải tiến và Keras đều có độ chính xác cao hơn mô hình cơ bản, với mô hình Keras thường có sai số thấp hơn trong các trường hợp phức tạp.
    3. Việc sử dụng các chỉ số kỹ thuật (RSI, MACD) và dữ liệu thời gian đã cải thiện đáng kể hiệu suất của cả mô hình hồi quy tuyến tính cải tiến và mô hình Keras.
    4. Mô hình Keras có thể dự đoán giá cổ phiếu trong tương lai với độ linh hoạt cao hơn nhờ khả năng học các mẫu phi tuyến.
    """
)

st.subheader('***Kết luận:***')
st.markdown(
    """
    ***Việc sử dụng mô hình hồi quy tuyến tính (Linear Regression) là khả thi khi kết hợp với các chỉ số kỹ thuật, nhưng mô hình Keras LSTM mang lại khả năng dự đoán mạnh mẽ hơn trong các tình huống phức tạp. Sự kết hợp giữa dữ liệu kỹ thuật và mô hình học sâu là chìa khóa để cải thiện độ chính xác dự đoán giá cổ phiếu.***
    """
)
st.markdown(
    """
    ***Lưu ý:*** Mô hình Keras vẫn có thể được tối ưu hóa thêm bằng cách điều chỉnh kiến trúc mạng, thêm các đặc trưng mới, hoặc sử dụng các kỹ thuật như cross-validation.
    """
)