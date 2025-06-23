import os
import streamlit as st
from Config import Config

def menu():
    st.sidebar.page_link('app.py', label='Trang chính')
    st.sidebar.page_link('pages/Analyse_Data.py', label='Phân tích dữ liệu')
    st.sidebar.page_link('pages/Enhance_Model.py', label='Xây dựng mô hình RNN')
    st.sidebar.page_link('pages/Evaluation.py', label='Đánh giá mô hình & Kết luận')

if __name__ == '__main__':
    st.set_page_config(
        page_title=Config.APP_NAME,
        page_icon=Config.APP_ICON,
        layout="wide"
    )

    st.title(f'{Config.APP_NAME}')
    st.write('Version:‎ ‎ ‎ ‎', Config.APP_VERSION)
    st.markdown('Trần Công Minh  -  CSI10')

    st.divider()
  
    st.subheader('Ý tưởng:')
    st.markdown(
        """
        <p style="font-size:18px; text-align:justify;">
        Phân tích độ hiệu quả và tính khả thi của mô hình Mạng thần kinh hồi quy (Recurrent Neural Network - RNN), cụ thể là LSTM, trong việc dự đoán giá cổ phiếu, đồng thời so sánh với các mô hình hồi quy tuyến tính để đánh giá hiệu suất.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.subheader('Triển khai:')
    st.markdown(
        """
        <ul style="font-size:18px; text-align:justify; list-style-type:disc; line-height:32px;">
            <li>Lựa chọn khoảng thời gian và mã cổ phiếu cụ thể để thu thập dữ liệu giá cổ phiếu.</li>
            <li>Thu thập dữ liệu: Sử dụng API Yahoo Finance (yfinance) để tải dữ liệu giá cổ phiếu trong khoảng thời gian đã chọn.</li>
            <li>Tiền xử lý dữ liệu: Loại bỏ dữ liệu thiếu, tính toán các chỉ số kỹ thuật (RSI, MACD), chuẩn hóa dữ liệu, và chia thành tập huấn luyện và tập kiểm tra.</li>
            <li>Xây dựng mô hình: Sử dụng mô hình LSTM (một loại RNN) để dự đoán giá cổ phiếu, đồng thời triển khai các mô hình hồi quy tuyến tính cơ bản và cải tiến để so sánh.</li>
            <li>Đánh giá mô hình: So sánh hiệu suất của LSTM với các mô hình hồi quy tuyến tính thông qua các chỉ số như MSE, MAE, R² và biểu đồ trực quan hóa dự đoán.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    menu()