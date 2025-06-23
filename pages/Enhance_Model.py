import streamlit as st
from app import menu
from Config import Config
from models.KerasSPPModule import KerasSPPModule as KSPPM

st.set_page_config(
    page_title=Config.APP_NAME,
    page_icon=Config.APP_ICON,
    layout="wide"
)
menu()

st.title('Triển khai mô hình RNN (LSTM) cho dự đoán giá cổ phiếu')

st.markdown(
    """
    Mô hình Mạng thần kinh hồi quy (RNN), cụ thể là LSTM, được sử dụng để dự đoán giá cổ phiếu dựa trên dữ liệu giá đóng cửa, tận dụng khả năng ghi nhớ các mẫu thời gian dài hạn.
    """
)

st.subheader("Mạng Thần Kinh Hồi Quy (RNN)")
st.markdown(
    """
    **Định nghĩa**: RNN là một loại mạng thần kinh nhân tạo được thiết kế cho dữ liệu tuần tự, nơi đầu ra của một bước thời gian được sử dụng làm đầu vào cho bước tiếp theo. Điều này cho phép RNN lưu giữ "bộ nhớ" về các dữ liệu trước đó thông qua trạng thái ẩn.

    **Cách hoạt động**:
    - Tại mỗi bước thời gian t, RNN xử lý đầu vào x[t] và trạng thái ẩn trước đó h[t-1] để tạo ra trạng thái ẩn mới h[t]:
    """
)

st.latex(r"h_t = \text{activation}(W_{xh}x[t] + W_{hh}h[t-1] + b_h)")

st.markdown(
    """
      với W_{xh}, W_{hh} là các ma trận trọng số, b_h là bias, và hàm kích hoạt thường là `tanh` hoặc `sigmoid`.
    - Đầu ra có thể được tính từ h_t để dự đoán.
    - RNN được huấn luyện bằng lan truyền ngược qua thời gian (BPTT), nhưng gặp vấn đề về gradient biến mất hoặc bùng nổ, hạn chế khả năng học các phụ thuộc dài hạn.

    **Hạn chế**: Do vấn đề gradient, RNN không hiệu quả với các chuỗi dài, như dữ liệu giá cổ phiếu trong nhiều tháng.
    """
)

st.subheader("Long Short-Term Memory (LSTM)")
st.markdown(
    """
    **Định nghĩa**: LSTM là một biến thể cải tiến của RNN, được thiết kế để học các phụ thuộc dài hạn. Nó sử dụng ô nhớ (memory cell) và ba cổng (gates) để kiểm soát dòng thông tin.

    **Cách hoạt động**:
    - **Ô nhớ (Cell State)**: Lưu giữ thông tin qua các bước thời gian, cho phép ghi nhớ dài hạn.
    - **Cổng quên (Forget Gate)**: Quyết định thông tin nào cần loại bỏ:
    """
)

st.latex(r"f_t = \sigma(W_f \cdot [h[t-1], x[t]] + b_f)")

st.markdown("- **Cổng vào (Input Gate)**: Quyết định thông tin mới được thêm vào:")

st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i), \quad \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)")

st.markdown("- **Cập nhật ô nhớ**:")

st.latex(r"C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t")

st.markdown("- **Cổng ra (Output Gate)**: Tạo đầu ra và cập nhật trạng thái ẩn:")

st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o), \quad h_t = o_t \cdot \tanh(C_t)")

st.markdown(
    """
    - Các cổng này cho phép LSTM chọn lọc thông tin quan trọng, tránh vấn đề gradient biến mất.

    **Ưu điểm**: LSTM có thể học các mẫu phức tạp và phụ thuộc dài hạn, rất phù hợp cho dữ liệu giá cổ phiếu.
    """
)

st.subheader("Tại sao LSTM phù hợp để dự đoán giá cổ phiếu")
st.markdown(
    """
    - **Phụ thuộc thời gian**: Giá cổ phiếu là dữ liệu chuỗi thời gian, nơi giá trị quá khứ ảnh hưởng đến tương lai. LSTM ghi nhớ các mẫu như xu hướng tăng/giảm qua nhiều ngày hoặc tháng.
    - **Mô hình hóa phi tuyến**: Giá cổ phiếu bị ảnh hưởng bởi các yếu tố phi tuyến (ví dụ: tâm lý thị trường, sự kiện kinh tế). LSTM có khả năng học các mối quan hệ phức tạp này.
    - **Bộ nhớ dài hạn**: LSTM có thể ghi nhớ các sự kiện quan trọng từ quá khứ (ví dụ: biến động lớn cách đây vài tháng), giúp dự đoán chính xác hơn.
    - **Xử lý nhiễu**: Dữ liệu giá cổ phiếu có nhiều nhiễu. Các cổng của LSTM giúp tập trung vào các mẫu quan trọng, bỏ qua nhiễu.
    - **Khả năng tuần tự**: LSTM xử lý dữ liệu theo chuỗi (ví dụ: 60 ngày giá đóng cửa), cho phép học cách các giá trị trước ảnh hưởng đến giá trị sau.
    """
)

# st.subheader("Triển khai mô hình LSTM")
# st.markdown(
#     """
#     **Các bước thực hiện**:
#     1. **Thu thập dữ liệu**: Tải giá cổ phiếu (giá đóng cửa) bằng API Yahoo Finance.
#     2. **Tiền xử lý**: Chuẩn hóa dữ liệu và tạo chuỗi (look-back = 60 ngày) để đưa vào LSTM.
#     3. **Xây dựng mô hình**: Mô hình LSTM với 2 tầng LSTM (50 đơn vị), dropout (0.2) để tránh overfitting, và tầng dense để dự đoán giá.
#     4. **Huấn luyện**: Huấn luyện trên tập dữ liệu (tỷ lệ huấn luyện: {train_ratio:.2f}) trong 50 epoch.
#     5. **Dự đoán**: Dự đoán giá trên tập kiểm tra và hiển thị kết quả qua biểu đồ cùng các chỉ số MSE, MAE, R².
#     """.format(train_ratio=0.7)
# )

st.write("Các mã cổ phiếu phổ biến:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Nhập mã cổ phiếu", "AAPL")
period = st.selectbox("Chọn khoảng thời gian", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Chọn tỷ lệ huấn luyện", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

ksppm = KSPPM(stock_symbol, period, train_ratio)
ksppm.fetch_data()
ksppm.prepare_data()
ksppm.train_model()
ksppm.predict()
ksppm.visualize()