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

st.title('Implementing RNN (LSTM) Model for Stock Price Prediction')

st.markdown(
    """
    Recurrent Neural Network (RNN) models, specifically LSTM, are used to predict stock prices based on closing price data, leveraging the ability to remember long-term temporal patterns.
    """
)

st.subheader("Recurrent Neural Network (RNN)")
st.markdown(
    """
    **Definition**: RNN is a type of artificial neural network designed for sequential data, where the output of one time step is used as input for the next. This allows RNNs to retain "memory" of previous data through hidden states.

    **How it works**:
    - At each time step t, RNN processes input x[t] and the previous hidden state h[t-1] to generate a new hidden state h[t]:
    """
)

st.latex(r"h_t = \text{activation}(W_{xh}x[t] + W_{hh}h[t-1] + b_h)")

st.markdown(
    """
      where W_{xh}, W_{hh} are weight matrices, b_h is bias, and the activation function is typically `tanh` or `sigmoid`.
    - The output can be calculated from h_t for prediction.
    - RNNs are trained using Backpropagation Through Time (BPTT), but suffer from vanishing or exploding gradient problems, limiting their ability to learn long-term dependencies.

    **Limitations**: Due to gradient problems, RNNs are ineffective with long sequences, such as stock price data over many months.
    """
)

st.subheader("Long Short-Term Memory (LSTM)")
st.markdown(
    """
    **Definition**: LSTM is an improved variant of RNN, designed to learn long-term dependencies. It uses a memory cell and three gates to control the flow of information.

    **How it works**:
    - **Cell State**: Retains information across time steps, enabling long-term memory.
    - **Forget Gate**: Decides what information to discard:
    """
)

st.latex(r"f_t = \sigma(W_f \cdot [h[t-1], x[t]] + b_f)")

st.markdown("- **Input Gate**: Decides what new information to add:")

st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i), \quad \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)")

st.markdown("- **Update Cell State**:")

st.latex(r"C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t")

st.markdown("- **Output Gate**: Generates output and updates the hidden state:")

st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o), \quad h_t = o_t \cdot \tanh(C_t)")

st.markdown(
    """
    - These gates allow LSTM to selectively retain important information, avoiding the vanishing gradient problem.

    **Advantages**: LSTM can learn complex patterns and long-term dependencies, making it well-suited for stock price data.
    """
)

st.subheader("Why LSTM is Suitable for Stock Price Prediction")
st.markdown(
    """
    - **Time Dependence**: Stock prices are time-series data where past values influence the future. LSTM remembers patterns like upward/downward trends over days or months.
    - **Non-linear Modeling**: Stock prices are affected by non-linear factors (e.g., market sentiment, economic events). LSTM is capable of learning these complex relationships.
    - **Long-term Memory**: LSTM can remember significant past events (e.g., major fluctuations months ago), leading to more accurate predictions.
    - **Noise Handling**: Stock price data is noisy. LSTM gates help focus on important patterns while ignoring noise.
    - **Sequential Capability**: LSTM processes data in sequences (e.g., 60 days of closing prices), allowing it to learn how previous values affect subsequent ones.
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

st.write("Popular Stock Symbols:")
st.markdown("""AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, KO, PEP, DIS, NFLX, INTC""")
stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Period", ["3mo", "6mo", "1y", "2y", "5y"])
train_ratio = st.slider("Select Training Ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)

ksppm = KSPPM(stock_symbol, period, train_ratio)
ksppm.fetch_data()
ksppm.prepare_data()
ksppm.train_model()
ksppm.predict()
ksppm.visualize()