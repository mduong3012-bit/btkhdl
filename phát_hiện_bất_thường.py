import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# Cấu hình trang web Streamlit
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận", layout="wide")

st.title("🛡️ Ứng dụng Phát hiện Giao dịch Bất thường (Anomaly Detection)")
st.write("Hệ thống sử dụng thuật toán **Isolation Forest** để phân tích và phát hiện các giao dịch đáng ngờ.")

# 1. Thành phần tải file lên giao diện web
uploaded_file = st.file_uploader("Tải lên file dữ liệu giao dịch của bạn (.csv)", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    
    st.success("Tải dữ liệu lên thành công!")
    
    # Hiển thị dữ liệu thô dạng bảng trên giao diện
    st.subheader("📊 Xem trước dữ liệu vừa tải lên")
    st.dataframe(df.head(10))

    # Thanh điều chỉnh tỷ lệ gian lận (Contamination) trên giao diện web
    st.sidebar.header("Cấu hình mô hình")
    contamination = st.sidebar.slider(
        "Tỷ lệ gian lận dự kiến (Contamination)", 
        min_value=0.005, max_value=0.05, value=0.01, step=0.005
    )

    # 2. Tiền xử lý dữ liệu để đưa vào mô hình
    df_model = df.copy()
    df_model['Timestamp'] = (pd.to_datetime(df_model['Timestamp']).astype('int64') // 10**9).astype(int)

    le = LabelEncoder()
    categorical_cols = ['AccountID', 'Merchant', 'TransactionType', 'Location']
    for col in categorical_cols:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    X = df_model.drop(columns=['TransactionID'])

    # 3. Chạy thuật toán Isolation Forest
    with st.spinner('Hệ thống đang phân tích dữ liệu...'):
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        df['anomaly_score'] = model.fit_predict(X)
        df['is_fraud'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # 4. Hiển thị kết quả thống kê tổng quan
    total_tx = len(df)
    fraud_tx = df['is_fraud'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số giao dịch", f"{total_tx:,}")
    col2.metric("Số giao dịch bất thường", f"{fraud_tx:,}", delta=f"{fraud_tx/total_tx*100:.2f}%", delta_color="inverse")
    col3.metric("Trạng thái hệ thống", "Hoạt động tốt")

    # 5. Vẽ đồ thị trực quan hóa
    st.subheader("📈 Biểu đồ phân tích (Điểm đỏ là giao dịch bất thường)")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=df, x=df.index, y='Amount', hue='is_fraud', palette={0: 'blue', 1: 'red'}, alpha=0.6, ax=ax)
    ax.set_xlabel("Số thứ tự giao dịch")
    ax.set_ylabel("Số tiền (Amount)")
    st.pyplot(fig)

    # 6. Bộ lọc và hiển thị danh sách gian lận
    st.subheader("🚨 Danh sách các giao dịch nghi ngờ gian lận")
    fraud_list = df[df['is_fraud'] == 1].drop(columns=['anomaly_score', 'is_fraud'])
    st.dataframe(fraud_list)

else:
    st.info("Nhắc nhở: Vui lòng kéo thả hoặc chọn file để bắt đầu phân tích.")
