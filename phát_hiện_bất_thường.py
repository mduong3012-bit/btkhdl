import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import time
import random
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder

# Cấu hình trang web Streamlit
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận", layout="wide")

st.title("🛡️ Ứng dụng Phát hiện Giao dịch Bất thường (Anomaly Detection)")
st.write("Hệ thống sử dụng thuật toán **Isolation Forest** để phân tích và phát hiện các giao dịch đáng ngờ.")

# Session state cho giám sát thời gian thực
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Hàm sinh dữ liệu giao dịch ngẫu nhiên
def generate_random_transaction():
    accounts = ['ACC001', 'ACC002', 'ACC003', 'ACC004', 'ACC005']
    merchants = ['Shop_A', 'Shop_B', 'Shop_C', 'Restaurant', 'GasStation']
    locations = ['HN', 'SG', 'DN', 'HCM', 'CT']
    trans_types = ['Online', 'ATM', 'POS', 'Transfer']
    
    return {
        'AccountID': random.choice(accounts),
        'Amount': round(random.uniform(50000, 5000000), 2),
        'Merchant': random.choice(merchants),
        'Location': random.choice(locations),
        'TransactionType': random.choice(trans_types),
        'Timestamp': datetime.now() - timedelta(seconds=random.randint(0, 3600))
    }

# File uploader hỗ trợ csv và xlsx
uploaded_file = st.file_uploader("Tải lên file dữ liệu giao dịch của bạn (.csv hoặc .xlsx)", type=["csv", "xlsx"])

# Thư mục lưu mô hình
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")


def load_file(file_obj):
    file_name = file_obj.name.lower()
    # Excel
    if file_name.endswith('.xlsx'):
        try:
            return pd.read_excel(file_obj, engine='openpyxl')
        except Exception as e:
            st.error(f"Lỗi đọc file Excel: {e}")
            return None
    # CSV
    file_obj.seek(0)
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding='latin1')
    except pd.errors.ParserError:
        file_obj.seek(0)
        try:
            return pd.read_csv(file_obj, sep=None, engine='python', on_bad_lines='warn')
        except pd.errors.ParserError:
            file_obj.seek(0)
            return pd.read_csv(file_obj, sep=';', engine='python', on_bad_lines='warn')


if uploaded_file is not None:
    df = load_file(uploaded_file)
    if df is None:
        st.stop()

    st.success("Tải dữ liệu lên thành công!")
    st.subheader("📊 Xem trước dữ liệu vừa tải lên")
    st.dataframe(df.head(10))

    # Chọn chế độ hoạt động
    mode = st.sidebar.selectbox("Chế độ hoạt động", 
                               ["1. Train (Huấn luyện)", 
                                "2. Predict (Dự đoán)", 
                                "3. Giám sát thời gian thực"])

    # Kiểm tra cột bắt buộc
    required_columns = ['Timestamp', 'Amount']
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        st.error(f"File thiếu cột bắt buộc: {', '.join(missing_required)}")
        st.stop()

    # Contamination slider
    st.sidebar.header("Cấu hình mô hình")
    contamination = st.sidebar.slider("Tỷ lệ gian lận dự kiến (Contamination)", min_value=0.005, max_value=0.05, value=0.01, step=0.005)

    # Tiền xử lý: Timestamp -> Hour, Is_Weekend, unix
    df_model = df.copy()
    try:
        df_model['Timestamp'] = pd.to_datetime(df_model['Timestamp'], errors='coerce')
        if df_model['Timestamp'].isna().any():
            raise ValueError('Một hoặc nhiều giá trị Timestamp không thể chuyển đổi')
    except Exception as e:
        st.error(f"Lỗi với cột Timestamp: {e}")
        st.stop()

    df_model['Hour'] = df_model['Timestamp'].dt.hour
    df_model['Is_Weekend'] = df_model['Timestamp'].dt.weekday >= 5
    df_model['Timestamp'] = (df_model['Timestamp'].astype('int64') // 10**9).astype(int)

    # Cột phân loại có thể có
    candidate_cat_cols = ['AccountID', 'Merchant', 'TransactionType', 'Location']
    categorical_cols = [c for c in candidate_cat_cols if c in df_model.columns]

    # Drop TransactionID nếu có
    drop_cols = [c for c in ['TransactionID'] if c in df_model.columns]

    # Sử dụng OrdinalEncoder để xử lý giá trị lạ (unknown_value=-1)
    encoder = None
    if "Train" in mode:
        if categorical_cols:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoder.fit(df_model[categorical_cols].astype(str))
            joblib.dump({'encoder': encoder, 'cat_cols': categorical_cols}, ENCODER_PATH)
    else:
        if os.path.exists(ENCODER_PATH):
            enc_obj = joblib.load(ENCODER_PATH)
            encoder = enc_obj.get('encoder')
            categorical_cols = enc_obj.get('cat_cols', categorical_cols)
        else:
            st.error('Không tìm thấy bộ mã hóa đã lưu. Chạy chế độ Train trước.')
            st.stop()

    # Áp mã hóa nếu có
    if categorical_cols and encoder is not None:
        try:
            cat_enc = encoder.transform(df_model[categorical_cols].astype(str))
            cat_df = pd.DataFrame(cat_enc, columns=[f'enc_{c}' for c in categorical_cols], index=df_model.index)
            df_model = pd.concat([df_model.drop(columns=categorical_cols), cat_df], axis=1)
        except Exception as e:
            st.error(f"Lỗi mã hóa: {e}")
            st.stop()

    # Chuẩn bị X cho mô hình
    X = df_model.drop(columns=drop_cols)

    # Chạy mô hình theo chế độ
    with st.spinner('Hệ thống đang phân tích dữ liệu...'):
        if "Train" in mode:
            model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            model.fit(X)
            joblib.dump({'model': model, 'features': X.columns.tolist()}, MODEL_PATH)
            preds = model.predict(X)
            df['anomaly_score'] = preds
            df['is_fraud'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
            st.success('Huấn luyện hoàn tất — mô hình đã được lưu.')
        elif "Predict" in mode:
            if os.path.exists(MODEL_PATH):
                model_obj = joblib.load(MODEL_PATH)
                model = model_obj.get('model')
                preds = model.predict(X)
                df['anomaly_score'] = preds
                df['is_fraud'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
            else:
                st.error('Không tìm thấy mô hình đã lưu. Chạy chế độ Train trước.')
                st.stop()
        elif "Giám sát" in mode:
            # Real-time monitoring mode - sẽ xử lý bên dưới
            st.info("Chế độ giám sát thời gian thực - xem bên dưới")
            preds = None

    # Thống kê
    total_tx = len(df)
    fraud_tx = int(df['is_fraud'].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số giao dịch", f"{total_tx:,}")
    col2.metric("Số giao dịch bất thường", f"{fraud_tx:,}", delta=f"{fraud_tx/total_tx*100:.2f}%", delta_color="inverse")
    col3.metric("Trạng thái hệ thống", "Hoạt động tốt")

    # Biểu đồ và danh sách - chỉ hiển thị nếu không ở chế độ giám sát
    if "Giám sát" not in mode:
        # Biểu đồ tương tác Plotly
        st.subheader("📈 Biểu đồ phân tích (Di chuột để xem chi tiết)")
        color_map = {0: 'Normal', 1: 'Anomaly'}
        df_plot = df.copy()
        df_plot['label'] = df_plot['is_fraud'].map({0: 'Normal', 1: 'Anomaly'})
        hover_cols = [c for c in ['AccountID', 'Amount', 'Merchant', 'Location'] if c in df.columns]
        fig = px.scatter(df_plot, x=df_plot.index, y='Amount', color='label', color_discrete_map={'Normal':'blue','Anomaly':'red'},
                         hover_data=hover_cols, labels={'x':'Số thứ tự giao dịch', 'Amount':'Số tiền'})
        st.plotly_chart(fig, use_container_width=True)

        # Biểu đồ Pie
        st.subheader('Tỷ lệ giao dịch: Bình thường vs Bất thường')
        pie_df = df_plot['label'].value_counts().reset_index()
        pie_df.columns = ['label', 'count']
        fig2 = px.pie(pie_df, names='label', values='count', color='label', color_discrete_map={'Normal':'blue','Anomaly':'red'},
                      title='Tỷ lệ giao dịch')
        st.plotly_chart(fig2, use_container_width=True)

        # Danh sách giao dịch nghi ngờ
        st.subheader("🚨 Danh sách các giao dịch nghi ngờ gian lận")
        fraud_list = df[df['is_fraud'] == 1].drop(columns=['anomaly_score', 'is_fraud'])
        st.dataframe(fraud_list)
    else:
        # Chế độ giám sát thời gian thực
        st.divider()
        st.subheader("🔴 Giám sát thời gian thực")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_btn = st.button("▶️ Bắt đầu giám sát", key="start_monitoring")
        with col_btn2:
            stop_btn = st.button("⏹️ Dừng giám sát", key="stop_monitoring")
        
        # Kiểm tra mô hình có lưu chưa
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Không tìm thấy mô hình đã lưu. Vui lòng chạy chế độ Train trước.")
        elif start_btn:
            st.session_state.monitoring = True
            st.session_state.alerts = []
        
        if stop_btn:
            st.session_state.monitoring = False
        
        # Vòng lặp giám sát
        alert_placeholder = st.empty()
        table_placeholder = st.empty()
        
        if st.session_state.monitoring and os.path.exists(MODEL_PATH):
            model_obj = joblib.load(MODEL_PATH)
            model = model_obj.get('model')
            enc_obj = joblib.load(ENCODER_PATH)
            encoder = enc_obj.get('encoder')
            cat_cols = enc_obj.get('cat_cols', [])
            
            transaction_count = 0
            max_transactions = 10  # Giới hạn để test
            
            while st.session_state.monitoring and transaction_count < max_transactions:
                # Sinh giao dịch ngẫu nhiên
                trans = generate_random_transaction()
                transaction_count += 1
                
                # Chuẩn bị dữ liệu
                trans_df = pd.DataFrame([trans])
                trans_model = trans_df.copy()
                
                # Feature engineering
                trans_model['Hour'] = trans_model['Timestamp'].dt.hour
                trans_model['Is_Weekend'] = trans_model['Timestamp'].dt.weekday >= 5
                trans_model['Timestamp'] = (trans_model['Timestamp'].astype('int64') // 10**9).astype(int)
                
                # Mã hóa
                if cat_cols:
                    cat_enc = encoder.transform(trans_model[cat_cols].astype(str))
                    cat_df = pd.DataFrame(cat_enc, columns=[f'enc_{c}' for c in cat_cols])
                    trans_model = pd.concat([trans_model.drop(columns=cat_cols), cat_df], axis=1)
                
                X_trans = trans_model.drop(columns=['TransactionID'] if 'TransactionID' in trans_model.columns else [])
                
                # Dự đoán
                pred = model.predict(X_trans)[0]
                is_fraud = 1 if pred == -1 else 0
                
                # Ghi nhận cảnh báo
                if is_fraud:
                    st.session_state.alerts.append({
                        'Timestamp': trans.get('Timestamp', 'N/A'),
                        'AccountID': trans.get('AccountID', 'N/A'),
                        'Amount': trans.get('Amount', 'N/A'),
                        'Merchant': trans.get('Merchant', 'N/A'),
                        'Location': trans.get('Location', 'N/A'),
                        'Status': '⚠️ BẤT THƯỜNG'
                    })
                    with alert_placeholder.container():
                        st.error(f"🚨 PHÁT HIỆN GIAO DỊCH ĐÁNG NGỜ! | {trans.get('AccountID')} | {trans.get('Amount'):,.0f} VND")
                else:
                    with alert_placeholder.container():
                        st.success(f"✓ Giao dịch bình thường | {trans.get('AccountID')} | {trans.get('Amount'):,.0f} VND")
                
                # Hiển thị bảng danh sách cảnh báo
                if st.session_state.alerts:
                    with table_placeholder.container():
                        st.subheader("📋 Danh sách cảnh báo")
                        alerts_df = pd.DataFrame(st.session_state.alerts)
                        st.dataframe(alerts_df, use_container_width=True)
                
                # Độ trễ 1 giây
                time.sleep(1)

else:
    st.info("Nhắc nhở: Vui lòng kéo thả hoặc chọn file để bắt đầu phân tích.")