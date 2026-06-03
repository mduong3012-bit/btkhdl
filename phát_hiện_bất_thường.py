import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import OneClassSVM

# Cấu hình trang web Streamlit
st.set_page_config(page_title="Hệ thống Phát hiện Gian lận", layout="wide")

st.title("🛡️ Ứng dụng Phát hiện Giao dịch Bất Thường (Anomaly Detection)")
st.write("Hệ thống sử dụng thuật toán không giám sát để phát hiện các giao dịch bất thường, với benchmark giữa Isolation Forest và One-Class SVM.")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
IF_MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest_model.joblib")
OCSVM_MODEL_PATH = os.path.join(MODEL_DIR, "one_class_svm_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")


def load_file(file_obj):
    file_name = file_obj.name.lower()
    if file_name.endswith(".xlsx"):
        try:
            return pd.read_excel(file_obj, engine="openpyxl")
        except Exception as exc:
            st.error(f"Lỗi đọc file Excel: {exc}")
            return None

    file_obj.seek(0)
    try:
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding="latin1")
    except pd.errors.ParserError:
        file_obj.seek(0)
        try:
            return pd.read_csv(file_obj, sep=None, engine="python", on_bad_lines="warn")
        except pd.errors.ParserError:
            file_obj.seek(0)
            return pd.read_csv(file_obj, sep=";", engine="python", on_bad_lines="warn")


def validate_and_engineer(df):
    required_columns = ["Timestamp", "Amount"]
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        st.error(f"File thiếu cột bắt buộc: {', '.join(missing_required)}")
        st.stop()

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        st.error("Một hoặc nhiều giá trị Timestamp không thể chuyển đổi thành định dạng thời gian.")
        st.stop()

    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.weekday
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Timestamp"] = (df["Timestamp"].astype("int64") // 10**9).astype(int)

    if "AccountID" in df.columns:
        account_mean = df.groupby("AccountID")["Amount"].transform("mean")
        account_mean = account_mean.replace(0, df["Amount"].mean() if df["Amount"].mean() != 0 else 1)
        df["Amount_to_Mean_Ratio"] = df["Amount"] / account_mean
    else:
        overall_mean = df["Amount"].mean() if df["Amount"].mean() != 0 else 1
        df["Amount_to_Mean_Ratio"] = df["Amount"] / overall_mean

    return df


def build_preprocessor(df, categorical_cols, numeric_cols):
    encoder = None
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(df[categorical_cols].astype(str))

    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])

    processor = {
        "encoder": encoder,
        "scaler": scaler,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "feature_names": numeric_cols + ([f"enc_{c}" for c in categorical_cols] if categorical_cols else []),
    }
    return processor


def transform_features(df, processor):
    df_out = pd.DataFrame(index=df.index)
    df_out[processor["numeric_cols"] ] = processor["scaler"].transform(df[processor["numeric_cols"]])
    if processor["categorical_cols"]:
        cat_enc = processor["encoder"].transform(df[processor["categorical_cols"]].astype(str))
        cat_df = pd.DataFrame(cat_enc, columns=[f"enc_{c}" for c in processor["categorical_cols"]], index=df.index)
        df_out = pd.concat([df_out, cat_df], axis=1)
    return df_out


def tune_isolation_forest(X, y):
    best = {
        "f1": -1,
        "contamination": None,
        "n_estimators": None,
    }
    for contamination in [0.005, 0.01, 0.02, 0.03]:
        for n_estimators in [50, 100, 150]:
            model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
            preds = model.fit_predict(X)
            pred_labels = np.where(preds == -1, 1, 0)
            score = f1_score(y, pred_labels, zero_division=0)
            if score > best["f1"]:
                best.update({"f1": score, "contamination": contamination, "n_estimators": n_estimators})
    return best


def evaluate_model(name, y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, title):
    labels = ["Normal", "Anomaly"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        hovertemplate="%{y} vs %{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    return fig


uploaded_file = st.file_uploader("Tải lên dữ liệu giao dịch (.csv hoặc .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df_raw = load_file(uploaded_file)
    if df_raw is None:
        st.stop()

    st.success("Tải dữ liệu lên thành công!")
    st.subheader("📊 Xem trước dữ liệu vừa tải lên")
    st.dataframe(df_raw.head(10))

    label_exists = "is_fraud_ground_truth" in df_raw.columns
    labels_available = st.sidebar.checkbox("Dữ liệu có sẵn nhãn đối chứng", value=label_exists)
    if labels_available and not label_exists:
        st.error("Không tìm thấy cột 'is_fraud_ground_truth' trong dữ liệu.")
        st.stop()

    mode = st.sidebar.radio("Chế độ hoạt động", ("Train (Huấn luyện)", "Predict (Dự đoán)"))
    selected_model = st.sidebar.selectbox("Chọn mô hình", ("Isolation Forest", "One-Class SVM"))

    st.sidebar.header("Cấu hình mô hình")
    contamination = st.sidebar.slider("Contamination (dành cho Isolation Forest)", min_value=0.005, max_value=0.05, value=0.01, step=0.005)
    nu_value = st.sidebar.slider("Nu (dành cho One-Class SVM)", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    df = validate_and_engineer(df_raw)
    df["Is_Weekend"] = df["Is_Weekend"].astype(int)

    candidate_cat_cols = ["AccountID", "Merchant", "TransactionType", "Location"]
    categorical_cols = [c for c in candidate_cat_cols if c in df.columns]
    numeric_cols = [
        "Timestamp",
        "Amount",
        "Hour_sin",
        "Hour_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "Is_Weekend",
        "Amount_to_Mean_Ratio",
    ]

    preprocessor = None
    if mode.startswith("Train"):
        preprocessor = build_preprocessor(df, categorical_cols, numeric_cols)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
    else:
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            categorical_cols = preprocessor["categorical_cols"]
            numeric_cols = preprocessor["numeric_cols"]
        else:
            st.error("Không tìm thấy bộ biến đổi tiền xử lý đã lưu. Vui lòng chạy chế độ Train trước.")
            st.stop()

    X = transform_features(df, preprocessor)

    y_true = None
    if labels_available:
        y_true = df["is_fraud_ground_truth"].astype(int).values

    results = []
    best_if_params = None
    if mode.startswith("Train"):
        if labels_available:
            best_if_params = tune_isolation_forest(X, y_true)
            st.sidebar.write(f"Tuning IA: contamination={best_if_params['contamination']}, n_estimators={best_if_params['n_estimators']}, F1={best_if_params['f1']:.3f}")
        else:
            best_if_params = {"contamination": contamination, "n_estimators": 100, "f1": None}

        if best_if_params is None or best_if_params["contamination"] is None:
            best_if_params = {"contamination": contamination, "n_estimators": 100}

        if "Isolation Forest" in ("Isolation Forest",):
            if_params = {
                "contamination": best_if_params["contamination"],
                "n_estimators": best_if_params["n_estimators"],
                "random_state": 42,
            }
            model_if = IsolationForest(**if_params)
            model_if.fit(X)
            joblib.dump({"model": model_if, "features": X.columns.tolist()}, IF_MODEL_PATH)
            preds_if = model_if.predict(X)
            y_pred_if = np.where(preds_if == -1, 1, 0)
            df["anomaly_score_if"] = preds_if
            df["is_fraud_if"] = y_pred_if
            if labels_available:
                results.append(evaluate_model("Isolation Forest", y_true, y_pred_if))

        model_ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu_value)
        model_ocsvm.fit(X)
        joblib.dump({"model": model_ocsvm, "features": X.columns.tolist()}, OCSVM_MODEL_PATH)
        preds_ocsvm = model_ocsvm.predict(X)
        y_pred_ocsvm = np.where(preds_ocsvm == -1, 1, 0)
        df["anomaly_score_ocsvm"] = preds_ocsvm
        df["is_fraud_ocsvm"] = y_pred_ocsvm
        if labels_available:
            results.append(evaluate_model("One-Class SVM", y_true, y_pred_ocsvm))

        st.success("Huấn luyện hoàn tất — các mô hình và bộ biến đổi đã được lưu.")

    else:
        model_path = IF_MODEL_PATH if selected_model == "Isolation Forest" else OCSVM_MODEL_PATH
        if not os.path.exists(model_path):
            st.error(f"Không tìm thấy mô hình {selected_model} đã lưu. Vui lòng chạy chế độ Train trước.")
            st.stop()
        model_obj = joblib.load(model_path)
        model = model_obj["model"]
        preds = model.predict(X)
        df["anomaly_score"] = preds
        df["is_fraud"] = np.where(preds == -1, 1, 0)
        if labels_available:
            results.append(evaluate_model(selected_model, y_true, df["is_fraud"].values))

    if mode.startswith("Predict"):
        st.sidebar.success(f"Sử dụng mô hình: {selected_model}")

    total_tx = len(df)
    fraud_tx = int(df["is_fraud"] .sum()) if "is_fraud" in df.columns else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số giao dịch", f"{total_tx:,}")
    col2.metric("Số giao dịch bất thường", f"{fraud_tx:,}", delta=f"{fraud_tx/total_tx*100:.2f}%", delta_color="inverse")
    col3.metric("Trạng thái hệ thống", "Hoạt động tốt")

    st.subheader("📈 Biểu đồ phân tích giao dịch")
    df_plot = df.copy()
    if mode.startswith("Train"):
        df_plot["label"] = np.where(df_plot["is_fraud_if"] == 1, "Anomaly", "Normal")
    else:
        df_plot["label"] = np.where(df_plot["is_fraud"] == 1, "Anomaly", "Normal")

    hover_cols = [c for c in ["AccountID", "Amount", "Merchant", "Location", "Hour", "DayOfWeek"] if c in df_plot.columns]
    fig = px.scatter(
        df_plot,
        x=df_plot.index,
        y="Amount",
        color="label",
        color_discrete_map={"Normal": "blue", "Anomaly": "red"},
        hover_data=hover_cols,
        labels={"x": "Số thứ tự giao dịch", "Amount": "Số tiền"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Tỷ lệ giao dịch")
    if mode.startswith("Train"):
        pie_label = df_plot["label"].value_counts().reset_index()
    else:
        pie_label = df_plot["label"].value_counts().reset_index()
    pie_label.columns = ["label", "count"]
    fig2 = px.pie(
        pie_label,
        names="label",
        values="count",
        color="label",
        color_discrete_map={"Normal": "blue", "Anomaly": "red"},
        title="Tỷ lệ giao dịch",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if labels_available and results:
        st.subheader("📈 Báo cáo thực nghiệm Thuật toán")
        metric_df = pd.DataFrame([{
            "Model": r["model"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1_score"],
        } for r in results])
        st.dataframe(metric_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}"}))

        for r in results:
            st.write(f"### Ma trận nhầm lẫn - {r['model']}")
            cm_fig = plot_confusion_matrix(r["confusion_matrix"], f"Confusion Matrix ({r['model']})")
            st.plotly_chart(cm_fig, use_container_width=True)

    st.subheader("🚨 Danh sách các giao dịch nghi ngờ gian lận")
    if mode.startswith("Train"):
        fraud_list = df[df["is_fraud_if"] == 1].drop(columns=[col for col in ["anomaly_score_if", "is_fraud_if", "anomaly_score_ocsvm", "is_fraud_ocsvm"] if col in df.columns])
    else:
        fraud_list = df[df["is_fraud"] == 1].drop(columns=["anomaly_score", "is_fraud"])
    st.dataframe(fraud_list)

else:
    st.info("Nhắc nhở: Vui lòng kéo thả hoặc chọn file để bắt đầu phân tích.")