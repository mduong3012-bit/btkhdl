# btkhdl
# 🛡️ Hệ Thống Phát Hiện Giao Dịch Bất Thường (Fraud & Anomaly Detection Web App)

Đây là bài tập lớn xây dựng ứng dụng web phân tích và phát hiện các giao giao dịch tài chính bất thường (nghi ngờ gian lận) dựa trên thuật toán Học máy không giám sát (Unsupervised Machine Learning) **Isolation Forest**. Ứng dụng được đóng gói trực quan hóa bằng framework **Streamlit**.

---

## 🚀 Tính Năng Chính Của Ứng Dụng
- **Tải tệp dữ liệu tùy chỉnh:** Cho phép người dùng kéo thả hoặc tải lên các file dữ liệu giao dịch định dạng `.csv`.
- **Mô hình hóa thời gian thực:** Tích hợp thanh trượt (Slider) để điều chỉnh tỷ lệ gian lận kỳ vọng (`contamination`), mô hình phía sau sẽ tự động tính toán lại ngay lập tức.
- **Trực quan hóa đa chiều:** Hiển thị biểu đồ phân tán (*Scatter Plot*) chỉ rõ các điểm giao dịch an toàn (Màu xanh) và điểm bất thường (Màu đỏ).
- **Trích xuất dữ liệu:** Tự động lọc và hiển thị danh sách chi tiết các giao dịch bị gắn nhãn nguy hiểm để phục vụ công tác hậu kiểm.

---

## 🛠️ Công Nghệ & Thư Viện Sử Dụng
Dự án được phát triển bằng ngôn ngữ **Python** và các thư viện chuyên dụng cho Khoa học dữ liệu:
- `Streamlit`: Xây dựng giao diện Web App.
- `Pandas` & `Numpy`: Xử lý và tiền xử lý cấu trúc dữ liệu bảng.
- `Scikit-learn`: Triển khai mô hình `Isolation Forest` và mã hóa dữ liệu `LabelEncoder`.
- `Matplotlib` & `Seaborn`: Vẽ biểu đồ trực quan hóa.

---

## 📦 Hướng Dẫn Cài Đặt Và Chạy Ứng Dụng

Để vận hành hệ thống này trên máy tính cá nhân hoặc **GitHub Codespaces**, vui lòng thực hiện theo các bước sau trong Terminal:

### Bước 1: Khởi tạo và kích hoạt môi trường ảo (venv)
```bash
# Tạo môi trường ảo tên là venv
python3 -m venv venv

# Kích hoạt môi trường ảo (Dành cho Linux/MacOS/GitHub Codespaces)
source venv/bin/activate

# Kích hoạt môi trường ảo (Dành cho Windows nếu chạy CMD)
# venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện cần thiết
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn openpyxl
```

### Bước 3: Chạy ứng dụng Streamlit
```bash
streamlit run phát_hiện_bất_thường.py
```

### Lưu ý
- Ứng dụng hiện hỗ trợ tải lên tệp dữ liệu định dạng `.csv` và `.xlsx`.
- Nếu bạn muốn chạy app ở chế độ public, file cấu hình `.streamlit/config.toml` đã được chuẩn bị để lắng nghe trên `0.0.0.0`.
- Truy cập giao diện qua `http://localhost:8501` khi chạy trên máy địa phương.
