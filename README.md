# Phishing URL Detection using Machine Learning 🛡️

> **Bài tập lớn môn học: Học máy (Machine Learning - CO3117)**
> **Học kỳ:** 262
> **Giảng viên hướng dẫn:** Võ Thanh Hùng

## 📖 Giới thiệu (Introduction)

Dự án này xây dựng một hệ thống tự động phát hiện các đường dẫn lừa đảo (Phishing URLs) sử dụng các kỹ thuật Học máy. Thay vì sử dụng danh sách đen (Blacklist) truyền thống, hệ thống phân tích các đặc trưng của URL (cấu trúc, từ ngữ, tên miền) để dự đoán khả năng độc hại, giúp phát hiện các cuộc tấn công mới (Zero-day phishing attacks).

## 🚀 Tính năng nổi bật (Key Features)

*   **Hybrid Feature Engineering:** Kết hợp giữa đặc trưng văn bản (Lexical) và đặc trưng máy chủ (Host-based).
*   **Semantic Analysis:** Phát hiện các từ khóa nhạy cảm (ví dụ: *login, verify, secure...*) thường được kẻ tấn công sử dụng.
*   **High Performance:** Sử dụng các mô hình Ensemble Learning mạnh mẽ (Random Forest, XGBoost) cho độ chính xác cao (~89%).
*   **Real-time Optimization:** Loại bỏ các truy vấn mạng chậm (WHOIS) để đảm bảo tốc độ dự đoán cực nhanh (~4 giây để huấn luyện mô hình XGBoost).

## 📂 Cấu trúc dự án (Project Structure)
phishing-url-detection/
├── data/
│ ├── raw/ # Dữ liệu thô tải từ Kaggle
│ └── processed/ # Dữ liệu đã làm sạch và trích xuất đặc trưng
├── notebooks/
│ ├── 01_Data_Preprocessing.ipynb # Làm sạch và chuẩn hóa dữ liệu
│ ├── 04_Hybrid_Model.ipynb # Trích xuất đặc trưng (Feature Extraction)
│ └── 05_Model_Comparison_and_Tuning.ipynb # Huấn luyện, so sánh và test mô hình
├── src/
│ └── features/
│ ├── lexical_features.py # Hàm xử lý chuỗi URL
│ └── host_features.py # Hàm xử lý domain/host
├── report/ # Báo cáo PDF và Slide
├── requirements.txt # Các thư viện cần thiết
└── README.md # Tài liệu hướng dẫn

## 🛠️ Cài đặt và Môi trường (Installation)

Dự án được phát triển trên Python 3.8+. Để chạy dự án, vui lòng thực hiện các bước sau:

1.  **Clone repository:**
    ```bash
    git clone https://github.com/thiennguyencrystal/phishing-url-detection
    cd phishing-url-detection
    ```

2.  **Tạo môi trường ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện phụ thuộc:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Hướng dẫn sử dụng (Usage)

Vui lòng chạy các Notebook theo thứ tự sau để tái tạo kết quả:

### Bước 1: Chuẩn bị dữ liệu
*   Chạy `notebooks/01_Data_Preprocessing.ipynb`.
*   Notebook này sẽ đọc dữ liệu thô, xử lý trùng lặp và lưu file sạch vào `data/processed/dataset_processed.csv`.

### Bước 2: Trích xuất đặc trưng (Quan trọng)
*   Chạy `notebooks/04_Hybrid_Model.ipynb`.
*   Notebook này áp dụng các hàm từ `src/features/` để chuyển đổi URL thành vector số.
*   **Lưu ý:** Quá trình này có thể mất vài phút. Kết quả được lưu tại `data/processed/dataset_final_features.csv`.

### Bước 3: Huấn luyện và So sánh
*   Chạy `notebooks/05_Model_Comparison_and_Tuning.ipynb`.
*   Notebook này sẽ:
    1.  So sánh hiệu suất của Logistic Regression, Random Forest và XGBoost.
    2.  Tinh chỉnh tham số (Hyperparameter Tuning) cho mô hình tốt nhất.
    3.  Cho phép bạn nhập một URL bất kỳ để kiểm tra thử (Demo).

## 📊 Kết quả thực nghiệm (Results)

| Mô hình                |  Accuracy  | Precision (Phishing) | Recall (Phishing) |  F1-Score  | Thời gian Train |
| :--------------------- | :--------: | :------------------: | :---------------: | :--------: | :-------------: |
| Logistic Regression    |    85.4%   |        0.8235        |     **0.4497**    |   0.5817   |      28.9s      |
| Random Forest          | **92.53%** |        0.8709        |     **0.7847**    | **0.8255** |      60.6s      |
| **XGBoost (Selected)** |   91.71%   |      **0.8797**      |       0.7323      |   0.7993   |    **5.76s**    |


> **Kết luận:** Nhóm chọn **XGBoost** là mô hình cuối cùng nhờ sự cân bằng tốt giữa độ chính xác và tốc độ vượt trội, phù hợp cho các ứng dụng thực tế.

## ⚠️ Hạn chế (Limitations)

*   Mô hình dựa chủ yếu vào đặc trưng cấu trúc (Lexical), nên có thể gặp khó khăn với các trang web uy tín nhưng có cấu trúc đơn giản (ví dụ: thiếu `www`).
*   Do loại bỏ tính năng WHOIS để tối ưu tốc độ, mô hình thiếu thông tin về tuổi đời tên miền.

## 👥 Nhóm thực hiện (Authors)

*   **Nguyễn Phan Thanh Thiên** - *MSSV: 2313220*
*   **Bùi Quang Huy** - *MSSV: 2311124*

---
*Dự án này được thực hiện nhằm mục đích học tập và nghiên cứu.*