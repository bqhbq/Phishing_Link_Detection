"""Thư viện dùng chung cho pipeline `notebooks_new/` (phát hiện Phishing URL).

Các module:
    config        – đường dẫn, hằng số, seed
    features      – chuẩn hóa URL + trích xuất đặc trưng (registry tường minh, dùng chung train/inference)
    preprocessing – transformer tùy biến (IQR clipping) cho sklearn Pipeline
    evaluation    – metrics, chọn ngưỡng, biểu đồ đánh giá, bootstrap CI
"""
