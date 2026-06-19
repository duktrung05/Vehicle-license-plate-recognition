# 🚗 Hệ Thống Nhận Diện Biển Số Xe Tự Động (Automatic Vehicle License Plate Recognition)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)

Một ứng dụng end-to-end xử lý và nhận diện biển số xe từ hình ảnh/video sử dụng các kỹ thuật Xử lý ảnh truyền thống kết hợp Deep Learning (YOLOv8 & OCR). Dự án được tối ưu hóa hiệu năng, đi kèm giao diện Web trực quan để demo và thử nghiệm nhanh.

---

## 📌 Tính Năng Cốt Lõi (Key Features)

* **Phát hiện biển số (License Plate Detection):** Sử dụng YOLOv8 được tinh chỉnh (fine-tune) để khoanh vùng chính xác vị trí biển số xe trong nhiều điều kiện ánh sáng khác nhau.
* **Tiền xử lý ảnh nâng cao (Advanced Image Preprocessing):** Áp dụng các bộ lọc đồ họa (Grayscale, Bilateral Filter, Canny Edge, Otsu Thresholding) để làm nét ký tự, giảm nhiễu trước khi nhận diện.
* **Nhận diện ký tự (OCR Engine):** Phân đoạn ký tự và đọc nội dung biển số xe với độ chính xác cao, hỗ trợ cả biển số 1 dòng và 2 dòng.
* **Giao diện Web trực quan (Web UI):** Tích hợp Streamlit cho phép người dùng upload ảnh, xem trực tiếp kết quả bounding box và text trả về theo thời gian thực.

---

## 🏗️ Kiến Trúc Hệ Thống (System Architecture)

Dự án được thiết kế theo cấu trúc mô-đun hóa (Modular Design) giúp dễ dàng bảo trì, mở rộng và viết Unit Test:

```text
xulyanh_v3/
├── xulyanh/
│   ├── data/
│   │   └── samples/              # Ảnh mẫu phục vụ demo nhanh
│   ├── frontend/
│   │   └── streamlit_app.py      # Giao diện Web (Streamlit)
│   ├── models/
│   │   └── yolov8/               # Lưu trữ file trọng số model (.pt)
│   └── src/                      # Source code xử lý chính
│       ├── detection/            # Mô-đun phát hiện vùng chứa biển số
│       ├── preprocessing/        # Mô-đun tiền xử lý ảnh (bộ lọc, góc nghiêng)
│       ├── recognition/          # Mô-đun OCR nhận diện ký tự
│       └── pipeline/             # Kết nối các mô-đun thành luồng xử lý hoàn chỉnh
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn dự án

```
### Cách chạy:
cd xulyanh_v3
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
1. Cài thư viện cần thiết:
```bash
   pip install -r requirements.txt
```
2. Chạy app Streamlit:
```bash
   streamlit run xulyanh/frontend/streamlit_app.py
```
3. Trình duyệt sẽ tự mở `http://localhost:8501` — tại đây bạn:
   - Bấm **Upload** ảnh
   - Chọn ảnh mẫu có sẵn trong thư mục `xulyanh/data/samples/`
   - Xem kết quả nhận diện biển số trực tiếp trên giao diện

Bạn cũng có thể dùng ảnh biển số xe của riêng mình để test, không bắt buộc dùng ảnh mẫu.
