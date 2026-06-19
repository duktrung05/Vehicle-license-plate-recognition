## Test 

Dự án có sẵn vài ảnh biển số mẫu trong `xulyanh/data/samples/` để bạn test ngay không cần tự chuẩn bị dữ liệu.

### Cách chạy:

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
