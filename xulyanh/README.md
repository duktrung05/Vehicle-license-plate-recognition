```bash
cd xulyanh
python -m pip install -r requirements.txt
```

### 1. Chạy Giao diện Người dùng (Streamlit Frontend)
Chạy lệnh sau từ thư mục `xulyanh`:
```bash
python -m streamlit run frontend/streamlit_app.py
```

### 2. Chạy API Backend (FastAPI Web Service)
Nếu bạn cần sử dụng hệ thống qua các API endpoint (tích hợp hệ thống bên thứ ba, Mobile app, v.v.):

Chạy lệnh sau từ thư mục `xulyanh`:
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

