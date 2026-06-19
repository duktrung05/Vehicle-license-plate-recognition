# ─── Base Stage ──────────────────────────────────────────────────────────────
FROM python:3.10-slim as base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create output directories
RUN mkdir -p outputs/uploaded outputs/crops outputs/processed_plates \
    outputs/result_images outputs/result_videos database

ENV PYTHONPATH=/app:/app/src

# ─── Backend Stage ───────────────────────────────────────────────────────────
FROM base as backend
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ─── Frontend Stage ──────────────────────────────────────────────────────────
FROM base as frontend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
