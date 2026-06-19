"""
FastAPI Backend for License Plate Recognition System
Run: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from backend.routers import predict, history, evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="License Plate Recognition API",
    description="""
    ## 🚗 Vietnamese License Plate Recognition System
    
    Full pipeline: YOLOv8 Detection → Preprocessing → OCR → Validation → Storage
    
    ### Features
    - **Image Recognition**: Upload single images
    - **Video Processing**: Process video files
    - **History**: View and filter past detections
    - **Evaluation**: Compute accuracy metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(history.router)
app.include_router(evaluation.router)

# Serve output files statically
outputs_dir = Path(__file__).parent.parent / "outputs"
outputs_dir.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "License Plate Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        from storage.database import PlateDatabase
        db = PlateDatabase()
        stats = db.get_stats()
        db_ok = True
    except Exception:
        db_ok = False

    # Check model
    model_path = Path(__file__).parent.parent / "models" / "yolov8" / "best.pt"

    try:
        from detection.detector import LicensePlateDetector
        det = LicensePlateDetector()
        detector_info = det.get_model_info()
        detector_backend = detector_info["backend"]
    except Exception:
        detector_backend = "unavailable"

    try:
        from recognition.ocr_engine import OCREngine
        ocr = OCREngine()
        ocr_info = ocr.get_engine_info()
        ocr_backend = ocr_info["active_engine"]
    except Exception:
        ocr_backend = "unavailable"

    return {
        "status": "healthy",
        "version": "1.0.0",
        "detector_backend": detector_backend,
        "ocr_backend": ocr_backend,
        "model_file_exists": model_path.exists(),
        "database_connected": db_ok,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
