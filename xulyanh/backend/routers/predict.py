"""
Prediction API routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2] / "src"))

from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline
from backend.schemas.response_schema import PredictResponse, VideoResultResponse, DetectionResult

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Lazy-init pipelines
_image_pipeline = None
_video_pipeline = None


def get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        model_path = str(Path(__file__).parents[2] / "models" / "yolov8" / "best.pt")
        _image_pipeline = ImagePipeline(
            model_path=model_path if Path(model_path).exists() else None
        )
    return _image_pipeline


def get_video_pipeline():
    global _video_pipeline
    if _video_pipeline is None:
        model_path = str(Path(__file__).parents[2] / "models" / "yolov8" / "best.pt")
        _video_pipeline = VideoPipeline(
            model_path=model_path if Path(model_path).exists() else None
        )
    return _video_pipeline


@router.post("/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    """Upload and process a single image for license plate recognition."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    pipeline = get_image_pipeline()
    result = pipeline.process_bytes(contents, filename=file.filename)

    detections = [DetectionResult(**d) for d in result.get("detections", [])]

    return PredictResponse(
        status=result.get("status", "error"),
        source_file=file.filename,
        detections=detections,
        processing_time_ms=result.get("processing_time_ms", 0),
        total_plates=len(detections),
        result_image_path=result.get("result_image_path"),
        error=result.get("error"),
    )


@router.post("/video", response_model=VideoResultResponse)
async def predict_video(file: UploadFile = File(...)):
    """Upload and process a video file for license plate recognition."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    contents = await file.read()
    pipeline = get_video_pipeline()
    result = pipeline.process_video_bytes(contents, filename=file.filename)

    detections = [DetectionResult(**d) for d in result.get("detections", [])]

    return VideoResultResponse(
        status=result.get("status", "error"),
        source_file=file.filename,
        output_path=result.get("output_path"),
        total_frames=result.get("total_frames", 0),
        processed_frames=result.get("processed_frames", 0),
        unique_plates=result.get("unique_plates", 0),
        detections=detections,
        processing_time_sec=result.get("processing_time_sec", 0),
        error=result.get("error"),
    )


@router.get("/result-image/{filename}")
async def get_result_image(filename: str):
    """Serve a result image by filename."""
    base = Path(__file__).parents[2] / "outputs" / "result_images"
    path = base / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path))
