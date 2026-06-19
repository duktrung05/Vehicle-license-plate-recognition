"""
Pydantic response schemas for the FastAPI backend
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime


class DetectionResult(BaseModel):
    id: Optional[int] = None
    bbox: Optional[List[float]] = None
    detection_confidence: float = 0.0
    ocr_confidence: float = 0.0
    raw_text: str = ""
    formatted_text: str = ""
    is_valid: bool = False
    province: Optional[str] = None
    province_code: Optional[str] = None
    plate_type: Optional[str] = None
    crop_path: Optional[str] = None


class PredictResponse(BaseModel):
    status: str = "success"
    source_file: str = ""
    detections: List[DetectionResult] = []
    processing_time_ms: float = 0.0
    total_plates: int = 0
    result_image_path: Optional[str] = None
    error: Optional[str] = None


class VideoResultResponse(BaseModel):
    status: str = "success"
    source_file: str = ""
    output_path: Optional[str] = None
    total_frames: int = 0
    processed_frames: int = 0
    unique_plates: int = 0
    detections: List[DetectionResult] = []
    processing_time_sec: float = 0.0
    error: Optional[str] = None


class PlateLogEntry(BaseModel):
    id: int
    timestamp: str
    source_type: str
    source_file: Optional[str] = None
    plate_text: str
    confidence: float
    bbox: Optional[str] = None
    province_code: Optional[str] = None
    province_name: Optional[str] = None
    plate_type: Optional[str] = None
    is_valid: bool
    processing_time_ms: float
    image_path: Optional[str] = None
    crop_path: Optional[str] = None
    metadata: Optional[str] = None


class HistoryResponse(BaseModel):
    total: int
    offset: int
    limit: int
    logs: List[PlateLogEntry]


class StatsResponse(BaseModel):
    total_detections: int
    valid_plates: int
    validation_rate: float
    avg_confidence: float
    today_detections: int
    top_provinces: List[Dict[str, Any]] = []


class EvaluationRequest(BaseModel):
    predictions: List[str]
    ground_truths: List[str]
    dataset_name: str = ""
    notes: str = ""


class EvaluationResponse(BaseModel):
    exact_match_rate: float
    mean_char_accuracy: float
    mean_edit_distance: float
    total_samples: int
    exact_matches: int
    evaluation_id: Optional[int] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    detector_backend: str
    ocr_backend: str
    database_connected: bool
