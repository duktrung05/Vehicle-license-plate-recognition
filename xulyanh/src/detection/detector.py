import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed. Using contour-based detector.")


class LicensePlateDetector:

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.model_path = model_path

        if model_path and Path(model_path).is_file() and YOLO_AVAILABLE:
            self._load_model(model_path)
        else:
            logger.info("No valid YOLO model found. Using contour-based fallback.")

    def _load_model(self, model_path: str):
        """Load YOLOv8 model."""
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Model classes: {self.model.names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> List[dict]:
        if self.model is not None:
            return self._detect_yolo(image)

        return self._detect_contour(image)

    def _is_plate_class(self, class_name: str) -> bool:
        normalized = (
            str(class_name)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        plate_class_names = {
            "bien_so_xe",
            "license_plate",
            "licence_plate",
            "plate",
            "plates",
            "number_plate",
        }

        return normalized in plate_class_names

    def _detect_yolo(self, image: np.ndarray) -> List[dict]:
        """Run YOLOv8 inference and only keep license plate class."""
        detections = []

        try:
            results = self.model(
                image,
                conf=self.conf_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return self._detect_contour(image)

        h, w = image.shape[:2]

        for result in results:
            if result.boxes is None:
                continue

            names = getattr(result, "names", None) or getattr(self.model, "names", {})

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = str(names.get(cls_id, cls_id))
                conf = float(box.conf[0])

                # Important: skip character detections
                if not self._is_plate_class(cls_name):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Clamp bbox to image bounds
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "crop": crop,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "class": "license_plate",
                    }
                )

        return detections

    def _detect_contour(self, image: np.ndarray) -> List[dict]:
        detections = []
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(
            edged.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = cw / ch if ch > 0 else 0

                if 1.5 < aspect_ratio < 6.0 and cw > 60 and ch > 15:
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(w, x + cw)
                    y2 = min(h, y + ch)

                    crop = image[y1:y2, x1:x2]

                    detections.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": 0.6,
                            "crop": crop,
                            "class_id": -1,
                            "class_name": "contour_fallback",
                            "class": "license_plate",
                        }
                    )

                    break

        return detections

    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes on image."""
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return result

    def get_model_info(self) -> dict:
        """Return model information."""
        return {
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "backend": "YOLOv8" if self.model is not None else "Contour-based",
            "conf_threshold": self.conf_threshold,
            "device": self.device,
            "classes": getattr(self.model, "names", None) if self.model is not None else None,
        }