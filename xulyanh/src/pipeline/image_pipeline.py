import cv2
import numpy as np
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Add parent to path
import sys
sys.path.append(str(Path(__file__).parents[2]))

from detection.detector import LicensePlateDetector
from preprocessing.plate_preprocessor import PlatePreprocessor
from recognition.ocr_engine import OCREngine
from postprocessing.plate_formatter import PlateFormatter
from storage.database import PlateDatabase


BASE_DIR = Path(__file__).parents[3]
OUTPUTS_DIR = BASE_DIR / "outputs"


def ensure_dirs():
    for sub in ["uploaded", "crops", "processed_plates", "result_images"]:
        (OUTPUTS_DIR / sub).mkdir(parents=True, exist_ok=True)


class ImagePipeline:
    """
    Full pipeline for single image license plate recognition.

    Fix chính:
      - Tách layout:
          one_row_auto
          two_row_motorbike
          two_row_auto
      - Truyền layout_hint xuống OCR và Formatter.
      - Không xử mọi biển 2 dòng như xe máy.
      - Không tự thêm số 0 đầu dòng dưới.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        use_gpu: bool = False,
        save_results: bool = True,
        speed_mode: str = "balanced",
    ):
        ensure_dirs()
        self.save_results = save_results
        self.speed_mode = speed_mode

        self.detector = LicensePlateDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
        )
        self.preprocessor = PlatePreprocessor()
        self.ocr = OCREngine(use_gpu=use_gpu)
        self.formatter = PlateFormatter()
        self.db = PlateDatabase()

        logger.info("ImagePipeline initialized")
        logger.info(f"Detector: {self.detector.get_model_info()}")
        logger.info(f"OCR: {self.ocr.get_engine_info()}")

    # ======================================================================
    # MAIN PROCESS
    # ======================================================================

    def process(
        self,
        image: np.ndarray,
        source_file: str = "",
        log_to_db: bool = True,
        speed_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single image through the full pipeline.
        Returns structured result dict.
        """
        start_time = time.time()

        results = {
            "source_file": source_file,
            "detections": [],
            "annotated_image": None,
            "processing_time_ms": 0,
            "status": "success",
            "error": None,
        }

        try:
            detections = self.detector.detect(image)
            logger.info(f"Found {len(detections)} plate(s)")

            for i, det in enumerate(detections):
                crop = det.get("crop")
                bbox = det.get("bbox")
                det_conf = float(det.get("confidence", 0.0))

                if crop is None or crop.size == 0:
                    logger.warning("Empty crop, skip detection")
                    continue

                # Upscale small crops (common in video frames)
                # to preserve character strokes for OCR accuracy
                crop_h, crop_w = crop.shape[:2]
                if crop_h < 120 and crop_h > 0:
                    upscale_factor = max(1.5, 150.0 / crop_h)
                    crop = cv2.resize(
                        crop,
                        (int(crop_w * upscale_factor), int(crop_h * upscale_factor)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    logger.debug(f"Upscaled small crop from {crop_h}px to {crop.shape[0]}px height")

                plate_text = ""
                formatted = ""
                is_valid = False
                metadata = {}
                ocr_conf = 0.0
                proposals = []
                layout_hint = "unknown"

                # ----------------------------------------------------------
                # Step 1: classify layout + OCR
                # ----------------------------------------------------------
                layout_hint, plate_text, ocr_conf, proposals = self._ocr_by_layout(
                    crop,
                    speed_mode=speed_mode or self.speed_mode,
                )

                # ----------------------------------------------------------
                # Step 2: format + validate
                # ----------------------------------------------------------
                formatted, is_valid, metadata = self._format_plate(
                    plate_text,
                    layout_hint=layout_hint,
                )

                # Nếu formatter trả rỗng hoặc không hợp lệ thì hiển thị raw để debug (giữ nguyên dấu chấm)
                if not formatted or not is_valid:
                    formatted = plate_text

                # ----------------------------------------------------------
                # Step 3: save crop / processed image
                # ----------------------------------------------------------
                crop_path = ""
                processed_path = ""

                if self.save_results:
                    ts = int(time.time() * 1000)

                    crop_path = str(
                        OUTPUTS_DIR / "crops" / f"crop_{ts}_{i}.jpg"
                    )
                    cv2.imwrite(crop_path, crop)

                    if proposals:
                        processed_path = str(
                            OUTPUTS_DIR / "processed_plates" / f"proc_{ts}_{i}.jpg"
                        )
                        cv2.imwrite(processed_path, proposals[0])

                # ----------------------------------------------------------
                # Step 4: log DB
                # ----------------------------------------------------------
                elapsed = (time.time() - start_time) * 1000
                final_conf = float(np.mean([det_conf, ocr_conf]))

                db_id = None
                if log_to_db:
                    db_id = self.db.log_detection(
                        plate_text=formatted,
                        confidence=final_conf,
                        source_type="image",
                        source_file=source_file,
                        bbox=bbox,
                        province_code=metadata.get("province_code"),
                        province_name=metadata.get("province_name"),
                        plate_type=metadata.get("plate_type"),
                        is_valid=is_valid,
                        processing_time_ms=elapsed,
                        crop_path=crop_path,
                    )

                results["detections"].append({
                    "id": db_id,
                    "bbox": bbox,
                    "detection_confidence": det_conf,
                    "ocr_confidence": ocr_conf,
                    "raw_text": plate_text,
                    "formatted_text": formatted,
                    "is_valid": is_valid,
                    "province": metadata.get("province_name", "Unknown"),
                    "province_code": metadata.get("province_code"),
                    "plate_type": metadata.get("plate_type"),
                    "layout_hint": layout_hint,
                    "crop_path": crop_path,
                    "processed_path": processed_path,
                })

            # --------------------------------------------------------------
            # Step 5: annotate image
            # --------------------------------------------------------------
            annotated = self.detector.draw_detections(image, detections)

            if results["detections"]:
                # Sort detections strictly from top to bottom by vertical center
                sorted_dets = sorted(
                    results["detections"],
                    key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2
                )

                # Set parameters for index badges
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Draw index badge just above the top-left of each bounding box
                for idx, det_result in enumerate(sorted_dets):
                    x1, y1, x2, y2 = det_result["bbox"]
                    
                    # Draw index badge just above the top-left of the bounding box
                    badge_w, badge_h = 24, 18
                    by1 = max(0, y1 - badge_h - 2)
                    by2 = by1 + badge_h
                    bx1 = x1
                    bx2 = x1 + badge_w
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (41, 22, 15), cv2.FILLED)
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 212, 0), 1)
                    cv2.putText(
                        annotated,
                        str(idx + 1),
                        (bx1 + 6, by1 + 13),
                        font,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            if self.save_results:
                out_path = str(
                    OUTPUTS_DIR / "result_images" / f"result_{int(time.time() * 1000)}.jpg"
                )
                cv2.imwrite(out_path, annotated)
                results["result_image_path"] = out_path

            results["annotated_image"] = annotated

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            results["status"] = "error"
            results["error"] = str(e)

        results["processing_time_ms"] = (time.time() - start_time) * 1000
        return results

    # ======================================================================
    # OCR LAYOUT LOGIC
    # ======================================================================

    def _ocr_by_layout(
        self,
        crop: np.ndarray,
        speed_mode: str = "balanced",
    ) -> Tuple[str, str, float, List[np.ndarray]]:
        h, w = crop.shape[:2]
        aspect = w / max(1, h)

        # Force layout based on aspect ratio limits
        if aspect >= 2.2:
            is_two_row = False
        elif aspect < 1.7:
            is_two_row = True   
        else:
            try:
                is_two_row = self.preprocessor.is_two_row_plate(crop)
            except Exception as e:
                logger.warning(f"is_two_row_plate failed: {e}")
                is_two_row = False

        logger.debug(
            f"Plate crop shape: w={w}, h={h}, aspect={aspect:.2f}, "
            f"is_two_row={is_two_row}"
        )

        if is_two_row:
            # 1. Run split 2-row OCR
            layout_hint_2r, plate_text_2r, ocr_conf_2r, proposals_2r = self._ocr_two_row(
                crop,
                speed_mode=speed_mode,
            )
            # 2. Run holistic 1-row OCR
            layout_hint_1r, plate_text_1r, ocr_conf_1r, proposals_1r = self._ocr_one_row_auto(
                crop,
                speed_mode=speed_mode,
            )
            
            # If aspect ratio is small, force the holistic candidate to format as 2-row
            layout_hint_1r_for_format = layout_hint_2r if aspect < 1.6 else layout_hint_1r
            
            # Format and validate both candidates to see if they are valid civilian plates
            formatted_2r, is_valid_2r, _ = self._format_plate(plate_text_2r, layout_hint_2r)
            formatted_1r, is_valid_1r, _ = self._format_plate(plate_text_1r, layout_hint_1r_for_format)
            
            # If holistic is valid but split is not, choose holistic!
            if is_valid_1r and not is_valid_2r:
                logger.info(f"Hybrid OCR: Choosing holistic '{plate_text_1r}' over split '{plate_text_2r}' because holistic is valid.")
                return layout_hint_1r_for_format, plate_text_1r, ocr_conf_1r, proposals_1r
                
            # If both are valid, compare confidences and fallback based on layout
            if is_valid_1r and is_valid_2r:
                # For motorcycle 2-row plates, we ALWAYS prefer split mode (as splitting is much more stable)
                if aspect < 1.6:
                    logger.info(f"Hybrid OCR: Choosing split '{plate_text_2r}' over holistic '{plate_text_1r}' for motorcycle 2-row plate.")
                    return layout_hint_2r, plate_text_2r, ocr_conf_2r, proposals_2r

                if ocr_conf_1r > ocr_conf_2r + 0.05:
                    logger.info(f"Hybrid OCR: Choosing holistic '{plate_text_1r}' (conf={ocr_conf_1r:.3f}) over split '{plate_text_2r}' (conf={ocr_conf_2r:.3f}) because of higher confidence.")
                    return layout_hint_1r_for_format, plate_text_1r, ocr_conf_1r, proposals_1r
                elif ocr_conf_2r > ocr_conf_1r + 0.05:
                    logger.info(f"Hybrid OCR: Choosing split '{plate_text_2r}' (conf={ocr_conf_2r:.3f}) over holistic '{plate_text_1r}' (conf={ocr_conf_1r:.3f}) because of higher confidence.")
                    return layout_hint_2r, plate_text_2r, ocr_conf_2r, proposals_2r

                # If holistic is longer (read more completely, e.g. 5-digit bottom row vs 4-digit), choose holistic
                if len(plate_text_1r) > len(plate_text_2r):
                    logger.info(f"Hybrid OCR: Choosing holistic '{plate_text_1r}' over split '{plate_text_2r}' because holistic is longer.")
                    return layout_hint_1r_for_format, plate_text_1r, ocr_conf_1r, proposals_1r

                # Default fallback is split for 2-row plates
                logger.info(f"Hybrid OCR: Choosing split '{plate_text_2r}' over holistic '{plate_text_1r}' for 2-row plate.")
                return layout_hint_2r, plate_text_2r, ocr_conf_2r, proposals_2r
                    
            # Default fallback to split 2-row
            return layout_hint_2r, plate_text_2r, ocr_conf_2r, proposals_2r

        return self._ocr_one_row_auto(crop, speed_mode=speed_mode)

    def _ocr_one_row_auto(
        self,
        crop: np.ndarray,
        speed_mode: str = "balanced",
    ) -> Tuple[str, str, float, List[np.ndarray]]:
        """
        Biển ô tô / biển ngang:
          30A-583.73
          51A-123.45
        """
        layout_hint = "one_row_auto"

        proposals = self.preprocessor.get_plate_region_proposals(crop, speed_mode=speed_mode)

        try:
            plate_text, ocr_conf = self.ocr.read_plate_ensemble(
                proposals,
                layout_hint=layout_hint,
            )
        except TypeError:
            # fallback nếu bạn chưa thay OCR mới
            plate_text, ocr_conf = self.ocr.read_plate_ensemble(proposals)

        logger.info(
            f"OCR one_row_auto: raw={plate_text}, conf={ocr_conf:.3f}"
        )

        return layout_hint, plate_text, ocr_conf, proposals

    def _ocr_two_row(
        self,
        crop: np.ndarray,
        speed_mode: str = "balanced",
    ) -> Tuple[str, str, float, List[np.ndarray]]:
        """
        Biển 2 dòng có các loại:
          - Xe máy: 59-V1 074.73 / 59-AA 123.45 (50cc)
          - Ô tô vuông: 51F-869.47 / 51LD-123.45
          - Quân sự: AA-12-34 / TM-123.45
        """
        row1_proposals, row2_proposals = self.preprocessor.get_two_row_proposals(crop, speed_mode=speed_mode)
        proposals = row1_proposals + row2_proposals

        layout_hint = self._classify_two_row_layout(row1_proposals, row2_proposals, crop)

        if layout_hint in {"two_row_auto", "military"}:
            # Biển ô tô vuông hoặc quân sự 2 dòng
            if hasattr(self.ocr, "read_square_auto_plate"):
                plate_text, ocr_conf = self.ocr.read_square_auto_plate(
                    row1_proposals,
                    row2_proposals,
                )
            else:
                # fallback nếu OCR chưa có hàm mới
                plate_text, ocr_conf = self._fallback_read_two_row_as_auto(
                    row1_proposals,
                    row2_proposals,
                )
            if layout_hint == "military":
                layout_hint = "military"

        else:
            # Mặc định: biển xe máy 2 dòng
            layout_hint = "two_row_motorbike"
            plate_text, ocr_conf = self.ocr.read_two_row_plate(
                row1_proposals,
                row2_proposals,
            )

        logger.info(
            f"OCR two_row: layout={layout_hint}, raw={plate_text}, conf={ocr_conf:.3f}"
        )

        return layout_hint, plate_text, ocr_conf, proposals

    def _classify_two_row_layout(
        self,
        row1_proposals: List[np.ndarray],
        row2_proposals: List[np.ndarray],
        crop: Optional[np.ndarray] = None,
    ) -> str:
        """
        Phân biệt:
          - two_row_motorbike: top row DDLD, ví dụ 59V1, 52U7 hoặc xe máy 50cc (DDLL)
          - two_row_auto:      top row DDL (ví dụ 51F) hoặc ô tô đặc biệt (DDLL)
          - military:          biển đỏ quân đội dòng trên 2 chữ cái (LL)
        """
        # Quick read top proposals to check for automobile-only prefixes
        for img in (row1_proposals or [])[:2]:
            try:
                t, _ = self.ocr.read_plate(img)
                clean_t = self._clean_text(t)
                if len(clean_t) >= 2 and clean_t[:2] in {"30", "31", "32", "33", "40", "51"}:
                    return "two_row_auto"
            except Exception:
                pass

        # ── Nhận diện bằng Tỉ lệ khung hình (Aspect Ratio) ──
        # Biển xe máy tiêu chuẩn: 280/200 = 1.4 (CCTV góc cao thường nén về 1.1 - 1.35)
        # Biển ô tô vuông tiêu chuẩn: 330/165 = 2.0 (CCTV góc cao thường nén về 1.5 - 1.8)
        # Do đó, nếu aspect ratio < 1.30, 100% là biển xe máy!
        if crop is not None:
            h, w = crop.shape[:2]
            aspect_ratio = w / max(1, h)
            if aspect_ratio < 1.40:
                # Thử kiểm tra xem có phải biển quân sự 2 dòng (bắt đầu bằng 2 chữ cái) không
                # Lấy nhanh text từ proposals
                for proposals in [row1_proposals, row2_proposals]:
                    for img in proposals[:1]:
                        try:
                            t, _ = self.ocr.read_plate(img)
                            clean = self._clean_text(t)
                            if len(clean) == 2 and clean.isalpha():
                                return "military"
                        except Exception:
                            pass
                return "two_row_motorbike"

        top_motorbike = ""
        top_auto = ""
        top_raw = ""

        # Lấy raw OCR của dòng trên để biết nó dài khoảng 3 hay 4 ký tự.
        raw_candidates = []
        for img in row1_proposals or []:
            try:
                raw_text, raw_conf = self.ocr.read_plate(img)
                clean_raw = self._clean_text(raw_text)
                if clean_raw:
                    raw_candidates.append((float(raw_conf), clean_raw))
            except Exception:
                pass

        if raw_candidates:
            raw_candidates.sort(key=lambda x: x[0], reverse=True)
            top_raw = raw_candidates[0][1]

        # Nếu OCR mới có _read_row_ensemble thì dùng để classify tốt hơn.
        if hasattr(self.ocr, "_read_row_ensemble"):
            try:
                top_motorbike, _ = self.ocr._read_row_ensemble(
                    row1_proposals,
                    row_kind="motorbike_top",
                )
            except Exception:
                top_motorbike = ""

            try:
                top_auto, _ = self.ocr._read_row_ensemble(
                    row1_proposals,
                    row_kind="auto_top",
                )
            except Exception:
                top_auto = ""

        top_motorbike = self._clean_text(top_motorbike)
        top_auto = self._clean_text(top_auto)
        top_raw = self._clean_text(top_raw)

        # ── ĐÈ ĐỊNH DẠNG: Sử dụng mã tỉnh ô tô đặc trưng ──
        for text in [top_raw, top_motorbike, top_auto]:
            if len(text) >= 2 and text[:2] in {"30", "31", "32", "33", "40", "51"}:
                return "two_row_auto"

        # ── Nhận diện biển quân sự (dòng trên chỉ có 2 chữ cái như AA, TM) ──
        for text in [top_raw, top_motorbike, top_auto]:
            if len(text) == 2 and text.isalpha():
                return "military"

        # ── Nhận diện biển có 2 chữ cái series dạng DDLL (ví dụ: 51LD, 59AA) ──
        for text in [top_motorbike, top_auto, top_raw]:
            if len(text) == 4 and re.fullmatch(r"\d{2}[A-Z]{2}", text):
                series = text[2:4]
                if series in {"LD", "DA", "MK", "MD", "KT", "NG", "NN", "QT"}:
                    return "two_row_auto"
                else:
                    return "two_row_motorbike"

        is_motorbike_top = bool(re.fullmatch(r"\d{2}[A-Z]\d", top_motorbike))
        is_auto_top = bool(re.fullmatch(r"\d{2}[A-Z]", top_auto))

        logger.debug(
            f"Classify two-row: top_raw={top_raw}, "
            f"top_motorbike={top_motorbike}, top_auto={top_auto}, "
            f"is_motorbike_top={is_motorbike_top}, is_auto_top={is_auto_top}"
        )

        # Nếu raw dòng trên chỉ khoảng 3 ký tự và auto_top hợp lệ -> ô tô vuông.
        if is_auto_top and not is_motorbike_top:
            if top_auto[:2] not in {"52", "59"} and len(top_raw) <= 3:
                return "two_row_auto"

        if is_auto_top and len(top_raw) <= 3:
            if top_auto[:2] not in {"52", "59"}:
                return "two_row_auto"

        # Nếu top row có đủ DDLD -> xe máy.
        if is_motorbike_top:
            return "two_row_motorbike"

        # Fallback bằng độ dài raw.
        if is_auto_top and len(top_raw) == 3:
            if top_auto[:2] in {"29", "52", "59"}:
                return "two_row_motorbike"
            return "two_row_auto"

        return "two_row_motorbike"

    def _fallback_read_two_row_as_auto(
        self,
        row1_proposals: List[np.ndarray],
        row2_proposals: List[np.ndarray],
    ) -> Tuple[str, float]:
        """
        Fallback nếu OCREngine chưa có read_square_auto_plate.
        Output cố gắng dạng:
          51F + 86947 = 51F86947
        """
        text1, conf1 = self._read_best_raw(row1_proposals)
        text2, conf2 = self._read_best_raw(row2_proposals)

        top = self._normalize_auto_top(text1)
        bottom = self._digits_only(text2)

        if len(bottom) > 5:
            bottom = bottom[-5:]

        combined = top + bottom
        avg_conf = (conf1 + conf2) / 2.0 if (conf1 + conf2) > 0 else 0.0

        return combined, avg_conf

    # ======================================================================
    # FORMATTER
    # ======================================================================

    def _format_plate(
        self,
        plate_text: str,
        layout_hint: str,
    ) -> Tuple[str, bool, dict]:
        """
        Gọi formatter với layout_hint.
        Có fallback để không crash nếu bạn chưa thay formatter mới.
        """
        try:
            return self.formatter.format(
                plate_text,
                layout_hint=layout_hint,
            )
        except TypeError:
            logger.warning(
                "PlateFormatter.format() chưa hỗ trợ layout_hint. "
                "Nên thay formatter bản mới."
            )
            return self.formatter.format(plate_text)

    # ======================================================================
    # SMALL HELPERS
    # ======================================================================

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = str(text).upper()
        return re.sub(r"[^A-Z0-9]", "", text)

    def _digits_only(self, text: str) -> str:
        clean = self._clean_text(text)

        digit_fix = {
            "O": "0", "Q": "0", "D": "0",
            "I": "1", "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "B": "8",
            "T": "7",
        }

        result = []
        for ch in clean:
            ch = digit_fix.get(ch, ch)
            if ch.isdigit():
                result.append(ch)

        return "".join(result)

    def _normalize_auto_top(self, text: str) -> str:
        """
        Dòng trên ô tô vuông:
          51F
        """
        clean = self._clean_text(text)

        digit_fix = {
            "O": "0", "Q": "0", "D": "0",
            "I": "1", "L": "1",
            "Z": "2",
            "S": "5",
            "G": "6",
            "B": "8",
        }

        letter_fix = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "4": "A",
            "5": "S",
            "6": "G",
            "7": "T",
            "8": "B",
        }

        candidates = []

        for i in range(0, max(1, len(clean) - 2)):
            part = clean[i:i + 3]

            if len(part) != 3:
                continue

            c0 = digit_fix.get(part[0], part[0])
            c1 = digit_fix.get(part[1], part[1])
            c2 = letter_fix.get(part[2], part[2])

            fixed = c0 + c1 + c2

            if re.fullmatch(r"\d{2}[A-Z]", fixed):
                score = 0

                if i == 0:
                    score += 5

                changed = sum(1 for a, b in zip(part, fixed) if a != b)
                score -= changed * 2

                candidates.append((score, fixed))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return clean[:3]

    def _read_best_raw(
        self,
        images: List[np.ndarray],
    ) -> Tuple[str, float]:
        best_text = ""
        best_conf = 0.0

        for img in images or []:
            text, conf = self.ocr.read_plate(img)

            if conf > best_conf and text:
                best_text = text
                best_conf = float(conf)

        return best_text, best_conf

    # ======================================================================
    # FILE / BYTES ENTRYPOINTS
    # ======================================================================

    def process_file(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image file.
        """
        img = cv2.imread(image_path)

        if img is None:
            return {
                "status": "error",
                "error": f"Cannot read image: {image_path}",
                "detections": [],
            }

        if self.save_results:
            dest = str(OUTPUTS_DIR / "uploaded" / Path(image_path).name)
            cv2.imwrite(dest, img)

        return self.process(img, source_file=image_path)

    def process_bytes(
        self,
        image_bytes: bytes,
        filename: str = "upload",
    ) -> Dict[str, Any]:
        """
        Process image from bytes.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {
                "status": "error",
                "error": "Cannot decode image",
                "detections": [],
            }

        return self.process(img, source_file=filename)