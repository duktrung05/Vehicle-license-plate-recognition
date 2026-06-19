"""
Video Processing Pipeline - Enhanced Version
- Frame diff skip: bỏ qua frame tĩnh để tăng tốc
- Per-plate tracking: gộp nhiều lần nhận diện cùng biển, giữ confidence cao nhất
- Annotate thống nhất với image pipeline
- Progress callback chi tiết
"""
 
import cv2
import numpy as np
import time
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import logging
 
logger = logging.getLogger(__name__)
 
import sys
sys.path.append(str(Path(__file__).parents[0]))
 
BASE_DIR = Path(__file__).parents[3]
OUTPUTS_DIR = BASE_DIR / "outputs"


class PlateTracker:
    def __init__(self, iou_threshold: float = 0.25, max_dist_pct: float = 0.15):
        self.tracks = []  # list of dict
        self.next_track_id = 1
        self.iou_threshold = iou_threshold
        self.max_dist_pct = max_dist_pct

    def _get_iou(self, boxA, boxB):
        if not boxA or not boxB:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def update(self, detections: List[Dict], frame_width: int, frame_height: int, frame_idx: int):
        # Sort current detections by vertical center to match the badge order (badge #1, #2...)
        sorted_dets = sorted(
            detections,
            key=lambda d: (d["bbox"][1] + d["bbox"][3]) / 2 if d.get("bbox") is not None else 0
        )

        matched_dets = [False] * len(sorted_dets)
        matched_tracks = [False] * len(self.tracks)

        # 1. First pass: Match by IoU
        for i, det in enumerate(sorted_dets):
            bbox = det.get("bbox")
            if not bbox:
                continue
            
            best_iou = -1
            best_track_idx = -1
            
            for j, track in enumerate(self.tracks):
                if matched_tracks[j]:
                    continue
                # Calculate IoU with track's last bbox
                iou = self._get_iou(bbox, track["last_bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = j
            
            if best_iou >= self.iou_threshold and best_track_idx != -1:
                self.tracks[best_track_idx]["detections"].append(det)
                self.tracks[best_track_idx]["last_bbox"] = bbox
                self.tracks[best_track_idx]["last_seen"] = frame_idx
                matched_dets[i] = True
                matched_tracks[best_track_idx] = True

        # 2. Second pass: Match unmatched detections by centroid distance
        diag = np.sqrt(frame_width**2 + frame_height**2)
        max_dist = self.max_dist_pct * diag

        for i, det in enumerate(sorted_dets):
            if matched_dets[i]:
                continue
            bbox = det.get("bbox")
            if not bbox:
                continue
            
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            
            best_dist = float("inf")
            best_track_idx = -1
            
            for j, track in enumerate(self.tracks):
                if matched_tracks[j]:
                    continue
                tb = track["last_bbox"]
                tcx = (tb[0] + tb[2]) / 2.0
                tcy = (tb[1] + tb[3]) / 2.0
                dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_track_idx = j
            
            if best_dist <= max_dist and best_track_idx != -1:
                self.tracks[best_track_idx]["detections"].append(det)
                self.tracks[best_track_idx]["last_bbox"] = bbox
                self.tracks[best_track_idx]["last_seen"] = frame_idx
                matched_dets[i] = True
                matched_tracks[best_track_idx] = True

        # 3. Create new tracks for unmatched detections
        for i, det in enumerate(sorted_dets):
            if matched_dets[i]:
                continue
            bbox = det.get("bbox")
            if not bbox:
                continue
            
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks.append({
                "track_id": track_id,
                "last_bbox": bbox,
                "first_seen": frame_idx,
                "last_seen": frame_idx,
                "detections": [det]
            })

    def get_consensus_detections(self, image_pipeline_formatter, fps: float) -> List[Dict]:
        consensus_list = []
        for track in self.tracks:
            dets = track["detections"]
            # Filter detections that have OCR texts
            valid_dets = [d for d in dets if (d.get("formatted_text") or d.get("raw_text"))]
            if not valid_dets:
                continue
            
            # Determine most frequent clean text length
            clean_texts = []
            for d in valid_dets:
                text = d.get("formatted_text") or d.get("raw_text") or ""
                clean = re.sub(r"[^A-Z0-9]", "", text.upper())
                if clean:
                    clean_texts.append(clean)
            
            if not clean_texts:
                continue
                
            len_counts = {}
            for ct in clean_texts:
                l = len(ct)
                len_counts[l] = len_counts.get(l, 0) + 1
            best_len = max(len_counts, key=len_counts.get)
            
            # Filter to detections matching the best length
            matching_dets = []
            for d in valid_dets:
                text = d.get("formatted_text") or d.get("raw_text") or ""
                clean = re.sub(r"[^A-Z0-9]", "", text.upper())
                if len(clean) == best_len:
                    matching_dets.append((d, clean))
            
            # Character-level voting weighted by OCR confidence
            char_votes = {} # pos -> {char -> total_confidence}
            for d, clean in matching_dets:
                conf = d.get("ocr_confidence", 0.5)
                
                # Get raw alphanumeric characters for comparison to check if formatter corrected it
                raw_text = d.get("raw_text") or ""
                clean_raw = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                
                for pos, ch in enumerate(clean):
                    if pos not in char_votes:
                        char_votes[pos] = {}
                        
                    # Weight penalty: if character was changed/forced by formatter,
                    # reduce its voting weight significantly (to 15%) because 
                    # the OCR engine didn't actually read this character directly.
                    weight = conf
                    if pos < len(clean_raw) and clean_raw[pos] != ch:
                        weight = conf * 0.15
                        
                    char_votes[pos][ch] = char_votes[pos].get(ch, 0.0) + weight
            
            # Build consensus clean text
            consensus_chars = []
            for pos in range(best_len):
                votes = char_votes.get(pos, {})
                if votes:
                    best_ch = max(votes, key=votes.get)
                    consensus_chars.append(best_ch)
                else:
                    consensus_chars.append("?")
            consensus_clean = "".join(consensus_chars)
            
            # Select representative detection (highest confidence)
            best_det, _ = max(matching_dets, key=lambda x: x[0].get("ocr_confidence", 0.0))
            
            # Re-format consensus text using formatter
            formatted_text = consensus_clean
            is_valid = False
            province = "Unknown"
            province_code = None
            plate_type = "Unknown"
            
            layout_hint = best_det.get("layout_hint", "unknown")
            try:
                formatted_text, is_valid, metadata = image_pipeline_formatter.format(consensus_clean, layout_hint=layout_hint)
                province = metadata.get("province_name", "Unknown")
                province_code = metadata.get("province_code")
                plate_type = metadata.get("plate_type")
            except Exception:
                # Fallback to formatting of the best representative detection
                formatted_text = best_det.get("formatted_text", consensus_clean)
                is_valid = best_det.get("is_valid", False)
                province = best_det.get("province", "Unknown")
                province_code = best_det.get("province_code")
                plate_type = best_det.get("plate_type")
            
            # Use format fallback if consensus formatting didn't produce a valid string
            if not formatted_text:
                formatted_text = best_det.get("formatted_text", consensus_clean)
                is_valid = best_det.get("is_valid", False)
                province = best_det.get("province", "Unknown")
                province_code = best_det.get("province_code")
                plate_type = best_det.get("plate_type")

            # Determine most frequent raw text
            raw_text_counts = {}
            for d in valid_dets:
                rt = d.get("raw_text", "")
                if rt:
                    raw_text_counts[rt] = raw_text_counts.get(rt, 0) + 1
            most_frequent_raw = max(raw_text_counts, key=raw_text_counts.get) if raw_text_counts else best_det.get("raw_text", "")
            
            # Get max confidence seen
            max_ocr_conf = max(d.get("ocr_confidence", 0.0) for d in dets)
            
            consensus_list.append({
                "id": best_det.get("id"),
                "bbox": best_det.get("bbox"),
                "detection_confidence": best_det.get("detection_confidence", 0.8),
                "ocr_confidence": max_ocr_conf,
                "raw_text": most_frequent_raw,
                "formatted_text": formatted_text,
                "is_valid": is_valid,
                "province": province,
                "province_code": province_code,
                "plate_type": plate_type,
                "layout_hint": layout_hint,
                "crop_path": best_det.get("crop_path"),
                "processed_path": best_det.get("processed_path"),
                "frame_number": best_det.get("frame_number"),
                "timestamp_sec": round(track["first_seen"] / fps, 2),
                "duration_frames": len(dets),
                "first_seen_frame": track["first_seen"],
                "last_seen_frame": track["last_seen"],
            })
            
        return consensus_list


class VideoPipeline:
    """
    Processes video files for license plate recognition.
    Improvements over v1:
    - Frame diff: skip frames that are nearly identical to previous (static scenes)
    - Smart dedup: merge multiple detections of same plate across frames
    - Annotate video with per-frame HUD showing all plates found so far
    - Detailed progress with ETA
    """
 
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        frame_skip: int = 5,           # process every 5th frame for good coverage
        use_gpu: bool = False,
        diff_threshold: float = 1.5,   # mean pixel diff để skip frame tĩnh
        dedup_threshold: float = 0.70, # similarity threshold for plate deduplication
        max_plates: Optional[int] = None, # default max plates (None = scan entire video)
        fast_mode: bool = True,
        speed_mode: Optional[str] = None,
    ):
        self.frame_skip = frame_skip
        self.diff_threshold = diff_threshold
        self.dedup_threshold = dedup_threshold
        self.max_plates = max_plates
 
        if speed_mode is None:
            speed_mode = "fast" if fast_mode else "balanced"
 
        # Import ở đây để tránh circular import
        from pipeline.image_pipeline import ImagePipeline
        self.image_pipeline = ImagePipeline(
            model_path=model_path,
            conf_threshold=conf_threshold,
            use_gpu=use_gpu,
            save_results=False,
            speed_mode=speed_mode,
        )
 
        (OUTPUTS_DIR / "result_videos").mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / "uploaded").mkdir(parents=True, exist_ok=True)
 
    def _resize_for_ocr(self, frame: np.ndarray, max_width: int = 1920) -> np.ndarray:
        """Thu nhỏ frame để OCR nhanh hơn, dùng INTER_AREA để giữ nét chữ"""
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        # INTER_AREA produces sharper text edges when downscaling
        # (INTER_LINEAR blurs thin strokes like '1' making OCR read '7')
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
 
    def _edit_distance(self, s1: str, s2: str) -> float:
        """Levenshtein distance normalized về 0-1"""
        if not s1 or not s2:
            return 0.0
        m, n = len(s1), len(s2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1): dp[i][0] = i
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = dp[i-1][j-1] if s1[i-1]==s2[j-1] \
                           else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        similarity = 1 - dp[m][n] / max(m, n)
        return similarity
 
    def _reformat_plate(self, reference_formatted: str, new_clean: str) -> str:
        """
        Apply the same formatting pattern (dashes, dots, spaces) from reference_formatted
        to the new_clean characters. This preserves the plate format while updating characters.
        """
        import re
        # Extract formatting characters and their positions from reference
        alphanums_in_ref = re.sub(r"[^A-Z0-9]", "", reference_formatted.upper())
        
        if len(alphanums_in_ref) != len(new_clean):
            # If lengths differ, use the image pipeline's formatter
            try:
                formatted, _, _ = self.image_pipeline.formatter.format(new_clean)
                if formatted:
                    return formatted
            except Exception:
                pass
            return new_clean
        
        # Map each alphanumeric position in the reference to the new character
        result = []
        clean_idx = 0
        for ch in reference_formatted:
            upper_ch = ch.upper()
            if re.match(r"[A-Z0-9]", upper_ch):
                if clean_idx < len(new_clean):
                    result.append(new_clean[clean_idx])
                    clean_idx += 1
                else:
                    result.append(ch)
            else:
                result.append(ch)  # Keep formatting chars (-, ., space)
        
        return "".join(result)
 
    # ─────────────────────────────────────────────────────────────────
    #  PUBLIC: process_video
    # ─────────────────────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file and return aggregated results.
 
        progress_callback(current_frame, total_frames, stats_dict)
        stats_dict keys: plates_found, fps_live, eta_sec
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "error", "error": f"Cannot open video: {video_path}"}
 
        fps       = cap.get(cv2.CAP_PROP_FPS) or 25
        total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = min(total_raw, max_frames) if max_frames else total_raw
 
        # Calculate video duration and dynamically adjust skip parameters
        video_duration = total_frames / fps
        if video_duration <= 5.0:
            effective_skip = max(2, self.frame_skip // 3)
            effective_diff = self.diff_threshold * 0.5
        else:
            effective_skip = self.frame_skip
            effective_diff = self.diff_threshold
 
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
        tracker = PlateTracker(iou_threshold=0.25, max_dist_pct=0.15)
        frame_idx       = 0
        processed_cnt   = 0
        skipped_diff    = 0
        prev_gray       = None
        start_time      = time.time()
        last_detections = []                # detections từ frame được xử lý gần nhất
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break
 
            should_process = (frame_idx % effective_skip == 0)
 
            # Frame diff: nếu ảnh hầu như không thay đổi → skip
            if should_process and prev_gray is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                if diff.mean() < effective_diff:
                    should_process = False
                    skipped_diff += 1
 
            if should_process:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = gray
 
                frame_resized = self._resize_for_ocr(frame)
                result = self.image_pipeline.process(frame_resized, source_file=video_path, log_to_db=False)
                processed_cnt += 1
                last_detections = result.get("detections", [])
 
                # Assign frame_number and timestamp_sec to each detection
                for det in last_detections:
                    det["frame_number"] = frame_idx
                    det["timestamp_sec"] = round(frame_idx / fps, 2)
 
                # Update tracker with detections from current frame
                tracker.update(last_detections, width, height, frame_idx)
 
                # Early stopping if enough plates/tracks are found
                active_tracks_count = len(tracker.tracks)
                if (self.max_plates is not None 
                    and active_tracks_count >= self.max_plates 
                    and processed_cnt >= max(10, active_tracks_count * 5)):
                        out_frame = result.get("annotated_image")
                        if out_frame is None:
                            out_frame = frame_resized
                        if out_frame.shape[1] != width or out_frame.shape[0] != height:
                            out_frame = cv2.resize(out_frame, (width, height))
                        if writer:
                            writer.write(out_frame)
                        frame_idx += 1
                        break
 
                # Dùng annotated_image từ image_pipeline nếu có
                out_frame = result.get("annotated_image")
                if out_frame is None:
                    out_frame = frame_resized
                
                # Resize back to original size to match VideoWriter resolution
                if out_frame.shape[1] != width or out_frame.shape[0] != height:
                    out_frame = cv2.resize(out_frame, (width, height))
            else:
                out_frame = frame
 
            if writer:
                writer.write(out_frame)
 
            frame_idx += 1
 
            # Progress callback
            if progress_callback:
                elapsed = time.time() - start_time
                live_fps = processed_cnt / elapsed if elapsed > 0 else 0
                remaining = total_frames - frame_idx
                eta = (remaining / self.frame_skip) / live_fps if live_fps > 0 else 0
                progress_callback(frame_idx, total_frames, {
                    "plates_found": len(tracker.tracks),
                    "fps_live": round(live_fps, 1),
                    "eta_sec": round(eta),
                })
 
        cap.release()
        if writer:
            writer.release()
 
        elapsed = time.time() - start_time
 
        # Compile consensus detections
        consensus_detections = tracker.get_consensus_detections(self.image_pipeline.formatter, fps)
 
        # Log unique plates to DB at the end of video processing
        for det in consensus_detections:
            self.image_pipeline.db.log_detection(
                plate_text=det.get("formatted_text"),
                confidence=det.get("ocr_confidence", 0.0),
                source_type="video",
                source_file=video_path,
                bbox=det.get("bbox"),
                province_code=det.get("province_code"),
                province_name=det.get("province"),
                plate_type=det.get("plate_type"),
                is_valid=det.get("is_valid", False),
                processing_time_ms=elapsed * 1000 / max(1, processed_cnt),
                crop_path=det.get("crop_path"),
            )
 
        elapsed = time.time() - start_time
 
        return {
            "status": "success",
            "source_file": video_path,
            "output_path": output_path,
            "total_frames": frame_idx,
            "processed_frames": processed_cnt,
            "skipped_static_frames": skipped_diff,
            "unique_plates": len(consensus_detections),
            "detections": consensus_detections,
            "processing_time_sec": round(elapsed, 2),
            "fps_processing": round(processed_cnt / elapsed, 2) if elapsed > 0 else 0,
        }
 
    # ─────────────────────────────────────────────────────────────────
    #  PUBLIC: process_video_bytes
    # ─────────────────────────────────────────────────────────────────
    def process_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "upload.mp4",
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Process video from raw bytes."""
        temp_path = str(OUTPUTS_DIR / "uploaded" / filename)
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
 
        out_path = str(OUTPUTS_DIR / "result_videos" / f"result_{filename}")
        return self.process_video(
            temp_path,
            output_path=out_path,
            progress_callback=progress_callback,
        )
 
    # ─────────────────────────────────────────────────────────────────
    #  PRIVATE: vẽ HUD tóm tắt góc dưới phải
    # ─────────────────────────────────────────────────────────────────
    def _draw_summary_hud(
        self,
        frame: np.ndarray,
        seen_plates: Dict[str, Dict],
        max_show: int = 6,
    ) -> np.ndarray:
        """Vẽ bảng tóm tắt biển số đã nhận diện lên góc dưới phải của frame (đã vô hiệu hóa theo yêu cầu)."""
        return frame
        h, w = frame.shape[:2]
 
        plates = list(seen_plates.values())
        # Sắp xếp theo timestamp để hiển thị biển mới nhất trước
        plates.sort(key=lambda d: d.get("timestamp_sec", 0), reverse=True)
        plates = plates[:max_show]
 
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.45, w / 2000)
        line_h     = int(22 * (w / 1280))
        pad        = 8
        box_w      = int(230 * (w / 1280))
 
        total_h = pad * 2 + 18 + len(plates) * (line_h + 4)
        box_x1  = w - box_w - 10
        box_y1  = h - total_h - 10
        box_x2  = w - 10
        box_y2  = h - 10
 
        # Nền mờ
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
 
        # Viền
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 200, 0), 1)
 
        # Header
        header_text = f"BIEN SO ({len(seen_plates)})"
        cv2.putText(frame, header_text,
                    (box_x1 + pad, box_y1 + pad + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85,
                    (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(frame,
                 (box_x1 + 4, box_y1 + pad + 18),
                 (box_x2 - 4, box_y1 + pad + 18),
                 (255, 200, 0), 1)
 
        # Từng biển
        for i, det in enumerate(plates):
            y = box_y1 + pad + 24 + i * (line_h + 4)
            plate_text = det.get("formatted_text", det.get("raw_text", "???"))
            conf       = det.get("ocr_confidence", 0)
            is_valid   = det.get("is_valid", False)
            ts         = det.get("timestamp_sec", 0)
 
            color = (80, 255, 120) if is_valid else (120, 120, 255)
 
            cv2.putText(frame, plate_text,
                        (box_x1 + pad, y + line_h - 4),
                        font, font_scale,
                        color, 1, cv2.LINE_AA)
 
            ts_str = f"{ts:.1f}s"
            cv2.putText(frame, ts_str,
                        (box_x2 - 38, y + line_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7,
                        (150, 150, 150), 1, cv2.LINE_AA)
 
        return frame