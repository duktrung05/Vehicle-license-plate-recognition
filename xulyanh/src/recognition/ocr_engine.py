import re
import numpy as np
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("easyocr not installed.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed.")


# Ý nghĩa: Tập hợp các mã tỉnh/thành phố hợp lệ của Việt Nam. 
# Được sử dụng để xác thực (validate) 2 ký tự đầu tiên của biển số, và cộng điểm ưu tiên nếu nhận diện đúng.
VALID_PROVINCE_CODES = {
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21",
    "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "43",
    "47", "48", "49", "50", "51", "52", "53", "54", "55", "56",
    "57", "58", "59", "60", "61", "62", "63", "64", "65", "66",
    "67", "68", "69", "70", "71", "72", "73", "74", "75", "76",
    "77", "78", "79", "80", "81", "82", "83", "84", "85", "86",
    "88", "89", "90", "92", "93", "94", "95", "97", "98", "99",
}


class OCREngine:
    DIGIT_FIX = {
        "O": "0", "Q": "0",
        "I": "1", "|": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
        "T": "1",
    }

    LETTER_FIX = {
        "0": "O",
        "1": "I",
    }

    DIGIT_FIX_FORCE = {
        "O": "0", "Q": "0", "D": "0", "U": "0",
        "I": "1", "L": "1", "|": "1", "J": "1", "T": "1",
        "Z": "2",
        "S": "5",
        "G": "9",
        "Y": "7",
        "B": "8",
        "A": "4", "H": "4",
    }


    LETTER_FIX_FORCE = {
        "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
        "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
    }

    def __init__(self, languages: List[str] = ["en"], use_gpu: bool = False):
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None
        self._init_easyocr()

    def _init_easyocr(self):
        if EASYOCR_AVAILABLE:
            try:
                self.reader = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    verbose=False,
                )
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"EasyOCR init failed: {e}")
                self.reader = None

    def read_plate(self, image: np.ndarray) -> Tuple[str, float]:
        # Ý nghĩa: Hàm cổng kết nối (entry point) để đọc biển số từ một ảnh đầu vào.
        # Fallback logic: Ưu tiên dùng EasyOCR -> Nếu không có sẽ dùng Tesseract -> Trả về kết quả giả (mock) nếu không cài cả hai.
        if self.reader is not None:
            return self._read_easyocr(image)

        if TESSERACT_AVAILABLE:
            return self._read_tesseract(image)

        logger.warning("No OCR engine available. Using mock result.")
        return self._mock_read(image)

    def _read_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            results = self.reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.",
            )

            if not results:
                return "", 0.0

            items = []

            for bbox, text, conf in results:
                text = str(text).upper().replace(" ", "")
                text = re.sub(r"[^A-Z0-9\-.]", "", text)

                if not text:
                    continue

                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]

                items.append({
                    "text": text,
                    "conf": float(conf),
                    "x": sum(xs) / len(xs),
                    "y": sum(ys) / len(ys),
                    "h": max(ys) - min(ys),
                })

            if not items:
                return "", 0.0

            items = sorted(items, key=lambda i: i["y"])
            lines = []

            # Ý nghĩa: Thuật toán gom nhóm các ký tự (bounding box) đơn lẻ thành từng dòng (lines).
            # Dựa trên khoảng cách trục Y (chiều dọc). Nếu tọa độ Y gần nhau (< 80% chiều cao ký tự), chúng được gom vào cùng một dòng.
            for item in items:
                placed = False

                for line in lines:
                    if abs(item["y"] - line["y"]) < max(10, item["h"] * 0.8):
                        line["items"].append(item)
                        line["y"] = (line["y"] + item["y"]) / 2
                        placed = True
                        break

                if not placed:
                    lines.append({
                        "y": item["y"],
                        "items": [item],
                    })

            lines = sorted(lines, key=lambda l: l["y"])

            line_texts = []
            confidences = []

            for line in lines:
                sorted_items = sorted(line["items"], key=lambda i: i["x"])
                line_texts.append("".join(i["text"] for i in sorted_items))
                confidences.extend(i["conf"] for i in sorted_items)

            combined = "".join(line_texts)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0

            return combined, avg_conf

        except Exception as e:
            logger.error(f"EasyOCR read failed: {e}")
            return "", 0.0

    def _read_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        try:
            config = (
                "--psm 7 --oem 3 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
            )

            text = pytesseract.image_to_string(image, config=config)
            text = text.upper().replace(" ", "").replace("\n", "").strip()

            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            confs = []
            for c in data.get("conf", []):
                try:
                    c = float(c)
                    if c >= 0:
                        confs.append(c)
                except Exception:
                    pass

            avg_conf = float(np.mean(confs)) / 100.0 if confs else 0.5

            return text, avg_conf

        except Exception as e:
            logger.error(f"Tesseract read failed: {e}")
            return "", 0.0

    def _mock_read(self, image: np.ndarray) -> Tuple[str, float]:
        return "51A-12345", 0.75

    # ======================================================================
    # PUBLIC OCR HELPERS
    # ======================================================================

    def read_plate_ensemble(
        self,
        plate_images: List[np.ndarray],
        layout_hint: Optional[str] = None,
    ) -> Tuple[str, float]:
        # Ý nghĩa: Cơ chế đọc biển số sử dụng nhiều ảnh biến thể (proposals - như ảnh binarized, grayscale, inverted...).
        # Sau đó chạy OCR trên tất cả, chuẩn hóa, rồi thực hiện bỏ phiếu ở cấp độ từng ký tự (Character-level Consensus).
        # Cách tiếp cận này giúp khắc phục triệt để lỗi khi 1-2 ký tự bị sai lệch làm lệch hướng kết quả bầu chọn chuỗi nguyên bản.
        proposal_results = []
        counts = {}

        for idx, img in enumerate(plate_images or []):
            text, conf = self.read_plate(img)
            raw = self._clean_ocr_text(text)

            if not raw:
                continue

            normalized = self._normalize_full_plate(
                raw,
                layout_hint=layout_hint,
            )
            if not normalized:
                continue

            counts[normalized] = counts.get(normalized, 0) + 1
            proposal_results.append((normalized, conf, raw, idx))

        if not proposal_results:
            return "", 0.0

        # 1. Nhóm theo độ dài chuỗi chuẩn hóa phổ biến nhất (sử dụng trọng số để ưu tiên chất lượng)
        len_weights = {}
        for norm, conf, raw, idx in proposal_results:
            l = len(norm)
            len_weights[l] = len_weights.get(l, 0.0) + conf

        best_len = max(len_weights.keys(), key=lambda k: len_weights[k])

        # 2. Khởi tạo mảng bỏ phiếu cho từng vị trí ký tự
        votes = [{} for _ in range(best_len)]
        confs_sum = [{} for _ in range(best_len)]

        for norm, conf, raw, idx in proposal_results:
            if len(norm) != best_len:
                continue

            for pos, ch in enumerate(norm):
                votes[pos][ch] = votes[pos].get(ch, 0.0) + conf
                confs_sum[pos][ch] = confs_sum[pos].get(ch, 0.0) + conf

        # 3. Tổng hợp ký tự chiến thắng tại mỗi vị trí
        consensus_chars = []
        avg_confs = []

        for pos in range(best_len):
            if not votes[pos]:
                consensus_chars.append("0")
                avg_confs.append(0.5)
                continue

            best_char = max(votes[pos].keys(), key=lambda c: votes[pos][c])
            consensus_chars.append(best_char)

            # Tính độ tin cậy trung bình của ký tự chiến thắng
            winning_count = sum(
                1 for norm, _, _, _ in proposal_results
                if len(norm) == best_len and norm[pos] == best_char
            )
            avg_conf = confs_sum[pos][best_char] / max(1, winning_count)
            avg_confs.append(avg_conf)

        consensus_text = "".join(consensus_chars)
        consensus_conf = float(np.mean(avg_confs)) if avg_confs else 0.0

        logger.debug(
            f"OCR ensemble character-consensus: {consensus_text} (conf={consensus_conf:.4f})"
        )

        return consensus_text, consensus_conf

    def read_two_row_plate(
        self,
        row1_proposals: List[np.ndarray],
        row2_proposals: List[np.ndarray],
    ) -> Tuple[str, float]:
        # Ý nghĩa: Đọc biển số xe máy (2 dòng) bằng cách chia làm ảnh dòng trên (motorbike_top) và dòng dưới (bottom).
        # Sử dụng kỹ thuật OCR ensemble riêng biệt cho từng dòng rồi ghép chuỗi lại.
        text1, conf1 = self._read_row_ensemble(
            row1_proposals,
            row_kind="motorbike_top",
        )

        text2, conf2 = self._read_row_ensemble(
            row2_proposals,
            row_kind="bottom",
        )

        combined = text1 + text2
        avg_conf = (conf1 + conf2) / 2.0 if (conf1 + conf2) > 0 else 0.0

        logger.debug(
            f"Two-row motorbike OCR: row1={text1}, row2={text2}, "
            f"combined={combined}, conf={avg_conf:.3f}"
        )

        return combined, avg_conf

    def read_square_auto_plate(
        self,
        row1_proposals: List[np.ndarray],
        row2_proposals: List[np.ndarray],
        layout_hint: Optional[str] = None,
    ) -> Tuple[str, float]:
        row_kind = "military_top" if layout_hint == "military" else "auto_top"
        text1, conf1 = self._read_row_ensemble(
            row1_proposals,
            row_kind=row_kind,
        )

        text2, conf2 = self._read_row_ensemble(
            row2_proposals,
            row_kind="bottom",
        )

        combined = text1 + text2
        avg_conf = (conf1 + conf2) / 2.0 if (conf1 + conf2) > 0 else 0.0

        logger.debug(
            f"Square auto OCR: row1={text1}, row2={text2}, "
            f"combined={combined}, conf={avg_conf:.3f}"
        )

        return combined, avg_conf

    # ======================================================================
    # ROW OCR
    # ======================================================================

    def _read_row_ensemble(
        self,
        row_images: List[np.ndarray],
        row_kind: str,
    ) -> Tuple[str, float]:
        proposal_results = []
        counts = {} 

        for idx, img in enumerate(row_images or []):
            text, conf = self.read_plate(img)
            raw = self._clean_ocr_text(text)

            if not raw:
                continue

            if row_kind == "motorbike_top":
                normalized = self._normalize_motorbike_top_row(raw)
            elif row_kind == "auto_top":
                normalized = self._normalize_auto_top_row(raw)
            elif row_kind == "military_top":
                normalized = self._normalize_military_top_row(raw)
            elif row_kind == "bottom":
                normalized = self._normalize_bottom_row(text)
            else:
                normalized = raw

            if not normalized:
                continue

            counts[normalized] = counts.get(normalized, 0) + 1
            proposal_results.append((normalized, conf, raw, idx))

        if not proposal_results:
            return "", 0.0

        # 1. Nhóm theo độ dài chuỗi chuẩn hóa phổ biến nhất (sử dụng trọng số để ưu tiên chất lượng)
        len_weights = {}
        for norm, conf, raw, idx in proposal_results:
            l = len(norm)
            len_weights[l] = len_weights.get(l, 0.0) + conf

        best_len = max(len_weights.keys(), key=lambda k: len_weights[k])

        # 2. Khởi tạo mảng bỏ phiếu cho từng vị trí ký tự
        votes = [{} for _ in range(best_len)]
        confs_sum = [{} for _ in range(best_len)]

        for norm, conf, raw, idx in proposal_results:
            if len(norm) != best_len:
                continue

            for pos, ch in enumerate(norm):
                votes[pos][ch] = votes[pos].get(ch, 0.0) + conf
                confs_sum[pos][ch] = confs_sum[pos].get(ch, 0.0) + conf

        # 3. Tổng hợp ký tự chiến thắng tại mỗi vị trí
        consensus_chars = []
        avg_confs = []

        for pos in range(best_len):
            if not votes[pos]:
                consensus_chars.append("0")
                avg_confs.append(0.5)
                continue

            best_char = max(votes[pos].keys(), key=lambda c: votes[pos][c])
            consensus_chars.append(best_char)

            # Tính độ tin cậy trung bình của ký tự chiến thắng
            winning_count = sum(
                1 for norm, _, _, _ in proposal_results
                if len(norm) == best_len and norm[pos] == best_char
            )
            avg_conf = confs_sum[pos][best_char] / max(1, winning_count)
            avg_confs.append(avg_conf)

        consensus_text = "".join(consensus_chars)
        consensus_conf = float(np.mean(avg_confs)) if avg_confs else 0.0

        logger.debug(
            f"Row OCR character-consensus [{row_kind}]: {consensus_text} (conf={consensus_conf:.4f})"
        )

        return consensus_text, consensus_conf

    # ======================================================================
    # NORMALIZATION
    # ======================================================================

    def _clean_ocr_text(self, text: str) -> str:
        if not text:
            return ""

        text = str(text).upper()
        text = text.replace(" ", "")
        text = text.replace("\n", "")

        return re.sub(r"[^A-Z0-9]", "", text)

    def _to_digit(self, c: str) -> str:
        c = c.upper()
        return self.DIGIT_FIX.get(c, c)

    def _to_letter(self, c: str) -> str:
        c = c.upper()
        return self.LETTER_FIX.get(c, c)

    def _apply_template(
        self,
        text: str,
        template: str,
        validate_province: bool = True,
    ) -> Optional[str]:
        # Ý nghĩa: Đối chiếu và ép kiểu chuỗi OCR thô theo một khuôn mẫu (template) cụ thể.
        # Template gồm các ký tự 'D' (Digit - số) và 'L' (Letter - chữ). 
        # Ví dụ: template "DDLDDDDD" cho biển "51A12345". 
        # Hàm này bắt buộc các vị trí phải đúng kiểu chữ/số, nếu sai sẽ kết hợp từ điển sửa lỗi (FIX_FORCE) để khôi phục.
        if len(text) != len(template):
            return None

        result = []

        for ch, kind in zip(text, template):
            if kind == "D":
                fixed = self._to_digit(ch)
                if not fixed.isdigit():
                    fixed = self.DIGIT_FIX_FORCE.get(ch, ch)
                if not fixed.isdigit():
                    return None
                result.append(fixed)

            elif kind == "L":
                fixed = self._to_letter(ch)
                if not fixed.isalpha():
                    fixed = self.LETTER_FIX_FORCE.get(ch, ch)
                if not fixed.isalpha():
                    return None
                
                # Correct illegal civilian letters (O, Q, I, W) if it's not a military template
                is_military = template.startswith("LL")
                if not is_military:
                    if fixed == "O": fixed = "U"
                    elif fixed == "Q": fixed = "U"
                    elif fixed == "W": fixed = "V"
                
                result.append(fixed)

            else:
                result.append(ch)

        fixed_text = "".join(result)
        
        # Bỏ qua mã tỉnh dân sự với biển quân sự
        is_military = template.startswith("LL")
        if validate_province and not is_military and fixed_text[:2] not in VALID_PROVINCE_CODES:
            return None

        return fixed_text

    def _normalize_motorbike_top_row(self, text: str) -> str:
        clean = self._clean_ocr_text(text)
        candidates = []

        # Thử cả 2 mẫu DDLD (thường) và DDLL (50cc)
        for template in ["DDLD", "DDLL"]:
            for i in range(0, max(1, len(clean) - 3)):
                part = clean[i:i + 4]

                if len(part) != 4:
                    continue

                fixed = self._apply_template(part, template)

                if not fixed:
                    continue

                # For motorcycle 2-letter series (DDLL), the first letter of the series must be 'A' or 'M'
                if template == "DDLL" and len(fixed) == 4:
                    if fixed[2] not in {"A", "M"}:
                        continue

                score = 0
                if fixed[:2] in VALID_PROVINCE_CODES:
                    score += 30
                if i == 0:
                    score += 5

                changed = sum(1 for a, b in zip(part, fixed) if a != b)
                score -= changed * 2

                candidates.append((score, fixed))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return ""

    def _normalize_auto_top_row(self, text: str) -> str:
        clean = self._clean_ocr_text(text)
        candidates = []

        # Thử mẫu DDLL (4 ký tự) trước
        for i in range(0, max(1, len(clean) - 3)):
            part = clean[i:i + 4]
            if len(part) == 4:
                fixed = self._apply_template(part, "DDLL")
                if fixed:
                    score = 10
                    if fixed[:2] in VALID_PROVINCE_CODES:
                        score += 30
                    if i == 0:
                        score += 5
                    changed = sum(1 for a, b in zip(part, fixed) if a != b)
                    score -= changed * 2
                    candidates.append((score, fixed))

        # Thử mẫu DDL (3 ký tự)
        for i in range(0, max(1, len(clean) - 2)):
            part = clean[i:i + 3]
            if len(part) == 3:
                fixed = self._apply_template(part, "DDL")
                if fixed:
                    score = 0
                    if fixed[:2] in VALID_PROVINCE_CODES:
                        score += 30
                    if i == 0:
                        score += 5
                    changed = sum(1 for a, b in zip(part, fixed) if a != b)
                    score -= changed * 2
                    candidates.append((score, fixed))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return ""

    def _normalize_military_top_row(self, text: str) -> str:
        clean = self._clean_ocr_text(text)
        candidates = []

        # Thử mẫu LL (2 ký tự, quân sự dòng trên)
        for i in range(0, max(1, len(clean) - 1)):
            part = clean[i:i + 2]
            if len(part) == 2:
                fixed = self._apply_template(part, "LL")
                if fixed:
                    score = 20
                    if i == 0:
                        score += 5
                    changed = sum(1 for a, b in zip(part, fixed) if a != b)
                    score -= changed * 2
                    candidates.append((score, fixed))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return ""

    def _normalize_bottom_row(self, text: str) -> str:
        # Keep track of original indices of clean characters in text
        clean_chars = []
        orig_indices = []
        for idx, ch in enumerate(text):
            ch_upper = ch.upper()
            if re.match(r"[A-Z0-9]", ch_upper):
                clean_chars.append(ch_upper)
                orig_indices.append(idx)
        clean = "".join(clean_chars)
        
        # Try matching templates "DDDDD" and "DDDD" using sliding window to avoid naive truncation
        candidates = []
        for template in ["DDDDD", "DDDD"]:
            t_len = len(template)
            c_len = len(clean)

            if c_len >= t_len:
                for shift in range(c_len - t_len + 1):
                    part = clean[shift : shift + t_len]
                    fixed = self._apply_template(part, template, validate_province=False)
                    if not fixed:
                        continue

                    score = 0
                    score -= shift * 5
                    extra_chars = c_len - t_len
                    score -= extra_chars * 15

                    changed = sum(1 for a, b in zip(part, fixed) if a != b)
                    score -= changed * 2

                    if shift == 0:
                        score += 5  # shift-0 bonus reduced to 5 to avoid overpowering other features

                    if fixed and fixed[0] in "456789":
                        score += 5

                    # Dot alignment reward
                    if len(orig_indices) == c_len:
                        if t_len == 5:
                            # Dot should be between 3rd and 4th digits (index 2 and 3 in the window)
                            start_idx = orig_indices[shift + 2]
                            end_idx = orig_indices[shift + 3]
                            if "." in text[start_idx : end_idx]:
                                score += 20  # Dot aligned reward
                        elif t_len == 4:
                            # Dot should be between 2nd and 3rd digits (index 1 and 2 in the window)
                            start_idx = orig_indices[shift + 1]
                            end_idx = orig_indices[shift + 2]
                            if "." in text[start_idx : end_idx]:
                                score += 20  # Dot aligned reward

                    candidates.append((score, fixed))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # fallback: legacy behavior
        digits = []
        for ch in clean:
            fixed = self._to_digit(ch)
            if not fixed.isdigit():
                fixed = self.DIGIT_FIX_FORCE.get(ch, ch)
            if fixed.isdigit():
                digits.append(fixed)

        digits = "".join(digits)

        if len(digits) > 5:
            digits = digits[-5:]

        return digits

    def _normalize_full_plate(
        self,
        text: str,
        layout_hint: Optional[str] = None,
    ) -> str:
        # Ý nghĩa: Chuẩn hóa toàn bộ chuỗi ký tự nhận diện được bằng cách lần lượt thử nghiệm với các templates hợp lệ.
        # Tham số layout_hint (1 dòng, 2 dòng, ô tô, xe máy, quân sự) giúp khoanh vùng danh sách templates phù hợp nhất.
        clean = self._clean_ocr_text(text)
        hint = (layout_hint or "").lower()

        if not clean:
            return ""

        # Phát hiện dạng biển quân sự
        is_military = hint == "military" or (len(clean) >= 2 and clean[:2].isalpha())

        if is_military:
            templates = [
                "LLDDDD",    # AA1234
                "LLDDDDD",   # TM12345
            ]
        elif hint in {"two_row_motorbike", "two_row", "motorbike_2row", "motorbike"}:
            templates = [
                "DDLDDDDDD",   # 59V107473
                "DDLDDDDD",    # 52U78693
                "DDLLDDDDD",   # 59AA12345 (50cc)
                "DDLLDDDD",    # 59AA1234 (50cc)
            ]
        elif hint in {"two_row_auto", "auto_square", "square_auto"}:
            templates = [
                "DDLDDDDD",    # 51F86947
                "DDLDDDD",     # 51F1234
                "DDLLDDDDD",   # 51LD12345
                "DDLLDDDD",    # 51LD1234
            ]
        elif hint in {"one_row_auto", "one_row", "auto", "car", "standard"}:
            templates = [
                "DDLDDDDD",    # 30A58373
                "DDLLDDDDD",   # 30AB12345 / 80NG12345
                "DDLDDDD",     # 51A1234
                "DDLLDDDD",    # 30AB1234
            ]
        else:
            templates = [
                "DDLDDDDDD",
                "DDLDDDDD",
                "DDLLDDDDD",
                "DDLLDDDD",
                "DDLDDDDDD",
                "LLDDDD",
                "LLDDDDD",
            ]

        candidates = []

        for template in templates:
            t_len = len(template)
            c_len = len(clean)

            if c_len >= t_len:
                for shift in range(c_len - t_len + 1):
                    part = clean[shift : shift + t_len]
                    fixed = self._apply_template(part, template)
                    if not fixed:
                        continue

                    score = 0
                    if template.startswith("LL"):
                        if fixed[:2].isalpha():
                            score += 30
                    else:
                        if fixed[:2] in VALID_PROVINCE_CODES:
                            score += 30

                    # Penalties for sliding window shift and extra characters
                    score -= shift * 2
                    extra_chars = c_len - t_len
                    score -= extra_chars * 15

                    changed = sum(1 for a, b in zip(part, fixed) if a != b)
                    score -= changed * 2

                    candidates.append((score, fixed))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # fallback: sửa 2 ký tự đầu về số nếu OCR nhầm tỉnh (và không phải biển quân sự)
        if len(clean) >= 2 and not clean[:2].isalpha():
            prefix = "".join(self.DIGIT_FIX_FORCE.get(c, self._to_digit(c)) for c in clean[:2])
            rest = clean[2:]
            if prefix.isdigit():
                return prefix + rest

        return clean

    # ======================================================================
    # SCORING
    # ======================================================================

    def _score_full_plate(
        self,
        raw: str,
        normalized: str,
        conf: float,
        layout_hint: Optional[str] = None,
    ) -> tuple:
        # Ý nghĩa: Hàm chấm điểm kết quả OCR để chọn ra ứng viên (candidate) xuất sắc nhất trong thuật toán ensemble.
        # Tiêu chí: Khớp mã tỉnh (+40), đúng định dạng chuẩn xe máy/ô tô/quân sự (+60 hoặc +35).
        # Đặc biệt là bị TRỪ ĐIỂM (- changed * 2) nếu chuỗi đã chuẩn hóa có quá nhiều ký tự bị bóp méo khác với raw OCR.
        clean = self._clean_ocr_text(normalized)
        hint = (layout_hint or "").lower()

        score = 0

        is_military_plate = len(clean) >= 2 and clean[:2].isalpha()
        if is_military_plate:
            score += 40
        else:
            if len(clean) >= 2 and clean[:2] in VALID_PROVINCE_CODES:
                score += 40

        is_standard = bool(
            re.match(r"^\d{2}[A-Z]{1,2}\d{4,5}$", clean)
        )

        is_motorbike_2row = bool(
            re.match(r"^\d{2}[A-Z]\d\d{4,5}$", clean) or re.match(r"^\d{2}[A-Z]{2}\d{4,5}$", clean)
        )

        is_military_layout = bool(
            re.match(r"^[A-Z]{2}\d{4,5}$", clean)
        )

        if hint == "military":
            if is_military_layout:
                score += 60
            else:
                score -= 30
        elif hint in {"one_row_auto", "one_row", "auto", "car", "standard"}:
            if is_standard:
                score += 60
            if is_motorbike_2row:
                score -= 20
            if is_military_layout:
                score -= 20
        elif hint in {"two_row_motorbike", "two_row", "motorbike_2row", "motorbike"}:
            if is_motorbike_2row:
                score += 60
            if is_standard:
                score -= 10
            if is_military_layout:
                score -= 10
        elif hint in {"two_row_auto", "auto_square", "square_auto"}:
            if is_standard:
                score += 60
            if is_military_layout:
                score -= 10
        else:
            if is_standard:
                score += 35
            if is_motorbike_2row:
                score += 35
            if is_military_layout:
                score += 35

        changed = self._calculate_mismatches(raw, clean)
        score -= int(changed * 2)

        return (
            score,
            float(conf),
            -int(changed),
            len(clean),
        )

    def _calculate_mismatches(self, raw: str, clean: str) -> float:
        # Ý nghĩa: Tính toán mức độ khác biệt (khoảng cách sai số) giữa chuỗi gốc (raw) và chuỗi đã chuẩn hóa (clean).
        # Có hỗ trợ trượt cửa sổ (sliding window) cho trường hợp 2 chuỗi lệch độ dài để tìm đoạn khớp nhất.
        if not raw or not clean:
            return max(len(raw), len(clean))
        
        len_r = len(raw)
        len_c = len(clean)
        
        if len_r == len_c:
            return sum(1 for a, b in zip(raw, clean) if a != b)
            
        if len_r < len_c:
            min_changes = len_c
            for shift in range(len_c - len_r + 1):
                window = clean[shift:shift+len_r]
                changes = sum(1 for a, b in zip(raw, window) if a != b)
                ignored = len_c - len_r
                total = changes + ignored * 0.5
                if total < min_changes:
                    min_changes = total
            return min_changes
        else:
            min_changes = len_r
            for shift in range(len_r - len_c + 1):
                window = raw[shift:shift+len_c]
                changes = sum(1 for a, b in zip(window, clean) if a != b)
                ignored = len_r - len_c
                total = changes + ignored * 0.5
                if total < min_changes:
                    min_changes = total
            return min_changes

    def _score_row_candidate(
        self,
        raw: str,
        normalized: str,
        conf: float,
        row_kind: str,
    ) -> tuple:
        clean = self._clean_ocr_text(normalized)
        score = 0

        if row_kind == "motorbike_top":
            # motorbike_top có thể là DDLD hoặc DDLL (50cc)
            if re.match(r"^\d{2}[A-Z]\d$", clean):
                score += 62  # Ưu tiên xe máy thường (DDLD) hơn xe máy 50cc (DDLL)
            elif re.match(r"^\d{2}[A-Z]{2}$", clean):
                score += 60
            if len(clean) == 4:
                score += 10
            if len(clean) >= 2 and clean[:2] in VALID_PROVINCE_CODES:
                score += 30
            
            # Phạt cực nặng các chữ cái không bao giờ xuất hiện trong sê-ri dân sự VN (O, Q, I, W)
            if len(clean) >= 3 and clean[2] in {"O", "Q", "I", "W"}:
                score -= 150
            if len(clean) >= 4 and clean[3] in {"O", "Q", "I", "W"} and clean[3].isalpha():
                score -= 150
            if clean.startswith("59"):
                if len(clean) >= 4 and clean[2].isalpha() and clean[3].isdigit():
                    if clean[3] not in {"1", "2", "3"}:
                        score -= 50

        elif row_kind == "auto_top":
            # auto_top có thể là DDL hoặc DDLL
            if re.match(r"^\d{2}[A-Z]$", clean):
                score += 62  # Ưu tiên xe con thường (DDL) hơn xe đặc biệt (DDLL)
            elif re.match(r"^\d{2}[A-Z]{2}$", clean):
                score += 60
            if len(clean) in {3, 4}:
                score += 10
            if len(clean) >= 2 and clean[:2] in VALID_PROVINCE_CODES:
                score += 30

        elif row_kind == "bottom":
            if clean.isdigit():
                score += 40
            if len(clean) in {4, 5}:
                score += 30
            else:
                score -= abs(len(clean) - 5) * 5

        changed = self._calculate_mismatches(raw, clean)
        score -= int(changed * 2)

        return (
            score,
            float(conf),
            -int(changed),
            len(clean),
        )

    # ======================================================================
    # UTILITY
    # ======================================================================

    def get_engine_info(self) -> dict:
        return {
            "easyocr_available": EASYOCR_AVAILABLE,
            "tesseract_available": TESSERACT_AVAILABLE,
            "active_engine": (
                "EasyOCR" if self.reader
                else ("Tesseract" if TESSERACT_AVAILABLE else "Mock")
            ),
            "languages": self.languages,
            "gpu": self.use_gpu,
        }