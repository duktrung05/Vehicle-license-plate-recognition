"""
License Plate Formatter & Post-Processor
Supports standard cars, square plates, motorcycles (including 50cc), specialized vehicles, military, and diplomatic formats.
"""

import re
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

VALID_PROVINCE_CODES = {
    "11","12","14","15","16","17","18","19","20","21","22","23",
    "24","25","26","27","28","29","30","31","32","33","34","35",
    "36","37","38","39","40","41","43","47","48","49","50","51","52",
    "53","54","55","56","57","58","59","60","61","62","63","64",
    "65","66","67","68","69","70","71","72","73","74","75","76",
    "77","78","79","80","81","82","83","84","85","86","88","89",
    "90","92","93","94","95","97","98","99",
}

PROVINCE_NAMES = {
    "11":"Cao Bằng","12":"Lạng Sơn","14":"Quảng Ninh","15":"Hải Phòng",
    "16":"Hải Phòng","17":"Thái Bình","18":"Nam Định","19":"Phú Thọ",
    "20":"Thái Nguyên","21":"Yên Bái","22":"Tuyên Quang","23":"Hà Giang",
    "24":"Lào Cai","25":"Lai Châu","26":"Sơn La","27":"Điện Biên",
    "28":"Hòa Bình","29":"Hà Nội","30":"Hà Nội","31":"Hà Nội",
    "32":"Hà Nội","33":"Hà Nội","34":"Hải Dương","35":"Ninh Bình",
    "36":"Thanh Hóa","37":"Nghệ An","38":"Hà Tĩnh","39":"Đồng Nai",
    "40":"Hà Nội","41":"TP. HCM","43":"Đà Nẵng","47":"Đắk Lắk","48":"Đắk Nông",
    "49":"Lâm Đồng","50":"TP. HCM","51":"TP. HCM","52":"TP. HCM",
    "53":"TP. HCM","54":"TP. HCM","55":"TP. HCM","56":"TP. HCM",
    "57":"TP. HCM","58":"TP. HCM","59":"TP. HCM","60":"Đồng Nai",
    "61":"Bình Dương","62":"Long An","63":"Tiền Giang","64":"Vĩnh Long",
    "65":"Cần Thơ","66":"Đồng Tháp","67":"An Giang","68":"Kiên Giang",
    "69":"Cà Mau","70":"Tây Ninh","71":"Bến Tre","72":"Bà Rịa-Vũng Tàu",
    "73":"Quảng Bình","74":"Quảng Trị","75":"Thừa Thiên Huế","76":"Quảng Ngãi",
    "77":"Bình Định","78":"Phú Yên","79":"Khánh Hòa","80":"Cục CSGT Bộ Công an",
    "81":"Gia Lai","82":"Kon Tum","83":"Sóc Trăng","84":"Trà Vinh",
    "85":"Ninh Thuận","86":"Bình Thuận","88":"Vĩnh Phúc","89":"Hưng Yên",
    "90":"Hà Nam","92":"Quảng Nam","93":"Bình Phước","94":"Bạc Liêu",
    "95":"Hậu Giang","97":"Bắc Kạn","98":"Bắc Giang","99":"Bắc Ninh",
}

class PlateFormatter:
    """
    Formatter cho các loại biển số Việt Nam.
    """

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

    # Sửa lỗi ký tự bắt buộc là chữ
    LETTER_FIX_FORCE = {
        "0": "O", "1": "I", "2": "Z", "3": "E", "4": "A",
        "5": "S", "6": "G", "7": "T", "8": "B", "9": "G",
    }

    VALID_MOTORCYCLE_DISTRICTS = {
        "29": {"B1", "C1", "D1", "D2", "E1", "E2", "F1", "G1", "H1", "H9", "K1", "L1", "L5", "M1", "N1", "P1", "S1", "S2", "T1", "U1", "V1", "V2", "V3", "V4", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Z1"},
        "59": {"B1", "B2", "T1", "T2", "F1", "F2", "C1", "C2", "C3", "C4", "H1", "H2", "K1", "K2", "L1", "L2", "U1", "U2", "M1", "M2", "G1", "G2", "X1", "X2", "X3", "X4", "P1", "P2", "D1", "D2", "D3", "N1", "N2", "S1", "S2", "S3", "V1", "V2", "V3", "E1", "E2", "Y1", "Y2", "Y3", "Y4", "Y5", "Z1", "Z2"},
        "79": {"C1", "D1", "H1", "N1", "N2", "S1", "V1", "X1", "Z1"},
        "73": {"B1", "C1", "D1", "E1", "F1", "G1", "H1", "K1"},
        "23": {"B1", "C1", "D1", "E1", "F1", "G1", "H1"},
        "19": {"B1", "C1", "D1", "E1", "F1", "G1", "H1", "K1", "L1", "M1", "N1", "P1", "S1", "S2"},
        "72": {"C1", "D1", "E1", "F1", "G1", "H1", "K1"},
        "98": {"B1", "D1", "E1", "F1", "G1", "H1", "K1", "L1", "M1", "N1", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9"},
    }

    def __init__(self, strict_validation: bool = False):
        self.strict_validation = strict_validation

    def format(
        self,
        raw_text: str,
        layout_hint: Optional[str] = None
    ) -> Tuple[str, bool, dict]:
        if not raw_text:
            return "", False, {"error": "Empty input"}

        raw = str(raw_text).upper()
        clean = self._clean(raw)

        formatted, pattern = self._format_pattern(
            clean=clean,
            raw=raw,
            layout_hint=layout_hint,
        )

        is_valid = self._validate(formatted, pattern)
        metadata = self._get_metadata(formatted, is_valid, pattern)
        metadata["layout_hint"] = layout_hint
        metadata["cleaned"] = clean

        return formatted, is_valid, metadata

    def batch_format(
        self,
        raw_texts: List[str],
        layout_hint: Optional[str] = None
    ) -> list:
        return [self.format(t, layout_hint=layout_hint) for t in raw_texts]

    def _clean(self, text: str) -> str:
        text = text.upper()
        return re.sub(r"[^A-Z0-9]", "", text)

    def _to_digit(self, c: str) -> str:
        c = c.upper()
        return self.DIGIT_FIX.get(c, c)

    def _to_letter(self, c: str) -> str:
        c = c.upper()
        return self.LETTER_FIX.get(c, c)

    def _apply_template(self, text: str, template: str) -> Optional[str]:
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

        # Biển quân sự không kiểm tra mã tỉnh dân sự
        is_military = template.startswith("LL")
        if not is_military and self.strict_validation and fixed_text[:2] not in VALID_PROVINCE_CODES:
            return None

        return fixed_text

    def _generate_similar_candidates(self, fixed: str, template: str) -> List[str]:
        confusion_digit = {
            "1": ["4", "7"],
            "4": ["1"],
            "2": ["7"],
            "7": ["2", "1"],
            "3": ["9", "8"],
            "9": ["3"],
            "8": ["0", "3"],
            "0": ["8"],
            "5": ["6"],
            "6": ["5"],
        }
        confusion_letter = {
            "D": ["G", "U"],
            "G": ["D"],
            "U": ["D"],
            "K": ["X"],
            "X": ["K"],
        }

        options = []
        for i, (ch, kind) in enumerate(zip(fixed, template)):
            pos_options = [ch]
            if kind == "D" and ch in confusion_digit:
                pos_options.extend(confusion_digit[ch])
            elif kind == "L" and ch in confusion_letter:
                pos_options.extend(confusion_letter[ch])
            options.append(pos_options)
            
        import itertools
        all_combos = itertools.product(*options)
        
        results = []
        for combo in all_combos:
            combo_str = "".join(combo)
            diffs = sum(1 for a, b in zip(fixed, combo_str) if a != b)
            if diffs <= 2:
                results.append(combo_str)
                
        return results

    def _format_pattern(
        self,
        clean: str,
        raw: str = "",
        layout_hint: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        if not clean:
            return "", None

        candidates = []

        def add_candidate(name: str, template: str):
            t_len = len(template)
            c_len = len(clean)

            if c_len < t_len:
                return

            for shift in range(c_len - t_len + 1):
                part = clean[shift : shift + t_len]
                base_fixed = self._apply_template(part, template)
                if not base_fixed:
                    continue

                similar_fixed_list = self._generate_similar_candidates(base_fixed, template)
                for fixed in similar_fixed_list:
                    formatted = None
                    pattern = None

                    # Xe máy 2 dòng, 5 số: 59V107473 -> 59-V1 074.73
                    if name == "motorbike_2row_5":
                        formatted = f"{fixed[:2]}-{fixed[2:4]} {fixed[4:7]}.{fixed[7:9]}"
                        pattern = "motorbike_2row"

                    # Xe máy 2 dòng cũ, 4 số: 52U78693 -> 52-U7 8693
                    elif name == "motorbike_2row_4":
                        formatted = f"{fixed[:2]}-{fixed[2:4]} {fixed[4:8]}"
                        pattern = "motorbike_2row_old"

                    # Xe máy 50cc 2 dòng, 5 số: 59AA12345 -> 59-AA 123.45
                    elif name == "motorbike_50cc_5":
                        if fixed[2] not in {"A", "M"}:
                            continue
                        formatted = f"{fixed[:2]}-{fixed[2:4]} {fixed[4:7]}.{fixed[7:9]}"
                        pattern = "motorbike_2row"

                    # Xe máy 50cc 2 dòng, 4 số: 59AA1234 -> 59-AA 1234
                    elif name == "motorbike_50cc_4":
                        if fixed[2] not in {"A", "M"}:
                            continue
                        formatted = f"{fixed[:2]}-{fixed[2:4]} {fixed[4:8]}"
                        pattern = "motorbike_2row_old"

                    # Biển ô tô / tiêu chuẩn 1 chữ, 5 số: 30A58373 -> 30A-583.73
                    elif name == "standard_1_5":
                        formatted = f"{fixed[:3]}-{fixed[3:6]}.{fixed[6:8]}"
                        pattern = "standard"

                    # Biển ô tô / tiêu chuẩn 1 chữ, 4 số: 51A1234 -> 51A-1234
                    elif name == "standard_1_4":
                        formatted = f"{fixed[:3]}-{fixed[3:7]}"
                        pattern = "standard"

                    # Biển ô tô 2 chữ, 5 số: 30AB12345 -> 30AB-123.45 / Ngoại giao 80-NG-123-45
                    elif name == "standard_2_5":
                        if fixed.startswith("80NG") and len(fixed) == 9:
                            formatted = f"80-NG-{fixed[4:7]}-{fixed[7:9]}"
                            pattern = "diplomatic"
                        elif fixed.startswith("80NN") and len(fixed) == 9:
                            formatted = f"80-NN-{fixed[4:7]}.{fixed[7:9]}"
                            pattern = "diplomatic"
                        else:
                            formatted = f"{fixed[:4]}-{fixed[4:7]}.{fixed[7:9]}"
                            pattern = "standard"

                    # Biển ô tô 2 chữ, 4 số: 30AB1234 -> 30AB-1234
                    elif name == "standard_2_4":
                        formatted = f"{fixed[:4]}-{fixed[4:8]}"
                        pattern = "standard"

                    # Biển Quân sự 4 số: AA1234 -> AA-12-34
                    elif name == "military_4":
                        formatted = f"{fixed[:2]}-{fixed[2:4]}-{fixed[4:6]}"
                        pattern = "military"

                    # Biển Quân sự 5 số: TM12345 -> TM-123.45
                    elif name == "military_5":
                        formatted = f"{fixed[:2]}-{fixed[2:5]}.{fixed[5:7]}"
                        pattern = "military"

                    # Biển ngắn
                    elif name == "short_1_3":
                        formatted = f"{fixed[:3]}-{fixed[3:6]}"
                        pattern = "short"

                    elif name == "short_1_4":
                        formatted = f"{fixed[:3]}-{fixed[3:7]}"
                        pattern = "short"

                    if formatted and pattern:
                        score_tuple = self._score_candidate(
                            original=part,
                            fixed=fixed,
                            pattern=pattern,
                            variant=name,
                            layout_hint=layout_hint,
                            raw=raw,
                        )
                        score = score_tuple[0]
                        score -= shift * 5
                        extra_chars = c_len - t_len
                        score -= extra_chars * 15

                        new_score_tuple = (score,) + score_tuple[1:]
                        candidates.append((new_score_tuple, formatted, pattern))

        # Đăng ký các candidate mẫu dựa trên layout_hint
        hint = (layout_hint or "").lower()
        if hint in {"two_row_motorbike", "motorbike_2row", "motorbike"}:
            add_candidate("motorbike_2row_5", "DDLDDDDDD")
            add_candidate("motorbike_2row_4", "DDLDDDDD")
            add_candidate("motorbike_50cc_5", "DDLLDDDDD")
            add_candidate("motorbike_50cc_4", "DDLLDDDD")
            add_candidate("military_4", "LLDDDD")
            add_candidate("military_5", "LLDDDDD")
        elif hint in {"two_row_auto", "auto_square", "square_auto"}:
            add_candidate("standard_1_5", "DDLDDDDD")
            add_candidate("standard_1_4", "DDLDDDD")
            add_candidate("standard_2_5", "DDLLDDDDD")
            add_candidate("standard_2_4", "DDLLDDDD")
            add_candidate("military_4", "LLDDDD")
            add_candidate("military_5", "LLDDDDD")
        elif hint in {"one_row_auto", "one_row", "auto", "car", "standard"}:
            add_candidate("standard_1_5", "DDLDDDDD")
            add_candidate("standard_1_4", "DDLDDDD")
            add_candidate("standard_2_5", "DDLLDDDDD")
            add_candidate("standard_2_4", "DDLLDDDD")
        else:
            add_candidate("motorbike_2row_5", "DDLDDDDDD")
            add_candidate("motorbike_2row_4", "DDLDDDDD")
            add_candidate("motorbike_50cc_5", "DDLLDDDDD")
            add_candidate("motorbike_50cc_4", "DDLLDDDD")
            add_candidate("standard_1_5", "DDLDDDDD")
            add_candidate("standard_1_4", "DDLDDDD")
            add_candidate("standard_2_5", "DDLLDDDDD")
            add_candidate("standard_2_4", "DDLLDDDD")
            add_candidate("military_4", "LLDDDD")
            add_candidate("military_5", "LLDDDDD")
            add_candidate("short_1_3", "DDLDDD")
            add_candidate("short_1_4", "DDLDDDD")

        if not candidates:
            return clean, None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][2]

    def _score_candidate(
        self,
        original: str,
        fixed: str,
        pattern: str,
        variant: str,
        layout_hint: Optional[str],
        raw: str,
    ) -> tuple:
        score = 0

        is_military = variant.startswith("military")
        if is_military:
            if fixed[:2].isalpha():
                score += 30
        else:
            if fixed[:2] in VALID_PROVINCE_CODES:
                score += 30
                if fixed[:2] == original[:2]:
                    score += 20
            elif self.strict_validation:
                score -= 100

        # Tính toán penalty thông minh cho các ký tự thay đổi
        # - Chữ số nhầm lẫn thông thường (e.g. I->1, O->0): phạt nhẹ (2 điểm)
        # - Ký tự nhầm lẫn đã biết (e.g. D<->G<->U): phạt nhẹ (4 điểm)
        # - Các cặp nhầm lẫn thường gặp nhưng nguy hiểm (e.g. 2<->7, 1<->4, 1<->7, N<->M): phạt trung bình/nặng (12 điểm)
        # - Thay đổi ký tự ép buộc nặng nề (e.g. A->4, H->4): phạt rất nặng (20 điểm)
        penalty = 0
        changed = 0
        known_confusions = {
            ("3", "9"), ("9", "3"),
            ("3", "8"), ("8", "3"),
            ("8", "0"), ("0", "8"),
            ("5", "6"), ("6", "5"),
            ("D", "G"), ("G", "D"),
            ("D", "U"), ("U", "D"),
            ("L", "1"), ("1", "L"),
            ("L", "4"), ("4", "L"),
            ("D", "0"), ("0", "D"),
            ("D", "O"), ("O", "D"),
            ("K", "X"), ("X", "K"),
        }
        for a, b in zip(original, fixed):
            if a != b:
                changed += 1
                if b.isdigit() and a in self.DIGIT_FIX and b == self.DIGIT_FIX[a]:
                    penalty += 2
                elif b.isalpha() and a in self.LETTER_FIX and b == self.LETTER_FIX[a]:
                    penalty += 2
                elif (a, b) in {("1", "7"), ("7", "1"), ("1", "4"), ("4", "1"), ("2", "7"), ("7", "2"), ("N", "M"), ("M", "N")}:
                    penalty += 12
                elif (a, b) in known_confusions:
                    penalty += 4
                else:
                    penalty += 20
        score -= penalty

        hint = (layout_hint or "").lower()

        # Ưu tiên mạnh theo layout từ pipeline
        if hint in {"two_row", "two_row_motorbike", "motorbike_2row", "motorbike"}:
            if pattern.startswith("motorbike"):
                score += 100
                if variant in {"motorbike_2row_5", "motorbike_2row_4"}:
                    score += 1 # Ưu tiên xe máy thường trên 50cc so với xe máy 50cc
            elif pattern == "military":
                score += 50
            else:
                score -= 120

        elif hint in {"one_row", "one_row_auto", "two_row_auto", "auto", "car", "standard", "auto_square"}:
            if pattern in {"standard", "diplomatic"}:
                score += 100
            elif pattern == "military":
                score += 50
            else:
                score -= 120
                
        elif hint == "military":
            if pattern == "military":
                score += 100
            else:
                score -= 30

        else:
            # Không có hint thì đoán
            if variant == "standard_1_5":
                score += 8
            if variant == "motorbike_2row_4":
                score += 5
            if variant.startswith("motorbike") and len(fixed) >= 4 and fixed[3] in {"8", "9"}:
                score -= 12

        # Phạt cực nặng các chữ cái không bao giờ xuất hiện trong sê-ri dân sự VN (O, Q, I, W)
        if pattern.startswith("motorbike") or pattern == "standard":
            if len(fixed) >= 3 and fixed[2] in {"O", "Q", "I", "W"}:
                score -= 150
            if len(fixed) >= 4 and fixed[3] in {"O", "Q", "I", "W"} and fixed[3].isalpha():
                score -= 150

        # Phạt cực nặng các sê-ri hai chữ cái không hợp lệ cho ô tô dân sự
        if pattern == "standard" and variant in {"standard_2_5", "standard_2_4"}:
            series = fixed[2:4]
            invalid_letters = {"I", "O", "Q", "W"}
            if any(c in invalid_letters for c in series):
                score -= 150

        # Phạt cực nặng nếu số sê-ri xe máy TP.HCM không hợp lệ (không phải 1, 2, 3)
        # Xe máy dân sự TP.HCM (59) bắt buộc dùng số sê-ri sau chữ cái là 1, 2, 3 (e.g. 59-V1, 59-V2, 59-V3)
        if fixed.startswith("59") and pattern.startswith("motorbike"):
            if len(fixed) >= 4 and fixed[2].isalpha() and fixed[3].isdigit():
                if fixed[3] not in {"1", "2", "3"}:
                    score -= 50

        # Quy tắc luật biển số xe máy hiện đại (5 chữ số dòng dưới, tức độ dài = 9):
        # - TP.HCM chỉ dùng "59" cho xe máy. Các mã cũ "50"-"58" chỉ dùng cho xe máy 4 số hoặc ô tô, không dùng cho xe máy 5 số.
        # - Hà Nội chỉ dùng "29" cho xe máy. Các mã "30"-"33", "40" chỉ dùng cho ô tô, không dùng cho xe máy.
        if pattern.startswith("motorbike") and variant in {"motorbike_2row_5", "motorbike_50cc_5"}:
            if fixed[:2] in {"50", "51", "52", "53", "54", "55", "56", "57", "58"}:
                score -= 100
            if fixed[:2] in {"30", "31", "32", "33", "40"}:
                score -= 100

        # Kiểm tra mã quận/huyện hợp lệ cho xe máy
        if pattern.startswith("motorbike") and len(fixed) >= 8:
            prov = fixed[:2]
            dist = fixed[2:4]
            # Không có biển xe máy dân sự nào dùng số quận/huyện là '0'
            if dist[1] == "0":
                score -= 120
            # Kiểm tra chữ cái sê-ri hợp lệ cho các tỉnh đã định nghĩa
            elif prov in self.VALID_MOTORCYCLE_DISTRICTS:
                valid_letters = {code[0] for code in self.VALID_MOTORCYCLE_DISTRICTS[prov]} | {"A"}
                if fixed[2] not in valid_letters:
                    score -= 120
                # Kiểm tra mã đầy đủ cho biển 5 số sê-ri chuẩn (độ dài = 9)
                elif len(fixed) == 9 and dist[1].isdigit():
                    if dist not in self.VALID_MOTORCYCLE_DISTRICTS[prov]:
                        score -= 120

        return (score, -changed, len(fixed))

    def _validate(self, plate: str, pattern: Optional[str] = None) -> bool:
        if not plate or not pattern:
            return False

        clean = self._clean(plate)

        if not (6 <= len(clean) <= 10):
            return False

        if pattern == "military":
            return bool(re.match(r"^[A-Z]{2}\d{4,5}$", clean))

        if clean[:2] not in VALID_PROVINCE_CODES:
            return not self.strict_validation

        # Civilian plates cannot contain O, Q, I, W in series
        if pattern.startswith("motorbike") or pattern == "standard":
            if len(clean) >= 3 and clean[2] in {"O", "Q", "I", "W"}:
                return False
            if len(clean) >= 4 and clean[3] in {"O", "Q", "I", "W"} and clean[3].isalpha():
                return False

        # Civilian car plates with 2 letters must be in valid list
        if pattern == "standard" and len(clean) >= 4 and clean[2:4].isalpha():
            series = clean[2:4]
            invalid_letters = {"I", "O", "Q", "W"}
            if any(c in invalid_letters for c in series):
                return False

        # Motorbike civilian plates in HCM (59) must use series number 1, 2, 3
        if clean.startswith("59") and pattern.startswith("motorbike"):
            if len(clean) >= 4 and clean[2].isalpha() and clean[3].isdigit():
                if clean[3] not in {"1", "2", "3"}:
                    return False

        # Motorbike district validation for both 4-digit and 5-digit bottom rows
        if pattern.startswith("motorbike") and len(clean) in {8, 9}:
            prov = clean[:2]
            dist = clean[2:4]
            if dist[1] == "0":
                return False
            
            # Motorbike cannot use car-only province codes
            if prov in {"50", "51", "52", "53", "54", "55", "56", "57", "58"}:
                return False
            if prov in {"30", "31", "32", "33", "40"}:
                return False
            
            if prov in self.VALID_MOTORCYCLE_DISTRICTS:
                valid_letters = {code[0] for code in self.VALID_MOTORCYCLE_DISTRICTS[prov]} | {"A"}
                if clean[2] not in valid_letters:
                    return False
                if dist[1].isdigit():
                    if dist not in self.VALID_MOTORCYCLE_DISTRICTS[prov]:
                        return False

        # Verify specific layout formats
        if pattern.startswith("motorbike") or pattern == "short":
            valid_patterns = [
                r"^\d{2}[A-Z]\d\d{5}$",       # 59V107473
                r"^\d{2}[A-Z]\d\d{4}$",       # 52U78693
                r"^\d{2}A[A-Z]\d{5}$",        # 59AA12345 (50cc)
                r"^\d{2}A[A-Z]\d{4}$",        # 59AA1234 (50cc)
                r"^\d{2}[A-Z]\d{3,4}$",       # 29A123 / 29A1234 (short)
            ]
        elif pattern in {"standard", "diplomatic"}:
            valid_patterns = [
                r"^\d{2}[A-Z]\d{4,5}$",       # 30A58373 / 30A1234
                r"^\d{2}[A-Z]{2}\d{4,5}$",    # 30AB12345 / 80NG12345
            ]
        else:
            valid_patterns = [r"^\d{2}[A-Z]{1,2}\d{4,5}$"]

        return any(re.match(p, clean) for p in valid_patterns)

    def _get_metadata(
        self,
        plate: str,
        is_valid: bool,
        pattern: Optional[str],
    ) -> dict:
        clean = self._clean(plate)

        metadata = {
            "is_valid": is_valid,
            "pattern_matched": pattern is not None,
            "pattern_name": pattern,
            "province_code": None,
            "province_name": None,
            "plate_type": None,
        }

        if pattern == "military":
            metadata["plate_type"] = "Military Vehicle"
            metadata["province_name"] = "Military"
            return metadata

        if len(clean) >= 2 and clean[:2].isdigit():
            code = clean[:2]
            metadata["province_code"] = code
            metadata["province_name"] = PROVINCE_NAMES.get(code, "Unknown")

        if pattern == "motorbike_2row":
            metadata["plate_type"] = "Motorcycle 2-row, 5-digit"
        elif pattern == "motorbike_2row_old":
            metadata["plate_type"] = "Motorcycle 2-row, 4-digit"
        elif pattern == "standard":
            metadata["plate_type"] = "Car / Standard"
        elif pattern == "diplomatic":
            metadata["plate_type"] = "Diplomatic / Foreign"
        elif pattern == "short":
            metadata["plate_type"] = "Short motorcycle"

        return metadata