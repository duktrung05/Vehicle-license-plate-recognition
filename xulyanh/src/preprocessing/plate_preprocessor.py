import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class PlatePreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (240, 80),
        enhance_contrast: bool = True,
    ):
        self.target_size = target_size  # width, height
        self.enhance_contrast = enhance_contrast

    def _should_invert_plate(self, img: np.ndarray) -> bool:
        if img is None or img.size == 0:
            return False
        
        try:
            h, w = img.shape[:2]
            border_y = max(2, int(h * 0.06))
            border_x = max(2, int(w * 0.06))
            
            border_pixels = []
            border_pixels.append(img[:border_y, :])
            border_pixels.append(img[-border_y:, :])
            border_pixels.append(img[border_y:-border_y, :border_x])
            border_pixels.append(img[border_y:-border_y, -border_x:])
            
            combined_borders = np.vstack([p.reshape(-1, 3) for p in border_pixels])
            
            gray_borders = cv2.cvtColor(combined_borders.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
            mean_gray = np.mean(gray_borders)
            
            if mean_gray < 115:
                return True
                
            b, g, r = np.mean(combined_borders, axis=0)
            
            # Blue plate: B channel is dominant
            if b > r + 25 and b > g + 10:
                return True
            # Green plate: G channel is dominant
            if g > r + 20 and g > b + 10:
                return True
            # Red plate: R channel is dominant
            if r > b + 30 and r > g + 30:
                return True
        except Exception as e:
            logger.debug(f"_should_invert_plate error: {e}")
            
        return False

    def check_and_invert(self, img: np.ndarray) -> np.ndarray:
        if self._should_invert_plate(img):
            logger.info("Detected dark background/colored plate. Inverting crop for normalization.")
            return cv2.bitwise_not(img)
        return img

    # ======================================================================
    # MAIN PREPROCESS
    # ======================================================================

    def preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Standard binary preprocess.
        Returns grayscale binary image.
        """
        if plate_crop is None or plate_crop.size == 0:
            logger.warning("Empty plate crop received")
            return np.zeros(
                (self.target_size[1], self.target_size[0]),
                dtype=np.uint8,
            )

        plate_crop = self.check_and_invert(plate_crop)

        resized = self._resize_to_height(plate_crop, target_height=80)
        deskewed = self._deskew(resized)
        gray = self._to_grayscale(deskewed)
        denoised = self._denoise(gray)

        if self.enhance_contrast:
            enhanced = self._enhance_contrast(denoised)
        else:
            enhanced = denoised

        binary = self._binarize(enhanced)
        return binary

    def preprocess_for_ocr(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Return BGR image suitable for EasyOCR.
        """
        binary = self.preprocess(plate_crop)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ======================================================================
    # TWO-ROW DETECTION
    # ======================================================================

    def is_two_row_plate(self, image: np.ndarray) -> bool:
        """
        Detect whether plate has 2 text rows using horizontal projection.
        """
        if image is None or image.size == 0:
            return False

        image = self.check_and_invert(image)

        h, w = image.shape[:2]
        aspect = w / max(1, h)
        if aspect >= 2.2:
            return False
        if aspect < 1.7:
            return True

        try:
            return self._has_two_rows_by_projection(image)
        except Exception as e:
            logger.debug(f"is_two_row_plate failed: {e}")
            return False

    def _has_two_rows_by_projection(self, image: np.ndarray) -> bool:
        """
        Use horizontal projection to detect 2 separate text bands.
        """
        mask = self._get_text_mask(image)
        h, _ = mask.shape

        if h < 24:
            return False

        proj = np.sum(mask > 0, axis=1).astype(np.float32)
        proj = self._smooth_1d(proj, k=5)

        total_ink = float(np.sum(proj))
        if total_ink <= 0:
            return False

        split_y = self._find_split_y_from_projection(proj)

        top_part = proj[:split_y]
        bottom_part = proj[split_y:]

        if top_part.size == 0 or bottom_part.size == 0:
            return False

        top_ink = float(np.sum(top_part))
        bottom_ink = float(np.sum(bottom_part))

        top_peak = float(np.max(top_part))
        bottom_peak = float(np.max(bottom_part))
        valley = float(proj[split_y])

        weaker_peak = min(top_peak, bottom_peak)

        if weaker_peak <= 0:
            return False

        top_ratio = top_ink / total_ink
        bottom_ratio = bottom_ink / total_ink

        # Cả hai nửa phải có chữ tương đối rõ
        enough_ink = top_ratio > 0.18 and bottom_ratio > 0.18

        # Giữa 2 dòng phải có valley thấp hơn peak
        clear_valley = valley < weaker_peak * 0.45

        # Split không được quá sát mép
        split_ok = int(h * 0.30) <= split_y <= int(h * 0.70)

        return bool(enough_ink and clear_valley and split_ok)

    # ======================================================================
    # TWO-ROW SPLIT
    # ======================================================================

    # ======================================================================
    # TWO-ROW SPLIT
    # ======================================================================

    def split_two_row_plate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a 2-row plate into top row and bottom row.
        """
        if image is None or image.size == 0:
            return image, image

        image = self.check_and_invert(image)

        h, w = image.shape[:2]
        mask = self._get_text_mask(image)

        if h < 24:
            mid = h // 2
            return image[:mid].copy(), image[mid:].copy()

        proj = np.sum(mask > 0, axis=1).astype(np.float32)
        proj = self._smooth_1d(proj, k=5)

        split_y = self._find_split_y_from_projection(proj)

        # Cho overlap lớn hơn để không cắt mất nét chữ khi góc chụp nghiêng/nén
        aspect = w / max(1, h)
        if aspect < 1.2:  # biển rất vuông
            overlap = max(8, int(h * 0.12))  # tăng từ 0.08 lên 0.12
        else:
            overlap = max(5, int(h * 0.08))

        top_end = min(h, split_y + overlap)
        bottom_start = max(0, split_y - overlap)

        row1 = image[:top_end].copy()
        row2 = image[bottom_start:].copy()

        # Trim bottom of row2 to remove plate border (reduced from 0.08 to 0.02 to avoid character clipping)
        r2_h = row2.shape[0]
        trim_h = int(r2_h * 0.02)
        if trim_h > 0 and r2_h - trim_h > 10:
            row2 = row2[:-trim_h, :]

        return row1, row2

    def get_two_row_proposals(
        self,
        image: np.ndarray,
        fast_mode: bool = False,
        speed_mode: Optional[str] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Split a 2-row plate and return OCR proposals for each row.
        """
        image = self.check_and_invert(image)
        row1, row2 = self.split_two_row_plate(image)

        h, w = image.shape[:2]
        aspect_ratio = w / max(1, h)

        is_fast = (speed_mode == "fast" or (speed_mode is None and fast_mode))

        # If the plate is very square (aspect ratio < 1.30, e.g. steep motorcycle angle),
        # we force add_border = True to prevent border noise/caps from prepending characters.
        if aspect_ratio < 1.30:
            row1_border = True
            row2_border = True
        else:
            row1_border = not is_fast
            row2_border = False

        # For row1, use target_height=80 and bypass_trim=True to retain character detail
        row1_proposals = self._make_ocr_proposals(
            row1,
            target_height=80,
            include_inverted=True,
            fast_mode=fast_mode,
            speed_mode=speed_mode,
            bypass_trim=True,
            add_border=row1_border,
        )

        row2_proposals = self._make_ocr_proposals(
            row2,
            target_height=80,
            include_inverted=True,
            fast_mode=fast_mode,
            speed_mode=speed_mode,
            bypass_trim=False,
            add_border=row2_border,
        )

        return row1_proposals, row2_proposals

    # ======================================================================
    # ONE-ROW PROPOSALS
    # ======================================================================

    def get_plate_region_proposals(
        self,
        image: np.ndarray,
        fast_mode: bool = False,
        speed_mode: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Generate OCR proposals for a full plate crop.
        """
        image = self.check_and_invert(image)
        return self._make_ocr_proposals(
            image,
            target_height=80,
            include_inverted=True,
            fast_mode=fast_mode,
            speed_mode=speed_mode,
        )

    # ======================================================================
    # OCR PROPOSAL GENERATION
    # ======================================================================

    def _make_ocr_proposals(
        self,
        image: np.ndarray,
        target_height: int = 72,
        include_inverted: bool = True,
        fast_mode: bool = False,
        speed_mode: Optional[str] = None,
        bypass_trim: bool = False,
        add_border: bool = True,
    ) -> List[np.ndarray]:
        """
        Create multiple image versions for OCR ensemble.
        Supports speed_mode: "fast", "balanced", "accurate"
        """
        proposals: List[np.ndarray] = []

        if image is None or image.size == 0:
            return proposals

        image = self._ensure_bgr(image)

        if bypass_trim:
            base = image
        else:
            # Trim content first, but keep margin
            base = self._trim_content(image, margin_ratio=0.10)
            base = self._ensure_bgr(base)
        
        h_orig = image.shape[0]
        if add_border:
            # Use smaller initial padding for smaller crops
            init_pad = 2 if h_orig < 80 else 4
            base = self._add_white_border(base, init_pad)

        # Map fast_mode to speed_mode for compatibility
        if speed_mode is None:
            speed_mode = "fast" if fast_mode else "balanced"

        if speed_mode == "fast":
            # Fast mode: skip heavy enhancements, just resize and return 1 proposal
            resized_base = self._resize_to_height(
                base,
                target_height=target_height,
                max_width=720,
            )
            return [resized_base]

        # Base proposals
        resized_base = self._resize_to_height(
            base,
            target_height=target_height,
            max_width=720,
        )
        proposals.append(resized_base)

        gray_base = self._to_grayscale(resized_base)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        
        # Add CLAHE and Sharpen for the base unpadded version
        enhanced_base = clahe.apply(gray_base)
        proposals.append(cv2.cvtColor(enhanced_base, cv2.COLOR_GRAY2BGR))
        proposals.append(cv2.cvtColor(self._sharpen(enhanced_base), cv2.COLOR_GRAY2BGR))

        # Padding proposals based on speed mode
        if add_border:
            if speed_mode == "balanced":
                # Just 2 extra padded versions, no CLAHE/Sharpen on them to save speed
                pads = [2, 4] if h_orig < 60 else [4, 8]
                for pad in pads:
                    padded = self._add_white_border(base, pad)
                    resized = self._resize_to_height(padded, target_height=target_height, max_width=720)
                    proposals.append(resized)
            else:  # "accurate"
                # Full ensemble with 4 paddings and CLAHE/Sharpen on each
                if h_orig < 50:
                    pads = [0, 1, 2, 3]
                elif h_orig < 80:
                    pads = [0, 2, 4, 6]
                else:
                    pads = [0, 4, 8, 12]
                for pad in pads:
                    padded = self._add_white_border(base, pad)
                    resized = self._resize_to_height(
                        padded,
                        target_height=target_height,
                        max_width=720,
                    )
                    proposals.append(resized)

                    gray = self._to_grayscale(resized)
                    enhanced = clahe.apply(gray)
                    proposals.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
                    proposals.append(cv2.cvtColor(self._sharpen(enhanced), cv2.COLOR_GRAY2BGR))

        # Add gamma correction proposals for dark/night images
        gray_orig = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_orig)
        
        if speed_mode == "balanced":
            # Only apply 1 gamma and only if dark (< 110)
            gammas = [1.8] if brightness < 110 else []
        else:
            gammas = [1.5, 2.0]
            
        for g in gammas:
            gamma_img = self._adjust_gamma(base, gamma=g)
            resized_g = self._resize_to_height(
                gamma_img,
                target_height=target_height,
                max_width=720,
            )
            proposals.append(resized_g)
            
            gray_g = self._to_grayscale(resized_g)
            clahe_g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced_g = clahe_g.apply(gray_g)
            proposals.append(cv2.cvtColor(enhanced_g, cv2.COLOR_GRAY2BGR))

        # Binary proposals from non-padded base
        # OTSU normal
        _, otsu = cv2.threshold(
            gray_base,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        proposals.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

        # OTSU inverted
        if include_inverted:
            otsu_inv = cv2.bitwise_not(otsu)
            proposals.append(cv2.cvtColor(otsu_inv, cv2.COLOR_GRAY2BGR))

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray_base,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=3,
        )
        proposals.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

        # Adaptive inverted
        if include_inverted:
            adaptive_inv = cv2.bitwise_not(adaptive)
            proposals.append(cv2.cvtColor(adaptive_inv, cv2.COLOR_GRAY2BGR))

        # Standard preprocess
        try:
            std = self.preprocess_for_ocr(base)
            proposals.append(std)
        except Exception as e:
            logger.debug(f"standard preprocess proposal failed: {e}")

        return self._deduplicate_proposals(proposals)

    def _deduplicate_proposals(self, proposals: List[np.ndarray]) -> List[np.ndarray]:
        """
        Remove invalid and exact duplicate proposal shapes/content lightly.
        """
        valid = []
        seen = set()

        for img in proposals:
            if img is None or img.size == 0:
                continue

            key = (
                img.shape[0],
                img.shape[1],
                int(np.mean(img)),
                int(np.std(img)),
            )

            if key in seen:
                continue

            seen.add(key)
            valid.append(img)

        return valid

    # ======================================================================
    # PRIVATE IMAGE HELPERS
    # ======================================================================

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return image

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Resize to self.target_size while preserving aspect ratio.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        if h <= 0 or w <= 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)

        scale = min(target_w / max(1, w), target_h / max(1, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )

        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left

        fill = [255, 255, 255] if len(resized.shape) == 3 else 255

        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=fill,
        )

    def _resize_to_height(
        self,
        image: np.ndarray,
        target_height: int = 72,
        max_width: int = 720,
    ) -> np.ndarray:
        """
        Resize preserving aspect ratio by target height.
        """
        h, w = image.shape[:2]

        if h <= 0 or w <= 0:
            return image

        scale = target_height / max(1, h)
        new_w = max(24, int(w * scale))
        new_w = min(new_w, max_width)

        return cv2.resize(
            image,
            (new_w, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def _add_white_border(self, image: np.ndarray, pad: int) -> np.ndarray:
        fill = [255, 255, 255] if len(image.shape) == 3 else 255

        return cv2.copyMakeBorder(
            image,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=fill,
        )

    def _get_text_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Return binary inverted mask where text strokes are white.
        """
        gray = self._to_grayscale(image)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Clean tiny noise, keep character strokes
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def _trim_content(
        self,
        image: np.ndarray,
        margin_ratio: float = 0.10,
    ) -> np.ndarray:
        """
        Crop around text/content with margin.
        If text mask is unreliable, return original.
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        aspect = w / max(1, h)

        if aspect >= 2.2:
            crop_x = 0.045
            crop_y = 0.04
        else:
            crop_x = 0.03
            crop_y = 0.03

        dy = int(h * crop_y)
        dx = int(w * crop_x)

        if h - 2 * dy > 10 and w - 2 * dx > 10:
            return image[dy:h-dy, dx:w-dx]
        return image

    def _adjust_gamma(self, image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _smooth_1d(self, arr: np.ndarray, k: int = 5) -> np.ndarray:
        if arr.size == 0:
            return arr

        k = max(3, int(k))
        if k % 2 == 0:
            k += 1

        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(arr, kernel, mode="same")

    def _find_split_y_from_projection(self, proj: np.ndarray) -> int:
        """
        Find valley between top and bottom text peaks.
        """
        h = len(proj)

        if h <= 0:
            return 0

        if h < 24:
            return h // 2

        top_start = int(h * 0.05)
        top_end = int(h * 0.52)
        bot_start = int(h * 0.48)
        bot_end = int(h * 0.95)

        top_end = max(top_end, top_start + 1)
        bot_end = max(bot_end, bot_start + 1)

        top_region = proj[top_start:top_end]
        bot_region = proj[bot_start:bot_end]

        if top_region.size == 0 or bot_region.size == 0:
            return h // 2

        top_peak = top_start + int(np.argmax(top_region))
        bot_peak = bot_start + int(np.argmax(bot_region))

        if bot_peak <= top_peak:
            return h // 2

        # Search valley around the middle gap
        gap_start = max(top_peak + 1, int(h * 0.30))
        gap_end = min(bot_peak, int(h * 0.70))

        if gap_end <= gap_start:
            gap_start = top_peak + 1
            gap_end = bot_peak

        if gap_end <= gap_start:
            return h // 2

        valley_region = proj[gap_start:gap_end]
        split_y = gap_start + int(np.argmin(valley_region))

        return int(np.clip(split_y, int(h * 0.30), int(h * 0.70)))

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew image using Hough lines.
        Important: rotate by -median_angle.
        """
        try:
            gray = self._to_grayscale(image)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

            if lines is None or len(lines) == 0:
                return image

            angles = []

            for line in lines[:20]:
                rho, theta = line[0]
                angle = (theta * 180.0 / np.pi) - 90.0

                if -15.0 < angle < 15.0:
                    angles.append(angle)

            if not angles:
                return image

            median_angle = float(np.median(angles))

            if abs(median_angle) <= 0.5:
                return image

            h, w = image.shape[:2]
            center = (w // 2, h // 2)

            # Fix: rotate opposite direction
            M = cv2.getRotationMatrix2D(center, -median_angle, 1.0)

            return cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

        except Exception as e:
            logger.debug(f"Deskew failed: {e}")
            return image

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            gray,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=3,
        )

        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        return cv2.filter2D(gray, -1, kernel)