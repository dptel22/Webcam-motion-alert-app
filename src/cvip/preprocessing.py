"""CVIP image preprocessing, lighting normalization, and adaptive background subtraction."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PreprocessedFrame:
    """Outputs produced by the classical CVIP preprocessing stage."""

    enhanced: np.ndarray
    denoised: np.ndarray
    fg_mask: np.ndarray
    motion_boxes: list[np.ndarray]


class CVIPPreprocessor:
    """Applies CLAHE contrast normalization, denoising, and MOG2 foreground segmentation."""

    def __init__(
        self,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple[int, int] = (8, 8),
        blur_kernel: int = 5,
        mog2_history: int = 300,
        mog2_var_threshold: float = 36.0,
        min_contour_area: int = 1800,
    ) -> None:
        self.use_clahe = use_clahe
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.min_contour_area = min_contour_area

        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size,
        )
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var_threshold,
            detectShadows=True,
        )
        self._morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def reset_background(self) -> None:
        """Reset the adaptive MOG2 background model."""
        history = self._bg_subtractor.getHistory()
        var_thresh = self._bg_subtractor.getVarThreshold()
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_thresh,
            detectShadows=True,
        )

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Normalize uneven lighting by applying CLAHE to the LAB luminance channel."""
        if not self.use_clahe:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        merged = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """Reduce high-frequency sensor noise with Gaussian smoothing."""
        if self.blur_kernel <= 1:
            return frame
        return cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)

    def extract_foreground_mask(self, denoised_frame: np.ndarray) -> np.ndarray:
        """Segment moving foreground objects using MOG2 and morphological filtering."""
        raw_mask = self._bg_subtractor.apply(denoised_frame)
        # MOG2 marks shadows as 127 and definite foreground as 255.
        _, binary_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self._morph_open_kernel)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._morph_close_kernel)
        return cleaned

    def detect_motion_boxes(self, fg_mask: np.ndarray) -> list[np.ndarray]:
        """Extract bounding boxes [x1, y1, x2, y2] from significant foreground contours."""
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[np.ndarray] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
        return boxes

    def process(self, frame: np.ndarray, extract_motion: bool = True) -> PreprocessedFrame:
        """Run full CVIP preprocessing and optional MOG2 motion segmentation."""
        enhanced = self.apply_clahe(frame)
        denoised = self.denoise(enhanced)
        if extract_motion:
            fg_mask = self.extract_foreground_mask(denoised)
            motion_boxes = self.detect_motion_boxes(fg_mask)
        else:
            fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            motion_boxes = []
        return PreprocessedFrame(
            enhanced=enhanced,
            denoised=denoised,
            fg_mask=fg_mask,
            motion_boxes=motion_boxes,
        )
