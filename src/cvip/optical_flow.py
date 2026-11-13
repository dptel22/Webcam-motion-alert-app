"""Sparse Lucas-Kanade Optical Flow for inter-frame motion estimation and debug visualization."""

from dataclasses import dataclass
import cv2
import numpy as np


FlowSegment = tuple[tuple[int, int], tuple[int, int]]


@dataclass
class FlowResult:
    """Shifted bounding boxes and flow vector segments for visualization."""

    shifted_boxes: list[np.ndarray]
    vectors: list[FlowSegment]


class OpticalFlowTracker:
    """Tracks Shi-Tomasi corners across frames using pyramidal Lucas-Kanade Optical Flow."""

    def __init__(
        self,
        max_corners_per_box: int = 25,
        quality_level: float = 0.05,
        min_distance: int = 5,
        win_size: tuple[int, int] = (15, 15),
        max_level: int = 2,
    ) -> None:
        self.max_corners_per_box = max_corners_per_box
        self.quality_level = quality_level
        self.min_distance = min_distance
        self._lk_params = {
            "winSize": win_size,
            "maxLevel": max_level,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        }

    @staticmethod
    def _clamp_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(np.float32)
        bw = max(4.0, x2 - x1)
        bh = max(4.0, y2 - y1)
        x1 = float(np.clip(x1, 0.0, max(0.0, width - bw)))
        y1 = float(np.clip(y1, 0.0, max(0.0, height - bh)))
        return np.array([x1, y1, min(float(width), x1 + bw), min(float(height), y1 + bh)], dtype=np.float32)

    def _track_points(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track points with forward LK flow; return matched (old_pts, new_pts)."""
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            points,
            None,
            **self._lk_params,
        )
        if next_pts is None or status is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        valid = status.reshape(-1) == 1
        return points.reshape(-1, 2)[valid], next_pts.reshape(-1, 2)[valid]

    def estimate_box_shifts(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        boxes_xyxy: list[np.ndarray],
    ) -> FlowResult:
        """Shift each bounding box by the median optical flow displacement inside its ROI."""
        height, width = curr_gray.shape[:2]
        shifted_boxes: list[np.ndarray] = []
        vectors: list[FlowSegment] = []

        for box in boxes_xyxy:
            x1, y1, x2, y2 = np.round(box).astype(int)
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            roi = prev_gray[y1:y2, x1:x2]
            if roi.shape[0] < 6 or roi.shape[1] < 6:
                shifted_boxes.append(box.astype(np.float32))
                continue

            corners = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=self.max_corners_per_box,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
            )
            if corners is None or len(corners) == 0:
                shifted_boxes.append(box.astype(np.float32))
                continue

            corners[:, 0, 0] += x1
            corners[:, 0, 1] += y1
            old_pts, new_pts = self._track_points(prev_gray, curr_gray, corners)

            if len(old_pts) < 2:
                shifted_boxes.append(box.astype(np.float32))
                continue

            deltas = new_pts - old_pts
            # Median displacement resists background-corner outliers near box edges.
            dx, dy = np.median(deltas, axis=0)
            shifted = box.astype(np.float32) + np.array([dx, dy, dx, dy], dtype=np.float32)
            shifted_boxes.append(self._clamp_box(shifted, width, height))

            for (ox, oy), (nx, ny) in zip(old_pts, new_pts):
                vectors.append(((int(ox), int(oy)), (int(nx), int(ny))))

        return FlowResult(shifted_boxes=shifted_boxes, vectors=vectors)

    def compute_global_flow_vectors(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        max_corners: int = 80,
    ) -> list[FlowSegment]:
        """Compute global sparse optical flow vectors for debug visualization when no boxes exist."""
        corners = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=max_corners,
            qualityLevel=0.05,
            minDistance=15,
        )
        old_pts, new_pts = self._track_points(prev_gray, curr_gray, corners)
        vectors: list[FlowSegment] = []
        for (ox, oy), (nx, ny) in zip(old_pts, new_pts):
            if np.hypot(nx - ox, ny - oy) >= 1.0:
                vectors.append(((int(ox), int(oy)), (int(nx), int(ny))))
        return vectors
