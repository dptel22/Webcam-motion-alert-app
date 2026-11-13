"""8-state Constant-Velocity Kalman Filter for bounding-box tracking."""

import numpy as np


def bbox_to_z(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to measurement vector [cx, cy, aspect_ratio, height]."""
    w = max(1.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
    h = max(1.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
    cx = float(bbox_xyxy[0]) + 0.5 * w
    cy = float(bbox_xyxy[1]) + 0.5 * h
    aspect = w / h
    return np.array([[cx], [cy], [aspect], [h]], dtype=np.float32)


def x_to_bbox(state: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, aspect_ratio, height, ...] to [x1, y1, x2, y2]."""
    flat = state.reshape(-1)
    cx, cy, aspect, h = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
    h = max(1.0, h)
    aspect = max(0.05, aspect)
    w = aspect * h
    return np.array(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        dtype=np.float32,
    )


class KalmanBoxTracker:
    """Constant-velocity Kalman Filter over state [cx, cy, a, h, vx, vy, va, vh]."""

    def __init__(self, initial_bbox: np.ndarray) -> None:
        # 8-D state, 4-D measurement
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0

        self._H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self._H[i, i] = 1.0

        self._x = np.zeros((8, 1), dtype=np.float32)
        self._x[:4] = bbox_to_z(initial_bbox)

        # Initial uncertainty: higher on unobserved velocities
        self._P = np.diag([10.0, 10.0, 1e-2, 10.0, 100.0, 100.0, 1e-3, 100.0]).astype(np.float32)
        # Process noise
        self._Q = np.diag([1.0, 1.0, 1e-3, 1.0, 0.05, 0.05, 1e-4, 0.05]).astype(np.float32)
        # Measurement noise
        self._R = np.diag([1.0, 1.0, 1e-2, 1.0]).astype(np.float32)

    def predict(self) -> np.ndarray:
        """Advance the state by one frame and return predicted [x1, y1, x2, y2]."""
        # Guard against negative height/aspect growth from velocity extrapolation
        if (self._x[3, 0] + self._x[7, 0]) <= 1.0:
            self._x[7, 0] = 0.0
        if (self._x[2, 0] + self._x[6, 0]) <= 0.05:
            self._x[6, 0] = 0.0

        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self.current_bbox()

    def update(self, bbox_xyxy: np.ndarray, measurement_scale: float = 1.0) -> np.ndarray:
        """Update the Kalman state with a new [x1, y1, x2, y2] observation."""
        z = bbox_to_z(bbox_xyxy)
        y = z - (self._H @ self._x)
        R = self._R * measurement_scale
        S = self._H @ self._P @ self._H.T + R
        K = self._P @ self._H.T @ np.linalg.inv(S)

        self._x = self._x + (K @ y)
        identity = np.eye(8, dtype=np.float32)
        self._P = (identity - K @ self._H) @ self._P
        return self.current_bbox()

    def current_bbox(self) -> np.ndarray:
        """Return current state estimate as [x1, y1, x2, y2]."""
        return x_to_bbox(self._x)

    def velocity(self) -> tuple[float, float]:
        """Return estimated center velocity (vx, vy) in pixels/frame."""
        return float(self._x[4, 0]), float(self._x[5, 0])
