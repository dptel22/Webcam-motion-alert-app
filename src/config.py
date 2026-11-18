"""Application configuration and tracking mode definitions."""

from dataclasses import dataclass, field
from enum import Enum


class TrackingMode(str, Enum):
    """Selectable tracking pipeline modes for live ablation and comparison."""

    CVIP_MOG2 = "cvip"       # CVIP project: CLAHE + MOG2 + contours + Kalman/Hungarian IoU + LK flow (no DL)
    DL_SORT = "dl_sort"      # DL project: YOLOv8 + MobileNetV3 Re-ID + Kalman/Hungarian (no CVIP stages)
    HYBRID = "hybrid"        # Combined: CLAHE + Optical Flow + YOLO + Kalman + Re-ID

    @classmethod
    def cycle_order(cls) -> list["TrackingMode"]:
        """Return the ordered list of modes for live hotkey cycling."""
        return [cls.HYBRID, cls.DL_SORT, cls.CVIP_MOG2]

    def next_mode(self) -> "TrackingMode":
        """Return the next mode in the cycle order."""
        order = self.cycle_order()
        idx = order.index(self)
        return order[(idx + 1) % len(order)]


@dataclass
class TrackerConfig:
    """Configuration for CVIP preprocessing, DL inference, tracking, alerts, and logging."""

    # Input / execution
    source: str = "0"
    mode: TrackingMode = TrackingMode.HYBRID
    debug_view: bool = False
    save_video: str | None = None
    metrics_dir: str = "outputs"
    max_frames: int | None = None
    max_frame_width: int = 960  # wider inputs are downscaled before processing

    # CVIP preprocessing & background subtraction
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple[int, int] = (8, 8)
    blur_kernel: int = 5
    mog2_history: int = 300
    mog2_var_threshold: float = 36.0
    min_contour_area: int = 1800

    # Deep Learning detector & Re-ID
    detector_model: str = "yolov8n.pt"
    detector_imgsz: int = 416
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    detect_interval: int = 2
    use_reid: bool = True
    reid_ema_alpha: float = 0.8
    target_classes: list[int] | None = None

    # Kalman + Hungarian Multi-Object Tracker
    max_age: int = 30
    min_hits: int = 3
    iou_weight: float = 0.5
    reid_weight: float = 0.5
    max_iou_distance: float = 0.75
    max_cosine_distance: float = 0.45
    trail_length: int = 30

    # Alerting (disabled by default during testing/demos)
    enable_alerts: bool = False
    alert_cooldown_sec: float = 30.0
    snapshot_dir: str = "images"
    alert_classes: list[str] = field(default_factory=lambda: ["person"])
