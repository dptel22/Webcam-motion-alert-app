"""Core tracking pipeline uniting CVIP preprocessing, DL detection/Re-ID, Kalman tracking, alerts, and metrics."""

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from src.alerts.emailing import AlertManager
from src.config import TrackerConfig, TrackingMode
from src.cvip.optical_flow import FlowSegment, OpticalFlowTracker
from src.cvip.preprocessing import CVIPPreprocessor
from src.dl.detector import Detection, YOLODetector
from src.dl.reid import ReIDExtractor
from src.eval.metrics import FrameMetrics, MetricsLogger
from src.tracking.tracker import MultiObjectTracker, Track, TrackState

_PALETTE: list[tuple[int, int, int]] = [
    (0, 210, 120),
    (255, 165, 0),
    (80, 180, 255),
    (220, 80, 220),
    (0, 220, 255),
    (120, 255, 120),
    (255, 100, 100),
    (180, 255, 80),
]


@dataclass
class PipelineResult:
    """Visual panels, active tracks, and telemetry produced for one frame."""

    annotated: np.ndarray
    clahe: np.ndarray
    mog2: np.ndarray
    flow: np.ndarray
    quad: np.ndarray
    tracks: list[Track]
    metrics: FrameMetrics


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def _draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    canvas = frame.copy()
    for track in tracks:
        color = _color_for_id(track.track_id)
        x1, y1, x2, y2 = np.round(track.bbox()).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"#{track.track_id} {track.class_name} {track.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )

        # Draw center-point trail
        pts = list(track.history)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 3, color, -1)

        # Draw velocity vector from bounding-box center
        cx = int(round(0.5 * (x1 + x2)))
        cy = int(round(0.5 * (y1 + y2)))
        vx, vy = track.kf.velocity()
        if abs(vx) > 0.25 or abs(vy) > 0.25:
            end_pt = (int(round(cx + vx * 5.0)), int(round(cy + vy * 5.0)))
            cv2.arrowedLine(canvas, (cx, cy), end_pt, (0, 255, 255), 2, tipLength=0.3)

    return canvas


def _draw_flow(frame: np.ndarray, vectors: list[FlowSegment]) -> np.ndarray:
    canvas = frame.copy()
    for start, end in vectors:
        cv2.arrowedLine(canvas, start, end, (0, 255, 180), 1, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(canvas, end, 2, (0, 200, 255), -1)
    return canvas


def _label_tile(tile: np.ndarray, text: str) -> np.ndarray:
    out = tile.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (6, 6), (14 + tw, 14 + th), (20, 20, 20), -1)
    cv2.putText(out, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 240, 245), 1, cv2.LINE_AA)
    return out


def _build_quad(
    annotated: np.ndarray,
    clahe: np.ndarray,
    mog2: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    h, w = annotated.shape[:2]
    if h < 2 or w < 2:
        return annotated.copy()
    tw = w // 2
    th = h // 2
    panels = [
        _label_tile(cv2.resize(annotated, (tw, th)), "Annotated"),
        _label_tile(cv2.resize(clahe, (w - tw, th)), "CLAHE"),
        _label_tile(cv2.resize(mog2, (tw, h - th)), "MOG2 Mask"),
        _label_tile(cv2.resize(flow, (w - tw, h - th)), "Optical Flow"),
    ]
    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))
    return np.vstack((top, bottom))


class TrackingPipeline:
    """Thread-safe multi-mode CVIP + Deep Learning tracking pipeline."""

    def __init__(
        self,
        config: TrackerConfig,
        detector=None,
        reid=None,
        alert_manager: AlertManager | None = None,
        metrics_logger: MetricsLogger | None = None,
    ) -> None:
        self.config = config
        self._lock = threading.RLock()

        self.preprocessor = CVIPPreprocessor(
            use_clahe=config.use_clahe,
            clahe_clip_limit=config.clahe_clip_limit,
            clahe_grid_size=config.clahe_grid_size,
            blur_kernel=config.blur_kernel,
            mog2_history=config.mog2_history,
            mog2_var_threshold=config.mog2_var_threshold,
            min_contour_area=config.min_contour_area,
        )
        self.flow_tracker = OpticalFlowTracker()
        self.tracker = MultiObjectTracker(
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_weight=config.iou_weight,
            reid_weight=config.reid_weight,
            max_iou_distance=config.max_iou_distance,
            max_cosine_distance=config.max_cosine_distance,
            ema_alpha=config.reid_ema_alpha,
            trail_length=config.trail_length,
        )
        self.detector = (
            detector
            if detector is not None
            else YOLODetector(
                model_name=config.detector_model,
                conf_threshold=config.conf_threshold,
                iou_threshold=config.iou_threshold,
                target_classes=config.target_classes,
            )
        )
        self.reid = reid if reid is not None else ReIDExtractor()
        self.alert_manager = (
            alert_manager
            if alert_manager is not None
            else AlertManager(
                snapshot_dir=config.snapshot_dir,
                cooldown_sec=config.alert_cooldown_sec,
                enabled=config.enable_alerts,
            )
        )
        self.metrics_logger = metrics_logger if metrics_logger is not None else MetricsLogger()

        self.frame_idx = 0
        self._prev_gray: np.ndarray | None = None
        self._start_time: float | None = None

    def set_mode(self, mode: TrackingMode | str) -> None:
        """Update active tracking mode safely."""
        with self._lock:
            self.config.mode = mode if isinstance(mode, TrackingMode) else TrackingMode(mode)

    def update_settings(self, **kwargs) -> None:
        """Update runtime configuration flags safely."""
        with self._lock:
            if "mode" in kwargs and kwargs["mode"] is not None:
                self.set_mode(kwargs["mode"])
            if "use_clahe" in kwargs and kwargs["use_clahe"] is not None:
                val = bool(kwargs["use_clahe"])
                self.config.use_clahe = val
                self.preprocessor.use_clahe = val
            if "use_reid" in kwargs and kwargs["use_reid"] is not None:
                self.config.use_reid = bool(kwargs["use_reid"])
            if "detect_interval" in kwargs and kwargs["detect_interval"] is not None:
                self.config.detect_interval = max(1, int(kwargs["detect_interval"]))
            if "enable_alerts" in kwargs and kwargs["enable_alerts"] is not None:
                val = bool(kwargs["enable_alerts"])
                self.config.enable_alerts = val
                self.alert_manager.enabled = val

    def reset(self) -> None:
        """Reset tracker state, background model, metrics, and frame counter."""
        with self._lock:
            self.tracker.reset()
            self.preprocessor.reset_background()
            self.metrics_logger.clear()
            self._prev_gray = None
            self.frame_idx = 0
            self._start_time = None

    def _compute_vis_flow(self, curr_gray: np.ndarray, boxes: list[np.ndarray]) -> list[FlowSegment]:
        if self._prev_gray is None:
            return []
        if boxes:
            return self.flow_tracker.estimate_box_shifts(self._prev_gray, curr_gray, boxes).vectors
        return self.flow_tracker.compute_global_flow_vectors(self._prev_gray, curr_gray)

    def _maybe_trigger_alert(self, annotated: np.ndarray, active_tracks: list[Track]) -> None:
        if not self.config.enable_alerts or not active_tracks:
            return
        target_classes = set(self.config.alert_classes)
        matched = (
            any(t.class_name in target_classes for t in active_tracks)
            if target_classes
            else True
        )
        if matched:
            self.alert_manager.trigger(annotated)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run one full frame through preprocessing, detection/flow, tracking, and telemetry."""
        if frame is None or frame.size == 0 or frame.ndim != 3:
            raise ValueError("Expected non-empty 3-channel BGR frame")
        with self._lock:
            t0 = time.perf_counter()
            if self._start_time is None:
                self._start_time = t0

            prep = self.preprocessor.process(frame, extract_motion=True)
            curr_gray = cv2.cvtColor(prep.denoised, cv2.COLOR_BGR2GRAY)

            mode = self.config.mode
            flow_vectors: list[FlowSegment] = []
            detections_count = 0

            if mode == TrackingMode.CVIP_MOG2:
                detections = [
                    Detection(bbox=box, confidence=1.0, class_id=0, class_name="motion")
                    for box in prep.motion_boxes
                ]
                detections_count = len(detections)
                active_tracks = self.tracker.step(detections, mode=TrackingMode.CVIP_MOG2)
                flow_vectors = self._compute_vis_flow(
                    curr_gray, [t.bbox() for t in active_tracks] or prep.motion_boxes
                )

            elif mode == TrackingMode.DL_SORT:
                detections = self.detector.detect(frame)
                detections_count = len(detections)
                active_tracks = self.tracker.step(detections, mode=TrackingMode.DL_SORT)
                flow_vectors = self._compute_vis_flow(curr_gray, [t.bbox() for t in active_tracks])

            else:  # TrackingMode.HYBRID
                interval = max(1, int(self.config.detect_interval))
                tracked_by_id = self.tracker.tracked_boxes_by_id(include_tentative=True)
                use_flow_step = (
                    self._prev_gray is not None
                    and (self.frame_idx % interval != 0)
                    and bool(self.tracker.active_tracks())
                )
                if use_flow_step:
                    flow_res = self.flow_tracker.estimate_box_shifts(
                        self._prev_gray,
                        curr_gray,
                        list(tracked_by_id.values()),
                    )
                    flow_boxes_by_id = dict(zip(tracked_by_id.keys(), flow_res.shifted_boxes))
                    flow_vectors = flow_res.vectors
                    active_tracks = self.tracker.step_with_flow(flow_boxes_by_id)
                else:
                    detections = self.detector.detect(prep.enhanced)
                    detections_count = len(detections)
                    if detections and self.config.use_reid:
                        embeddings = self.reid.extract(prep.enhanced, [d.bbox for d in detections])
                        for det, emb in zip(detections, embeddings):
                            det.embedding = emb
                    active_tracks = self.tracker.step(detections, mode=TrackingMode.HYBRID)
                    flow_vectors = self._compute_vis_flow(curr_gray, [t.bbox() for t in active_tracks])

            self._prev_gray = curr_gray

            annotated = _draw_tracks(frame, active_tracks)
            clahe = prep.enhanced.copy()
            mog2 = cv2.cvtColor(prep.fg_mask, cv2.COLOR_GRAY2BGR)
            flow = _draw_flow(prep.denoised, flow_vectors)
            quad = _build_quad(annotated, clahe, mog2, flow)

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            fps = 1000.0 / max(latency_ms, 1e-3)
            metric = FrameMetrics(
                frame_idx=self.frame_idx,
                timestamp_sec=t1 - self._start_time,
                mode=mode.value,
                fps=fps,
                latency_ms=latency_ms,
                active_tracks=len(active_tracks),
                detections_count=detections_count,
                total_tracks_created=self.tracker.total_tracks_created,
            )
            self.metrics_logger.record(metric)
            self.frame_idx += 1

            self._maybe_trigger_alert(annotated, active_tracks)

            return PipelineResult(
                annotated=annotated,
                clahe=clahe,
                mog2=mog2,
                flow=flow,
                quad=quad,
                tracks=active_tracks,
                metrics=metric,
            )

    def get_track_rows(self) -> list[dict]:
        """Return serialized track rows for active and lost tracks."""
        with self._lock:
            rows: list[dict] = []
            for track in self.tracker.tracks:
                if track.state not in (TrackState.CONFIRMED, TrackState.LOST):
                    continue
                vx, vy = track.kf.velocity()
                rows.append(
                    {
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "confidence": round(float(track.confidence), 4),
                        "vx": round(float(vx), 2),
                        "vy": round(float(vy), 2),
                        "age": track.age,
                        "hits": track.hits,
                        "state": track.state.value,
                    }
                )
            return rows


def generate_sample_video(
    output_path: str | Path,
    num_frames: int = 60,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Generate a synthetic MP4 with two textured moving objects crossing paths."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, 24.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {dest}")

    # Pre-render static textured background
    bg = np.full((height, width, 3), 32, dtype=np.uint8)
    for gx in range(0, width, 40):
        cv2.line(bg, (gx, 0), (gx, height), (45, 45, 45), 1)
    for gy in range(0, height, 40):
        cv2.line(bg, (0, gy), (width, gy), (45, 45, 45), 1)

    box_w = max(36, width // 10)
    box_h = max(48, height // 4)

    def _paint_target(canvas: np.ndarray, x: int, y: int, primary: tuple[int, int, int], accent: tuple[int, int, int]) -> None:
        cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), primary, -1)
        step = max(8, box_w // 5)
        for dy in range(4, box_h - 4, step):
            for dx in range(4, box_w - 4, step):
                col = accent if ((dx + dy) // step) % 2 == 0 else (20, 20, 20)
                cv2.rectangle(canvas, (x + dx, y + dy), (x + dx + step - 2, y + dy + step - 2), col, -1)

    try:
        denom = max(1, num_frames - 1)
        for idx in range(num_frames):
            t = idx / denom
            frame = bg.copy()

            x1 = int(20 + t * max(1, width - box_w - 40))
            y1 = int(20 + t * max(1, height - box_h - 40))
            _paint_target(frame, x1, y1, (210, 210, 210), (40, 120, 240))

            x2 = int(max(20, width - box_w - 20) - t * max(1, width - box_w - 40))
            y2 = int(max(20, height - box_h - 20) - t * max(1, height - box_h - 40))
            _paint_target(frame, x2, y2, (190, 230, 190), (220, 70, 70))

            writer.write(frame)
    finally:
        writer.release()

    return dest
