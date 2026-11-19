"""Core tracking pipeline uniting CVIP preprocessing, DL detection/Re-ID, Kalman tracking, alerts, and metrics.

Each tracking mode is a project preset with strict stage gating:

* ``cvip``    - CVIP project: CLAHE -> MOG2 -> contours -> Kalman/Hungarian IoU -> Lucas-Kanade flow.
                Never touches (or even constructs) the YOLO detector or the Re-ID network.
* ``dl_sort`` - DL project: YOLOv8 on the raw frame -> MobileNetV3 Re-ID (optional) -> Kalman/Hungarian
                with fused IoU + appearance cost. Never runs CLAHE, MOG2, contours, or optical flow.
* ``hybrid``  - Combined: CLAHE -> YOLO + Re-ID on keyframes, Lucas-Kanade flow between them.

Visual panels are rendered lazily (only the view a client asks for is ever drawn), from immutable
per-frame snapshots, so drawing and JPEG encoding never need the pipeline lock.
"""

from collections import deque
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

# Pipeline stages (= selectable views) per mode, in pipeline order. "quad" is always allowed too.
MODE_STAGES: dict[TrackingMode, tuple[str, ...]] = {
    TrackingMode.CVIP_MOG2: ("input", "clahe", "mog2", "flow", "annotated"),
    TrackingMode.DL_SORT: ("input", "detections", "annotated"),
    TrackingMode.HYBRID: ("input", "clahe", "flow", "detections", "annotated"),
}
ALL_VIEWS: tuple[str, ...] = ("input", "clahe", "mog2", "flow", "detections", "annotated", "quad")
STAGE_TIMING_KEYS: tuple[str, ...] = ("preprocess", "mog2", "flow", "detect", "reid", "track", "render")
VIEW_LABELS: dict[str, str] = {
    "input": "Input",
    "clahe": "CLAHE",
    "mog2": "MOG2 Mask",
    "flow": "Optical Flow",
    "detections": "YOLO Detections",
    "annotated": "Annotated",
    "quad": "Quad",
}


def stages_for_mode(mode: TrackingMode | str) -> list[str]:
    """Return the ordered list of pipeline stages (views) produced by a mode."""
    return list(MODE_STAGES[TrackingMode(mode)])


def views_for_mode(mode: TrackingMode | str) -> list[str]:
    """Return every view a client may request in a mode (its stages plus "quad")."""
    return stages_for_mode(mode) + ["quad"]


@dataclass(frozen=True)
class TrackDraw:
    """Immutable per-frame drawing snapshot of a track (Track objects keep mutating)."""

    track_id: int
    bbox: tuple[int, int, int, int]
    label: str
    history: tuple[tuple[int, int], ...]
    velocity: tuple[float, float]


def _snapshot_tracks(tracks: list[Track]) -> list[TrackDraw]:
    out: list[TrackDraw] = []
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in np.round(t.bbox()))
        vx, vy = t.kf.velocity()
        out.append(
            TrackDraw(
                track_id=t.track_id,
                bbox=(x1, y1, x2, y2),
                label=f"#{t.track_id} {t.class_name} {t.confidence:.2f}",
                history=tuple(t.history),
                velocity=(float(vx), float(vy)),
            )
        )
    return out


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[(track_id - 1) % len(_PALETTE)]


def _put_label(canvas: np.ndarray, text: str, x: int, y: int, bg: tuple[int, int, int], fg=(15, 15, 15)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(th + 6, y)
    cv2.rectangle(canvas, (x, ty - th - 6), (x + tw + 8, ty), bg, -1)
    cv2.putText(canvas, text, (x + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA)


def _draw_tracks(frame: np.ndarray, tracks: list[TrackDraw]) -> np.ndarray:
    canvas = frame.copy()
    for track in tracks:
        color = _color_for_id(track.track_id)
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _put_label(canvas, track.label, x1, y1, color)

        # Center-point trail (one polyline call instead of N line calls)
        if len(track.history) > 1:
            pts = np.asarray(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        if track.history:
            cv2.circle(canvas, track.history[-1], 3, color, -1)

        # Velocity vector from bounding-box center
        vx, vy = track.velocity
        if abs(vx) > 0.25 or abs(vy) > 0.25:
            cx = int(round(0.5 * (x1 + x2)))
            cy = int(round(0.5 * (y1 + y2)))
            end_pt = (int(round(cx + vx * 5.0)), int(round(cy + vy * 5.0)))
            cv2.arrowedLine(canvas, (cx, cy), end_pt, (0, 255, 255), 2, tipLength=0.3)
    return canvas


def _draw_detections(frame: np.ndarray, detections: list[Detection], stale_frames: int | None) -> np.ndarray:
    """Draw raw detector output (class + confidence) before any tracking/association."""
    canvas = frame.copy()
    color = (0, 140, 255)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in np.round(det.bbox))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _put_label(canvas, f"{det.class_name} {det.confidence:.2f}", x1, y1, color)
    if stale_frames is None:
        header = "YOLO: no detection run yet"
    elif stale_frames > 0:
        header = f"YOLO: {len(detections)} det ({stale_frames} frame(s) ago)"
    else:
        header = f"YOLO: {len(detections)} det"
    _put_label(canvas, header, 6, 22, (20, 20, 20), fg=(235, 240, 245))
    return canvas


def _draw_flow(frame: np.ndarray, vectors: list[FlowSegment]) -> np.ndarray:
    canvas = frame.copy()
    if vectors:
        lines = [np.array([s, e], dtype=np.int32) for s, e in vectors]
        cv2.polylines(canvas, lines, False, (0, 255, 180), 1, cv2.LINE_AA)
        for _start, end in vectors:
            cv2.circle(canvas, end, 2, (0, 200, 255), -1)
    return canvas


def _label_tile(tile: np.ndarray, text: str) -> np.ndarray:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(tile, (6, 6), (14 + tw, 14 + th), (20, 20, 20), -1)
    cv2.putText(tile, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 240, 245), 1, cv2.LINE_AA)
    return tile


def _build_quad(tiles: list[tuple[str, np.ndarray]], size: tuple[int, int]) -> np.ndarray:
    """Compose up to four labelled tiles into a 2x2 grid of the given (h, w)."""
    h, w = size
    canvas = np.full((h, w, 3), 18, dtype=np.uint8)
    if h < 2 or w < 2 or not tiles:
        return canvas
    tw, th = w // 2, h // 2
    slots = [(0, 0, tw, th), (tw, 0, w - tw, th), (0, th, tw, h - th), (tw, th, w - tw, h - th)]
    for (label, img), (x, y, sw, sh) in zip(tiles[:4], slots):
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        canvas[y : y + sh, x : x + sw] = _label_tile(small, label)
    return canvas


class PipelineResult:
    """Telemetry plus lazily rendered visual panels for one processed frame.

    Only data needed to draw a panel is stored; the panel itself is rendered (once, cached)
    the first time a consumer asks for it. Panels not produced by the frame's mode are None.
    """

    def __init__(
        self,
        *,
        mode: TrackingMode,
        frame: np.ndarray,
        tracks: list[Track],
        metrics: FrameMetrics,
        stage_ms: dict[str, float],
        track_draws: list[TrackDraw],
        enhanced: np.ndarray | None = None,
        denoised: np.ndarray | None = None,
        fg_mask: np.ndarray | None = None,
        flow_vectors: list[FlowSegment] | None = None,
        detections: list[Detection] | None = None,
        detect_frame: np.ndarray | None = None,
        detections_stale: int | None = None,
    ) -> None:
        self.mode = mode
        self.frame = frame
        self.tracks = tracks
        self.metrics = metrics
        self.stage_ms = stage_ms
        self.track_draws = track_draws
        self.enhanced = enhanced
        self.denoised = denoised
        self.fg_mask = fg_mask
        self.flow_vectors = flow_vectors or []
        self.detections = detections or []
        self.detect_frame = detect_frame
        self.detections_stale = detections_stale
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def stages(self) -> list[str]:
        return stages_for_mode(self.mode)

    def has_view(self, view: str) -> bool:
        return view == "quad" or view in MODE_STAGES[self.mode]

    def panel(self, view: str) -> np.ndarray | None:
        """Return the rendered BGR panel for a view (cached), or None if the mode lacks it."""
        if not self.has_view(view):
            return None
        with self._lock:
            cached = self._cache.get(view)
        if cached is not None:
            return cached
        t0 = time.perf_counter()
        img = self._render(view)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            existing = self._cache.get(view)
            if existing is not None:
                return existing
            self._cache[view] = img
            if view not in ("input", "quad"):
                self.stage_ms["render"] = round(self.stage_ms.get("render", 0.0) + dt_ms, 3)
        return img

    def _render(self, view: str) -> np.ndarray:
        if view == "input":
            return self.frame
        if view == "annotated":
            return _draw_tracks(self.frame, self.track_draws)
        if view == "clahe":
            return self.enhanced
        if view == "mog2":
            return cv2.cvtColor(self.fg_mask, cv2.COLOR_GRAY2BGR)
        if view == "flow":
            return _draw_flow(self.denoised, self.flow_vectors)
        if view == "detections":
            base = self.detect_frame if self.detect_frame is not None else self.frame
            return _draw_detections(base, self.detections, self.detections_stale)
        if view == "quad":
            stages = self.stages
            order = ["annotated"] + [s for s in stages if s not in ("annotated", "input")]
            if len(order) < 4 and "input" in stages:
                order.append("input")
            tiles = [(VIEW_LABELS[s], self.panel(s)) for s in order[:4]]
            return _build_quad(tiles, self.frame.shape[:2])
        raise ValueError(f"Unknown view: {view}")

    # Attribute-style access kept for the CLI runner and existing callers.
    @property
    def input(self) -> np.ndarray:
        return self.frame

    @property
    def annotated(self) -> np.ndarray:
        return self.panel("annotated")

    @property
    def clahe(self) -> np.ndarray | None:
        return self.panel("clahe")

    @property
    def mog2(self) -> np.ndarray | None:
        return self.panel("mog2")

    @property
    def flow(self) -> np.ndarray | None:
        return self.panel("flow")

    @property
    def quad(self) -> np.ndarray:
        return self.panel("quad")


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


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
        # DL components are created lazily on first use so the CVIP mode never builds them.
        self._detector = detector
        self._reid = reid
        self._detector_used = False
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
        self._frame_times: deque[float] = deque(maxlen=15)
        self._last_detections: list[Detection] = []
        self._last_detect_idx: int | None = None
        self.last_stage_ms: dict[str, float] = {}

    # ------------------------------------------------------------------ DL components
    @property
    def detector(self):
        if self._detector is None:
            self._detector = YOLODetector(
                model_name=self.config.detector_model,
                conf_threshold=self.config.conf_threshold,
                iou_threshold=self.config.iou_threshold,
                target_classes=self.config.target_classes,
                imgsz=self.config.detector_imgsz,
            )
        return self._detector

    @detector.setter
    def detector(self, value) -> None:
        self._detector = value
        self._detector_used = False

    @property
    def reid(self):
        if self._reid is None:
            from src.dl.reid import ReIDExtractor  # lazy: imports torch

            self._reid = ReIDExtractor()
        return self._reid

    @reid.setter
    def reid(self, value) -> None:
        self._reid = value

    @property
    def detector_loaded(self) -> bool:
        """True once detector weights are in memory (or an injected detector has been used)."""
        det = self._detector
        if det is None:
            return False
        loaded = getattr(det, "is_loaded", None)
        if isinstance(loaded, bool):
            return loaded
        return self._detector_used

    # ------------------------------------------------------------------ settings
    def set_mode(self, mode: TrackingMode | str) -> None:
        """Switch the active mode, resetting state that must not leak across projects."""
        with self._lock:
            new = mode if isinstance(mode, TrackingMode) else TrackingMode(mode)
            old = self.config.mode
            if new == old:
                return
            self.config.mode = new
            # Previous gray frame may come from a different preprocessing chain.
            self._prev_gray = None
            self._last_detections = []
            self._last_detect_idx = None
            if TrackingMode.CVIP_MOG2 in (old, new):
                # Motion-blob tracks and YOLO tracks are not interchangeable.
                self.tracker.reset()
            if new == TrackingMode.CVIP_MOG2:
                # MOG2 is frozen outside CVIP mode; start from a fresh background model.
                self.preprocessor.reset_background()

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
            self._frame_times.clear()
            self._last_detections = []
            self._last_detect_idx = None
            self.last_stage_ms = {}

    # ------------------------------------------------------------------ helpers
    def _limit_width(self, frame: np.ndarray) -> np.ndarray:
        max_w = int(self.config.max_frame_width or 0)
        h, w = frame.shape[:2]
        if max_w <= 0 or w <= max_w:
            return frame
        new_h = max(1, int(round(h * max_w / float(w))))
        return cv2.resize(frame, (max_w, new_h), interpolation=cv2.INTER_AREA)

    def _compute_vis_flow(self, curr_gray: np.ndarray, boxes: list[np.ndarray]) -> list[FlowSegment]:
        if self._prev_gray is None or self._prev_gray.shape != curr_gray.shape:
            return []
        if boxes:
            return self.flow_tracker.estimate_box_shifts(self._prev_gray, curr_gray, boxes).vectors
        return self.flow_tracker.compute_global_flow_vectors(self._prev_gray, curr_gray)

    def _preprocess(self, frame: np.ndarray, stage_ms: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = time.perf_counter()
        enhanced = self.preprocessor.apply_clahe(frame)
        denoised = self.preprocessor.denoise(enhanced)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        stage_ms["preprocess"] = _ms_since(t)
        return enhanced, denoised, gray

    def _detect(self, image: np.ndarray, stage_ms: dict[str, float]) -> list[Detection]:
        t = time.perf_counter()
        detections = self.detector.detect(image)
        self._detector_used = True
        stage_ms["detect"] = _ms_since(t)
        if detections and self.config.use_reid:
            t = time.perf_counter()
            embeddings = self.reid.extract(image, [d.bbox for d in detections])
            for det, emb in zip(detections, embeddings):
                det.embedding = emb
            stage_ms["reid"] = _ms_since(t)
        self._last_detections = detections
        self._last_detect_idx = self.frame_idx
        return detections

    def _detections_view_data(self) -> dict:
        stale = None if self._last_detect_idx is None else self.frame_idx - self._last_detect_idx
        return {"detections": list(self._last_detections), "detections_stale": stale}

    # ------------------------------------------------------------------ per-mode stages
    def _run_cvip(self, frame: np.ndarray, stage_ms: dict[str, float]) -> dict:
        enhanced, denoised, gray = self._preprocess(frame, stage_ms)

        t = time.perf_counter()
        fg_mask = self.preprocessor.extract_foreground_mask(denoised)
        motion_boxes = self.preprocessor.detect_motion_boxes(fg_mask)
        stage_ms["mog2"] = _ms_since(t)

        t = time.perf_counter()
        detections = [
            Detection(bbox=box, confidence=1.0, class_id=0, class_name="motion") for box in motion_boxes
        ]
        active_tracks = self.tracker.step(detections, mode=TrackingMode.CVIP_MOG2)
        stage_ms["track"] = _ms_since(t)

        t = time.perf_counter()
        flow_vectors = self._compute_vis_flow(gray, [tr.bbox() for tr in active_tracks] or motion_boxes)
        stage_ms["flow"] = _ms_since(t)
        self._prev_gray = gray

        return {
            "tracks": active_tracks,
            "detections_count": len(detections),
            "enhanced": enhanced,
            "denoised": denoised,
            "fg_mask": fg_mask,
            "flow_vectors": flow_vectors,
        }

    def _run_dl_sort(self, frame: np.ndarray, stage_ms: dict[str, float]) -> dict:
        interval = max(1, int(self.config.detect_interval))
        detections_count = 0
        if self.frame_idx % interval == 0:
            detections = self._detect(frame, stage_ms)
            detections_count = len(detections)
            t = time.perf_counter()
            active_tracks = self.tracker.step(detections, mode=TrackingMode.DL_SORT)
        else:
            # Non-detection frame: cheap constant-velocity Kalman prediction only.
            t = time.perf_counter()
            active_tracks = self.tracker.step_predict()
        stage_ms["track"] = _ms_since(t)
        return {
            "tracks": active_tracks,
            "detections_count": detections_count,
            "detect_frame": frame,
            **self._detections_view_data(),
        }

    def _run_hybrid(self, frame: np.ndarray, stage_ms: dict[str, float]) -> dict:
        enhanced, denoised, gray = self._preprocess(frame, stage_ms)
        interval = max(1, int(self.config.detect_interval))
        detections_count = 0
        use_flow_step = (
            self._prev_gray is not None
            and self._prev_gray.shape == gray.shape
            and (self.frame_idx % interval != 0)
            and bool(self.tracker.active_tracks())
        )
        if use_flow_step:
            tracked_by_id = self.tracker.tracked_boxes_by_id(include_tentative=True)
            t = time.perf_counter()
            flow_res = self.flow_tracker.estimate_box_shifts(self._prev_gray, gray, list(tracked_by_id.values()))
            stage_ms["flow"] = _ms_since(t)
            t = time.perf_counter()
            active_tracks = self.tracker.step_with_flow(dict(zip(tracked_by_id.keys(), flow_res.shifted_boxes)))
            stage_ms["track"] = _ms_since(t)
            flow_vectors = flow_res.vectors
        else:
            detections = self._detect(enhanced, stage_ms)
            detections_count = len(detections)
            t = time.perf_counter()
            active_tracks = self.tracker.step(detections, mode=TrackingMode.HYBRID)
            stage_ms["track"] = _ms_since(t)
            t = time.perf_counter()
            flow_vectors = self._compute_vis_flow(gray, [tr.bbox() for tr in active_tracks])
            stage_ms["flow"] = _ms_since(t)
        self._prev_gray = gray
        return {
            "tracks": active_tracks,
            "detections_count": detections_count,
            "enhanced": enhanced,
            "denoised": denoised,
            "flow_vectors": flow_vectors,
            "detect_frame": enhanced,
            **self._detections_view_data(),
        }

    # ------------------------------------------------------------------ main entry
    def _maybe_trigger_alert(self, result: PipelineResult) -> None:
        if not self.config.enable_alerts or not result.tracks:
            return
        target_classes = set(self.config.alert_classes)
        matched = any(t.class_name in target_classes for t in result.tracks) if target_classes else True
        if not matched:
            return
        remaining = getattr(self.alert_manager, "cooldown_remaining", None)
        if callable(remaining) and remaining() > 0:
            return  # skip rendering the annotated frame while the alert is cooling down
        self.alert_manager.trigger(result.annotated)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run one frame through the active mode's stages and return lazily rendered results."""
        if frame is None or frame.size == 0 or frame.ndim != 3:
            raise ValueError("Expected non-empty 3-channel BGR frame")
        frame = self._limit_width(frame)
        with self._lock:
            t0 = time.perf_counter()
            if self._start_time is None:
                self._start_time = t0

            mode = self.config.mode
            stage_ms: dict[str, float] = {}
            if mode == TrackingMode.CVIP_MOG2:
                data = self._run_cvip(frame, stage_ms)
            elif mode == TrackingMode.DL_SORT:
                data = self._run_dl_sort(frame, stage_ms)
            else:
                data = self._run_hybrid(frame, stage_ms)

            active_tracks = data.pop("tracks")
            detections_count = data.pop("detections_count")
            track_draws = _snapshot_tracks(active_tracks)

            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            # FPS is the real frame rate over a rolling window of recent frames
            # (wall clock, so it includes capture and pacing). Using 1/processing-time
            # made cheap frames (Kalman-only frames between detections) report
            # absurd values, and per-frame intervals alternate wildly with
            # detect_interval > 1.
            self._frame_times.append(t1)
            if len(self._frame_times) >= 2:
                span = max(self._frame_times[-1] - self._frame_times[0], 1e-3)
                fps = (len(self._frame_times) - 1) / span
            else:
                fps = 1000.0 / max(latency_ms, 1.0)
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

            result = PipelineResult(
                mode=mode,
                frame=frame,
                tracks=active_tracks,
                metrics=metric,
                stage_ms=stage_ms,
                track_draws=track_draws,
                **data,
            )
            self.last_stage_ms = stage_ms

        # Drawing / alert snapshotting happen outside the pipeline lock.
        self._maybe_trigger_alert(result)
        return result

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
