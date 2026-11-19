"""Tests for TrackingPipeline, PipelineResult, and synthetic sample video generation."""

from pathlib import Path
import threading
import time

import cv2
import numpy as np

from src.alerts.emailing import AlertManager
from src.config import TrackerConfig, TrackingMode
from src.dl.detector import Detection
from src.eval.metrics import FrameMetrics, MetricsLogger
from src.pipeline import PipelineResult, TrackingPipeline, generate_sample_video


class _FakeDetector:
    """Deterministic stub detector returning scripted boxes per frame."""

    def __init__(self, boxes_fn) -> None:
        self.boxes_fn = boxes_fn
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        dets = self.boxes_fn(self.calls, frame)
        self.calls += 1
        return dets


class _FakeReID:
    """Deterministic stub Re-ID extractor returning L2-normalized vectors."""

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, frame: np.ndarray, bboxes: list[np.ndarray]) -> list[np.ndarray | None]:
        self.calls += 1
        vecs: list[np.ndarray | None] = []
        for idx, _ in enumerate(bboxes):
            v = np.zeros(16, dtype=np.float32)
            v[idx % 16] = 1.0
            vecs.append(v)
        return vecs


def _draw_textured_box(frame: np.ndarray, x: int, y: int, w: int = 60, h: int = 80) -> None:
    """Draw a high-contrast textured patch suitable for MOG2 and Optical Flow."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), -1)
    for dy in range(4, h - 4, 12):
        for dx in range(4, w - 4, 12):
            color = (30, 80, 200) if ((dx + dy) // 12) % 2 == 0 else (20, 20, 20)
            cv2.rectangle(frame, (x + dx, y + dy), (x + dx + 8, y + dy + 8), color, -1)


def _make_sequence(num_frames: int = 8, width: int = 320, height: int = 180) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        frame = np.full((height, width, 3), 25, dtype=np.uint8)
        _draw_textured_box(frame, 20 + i * 15, 40)
        frames.append(frame)
    return frames


def test_pipeline_result_and_cvip_mog2_mode() -> None:
    cfg = TrackerConfig(
        mode=TrackingMode.CVIP_MOG2,
        min_hits=2,
        min_contour_area=400,
    )
    pipeline = TrackingPipeline(cfg)

    frames = _make_sequence(num_frames=8)
    last_result: PipelineResult | None = None
    for frame in frames:
        last_result = pipeline.process_frame(frame)

    assert isinstance(last_result, PipelineResult)
    for panel in (
        last_result.annotated,
        last_result.clahe,
        last_result.mog2,
        last_result.flow,
        last_result.quad,
    ):
        assert isinstance(panel, np.ndarray)
        assert panel.ndim == 3
        assert panel.shape[2] == 3

    assert isinstance(last_result.metrics, FrameMetrics)
    assert last_result.metrics.mode == "cvip"
    assert len(last_result.tracks) >= 1
    assert pipeline.metrics_logger.summary()["total_frames"] == 8


def test_pipeline_dl_sort_mode_uses_detector_and_returns_track_rows() -> None:
    def scripted(call_idx: int, _frame: np.ndarray) -> list[Detection]:
        x1 = 30.0 + call_idx * 10.0
        return [
            Detection(
                bbox=np.array([x1, 40.0, x1 + 50.0, 120.0], dtype=np.float32),
                confidence=0.92,
                class_id=0,
                class_name="person",
            )
        ]

    det = _FakeDetector(scripted)
    reid = _FakeReID()
    # detect_interval=1: DL_SORT now honours detect_interval (Kalman-predict-only in between).
    cfg = TrackerConfig(mode=TrackingMode.DL_SORT, min_hits=2, detect_interval=1)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)

    for frame in _make_sequence(num_frames=5):
        res = pipeline.process_frame(frame)

    assert det.calls == 5
    assert reid.calls == 5  # DL_SORT now fuses Re-ID embeddings too
    assert len(res.tracks) == 1
    rows = pipeline.get_track_rows()
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {
        "track_id",
        "class_name",
        "confidence",
        "vx",
        "vy",
        "age",
        "hits",
        "state",
    }
    assert row["class_name"] == "person"
    assert row["state"] == "confirmed"


def test_pipeline_hybrid_mode_skips_detector_on_non_keyframes_and_uses_reid() -> None:
    def scripted(call_idx: int, _frame: np.ndarray) -> list[Detection]:
        # Keyframes will be frame 0 (call 0) and frame 3 (call 1)
        frame_guess = call_idx * 3
        x1 = 20.0 + frame_guess * 15.0
        return [
            Detection(
                bbox=np.array([x1, 40.0, x1 + 60.0, 120.0], dtype=np.float32),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )
        ]

    det = _FakeDetector(scripted)
    reid = _FakeReID()
    cfg = TrackerConfig(
        mode=TrackingMode.HYBRID,
        detect_interval=3,
        use_reid=True,
        min_hits=1,
    )
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)

    frames = _make_sequence(num_frames=6)
    results = [pipeline.process_frame(f) for f in frames]

    # Detector and ReID only run on keyframes 0 and 3
    assert det.calls == 2
    assert reid.calls == 2
    # Intermediate frames 1, 2, 4, 5 maintain active tracks via Optical Flow
    assert all(len(r.tracks) == 1 for r in results)


def test_pipeline_alerts_trigger_only_for_matching_classes(tmp_path: Path) -> None:
    sent: list[str] = []
    alert_mgr = AlertManager(
        snapshot_dir=tmp_path / "snaps",
        cooldown_sec=0.0,
        enabled=True,
        sender_fn=lambda p: sent.append(p),
    )

    car_det = _FakeDetector(
        lambda idx, _f: [
            Detection(
                bbox=np.array([20.0 + idx * 5, 30.0, 80.0 + idx * 5, 110.0], dtype=np.float32),
                confidence=0.88,
                class_id=2,
                class_name="car",
            )
        ]
    )
    cfg = TrackerConfig(
        mode=TrackingMode.DL_SORT,
        min_hits=1,
        enable_alerts=True,
        alert_classes=["person"],
    )
    pipeline = TrackingPipeline(cfg, detector=car_det, reid=_FakeReID(), alert_manager=alert_mgr)
    pipeline.process_frame(_make_sequence(1)[0])
    alert_mgr.wait(timeout=2.0)
    assert len(sent) == 0

    # Now switch to person detection -> should trigger
    person_det = _FakeDetector(
        lambda idx, _f: [
            Detection(
                bbox=np.array([20.0 + idx * 5, 30.0, 80.0 + idx * 5, 110.0], dtype=np.float32),
                confidence=0.95,
                class_id=0,
                class_name="person",
            )
        ]
    )
    pipeline.detector = person_det
    pipeline.reset()
    pipeline.process_frame(_make_sequence(1)[0])
    alert_mgr.wait(timeout=2.0)
    assert len(sent) == 1


def test_pipeline_thread_safety_with_concurrent_updates() -> None:
    det = _FakeDetector(
        lambda _idx, _f: [
            Detection(
                bbox=np.array([40.0, 40.0, 100.0, 120.0], dtype=np.float32),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )
        ]
    )
    reid = _FakeReID()
    cfg = TrackerConfig(mode=TrackingMode.HYBRID, min_hits=1)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)

    errors: list[Exception] = []
    frames = _make_sequence(num_frames=10)

    def worker_process() -> None:
        try:
            for f in frames:
                pipeline.process_frame(f)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def worker_control() -> None:
        try:
            for mode in (TrackingMode.CVIP_MOG2, TrackingMode.DL_SORT, TrackingMode.HYBRID):
                pipeline.set_mode(mode)
                pipeline.update_settings(use_clahe=False, detect_interval=2)
                pipeline.update_settings(use_clahe=True, use_reid=True)
                time.sleep(0.005)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t1 = threading.Thread(target=worker_process)
    t2 = threading.Thread(target=worker_control)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not errors


def test_generate_sample_video_creates_readable_mp4_with_motion(tmp_path: Path) -> None:
    out = generate_sample_video(tmp_path / "demo.mp4", num_frames=12, width=320, height=180)
    assert out.exists()
    assert out.stat().st_size > 0

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    read_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_frames.append(frame)
    cap.release()

    assert len(read_frames) == 12
    assert read_frames[0].shape == (180, 320, 3)
    assert not np.array_equal(read_frames[0], read_frames[-1])


def test_pipeline_hybrid_confirms_late_entering_object_across_flow_frames() -> None:
    """A new TENTATIVE object entering while another track is CONFIRMED must survive flow steps."""

    def scripted(call_idx: int, _frame: np.ndarray) -> list[Detection]:
        # Object 1 present on all keyframes (calls 0..4 => frames 0, 2, 4, 6, 8)
        dets = [
            Detection(
                bbox=np.array([20.0 + call_idx * 10, 20.0, 80.0 + call_idx * 10, 90.0], dtype=np.float32),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )
        ]
        # Object 2 enters on keyframe call 2 (frame 4) after Object 1 is already CONFIRMED
        if call_idx >= 2:
            shift = (call_idx - 2) * 10
            dets.append(
                Detection(
                    bbox=np.array([150.0 + shift, 50.0, 210.0 + shift, 130.0], dtype=np.float32),
                    confidence=0.88,
                    class_id=0,
                    class_name="person",
                )
            )
        return dets

    det = _FakeDetector(scripted)
    reid = _FakeReID()
    cfg = TrackerConfig(
        mode=TrackingMode.HYBRID,
        detect_interval=2,
        min_hits=3,
        use_reid=True,
    )
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)

    frames = _make_sequence(num_frames=9)
    results = [pipeline.process_frame(f) for f in frames]

    # By frame 8 (keyframe call 4 => Object 2 has 3 hits: frames 4, 6, 8), both tracks must be CONFIRMED
    assert len(results[-1].tracks) == 2
    assert pipeline.tracker.total_tracks_created == 2

