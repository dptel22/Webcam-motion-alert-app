"""Strict per-mode stage gating: CVIP never touches DL models, DL never runs CVIP stages."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.config import TrackerConfig, TrackingMode
from src.dl.detector import Detection
from src.pipeline import MODE_STAGES, TrackingPipeline, stages_for_mode, views_for_mode
from src.tracking.tracker import MultiObjectTracker


def _frames(n: int = 6, width: int = 320, height: int = 180) -> list[np.ndarray]:
    out = []
    for i in range(n):
        frame = np.full((height, width, 3), 25, dtype=np.uint8)
        x = 20 + i * 15
        cv2.rectangle(frame, (x, 40), (x + 60, 120), (220, 220, 220), -1)
        for dy in range(4, 76, 12):
            for dx in range(4, 56, 12):
                cv2.rectangle(frame, (x + dx, 40 + dy), (x + dx + 8, 48 + dy), (30, 80, 200), -1)
        out.append(frame)
    return out


def _spy_detector() -> MagicMock:
    det = MagicMock(name="detector")
    det.is_loaded = True

    def _detect(frame):
        return [
            Detection(
                bbox=np.array([40.0, 40.0, 110.0, 120.0], dtype=np.float32),
                confidence=0.9,
                class_id=0,
                class_name="person",
            )
        ]

    det.detect.side_effect = _detect
    return det


def _spy_reid() -> MagicMock:
    reid = MagicMock(name="reid")

    def _extract(frame, bboxes):
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = 1.0
        return [vec.copy() for _ in bboxes]

    reid.extract.side_effect = _extract
    return reid


def _spy_cvip(pipeline: TrackingPipeline) -> dict[str, MagicMock]:
    """Wrap every CVIP stage entry point with a pass-through spy."""
    spies = {}
    for obj, names in (
        (pipeline.preprocessor, ("process", "apply_clahe", "denoise", "extract_foreground_mask", "detect_motion_boxes")),
        (pipeline.flow_tracker, ("estimate_box_shifts", "compute_global_flow_vectors")),
    ):
        for name in names:
            spy = MagicMock(wraps=getattr(obj, name))
            setattr(obj, name, spy)
            spies[name] = spy
    return spies


def test_mode_stage_lists() -> None:
    assert stages_for_mode("cvip") == ["input", "clahe", "mog2", "flow", "annotated"]
    assert stages_for_mode("dl_sort") == ["input", "detections", "annotated"]
    assert stages_for_mode("hybrid") == ["input", "clahe", "flow", "detections", "annotated"]
    for mode in MODE_STAGES:
        assert views_for_mode(mode)[-1] == "quad"


def test_cvip_mode_never_calls_detector_or_reid() -> None:
    det, reid = _spy_detector(), _spy_reid()
    cfg = TrackerConfig(mode=TrackingMode.CVIP_MOG2, min_hits=1, min_contour_area=400)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)
    spies = _spy_cvip(pipeline)

    results = [pipeline.process_frame(f) for f in _frames(6)]

    det.detect.assert_not_called()
    reid.extract.assert_not_called()
    assert spies["apply_clahe"].call_count == 6
    assert spies["extract_foreground_mask"].call_count == 6
    assert spies["detect_motion_boxes"].call_count == 6
    assert set(results[-1].stage_ms) == {"preprocess", "mog2", "track", "flow"}
    assert results[-1].panel("detections") is None


def test_cvip_mode_never_constructs_default_dl_models() -> None:
    cfg = TrackerConfig(mode=TrackingMode.CVIP_MOG2, min_contour_area=400)
    pipeline = TrackingPipeline(cfg)
    for frame in _frames(4):
        pipeline.process_frame(frame)
    assert pipeline._detector is None
    assert pipeline._reid is None
    assert pipeline.detector_loaded is False


def test_dl_sort_mode_never_runs_cvip_stages_and_feeds_raw_frame() -> None:
    det, reid = _spy_detector(), _spy_reid()
    cfg = TrackerConfig(mode=TrackingMode.DL_SORT, min_hits=1, detect_interval=1, use_reid=True)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)
    spies = _spy_cvip(pipeline)

    frames = _frames(4)
    results = [pipeline.process_frame(f) for f in frames]

    for name, spy in spies.items():
        assert spy.call_count == 0, f"{name} must not run in dl_sort"
    assert det.detect.call_count == 4
    assert reid.extract.call_count == 4
    # Detector receives the untouched input frame (no CLAHE / blur).
    assert det.detect.call_args_list[0].args[0] is frames[0]
    assert set(results[-1].stage_ms) == {"detect", "reid", "track"}
    assert results[-1].panel("clahe") is None
    assert results[-1].panel("mog2") is None
    assert results[-1].panel("flow") is None
    # Re-ID embeddings are fused into DL_SORT tracks
    assert len(results[-1].tracks) == 1
    assert results[-1].tracks[0].embedding is not None
    assert pipeline.detector_loaded is True


def test_dl_sort_without_reid_skips_reid() -> None:
    det, reid = _spy_detector(), _spy_reid()
    cfg = TrackerConfig(mode=TrackingMode.DL_SORT, min_hits=1, detect_interval=1, use_reid=False)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)
    pipeline.process_frame(_frames(1)[0])
    reid.extract.assert_not_called()
    assert pipeline._reid is reid


def test_dl_sort_honours_detect_interval_with_kalman_prediction() -> None:
    det, reid = _spy_detector(), _spy_reid()
    cfg = TrackerConfig(mode=TrackingMode.DL_SORT, min_hits=1, detect_interval=3)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)

    results = [pipeline.process_frame(f) for f in _frames(6)]

    assert det.detect.call_count == 2  # frames 0 and 3
    assert all(len(r.tracks) == 1 for r in results)  # tracks coast on Kalman prediction
    assert pipeline.tracker.total_tracks_created == 1
    assert "detect" not in results[1].stage_ms and "track" in results[1].stage_ms
    assert results[1].detections_stale == 1
    assert results[3].detections_stale == 0


def test_hybrid_runs_both_projects() -> None:
    det, reid = _spy_detector(), _spy_reid()
    cfg = TrackerConfig(mode=TrackingMode.HYBRID, min_hits=1, detect_interval=1)
    pipeline = TrackingPipeline(cfg, detector=det, reid=reid)
    spies = _spy_cvip(pipeline)
    frames = _frames(3)
    for frame in frames:
        res = pipeline.process_frame(frame)
    assert det.detect.call_count == 3
    assert spies["apply_clahe"].call_count == 3
    # Hybrid has no MOG2 stage
    assert spies["extract_foreground_mask"].call_count == 0
    assert det.detect.call_args_list[0].args[0] is not frames[0]  # gets the CLAHE-enhanced frame
    assert {"preprocess", "detect", "reid", "track", "flow"} <= set(res.stage_ms)
    assert res.panel("mog2") is None
    assert res.panel("detections") is not None


def test_mog2_reset_only_on_switch_into_cvip() -> None:
    pipeline = TrackingPipeline(TrackerConfig(mode=TrackingMode.HYBRID), detector=_spy_detector(), reid=_spy_reid())
    with patch.object(pipeline.preprocessor, "reset_background", wraps=pipeline.preprocessor.reset_background) as spy:
        pipeline.set_mode(TrackingMode.DL_SORT)
        assert spy.call_count == 0
        pipeline.set_mode(TrackingMode.CVIP_MOG2)
        assert spy.call_count == 1
        pipeline.set_mode(TrackingMode.CVIP_MOG2)  # no-op
        assert spy.call_count == 1


def test_detections_view_draws_raw_boxes_before_tracking() -> None:
    det = _spy_detector()
    cfg = TrackerConfig(mode=TrackingMode.DL_SORT, min_hits=5, detect_interval=1, use_reid=False)
    pipeline = TrackingPipeline(cfg, detector=det, reid=_spy_reid())
    frame = _frames(1)[0]
    res = pipeline.process_frame(frame)
    # min_hits=5 -> no confirmed track yet, but raw detections are already drawn
    assert res.tracks == []
    panel = res.panel("detections")
    assert panel.shape == frame.shape
    assert not np.array_equal(panel, frame)
    assert np.array_equal(res.panel("input"), frame)


def test_quad_is_lazy_and_uses_mode_stages() -> None:
    pipeline = TrackingPipeline(
        TrackerConfig(mode=TrackingMode.DL_SORT, detect_interval=1), detector=_spy_detector(), reid=_spy_reid()
    )
    res = pipeline.process_frame(_frames(1)[0])
    assert "quad" not in res._cache and "annotated" not in res._cache
    quad = res.quad
    assert quad.shape == res.frame.shape
    assert {"quad", "annotated", "detections"} <= set(res._cache)
    assert "render" in res.stage_ms


def test_wide_input_is_downscaled_to_960() -> None:
    pipeline = TrackingPipeline(TrackerConfig(mode=TrackingMode.CVIP_MOG2))
    res = pipeline.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert res.frame.shape == (540, 960, 3)


@pytest.mark.parametrize("mode, expect_match", [(TrackingMode.DL_SORT, True), (TrackingMode.CVIP_MOG2, False)])
def test_tracker_fuses_reid_cost_in_dl_sort(mode, expect_match) -> None:
    tracker = MultiObjectTracker(min_hits=1, max_age=10)
    emb = np.zeros(16, dtype=np.float32)
    emb[3] = 1.0
    first = Detection(np.array([0, 0, 40, 80], dtype=np.float32), 0.9, 0, "person", embedding=emb.copy())
    tracker.step([first], mode=mode)
    # Same appearance, but no spatial overlap: only Re-ID can associate it.
    jumped = Detection(np.array([200, 0, 240, 80], dtype=np.float32), 0.9, 0, "person", embedding=emb.copy())
    active = tracker.step([jumped], mode=mode)
    assert (active[0].track_id == 1) is expect_match
