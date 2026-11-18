"""Web API tests for alert endpoints, per-mode stages/views, thumbnails, and FPS reporting."""

import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.config import TrackerConfig, TrackingMode
from src.pipeline import TrackingPipeline
from src.web_server import TrackingSessionManager, create_app
from tests.test_web_api import (
    _StubDetector,
    _StubReID,
    _make_request,
    _route_endpoint,
    _step_coro,
    _wait_for_frames,
)


class _FakeSender:
    """Records sends instead of talking to SMTP; optionally fails."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def __call__(self, path: str) -> None:
        self.calls.append(path)
        if self.error is not None:
            raise self.error


def _make_manager(tmp_path: Path, sender=None, mode: TrackingMode = TrackingMode.HYBRID) -> TrackingSessionManager:
    cfg = TrackerConfig(
        metrics_dir=str(tmp_path / "outputs"),
        snapshot_dir=str(tmp_path / "images"),
        min_hits=1,
        mode=mode,
    )

    def factory(c: TrackerConfig) -> TrackingPipeline:
        pipeline = TrackingPipeline(c, detector=_StubDetector(), reid=_StubReID())
        if sender is not None:
            pipeline.alert_manager.sender_fn = sender
        return pipeline

    return TrackingSessionManager(cfg, pipeline_factory=factory)


def _post(app, path: str, payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    return _step_coro(_route_endpoint(app, path)(_make_request("POST", path, body=body)))


def _get(app, path: str, query: str = "", route: str | None = None):
    return _step_coro(_route_endpoint(app, route or path)(_make_request("GET", path, query)))


def _wait_for_history(mgr: TrackingSessionManager, status: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    data = mgr.get_alerts()
    while time.monotonic() < deadline:
        data = mgr.get_alerts()
        if data["history"] and data["history"][0]["status"] == status:
            return data
        time.sleep(0.02)
    return data


def test_alerts_endpoint_shape_masks_recipient(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EMAIL_RECEIVER", "someone@example.com")
    monkeypatch.delenv("GOOGLE_APP_PASSWORD", raising=False)
    app = create_app(_make_manager(tmp_path))

    resp = _get(app, "/api/alerts")
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert set(data) == {
        "enabled",
        "configured",
        "recipient",
        "trigger_classes",
        "cooldown_sec",
        "cooldown_remaining_sec",
        "history",
    }
    assert data["configured"] is False
    assert data["recipient"] == "so•••@example.com"
    assert "someone" not in resp.body.decode("utf-8")
    assert data["trigger_classes"] == ["person"]
    assert data["history"] == []


def test_test_alert_records_sent_and_serves_snapshot(tmp_path: Path) -> None:
    sender = _FakeSender()
    mgr = _make_manager(tmp_path, sender=sender)
    app = create_app(mgr)

    resp = _post(app, "/api/alerts/test")
    assert resp.status_code == 200
    data = _wait_for_history(mgr, "sent")
    assert len(sender.calls) == 1
    rec = data["history"][0]
    assert rec["status"] == "sent"
    assert rec["test"] is True
    assert rec["error"] is None
    assert rec["snapshot_url"] == f"/api/alerts/{rec['id']}/snapshot.jpg"

    snap = _step_coro(
        _route_endpoint(app, "/api/alerts/{alert_id}/snapshot.jpg")(
            _make_request_with_params(rec["snapshot_url"], {"alert_id": str(rec["id"])})
        )
    )
    assert snap.status_code == 200
    assert snap.media_type == "image/jpeg"
    assert snap.body.startswith(b"\xff\xd8")


def test_test_alert_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    sender = _FakeSender(RuntimeError("GOOGLE_APP_PASSWORD environment variable is not set"))
    mgr = _make_manager(tmp_path, sender=sender)
    app = create_app(mgr)

    resp = _post(app, "/api/alerts/test")
    assert resp.status_code == 200
    data = _wait_for_history(mgr, "failed")
    rec = data["history"][0]
    assert rec["status"] == "failed"
    assert "GOOGLE_APP_PASSWORD" in rec["error"]


def test_unknown_alert_snapshot_is_404(tmp_path: Path) -> None:
    app = create_app(_make_manager(tmp_path))
    ep = _route_endpoint(app, "/api/alerts/{alert_id}/snapshot.jpg")
    for raw in ("999", "abc"):
        resp = _step_coro(ep(_make_request_with_params(f"/api/alerts/{raw}/snapshot.jpg", {"alert_id": raw})))
        assert resp.status_code == 404


def test_status_reports_stages_per_mode_and_resets_view(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    app = create_app(mgr)

    data = json.loads(_post(app, "/api/control", {"mode": "cvip", "view": "mog2"}).body)
    assert data["stages"] == ["input", "clahe", "mog2", "flow", "annotated"]
    assert data["view"] == "mog2"

    # mog2 isn't part of the DL pipeline, so switching mode falls back to the tracked view.
    data = json.loads(_post(app, "/api/control", {"mode": "dl_sort"}).body)
    assert data["stages"] == ["input", "detections", "annotated"]
    assert data["view"] == "annotated"

    data = json.loads(_post(app, "/api/control", {"view": "detections"}).body)
    assert data["view"] == "detections"


def test_input_and_detections_views_and_thumbnail_width(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path, mode=TrackingMode.HYBRID)
    app = create_app(mgr)
    try:
        mgr.start("sample")
        status = _wait_for_frames(mgr, min_frames=3)
        assert status["stage_ms"], "stage timings should be reported while running"

        for view in ("input", "detections", "annotated"):
            full = _get(app, "/api/frame.jpg", f"view={view}")
            assert full.status_code == 200
            assert full.body.startswith(b"\xff\xd8")

        thumb = _get(app, "/api/frame.jpg", "view=input&w=240")
        img = cv2.imdecode(np.frombuffer(thumb.body, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[1] == 240
    finally:
        mgr.stop()


def test_fps_is_real_frame_rate_not_inverse_processing_time() -> None:
    """Cheap frames (e.g. Kalman-only between detections) must not report absurd FPS."""
    cfg = TrackerConfig(mode=TrackingMode.DL_SORT, detect_interval=3, min_hits=1)
    pipeline = TrackingPipeline(cfg, detector=_StubDetector(), reid=_StubReID())
    frame = np.full((180, 320, 3), 40, dtype=np.uint8)

    for _ in range(6):
        pipeline.process_frame(frame)
        time.sleep(0.02)  # ~50 FPS pacing, like a camera

    recent = pipeline.metrics_logger.recent(limit=6)
    later = [m["fps"] if isinstance(m, dict) else m.fps for m in recent[1:]]
    assert later and max(later) < 80, later


def _make_request_with_params(path: str, path_params: dict):
    req = _make_request("GET", path)
    req.scope["path_params"] = path_params
    return req
