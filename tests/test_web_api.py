"""Tests for TrackingSessionManager, Starlette web routes, and CLI argument parsing."""

from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import pytest
from starlette.datastructures import FormData, UploadFile
from starlette.requests import Request
from starlette.routing import Mount, Route

from src.app import parse_args
from src.config import TrackerConfig
from src.dl.detector import Detection
from src.pipeline import TrackingPipeline, generate_sample_video
from src.web_server import TrackingSessionManager, create_app


class _StubDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        return [
            Detection(
                bbox=np.array([40.0, 40.0, 110.0, 140.0], dtype=np.float32),
                confidence=0.91,
                class_id=0,
                class_name="person",
            )
        ]


class _StubReID:
    def extract(self, frame: np.ndarray, bboxes: list[np.ndarray]) -> list[np.ndarray | None]:
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = 1.0
        return [vec for _ in bboxes]


def _make_manager(tmp_path: Path) -> TrackingSessionManager:
    cfg = TrackerConfig(
        metrics_dir=str(tmp_path / "outputs"),
        snapshot_dir=str(tmp_path / "images"),
        min_hits=1,
    )
    return TrackingSessionManager(
        cfg,
        pipeline_factory=lambda c: TrackingPipeline(c, detector=_StubDetector(), reid=_StubReID()),
    )


def _wait_for_frames(mgr: TrackingSessionManager, min_frames: int = 2, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last_status = {}
    while time.monotonic() < deadline:
        last_status = mgr.get_status()
        if last_status["summary"]["total_frames"] >= min_frames:
            return last_status
        time.sleep(0.05)
    return last_status


def _step_coro(coro):
    """Drive a pure coroutine to completion without spawning an OS socket event loop."""
    try:
        while True:
            coro.send(None)
    except StopIteration as stop:
        return stop.value


def _make_request(
    method: str = "GET",
    path: str = "/",
    query_string: str = "",
    body: bytes = b"",
    form_data: FormData | None = None,
) -> Request:
    sent_body = False

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string.encode("ascii"),
        "headers": [],
        "scheme": "http",
        "server": ("127.0.0.1", 8000),
    }
    req = Request(scope, receive)
    if form_data is not None:
        req._form = form_data
    return req


def _route_endpoint(app, path: str):
    for route in app.routes:
        if isinstance(route, Route) and route.path == path:
            return route.endpoint
    raise AssertionError(f"Route {path} not found")


def test_app_routes_and_initial_status_shape(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    app = create_app(mgr)

    index_resp = _step_coro(_route_endpoint(app, "/")(_make_request("GET", "/")))
    assert index_resp.status_code == 200
    assert Path(index_resp.path).name == "index.html"

    mounts = [r for r in app.routes if isinstance(r, Mount) and r.path == "/static"]
    assert len(mounts) == 1

    status_resp = _step_coro(_route_endpoint(app, "/api/status")(_make_request("GET", "/api/status")))
    assert status_resp.status_code == 200
    data = json.loads(status_resp.body)
    expected_keys = {
        "running",
        "mode",
        "source",
        "use_clahe",
        "use_reid",
        "detect_interval",
        "enable_alerts",
        "view",
        "error",
        "summary",
        "tracks",
    }
    assert expected_keys.issubset(data.keys())
    assert data["running"] is False


def test_control_sample_start_update_reset_stop_and_validation(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    app = create_app(mgr)
    control_ep = _route_endpoint(app, "/api/control")

    try:
        payload = json.dumps(
            {
                "action": "sample",
                "mode": "cvip",
                "use_clahe": False,
                "use_reid": False,
                "detect_interval": 4,
                "enable_alerts": False,
                "view": "quad",
            }
        ).encode("utf-8")
        resp = _step_coro(control_ep(_make_request("POST", "/api/control", body=payload)))
        assert resp.status_code == 200
        data = json.loads(resp.body)
        assert data["running"] is True
        assert data["mode"] == "cvip"
        assert data["use_clahe"] is False
        assert data["use_reid"] is False
        assert data["detect_interval"] == 4
        assert data["view"] == "quad"

        status = _wait_for_frames(mgr, min_frames=2)
        assert status["summary"]["total_frames"] >= 2

        stop_resp = _step_coro(
            control_ep(_make_request("POST", "/api/control", body=b'{"action":"stop"}'))
        )
        assert stop_resp.status_code == 200
        assert json.loads(stop_resp.body)["running"] is False

        reset_resp = _step_coro(
            control_ep(_make_request("POST", "/api/control", body=b'{"action":"reset"}'))
        )
        assert reset_resp.status_code == 200
        assert json.loads(reset_resp.body)["tracks"] == []

        bad_action = _step_coro(
            control_ep(_make_request("POST", "/api/control", body=b'{"action":"explode"}'))
        )
        assert bad_action.status_code == 400

        bad_mode = _step_coro(
            control_ep(_make_request("POST", "/api/control", body=b'{"mode":"unknown_mode"}'))
        )
        assert bad_mode.status_code == 400
    finally:
        mgr.stop()


def test_frame_jpg_and_mjpeg_stream(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    app = create_app(mgr)
    frame_ep = _route_endpoint(app, "/api/frame.jpg")
    mjpeg_ep = _route_endpoint(app, "/stream/mjpeg")

    try:
        initial = _step_coro(frame_ep(_make_request("GET", "/api/frame.jpg", "view=annotated")))
        assert initial.status_code == 200
        assert initial.body.startswith(b"\xff\xd8")

        mgr.start("sample")
        _wait_for_frames(mgr, min_frames=2)

        for view in ("annotated", "clahe", "mog2", "flow", "quad"):
            jpg = _step_coro(frame_ep(_make_request("GET", "/api/frame.jpg", f"view={view}")))
            assert jpg.status_code == 200
            assert jpg.media_type == "image/jpeg"
            assert jpg.body.startswith(b"\xff\xd8")

        stream_resp = _step_coro(mjpeg_ep(_make_request("GET", "/stream/mjpeg", "view=annotated")))
        assert stream_resp.status_code == 200
        assert "multipart/x-mixed-replace" in stream_resp.media_type
        first_chunk = _step_coro(stream_resp.body_iterator.__anext__())
        assert b"--frame" in first_chunk
        assert b"\xff\xd8" in first_chunk
    finally:
        mgr.stop()


def test_upload_endpoint_saves_and_plays_video(tmp_path: Path) -> None:
    sample_file = generate_sample_video(tmp_path / "upload_src.mp4", num_frames=10, width=320, height=180)
    mgr = _make_manager(tmp_path)
    app = create_app(mgr)
    upload_ep = _route_endpoint(app, "/api/upload")

    try:
        bad_upload = UploadFile(file=BytesIO(b"hello"), filename="notes.txt")
        bad = _step_coro(upload_ep(_make_request("POST", "/api/upload", form_data=FormData([("file", bad_upload)]))))
        assert bad.status_code == 400

        # Real file read via direct Async UploadFile substitute using synchronous BytesIO
        class _MemUpload:
            filename = "../my clip.mp4"

            def __init__(self, data: bytes) -> None:
                self._data = data

            async def read(self) -> bytes:
                return self._data

        ok = _step_coro(
            upload_ep(
                _make_request(
                    "POST",
                    "/api/upload",
                    form_data=FormData([("file", _MemUpload(sample_file.read_bytes()))]),
                )
            )
        )
        assert ok.status_code == 200
        body = json.loads(ok.body)
        assert body["running"] is True
        uploaded = tmp_path / "outputs" / "uploads" / "my_clip.mp4"
        assert uploaded.exists()

        _wait_for_frames(mgr, min_frames=2)
    finally:
        mgr.stop()


def test_metrics_and_export_endpoints(tmp_path: Path) -> None:
    mgr = _make_manager(tmp_path)
    app = create_app(mgr)
    metrics_ep = _route_endpoint(app, "/api/metrics")
    export_ep = _route_endpoint(app, "/api/export")

    try:
        mgr.start("sample")
        _wait_for_frames(mgr, min_frames=3)
        mgr.stop()

        metrics_resp = _step_coro(metrics_ep(_make_request("GET", "/api/metrics", "limit=2")))
        assert metrics_resp.status_code == 200
        mdata = json.loads(metrics_resp.body)
        assert "summary" in mdata and "recent" in mdata
        assert len(mdata["recent"]) <= 2

        for fmt, expected_ct in (
            ("csv", "text/csv"),
            ("json", "application/json"),
            ("plot", "image/png"),
        ):
            exp = _step_coro(export_ep(_make_request("GET", "/api/export", f"format={fmt}")))
            assert exp.status_code == 200
            assert exp.media_type == expected_ct
            assert Path(exp.path).exists()
            assert Path(exp.path).stat().st_size > 0

        bad_exp = _step_coro(export_ep(_make_request("GET", "/api/export", "format=xml")))
        assert bad_exp.status_code == 400
    finally:
        mgr.stop()


def test_cli_parse_args() -> None:
    args = parse_args(
        [
            "--source",
            "sample",
            "--mode",
            "cvip",
            "--debug-view",
            "--save-video",
            "out.mp4",
            "--alerts",
            "--max-frames",
            "25",
            "--web",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ]
    )
    assert args.source == "sample"
    assert args.mode == "cvip"
    assert args.debug_view is True
    assert args.save_video == "out.mp4"
    assert args.alerts is True
    assert args.max_frames == 25
    assert args.web is True
    assert args.host == "0.0.0.0"
    assert args.port == 9000
