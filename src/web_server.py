"""Starlette web server and background session manager for live video tracking.

Performance notes:
* The tracking thread publishes one immutable PipelineResult per frame plus a sequence number.
* Panels are rendered lazily and JPEG-encoded at most once per (view, width) per frame, in a
  worker thread (never on the asyncio event loop); every client shares the cached bytes.
* MJPEG streams only push a part when a new frame sequence exists (event-driven wake-up).
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src.alerts.emailing import is_email_configured, mask_email, resolve_receiver
from src.config import TrackerConfig, TrackingMode
from src.pipeline import (
    ALL_VIEWS,
    STAGE_TIMING_KEYS,
    PipelineResult,
    TrackingPipeline,
    generate_sample_video,
    stages_for_mode,
    views_for_mode,
)

_ALLOWED_UPLOAD_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_VALID_VIEWS = set(ALL_VIEWS)
_WEB_DIR = Path(__file__).resolve().parent / "web"
_JPEG_QUALITY = 75
_MIN_WIDTH, _MAX_WIDTH = 64, 1280
_WEBCAM_SIZE = (640, 480)


def _make_placeholder_frame(text: str = "Waiting for video stream...") -> np.ndarray:
    img = np.full((360, 640, 3), 24, dtype=np.uint8)
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(
        img,
        text,
        (max(10, (640 - tw) // 2), 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (180, 190, 200),
        2,
        cv2.LINE_AA,
    )
    return img


def _resize_to_width(img: np.ndarray, width: int | None) -> np.ndarray:
    """Downscale (never upscale) so the image is at most `width` pixels wide."""
    if width is None:
        return img
    h, w = img.shape[:2]
    if w <= width:
        return img
    new_h = max(1, int(round(h * width / float(w))))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def _encode_jpeg(img: np.ndarray, width: int | None = None) -> bytes:
    ok, buf = cv2.imencode(".jpg", _resize_to_width(img, width), [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _make_placeholder_jpeg(text: str = "Waiting for video stream...", width: int | None = None) -> bytes:
    return _encode_jpeg(_make_placeholder_frame(text), width)


def _parse_width(raw: str | None) -> int | None:
    """Parse the optional ?w= thumbnail width, clamped to [64, 1280]; invalid -> None."""
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return max(_MIN_WIDTH, min(_MAX_WIDTH, value))


def _sanitize_filename(filename: str) -> str:
    base = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).lstrip(".")
    return cleaned or "upload.mp4"


class TrackingSessionManager:
    """Manages a background tracking loop and thread-safe frame/telemetry state."""

    def __init__(
        self,
        config: TrackerConfig,
        pipeline_factory: Callable[[TrackerConfig], TrackingPipeline] | None = None,
    ) -> None:
        self.config = config
        self._pipeline_factory = pipeline_factory or TrackingPipeline
        self.pipeline = self._pipeline_factory(config)

        self._ctrl_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._running = False
        self._view = "quad" if config.debug_view else "annotated"
        self._error: str | None = None

        # Latest published frame (immutable) + monotonically increasing sequence number.
        self._latest_result: PipelineResult | None = None
        self._latest_tracks: list[dict[str, Any]] = []
        self._seq = 0

        # {(view, width): (seq, jpeg_bytes)} - each frame is encoded at most once per key.
        self._encode_cache: dict[tuple[str, int | None], tuple[int, bytes]] = {}
        self._cache_lock = threading.Lock()
        self._placeholder_cache: dict[tuple[str, int | None], bytes] = {}

        # asyncio waiters woken (thread-safely) whenever a new frame is published.
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = set()
        self._waiters_lock = threading.Lock()

        self._placeholder_jpeg = self._placeholder("Waiting for video stream...", None)

    # ------------------------------------------------------------------ frame publication
    @property
    def frame_seq(self) -> int:
        return self._seq

    def _publish(self, result: PipelineResult | None, track_rows: list[dict[str, Any]]) -> None:
        with self._frame_lock:
            self._latest_result = result
            self._latest_tracks = track_rows
            self._seq += 1
        with self._cache_lock:
            # Old entries can never be served again; drop them to bound memory.
            self._encode_cache.clear()
        self._notify_waiters()

    def _bump_seq(self) -> None:
        """Signal streams to re-render (e.g. mode/view change while paused)."""
        with self._frame_lock:
            self._seq += 1
        self._notify_waiters()

    def _notify_waiters(self) -> None:
        with self._waiters_lock:
            waiters = list(self._waiters)
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # loop already closed
                pass

    async def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> int:
        """Wait (without blocking the event loop) until frame_seq != last_seq or timeout."""
        if self._seq != last_seq:
            return self._seq
        entry = (asyncio.get_running_loop(), asyncio.Event())
        with self._waiters_lock:
            self._waiters.add(entry)
        try:
            if self._seq == last_seq:
                try:
                    await asyncio.wait_for(entry[1].wait(), timeout)
                except asyncio.TimeoutError:
                    pass
        finally:
            with self._waiters_lock:
                self._waiters.discard(entry)
        return self._seq

    # ------------------------------------------------------------------ capture loop
    def _open_capture(self, source: str) -> tuple[cv2.VideoCapture, bool]:
        if source == "sample":
            sample_path = Path(self.config.metrics_dir) / "sample_demo.mp4"
            if not sample_path.exists():
                generate_sample_video(sample_path)
            return cv2.VideoCapture(str(sample_path)), True
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            # Keep only the newest frame and request a modest resolution to avoid stale/lagging frames.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, _WEBCAM_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _WEBCAM_SIZE[1])
            return cap, False
        return cv2.VideoCapture(source), Path(source).is_file()

    def _run_loop(self, source: str, stop_event: threading.Event) -> None:
        cap, should_loop = self._open_capture(source)
        if not cap.isOpened():
            with self._ctrl_lock:
                self._error = f"Failed to open video source: {source}"
                if self._thread is threading.current_thread():
                    self._running = False
            cap.release()
            return

        target_dt = 1.0 / 22.0
        consecutive_failures = 0

        try:
            while not stop_event.is_set():
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if should_loop and consecutive_failures < 5:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self._ctrl_lock:
                        self._error = f"Stream ended or failed on source: {source}"
                    break

                consecutive_failures = 0
                result = self.pipeline.process_frame(frame)
                self._publish(result, self.pipeline.get_track_rows())

                elapsed = time.monotonic() - t0
                if should_loop and elapsed < target_dt:
                    stop_event.wait(target_dt - elapsed)
        except Exception as exc:  # pragma: no cover
            with self._ctrl_lock:
                self._error = str(exc)
        finally:
            cap.release()
            with self._ctrl_lock:
                if self._thread is threading.current_thread():
                    self._running = False

    def start(self, source: str | None = None) -> None:
        """Start (or restart) background video processing on the given source."""
        self.stop()
        with self._ctrl_lock:
            if source is not None:
                self.config.source = source
            active_source = self.config.source
            self._error = None
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                args=(active_source, stop_event),
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background video processing thread if running."""
        with self._ctrl_lock:
            thread = self._thread
            self._stop_event.set()
            self._running = False
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        with self._ctrl_lock:
            if self._thread is thread:
                self._thread = None

    def reset(self) -> None:
        """Reset pipeline tracks, background model, and cached frames."""
        self.pipeline.reset()
        self._publish(None, [])

    # ------------------------------------------------------------------ settings / status
    def update_config(
        self,
        *,
        mode: str | TrackingMode | None = None,
        use_clahe: bool | None = None,
        use_reid: bool | None = None,
        detect_interval: int | None = None,
        enable_alerts: bool | None = None,
        view: str | None = None,
    ) -> None:
        """Validate and apply runtime configuration updates."""
        resolved = None
        if mode is not None:
            resolved = mode if isinstance(mode, TrackingMode) else TrackingMode(mode)
        if view is not None and view not in _VALID_VIEWS:
            raise ValueError(f"Invalid view: {view}")

        if resolved is not None:
            self.pipeline.set_mode(resolved)
        self.pipeline.update_settings(
            use_clahe=use_clahe,
            use_reid=use_reid,
            detect_interval=detect_interval,
            enable_alerts=enable_alerts,
        )
        with self._ctrl_lock:
            if view is not None:
                self._view = view
            if self._view not in views_for_mode(self.config.mode):
                # e.g. "mog2" selected and then switching to dl_sort
                self._view = "annotated"
        if resolved is not None or view is not None:
            self._bump_seq()

    def get_status(self) -> dict[str, Any]:
        """Return full session status, settings, summary metrics, and active tracks.

        Cheap by design: no pipeline lock, O(1) summary, no image work.
        """
        with self._ctrl_lock:
            running = self._running
            view = self._view
            error = self._error
        with self._frame_lock:
            tracks = list(self._latest_tracks)
        mode = self.config.mode
        stage_ms = dict(self.pipeline.last_stage_ms)
        return {
            "running": running,
            "mode": mode.value,
            "source": self.config.source,
            "use_clahe": self.config.use_clahe,
            "use_reid": self.config.use_reid,
            "detect_interval": self.config.detect_interval,
            "enable_alerts": self.config.enable_alerts,
            "view": view,
            "error": error,
            "summary": self.pipeline.metrics_logger.summary(),
            "tracks": tracks,
            "stages": stages_for_mode(mode),
            "stage_ms": {k: stage_ms[k] for k in STAGE_TIMING_KEYS if k in stage_ms},
            "detector_loaded": bool(self.pipeline.detector_loaded),
        }

    # ------------------------------------------------------------------ JPEG frames
    def _placeholder(self, text: str, width: int | None) -> bytes:
        key = (text, width)
        cached = self._placeholder_cache.get(key)
        if cached is None:
            cached = _make_placeholder_jpeg(text, width)
            if len(self._placeholder_cache) < 64:
                self._placeholder_cache[key] = cached
        return cached

    def peek_jpeg(self, view: str = "annotated", width: int | None = None) -> bytes | None:
        """Return cached JPEG bytes for the current frame if already encoded (cheap, no encoding)."""
        key_view = view if view in _VALID_VIEWS else "annotated"
        seq = self._seq
        with self._cache_lock:
            hit = self._encode_cache.get((key_view, width))
        if hit is not None and hit[0] == seq:
            return hit[1]
        return None

    def get_jpeg(self, view: str = "annotated", width: int | None = None) -> bytes:
        """Return JPEG bytes of the latest frame for a view, encoding at most once per frame.

        Blocking (render + encode); call from a worker thread in async handlers.
        """
        key_view = view if view in _VALID_VIEWS else "annotated"
        with self._frame_lock:
            result = self._latest_result
            seq = self._seq
        mode = self.config.mode
        if key_view not in views_for_mode(mode) or (result is not None and not result.has_view(key_view)):
            return self._placeholder(f"Not used in {mode.value}", width)
        if result is None:
            return self._placeholder("Waiting for video stream...", width)

        cache_key = (key_view, width)
        with self._cache_lock:
            hit = self._encode_cache.get(cache_key)
        if hit is not None and hit[0] == seq:
            return hit[1]

        panel = result.panel(key_view)
        if panel is None:
            return self._placeholder(f"Not used in {mode.value}", width)
        data = _encode_jpeg(panel, width) or self._placeholder_jpeg
        with self._cache_lock:
            if self._seq == seq:
                self._encode_cache[cache_key] = (seq, data)
        return data

    async def get_jpeg_async(self, view: str = "annotated", width: int | None = None) -> bytes:
        """Event-loop friendly get_jpeg: cache hits are served inline, misses encode in a thread."""
        cached = self.peek_jpeg(view, width)
        if cached is not None:
            return cached
        return await asyncio.to_thread(self.get_jpeg, view, width)

    # ------------------------------------------------------------------ alerts
    def get_alerts(self) -> dict[str, Any]:
        """Return alert configuration (with masked recipient) and recent alert history."""
        alert_mgr = self.pipeline.alert_manager
        history = [
            {
                "id": rec["id"],
                "time": rec["time"],
                "status": rec["status"],
                "error": rec["error"],
                "test": rec["test"],
                "snapshot_url": (
                    f"/api/alerts/{rec['id']}/snapshot.jpg" if rec["has_snapshot"] else None
                ),
            }
            for rec in alert_mgr.history()
        ]
        return {
            "enabled": bool(self.config.enable_alerts),
            "configured": is_email_configured(),
            "recipient": mask_email(resolve_receiver()),
            "trigger_classes": list(self.config.alert_classes),
            "cooldown_sec": float(alert_mgr.cooldown_sec),
            "cooldown_remaining_sec": round(float(alert_mgr.cooldown_remaining()), 3),
            "history": history,
        }

    def get_alert_snapshot(self, alert_id: int) -> bytes | None:
        """Return the in-memory JPEG snapshot for an alert record, or None if unknown."""
        return self.pipeline.alert_manager.get_snapshot(alert_id)

    def send_test_alert(self, wait_timeout: float = 0.5) -> None:
        """Send a test alert using the latest annotated frame (or a placeholder).

        Bypasses cooldown and the enable_alerts flag. Dispatches in a background thread
        and waits briefly so fast failures (e.g. missing password) are already recorded.
        Blocking: call via a threadpool from async handlers.
        """
        with self._ctrl_lock:
            running = self._running
        frame = None
        if running:
            with self._frame_lock:
                result = self._latest_result
            if result is not None:
                frame = result.annotated.copy()
        if frame is None:
            frame = _make_placeholder_frame("Test alert - no live video")
        alert_mgr = self.pipeline.alert_manager
        if alert_mgr.send_test(frame, blocking=False):
            alert_mgr.wait(timeout=wait_timeout)

    # ------------------------------------------------------------------ metrics
    def get_metrics(self, limit: int = 60) -> dict[str, Any]:
        """Return aggregated metrics summary and recent frame history."""
        return {
            "summary": self.pipeline.metrics_logger.summary(),
            "recent": self.pipeline.metrics_logger.recent(limit=max(1, limit)),
        }

    def export_metrics(self, fmt: str) -> tuple[Path, str, str]:
        """Export metrics in the requested format and return (path, media_type, filename)."""
        out_dir = Path(self.config.metrics_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = self.pipeline.metrics_logger
        fmt_clean = fmt.lower()
        if fmt_clean == "csv":
            dest = out_dir / "metrics.csv"
            logger.export_csv(dest)
            return dest, "text/csv", "metrics.csv"
        if fmt_clean == "json":
            dest = out_dir / "metrics.json"
            logger.export_json(dest)
            return dest, "application/json", "metrics.json"
        if fmt_clean == "plot":
            dest = out_dir / "metrics_plot.png"
            logger.export_plot(dest)
            return dest, "image/png", "metrics_plot.png"
        raise ValueError(f"Unsupported export format: {fmt}")


def create_app(session_manager: TrackingSessionManager | None = None) -> Starlette:
    """Create and configure the Starlette web application."""
    manager = session_manager or TrackingSessionManager(TrackerConfig())

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        try:
            yield
        finally:
            manager.stop()

    async def index(_request: Request) -> Response:
        index_path = _WEB_DIR / "index.html"
        if index_path.is_file():
            return FileResponse(index_path, media_type="text/html")
        return JSONResponse({"detail": "Frontend not found"}, status_code=404)

    async def api_status(_request: Request) -> JSONResponse:
        return JSONResponse(manager.get_status())

    async def api_control(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "JSON object expected"}, status_code=400)

        try:
            manager.update_config(
                mode=payload.get("mode"),
                use_clahe=payload.get("use_clahe"),
                use_reid=payload.get("use_reid"),
                detect_interval=payload.get("detect_interval"),
                enable_alerts=payload.get("enable_alerts"),
                view=payload.get("view"),
            )
        except (ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        action = payload.get("action")
        if action is not None:
            if action == "start":
                await asyncio.to_thread(manager.start, payload.get("source"))
            elif action == "stop":
                await asyncio.to_thread(manager.stop)
            elif action == "reset":
                manager.reset()
            elif action == "sample":
                await asyncio.to_thread(manager.start, "sample")
            elif action == "webcam":
                await asyncio.to_thread(manager.start, "0")
            else:
                return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

        return JSONResponse(manager.get_status())

    async def api_upload(request: Request) -> JSONResponse:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not getattr(upload, "filename", None):
            return JSONResponse({"error": "Missing video file upload"}, status_code=400)

        ext = Path(upload.filename).suffix.lower()
        if ext not in _ALLOWED_UPLOAD_EXTS:
            return JSONResponse(
                {"error": f"Unsupported file extension: {ext}"},
                status_code=400,
            )

        uploads_dir = Path(manager.config.metrics_dir) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        dest_path = uploads_dir / _sanitize_filename(upload.filename)
        content = await upload.read()
        await asyncio.to_thread(dest_path.write_bytes, content)

        await asyncio.to_thread(manager.start, str(dest_path))
        return JSONResponse(manager.get_status())

    async def api_metrics(request: Request) -> JSONResponse:
        raw_limit = request.query_params.get("limit", "60")
        try:
            limit = max(1, int(raw_limit))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)
        return JSONResponse(manager.get_metrics(limit=limit))

    async def api_export(request: Request) -> Response:
        fmt = request.query_params.get("format", "csv")
        try:
            path, media_type, filename = await asyncio.to_thread(manager.export_metrics, fmt)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return FileResponse(path, media_type=media_type, filename=filename)

    async def api_alerts(_request: Request) -> JSONResponse:
        return JSONResponse(manager.get_alerts())

    async def api_alert_snapshot(request: Request) -> Response:
        raw_id = request.path_params.get("alert_id")
        try:
            alert_id = int(raw_id)
        except (TypeError, ValueError):
            return JSONResponse({"error": f"Unknown alert id: {raw_id}"}, status_code=404)
        jpeg = manager.get_alert_snapshot(alert_id)
        if jpeg is None:
            return JSONResponse({"error": f"Unknown alert id: {alert_id}"}, status_code=404)
        return Response(content=jpeg, media_type="image/jpeg")

    async def api_alerts_test(_request: Request) -> JSONResponse:
        await asyncio.to_thread(manager.send_test_alert)
        return JSONResponse(manager.get_alerts())

    async def api_frame_jpg(request: Request) -> Response:
        view = request.query_params.get("view", "annotated")
        width = _parse_width(request.query_params.get("w"))
        jpeg = await manager.get_jpeg_async(view, width)
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def stream_mjpeg(request: Request) -> StreamingResponse:
        view = request.query_params.get("view", "annotated")
        width = _parse_width(request.query_params.get("w"))

        async def _frame_generator():
            last_seq: int | None = None
            while True:
                seq = manager.frame_seq
                if seq != last_seq:
                    last_seq = seq
                    jpeg = await manager.get_jpeg_async(view, width)
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    )
                    yield header + jpeg + b"\r\n"
                if await request.is_disconnected():
                    break
                # Sleep until the tracking thread publishes a new frame (or 1 s for disconnect checks).
                await manager.wait_for_frame(last_seq, timeout=1.0)

        return StreamingResponse(
            _frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/control", api_control, methods=["POST"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/api/metrics", api_metrics, methods=["GET"]),
        Route("/api/export", api_export, methods=["GET"]),
        Route("/api/alerts", api_alerts, methods=["GET"]),
        Route("/api/alerts/test", api_alerts_test, methods=["POST"]),
        Route("/api/alerts/{alert_id}/snapshot.jpg", api_alert_snapshot, methods=["GET"]),
        Route("/api/frame.jpg", api_frame_jpg, methods=["GET"]),
        Route("/stream/mjpeg", stream_mjpeg, methods=["GET"]),
        Mount("/static", StaticFiles(directory=str(_WEB_DIR), check_dir=False), name="static"),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.session_manager = manager
    return app
