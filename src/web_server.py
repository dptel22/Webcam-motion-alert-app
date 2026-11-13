"""Starlette web server and background session manager for live video tracking."""

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

from src.config import TrackerConfig, TrackingMode
from src.pipeline import TrackingPipeline, generate_sample_video

_ALLOWED_UPLOAD_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_VALID_VIEWS = {"annotated", "clahe", "mog2", "flow", "quad"}
_WEB_DIR = Path(__file__).resolve().parent / "web"


def _make_placeholder_jpeg() -> bytes:
    img = np.full((360, 640, 3), 24, dtype=np.uint8)
    cv2.putText(
        img,
        "Waiting for video stream...",
        (170, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (180, 190, 200),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


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
        self._latest_panels: dict[str, np.ndarray] = {}
        self._latest_tracks: list[dict[str, Any]] = []
        self._placeholder_jpeg = _make_placeholder_jpeg()

    def _open_capture(self, source: str) -> tuple[cv2.VideoCapture, bool]:
        if source == "sample":
            sample_path = Path(self.config.metrics_dir) / "sample_demo.mp4"
            if not sample_path.exists():
                generate_sample_video(sample_path)
            return cv2.VideoCapture(str(sample_path)), True
        if source.isdigit():
            return cv2.VideoCapture(int(source)), False
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
                track_rows = self.pipeline.get_track_rows()
                with self._frame_lock:
                    self._latest_panels = {
                        "annotated": result.annotated,
                        "clahe": result.clahe,
                        "mog2": result.mog2,
                        "flow": result.flow,
                        "quad": result.quad,
                    }
                    self._latest_tracks = track_rows

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
        with self._frame_lock:
            self._latest_panels.clear()
            self._latest_tracks.clear()

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
        if mode is not None:
            resolved = mode if isinstance(mode, TrackingMode) else TrackingMode(mode)
            self.pipeline.set_mode(resolved)
        self.pipeline.update_settings(
            use_clahe=use_clahe,
            use_reid=use_reid,
            detect_interval=detect_interval,
            enable_alerts=enable_alerts,
        )
        if view is not None:
            if view not in _VALID_VIEWS:
                raise ValueError(f"Invalid view: {view}")
            with self._ctrl_lock:
                self._view = view

    def get_status(self) -> dict[str, Any]:
        """Return full session status, settings, summary metrics, and active tracks."""
        with self._ctrl_lock:
            running = self._running
            view = self._view
            error = self._error
        with self._frame_lock:
            tracks = list(self._latest_tracks)
        return {
            "running": running,
            "mode": self.config.mode.value,
            "source": self.config.source,
            "use_clahe": self.config.use_clahe,
            "use_reid": self.config.use_reid,
            "detect_interval": self.config.detect_interval,
            "enable_alerts": self.config.enable_alerts,
            "view": view,
            "error": error,
            "summary": self.pipeline.metrics_logger.summary(),
            "tracks": tracks,
        }

    def get_jpeg(self, view: str = "annotated") -> bytes:
        """Encode the latest frame for the requested view as JPEG bytes."""
        key = view if view in _VALID_VIEWS else "annotated"
        with self._frame_lock:
            frame = self._latest_panels.get(key)
            if frame is not None:
                frame = frame.copy()
        if frame is None:
            return self._placeholder_jpeg
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else self._placeholder_jpeg

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
                manager.start(payload.get("source"))
            elif action == "stop":
                manager.stop()
            elif action == "reset":
                manager.reset()
            elif action == "sample":
                manager.start("sample")
            elif action == "webcam":
                manager.start("0")
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
        dest_path.write_bytes(content)

        manager.start(str(dest_path))
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
            path, media_type, filename = manager.export_metrics(fmt)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return FileResponse(path, media_type=media_type, filename=filename)

    async def api_frame_jpg(request: Request) -> Response:
        view = request.query_params.get("view", "annotated")
        jpeg = manager.get_jpeg(view=view)
        return Response(content=jpeg, media_type="image/jpeg")

    async def stream_mjpeg(request: Request) -> StreamingResponse:
        view = request.query_params.get("view", "annotated")

        async def _frame_generator():
            while True:
                jpeg = manager.get_jpeg(view=view)
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                yield header + jpeg + b"\r\n"
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.045)

        return StreamingResponse(
            _frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/control", api_control, methods=["POST"]),
        Route("/api/upload", api_upload, methods=["POST"]),
        Route("/api/metrics", api_metrics, methods=["GET"]),
        Route("/api/export", api_export, methods=["GET"]),
        Route("/api/frame.jpg", api_frame_jpg, methods=["GET"]),
        Route("/stream/mjpeg", stream_mjpeg, methods=["GET"]),
        Mount("/static", StaticFiles(directory=str(_WEB_DIR), check_dir=False), name="static"),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.session_manager = manager
    return app
