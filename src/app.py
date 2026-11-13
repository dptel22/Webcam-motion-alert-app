"""CLI entrypoint and native OpenCV runner for the motion tracking application."""

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import uvicorn

from src.config import TrackerConfig, TrackingMode
from src.pipeline import TrackingPipeline, generate_sample_video
from src.web_server import TrackingSessionManager, create_app


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Hybrid CVIP + Deep Learning Multi-Object Motion Tracker",
    )
    parser.add_argument("--source", default="0", help="Camera index, video path, or 'sample'")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in TrackingMode],
        default=TrackingMode.HYBRID.value,
        help="Tracking pipeline mode",
    )
    parser.add_argument("--debug-view", action="store_true", help="Show 2x2 CVIP + tracking quad view")
    parser.add_argument("--save-video", default=None, help="Optional output video file path")
    parser.add_argument("--alerts", action="store_true", help="Enable email alerts on target tracks")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N processed frames")
    parser.add_argument("--web", action="store_true", help="Launch Starlette web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Web server bind host")
    parser.add_argument("--port", type=int, default=8000, help="Web server bind port")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def _export_all_metrics(pipeline: TrackingPipeline) -> None:
    out_dir = Path(pipeline.config.metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline.metrics_logger.export_csv(out_dir / "metrics.csv")
    pipeline.metrics_logger.export_json(out_dir / "metrics.json")
    pipeline.metrics_logger.export_plot(out_dir / "metrics_plot.png")


def _open_cli_capture(config: TrackerConfig) -> cv2.VideoCapture:
    if config.source == "sample":
        sample_path = Path(config.metrics_dir) / "sample_demo.mp4"
        if not sample_path.exists():
            generate_sample_video(sample_path)
        return cv2.VideoCapture(str(sample_path))
    if config.source.isdigit():
        return cv2.VideoCapture(int(config.source))
    return cv2.VideoCapture(config.source)


def run_cli(config: TrackerConfig) -> None:
    """Run the native OpenCV display loop with interactive hotkeys."""
    pipeline = TrackingPipeline(config)
    cap = _open_cli_capture(config)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video source: {config.source}")

    debug_view = config.debug_view
    writer: cv2.VideoWriter | None = None

    try:
        while True:
            if config.max_frames is not None and pipeline.frame_idx >= config.max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            result = pipeline.process_frame(frame)
            display = result.quad if debug_view else result.annotated

            if config.save_video:
                if writer is None:
                    dest = Path(config.save_video)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    h, w = display.shape[:2]
                    writer = cv2.VideoWriter(
                        str(dest),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        24.0,
                        (w, h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open VideoWriter for: {dest}")
                writer.write(display)

            cv2.imshow("Motion Tracker", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                pipeline.set_mode(pipeline.config.mode.next_mode())
            elif key == ord("v"):
                debug_view = not debug_view
            elif key == ord("c"):
                pipeline.update_settings(use_clahe=not pipeline.config.use_clahe)
            elif key == ord("r"):
                pipeline.reset()
            elif key == ord("s"):
                _export_all_metrics(pipeline)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if pipeline.frame_idx > 0:
            _export_all_metrics(pipeline)


def main(argv: Sequence[str] | None = None) -> None:
    """Application entrypoint dispatching to web server or native OpenCV CLI."""
    args = parse_args(argv)
    config = TrackerConfig(
        source=args.source,
        mode=TrackingMode(args.mode),
        debug_view=args.debug_view,
        save_video=args.save_video,
        enable_alerts=args.alerts,
        max_frames=args.max_frames,
    )
    if args.web:
        app = create_app(TrackingSessionManager(config))
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        run_cli(config)
