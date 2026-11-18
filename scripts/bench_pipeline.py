r"""Benchmark TrackingPipeline per mode on the sample video.

Usage:
    .venvScriptspython.exe scripts/bench_pipeline.py [--frames N] [--modes cvip dl_sort hybrid]

Runs N frames of outputs/sample_demo.mp4 (generated if missing) through each mode and
prints mean ms/frame and FPS. A few warm-up frames per mode (model loading, MOG2
initialisation) are excluded from timing.
"""

import argparse
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.config import TrackerConfig, TrackingMode  # noqa: E402
from src.pipeline import TrackingPipeline, generate_sample_video  # noqa: E402


def _load_frames(path: Path, limit: int) -> list:
    if not path.exists():
        generate_sample_video(path)
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < limit:
            ok, frame = cap.read()
            if not ok:
                if not frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {path}")
    return frames


def bench_mode(mode: TrackingMode, frames: list, warmup: int) -> tuple[float, float]:
    cfg = TrackerConfig(mode=mode, enable_alerts=False)
    pipeline = TrackingPipeline(cfg)
    for frame in frames[:warmup]:
        pipeline.process_frame(frame)
    timings = []
    for frame in frames:
        t0 = time.perf_counter()
        result = pipeline.process_frame(frame)
        _ = result.annotated  # the default dashboard view
        timings.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = statistics.fmean(timings)
    return mean_ms, 1000.0 / mean_ms


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--video", default=str(ROOT / "outputs" / "sample_demo.mp4"))
    parser.add_argument("--modes", nargs="*", default=[m.value for m in TrackingMode])
    args = parser.parse_args(argv)

    frames = _load_frames(Path(args.video), args.frames)
    h, w = frames[0].shape[:2]
    print(f"Video: {args.video} ({w}x{h}), {len(frames)} timed frames, {args.warmup} warm-up")
    print(f"{'mode':<10}{'ms/frame':>12}{'FPS':>10}")
    for name in args.modes:
        mean_ms, fps = bench_mode(TrackingMode(name), frames, args.warmup)
        print(f"{name:<10}{mean_ms:>12.2f}{fps:>10.1f}")


if __name__ == "__main__":
    main()
