"""Runtime metrics logging, statistical aggregation, and evaluation export."""

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


@dataclass
class FrameMetrics:
    """Performance and tracking metrics recorded for a single processed frame."""

    frame_idx: int
    timestamp_sec: float
    mode: str
    fps: float
    latency_ms: float
    active_tracks: int
    detections_count: int
    total_tracks_created: int


class MetricsLogger:
    """Thread-safe telemetry logger aggregating frame metrics and exporting reports."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: list[FrameMetrics] = []
        self._reset_aggregates()

    def _reset_aggregates(self) -> None:
        # Running aggregates keep summary() O(#modes) instead of O(#frames); the
        # dashboard polls it several times per second during long sessions.
        self._sum_fps = 0.0
        self._sum_latency = 0.0
        self._peak_active = 0
        self._max_created = 0
        self._per_mode: dict[str, list[float]] = {}  # mode -> [frames, sum_fps, sum_latency]

    def record(self, metric: FrameMetrics) -> None:
        """Append a single frame's metric record."""
        with self._lock:
            self._records.append(metric)
            self._sum_fps += metric.fps
            self._sum_latency += metric.latency_ms
            self._peak_active = max(self._peak_active, metric.active_tracks)
            self._max_created = max(self._max_created, metric.total_tracks_created)
            agg = self._per_mode.setdefault(metric.mode, [0, 0.0, 0.0])
            agg[0] += 1
            agg[1] += metric.fps
            agg[2] += metric.latency_ms

    def clear(self) -> None:
        """Clear all recorded metrics."""
        with self._lock:
            self._records.clear()
            self._reset_aggregates()

    def summary(self) -> dict[str, Any]:
        """Compute aggregate benchmark metrics across all recorded frames and modes."""
        with self._lock:
            total_frames = len(self._records)
            if total_frames == 0:
                return {
                    "total_frames": 0,
                    "avg_fps": 0.0,
                    "avg_latency_ms": 0.0,
                    "peak_active_tracks": 0,
                    "total_tracks_created": 0,
                    "per_mode": {},
                }
            per_mode = {
                mode_name: {
                    "frames": int(count),
                    "avg_fps": sum_fps / count if count > 0 else 0.0,
                    "avg_latency_ms": sum_lat / count if count > 0 else 0.0,
                }
                for mode_name, (count, sum_fps, sum_lat) in self._per_mode.items()
            }
            return {
                "total_frames": total_frames,
                "avg_fps": self._sum_fps / total_frames,
                "avg_latency_ms": self._sum_latency / total_frames,
                "peak_active_tracks": self._peak_active,
                "total_tracks_created": self._max_created,
                "per_mode": per_mode,
            }

    def recent(self, limit: int = 60) -> list[dict[str, Any]]:
        """Return the last N records as plain dictionaries for UI/dashboard feeds."""
        if limit <= 0:
            return []
        with self._lock:
            return [asdict(m) for m in self._records[-limit:]]

    def export_csv(self, path: str | Path) -> None:
        """Write all recorded frame metrics to a CSV file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)

        fieldnames = [
            "frame_idx",
            "timestamp_sec",
            "mode",
            "fps",
            "latency_ms",
            "active_tracks",
            "detections_count",
            "total_tracks_created",
        ]

        with open(dest, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

    def export_json(self, path: str | Path) -> None:
        """Write summary metrics and full frame history to a JSON file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)
            summary = self.summary()

        data = {
            "summary": summary,
            "frames": [asdict(m) for m in records],
        }

        with open(dest, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def export_plot(self, path: str | Path) -> None:
        """Save a dual-panel PNG plot of FPS and Active Tracks using thread-safe FigureCanvasAgg."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)

        fig = Figure(figsize=(10, 6))
        FigureCanvasAgg(fig)
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        if records:
            frames = [m.frame_idx for m in records]
            fps = [m.fps for m in records]
            tracks = [m.active_tracks for m in records]

            ax1.plot(frames, fps, color="tab:blue", label="FPS", linewidth=1.5)
            ax1.set_ylabel("FPS")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.plot(frames, tracks, color="tab:orange", label="Active Tracks", linewidth=1.5)
            ax2.set_xlabel("Frame Index")
            ax2.set_ylabel("Active Tracks")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")
        else:
            ax1.text(0.5, 0.5, "No metrics recorded", ha="center", va="center")
            ax2.text(0.5, 0.5, "No metrics recorded", ha="center", va="center")

        fig.suptitle("Visual Tracking Runtime Metrics")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
