"""Tests for FrameMetrics and MetricsLogger evaluation module."""

import csv
import json
from pathlib import Path
import sys

# Ensure workspace root is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.eval.metrics import FrameMetrics, MetricsLogger


def test_frame_metrics_dataclass_fields():
    """Verify FrameMetrics dataclass instantiation and attribute accessibility."""
    metric = FrameMetrics(
        frame_idx=1,
        timestamp_sec=100.5,
        mode="hybrid",
        fps=30.0,
        latency_ms=33.3,
        active_tracks=3,
        detections_count=4,
        total_tracks_created=5,
    )
    assert metric.frame_idx == 1
    assert metric.timestamp_sec == 100.5
    assert metric.mode == "hybrid"
    assert metric.fps == 30.0
    assert metric.latency_ms == 33.3
    assert metric.active_tracks == 3
    assert metric.detections_count == 4
    assert metric.total_tracks_created == 5


def test_metrics_logger_empty_summary():
    """Verify summary() on an unpopulated logger returns safe zeroed defaults."""
    logger = MetricsLogger()
    summary = logger.summary()

    assert summary["total_frames"] == 0
    assert summary["avg_fps"] == 0.0
    assert summary["avg_latency_ms"] == 0.0
    assert summary["peak_active_tracks"] == 0
    assert summary["total_tracks_created"] == 0
    assert summary["per_mode"] == {}
    assert logger.recent() == []


def test_metrics_logger_record_and_summary():
    """Verify record() correctly aggregates summary statistics and per-mode breakdown."""
    logger = MetricsLogger()

    frames = [
        FrameMetrics(
            frame_idx=1,
            timestamp_sec=0.1,
            mode="hybrid",
            fps=30.0,
            latency_ms=30.0,
            active_tracks=2,
            detections_count=2,
            total_tracks_created=2,
        ),
        FrameMetrics(
            frame_idx=2,
            timestamp_sec=0.2,
            mode="hybrid",
            fps=20.0,
            latency_ms=50.0,
            active_tracks=4,
            detections_count=3,
            total_tracks_created=4,
        ),
        FrameMetrics(
            frame_idx=3,
            timestamp_sec=0.3,
            mode="cvip",
            fps=50.0,
            latency_ms=20.0,
            active_tracks=1,
            detections_count=1,
            total_tracks_created=4,
        ),
    ]

    for f in frames:
        logger.record(f)

    summary = logger.summary()
    assert summary["total_frames"] == 3
    assert summary["avg_fps"] == pytest.approx((30.0 + 20.0 + 50.0) / 3.0, rel=1e-3)
    assert summary["avg_latency_ms"] == pytest.approx((30.0 + 50.0 + 20.0) / 3.0, rel=1e-3)
    assert summary["peak_active_tracks"] == 4
    assert summary["total_tracks_created"] == 4

    per_mode = summary["per_mode"]
    assert "hybrid" in per_mode
    assert per_mode["hybrid"]["frames"] == 2
    assert per_mode["hybrid"]["avg_fps"] == pytest.approx(25.0, rel=1e-3)
    assert per_mode["hybrid"]["avg_latency_ms"] == pytest.approx(40.0, rel=1e-3)

    assert "cvip" in per_mode
    assert per_mode["cvip"]["frames"] == 1
    assert per_mode["cvip"]["avg_fps"] == pytest.approx(50.0, rel=1e-3)
    assert per_mode["cvip"]["avg_latency_ms"] == pytest.approx(20.0, rel=1e-3)


def test_metrics_logger_recent():
    """Verify recent(limit) returns the last N records formatted as dictionaries."""
    logger = MetricsLogger()
    for i in range(10):
        logger.record(
            FrameMetrics(
                frame_idx=i,
                timestamp_sec=float(i),
                mode="hybrid",
                fps=30.0 + i,
                latency_ms=30.0 - i,
                active_tracks=i,
                detections_count=i,
                total_tracks_created=i,
            )
        )

    recent_3 = logger.recent(limit=3)
    assert len(recent_3) == 3
    assert isinstance(recent_3[0], dict)
    assert recent_3[0]["frame_idx"] == 7
    assert recent_3[1]["frame_idx"] == 8
    assert recent_3[2]["frame_idx"] == 9

    # Limit larger than total recorded
    all_recent = logger.recent(limit=50)
    assert len(all_recent) == 10
    assert all_recent[0]["frame_idx"] == 0


def test_metrics_logger_export_csv(tmp_path: Path):
    """Verify export_csv produces a valid CSV file with header and metric rows."""
    logger = MetricsLogger()
    logger.record(
        FrameMetrics(
            frame_idx=1,
            timestamp_sec=1.0,
            mode="hybrid",
            fps=29.97,
            latency_ms=33.3,
            active_tracks=2,
            detections_count=3,
            total_tracks_created=2,
        )
    )
    logger.record(
        FrameMetrics(
            frame_idx=2,
            timestamp_sec=2.0,
            mode="dl_sort",
            fps=31.5,
            latency_ms=31.7,
            active_tracks=3,
            detections_count=3,
            total_tracks_created=3,
        )
    )

    csv_path = tmp_path / "metrics.csv"
    logger.export_csv(csv_path)

    assert csv_path.exists()
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["frame_idx"] == "1"
        assert reader[0]["mode"] == "hybrid"
        assert float(reader[0]["fps"]) == pytest.approx(29.97, rel=1e-3)
        assert reader[1]["frame_idx"] == "2"
        assert reader[1]["mode"] == "dl_sort"


def test_metrics_logger_export_json(tmp_path: Path):
    """Verify export_json produces a structured JSON with summary and raw frame records."""
    logger = MetricsLogger()
    logger.record(
        FrameMetrics(
            frame_idx=1,
            timestamp_sec=1.5,
            mode="hybrid",
            fps=25.0,
            latency_ms=40.0,
            active_tracks=1,
            detections_count=2,
            total_tracks_created=1,
        )
    )

    json_path = tmp_path / "metrics.json"
    logger.export_json(json_path)

    assert json_path.exists()
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    assert "summary" in data
    assert "frames" in data
    assert data["summary"]["total_frames"] == 1
    assert data["summary"]["avg_fps"] == 25.0
    assert len(data["frames"]) == 1
    assert data["frames"][0]["frame_idx"] == 1
    assert data["frames"][0]["mode"] == "hybrid"


def test_metrics_logger_export_plot(tmp_path: Path):
    """Verify export_plot saves a valid non-empty PNG plot using Matplotlib Agg backend."""
    logger = MetricsLogger()
    for i in range(5):
        logger.record(
            FrameMetrics(
                frame_idx=i,
                timestamp_sec=float(i),
                mode="hybrid",
                fps=25.0 + i,
                latency_ms=35.0,
                active_tracks=i,
                detections_count=i,
                total_tracks_created=i,
            )
        )

    plot_path = tmp_path / "metrics_plot.png"
    logger.export_plot(plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_metrics_logger_export_plot_empty(tmp_path: Path):
    """Verify export_plot handles an empty logger gracefully without raising errors."""
    logger = MetricsLogger()
    plot_path = tmp_path / "empty_plot.png"
    logger.export_plot(plot_path)

    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_metrics_logger_clear_resets_state():
    """Verify clear() empties records and resets summary metrics."""
    logger = MetricsLogger()
    logger.record(
        FrameMetrics(
            frame_idx=0,
            timestamp_sec=0.01,
            mode="hybrid",
            fps=30.0,
            latency_ms=33.3,
            active_tracks=2,
            detections_count=2,
            total_tracks_created=5,
        )
    )
    assert logger.summary()["total_frames"] == 1
    logger.clear()
    summary = logger.summary()
    assert summary["total_frames"] == 0
    assert summary["total_tracks_created"] == 0
    assert logger.recent() == []
