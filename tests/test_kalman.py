"""Unit tests for Kalman box tracking, state conversions, and coordinate integrity."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.tracking.kalman import KalmanBoxTracker, bbox_to_z, x_to_bbox


def test_bbox_to_z_and_x_to_bbox_roundtrip_preservation():
    """Verify roundtrip conversion between [x1, y1, x2, y2] and measurement vector."""
    test_boxes = [
        np.array([10.0, 20.0, 50.0, 80.0], dtype=np.float32),     # standard rectangle
        np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32),      # square box
        np.array([50.0, 30.0, 250.0, 60.0], dtype=np.float32),     # wide aspect ratio
        np.array([120.0, 40.0, 150.0, 340.0], dtype=np.float32),   # tall aspect ratio
    ]

    for box in test_boxes:
        z = bbox_to_z(box)
        assert z.shape == (4, 1)

        # Mathematical sanity check on measurement components: [cx, cy, aspect, h]
        w = box[2] - box[0]
        h = box[3] - box[1]
        np.testing.assert_allclose(z[0, 0], box[0] + 0.5 * w, atol=1e-5)
        np.testing.assert_allclose(z[1, 0], box[1] + 0.5 * h, atol=1e-5)
        np.testing.assert_allclose(z[2, 0], w / h, atol=1e-5)
        np.testing.assert_allclose(z[3, 0], h, atol=1e-5)

        # Roundtrip recovery
        recovered = x_to_bbox(z)
        assert recovered.shape == (4,)
        np.testing.assert_allclose(recovered, box, atol=1e-4)


def test_conversions_never_produce_inverted_boxes():
    """Verify x_to_bbox and bbox_to_z never produce inverted boxes (x2 > x1, y2 > y1)."""
    # Degenerate cases: inverted coordinates, zero-area boxes, and negative dimensions
    inverted_or_degenerate_inputs = [
        np.array([100.0, 100.0, 50.0, 50.0], dtype=np.float32),   # completely inverted
        np.array([60.0, 30.0, 40.0, 90.0], dtype=np.float32),     # inverted x only
        np.array([30.0, 90.0, 80.0, 20.0], dtype=np.float32),     # inverted y only
        np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32),     # zero width and height
    ]

    for degenerate in inverted_or_degenerate_inputs:
        z = bbox_to_z(degenerate)
        box = x_to_bbox(z)
        # Bounding box must satisfy x2 > x1 and y2 > y1
        assert box[2] > box[0], f"Expected x2 > x1, got {box[0]} to {box[2]}"
        assert box[3] > box[1], f"Expected y2 > y1, got {box[1]} to {box[3]}"

    # Negative / extreme states directly passed to x_to_bbox
    extreme_states = [
        np.array([[-50.0], [-50.0], [-10.0], [-20.0]], dtype=np.float32),
        np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32),
        np.array([[100.0], [100.0], [1e-6], [0.01]], dtype=np.float32),
    ]

    for state in extreme_states:
        box = x_to_bbox(state)
        assert box[2] > box[0]
        assert box[3] > box[1]


def test_kalman_linear_motion_velocity_convergence_and_prediction():
    """Verify KalmanBoxTracker converges to true velocity and anticipates next position."""
    initial_bbox = np.array([100.0, 100.0, 150.0, 200.0], dtype=np.float32)
    tracker = KalmanBoxTracker(initial_bbox)

    true_vx = 5.0
    true_vy = -2.0
    displacement = np.array([true_vx, true_vy, true_vx, true_vy], dtype=np.float32)

    curr_bbox = initial_bbox.copy()

    # Feed 10 frames of constant linear motion
    for frame_idx in range(1, 11):
        curr_bbox = curr_bbox + displacement
        predicted = tracker.predict()
        assert predicted.shape == (4,)
        updated = tracker.update(curr_bbox)
        assert updated.shape == (4,)

    # Verify velocity estimate has converged near true (vx, vy)
    est_vx, est_vy = tracker.velocity()
    assert abs(est_vx - true_vx) < 0.2, f"Estimated vx {est_vx} did not converge near {true_vx}"
    assert abs(est_vy - true_vy) < 0.2, f"Estimated vy {est_vy} did not converge near {true_vy}"

    # Verify predict() anticipates the next (11th) position within a few pixels
    next_anticipated = tracker.predict()
    expected_next = curr_bbox + displacement
    max_error = float(np.max(np.abs(next_anticipated - expected_next)))
    assert max_error < 1.0, f"Anticipated position error {max_error} exceeded threshold"


def test_kalman_stationary_target():
    """Verify KalmanBoxTracker velocity converges near zero for stationary objects."""
    box = np.array([50.0, 50.0, 90.0, 90.0], dtype=np.float32)
    tracker = KalmanBoxTracker(box)

    for _ in range(8):
        tracker.predict()
        tracker.update(box)

    vx, vy = tracker.velocity()
    assert abs(vx) < 0.1
    assert abs(vy) < 0.1
    np.testing.assert_allclose(tracker.current_bbox(), box, atol=0.5)
