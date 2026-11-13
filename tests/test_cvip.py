"""Unit tests for CVIP preprocessing, lighting normalization, and optical flow tracking."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from src.cvip.optical_flow import FlowResult, OpticalFlowTracker
from src.cvip.preprocessing import CVIPPreprocessor, PreprocessedFrame


def test_clahe_enhances_contrast_and_preserves_shape_dtype():
    """Verify CLAHE improves contrast while preserving image dimensions and uint8 dtype."""
    preprocessor = CVIPPreprocessor(use_clahe=True, clahe_clip_limit=3.0)

    # Low-contrast synthetic BGR frame with values restricted to [100, 115]
    low_contrast = np.full((120, 160, 3), 108, dtype=np.uint8)
    low_contrast[30:90, 40:120] = 114

    enhanced = preprocessor.apply_clahe(low_contrast)

    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == low_contrast.shape
    assert enhanced.dtype == np.uint8
    # Standard deviation should increase after CLAHE contrast equalization
    assert float(enhanced.std()) > float(low_contrast.std())
    assert int(enhanced.max() - enhanced.min()) >= int(low_contrast.max() - low_contrast.min())


def test_clahe_disabled_returns_identical_frame():
    """Verify apply_clahe returns unmodified image when use_clahe is False."""
    preprocessor = CVIPPreprocessor(use_clahe=False)
    frame = np.random.randint(50, 150, size=(80, 80, 3), dtype=np.uint8)
    result = preprocessor.apply_clahe(frame)
    np.testing.assert_array_equal(result, frame)


def test_mog2_detects_moving_bright_rectangle_across_consecutive_frames():
    """Verify CVIPPreprocessor with MOG2 detects a moving rectangle across frames."""
    preprocessor = CVIPPreprocessor(
        use_clahe=True,
        blur_kernel=5,
        mog2_history=100,
        mog2_var_threshold=25.0,
        min_contour_area=1000,
    )

    frame_h, frame_w = 240, 320
    bg = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Prime MOG2 background subtractor with static background frames
    for _ in range(8):
        preprocessor.process(bg, extract_motion=True)

    rect_w, rect_h = 60, 60
    # Frame 1: rectangle at (x=60, y=80)
    x1_f1, y1_f1 = 60, 80
    frame_t1 = bg.copy()
    cv2.rectangle(
        frame_t1,
        (x1_f1, y1_f1),
        (x1_f1 + rect_w, y1_f1 + rect_h),
        (255, 255, 255),
        -1,
    )

    res_t1 = preprocessor.process(frame_t1, extract_motion=True)
    assert isinstance(res_t1, PreprocessedFrame)
    assert res_t1.enhanced.shape == (frame_h, frame_w, 3)
    assert res_t1.fg_mask.shape == (frame_h, frame_w)
    assert len(res_t1.motion_boxes) >= 1

    box1 = res_t1.motion_boxes[0]
    # Check that the detected motion box covers the moving rectangle
    cx1 = 0.5 * (box1[0] + box1[2])
    cy1 = 0.5 * (box1[1] + box1[3])
    expected_cx1 = x1_f1 + 0.5 * rect_w
    expected_cy1 = y1_f1 + 0.5 * rect_h
    assert abs(cx1 - expected_cx1) < 10.0
    assert abs(cy1 - expected_cy1) < 10.0

    # Frame 2: rectangle moved to (x=110, y=80)
    x1_f2, y1_f2 = 110, 80
    frame_t2 = bg.copy()
    cv2.rectangle(
        frame_t2,
        (x1_f2, y1_f2),
        (x1_f2 + rect_w, y1_f2 + rect_h),
        (255, 255, 255),
        -1,
    )

    res_t2 = preprocessor.process(frame_t2, extract_motion=True)
    assert len(res_t2.motion_boxes) >= 1

    # Find the box covering the new position
    box2 = min(
        res_t2.motion_boxes,
        key=lambda b: abs(0.5 * (b[0] + b[2]) - (x1_f2 + 0.5 * rect_w)),
    )
    cx2 = 0.5 * (box2[0] + box2[2])
    cy2 = 0.5 * (box2[1] + box2[3])
    expected_cx2 = x1_f2 + 0.5 * rect_w
    expected_cy2 = y1_f2 + 0.5 * rect_h
    assert abs(cx2 - expected_cx2) < 10.0
    assert abs(cy2 - expected_cy2) < 10.0


def test_cvip_process_extract_motion_false():
    """Verify process() with extract_motion=False returns zero mask and empty boxes."""
    preprocessor = CVIPPreprocessor()
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    result = preprocessor.process(frame, extract_motion=False)
    assert len(result.motion_boxes) == 0
    assert np.all(result.fg_mask == 0)


def test_cvip_reset_background():
    """Verify reset_background reinitializes the internal MOG2 subtractor."""
    preprocessor = CVIPPreprocessor(mog2_history=200, mog2_var_threshold=40.0)
    preprocessor.reset_background()
    assert preprocessor._bg_subtractor.getHistory() == 200
    assert preprocessor._bg_subtractor.getVarThreshold() == 40.0


def test_optical_flow_estimate_box_shifts_textured_patch():
    """Verify OpticalFlowTracker shifts box by the true (dx, dy) translation of a textured patch."""
    tracker = OpticalFlowTracker(
        max_corners_per_box=30,
        quality_level=0.02,
        min_distance=4,
    )

    height, width = 180, 220
    prev_gray = np.zeros((height, width), dtype=np.uint8)

    # Create a high-contrast random textured patch
    np.random.seed(1337)
    patch_size = 50
    patch = np.random.randint(40, 240, size=(patch_size, patch_size), dtype=np.uint8)

    # Place patch in prev_gray at [40, 40, 90, 90]
    x_orig, y_orig = 40, 40
    prev_gray[y_orig : y_orig + patch_size, x_orig : x_orig + patch_size] = patch

    # Shift patch by (dx, dy) = (8, -5) in curr_gray
    dx_true, dy_true = 8, -5
    curr_gray = np.zeros((height, width), dtype=np.uint8)
    curr_gray[
        y_orig + dy_true : y_orig + dy_true + patch_size,
        x_orig + dx_true : x_orig + dx_true + patch_size,
    ] = patch

    input_box = np.array(
        [x_orig, y_orig, x_orig + patch_size, y_orig + patch_size],
        dtype=np.float32,
    )

    result: FlowResult = tracker.estimate_box_shifts(prev_gray, curr_gray, [input_box])

    assert len(result.shifted_boxes) == 1
    shifted = result.shifted_boxes[0]

    # Expected shifted box
    expected_box = input_box + np.array([dx_true, dy_true, dx_true, dy_true], dtype=np.float32)
    # Check that shifted box is within 1.0 pixel of true displacement
    np.testing.assert_allclose(shifted, expected_box, atol=1.0)

    # Flow vector segments should be populated
    assert len(result.vectors) > 5
    for (ox, oy), (nx, ny) in result.vectors:
        assert 0 <= ox < width and 0 <= oy < height
        assert 0 <= nx < width and 0 <= ny < height
        # Individual point displacements should be close to (dx_true, dy_true)
        assert abs((nx - ox) - dx_true) <= 1
        assert abs((ny - oy) - dy_true) <= 1


def test_optical_flow_empty_boxes_and_clamp():
    """Verify estimate_box_shifts handles empty box lists and clamps boundary boxes."""
    tracker = OpticalFlowTracker()
    prev_gray = np.zeros((100, 100), dtype=np.uint8)
    curr_gray = np.zeros((100, 100), dtype=np.uint8)

    # Empty boxes
    empty_res = tracker.estimate_box_shifts(prev_gray, curr_gray, [])
    assert empty_res.shifted_boxes == []
    assert empty_res.vectors == []

    # Box near boundary clamped correctly
    clamped = tracker._clamp_box(np.array([95.0, 95.0, 105.0, 105.0], dtype=np.float32), 100, 100)
    assert clamped[2] <= 100.0
    assert clamped[3] <= 100.0
    assert clamped[0] >= 0.0
    assert clamped[1] >= 0.0
