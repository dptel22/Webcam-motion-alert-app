"""Unit tests for MultiObjectTracker association, occlusion recovery, flow steps, and ReID."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.config import TrackingMode
from src.dl.detector import Detection
from src.dl.reid import ReIDExtractor
from src.tracking.tracker import MultiObjectTracker, TrackState, compute_iou, cosine_distance


@pytest.mark.parametrize("mode", [TrackingMode.DL_SORT, TrackingMode.CVIP_MOG2])
def test_tracker_assigns_persistent_ids_across_frames_two_objects(mode: TrackingMode):
    """Verify MultiObjectTracker assigns persistent track_ids across frames (min_hits=2)."""
    tracker = MultiObjectTracker(min_hits=2)

    # Initial detections for two distinct objects
    box_a = np.array([20.0, 30.0, 60.0, 80.0], dtype=np.float32)
    box_b = np.array([150.0, 160.0, 200.0, 210.0], dtype=np.float32)

    # Frame 1: Initial detections -> spawned as tentative (not yet active)
    det_a1 = Detection(bbox=box_a, confidence=0.9, class_id=0, class_name="person")
    det_b1 = Detection(bbox=box_b, confidence=0.85, class_id=1, class_name="bicycle")
    active_f1 = tracker.step([det_a1, det_b1], mode=mode)
    assert len(active_f1) == 0

    # Frame 2: Objects move slightly (hits=2 >= min_hits -> confirmed)
    shift = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    det_a2 = Detection(bbox=box_a + shift, confidence=0.92, class_id=0, class_name="person")
    det_b2 = Detection(bbox=box_b + shift, confidence=0.88, class_id=1, class_name="bicycle")
    active_f2 = tracker.step([det_a2, det_b2], mode=mode)
    assert len(active_f2) == 2
    ids_f2 = {t.track_id for t in active_f2}
    assert len(ids_f2) == 2

    # Frame 3: Objects move slightly further
    det_a3 = Detection(bbox=box_a + 2 * shift, confidence=0.91, class_id=0, class_name="person")
    det_b3 = Detection(bbox=box_b + 2 * shift, confidence=0.87, class_id=1, class_name="bicycle")
    active_f3 = tracker.step([det_a3, det_b3], mode=mode)
    assert len(active_f3) == 2
    ids_f3 = {t.track_id for t in active_f3}

    # IDs must persist across frames
    assert ids_f2 == ids_f3

    # Frame 4: Another step
    det_a4 = Detection(bbox=box_a + 3 * shift, confidence=0.93, class_id=0, class_name="person")
    det_b4 = Detection(bbox=box_b + 3 * shift, confidence=0.89, class_id=1, class_name="bicycle")
    active_f4 = tracker.step([det_a4, det_b4], mode=mode)
    assert len(active_f4) == 2
    ids_f4 = {t.track_id for t in active_f4}
    assert ids_f4 == ids_f2


def test_hybrid_mode_occlusion_recovery_with_matching_reid():
    """Verify HYBRID mode recovers the same track_id after a 3-frame occlusion with matching Re-ID."""
    tracker = MultiObjectTracker(min_hits=2, max_age=30, max_cosine_distance=0.45)

    # Unit L2-normalized embedding
    emb_target = np.zeros(576, dtype=np.float32)
    emb_target[0] = 1.0

    box_init = np.array([50.0, 50.0, 90.0, 100.0], dtype=np.float32)
    d1 = Detection(bbox=box_init, confidence=0.9, class_id=0, class_name="person", embedding=emb_target)
    tracker.step([d1], mode=TrackingMode.HYBRID)

    d2 = Detection(
        bbox=box_init + np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float32),
        confidence=0.92,
        class_id=0,
        class_name="person",
        embedding=emb_target,
    )
    active = tracker.step([d2], mode=TrackingMode.HYBRID)
    assert len(active) == 1
    orig_id = active[0].track_id
    assert active[0].state == TrackState.CONFIRMED

    # 3-frame occlusion: zero detections fed
    for _ in range(3):
        res = tracker.step([], mode=TrackingMode.HYBRID)
        assert len(res) == 0

    # Target track is now in LOST state, but retained because time_since_update (3) <= max_age (30)
    lost_tracks = [t for t in tracker.tracks if t.track_id == orig_id]
    assert len(lost_tracks) == 1
    assert lost_tracks[0].state == TrackState.LOST
    assert lost_tracks[0].time_since_update == 3

    # Object reappears slightly shifted with matching Re-ID embedding
    # Note: after 3 frames of prediction, the Kalman predicted box moved, so IoU distance
    # might be higher, but Re-ID cosine distance is near 0.0, allowing recovery.
    d_reappear = Detection(
        bbox=box_init + np.array([15.0, 5.0, 15.0, 5.0], dtype=np.float32),
        confidence=0.88,
        class_id=0,
        class_name="person",
        embedding=emb_target,
    )
    recovered_active = tracker.step([d_reappear], mode=TrackingMode.HYBRID)

    assert len(recovered_active) == 1
    assert recovered_active[0].track_id == orig_id
    assert recovered_active[0].state == TrackState.CONFIRMED


def test_hybrid_mode_different_embedding_after_occlusion_assigns_new_id():
    """Verify HYBRID mode assigns a new track_id when an object with different Re-ID appears."""
    tracker = MultiObjectTracker(min_hits=2, max_age=30, max_cosine_distance=0.45)

    emb_target = np.zeros(576, dtype=np.float32)
    emb_target[0] = 1.0

    box_init = np.array([50.0, 50.0, 90.0, 100.0], dtype=np.float32)
    d1 = Detection(bbox=box_init, confidence=0.9, class_id=0, class_name="person", embedding=emb_target)
    tracker.step([d1], mode=TrackingMode.HYBRID)

    d2 = Detection(
        bbox=box_init + np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        confidence=0.9,
        class_id=0,
        class_name="person",
        embedding=emb_target,
    )
    active = tracker.step([d2], mode=TrackingMode.HYBRID)
    orig_id = active[0].track_id

    # 3-frame occlusion
    for _ in range(3):
        tracker.step([], mode=TrackingMode.HYBRID)

    # Completely different orthogonal embedding (cosine_distance = 1.0 > 0.45)
    # and placed with no spatial overlap to avoid IoU match
    emb_different = np.zeros(576, dtype=np.float32)
    emb_different[1] = 1.0

    d_different = Detection(
        bbox=np.array([200.0, 200.0, 240.0, 250.0], dtype=np.float32),
        confidence=0.85,
        class_id=0,
        class_name="person",
        embedding=emb_different,
    )
    tracker.step([d_different], mode=TrackingMode.HYBRID)

    # The new detection should not match orig_id; a new tentative track must be spawned
    active_ids = {t.track_id for t in tracker.tracks if t.track_id != orig_id}
    assert len(active_ids) == 1
    new_id = list(active_ids)[0]
    assert new_id != orig_id

    # Confirm original track remained unrecovered
    orig_track = next(t for t in tracker.tracks if t.track_id == orig_id)
    assert orig_track.state == TrackState.LOST


def test_step_with_flow_advances_confirmed_tracks_without_dropping():
    """Verify step_with_flow updates confirmed tracks and retains them in active_tracks."""
    tracker = MultiObjectTracker(min_hits=2)

    box = np.array([40.0, 40.0, 80.0, 90.0], dtype=np.float32)
    d1 = Detection(bbox=box, confidence=0.9, class_id=0, class_name="person")
    tracker.step([d1])
    d2 = Detection(
        bbox=box + np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )
    active = tracker.step([d2])
    assert len(active) == 1
    track_id = active[0].track_id

    # Advance using optical flow box shift
    flow_shifted_box = box + np.array([5.0, 4.0, 5.0, 4.0], dtype=np.float32)
    flow_map = {track_id: flow_shifted_box}

    active_flow = tracker.step_with_flow(flow_map)
    assert len(active_flow) == 1
    assert active_flow[0].track_id == track_id
    assert active_flow[0].state == TrackState.CONFIRMED
    assert active_flow[0].time_since_update == 0

    # Ensure tracked_boxes_by_id reflects the active confirmed track
    boxes_by_id = tracker.tracked_boxes_by_id()
    assert track_id in boxes_by_id
    assert boxes_by_id[track_id].shape == (4,)


def test_reid_extractor_pretrained_false_generates_normalized_embeddings_and_skips_tiny():
    """Verify ReIDExtractor(pretrained=False) extracts L2-normalized (576,) vectors and skips <4px boxes."""
    extractor = ReIDExtractor(input_size=(128, 128), pretrained=False, device="cpu")

    # Synthetic BGR frame
    frame = np.full((160, 160, 3), 120, dtype=np.uint8)
    frame[30:90, 30:90] = [200, 50, 50]

    valid_box1 = np.array([20.0, 20.0, 80.0, 80.0], dtype=np.float32)   # 60x60
    valid_box2 = np.array([90.0, 90.0, 130.0, 140.0], dtype=np.float32) # 40x50
    tiny_box_w = np.array([10.0, 10.0, 12.0, 50.0], dtype=np.float32)   # width = 2 (< 4)
    tiny_box_h = np.array([10.0, 10.0, 50.0, 13.0], dtype=np.float32)   # height = 3 (< 4)
    tiny_box_wh = np.array([10.0, 10.0, 11.0, 11.0], dtype=np.float32)  # 1x1 (< 4)

    embeddings = extractor.extract(
        frame,
        [valid_box1, valid_box2, tiny_box_w, tiny_box_h, tiny_box_wh],
    )

    assert len(embeddings) == 5

    # Valid boxes should produce (576,) float32 vectors with L2 norm ~ 1.0
    for idx in (0, 1):
        emb = embeddings[idx]
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (576,)
        assert emb.dtype == np.float32
        norm = float(np.linalg.norm(emb))
        assert np.isclose(norm, 1.0, atol=1e-5), f"Embedding norm {norm} is not 1.0"

    # Tiny boxes should be skipped and return None
    assert embeddings[2] is None
    assert embeddings[3] is None
    assert embeddings[4] is None

    # Empty box list returns empty list
    assert extractor.extract(frame, []) == []
