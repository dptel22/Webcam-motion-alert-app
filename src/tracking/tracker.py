"""DeepSORT-style Multi-Object Tracker with Kalman Filter, Hungarian Matching, and Re-ID."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import TrackingMode
from src.dl.detector import Detection
from src.tracking.kalman import KalmanBoxTracker


_INVALID_COST = 1e6


class TrackState(str, Enum):
    """Lifecycle states of a tracked object."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between two L2-normalized vectors."""
    similarity = float(np.dot(emb_a, emb_b))
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


@dataclass
class Track:
    """Represents a single tracked object across video frames."""

    track_id: int
    kf: KalmanBoxTracker
    class_id: int
    class_name: str
    confidence: float
    trail_length: int = 30
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    state: TrackState = TrackState.TENTATIVE
    embedding: np.ndarray | None = None
    history: deque[tuple[int, int]] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.trail_length)
        self._record_center(self.kf.current_bbox())

    def _record_center(self, bbox: np.ndarray) -> None:
        cx = int(round(0.5 * (bbox[0] + bbox[2])))
        cy = int(round(0.5 * (bbox[1] + bbox[3])))
        self.history.append((cx, cy))

    def bbox(self) -> np.ndarray:
        """Return current Kalman-smoothed bounding box [x1, y1, x2, y2]."""
        return self.kf.current_bbox()

    def predict(self) -> np.ndarray:
        """Advance Kalman state for a new frame."""
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update_from_detection(
        self,
        det: Detection,
        min_hits: int,
        ema_alpha: float = 0.8,
    ) -> None:
        """Update track state, class info, and EMA appearance embedding from a matched detection."""
        updated_box = self.kf.update(det.bbox)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = det.confidence
        self.class_id = det.class_id
        self.class_name = det.class_name
        if self.hits >= min_hits:
            self.state = TrackState.CONFIRMED

        if det.embedding is not None:
            if self.embedding is None:
                self.embedding = det.embedding.copy()
            else:
                blended = ema_alpha * self.embedding + (1.0 - ema_alpha) * det.embedding
                norm = float(np.linalg.norm(blended))
                self.embedding = (blended / norm) if norm > 1e-8 else det.embedding.copy()

        self._record_center(updated_box)

    def update_from_flow(self, flow_bbox: np.ndarray) -> None:
        """Apply a higher-noise optical-flow observation between DL detection keyframes."""
        updated_box = self.kf.update(flow_bbox, measurement_scale=2.5)
        self.time_since_update = 0
        self._record_center(updated_box)


class MultiObjectTracker:
    """Tracks multiple objects using Kalman prediction, Hungarian matching, and optional Re-ID."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_weight: float = 0.5,
        reid_weight: float = 0.5,
        max_iou_distance: float = 0.75,
        max_cosine_distance: float = 0.45,
        ema_alpha: float = 0.8,
        trail_length: int = 30,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_weight = iou_weight
        self.reid_weight = reid_weight
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.ema_alpha = ema_alpha
        self.trail_length = trail_length

        self.tracks: list[Track] = []
        self._next_id = 1
        self.total_tracks_created = 0

    def reset(self) -> None:
        """Clear all active tracks and reset ID counter."""
        self.tracks.clear()
        self._next_id = 1
        self.total_tracks_created = 0

    def _build_cost_matrix(
        self,
        detections: list[Detection],
        use_reid: bool,
    ) -> np.ndarray:
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        cost = np.full((num_tracks, num_dets), _INVALID_COST, dtype=np.float32)

        for t_idx, track in enumerate(self.tracks):
            t_box = track.bbox()
            for d_idx, det in enumerate(detections):
                iou_dist = 1.0 - compute_iou(t_box, det.bbox)

                has_embeddings = (
                    use_reid
                    and track.embedding is not None
                    and det.embedding is not None
                )
                if has_embeddings:
                    cos_dist = cosine_distance(track.embedding, det.embedding)
                    # Allow strong Re-ID recovery even after brief occlusion (when IoU drops)
                    spatial_ok = iou_dist <= self.max_iou_distance
                    reid_recovery_ok = cos_dist <= (self.max_cosine_distance * 0.75)
                    if not (spatial_ok or reid_recovery_ok):
                        continue
                    cost[t_idx, d_idx] = self.iou_weight * iou_dist + self.reid_weight * cos_dist
                else:
                    if iou_dist <= self.max_iou_distance:
                        cost[t_idx, d_idx] = iou_dist

        return cost

    def _prune_stale_tracks(self, drop_tentative: bool = True) -> None:
        kept: list[Track] = []
        for track in self.tracks:
            if track.time_since_update == 0:
                kept.append(track)
                continue
            if track.state == TrackState.TENTATIVE:
                if not drop_tentative:
                    kept.append(track)
                continue
            track.state = TrackState.LOST
            if track.time_since_update <= self.max_age:
                kept.append(track)
        self.tracks = kept

    def _spawn_track(self, det: Detection) -> Track:
        state = TrackState.CONFIRMED if self.min_hits <= 1 else TrackState.TENTATIVE
        track = Track(
            track_id=self._next_id,
            kf=KalmanBoxTracker(det.bbox),
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            trail_length=self.trail_length,
            state=state,
            embedding=det.embedding.copy() if det.embedding is not None else None,
        )
        self._next_id += 1
        self.total_tracks_created += 1
        return track

    def step(
        self,
        detections: list[Detection],
        mode: TrackingMode = TrackingMode.HYBRID,
    ) -> list[Track]:
        """Run one predict-associate-update cycle with new detections."""
        for track in self.tracks:
            track.predict()

        # Re-ID appearance cost is fused whenever detections carry embeddings (DL_SORT and
        # HYBRID). CVIP motion blobs never have embeddings, so it is IoU-only there.
        use_reid = mode != TrackingMode.CVIP_MOG2
        unmatched_dets = set(range(len(detections)))

        if self.tracks and detections:
            cost_matrix = self._build_cost_matrix(detections, use_reid=use_reid)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for t_idx, d_idx in zip(row_ind, col_ind):
                if cost_matrix[t_idx, d_idx] >= _INVALID_COST * 0.5:
                    continue
                self.tracks[t_idx].update_from_detection(
                    detections[d_idx],
                    min_hits=self.min_hits,
                    ema_alpha=self.ema_alpha,
                )
                unmatched_dets.discard(d_idx)

        self._prune_stale_tracks()

        for d_idx in sorted(unmatched_dets):
            self.tracks.append(self._spawn_track(detections[d_idx]))

        return self.active_tracks()

    def step_with_flow(self, flow_boxes_by_id: dict[int, np.ndarray]) -> list[Track]:
        """Advance tracks on non-keyframe steps using Kalman + Optical Flow shifts."""
        for track in self.tracks:
            track.predict()
            if track.track_id in flow_boxes_by_id and track.state in (
                TrackState.CONFIRMED,
                TrackState.TENTATIVE,
            ):
                track.update_from_flow(flow_boxes_by_id[track.track_id])

        # Do not drop tentative tracks on non-keyframe flow steps before next detector check.
        self._prune_stale_tracks(drop_tentative=False)
        return self.active_tracks()

    def step_predict(self) -> list[Track]:
        """Advance all tracks with Kalman prediction only (no detector, no measurement).

        Used on non-detection frames in DL_SORT mode (detect_interval > 1). Confirmed tracks
        coast on their predicted boxes; nothing is pruned until the next detection step.
        """
        for track in self.tracks:
            track.predict()
            if track.state == TrackState.CONFIRMED:
                track._record_center(track.bbox())
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def active_tracks(self) -> list[Track]:
        """Return currently confirmed tracks visible in the current frame."""
        return [
            t for t in self.tracks
            if t.state == TrackState.CONFIRMED and t.time_since_update == 0
        ]

    def tracked_boxes_by_id(self, include_tentative: bool = False) -> dict[int, np.ndarray]:
        """Return {track_id: bbox} for visible tracks (used by Optical Flow)."""
        allowed = (TrackState.CONFIRMED, TrackState.TENTATIVE) if include_tentative else (TrackState.CONFIRMED,)
        return {
            t.track_id: t.bbox()
            for t in self.tracks
            if t.state in allowed and t.time_since_update == 0
        }
