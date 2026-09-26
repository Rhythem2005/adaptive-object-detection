"""Phase 6: Multi-Object Tracking via Kalman Filter + Hungarian Assignment.

Implements a SORT-family tracker that operates on post-NMS detections from the
Phase 5 Selective Re-Detection pipeline. Each track maintains a constant-velocity
Kalman filter with state [cx, cy, w, h, vx, vy, vw, vh].

Pipeline position (preserved from existing design):
    Frame Controller → Illumination → YOLO → Confidence Filter
        → Selective Re-Detection → Fusion/NMS → **Tracking** → Telemetry

Design decisions:
    - Uses actual elapsed Δt between processed frames (from capture_ts), not a
      fixed 1/30s, because frame-to-frame timing is variable due to inference
      latency variation (43–118ms) and frame drops in the adaptive controller.
    - Association uses SciPy's linear_sum_assignment (Hungarian algorithm) with
      class-aware IoU gating: only detections and tracks of the same class are
      eligible for matching, and pairs below the IoU threshold are rejected.
    - Standard SORT-family defaults: max_age=30, min_hits=3, iou_threshold=0.3.

Reference:
    Bewley et al., "Simple Online and Realtime Tracking" (SORT), ICIP 2016.
"""

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Bounding-Box Conversions ────────────────────────────────────────────────

def xyxy_to_cxcywh(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float64)


def cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = cxcywh[0], cxcywh[1], cxcywh[2], cxcywh[3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float64)


# ── IoU Computation ─────────────────────────────────────────────────────────

def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes.

    Args:
        box_a: [x1, y1, x2, y2] array.
        box_b: [x1, y1, x2, y2] array.

    Returns:
        float: IoU in [0.0, 1.0].
    """
    xa = max(float(box_a[0]), float(box_b[0]))
    ya = max(float(box_a[1]), float(box_b[1]))
    xb = min(float(box_a[2]), float(box_b[2]))
    yb = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of [x1,y1,x2,y2] boxes.

    Args:
        boxes_a: (N, 4) array.
        boxes_b: (M, 4) array.

    Returns:
        (N, M) IoU matrix.
    """
    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    iou_mat = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou_mat[i, j] = compute_iou(boxes_a[i], boxes_b[j])
    return iou_mat


# ── Kalman Filter (Constant-Velocity, 8-State) ─────────────────────────────

class KalmanBoxTracker:
    """Kalman filter for a single bounding box in [cx, cy, w, h] space.

    State vector: [cx, cy, w, h, vx, vy, vw, vh] (8D).
    Observation:  [cx, cy, w, h] (4D).
    Motion model: constant velocity with variable Δt.

    The process noise Q and measurement noise R use diagonal matrices with
    standard SORT-family magnitude tuning. Exact values are not critical for
    short-term tracking; the IoU-based association dominates performance.
    """

    def __init__(self, bbox_xyxy: np.ndarray):
        """Initialize with a detection bounding box in [x1,y1,x2,y2] format.

        Args:
            bbox_xyxy: Initial detection box [x1, y1, x2, y2].
        """
        z = xyxy_to_cxcywh(bbox_xyxy)

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = z  # Position from measurement
        # Velocities initialized to zero

        # State covariance — high initial uncertainty on velocities
        self.P = np.eye(8, dtype=np.float64)
        self.P[4, 4] = 1000.0  # vx uncertainty
        self.P[5, 5] = 1000.0  # vy uncertainty
        self.P[6, 6] = 1000.0  # vw uncertainty
        self.P[7, 7] = 1000.0  # vh uncertainty
        # Position initialized with moderate uncertainty
        self.P[0, 0] = 10.0
        self.P[1, 1] = 10.0
        self.P[2, 2] = 10.0
        self.P[3, 3] = 10.0

        # Observation matrix H: maps state to measurement space [cx, cy, w, h]
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise R (observation uncertainty)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

        # Process noise base values (scaled by Δt in predict)
        self._q_pos = 1.0    # position process noise
        self._q_vel = 0.01   # velocity process noise

    def predict(self, dt: float) -> np.ndarray:
        """Predict state forward by dt seconds.

        Args:
            dt: Time step in seconds (actual elapsed time).

        Returns:
            Predicted bounding box in [x1, y1, x2, y2] format.
        """
        if dt <= 0:
            dt = 1.0 / 30.0  # Fallback: assume 30 FPS if Δt invalid

        # State transition matrix F for constant-velocity with variable Δt
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = dt  # cx += vx * dt
        F[1, 5] = dt  # cy += vy * dt
        F[2, 6] = dt  # w  += vw * dt
        F[3, 7] = dt  # h  += vh * dt

        # Process noise Q — scaled by dt
        # Using discrete white-noise acceleration model
        Q = np.zeros((8, 8), dtype=np.float64)
        # Position noise ~ dt^2, velocity noise ~ dt
        q_p = self._q_pos
        q_v = self._q_vel
        for i in range(4):
            Q[i, i] = q_p * dt * dt
            Q[i + 4, i + 4] = q_v * dt
            Q[i, i + 4] = q_p * dt
            Q[i + 4, i] = q_p * dt

        # Predict step
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Enforce minimum w, h to prevent degenerate boxes
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

        return cxcywh_to_xyxy(self.x[:4])

    def update(self, bbox_xyxy: np.ndarray):
        """Update state with a matched detection.

        Args:
            bbox_xyxy: Matched detection box [x1, y1, x2, y2].
        """
        z = xyxy_to_cxcywh(bbox_xyxy)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(8, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Enforce minimum w, h
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    def get_state_xyxy(self) -> np.ndarray:
        """Return current state as [x1, y1, x2, y2]."""
        return cxcywh_to_xyxy(self.x[:4])


# ── Track Object ────────────────────────────────────────────────────────────

class Track:
    """A single tracked object with Kalman state and lifecycle management.

    Attributes:
        track_id:       Unique integer ID for this track.
        cls:            YOLO class ID.
        kalman:         KalmanBoxTracker instance.
        age:            Total frames since track creation.
        hits:           Number of frames with a matched detection.
        time_since_update: Consecutive frames without a matched detection.
        last_confidence: Confidence of the last matched detection.
        confirmed:      Whether track has accumulated enough hits (>= min_hits).
    """

    _next_id = 1  # Class-level counter for unique IDs

    def __init__(self, detection: Dict[str, Any]):
        """Create a new track from an unmatched detection.

        Args:
            detection: Dict with keys "box" (ndarray [x1,y1,x2,y2]),
                      "conf" (float), "cls" (int).
        """
        self.track_id = Track._next_id
        Track._next_id += 1

        self.cls = detection["cls"]
        self.kalman = KalmanBoxTracker(detection["box"].astype(np.float64))
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.last_confidence = detection["conf"]
        self.confirmed = False  # Will be set True once hits >= min_hits

    @classmethod
    def reset_id_counter(cls):
        """Reset the track ID counter (for testing purposes)."""
        cls._next_id = 1

    def predict(self, dt: float) -> np.ndarray:
        """Predict track's next position.

        Args:
            dt: Elapsed time in seconds since last prediction.

        Returns:
            Predicted bbox [x1, y1, x2, y2].
        """
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict(dt)

    def update(self, detection: Dict[str, Any]):
        """Update track with a matched detection.

        Args:
            detection: Dict with "box", "conf", "cls" keys.
        """
        self.kalman.update(detection["box"].astype(np.float64))
        self.hits += 1
        self.time_since_update = 0
        self.last_confidence = detection["conf"]

    def get_bbox(self) -> np.ndarray:
        """Return current bounding box as [x1, y1, x2, y2]."""
        return self.kalman.get_state_xyxy()

    def is_confirmed(self, min_hits: int) -> bool:
        """Check if track has enough hits to be considered confirmed."""
        self.confirmed = self.hits >= min_hits
        return self.confirmed


# ── Hungarian Association (Class-Aware, IoU-Gated) ──────────────────────────

def associate_detections_to_tracks(
    detections: List[Dict[str, Any]],
    tracks: List[Track],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match detections to tracks using class-aware IoU + Hungarian algorithm.

    Class-aware: a detection can only match a track of the same class.
    IoU-gated: matches with IoU below iou_threshold are rejected.

    Args:
        detections: List of detection dicts {"box", "conf", "cls"}.
        tracks:     List of active Track objects.
        iou_threshold: Minimum IoU for a valid match (default 0.3, SORT standard).

    Returns:
        tuple: (matches, unmatched_det_indices, unmatched_trk_indices)
            - matches: list of (detection_idx, track_idx) pairs
            - unmatched_det_indices: detection indices with no match
            - unmatched_trk_indices: track indices with no match
    """
    if len(detections) == 0 and len(tracks) == 0:
        return [], [], []

    if len(detections) == 0:
        return [], [], list(range(len(tracks)))

    if len(tracks) == 0:
        return [], list(range(len(detections))), []

    n_det = len(detections)
    n_trk = len(tracks)

    # Build detection and track bbox arrays
    det_boxes = np.array([d["box"] for d in detections], dtype=np.float64)
    trk_boxes = np.array([t.get_bbox() for t in tracks], dtype=np.float64)

    # Compute IoU matrix
    iou_mat = compute_iou_matrix(det_boxes, trk_boxes)

    # Apply class-aware gating: zero out IoU for class mismatches
    for d_idx in range(n_det):
        for t_idx in range(n_trk):
            if detections[d_idx]["cls"] != tracks[t_idx].cls:
                iou_mat[d_idx, t_idx] = 0.0

    # Convert IoU to cost (Hungarian minimizes cost)
    cost_matrix = 1.0 - iou_mat

    # Solve assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_det = set()
    matched_trk = set()

    for r, c in zip(row_indices, col_indices):
        if iou_mat[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_det.add(r)
            matched_trk.add(c)

    unmatched_det = [i for i in range(n_det) if i not in matched_det]
    unmatched_trk = [i for i in range(n_trk) if i not in matched_trk]

    return matches, unmatched_det, unmatched_trk


# ── Multi-Object Tracker ────────────────────────────────────────────────────

class MultiObjectTracker:
    """SORT-style multi-object tracker with Kalman filtering and Hungarian assignment.

    Lifecycle:
        - New track created for each unmatched detection.
        - Matched tracks: Kalman update + hits++ + time_since_update=0.
        - Unmatched tracks: predict-only + time_since_update++.
        - Tracks with time_since_update > max_age are terminated.
        - Only confirmed tracks (hits >= min_hits) are reported.

    Args:
        iou_threshold: Minimum IoU for association (SORT default: 0.3).
        max_age:       Frames to keep a track alive without a match (SORT default: 30).
                       At ~14 FPS effective rate, 30 frames ≈ ~2.1 seconds.
        min_hits:      Minimum hits before a track is reported (SORT default: 3).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._last_ts: Optional[float] = None

        # Telemetry accumulators (per-call, reset each update)
        self.last_telemetry: Dict[str, Any] = {}

    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """Process one frame's detections and return active confirmed tracks.

        Args:
            detections: Post-NMS detections from Phase 5, each a dict with
                       keys "box" (ndarray [x1,y1,x2,y2] float32),
                       "conf" (float), "cls" (int), "source" (str).
            timestamp:  Capture timestamp (time.perf_counter()) of this frame.
                       Used to compute Δt for Kalman prediction.

        Returns:
            List of active track dicts:
                {"track_id": int, "bbox": ndarray, "cls": int,
                 "confidence": float, "age": int, "hits": int}
        """
        t0 = time.perf_counter()

        # ── Compute Δt ──────────────────────────────────────────────────────
        if self._last_ts is not None and timestamp > self._last_ts:
            dt = timestamp - self._last_ts
        else:
            dt = 1.0 / 30.0  # First frame: assume 30 FPS as initialization
        self._last_ts = timestamp

        # ── Predict all existing tracks ─────────────────────────────────────
        for track in self.tracks:
            track.predict(dt)

        # ── Associate detections to tracks ──────────────────────────────────
        matches, unmatched_det, unmatched_trk = associate_detections_to_tracks(
            detections, self.tracks, self.iou_threshold
        )

        # ── Update matched tracks ───────────────────────────────────────────
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        # ── Create new tracks for unmatched detections ──────────────────────
        n_created = 0
        for det_idx in unmatched_det:
            new_track = Track(detections[det_idx])
            self.tracks.append(new_track)
            n_created += 1

        # ── Terminate stale tracks ──────────────────────────────────────────
        n_terminated = 0
        active_tracks = []
        for track in self.tracks:
            if track.time_since_update <= self.max_age:
                active_tracks.append(track)
            else:
                n_terminated += 1
        self.tracks = active_tracks

        # ── Build output (confirmed tracks only) ────────────────────────────
        results = []
        for track in self.tracks:
            if track.is_confirmed(self.min_hits) and track.time_since_update == 0:
                bbox = track.get_bbox()
                results.append({
                    "track_id": track.track_id,
                    "bbox": bbox.astype(np.float32),
                    "cls": track.cls,
                    "confidence": track.last_confidence,
                    "age": track.age,
                    "hits": track.hits,
                })

        t1 = time.perf_counter()
        tracking_ms = (t1 - t0) * 1000.0

        # ── Store telemetry ─────────────────────────────────────────────────
        self.last_telemetry = {
            "dt_seconds": round(dt, 6),
            "n_detections": len(detections),
            "n_tracks_before": len(self.tracks) + n_terminated,
            "n_matched": len(matches),
            "n_unmatched_det": len(unmatched_det),
            "n_unmatched_trk": len(unmatched_trk),
            "n_created": n_created,
            "n_terminated": n_terminated,
            "n_active_tracks": len(self.tracks),
            "n_confirmed_output": len(results),
            "tracking_ms": round(tracking_ms, 3),
        }

        return results

    def get_all_tracks(self) -> List[Track]:
        """Return all active tracks (for telemetry/debugging)."""
        return self.tracks

    def get_track_ages(self) -> List[int]:
        """Return ages of all active tracks."""
        return [t.age for t in self.tracks]
