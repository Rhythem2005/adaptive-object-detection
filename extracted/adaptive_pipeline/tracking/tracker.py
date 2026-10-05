"""Track state between detector calls.

Rewritten. Problems in the original that made it unusable for skipping:
  1. Output only included tracks matched in the *current* frame
     (time_since_update == 0), so on a frame without detection it output
     nothing -- tracking could never stand in for the detector.
  2. min_hits=3 with no warm-up exception deleted the first two detections
     of every object: the tracker could only *lower* recall vs. raw YOLO.
  3. Process noise Q had cross terms q*dt with diagonal q*dt^2 and 0.01*dt,
     which is not positive semi-definite -> covariance can go indefinite.
  4. dt came from perf_counter capture timestamps (non-deterministic).
  5. Per-track Python loops for IoU and Kalman; global class-level ID counter.

Now:
  * Detector is authoritative: on FULL frames the pipeline outputs the raw
    detections (identical to the baseline); the tracker only carries them
    forward. No min_hits gating.
  * Batched constant-velocity Kalman filter in frame units with
    box-size-proportional, diagonal (PSD) noise (ByteTrack/BoT-SORT style).
  * Optical-flow boxes enter as noisier measurements; detections as clean ones.
  * Propagated confidence = last detector conf * decay^(frames since det),
    so stale boxes rank lower in the PR curve instead of masquerading as fresh.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..shared.boxes import EMPTY, iou_matrix, nms_classaware

SP, SV = 1.0 / 20, 1.0 / 160      # std weights (position, velocity) per frame
FLOW_R_SCALE = 4.0                # flow measurement variance multiplier vs. detection


def _xyxy_to_xywh(b):
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                     b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def _xywh_to_xyxy(x):
    return np.stack([x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2,
                     x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2], 1)


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.X = np.zeros((0, 8))
        self.P = np.zeros((0, 8, 8))
        self.ids = np.zeros(0, np.int64)
        self.cls = np.zeros(0, np.float32)
        self.conf_det = np.zeros(0, np.float32)
        self.since_det = np.zeros(0, np.float32)   # frames since last detector confirmation
        self.missed = np.zeros(0, np.int64)        # detector checks this track failed
        self.fail_streak = np.zeros(0, np.int64)   # consecutive flow failures
        self._next_id = 1

    def __len__(self):
        return len(self.ids)

    def boxes(self):
        return _xywh_to_xyxy(self.X[:, :4]) if len(self) else np.zeros((0, 4))

    def visible(self):
        return self.missed == 0

    # ── Kalman ─────────────────────────────────────────────────────────────
    def predict(self, dt=1.0):
        if not len(self):
            return
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        w, h = self.X[:, 2], self.X[:, 3]
        std = np.stack([SP * w, SP * h, SP * w, SP * h, SV * w, SV * h, SV * w, SV * h], 1)
        Q = np.zeros_like(self.P)
        idx = np.arange(8)
        Q[:, idx, idx] = std ** 2 * dt
        self.X = self.X @ F.T
        self.P = F @ self.P @ F.T + Q
        self.X[:, 2:4] = np.maximum(self.X[:, 2:4], 1.0)

    def _kf_update(self, idx, boxes_xyxy, r_scale):
        if len(idx) == 0:
            return
        z = _xyxy_to_xywh(boxes_xyxy)
        X, P = self.X[idx], self.P[idx]
        std = np.stack([SP * z[:, 2], SP * z[:, 3], SP * z[:, 2], SP * z[:, 3]], 1)
        R = np.zeros((len(idx), 4, 4))
        R[:, np.arange(4), np.arange(4)] = std ** 2 * r_scale
        S = P[:, :4, :4] + R
        K = P[:, :, :4] @ np.linalg.inv(S)                    # (n,8,4)
        X = X + (K @ (z - X[:, :4])[:, :, None])[:, :, 0]
        P = P - K @ P[:, :4, :]
        X[:, 2:4] = np.maximum(X[:, 2:4], 1.0)
        self.X[idx], self.P[idx] = X, P

    # ── per-frame bookkeeping ──────────────────────────────────────────────
    def tick(self, dt=1.0):
        self.since_det += dt

    def apply_flow(self, new_boxes, ok):
        idx = np.where(ok)[0]
        self._kf_update(idx, new_boxes[idx], FLOW_R_SCALE)
        self.fail_streak[ok] = 0
        self.fail_streak[~ok] += 1

    # ── association ────────────────────────────────────────────────────────
    def _associate(self, dets):
        if len(self) == 0 or len(dets) == 0:
            return [], list(range(len(dets))), list(range(len(self)))
        iou = iou_matrix(dets[:, :4], self.boxes())
        iou[dets[:, 5][:, None] != self.cls[None, :]] = 0.0
        r, c = linear_sum_assignment(-iou)
        good = iou[r, c] >= self.cfg.assoc_iou
        matches = list(zip(r[good], c[good]))
        md = {m[0] for m in matches}
        mt = {m[1] for m in matches}
        return (matches, [i for i in range(len(dets)) if i not in md],
                [j for j in range(len(self)) if j not in mt])

    def _confirm(self, matches, dets):
        if not matches:
            return
        d = np.array([m[0] for m in matches])
        t = np.array([m[1] for m in matches])
        self._kf_update(t, dets[d, :4], 1.0)
        self.conf_det[t] = dets[d, 4]
        self.since_det[t] = 0
        self.missed[t] = 0
        self.fail_streak[t] = 0

    def _spawn(self, dets):
        if len(dets) == 0:
            return
        n = len(dets)
        z = _xyxy_to_xywh(dets[:, :4].astype(np.float64))
        X = np.zeros((n, 8))
        X[:, :4] = z
        std = np.stack([2 * SP * z[:, 2], 2 * SP * z[:, 3], 2 * SP * z[:, 2], 2 * SP * z[:, 3],
                        10 * SV * z[:, 2], 10 * SV * z[:, 3], 10 * SV * z[:, 2], 10 * SV * z[:, 3]], 1)
        P = np.zeros((n, 8, 8))
        P[:, np.arange(8), np.arange(8)] = std ** 2
        self.X = np.concatenate([self.X, X])
        self.P = np.concatenate([self.P, P])
        self.ids = np.concatenate([self.ids, np.arange(self._next_id, self._next_id + n)])
        self._next_id += n
        self.cls = np.concatenate([self.cls, dets[:, 5]])
        self.conf_det = np.concatenate([self.conf_det, dets[:, 4]])
        self.since_det = np.concatenate([self.since_det, np.zeros(n, np.float32)])
        self.missed = np.concatenate([self.missed, np.zeros(n, np.int64)])
        self.fail_streak = np.concatenate([self.fail_streak, np.zeros(n, np.int64)])

    def _keep(self, mask):
        for name in ("X", "P", "ids", "cls", "conf_det", "since_det", "missed", "fail_streak"):
            setattr(self, name, getattr(self, name)[mask])

    # ── detector feedback ──────────────────────────────────────────────────
    def update_full(self, dets):
        """Whole-frame detection: authoritative for every track."""
        dets = dets[dets[:, 4] >= self.cfg.track_min_conf]
        matches, ud, ut = self._associate(dets)
        self._confirm(matches, dets)
        if ut:
            self.missed[np.array(ut)] += 1
        self._keep(self.missed <= self.cfg.max_missed_full)
        self._spawn(dets[ud])

    def update_local(self, dets, refreshed_idx, drop_unconfirmed=False):
        """ROI detection. Re-anchors every track it matches.
        Returns the number of flagged tracks it did NOT confirm. Those are
        only removed when drop_unconfirmed=True; by default the caller
        escalates to a full pass, which then decides (and spawns new tracks)."""
        dets = dets[dets[:, 4] >= self.cfg.track_min_conf]
        matches, ud, _ = self._associate(dets)
        self._confirm(matches, dets)
        matched_t = {m[1] for m in matches}
        lost = [int(i) for i in refreshed_idx if int(i) not in matched_t]
        if drop_unconfirmed:
            if lost:
                self.missed[np.array(lost)] += 1
            self._keep(self.missed <= self.cfg.max_missed_full)
            self._spawn(dets[ud])
        return len(lost)

    # ── output on non-detector frames ──────────────────────────────────────
    def outputs(self):
        if not len(self):
            return EMPTY.copy(), np.zeros(0, np.int64)
        m = (self.missed == 0) & (self.fail_streak <= self.cfg.coast_max)
        b = self.boxes()[m]
        conf = self.conf_det[m] * self.cfg.conf_decay ** self.since_det[m]
        out = np.concatenate([b, conf[:, None], self.cls[m][:, None]], 1).astype(np.float32)
        ids = self.ids[m]
        keep = nms_classaware(np.concatenate([out, np.arange(len(out))[:, None]], 1), self.cfg.output_nms_iou) \
            if len(out) > 1 else np.concatenate([out, np.arange(len(out))[:, None]], 1)
        order = keep[:, 6].astype(int)
        return out[order], ids[order]
