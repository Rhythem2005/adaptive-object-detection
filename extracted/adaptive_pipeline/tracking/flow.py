"""Sparse optical-flow box propagation (median-flow style).

Why: a driving camera moves, so object motion between frames is dominated
by ego-motion and perspective scaling; a constant-velocity Kalman prior
alone drifts within a few frames. Pyramidal LK on a 5x5 grid per box,
filtered by forward-backward error, gives a measured displacement and
scale change at ~1-3 ms/frame at 640 px width on CPU, and -- importantly
-- a per-track reliability signal (fraction of points that survive the
FB check) that the scheduler uses to decide when to re-detect.
"""
import cv2
import numpy as np

_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)


class FlowPropagator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lk = dict(winSize=(cfg.flow_win, cfg.flow_win), maxLevel=cfg.flow_levels, criteria=_CRIT)

    def prepare(self, frame):
        h, w = frame.shape[:2]
        scale = self.cfg.flow_width / float(w)
        small = cv2.resize(frame, (self.cfg.flow_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale

    def propagate(self, prev_gray, gray, boxes, scale):
        """boxes: (N,4) full-res xyxy at t-1. Returns (new_boxes, ok, quality)."""
        n = len(boxes)
        new = boxes.copy().astype(np.float64)
        ok = np.zeros(n, bool)
        quality = np.zeros(n, np.float32)
        if n == 0 or prev_gray is None:
            return new, ok, quality

        g = self.cfg.flow_grid
        b = boxes.astype(np.float64) * scale
        w = (b[:, 2] - b[:, 0])[:, None]
        h = (b[:, 3] - b[:, 1])[:, None]
        t = np.linspace(0.15, 0.85, g)[None, :]
        gx = b[:, 0:1] + w * t                       # (N,g)
        gy = b[:, 1:2] + h * t
        px = np.repeat(gx, g, axis=1)                # (N,g*g)
        py = np.tile(gy, (1, g))
        p0 = np.stack([px.ravel(), py.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk)
        fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
        valid = ((st1.ravel() == 1) & (st2.ravel() == 1) & (fb <= self.cfg.flow_fb_max)).reshape(n, g * g)
        p0 = p0.reshape(n, g * g, 2).astype(np.float64)
        p1 = p1.reshape(n, g * g, 2).astype(np.float64)
        H, W = gray.shape

        k = valid.sum(1)
        quality = (k / (g * g)).astype(np.float32)
        enough = k >= self.cfg.flow_min_points

        # median displacement and median scale (distance-to-centroid ratio),
        # over FB-valid points only; NaN-masked so all boxes run vectorised
        import warnings
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            vm = valid[..., None]
            d = np.nanmedian(np.where(vm, p1 - p0, np.nan), axis=1)               # (N,2)
            c0 = np.nanmean(np.where(vm, p0, np.nan), axis=1, keepdims=True)
            c1 = np.nanmean(np.where(vm, p1, np.nan), axis=1, keepdims=True)
            r0 = np.linalg.norm(p0 - c0, axis=2)
            r1 = np.linalg.norm(p1 - c1, axis=2)
            s = np.nanmedian(np.where(valid & (r0 > 1.0), r1 / np.maximum(r0, 1e-9), np.nan), axis=1)
        measurable = (np.minimum(w[:, 0], h[:, 0]) >= 8) & np.isfinite(s)
        s = np.where(measurable, np.clip(np.nan_to_num(s, nan=1.0), 0.8, 1.25), 1.0)

        cx = (b[:, 0] + b[:, 2]) / 2 + np.nan_to_num(d[:, 0])
        cy = (b[:, 1] + b[:, 3]) / 2 + np.nan_to_num(d[:, 1])
        inside = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
        ok = enough & inside
        nw, nh = w[:, 0] * s, h[:, 0] * s
        prop = np.stack([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], 1) / scale
        new[ok] = prop[ok]
        return new, ok, quality
