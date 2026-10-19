"""Sparse optical-flow box propagation (median-flow style).

Why: a driving camera moves, so object motion between frames is dominated
by ego-motion and perspective scaling; a constant-velocity Kalman prior
    alone drifts within a few frames. Pyramidal LK on a configurable grid,
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
        self.diagnostics = {}

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
            self.diagnostics = {}
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
        point_mask = np.ones((n, g * g), bool)
        if self.cfg.flow_point_mode == "corners":
            points = p0.reshape(n, g * g, 2)
            point_mask[:] = False
            ph, pw = prev_gray.shape
            for i, box in enumerate(b):
                x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
                x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), [pw, ph])
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                corners = cv2.goodFeaturesToTrack(prev_gray[y1:y2, x1:x2], maxCorners=g*g,
                                                  qualityLevel=0.01, minDistance=3, blockSize=3)
                if corners is not None:
                    count = len(corners)
                    points[i, :count] = corners[:, 0] + [x1, y1]
                    point_mask[i, :count] = True
            p0 = points.reshape(-1, 1, 2)

        def failed():
            self.diagnostics = {"valid_fraction": quality.copy(), "fb_error": np.full(n, np.inf),
                                "spread": np.zeros(n), "dispersion": np.full(n, np.inf)}
            return new, ok, quality
        if prev_gray.shape != gray.shape:
            return failed()
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk)
        if p1 is None or st1 is None:
            return failed()
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **self.lk)
        if pb is None or st2 is None:
            return failed()
        fb = np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1)
        valid = ((st1.ravel() == 1) & (st2.ravel() == 1) & (fb <= self.cfg.flow_fb_max)).reshape(n, g * g)
        valid &= point_mask
        p0 = p0.reshape(n, g * g, 2).astype(np.float64)
        p1 = p1.reshape(n, g * g, 2).astype(np.float64)
        H, W = gray.shape

        k = valid.sum(1)
        # Legacy grid quality remains byte-compatible; corners use available points.
        quality = (k / np.maximum(point_mask.sum(1), 1)).astype(np.float32)
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
        if self.cfg.quality_aware_flow:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fb_median = np.nanmedian(np.where(valid, fb.reshape(n, g*g), np.nan), axis=1)
                lo = np.nanmin(np.where(valid[..., None], p0, np.nan), axis=1)
                hi = np.nanmax(np.where(valid[..., None], p0, np.nan), axis=1)
                spread = np.prod(np.maximum(hi - lo, 0), axis=1) / np.maximum(w[:, 0] * h[:, 0], 1)
                residual = np.linalg.norm((p1 - p0) - d[:, None, :], axis=2)
                dispersion = np.nanmedian(np.where(valid, residual, np.nan), axis=1) / np.maximum(np.hypot(w[:, 0], h[:, 0]), 1)
            ok &= (quality >= self.cfg.flow_min_valid_fraction) & (spread >= self.cfg.flow_min_spread)
            ok &= np.isfinite(d).all(axis=1) & (dispersion <= self.cfg.flow_max_dispersion)
            fb_trust = np.clip(1 - np.nan_to_num(fb_median, nan=self.cfg.flow_fb_max) / self.cfg.flow_fb_max, 0, 1)
            spread_trust = np.clip(np.nan_to_num(spread) / max(0.25, self.cfg.flow_min_spread), 0, 1)
            coherence = np.clip(1 - np.nan_to_num(dispersion, nan=self.cfg.flow_max_dispersion) / self.cfg.flow_max_dispersion, 0, 1)
            self.diagnostics = {"valid_fraction": quality.copy(), "fb_error": fb_median,
                                "spread": spread, "dispersion": dispersion}
            quality = (quality * fb_trust * spread_trust * coherence).astype(np.float32)
            quality[~ok] = 0
        else:
            self.diagnostics = {"valid_fraction": quality.copy()}
        nw, nh = w[:, 0] * s, h[:, 0] * s
        prop = np.stack([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], 1) / scale
        new[ok] = prop[ok]
        return new, ok, quality
