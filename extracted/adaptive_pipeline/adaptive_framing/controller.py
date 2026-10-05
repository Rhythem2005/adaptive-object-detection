"""Detector scheduler: decides FULL / LOCAL / SKIP for every frame.

Replaces the 'adaptive frame controller' that was really a latest-frame
buffer (frames were dropped when inference was slow -- a property of the
hardware, not a decision, and dropped frames had no output to evaluate).

All signals are cheap and computed BEFORE any detector call:
  staleness     frames since last FULL               (hard bound k_max)
  illumination  |L_t - L_keyframe| on thumbnail      (tunnels, headlights)
  novelty       fraction of thumbnail pixels changed vs. the last keyframe,
                *outside* propagated track boxes -> change the tracks do
                not explain (new objects, revealed regions, ego-motion)
  track health  per-track optical-flow forward-backward consistency
  uncertainty   tracks whose last detector conf was in [tau_low, tau_high)
                and that have not been re-confirmed for k_uncertain frames

Decision:
  FULL  if any global trigger fires, or too many tracks failed
  LOCAL if a few specific tracks need verification (<= max_local_rois)
  SKIP  otherwise (output propagated tracks)
Every decision is logged with its reasons, so each trigger's share of the
compute can be attributed and ablated.
"""
import cv2
import numpy as np

FULL, LOCAL, SKIP = "FULL", "LOCAL", "SKIP"


class FrameController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.key_thumb = None
        self.key_lum = None
        self.since_full = None

    def tick(self, dt=1):
        if self.since_full is not None:
            self.since_full += dt

    def on_full(self, thumb, lum):
        self.key_thumb, self.key_lum, self.since_full = thumb, lum, 0

    def novelty(self, thumb, track_boxes, thumb_scale):
        changed = cv2.absdiff(thumb, self.key_thumb) > self.cfg.novelty_pix_thr
        mask = np.ones_like(changed)
        th, tw = mask.shape
        for x1, y1, x2, y2 in track_boxes * thumb_scale:
            mask[max(0, int(y1) - 1):min(th, int(np.ceil(y2)) + 1),
                 max(0, int(x1) - 1):min(tw, int(np.ceil(x2)) + 1)] = False
        denom = mask.sum()
        if denom < 0.05 * mask.size:      # tracks cover nearly everything
            return 0.0
        return float((changed & mask).sum() / denom)

    def decide(self, thumb, lum, thumb_scale, tracker, flow_ok):
        c = self.cfg
        sig = {"novelty": np.nan, "fail_frac": np.nan, "n_need": 0}
        if c.policy == "always":
            return FULL, ["always"], [], sig
        if self.since_full is None:
            return FULL, ["init"], [], sig
        if c.policy == "fixed":
            return (FULL, ["interval"], [], sig) if self.since_full >= c.fixed_k else (SKIP, [], [], sig)

        reasons = []
        if self.since_full >= c.k_max:
            reasons.append("staleness")
        if c.use_illum_trigger and abs(lum - self.key_lum) > c.lum_jump:
            reasons.append("illumination")
        vis = tracker.visible()
        if c.use_novelty:
            nov = self.novelty(thumb, tracker.boxes()[vis], thumb_scale)
            sig["novelty"] = nov
            if nov > c.novelty_thr:
                reasons.append("novelty")

        n_vis = int(vis.sum())
        fail = np.where(vis & ~flow_ok)[0] if c.use_track_health else np.zeros(0, np.int64)
        fail_frac = len(fail) / max(n_vis, 1)
        sig["fail_frac"] = fail_frac
        if c.use_uncertainty:
            due = np.where(vis & (tracker.conf_det < c.tau_high) & (tracker.since_det >= c.k_uncertain))[0]
        else:
            due = np.zeros(0, np.int64)
        need = np.union1d(fail, due).astype(np.int64)
        sig["n_need"] = len(need)

        if reasons:
            return FULL, reasons, [], sig
        if c.use_track_health and fail_frac > c.fail_frac_thr:
            return FULL, ["track_failure"], [], sig
        if len(need) and c.use_local:
            if len(need) <= c.max_local_rois:
                r = (["flow_fail"] if len(fail) else []) + (["uncertain"] if len(np.setdiff1d(due, fail)) else [])
                return LOCAL, r, need, sig
            return FULL, ["too_many_refresh"], [], sig
        return SKIP, [], [], sig
