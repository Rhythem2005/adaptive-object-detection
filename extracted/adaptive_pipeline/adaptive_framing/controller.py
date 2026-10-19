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

    def early_full(self, lum):
        """Opt-in cheap guards; do not change the legacy action ordering."""
        if self.cfg.controller_mode != "risk" or self.cfg.policy != "adaptive":
            return []
        if self.since_full is None:
            return ["init"]
        reasons = []
        if self.since_full >= self.cfg.k_max:
            reasons.append("staleness")
        if self.cfg.use_illum_trigger and abs(lum - self.key_lum) > self.cfg.lum_jump:
            reasons.append("illumination")
        return reasons

    def decide(self, thumb, lum, thumb_scale, tracker, flow_ok, *, flow_quality=None,
               novelty_override=None, scene_valid=True, budget_allows_full=True):
        c = self.cfg
        sig = {"novelty": np.nan, "fail_frac": np.nan, "n_need": 0}
        if c.policy == "always":
            return FULL, ["always"], [], sig
        if self.since_full is None:
            return FULL, ["init"], [], sig
        if c.policy == "fixed":
            return (FULL, ["interval"], [], sig) if self.since_full >= c.fixed_k else (SKIP, [], [], sig)
        if c.controller_mode == "risk":
            return self._decide_risk(thumb, lum, thumb_scale, tracker, flow_ok, flow_quality,
                                     novelty_override, scene_valid, budget_allows_full)

        reasons = []
        if self.since_full >= c.k_max:
            reasons.append("staleness")
        if c.use_illum_trigger and abs(lum - self.key_lum) > c.lum_jump:
            reasons.append("illumination")
        if c.motion_compensation and not scene_valid:
            reasons.append("motion_unreliable")
        vis = tracker.visible()
        if c.use_novelty:
            nov = self.novelty(thumb, tracker.boxes()[vis], thumb_scale) if novelty_override is None else novelty_override
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

    def _decide_risk(self, thumb, lum, scale, tracker, flow_ok, quality,
                     novelty_override, scene_valid, budget_allows_full):
        c = self.cfg
        vis = tracker.visible()
        n = int(vis.sum())
        uncertainty = tracker.uncertainty()
        quality = tracker.flow_quality if quality is None else np.asarray(quality)
        if c.propagation != "flow":
            quality = np.clip(1 - uncertainty / c.risk_uncertainty, 0, 1)
        zeros = np.zeros(len(tracker))
        risk = np.maximum.reduce([1 - quality if c.use_track_health else zeros,
                                  uncertainty / c.risk_uncertainty if c.use_uncertainty else zeros,
                                  tracker.flow_innovation / c.risk_innovation if c.use_uncertainty else zeros]) \
            if len(tracker) else zeros
        risk = np.nan_to_num(risk, nan=2, posinf=2)
        highest = float(np.max(risk[vis])) if n else 0.0
        fail = np.flatnonzero(vis & ~flow_ok) if c.use_track_health else np.zeros(0, int)
        fail_fraction = len(fail) / max(n, 1)
        novelty = np.nan
        if c.use_novelty:
            novelty = self.novelty(thumb, tracker.boxes()[vis], scale) if novelty_override is None else novelty_override
        sig = {"novelty": novelty, "fail_frac": fail_fraction, "n_need": 0,
               "prediction_risk": highest, "flow_quality_mean": float(np.mean(quality[vis])) if n else np.nan,
               "uncertainty_mean": float(np.mean(uncertainty[vis])) if n else np.nan,
               "budget_deferred": 0}
        reasons = self.early_full(lum)
        if c.motion_compensation and not scene_valid:
            reasons.append("motion_unreliable")
        if c.use_novelty and novelty > c.novelty_thr:
            reasons.append("novelty")
        if c.use_track_health and fail_fraction > c.fail_frac_thr:
            reasons.append("track_failure")
        if reasons:
            return FULL, reasons, [], sig
        if not n and self.since_full >= c.adaptive_min_interval:
            return FULL, ["no_visible_tracks"], [], sig

        hard = vis & (risk >= c.risk_refresh)
        if c.use_track_health:
            hard |= vis & (~flow_ok | (quality < c.risk_flow_quality))
        due = vis & (tracker.conf_det < c.tau_high) & (tracker.since_det >= c.k_uncertain) \
            & (risk >= c.risk_refresh / 2) if c.use_uncertainty else np.zeros(len(tracker), bool)
        need = np.flatnonzero(hard | due)
        sig["n_need"] = len(need)
        if len(need):
            # Only defer a soft confidence concern; failed motion never waits for budget.
            if not hard.any() and not budget_allows_full:
                sig["budget_deferred"] = 1
            else:
                if c.use_local and len(need) <= c.max_local_rois:
                    return LOCAL, ["prediction_risk"], need, sig
                return FULL, ["risk_refresh"], [], sig

        fraction = np.clip(highest / c.risk_refresh, 0, 1)
        interval = int(round(c.k_max - fraction * (c.k_max - c.adaptive_min_interval)))
        sig["refresh_interval"] = interval
        if self.since_full >= interval:
            if budget_allows_full or highest >= c.risk_refresh:
                return FULL, ["adaptive_deadline"], [], sig
            sig["budget_deferred"] = 1
        return SKIP, [], [], sig
