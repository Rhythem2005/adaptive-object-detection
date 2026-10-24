"""Per-frame orchestration. This is the piece the original repo did not have:
nothing in it ever decided to skip the detector.

step(frame) for frame t:
  1. signals    thumbnail (160 px): LAB-L luminance + blurred gray;
                flow-resolution gray (640 px) if propagation == "flow"
  2. propagate  boxes(t-1) -> Kalman predict -> LK flow measurement update
  3. decide     FrameController -> FULL / LOCAL / SKIP (+ reasons)
  4. act        LOCAL: one batched YOLO pass on ROI crops around flagged
                       tracks. Confirmed tracks are re-anchored. If any
                       flagged track is NOT confirmed, or the ROIs would
                       cover > roi_max_area_frac of the frame, escalate to
                       FULL on this same frame (ROI pass is a confirmer,
                       never a deleter).
                FULL : YOLO on whole frame (optionally CLAHE / refine);
                       output = raw detections (== baseline on this frame);
                       tracker re-synchronised; keyframe signals reset
                SKIP : output = tracker state, no detector call
"""
import time

import cv2
import numpy as np

from .adaptive_framing.controller import FULL, LOCAL, SKIP, FrameController
from .adaptive_framing.budget import DetectorBudget, LocalCostGate
from .environmental_adaptation.illumination import IlluminationAdapter, mean_lab_l
from .selective_redetection.selective_redetection import (build_rois, merge_rois,
                                                          redetect, refine_keyframe)
from .tracking.flow import FlowPropagator
from .tracking.tracker import Tracker
from .tracking.verification import MotionCompensator, AppearanceVerifier


class AdaptivePipeline:
    def __init__(self, cfg, detector):
        if cfg.policy not in ("always", "fixed", "adaptive"):
            raise ValueError(cfg.policy)
        if cfg.propagation not in ("flow", "kalman", "hold"):
            raise ValueError(cfg.propagation)
        self.cfg = cfg
        self.detector = detector
        self.tracker = Tracker(cfg)
        self.controller = FrameController(cfg)
        self.illum = IlluminationAdapter(cfg)
        self.flow = FlowPropagator(cfg)
        self.budget = DetectorBudget(cfg)
        self.local_gate = LocalCostGate(cfg)
        self.motion = MotionCompensator(cfg)
        self.appearance = AppearanceVerifier(cfg)
        self.reset()

    def reset(self):
        """Call at every video boundary."""
        self.tracker.reset()
        self.controller.reset()
        self.illum.reset()
        self.prev_gray = None
        self.prev_idx = None
        self.last_prediction_details = np.zeros((0, 4))
        self.budget.reset()
        self.local_gate.reset()
        self.motion.reset()
        self.appearance.reset()

    def step(self, frame, frame_idx):
        c = self.cfg
        T = time.perf_counter
        t0 = T()
        H, W = frame.shape[:2]
        dt = 1 if self.prev_idx is None else max(1, frame_idx - self.prev_idx)
        temporal = c.policy != "always"

        # 1. signals (skipped entirely for the per-frame baseline unless CLAHE needs luminance)
        if not temporal and not c.use_clahe and not c.keyframe_refine:
            out, det_ms = self.detector.detect(frame)
            t5 = T()
            full_cost = self.detector.full_gflops(H, W)
            self.last_prediction_details = np.column_stack([np.full(len(out), -1), np.zeros((len(out), 3))])
            return out, {"frame": frame_idx, "decision": FULL, "reasons": "always", "since_full": 0,
                         "t_signal_ms": 0.0, "t_propagate_ms": 0.0, "t_decide_ms": 0.0,
                         "t_detect_ms": det_ms, "t_roi_ms": 0.0, "t_track_ms": 0.0,
                         "t_total_ms": (t5 - t0) * 1e3, "n_out": len(out), "n_tracks": 0,
                         "n_crops": 0, "n_need": 0, "local_unconfirmed": 0, "roi_pass": 0, "novelty": np.nan, "fail_frac": np.nan,
                         "lum": np.nan, "clahe": 0, "gflops": full_cost,
                         "full_reference_gflops": full_cost, "output_age_mean": 0.0}
        tw = c.thumb_w
        thumb_scale = tw / float(W)
        if c.fixed_signal_fastpath:
            # Fixed decisions never consume these signals. Keep flow preparation
            # below, including FULL frames used as the next SKIP's reference.
            thumb, lum = None, np.nan
        else:
            th = int(round(H * tw / W))
            small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            lum = mean_lab_l(small)
            thumb = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            self.illum.update(lum)
        gray = scale = None
        if temporal and (c.propagation == "flow" or c.motion_compensation or c.appearance_verification):
            gray, scale = self.flow.prepare(frame)
        t1 = T()

        # 2. propagate
        flow_ok = np.ones(len(self.tracker), bool)
        flow_quality = self.tracker.flow_quality.copy()
        self.controller.tick(dt)
        early = self.controller.early_full(lum)
        if c.observation_anchoring and c.policy == "fixed" and self.controller.since_full is not None \
                and self.controller.since_full >= c.fixed_k:
            early = ["interval"]
        previous_boxes = self.tracker.boxes()
        if temporal and len(self.tracker):
            prev_boxes = self.tracker.boxes()
            if c.propagation in ("flow", "kalman"):
                self.tracker.predict(dt)
            if c.propagation == "flow" and self.prev_gray is not None and not early:
                nb, flow_ok, flow_quality = self.flow.propagate(self.prev_gray, gray, prev_boxes, scale)
                self.tracker.apply_flow(nb, flow_ok, flow_quality)
        novelty_override = None
        scene_valid = True
        appearance_checked = appearance_failed = 0
        tv = T()
        if temporal and not early and c.motion_compensation:
            scene_valid = self.motion.update(self.prev_gray, gray, previous_boxes, scale)
            if scene_valid:
                novelty_override = self.motion.novelty(self.controller.key_thumb, thumb,
                    self.tracker.boxes()[self.tracker.visible()], thumb_scale, c.flow_width)
                scene_valid = self.motion.valid
        if temporal and not early and c.appearance_verification and len(self.tracker):
            eligible = np.flatnonzero(self.tracker.visible())
            verified, appearance_checked, appearance_failed = self.appearance.verify(
                gray, self.tracker.boxes()[eligible], self.tracker.ids[eligible], scale, flow_quality[eligible])
            flow_quality[eligible] = verified
            self.tracker.flow_quality = flow_quality.copy()
            if appearance_failed:
                flow_ok &= flow_quality >= c.risk_flow_quality
        verification_ms = (T() - tv) * 1e3 if c.motion_compensation or c.appearance_verification else 0.0
        motion_inlier_fraction = self.motion.inlier_fraction
        self.tracker.tick(dt)
        t2 = T()

        # 3. decide
        full_cost = self.detector.full_gflops(H, W)
        if early:
            decision, reasons, refresh = FULL, early, []
            sig = {"novelty": np.nan, "fail_frac": np.nan, "n_need": 0}
        else:
            decision, reasons, refresh, sig = self.controller.decide(thumb, lum, thumb_scale, self.tracker, flow_ok,
                flow_quality=flow_quality, novelty_override=novelty_override, scene_valid=scene_valid,
                budget_allows_full=self.budget.allows(full_cost, full_cost))
        since_full = self.controller.since_full if self.controller.since_full is not None else 0
        t3 = T()

        # 4. act
        det_ms = roi_ms = 0.0
        n_crops = 0
        gflops = 0.0
        pre = self.illum.enhance if self.illum.active else None
        t4 = None
        local_unconfirmed = 0
        roi_pass = 0
        local_attempt_ms = 0.0
        local_attempt_crops = 0
        local_cost_ratio = np.nan
        local_expected_success = self.local_gate.success_estimate
        if decision == LOCAL:
            margin = c.roi_margin
            if c.dynamic_roi_margin:
                margin += min(float(np.max(self.tracker.uncertainty()[refresh])), 0.5)
            rois = build_rois(self.tracker.boxes()[refresh], frame.shape, margin, c.roi_min_size)
            rois = merge_rois(rois, c.roi_merge_iou)
            area = float(((rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])).sum())
            if area > c.roi_max_area_frac * H * W:
                decision, reasons = FULL, reasons + ["roi_too_large"]
            elif c.local_cost_gate:
                roi_cost = self.detector.roi_gflops(len(rois))
                admitted, gate_reason, local_cost_ratio = self.local_gate.admit(len(rois), roi_cost, full_cost)
                expected_cost = roi_cost + (1 - local_expected_success) * full_cost
                if admitted and c.controller_mode == "risk" and not self.budget.allows(expected_cost, full_cost):
                    admitted, gate_reason = False, "local_budget"
                if not admitted:
                    decision, reasons = FULL, reasons + [gate_reason]
            if decision == LOCAL:
                roi_dets, roi_ms, n_crops = redetect(self.detector, frame, rois, c, preprocess=pre)
                local_attempt_ms, local_attempt_crops = roi_ms, n_crops
                roi_pass = 1
                gflops += self.detector.roi_gflops(n_crops)
                n_unconf = self.tracker.update_local(roi_dets, refresh, drop_unconfirmed=(c.local_miss_policy == "drop"))
                local_unconfirmed = n_unconf
                if n_unconf and c.local_miss_policy == "escalate":
                    decision, reasons = FULL, reasons + ["local_escalate"]
                else:
                    t4 = T()
                    out, ids = self.tracker.outputs()
        if decision == FULL and t4 is None:
            inp = self.illum.enhance(frame)
            out, det_ms = self.detector.detect(inp)
            gflops += self.detector.full_gflops(H, W)
            if c.keyframe_refine:
                out, r_ms, n2 = refine_keyframe(self.detector, inp, out, c)
                roi_ms += r_ms
                n_crops += n2
                gflops += self.detector.roi_gflops(n2)
            t4 = T()
            if temporal:
                self.tracker.update_full(out)
                self.controller.on_full(thumb, lum)
                self.motion.reset()
            self.local_gate.observe_full(det_ms)
        elif decision == SKIP:
            t4 = T()
            out, ids = self.tracker.outputs()
        if roi_pass:
            self.local_gate.observe_local(local_attempt_ms, decision == LOCAL and local_unconfirmed == 0, local_attempt_crops)
        if temporal and c.appearance_verification and gray is not None:
            self.appearance.anchor(gray, self.tracker.boxes(), self.tracker.ids, scale,
                                   confirmed=self.tracker.since_det == 0)
        budget_fraction = self.budget.record(gflops, full_cost)
        t5 = T()
        # Diagnostic export construction is outside pipeline timing, as are
        # frame-record construction and prediction-file serialization.
        if decision == FULL:
            self.last_prediction_details = np.column_stack([np.full(len(out), -1), np.zeros((len(out), 3))])
        else:
            lookup = {int(ident): i for i, ident in enumerate(self.tracker.ids)}
            indices = np.array([lookup[int(ident)] for ident in ids], dtype=int)
            self.last_prediction_details = np.column_stack([ids, self.tracker.since_det[indices],
                self.tracker.flow_quality[indices], self.tracker.flow_innovation[indices]])
        output_age = float(np.mean(self.last_prediction_details[:, 1])) if len(out) else np.nan
        self.prev_gray = gray
        self.prev_idx = frame_idx

        rec = {
            "frame": frame_idx, "decision": decision, "reasons": "|".join(reasons),
            "since_full": since_full,
            "t_signal_ms": (t1 - t0) * 1e3, "t_propagate_ms": (t2 - t1) * 1e3,
            "t_decide_ms": (t3 - t2) * 1e3, "t_detect_ms": det_ms, "t_roi_ms": roi_ms,
            "t_track_ms": (t5 - t4) * 1e3, "t_total_ms": (t5 - t0) * 1e3,
            "n_out": len(out), "n_tracks": len(self.tracker), "n_crops": n_crops,
            "n_need": sig["n_need"], "local_unconfirmed": local_unconfirmed, "roi_pass": roi_pass, "novelty": sig["novelty"], "fail_frac": sig["fail_frac"],
            "lum": lum, "clahe": int(self.illum.active), "gflops": gflops,
            "full_reference_gflops": full_cost,
            "prediction_risk": sig.get("prediction_risk", np.nan),
            "flow_quality_mean": sig.get("flow_quality_mean", float(np.mean(flow_quality)) if len(flow_quality) else np.nan),
            "uncertainty_mean": sig.get("uncertainty_mean", np.nan),
            "refresh_interval": sig.get("refresh_interval", c.fixed_k if c.policy == "fixed" else c.k_max),
            "budget_fraction": budget_fraction,
            "budget_overrun": int(c.controller_mode == "risk" and budget_fraction > c.detector_budget_fraction),
            "budget_deferred": sig.get("budget_deferred", 0),
            "local_cost_ratio": local_cost_ratio, "local_expected_success": local_expected_success,
            "local_attempt_ms": local_attempt_ms, "local_attempt_crops": local_attempt_crops,
            "t_verification_ms": verification_ms,
            "appearance_checked": appearance_checked, "appearance_failed": appearance_failed,
            "motion_inlier_fraction": motion_inlier_fraction if c.motion_compensation else np.nan,
            "output_age_mean": output_age,
        }
        return out, rec
