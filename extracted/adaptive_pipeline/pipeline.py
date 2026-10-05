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
from .environmental_adaptation.illumination import IlluminationAdapter, mean_lab_l
from .selective_redetection.selective_redetection import (build_rois, merge_rois,
                                                          redetect, refine_keyframe)
from .tracking.flow import FlowPropagator
from .tracking.tracker import Tracker


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
        self.reset()

    def reset(self):
        """Call at every video boundary."""
        self.tracker.reset()
        self.controller.reset()
        self.illum.reset()
        self.prev_gray = None
        self.prev_idx = None

    def step(self, frame, frame_idx):
        c = self.cfg
        T = time.perf_counter
        t0 = T()
        H, W = frame.shape[:2]
        dt = 1 if self.prev_idx is None else max(1, frame_idx - self.prev_idx)
        temporal = c.policy != "always"

        # 1. signals (skipped entirely for the per-frame baseline unless CLAHE needs luminance)
        if not temporal and not c.use_clahe:
            out, det_ms = self.detector.detect(frame)
            t5 = T()
            return out, {"frame": frame_idx, "decision": FULL, "reasons": "always", "since_full": 0,
                         "t_signal_ms": 0.0, "t_propagate_ms": 0.0, "t_decide_ms": 0.0,
                         "t_detect_ms": det_ms, "t_roi_ms": 0.0, "t_track_ms": 0.0,
                         "t_total_ms": (t5 - t0) * 1e3, "n_out": len(out), "n_tracks": 0,
                         "n_crops": 0, "n_need": 0, "local_unconfirmed": 0, "roi_pass": 0, "novelty": np.nan, "fail_frac": np.nan,
                         "lum": np.nan, "clahe": 0, "gflops": self.detector.full_gflops(H, W)}
        tw = c.thumb_w
        th = int(round(H * tw / W))
        small = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        lum = mean_lab_l(small)
        thumb = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        thumb_scale = tw / float(W)
        self.illum.update(lum)
        gray = scale = None
        if temporal and c.propagation == "flow":
            gray, scale = self.flow.prepare(frame)
        t1 = T()

        # 2. propagate
        flow_ok = np.ones(len(self.tracker), bool)
        if temporal and len(self.tracker):
            prev_boxes = self.tracker.boxes()
            if c.propagation in ("flow", "kalman"):
                self.tracker.predict(dt)
            if c.propagation == "flow" and self.prev_gray is not None:
                nb, flow_ok, _ = self.flow.propagate(self.prev_gray, gray, prev_boxes, scale)
                self.tracker.apply_flow(nb, flow_ok)
        self.tracker.tick(dt)
        self.controller.tick(dt)
        t2 = T()

        # 3. decide
        decision, reasons, refresh, sig = self.controller.decide(thumb, lum, thumb_scale, self.tracker, flow_ok)
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
        if decision == LOCAL:
            rois = build_rois(self.tracker.boxes()[refresh], frame.shape, c.roi_margin, c.roi_min_size)
            rois = merge_rois(rois, c.roi_merge_iou)
            area = float(((rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])).sum())
            if area > c.roi_max_area_frac * H * W:
                decision, reasons = FULL, reasons + ["roi_too_large"]
            else:
                roi_dets, roi_ms, n_crops = redetect(self.detector, frame, rois, c, preprocess=pre)
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
        elif decision == SKIP:
            t4 = T()
            out, ids = self.tracker.outputs()
        t5 = T()

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
        }
        return out, rec
