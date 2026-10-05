"""YOLOv8n wrapper.

Returns plain (N,6) arrays in *dataset* class ids, so every downstream
module and the evaluator see the same representation. Timing covers the
whole ultralytics predict call (pre-process + forward + NMS + .cpu()),
which is what a deployed pipeline pays; it is NOT comparable to
ultralytics' 'inference-only' speed figure.
"""
import math
import time
from typing import List, Tuple

import numpy as np

from ..shared.boxes import EMPTY


class YOLODetector:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        n = len(self.model.names)
        if cfg.class_map is None:
            self.lut = None
        else:
            self.lut = np.full(n, -1, np.int64)
            for k, v in cfg.class_map.items():
                self.lut[int(k)] = int(v)

    # ── internals ──────────────────────────────────────────────────────────
    def _predict(self, source, imgsz):
        c = self.cfg
        kw = dict(source=source, imgsz=imgsz, conf=c.det_conf, iou=c.det_iou,
                  max_det=c.max_det, device=c.device, verbose=False, save=False)
        if c.half:                       # only pass when used (deprecated kwarg in ultralytics>=8.4)
            kw["half"] = True
        return self.model.predict(**kw)

    def _to_array(self, result) -> np.ndarray:
        b = result.boxes
        if b is None or len(b) == 0:
            return EMPTY.copy()
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()[:, None]
        cls = b.cls.cpu().numpy().astype(np.int64)
        out = np.concatenate([xyxy, conf, cls[:, None]], 1).astype(np.float32)
        if self.lut is not None:
            mapped = self.lut[cls]
            keep = mapped >= 0
            out = out[keep]
            out[:, 5] = mapped[keep]
        return out

    # ── public ─────────────────────────────────────────────────────────────
    def detect(self, frame) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        dets = self._to_array(self._predict(frame, self.cfg.imgsz)[0])
        return dets, (time.perf_counter() - t0) * 1000.0

    def detect_batch(self, crops: List[np.ndarray], imgsz: int) -> Tuple[List[np.ndarray], float]:
        """One batched forward pass over several ROI crops."""
        if not crops:
            return [], 0.0
        t0 = time.perf_counter()
        results = self._predict(list(crops), imgsz)
        dets = [self._to_array(r) for r in results]
        return dets, (time.perf_counter() - t0) * 1000.0

    def warmup(self, frame, n=3):
        for _ in range(n):
            self.detect(frame)
        h, w = frame.shape[:2]
        crop = frame[: h // 3, : w // 3]
        self.detect_batch([crop, crop], self.cfg.roi_imgsz)

    # ── cost model (hardware-independent) ──────────────────────────────────
    def full_gflops(self, h, w) -> float:
        s = self.cfg.imgsz / max(h, w)
        nh = math.ceil(h * s / 32) * 32
        nw = math.ceil(w * s / 32) * 32
        return self.cfg.gflops_640 * nh * nw / 640.0 ** 2

    def roi_gflops(self, n_crops) -> float:
        return n_crops * self.cfg.gflops_640 * (self.cfg.roi_imgsz / 640.0) ** 2
