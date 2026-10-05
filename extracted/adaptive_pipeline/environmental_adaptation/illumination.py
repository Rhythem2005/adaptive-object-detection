"""Low-light handling.

Changes vs. original:
  * luminance (mean LAB L, same scale as tau_illum=45) is computed on the
    160-px thumbnail the scheduler already builds -> ~0.05 ms, not a
    full-resolution LAB conversion every frame;
  * hysteresis so CLAHE does not flicker on/off around the threshold;
  * CLAHE object is cached;
  * enhancement is applied ONLY to detector inputs (full frame on FULL,
    individual crops on LOCAL). Skipped frames never pay for it.
"""
import cv2
import numpy as np


def mean_lab_l(bgr_small: np.ndarray) -> float:
    return float(cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)[:, :, 0].mean())


class IlluminationAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=tuple(cfg.clahe_tile))
        self.low = False

    def reset(self):
        self.low = False

    def update(self, lum: float) -> bool:
        if self.low and lum > self.cfg.illum_thr + self.cfg.illum_hyst:
            self.low = False
        elif not self.low and lum < self.cfg.illum_thr:
            self.low = True
        return self.low

    @property
    def active(self) -> bool:
        return self.cfg.use_clahe and self.low

    def enhance(self, bgr: np.ndarray) -> np.ndarray:
        if not self.active:
            return bgr
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
