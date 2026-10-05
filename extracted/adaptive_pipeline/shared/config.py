"""Single configuration object for every experiment.

Every ablation is a PipelineConfig with a few fields overridden
(see experiments/presets.py), so a run is fully described by the
config dict saved next to its results.

Values marked [TUNE-ON-VAL] must be chosen on the validation split,
never on test.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Optional, Tuple

# Dataset class order (index = label id in the YOLO .txt files).
DATASET_NAMES = ["car", "bus", "truck", "pedestrian", "rider", "bicycle", "traffic light"]

# COCO id -> dataset id for *stock* YOLOv8n.
# !! This MUST be byte-identical to the mapping used for the frozen baseline.
# Stock COCO has no "rider" class; set this to whatever the baseline used.
# Set to None if the weights already output dataset ids.
# Verify by running preset `baseline` and reproducing the frozen numbers.
COCO_TO_DATASET: Optional[Dict[int, int]] = {
    2: 0,   # car
    5: 1,   # bus
    7: 2,   # truck
    0: 3,   # person -> pedestrian
    1: 5,   # bicycle
    9: 6,   # traffic light
}


@dataclass
class PipelineConfig:
    # ── Detector (must match the frozen baseline's eval settings) ──────────
    model_path: str = "yolov8n.pt"
    device: Optional[str] = None          # None = ultralytics default
    imgsz: int = 640
    det_conf: float = 0.001               # ultralytics val default (needed for mAP)
    det_iou: float = 0.7                  # ultralytics val default
    max_det: int = 300
    half: bool = False                    # ultralytics val uses half on CUDA
    class_map: Optional[Dict[int, int]] = field(default_factory=lambda: COCO_TO_DATASET)
    gflops_640: float = 8.7               # YOLOv8n @640x640 (ultralytics model card)

    # ── Scheduling policy ─────────────────────────────────────────────────
    policy: str = "adaptive"              # "always" | "fixed" | "adaptive"
    fixed_k: int = 4                      # fixed policy: FULL every k frames
    propagation: str = "flow"             # "flow" | "kalman" | "hold"

    k_max: int = 8                        # hard staleness bound (frames) [TUNE-ON-VAL]

    use_novelty: bool = True
    novelty_thr: float = 0.20             # changed fraction outside tracks [TUNE-ON-VAL]
    novelty_pix_thr: int = 20             # per-pixel |diff| on blurred thumbnail
    thumb_w: int = 160

    use_track_health: bool = True
    fail_frac_thr: float = 0.30           # failed-flow fraction -> FULL [TUNE-ON-VAL]

    use_local: bool = True                # LOCAL tier (ROI re-detection of tracks)
    max_local_rois: int = 3               # >3 ROIs @320 ~ cost of one full pass
    roi_imgsz: int = 320
    roi_margin: float = 0.5               # each side, as fraction of box w/h
    roi_min_size: int = 96
    roi_merge_iou: float = 0.3
    crop_border_px: int = 3
    roi_max_area_frac: float = 0.35       # ROIs covering more than this -> FULL instead
    local_miss_policy: str = "escalate"   # "escalate": unconfirmed track -> FULL on same frame
                                          # "drop": trust the ROI pass and delete the track

    use_uncertainty: bool = True          # re-verify uncertain tracks early
    tau_low: float = 0.25
    tau_high: float = 0.65
    k_uncertain: int = 3                  # [TUNE-ON-VAL]

    use_illum_trigger: bool = True
    lum_jump: float = 20.0                # |L - L_keyframe| -> FULL (tunnels etc.)

    # ── Environmental adaptation (detector inputs only) ───────────────────
    use_clahe: bool = False               # off by default: changes keyframe outputs
    illum_thr: float = 45.0               # LAB L mean, 8-bit (paper value)
    illum_hyst: float = 5.0
    clahe_clip: float = 2.0
    clahe_tile: Tuple[int, int] = (8, 8)

    # ── Original selective re-detection, as an optional keyframe refiner ──
    keyframe_refine: bool = False
    refine_alpha: float = 0.05
    refine_max_rois: int = 4
    fusion_nms_iou: float = 0.5

    # ── Tracker ───────────────────────────────────────────────────────────
    track_min_conf: float = 0.25          # detections that become tracks
    assoc_iou: float = 0.3
    max_missed_full: int = 1              # keyframes a track may miss (ID continuity)
    coast_max: int = 2                    # frames a flow-failed track is still output
    conf_decay: float = 0.97              # per-frame decay of propagated confidence
    output_nms_iou: float = 0.7

    # ── Optical flow propagation ──────────────────────────────────────────
    flow_width: int = 640
    flow_grid: int = 4                    # 4x4 points per box
    flow_win: int = 11                    # LK window
    flow_levels: int = 3                  # pyramid levels (large near-field motion)
    flow_fb_max: float = 1.5              # forward-backward error (flow px)
    flow_min_points: int = 4

    def with_overrides(self, **kw) -> "PipelineConfig":
        unknown = set(kw) - set(asdict(self))
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        return replace(self, **kw)

    def to_dict(self):
        d = asdict(self)
        if d["class_map"] is not None:
            d["class_map"] = {str(k): v for k, v in d["class_map"].items()}
        return d
