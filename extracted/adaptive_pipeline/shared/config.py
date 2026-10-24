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
    fixed_signal_fastpath: bool = False   # omit unused thumbnail/illumination signals
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

    # Experimental improvements are opt-in. Existing presets remain controls.
    observation_anchoring: bool = False
    quality_aware_flow: bool = False
    flow_point_mode: str = "grid"       # "grid" | "corners"
    flow_min_valid_fraction: float = 0.50
    flow_min_spread: float = 0.10        # valid-point span area / box area
    flow_max_dispersion: float = 0.20   # median motion residual / box diagonal
    flow_r_min: float = 1.0
    flow_r_max: float = 16.0
    quality_conf_decay: bool = False
    reliable_decay_floor: float = 0.25  # still decay even with reliable flow
    low_score_association: bool = False
    association_low_conf: float = 0.10
    # Phase 3 opt-ins. None preserves the original shared-threshold behaviour.
    track_birth_conf: Optional[float] = None
    low_association_iou: Optional[float] = None
    low_association_visible_only: bool = False
    threshold_aware_assignment: bool = False

    controller_mode: str = "legacy"     # "legacy" | "risk"
    adaptive_min_interval: int = 2
    risk_flow_quality: float = 0.50
    risk_uncertainty: float = 0.15       # normalized position/size std
    risk_innovation: float = 0.35        # normalized flow/prediction disagreement
    risk_refresh: float = 0.70
    detector_budget_fraction: float = 0.55  # soft rolling detector budget
    budget_window: int = 60

    local_cost_gate: bool = False
    local_max_crops: int = 1
    local_max_cost_fraction: float = 0.60
    local_cost_margin: float = 0.10
    local_success_prior: float = 0.65   # unvalidated prior, not a measured rate
    local_prior_strength: float = 4.0
    dynamic_roi_margin: bool = False
    local_births: bool = False
    local_birth_conf: float = 0.65
    local_max_births: int = 2

    motion_compensation: bool = False
    motion_max_points: int = 128
    motion_min_points: int = 12
    motion_min_inlier_fraction: float = 0.60
    motion_min_coverage: float = 0.60
    appearance_verification: bool = False
    appearance_max_tracks: int = 8
    appearance_patch_size: int = 32
    appearance_min_correlation: float = 0.50

    def __post_init__(self):
        if self.fixed_signal_fastpath and (self.policy != "fixed" or self.use_clahe
                or self.keyframe_refine or self.motion_compensation or self.appearance_verification):
            raise ValueError("fixed_signal_fastpath requires fixed policy without image adaptation/verification")
        if self.controller_mode not in ("legacy", "risk"):
            raise ValueError("controller_mode must be legacy or risk")
        if self.flow_point_mode not in ("grid", "corners"):
            raise ValueError("flow_point_mode must be grid or corners")
        if self.controller_mode == "risk" and not 1 <= self.adaptive_min_interval <= self.k_max:
            raise ValueError("adaptive_min_interval must be between 1 and k_max")
        if self.budget_window < 1 or not 0 < self.detector_budget_fraction <= 1:
            raise ValueError("invalid detector budget")
        if not 0 <= self.local_success_prior <= 1 or self.local_prior_strength <= 0:
            raise ValueError("invalid LOCAL success prior")
        if self.local_max_crops < 1 or not 0 < self.local_max_cost_fraction <= 1:
            raise ValueError("invalid LOCAL cost limit")
        if not 0 <= self.reliable_decay_floor <= 1:
            raise ValueError("reliable_decay_floor must be in [0, 1]")
        if not 0 < self.flow_r_min <= self.flow_r_max:
            raise ValueError("invalid flow measurement-noise range")
        if self.appearance_patch_size < 2 or self.appearance_max_tracks < 1:
            raise ValueError("invalid appearance verification limits")
        if self.motion_min_points < 3 or self.motion_max_points < self.motion_min_points:
            raise ValueError("invalid background feature limits")
        if self.flow_grid < 2 or not 0 < self.flow_min_points <= self.flow_grid ** 2:
            raise ValueError("invalid flow point budget")
        if self.flow_fb_max <= 0 or self.flow_max_dispersion <= 0:
            raise ValueError("flow error limits must be positive")
        if self.risk_uncertainty <= 0 or self.risk_innovation <= 0 or self.risk_refresh <= 0:
            raise ValueError("risk normalizers must be positive")
        for name in ("flow_min_valid_fraction", "flow_min_spread", "risk_flow_quality",
                     "motion_min_inlier_fraction", "motion_min_coverage", "local_birth_conf"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1 <= self.appearance_min_correlation <= 1 or self.local_max_births < 0:
            raise ValueError("invalid appearance correlation or birth budget")
        if not 0 <= self.association_low_conf <= self.track_min_conf or self.local_cost_margin < 0:
            raise ValueError("invalid association confidence or LOCAL cost margin")
        for name in ("track_birth_conf", "low_association_iou"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be None or in [0, 1]")
        if not 0 < self.conf_decay <= 1:
            raise ValueError("conf_decay must be in (0, 1]")

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
