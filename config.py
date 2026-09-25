"""Project configuration and settings for MINOR-PROJECT.

Centralizes configuration values including file paths, model selection,
camera/stream parameters, and logging/metrics thresholds.
"""

import os

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25

# ── Dataset ─────────────────────────────────────────────────────────────────
VIDEO_DIR = os.path.join("data", "bdd100k", "videos")

# Representative test videos (same set used in Phase 2 baseline)
TEST_VIDEOS = ["50.mp4", "544.mp4", "1600.mp4"]

# ── Phase 4: Environmental Adaptation / Illumination Handling ───────────────
# Paper: "A Unified Lightweight YOLO Framework with Adaptive Frame Control
#         and Selective Re-Detection for Real-Time Road-Scene Monitoring"
#
# Low-light threshold τ_illum.
# Paper-specified: "When Y falls below an empirical low-light threshold
# (τ_illum = 45 on an 8-bit scale), the frame is routed through CLAHE"
# Units: L-channel mean value, 8-bit scale (0–255).
ILLUMINATION_THRESHOLD = 45

# CLAHE clipLimit.
# Paper does NOT specify — using OpenCV cv2.createCLAHE default.
CLAHE_CLIP_LIMIT = 2.0

# CLAHE tileGridSize.
# Paper does NOT specify — using OpenCV cv2.createCLAHE default.
CLAHE_TILE_GRID_SIZE = (8, 8)

# ── Phase 5: Confidence/Uncertainty Filtering & Selective Re-Detection ──────
# Paper: "A Unified Lightweight YOLO Framework with Adaptive Frame Control
#         and Selective Re-Detection for Real-Time Road-Scene Monitoring"
#
# Lower confidence threshold τ_low (paper: 0.25).
# Proposals below τ_low are discarded as background noise (D_reject).
CONF_TAU_LOW = 0.25

# Upper confidence threshold τ_high (paper: 0.65).
# Proposals with s >= τ_high bypass secondary processing (D_accept).
CONF_TAU_HIGH = 0.65

# Bounding-box area threshold α as fraction of total frame area (paper: 5% = 0.05).
# Candidates for localized re-detection must satisfy:
#   τ_low <= s < τ_high  AND  Area <= α * Area_frame
AREA_ALPHA = 0.05

# Context expansion proportional margin β (paper: "symmetrically expanded
# by a fixed proportional margin before patch extraction").
# 1.0 expands width and height symmetrically by 1.0x on each side (3x total box size),
# ensuring sufficient visual context for YOLO shallow feature extraction.
ROI_CONTEXT_MARGIN = 1.0

# Minimum ROI dimension (pixels) to avoid feature collapse on sub-32px boxes.
ROI_MIN_SIZE = 64

# Maximum ROIs per frame to keep real-time latency bounded on edge hardware.
MAX_ROIS_PER_FRAME = 4

# IoU threshold for class-aware NMS fusion (paper: "class-aware Non-Maximum
# Suppression (NMS) deduplicates the aggregated pool").
FUSION_NMS_IOU = 0.50

# ── Output ──────────────────────────────────────────────────────────────────
PHASE3_RESULTS_DIR = os.path.join("benchmarks", "phase3")
PHASE4_RESULTS_DIR = os.path.join("benchmarks", "phase4")
PHASE5_RESULTS_DIR = os.path.join("benchmarks", "phase5")
