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

# ── Output ──────────────────────────────────────────────────────────────────
PHASE3_RESULTS_DIR = os.path.join("benchmarks", "phase3")
PHASE4_RESULTS_DIR = os.path.join("benchmarks", "phase4")
