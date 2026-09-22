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

# ── Output ──────────────────────────────────────────────────────────────────
PHASE3_RESULTS_DIR = os.path.join("benchmarks", "phase3")
