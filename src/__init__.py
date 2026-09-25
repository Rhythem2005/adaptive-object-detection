"""MINOR-PROJECT: Real-Time Video Ingestion and Object Detection Pipeline."""

from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.metrics import Phase3Metrics
from src.illumination import adapt_illumination
from src.selective_redetection import (
    filter_uncertain_candidates,
    create_context_rois,
    redetect_rois,
    fuse_and_suppress,
    selective_redetection_pipeline,
)

__all__ = [
    "LatestFrameBuffer",
    "VideoCaptureThread",
    "YOLODetector",
    "Phase3Metrics",
    "adapt_illumination",
    "filter_uncertain_candidates",
    "create_context_rois",
    "redetect_rois",
    "fuse_and_suppress",
    "selective_redetection_pipeline",
]

