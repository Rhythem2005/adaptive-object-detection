"""MINOR-PROJECT: Real-Time Video Ingestion and Object Detection Pipeline."""

from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.metrics import Phase3Metrics

__all__ = [
    "LatestFrameBuffer",
    "VideoCaptureThread",
    "YOLODetector",
    "Phase3Metrics",
]
