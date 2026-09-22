"""YOLO object detection inference module.

Loads the specified YOLO model weights and runs real-time inference
on frames retrieved from the frame manager.
"""

import time

from ultralytics import YOLO


class YOLODetector:
    """Wrapper for YOLOv8n single-frame inference.

    Args:
        model_path:     Path to the YOLO .pt weights file.
        conf_threshold: Minimum detection confidence (default 0.25).
    """

    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """Run inference on a single frame.

        Args:
            frame: numpy array (BGR image).

        Returns:
            tuple: (results, inference_ms)
                - results: ultralytics Results object list
                - inference_ms: float, wall-clock inference time in milliseconds
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            save=False,
        )
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000.0
        return results, inference_ms
