"""Pipeline metrics tracking and performance evaluation.

Monitors and records key metrics for Phase 3 (Adaptive Frame Controller):
- Per-inference frame_id and frame_age (staleness at inference time)
- Inference latency per frame
- Detection count per frame
- Aggregate statistics: mean, min, max for latency and frame age
- Effective inference FPS
"""

import csv
import os


class Phase3Metrics:
    """Accumulates per-inference records and computes aggregate statistics.

    Each record stores:
        frame_id      – monotonic ID assigned by LatestFrameBuffer.put()
        frame_age_ms  – time elapsed from frame capture to inference start
        inference_ms  – wall-clock inference duration
        n_detections  – number of bounding boxes returned by YOLO
    """

    def __init__(self):
        self.records = []

    def record(self, frame_id, frame_age_ms, inference_ms, n_detections):
        """Append a single inference measurement."""
        self.records.append({
            "frame_id": frame_id,
            "frame_age_ms": round(frame_age_ms, 3),
            "inference_ms": round(inference_ms, 2),
            "n_detections": n_detections,
        })

    def summary(self):
        """Compute aggregate statistics across all recorded inferences.

        Returns:
            dict with keys: total_inferences, avg_inference_ms, min_inference_ms,
            max_inference_ms, avg_frame_age_ms, min_frame_age_ms, max_frame_age_ms,
            avg_detections, inference_fps.
        """
        if not self.records:
            return {}

        inf_times = [r["inference_ms"] for r in self.records]
        ages = [r["frame_age_ms"] for r in self.records]
        dets = [r["n_detections"] for r in self.records]

        total_inf_s = sum(inf_times) / 1000.0

        return {
            "total_inferences": len(self.records),
            "avg_inference_ms": round(sum(inf_times) / len(inf_times), 2),
            "min_inference_ms": round(min(inf_times), 2),
            "max_inference_ms": round(max(inf_times), 2),
            "avg_frame_age_ms": round(sum(ages) / len(ages), 3),
            "min_frame_age_ms": round(min(ages), 3),
            "max_frame_age_ms": round(max(ages), 3),
            "avg_detections": round(sum(dets) / len(dets), 2),
            "inference_fps": round(
                len(self.records) / total_inf_s if total_inf_s > 0 else 0, 2
            ),
        }

    def to_csv(self, path):
        """Write per-inference records to a CSV file.

        Args:
            path: Output CSV file path.
        """
        if not self.records:
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)
