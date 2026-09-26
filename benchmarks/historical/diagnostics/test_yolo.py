"""Phase 2: YOLOv8n Baseline Detection Benchmark

Runs frame-by-frame streaming inference on representative BDD100K videos
and records baseline timing metrics (latency, FPS, total processing time).
"""

import time
import csv
import os

import cv2
from ultralytics import YOLO


# ── Configuration ───────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
VIDEO_DIR = "data/bdd100k/videos"
CONFIDENCE_THRESHOLD = 0.25

# Three representative videos spread across the dataset
TEST_VIDEOS = ["50.mp4", "544.mp4", "1600.mp4"]

RESULTS_CSV = "benchmarks/baseline_results.csv"


def run_baseline(model, video_path):
    """Run streaming YOLOv8n inference on a single video, frame by frame.

    Returns a dict of timing metrics for the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Native: {width}x{height} @ {fps_native:.2f} FPS, {total_frames} frames")

    inference_times = []
    detection_counts = []
    frame_idx = 0

    wall_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the single frame (stream=True yields results lazily)
        t0 = time.perf_counter()
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            save=False,
        )
        t1 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000.0
        inference_times.append(inference_ms)

        # Count detections for this frame (consume result without storing)
        n_dets = len(results[0].boxes) if results else 0
        detection_counts.append(n_dets)

        frame_idx += 1

    wall_end = time.perf_counter()
    cap.release()

    total_wall_s = wall_end - wall_start
    avg_inference_ms = sum(inference_times) / len(inference_times) if inference_times else 0
    e2e_fps = frame_idx / total_wall_s if total_wall_s > 0 else 0
    avg_dets = sum(detection_counts) / len(detection_counts) if detection_counts else 0

    metrics = {
        "video": os.path.basename(video_path),
        "resolution": f"{width}x{height}",
        "native_fps": round(fps_native, 2),
        "total_frames": frame_idx,
        "total_time_s": round(total_wall_s, 3),
        "avg_inference_ms": round(avg_inference_ms, 2),
        "e2e_fps": round(e2e_fps, 2),
        "avg_detections": round(avg_dets, 2),
    }

    print(f"  Frames processed : {frame_idx}")
    print(f"  Total wall time  : {total_wall_s:.3f} s")
    print(f"  Avg inference    : {avg_inference_ms:.2f} ms")
    print(f"  End-to-end FPS   : {e2e_fps:.2f}")
    print(f"  Avg detections   : {avg_dets:.2f}")
    print()

    return metrics


def main():
    print("=" * 60)
    print("Phase 2: YOLOv8n Baseline Detection Benchmark")
    print("=" * 60)
    print()

    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print()

    all_metrics = []

    for video_name in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"  [SKIP] {video_path} not found")
            continue

        print("-" * 60)
        metrics = run_baseline(model, video_path)
        if metrics:
            all_metrics.append(metrics)

    # ── Summary ─────────────────────────────────────────────────────────────
    if all_metrics:
        print("=" * 60)
        print("BASELINE SUMMARY")
        print("=" * 60)

        total_frames = sum(m["total_frames"] for m in all_metrics)
        total_time = sum(m["total_time_s"] for m in all_metrics)
        overall_fps = total_frames / total_time if total_time > 0 else 0
        avg_latency = sum(m["avg_inference_ms"] for m in all_metrics) / len(all_metrics)

        print(f"  Videos tested      : {len(all_metrics)}")
        print(f"  Total frames       : {total_frames}")
        print(f"  Total time         : {total_time:.3f} s")
        print(f"  Overall FPS        : {overall_fps:.2f}")
        print(f"  Mean inference     : {avg_latency:.2f} ms")
        print()

        # ── Save CSV ────────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"  Results saved to: {RESULTS_CSV}")

    print()
    print("Phase 2 baseline benchmark complete.")


if __name__ == "__main__":
    main()