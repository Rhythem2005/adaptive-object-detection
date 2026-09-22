"""Phase 3: Adaptive Frame Controller Pipeline

Coordinates the adaptive frame management pipeline:
    VIDEO CAPTURE (producer thread, real-time paced)
         ↓
    LATEST-FRAME BUFFER (capacity=1, replaces stale frames)
         ↓
    YOLOv8n INFERENCE (consumer on main thread)
         ↓
    METRICS COLLECTION

The producer continuously captures frames from the video file at native FPS
and writes them into a single-slot buffer.  The consumer retrieves the latest
available frame after each inference cycle, ensuring freshness over
completeness.  Frames that arrive while inference is busy are silently
replaced.
"""

import os
import time

import numpy as np

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VIDEO_DIR,
    TEST_VIDEOS,
    PHASE3_RESULTS_DIR,
)
from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.metrics import Phase3Metrics


def warmup_detector(detector):
    """Run a single inference on a dummy frame to trigger YOLO compilation.

    This ensures the first timed inference in the benchmark reflects
    steady-state latency rather than one-time model warmup overhead.
    """
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.detect(dummy)


def run_adaptive_pipeline(detector, video_path):
    """Run the Phase 3 adaptive frame controller pipeline on a single video.

    Uses real-time frame delivery (native FPS pacing) to accurately simulate
    a camera or live video source feeding into the adaptive frame controller.

    Args:
        detector:   YOLODetector instance (pre-loaded, warmed-up model).
        video_path: Path to the input video file.

    Returns:
        tuple: (buffer_stats dict, metrics_summary dict, Phase3Metrics instance)
               or None if the video could not be opened.
    """
    video_name = os.path.basename(video_path)
    print(f"  Video: {video_name}")

    # ── Set up producer–consumer pipeline ───────────────────────────────────
    buffer = LatestFrameBuffer()
    capture = VideoCaptureThread(video_path, buffer, realtime=True)
    metrics = Phase3Metrics()

    capture.start()

    # Wait briefly for capture thread to populate metadata
    time.sleep(0.1)
    if capture.is_stopped() and buffer.get() is None:
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    print(f"  Native: {capture.width}x{capture.height} "
          f"@ {capture.native_fps:.2f} FPS, {capture.total_frames} frames")

    # ── Inference loop (consumer) ───────────────────────────────────────────
    wall_start = time.perf_counter()

    while True:
        data = buffer.get()

        if data is not None:
            frame, frame_id, capture_ts = data
            inference_start = time.perf_counter()

            results, inference_ms = detector.detect(frame)

            n_dets = len(results[0].boxes) if results else 0
            frame_age_ms = (inference_start - capture_ts) * 1000.0

            metrics.record(frame_id, frame_age_ms, inference_ms, n_dets)

        elif capture.is_stopped():
            # Capture finished — drain any final frame that may have been
            # deposited between our get() and the stopped check.
            final = buffer.get()
            if final is not None:
                frame, frame_id, capture_ts = final
                inference_start = time.perf_counter()
                results, inference_ms = detector.detect(frame)
                n_dets = len(results[0].boxes) if results else 0
                frame_age_ms = (inference_start - capture_ts) * 1000.0
                metrics.record(frame_id, frame_age_ms, inference_ms, n_dets)
            break
        else:
            # Buffer empty but capture still running — brief yield
            time.sleep(0.001)

    wall_end = time.perf_counter()
    total_wall_s = wall_end - wall_start

    # ── Report ──────────────────────────────────────────────────────────────
    buf_stats = buffer.stats()
    met_summary = metrics.summary()

    print(f"  Wall time          : {total_wall_s:.3f} s")
    print(f"  Frames captured    : {buf_stats['captured']}")
    print(f"  Frames consumed    : {buf_stats['consumed']}")
    print(f"  Frames replaced    : {buf_stats['replaced']}")
    print(f"  Inferences run     : {met_summary.get('total_inferences', 0)}")
    print(f"  Avg inference      : {met_summary.get('avg_inference_ms', 0):.2f} ms")
    print(f"  Inference FPS      : {met_summary.get('inference_fps', 0):.2f}")
    print(f"  Avg frame age      : {met_summary.get('avg_frame_age_ms', 0):.3f} ms")
    print(f"  Max frame age      : {met_summary.get('max_frame_age_ms', 0):.3f} ms")
    print(f"  Avg detections     : {met_summary.get('avg_detections', 0):.2f}")
    print()

    # Verification assertions (soft — print warnings, don't crash)
    if buf_stats["captured"] != buf_stats["consumed"] + buf_stats["replaced"]:
        print("  [WARN] Counter mismatch: captured != consumed + replaced")
    else:
        print("  [OK] captured == consumed + replaced (buffer accounting verified)")

    # With realtime pacing at ~30 FPS and inference at ~18ms, the consumer
    # is faster than the producer most of the time.  Some frames will still
    # be replaced when inference occasionally exceeds the frame interval.
    if buf_stats["replaced"] > 0:
        print(f"  [OK] {buf_stats['replaced']} frames replaced "
              f"({buf_stats['replaced'] / buf_stats['captured'] * 100:.1f}% drop rate)")
    else:
        print("  [INFO] No frames replaced — inference kept up with capture rate")
    print()

    return buf_stats, met_summary, metrics


def main():
    """Execute the Phase 3 adaptive frame controller pipeline."""
    print("=" * 65)
    print("Phase 3: Adaptive Frame Controller Pipeline")
    print("=" * 65)
    print()

    # ── Load model and warm up ──────────────────────────────────────────────
    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Warming up model (one dummy inference)...")
    warmup_detector(detector)
    print("Warmup complete.")
    print()

    all_results = []

    for video_name in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"  [SKIP] {video_path} not found")
            continue

        print("-" * 65)
        result = run_adaptive_pipeline(detector, video_path)
        if result:
            buf_stats, met_summary, metrics_obj = result

            # Save per-video detailed CSV
            os.makedirs(PHASE3_RESULTS_DIR, exist_ok=True)
            csv_name = video_name.replace(".mp4", "_phase3.csv")
            csv_path = os.path.join(PHASE3_RESULTS_DIR, csv_name)
            metrics_obj.to_csv(csv_path)
            print(f"  Per-frame CSV: {csv_path}")
            print()

            all_results.append({
                "video": video_name,
                "captured": buf_stats["captured"],
                "consumed": buf_stats["consumed"],
                "replaced": buf_stats["replaced"],
                **met_summary,
            })

    # ── Cross-video summary ─────────────────────────────────────────────────
    if all_results:
        print("=" * 65)
        print("PHASE 3 SUMMARY")
        print("=" * 65)

        total_captured = sum(r["captured"] for r in all_results)
        total_consumed = sum(r["consumed"] for r in all_results)
        total_replaced = sum(r["replaced"] for r in all_results)
        avg_inf = sum(r["avg_inference_ms"] for r in all_results) / len(all_results)
        avg_age = sum(r["avg_frame_age_ms"] for r in all_results) / len(all_results)
        avg_fps = sum(r["inference_fps"] for r in all_results) / len(all_results)
        avg_dets = sum(r["avg_detections"] for r in all_results) / len(all_results)

        print(f"  Videos tested      : {len(all_results)}")
        print(f"  Total captured     : {total_captured}")
        print(f"  Total consumed     : {total_consumed}")
        print(f"  Total replaced     : {total_replaced}")
        if total_captured > 0:
            print(f"  Drop rate          : {total_replaced / total_captured * 100:.1f}%")
        print(f"  Mean inference     : {avg_inf:.2f} ms")
        print(f"  Mean frame age     : {avg_age:.3f} ms")
        print(f"  Mean inference FPS : {avg_fps:.2f}")
        print(f"  Mean detections    : {avg_dets:.2f}")
        print()

        # Save summary CSV
        import csv

        os.makedirs(PHASE3_RESULTS_DIR, exist_ok=True)
        summary_csv = os.path.join(PHASE3_RESULTS_DIR, "phase3_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Summary CSV: {summary_csv}")

    print()
    print("Phase 3 adaptive frame controller pipeline complete.")


if __name__ == "__main__":
    main()
