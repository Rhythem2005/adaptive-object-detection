"""Phase 4: Adaptive Frame Controller + Environmental Adaptation Pipeline

Coordinates the full Phase 3+4 pipeline:
    VIDEO CAPTURE (producer thread, real-time paced)
         ↓
    LATEST-FRAME BUFFER (capacity=1, replaces stale frames)
         ↓
    ENVIRONMENTAL ADAPTATION / ILLUMINATION HANDLING
         ↓
    YOLOv8n INFERENCE (consumer on main thread)
         ↓
    METRICS COLLECTION

Phase 3 (Adaptive Frame Controller) behavior is unchanged:
    The producer continuously captures frames from the video file at native FPS
    and writes them into a single-slot buffer.  The consumer retrieves the latest
    available frame after each inference cycle, ensuring freshness over
    completeness.  Frames that arrive while inference is busy are silently
    replaced.

Phase 4 (Environmental Adaptation) inserts between buffer retrieval and YOLO:
    Each frame's average luminance Y is computed (L channel of L*a*b*).
    If Y < τ_illum (paper: 45), CLAHE is applied to the L channel.
    Adequately illuminated frames bypass enhancement entirely (no-op).
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
    PHASE4_RESULTS_DIR,
    ILLUMINATION_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
)
from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.metrics import Phase3Metrics
from src.illumination import adapt_illumination


def warmup_detector(detector):
    """Run a single inference on a dummy frame to trigger YOLO compilation.

    This ensures the first timed inference in the benchmark reflects
    steady-state latency rather than one-time model warmup overhead.
    """
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.detect(dummy)


def run_adaptive_pipeline(detector, video_path):
    """Run the Phase 3+4 adaptive pipeline on a single video.

    Uses real-time frame delivery (native FPS pacing) to accurately simulate
    a camera or live video source feeding into the adaptive frame controller.

    Phase 4 environmental adaptation is applied between frame retrieval
    and YOLO inference.

    Args:
        detector:   YOLODetector instance (pre-loaded, warmed-up model).
        video_path: Path to the input video file.

    Returns:
        tuple: (buffer_stats dict, metrics_summary dict, Phase3Metrics instance,
                illumination_records list)
               or None if the video could not be opened.
    """
    video_name = os.path.basename(video_path)
    print(f"  Video: {video_name}")

    # ── Set up producer–consumer pipeline ───────────────────────────────────
    buffer = LatestFrameBuffer()
    capture = VideoCaptureThread(video_path, buffer, realtime=True)
    metrics = Phase3Metrics()

    # Per-frame illumination tracking
    illumination_records = []

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
            processing_start = time.perf_counter()

            # ── Phase 4: Environmental Adaptation ───────────────────────────
            # adapt_ms is self-timed inside adapt_illumination() via its own
            # perf_counter pair — independent of any outer timer.
            adapted_frame, luminance, was_enhanced, adapt_ms = adapt_illumination(
                frame,
                ILLUMINATION_THRESHOLD,
                CLAHE_CLIP_LIMIT,
                CLAHE_TILE_GRID_SIZE,
            )

            illumination_records.append({
                "frame_id": frame_id,
                "luminance": round(luminance, 2),
                "enhanced": was_enhanced,
                "adapt_ms": round(adapt_ms, 3),
            })

            # ── YOLOv8n inference on (possibly enhanced) frame ──────────────
            # inference_ms is self-timed inside detector.detect() via its own
            # perf_counter pair — strictly YOLO time, excludes illumination.
            results, inference_ms = detector.detect(adapted_frame)

            n_dets = len(results[0].boxes) if results else 0
            frame_age_ms = (processing_start - capture_ts) * 1000.0

            metrics.record(frame_id, frame_age_ms, inference_ms, n_dets)

        elif capture.is_stopped():
            # Capture finished — drain any final frame that may have been
            # deposited between our get() and the stopped check.
            final = buffer.get()
            if final is not None:
                frame, frame_id, capture_ts = final
                processing_start = time.perf_counter()

                adapted_frame, luminance, was_enhanced, adapt_ms = adapt_illumination(
                    frame,
                    ILLUMINATION_THRESHOLD,
                    CLAHE_CLIP_LIMIT,
                    CLAHE_TILE_GRID_SIZE,
                )

                illumination_records.append({
                    "frame_id": frame_id,
                    "luminance": round(luminance, 2),
                    "enhanced": was_enhanced,
                    "adapt_ms": round(adapt_ms, 3),
                })

                results, inference_ms = detector.detect(adapted_frame)
                n_dets = len(results[0].boxes) if results else 0
                frame_age_ms = (processing_start - capture_ts) * 1000.0
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

    # Illumination summary
    n_enhanced = sum(1 for r in illumination_records if r["enhanced"])
    n_total = len(illumination_records)
    lum_values = [r["luminance"] for r in illumination_records]
    adapt_times = [r["adapt_ms"] for r in illumination_records]

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
    print(f"  --- Phase 4: Illumination ---")
    print(f"  Threshold (τ_illum): {ILLUMINATION_THRESHOLD}")
    print(f"  Frames analyzed    : {n_total}")
    print(f"  Frames enhanced    : {n_enhanced} ({n_enhanced / n_total * 100:.1f}%)" if n_total > 0 else "  Frames enhanced    : 0")
    print(f"  Frames bypassed    : {n_total - n_enhanced}")
    if lum_values:
        print(f"  Avg luminance      : {sum(lum_values) / len(lum_values):.2f}")
        print(f"  Min luminance      : {min(lum_values):.2f}")
        print(f"  Max luminance      : {max(lum_values):.2f}")
    if adapt_times:
        print(f"  Avg adapt time     : {sum(adapt_times) / len(adapt_times):.3f} ms")
        print(f"  Max adapt time     : {max(adapt_times):.3f} ms")
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

    return buf_stats, met_summary, metrics, illumination_records


def main():
    """Execute the Phase 3+4 adaptive pipeline."""
    print("=" * 65)
    print("Phase 4: Adaptive Frame Controller + Environmental Adaptation")
    print("=" * 65)
    print()

    # ── Load model and warm up ──────────────────────────────────────────────
    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Illumination threshold (τ_illum): {ILLUMINATION_THRESHOLD}")
    print(f"CLAHE clipLimit: {CLAHE_CLIP_LIMIT}")
    print(f"CLAHE tileGridSize: {CLAHE_TILE_GRID_SIZE}")
    print("Warming up model (one dummy inference)...")
    warmup_detector(detector)
    print("Warmup complete.")
    print()

    all_results = []
    all_illumination = {}

    for video_name in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"  [SKIP] {video_path} not found")
            continue

        print("-" * 65)
        result = run_adaptive_pipeline(detector, video_path)
        if result:
            buf_stats, met_summary, metrics_obj, illum_records = result

            # Save per-video detailed CSV (Phase 3 metrics — unchanged format)
            os.makedirs(PHASE3_RESULTS_DIR, exist_ok=True)
            csv_name = video_name.replace(".mp4", "_phase3.csv")
            csv_path = os.path.join(PHASE3_RESULTS_DIR, csv_name)
            metrics_obj.to_csv(csv_path)
            print(f"  Per-frame CSV: {csv_path}")

            # Save per-video Phase 4 illumination CSV
            os.makedirs(PHASE4_RESULTS_DIR, exist_ok=True)
            illum_csv_name = video_name.replace(".mp4", "_illumination.csv")
            illum_csv_path = os.path.join(PHASE4_RESULTS_DIR, illum_csv_name)
            _save_illumination_csv(illum_records, illum_csv_path)
            print(f"  Illumination CSV: {illum_csv_path}")
            print()

            all_results.append({
                "video": video_name,
                "captured": buf_stats["captured"],
                "consumed": buf_stats["consumed"],
                "replaced": buf_stats["replaced"],
                "avg_adapt_ms": round(
                    sum(r["adapt_ms"] for r in illum_records) / len(illum_records), 3
                ) if illum_records else 0.0,
                **met_summary,
            })
            all_illumination[video_name] = illum_records

    # ── Cross-video summary ─────────────────────────────────────────────────
    if all_results:
        print("=" * 65)
        print("PHASE 3+4 SUMMARY")
        print("=" * 65)

        total_captured = sum(r["captured"] for r in all_results)
        total_consumed = sum(r["consumed"] for r in all_results)
        total_replaced = sum(r["replaced"] for r in all_results)
        avg_adapt = sum(r["avg_adapt_ms"] for r in all_results) / len(all_results)
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
        print(f"  Mean adapt (illum) : {avg_adapt:.3f} ms  (illumination stage only)")
        print(f"  Mean inference     : {avg_inf:.2f} ms  (YOLO only)")
        print(f"  Mean total/frame   : {avg_adapt + avg_inf:.2f} ms  (adapt + YOLO)")
        print(f"  Mean frame age     : {avg_age:.3f} ms")
        print(f"  Mean inference FPS : {avg_fps:.2f}")
        print(f"  Mean detections    : {avg_dets:.2f}")
        print()

        # Phase 4 aggregate
        all_illum_records = []
        for records in all_illumination.values():
            all_illum_records.extend(records)

        total_frames_p4 = len(all_illum_records)
        total_enhanced = sum(1 for r in all_illum_records if r["enhanced"])
        all_lums = [r["luminance"] for r in all_illum_records]
        all_adapts = [r["adapt_ms"] for r in all_illum_records]

        print(f"  --- Phase 4 Aggregate ---")
        print(f"  Total frames       : {total_frames_p4}")
        print(f"  Enhanced frames    : {total_enhanced} "
              f"({total_enhanced / total_frames_p4 * 100:.1f}%)" if total_frames_p4 > 0 else "")
        print(f"  Bypassed frames    : {total_frames_p4 - total_enhanced}")
        if all_lums:
            print(f"  Avg luminance      : {sum(all_lums) / len(all_lums):.2f}")
            print(f"  Min luminance      : {min(all_lums):.2f}")
            print(f"  Max luminance      : {max(all_lums):.2f}")
        if all_adapts:
            print(f"  Avg adapt overhead : {sum(all_adapts) / len(all_adapts):.3f} ms")
            print(f"  Max adapt overhead : {max(all_adapts):.3f} ms")
        print()

        # Save summary CSV
        import csv

        os.makedirs(PHASE4_RESULTS_DIR, exist_ok=True)
        summary_csv = os.path.join(PHASE4_RESULTS_DIR, "phase4_summary.csv")
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Summary CSV: {summary_csv}")

        # Also re-save Phase 3 summary (same format as before, for diff)
        os.makedirs(PHASE3_RESULTS_DIR, exist_ok=True)
        summary_csv_p3 = os.path.join(PHASE3_RESULTS_DIR, "phase3_summary.csv")
        with open(summary_csv_p3, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Phase 3 CSV: {summary_csv_p3}")

    print()
    print("Phase 4 adaptive pipeline complete.")


def _save_illumination_csv(records, path):
    """Write per-frame illumination records to CSV."""
    import csv
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
