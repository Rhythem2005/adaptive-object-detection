"""Phase 6 Validation Benchmark: Temporal Tracking (Kalman + Hungarian).

Runs the full Phase 3+4+5+6 pipeline on the 20-video evaluation set defined
in benchmarks/phase6_evaluation_videos.csv, with Phase 6 tracking appended
after Phase 5 Fusion/NMS output.

Pipeline order (preserved from existing design):
    Frame Controller → Illumination → YOLO → Confidence Filter
        → Selective Re-Detection → Fusion/NMS → Tracking → Telemetry

Reports:
    - FPS, pipeline/tracking latency, frame age
    - Detections/frame, active tracks/frame
    - Tracks created/terminated, matched/unmatched counts
    - Mean/max track age, mean track length
    - Overall and per-condition (day/evening/night) with sample counts
    - Phase 5-vs-6 comparison

Usage:
    python3 benchmarks/validate_phase6.py
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VIDEO_DIR,
    ILLUMINATION_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CONF_TAU_LOW,
    CONF_TAU_HIGH,
    AREA_ALPHA,
    ROI_CONTEXT_MARGIN,
    ROI_MIN_SIZE,
    MAX_ROIS_PER_FRAME,
    FUSION_NMS_IOU,
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_AGE,
    TRACKER_MIN_HITS,
    PHASE6_RESULTS_DIR,
)
from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.metrics import Phase3Metrics
from src.illumination import adapt_illumination
from src.selective_redetection import selective_redetection_pipeline
from src.tracker import MultiObjectTracker, Track


def warmup_detector(detector):
    """Run dummy inferences to trigger model compilation and warm-up."""
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.detect(dummy)
    dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    detector.detect(dummy_crop)


def _load_evaluation_videos(csv_path):
    """Load the Phase 6 evaluation video list.

    Returns:
        list of dicts: [{video, condition, path, selection_reason}, ...]
    """
    videos = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(row)
    return videos


def _find_video(video_entry):
    """Try to find the video file using the CSV path or fallback to VIDEO_DIR.

    Returns:
        str or None: Resolved path, or None if not found.
    """
    # Try the path from CSV first
    if os.path.exists(video_entry["path"]):
        return video_entry["path"]
    # Fallback: look in the standard VIDEO_DIR
    alt = os.path.join(VIDEO_DIR, video_entry["video"])
    if os.path.exists(alt):
        return alt
    return None


def run_phase6_pipeline(detector, video_path, video_condition):
    """Run Phase 3+4+5+6 pipeline on a single video.

    Args:
        detector:        YOLODetector instance.
        video_path:      Path to video file.
        video_condition: Illumination condition label (day/evening/night).

    Returns:
        dict with per-video benchmark results, or None on failure.
    """
    video_name = os.path.basename(video_path)
    print(f"  Video: {video_name} ({video_condition})")

    buffer = LatestFrameBuffer()
    capture = VideoCaptureThread(video_path, buffer, realtime=True)

    # Fresh tracker per video (resets track IDs)
    Track.reset_id_counter()
    tracker = MultiObjectTracker(
        iou_threshold=TRACKER_IOU_THRESHOLD,
        max_age=TRACKER_MAX_AGE,
        min_hits=TRACKER_MIN_HITS,
    )

    # Per-frame records for CSV output
    frame_records = []

    # Aggregate counters
    total_matched = 0
    total_unmatched_det = 0
    total_unmatched_trk = 0
    total_created = 0
    total_terminated = 0
    all_track_ages = []
    max_track_id_seen = 0

    capture.start()
    time.sleep(0.1)
    if capture.is_stopped() and buffer.get() is None:
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    print(f"  Native: {capture.width}x{capture.height} "
          f"@ {capture.native_fps:.2f} FPS, {capture.total_frames} frames")

    wall_start = time.perf_counter()
    frames_processed = 0

    while True:
        data = buffer.get()

        if data is not None:
            frame, frame_id, capture_ts = data
            processing_start = time.perf_counter()

            # ── Phase 4: Illumination ───────────────────────────────────
            adapted_frame, luminance, was_enhanced, adapt_ms = adapt_illumination(
                frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
            )

            # ── Phase 3/5: Primary YOLO + Selective Re-Detection ────────
            primary_results, primary_ms = detector.detect(adapted_frame)
            final_dets, p5_telemetry = selective_redetection_pipeline(
                adapted_frame, primary_results, detector,
                tau_low=CONF_TAU_LOW, tau_high=CONF_TAU_HIGH, alpha=AREA_ALPHA,
                margin=ROI_CONTEXT_MARGIN, min_size=ROI_MIN_SIZE,
                max_rois=MAX_ROIS_PER_FRAME, nms_iou_thresh=FUSION_NMS_IOU,
            )

            # ── Phase 6: Tracking ──────────────────────────────────────
            active_tracks = tracker.update(final_dets, timestamp=capture_ts)
            tel = tracker.last_telemetry

            total_detection_ms = primary_ms + p5_telemetry["total_phase5_ms"]
            frame_age_ms = (processing_start - capture_ts) * 1000.0
            pipeline_ms = total_detection_ms + tel["tracking_ms"]

            # Accumulate
            total_matched += tel["n_matched"]
            total_unmatched_det += tel["n_unmatched_det"]
            total_unmatched_trk += tel["n_unmatched_trk"]
            total_created += tel["n_created"]
            total_terminated += tel["n_terminated"]

            for t in tracker.get_all_tracks():
                all_track_ages.append(t.age)
                max_track_id_seen = max(max_track_id_seen, t.track_id)

            frame_records.append({
                "frame_id": frame_id,
                "luminance": round(luminance, 2),
                "enhanced": was_enhanced,
                "primary_ms": round(primary_ms, 3),
                "n_primary": p5_telemetry["n_primary"],
                "n_final_dets": p5_telemetry["n_final"],
                "redetect_ms": p5_telemetry["redetect_ms"],
                "fusion_ms": p5_telemetry["fusion_ms"],
                "total_detection_ms": round(total_detection_ms, 3),
                "tracking_ms": tel["tracking_ms"],
                "pipeline_ms": round(pipeline_ms, 3),
                "frame_age_ms": round(frame_age_ms, 3),
                "dt_seconds": tel["dt_seconds"],
                "n_active_tracks": tel["n_active_tracks"],
                "n_confirmed_output": tel["n_confirmed_output"],
                "n_matched": tel["n_matched"],
                "n_unmatched_det": tel["n_unmatched_det"],
                "n_unmatched_trk": tel["n_unmatched_trk"],
                "n_created": tel["n_created"],
                "n_terminated": tel["n_terminated"],
            })

            frames_processed += 1

        elif capture.is_stopped():
            final = buffer.get()
            if final is not None:
                frame, frame_id, capture_ts = final
                processing_start = time.perf_counter()
                adapted_frame, luminance, was_enhanced, adapt_ms = adapt_illumination(
                    frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
                )
                primary_results, primary_ms = detector.detect(adapted_frame)
                final_dets, p5_telemetry = selective_redetection_pipeline(
                    adapted_frame, primary_results, detector,
                    tau_low=CONF_TAU_LOW, tau_high=CONF_TAU_HIGH, alpha=AREA_ALPHA,
                    margin=ROI_CONTEXT_MARGIN, min_size=ROI_MIN_SIZE,
                    max_rois=MAX_ROIS_PER_FRAME, nms_iou_thresh=FUSION_NMS_IOU,
                )
                active_tracks = tracker.update(final_dets, timestamp=capture_ts)
                tel = tracker.last_telemetry
                total_detection_ms = primary_ms + p5_telemetry["total_phase5_ms"]
                frame_age_ms = (processing_start - capture_ts) * 1000.0
                pipeline_ms = total_detection_ms + tel["tracking_ms"]

                total_matched += tel["n_matched"]
                total_unmatched_det += tel["n_unmatched_det"]
                total_unmatched_trk += tel["n_unmatched_trk"]
                total_created += tel["n_created"]
                total_terminated += tel["n_terminated"]
                for t in tracker.get_all_tracks():
                    all_track_ages.append(t.age)
                    max_track_id_seen = max(max_track_id_seen, t.track_id)

                frame_records.append({
                    "frame_id": frame_id,
                    "luminance": round(luminance, 2),
                    "enhanced": was_enhanced,
                    "primary_ms": round(primary_ms, 3),
                    "n_primary": p5_telemetry["n_primary"],
                    "n_final_dets": p5_telemetry["n_final"],
                    "redetect_ms": p5_telemetry["redetect_ms"],
                    "fusion_ms": p5_telemetry["fusion_ms"],
                    "total_detection_ms": round(total_detection_ms, 3),
                    "tracking_ms": tel["tracking_ms"],
                    "pipeline_ms": round(pipeline_ms, 3),
                    "frame_age_ms": round(frame_age_ms, 3),
                    "dt_seconds": tel["dt_seconds"],
                    "n_active_tracks": tel["n_active_tracks"],
                    "n_confirmed_output": tel["n_confirmed_output"],
                    "n_matched": tel["n_matched"],
                    "n_unmatched_det": tel["n_unmatched_det"],
                    "n_unmatched_trk": tel["n_unmatched_trk"],
                    "n_created": tel["n_created"],
                    "n_terminated": tel["n_terminated"],
                })
                frames_processed += 1
            break
        else:
            time.sleep(0.001)

    wall_end = time.perf_counter()
    wall_s = wall_end - wall_start

    if frames_processed == 0:
        print("  [WARN] No frames processed")
        return None

    # ── Compute per-video summary ───────────────────────────────────────
    buf_stats = buffer.stats()
    detection_times = [r["total_detection_ms"] for r in frame_records]
    tracking_times = [r["tracking_ms"] for r in frame_records]
    pipeline_times = [r["pipeline_ms"] for r in frame_records]
    frame_ages = [r["frame_age_ms"] for r in frame_records]
    det_counts = [r["n_final_dets"] for r in frame_records]
    active_track_counts = [r["n_active_tracks"] for r in frame_records]
    confirmed_counts = [r["n_confirmed_output"] for r in frame_records]

    # Track length = hits count for all tracks that existed
    # Since tracks are per-video, max_track_id_seen gives total unique tracks created
    # Remaining active tracks at end
    remaining_tracks = tracker.get_all_tracks()
    track_lengths = [t.hits for t in remaining_tracks]

    e2e_fps = frames_processed / (sum(pipeline_times) / 1000.0) if sum(pipeline_times) > 0 else 0

    summary = {
        "video": os.path.basename(video_path),
        "condition": video_condition,
        "captured": buf_stats["captured"],
        "consumed": buf_stats["consumed"],
        "replaced": buf_stats["replaced"],
        "drop_rate_pct": round(buf_stats["replaced"] / buf_stats["captured"] * 100, 2) if buf_stats["captured"] > 0 else 0.0,
        "frames_processed": frames_processed,
        "wall_time_s": round(wall_s, 3),
        "e2e_fps": round(e2e_fps, 2),
        "avg_detection_ms": round(sum(detection_times) / len(detection_times), 2),
        "avg_tracking_ms": round(sum(tracking_times) / len(tracking_times), 3),
        "avg_pipeline_ms": round(sum(pipeline_times) / len(pipeline_times), 2),
        "avg_frame_age_ms": round(sum(frame_ages) / len(frame_ages), 3),
        "max_frame_age_ms": round(max(frame_ages), 3),
        "avg_dets_per_frame": round(sum(det_counts) / len(det_counts), 2),
        "avg_active_tracks": round(sum(active_track_counts) / len(active_track_counts), 2),
        "avg_confirmed_tracks": round(sum(confirmed_counts) / len(confirmed_counts), 2),
        "total_matched": total_matched,
        "total_unmatched_det": total_unmatched_det,
        "total_unmatched_trk": total_unmatched_trk,
        "total_created": total_created,
        "total_terminated": total_terminated,
        "unique_track_ids": max_track_id_seen,
        "mean_track_age": round(sum(all_track_ages) / len(all_track_ages), 2) if all_track_ages else 0.0,
        "max_track_age": max(all_track_ages) if all_track_ages else 0,
        "mean_track_length": round(sum(track_lengths) / len(track_lengths), 2) if track_lengths else 0.0,
    }

    # Print per-video summary
    print(f"  Wall time          : {wall_s:.3f} s")
    print(f"  Frames processed   : {frames_processed}")
    print(f"  Drop rate          : {summary['drop_rate_pct']:.2f}%")
    print(f"  E2E FPS            : {summary['e2e_fps']:.2f}")
    print(f"  Avg detection ms   : {summary['avg_detection_ms']:.2f}")
    print(f"  Avg tracking ms    : {summary['avg_tracking_ms']:.3f}")
    print(f"  Avg pipeline ms    : {summary['avg_pipeline_ms']:.2f}")
    print(f"  Avg frame age      : {summary['avg_frame_age_ms']:.3f} ms")
    print(f"  Avg dets/frame     : {summary['avg_dets_per_frame']:.2f}")
    print(f"  Avg active tracks  : {summary['avg_active_tracks']:.2f}")
    print(f"  Avg confirmed trks : {summary['avg_confirmed_tracks']:.2f}")
    print(f"  Tracks created     : {summary['total_created']}")
    print(f"  Tracks terminated  : {summary['total_terminated']}")
    print(f"  Total matched      : {summary['total_matched']}")
    print(f"  Total unmatched det: {summary['total_unmatched_det']}")
    print(f"  Mean track age     : {summary['mean_track_age']:.2f}")
    print(f"  Max track age      : {summary['max_track_age']}")
    print(f"  Mean track length  : {summary['mean_track_length']:.2f}")
    print()

    return summary, frame_records


def _save_csv(records, path):
    """Write list of dicts to CSV."""
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def _print_condition_stats(summaries, condition, label):
    """Print aggregate stats for a single condition."""
    subset = [s for s in summaries if s["condition"] == condition]
    if not subset:
        return
    n = len(subset)
    print(f"\n  --- {label} (n={n}) ---")
    print(f"  Mean E2E FPS         : {sum(s['e2e_fps'] for s in subset) / n:.2f}")
    print(f"  Mean drop rate       : {sum(s['drop_rate_pct'] for s in subset) / n:.2f}%")
    print(f"  Mean detection ms    : {sum(s['avg_detection_ms'] for s in subset) / n:.2f}")
    print(f"  Mean tracking ms     : {sum(s['avg_tracking_ms'] for s in subset) / n:.3f}")
    print(f"  Mean pipeline ms     : {sum(s['avg_pipeline_ms'] for s in subset) / n:.2f}")
    print(f"  Mean frame age       : {sum(s['avg_frame_age_ms'] for s in subset) / n:.3f} ms")
    print(f"  Mean dets/frame      : {sum(s['avg_dets_per_frame'] for s in subset) / n:.2f}")
    print(f"  Mean active tracks   : {sum(s['avg_active_tracks'] for s in subset) / n:.2f}")
    print(f"  Mean confirmed trks  : {sum(s['avg_confirmed_tracks'] for s in subset) / n:.2f}")
    print(f"  Total created        : {sum(s['total_created'] for s in subset)}")
    print(f"  Total terminated     : {sum(s['total_terminated'] for s in subset)}")
    print(f"  Total matched        : {sum(s['total_matched'] for s in subset)}")
    print(f"  Total unmatched det  : {sum(s['total_unmatched_det'] for s in subset)}")
    print(f"  Mean track age       : {sum(s['mean_track_age'] for s in subset) / n:.2f}")
    print(f"  Max track age        : {max(s['max_track_age'] for s in subset)}")
    print(f"  Mean track length    : {sum(s['mean_track_length'] for s in subset) / n:.2f}")


def main():
    """Execute Phase 6 benchmark across evaluation videos."""
    print("=" * 70)
    print("Phase 6 Benchmark: Temporal Tracking (Kalman + Hungarian)")
    print("=" * 70)
    print()

    # ── Configuration Report ────────────────────────────────────────────
    print(f"Model                : {MODEL_PATH}")
    print(f"Base confidence      : {CONFIDENCE_THRESHOLD}")
    print(f"Illumination τ_illum : {ILLUMINATION_THRESHOLD}")
    print(f"Uncertainty band     : τ_low={CONF_TAU_LOW}, τ_high={CONF_TAU_HIGH}, α={AREA_ALPHA*100:.0f}%")
    print(f"Context margin       : β={ROI_CONTEXT_MARGIN}, min_size={ROI_MIN_SIZE}px, max_rois={MAX_ROIS_PER_FRAME}")
    print(f"Fusion NMS IoU       : {FUSION_NMS_IOU}")
    print(f"Tracker IoU thresh   : {TRACKER_IOU_THRESHOLD}")
    print(f"Tracker max age      : {TRACKER_MAX_AGE}")
    print(f"Tracker min hits     : {TRACKER_MIN_HITS}")
    print()

    # ── Load Evaluation Video List ──────────────────────────────────────
    csv_path = os.path.join("benchmarks", "phase6_evaluation_videos.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found")
        sys.exit(1)

    eval_videos = _load_evaluation_videos(csv_path)
    print(f"Evaluation videos    : {len(eval_videos)} listed in CSV")

    # ── Initialize Detector ─────────────────────────────────────────────
    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    print("Warming up model...")
    warmup_detector(detector)
    print("Warmup complete.")
    print()

    # ── Run Pipeline ────────────────────────────────────────────────────
    all_summaries = []
    skipped = []

    for entry in eval_videos:
        video_path = _find_video(entry)
        if video_path is None:
            print(f"  [SKIP] {entry['video']} — not found")
            skipped.append(entry["video"])
            continue

        print("-" * 70)
        result = run_phase6_pipeline(detector, video_path, entry["condition"])
        if result:
            summary, frame_records = result
            all_summaries.append(summary)

            # Save per-video frame CSV
            os.makedirs(PHASE6_RESULTS_DIR, exist_ok=True)
            frame_csv = os.path.join(
                PHASE6_RESULTS_DIR,
                entry["video"].replace(".mp4", "_phase6.csv"),
            )
            _save_csv(frame_records, frame_csv)
            print(f"  Frame CSV: {frame_csv}")

    # ── Cross-Video Summary ─────────────────────────────────────────────
    if not all_summaries:
        print("\n[ERROR] No videos processed successfully.")
        sys.exit(1)

    n = len(all_summaries)
    print()
    print("=" * 70)
    print("PHASE 6 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Videos evaluated     : {n} of {len(eval_videos)} ({len(skipped)} skipped)")
    if skipped:
        print(f"  Skipped              : {', '.join(skipped)}")
    print()

    # Overall stats
    print("  --- Overall ---")
    print(f"  Mean E2E FPS         : {sum(s['e2e_fps'] for s in all_summaries) / n:.2f}")
    print(f"  Mean drop rate       : {sum(s['drop_rate_pct'] for s in all_summaries) / n:.2f}%")
    print(f"  Mean detection ms    : {sum(s['avg_detection_ms'] for s in all_summaries) / n:.2f}")
    print(f"  Mean tracking ms     : {sum(s['avg_tracking_ms'] for s in all_summaries) / n:.3f}")
    print(f"  Mean pipeline ms     : {sum(s['avg_pipeline_ms'] for s in all_summaries) / n:.2f}")
    print(f"  Mean frame age       : {sum(s['avg_frame_age_ms'] for s in all_summaries) / n:.3f} ms")
    print(f"  Max frame age        : {max(s['max_frame_age_ms'] for s in all_summaries):.3f} ms")
    print(f"  Mean dets/frame      : {sum(s['avg_dets_per_frame'] for s in all_summaries) / n:.2f}")
    print(f"  Mean active tracks   : {sum(s['avg_active_tracks'] for s in all_summaries) / n:.2f}")
    print(f"  Mean confirmed trks  : {sum(s['avg_confirmed_tracks'] for s in all_summaries) / n:.2f}")
    print(f"  Total created        : {sum(s['total_created'] for s in all_summaries)}")
    print(f"  Total terminated     : {sum(s['total_terminated'] for s in all_summaries)}")
    print(f"  Total matched        : {sum(s['total_matched'] for s in all_summaries)}")
    print(f"  Total unmatched det  : {sum(s['total_unmatched_det'] for s in all_summaries)}")
    print(f"  Mean track age       : {sum(s['mean_track_age'] for s in all_summaries) / n:.2f}")
    print(f"  Max track age        : {max(s['max_track_age'] for s in all_summaries)}")
    print(f"  Mean track length    : {sum(s['mean_track_length'] for s in all_summaries) / n:.2f}")

    # Per-condition stats
    _print_condition_stats(all_summaries, "day", "Day")
    _print_condition_stats(all_summaries, "evening_low_light", "Evening/Low-Light")
    _print_condition_stats(all_summaries, "night", "Night")

    # ── Save summary CSV ────────────────────────────────────────────────
    os.makedirs(PHASE6_RESULTS_DIR, exist_ok=True)
    summary_csv = os.path.join(PHASE6_RESULTS_DIR, "phase6_summary.csv")
    _save_csv(all_summaries, summary_csv)
    print(f"\n  Phase 6 Summary CSV  : {summary_csv}")

    # ── Phase 5 vs 6 Comparison ─────────────────────────────────────────
    phase5_csv = os.path.join("benchmarks", "phase5", "phase5_summary.csv")
    if os.path.exists(phase5_csv):
        print()
        print("=" * 70)
        print("PHASE 5 vs PHASE 6 COMPARISON")
        print("=" * 70)
        print()
        print("  NOTE: Phase 5 benchmark used 3 videos (50, 544, 1600).")
        print(f"         Phase 6 benchmark used {n} videos.")
        print("         Comparing only overlapping videos for fairness.")
        print()

        with open(phase5_csv) as f:
            p5_rows = {r["video"]: r for r in csv.DictReader(f)}

        for s6 in all_summaries:
            v = s6["video"]
            if v in p5_rows:
                p5 = p5_rows[v]
                p5_fps = float(p5["e2e_fps"])
                p5_drop = float(p5["drop_rate_pct"])
                p5_det_ms = float(p5["avg_total_detection_ms"])
                print(f"  {v}:")
                print(f"    Phase 5: FPS={p5_fps:.2f}, Drop={p5_drop:.2f}%, Det_ms={p5_det_ms:.2f}")
                print(f"    Phase 6: FPS={s6['e2e_fps']:.2f}, Drop={s6['drop_rate_pct']:.2f}%, "
                      f"Det_ms={s6['avg_detection_ms']:.2f}, Trk_ms={s6['avg_tracking_ms']:.3f}, "
                      f"Pipeline_ms={s6['avg_pipeline_ms']:.2f}")
                fps_delta = s6['e2e_fps'] - p5_fps
                print(f"    Δ FPS: {fps_delta:+.2f} ({fps_delta / p5_fps * 100:+.1f}%)")
                print()
    else:
        print("\n  [INFO] No Phase 5 summary found for comparison.")

    print()
    print("Phase 6 benchmark complete.")


if __name__ == "__main__":
    main()
