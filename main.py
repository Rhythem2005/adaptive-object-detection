"""Phase 5: Adaptive Frame Controller + Environmental Adaptation + Selective Re-Detection

Coordinates the full Phase 3+4+5 pipeline:
    VIDEO CAPTURE (producer thread, real-time paced)
         ↓
    LATEST-FRAME BUFFER (capacity=1, replaces stale frames)
         ↓
    ENVIRONMENTAL ADAPTATION / ILLUMINATION HANDLING (Phase 4)
         ↓
    YOLOv8n PRIMARY INFERENCE (consumer on main thread)
         ↓
    CONFIDENCE / UNCERTAINTY FILTERING (Phase 5)
         ↓
    SELECTIVE CONTEXT EXPANSION & LOCALIZED RE-DETECTION (Phase 5)
         ↓
    BOUNDING BOX FUSION & CLASS-AWARE NMS (Phase 5)
         ↓
    METRICS & TELEMETRY COLLECTION

Phase 3 (Adaptive Frame Controller) behavior is preserved:
    The producer continuously captures frames from the video file at native FPS
    and writes them into a single-slot buffer. The consumer retrieves the latest
    available frame after each detection cycle. Frames that arrive while inference
    is busy are silently replaced, maintaining stream freshness.

Phase 4 (Environmental Adaptation) behavior is preserved:
    Each frame's average luminance Y is computed (L channel of L*a*b*).
    If Y < τ_illum (45), CLAHE is applied to the L channel; normal frames bypass.

Phase 5 (Selective Re-Detection) operates downstream of primary detection:
    - Primary proposals with s >= τ_high (0.65) or Area > α * Area_frame (0.05) are accepted (D_accept).
    - Ambiguous small detections (0.25 <= s < 0.65 and Area <= 5%) form candidates (D_cand).
    - Candidates are symmetrically expanded by proportional margin β (1.0 = 3x box size) to preserve context.
    - YOLOv8n evaluates only the extracted ROIs.
    - Secondary detections are remapped to global coordinates.
    - Class-aware NMS deduplicates accepted primary and secondary proposals.
"""

import csv
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
    PHASE5_RESULTS_DIR,
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
)
from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.metrics import Phase3Metrics
from src.illumination import adapt_illumination
from src.selective_redetection import selective_redetection_pipeline


def warmup_detector(detector):
    """Run dummy inferences to trigger YOLO model compilation and GPU/CPU warm-up."""
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.detect(dummy)
    dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    detector.detect(dummy_crop)


def _process_frame(
    frame,
    frame_id,
    capture_ts,
    processing_start,
    detector,
    illumination_records,
    phase5_records,
    metrics,
):
    """Execute Phase 4 adaptation, primary YOLO, and Phase 5 selective re-detection for one frame."""
    # ── Phase 4: Environmental Adaptation ───────────────────────────────────
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

    # ── Primary YOLOv8n inference on full frame ─────────────────────────────
    primary_results, primary_ms = detector.detect(adapted_frame)

    # ── Phase 5: Confidence/Uncertainty Filtering -> Selective Re-Detection -> Fusion/NMS ──
    final_dets, p5_telemetry = selective_redetection_pipeline(
        adapted_frame,
        primary_results,
        detector,
        tau_low=CONF_TAU_LOW,
        tau_high=CONF_TAU_HIGH,
        alpha=AREA_ALPHA,
        margin=ROI_CONTEXT_MARGIN,
        min_size=ROI_MIN_SIZE,
        max_rois=MAX_ROIS_PER_FRAME,
        nms_iou_thresh=FUSION_NMS_IOU,
    )

    total_detection_ms = primary_ms + p5_telemetry["total_phase5_ms"]
    frame_age_ms = (processing_start - capture_ts) * 1000.0

    phase5_records.append({
        "frame_id": frame_id,
        "primary_ms": round(primary_ms, 3),
        "n_primary": p5_telemetry["n_primary"],
        "n_accepted": p5_telemetry["n_accepted"],
        "n_candidates": p5_telemetry["n_candidates"],
        "n_rois": p5_telemetry["n_rois"],
        "n_secondary": p5_telemetry["n_secondary"],
        "n_suppressed": p5_telemetry["n_suppressed"],
        "n_final": p5_telemetry["n_final"],
        "redetect_ms": p5_telemetry["redetect_ms"],
        "fusion_ms": p5_telemetry["fusion_ms"],
        "total_phase5_ms": p5_telemetry["total_phase5_ms"],
        "total_detection_ms": round(total_detection_ms, 3),
    })

    metrics.record(frame_id, frame_age_ms, total_detection_ms, len(final_dets))


def run_adaptive_pipeline(detector, video_path):
    """Run the Phase 3+4+5 adaptive pipeline on a single video.

    Uses real-time frame delivery (native FPS pacing) to simulate a live camera stream.

    Args:
        detector:   YOLODetector instance.
        video_path: Path to the input video file.

    Returns:
        tuple: (buffer_stats, metrics_summary, Phase3Metrics, illumination_records, phase5_records)
               or None if the video cannot be opened.
    """
    video_name = os.path.basename(video_path)
    print(f"  Video: {video_name}")

    buffer = LatestFrameBuffer()
    capture = VideoCaptureThread(video_path, buffer, realtime=True)
    metrics = Phase3Metrics()

    illumination_records = []
    phase5_records = []

    capture.start()

    time.sleep(0.1)
    if capture.is_stopped() and buffer.get() is None:
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    print(f"  Native: {capture.width}x{capture.height} "
          f"@ {capture.native_fps:.2f} FPS, {capture.total_frames} frames")

    wall_start = time.perf_counter()

    while True:
        data = buffer.get()

        if data is not None:
            frame, frame_id, capture_ts = data
            processing_start = time.perf_counter()
            _process_frame(
                frame, frame_id, capture_ts, processing_start,
                detector, illumination_records, phase5_records, metrics
            )

        elif capture.is_stopped():
            final = buffer.get()
            if final is not None:
                frame, frame_id, capture_ts = final
                processing_start = time.perf_counter()
                _process_frame(
                    frame, frame_id, capture_ts, processing_start,
                    detector, illumination_records, phase5_records, metrics
                )
            break
        else:
            time.sleep(0.001)

    wall_end = time.perf_counter()
    total_wall_s = wall_end - wall_start

    # ── Summary & Reporting ─────────────────────────────────────────────────
    buf_stats = buffer.stats()
    met_summary = metrics.summary()

    n_total = len(illumination_records)
    n_enhanced = sum(1 for r in illumination_records if r["enhanced"])
    lum_values = [r["luminance"] for r in illumination_records]
    adapt_times = [r["adapt_ms"] for r in illumination_records]

    n_p5 = len(phase5_records)
    primary_times = [r["primary_ms"] for r in phase5_records]
    redetect_times = [r["redetect_ms"] for r in phase5_records]
    fusion_times = [r["fusion_ms"] for r in phase5_records]
    p5_total_times = [r["total_phase5_ms"] for r in phase5_records]
    total_det_times = [r["total_detection_ms"] for r in phase5_records]
    cands_counts = [r["n_candidates"] for r in phase5_records]
    rois_counts = [r["n_rois"] for r in phase5_records]
    sec_counts = [r["n_secondary"] for r in phase5_records]
    final_counts = [r["n_final"] for r in phase5_records]

    print(f"  Wall time            : {total_wall_s:.3f} s")
    print(f"  Frames captured      : {buf_stats['captured']}")
    print(f"  Frames consumed      : {buf_stats['consumed']}")
    print(f"  Frames replaced      : {buf_stats['replaced']}")
    if buf_stats['captured'] > 0:
        print(f"  Frame drop rate      : {buf_stats['replaced'] / buf_stats['captured'] * 100:.1f}%")
    print(f"  Cycles run           : {met_summary.get('total_inferences', 0)}")
    print(f"  Effective stream FPS : {met_summary.get('inference_fps', 0):.2f}")
    print(f"  Avg frame age        : {met_summary.get('avg_frame_age_ms', 0):.3f} ms")
    print(f"  Max frame age        : {met_summary.get('max_frame_age_ms', 0):.3f} ms")
    print()
    print(f"  --- Phase 4: Environmental Adaptation ---")
    print(f"  Frames analyzed      : {n_total}")
    print(f"  Frames enhanced      : {n_enhanced} ({n_enhanced / n_total * 100:.1f}%)" if n_total > 0 else "  Frames enhanced      : 0")
    print(f"  Avg luminance        : {sum(lum_values) / len(lum_values):.2f}" if lum_values else "")
    print(f"  Avg adapt time       : {sum(adapt_times) / len(adapt_times):.3f} ms" if adapt_times else "")
    print()
    print(f"  --- Phase 5: Selective Re-Detection & Fusion ---")
    print(f"  Filtering thresholds : τ_low={CONF_TAU_LOW}, τ_high={CONF_TAU_HIGH}, α={AREA_ALPHA*100:.0f}%")
    print(f"  Context margin       : β={ROI_CONTEXT_MARGIN} (3x box size), min_size={ROI_MIN_SIZE}px, max_rois={MAX_ROIS_PER_FRAME}")
    print(f"  Avg primary dets     : {sum(r['n_primary'] for r in phase5_records) / n_p5:.2f}" if n_p5 else "")
    print(f"  Avg accepted (D_acc) : {sum(r['n_accepted'] for r in phase5_records) / n_p5:.2f}" if n_p5 else "")
    print(f"  Avg candidates (D_can): {sum(cands_counts) / n_p5:.2f}" if n_p5 else "")
    print(f"  Avg ROIs extracted   : {sum(rois_counts) / n_p5:.2f}" if n_p5 else "")
    print(f"  Avg secondary recov  : {sum(sec_counts) / n_p5:.2f}" if n_p5 else "")
    print(f"  Avg final detections : {sum(final_counts) / n_p5:.2f}" if n_p5 else "")
    print(f"  Avg primary time     : {sum(primary_times) / n_p5:.2f} ms" if n_p5 else "")
    print(f"  Avg redetect time    : {sum(redetect_times) / n_p5:.2f} ms" if n_p5 else "")
    print(f"  Avg fusion/NMS time  : {sum(fusion_times) / n_p5:.3f} ms" if n_p5 else "")
    print(f"  Avg total detect time: {sum(total_det_times) / n_p5:.2f} ms" if n_p5 else "")
    print()

    # Verification assertions
    if buf_stats["captured"] != buf_stats["consumed"] + buf_stats["replaced"]:
        print("  [WARN] Counter mismatch: captured != consumed + replaced")
    else:
        print("  [OK] captured == consumed + replaced (buffer accounting verified)")

    print()

    return buf_stats, met_summary, metrics, illumination_records, phase5_records


def _save_csv(records, path):
    """Write list of dicts to CSV."""
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    """Execute the Phase 3+4+5 pipeline across test videos."""
    print("=" * 70)
    print("Phase 5: Adaptive Stream + Environmental Adaptation + Selective Re-Detection")
    print("=" * 70)
    print()

    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    print(f"Model loaded         : {MODEL_PATH}")
    print(f"Base confidence      : {CONFIDENCE_THRESHOLD}")
    print(f"Illumination τ_illum : {ILLUMINATION_THRESHOLD}")
    print(f"Uncertainty band     : τ_low={CONF_TAU_LOW}, τ_high={CONF_TAU_HIGH}, α={AREA_ALPHA*100:.0f}%")
    print(f"Context margin       : β={ROI_CONTEXT_MARGIN}, min_size={ROI_MIN_SIZE}px, max_rois={MAX_ROIS_PER_FRAME}")
    print(f"Fusion NMS IoU       : {FUSION_NMS_IOU}")
    print("Warming up model...")
    warmup_detector(detector)
    print("Warmup complete.")
    print()

    all_summaries = []

    for video_name in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"  [SKIP] {video_path} not found")
            continue

        print("-" * 70)
        result = run_adaptive_pipeline(detector, video_path)
        if result:
            buf_stats, met_summary, metrics_obj, illum_records, p5_records = result

            # Save per-video Phase 5 per-frame CSV
            os.makedirs(PHASE5_RESULTS_DIR, exist_ok=True)
            p5_csv_name = video_name.replace(".mp4", "_phase5.csv")
            p5_csv_path = os.path.join(PHASE5_RESULTS_DIR, p5_csv_name)
            _save_csv(p5_records, p5_csv_path)
            print(f"  Phase 5 frame CSV    : {p5_csv_path}")

            # Save per-video Phase 4 illumination CSV
            os.makedirs(PHASE4_RESULTS_DIR, exist_ok=True)
            illum_csv_name = video_name.replace(".mp4", "_illumination.csv")
            illum_csv_path = os.path.join(PHASE4_RESULTS_DIR, illum_csv_name)
            _save_csv(illum_records, illum_csv_path)
            print(f"  Phase 4 illum CSV    : {illum_csv_path}")

            # Save Phase 3 metrics CSV
            os.makedirs(PHASE3_RESULTS_DIR, exist_ok=True)
            p3_csv_name = video_name.replace(".mp4", "_phase3.csv")
            p3_csv_path = os.path.join(PHASE3_RESULTS_DIR, p3_csv_name)
            metrics_obj.to_csv(p3_csv_path)
            print(f"  Phase 3 stream CSV   : {p3_csv_path}")

            n_p5 = len(p5_records)
            summary_row = {
                "video": video_name,
                "captured": buf_stats["captured"],
                "consumed": buf_stats["consumed"],
                "replaced": buf_stats["replaced"],
                "drop_rate_pct": round(buf_stats["replaced"] / buf_stats["captured"] * 100, 2) if buf_stats["captured"] > 0 else 0.0,
                "avg_primary_ms": round(sum(r["primary_ms"] for r in p5_records) / n_p5, 2) if n_p5 else 0.0,
                "avg_candidates": round(sum(r["n_candidates"] for r in p5_records) / n_p5, 2) if n_p5 else 0.0,
                "avg_rois": round(sum(r["n_rois"] for r in p5_records) / n_p5, 2) if n_p5 else 0.0,
                "avg_redetect_ms": round(sum(r["redetect_ms"] for r in p5_records) / n_p5, 2) if n_p5 else 0.0,
                "avg_fusion_ms": round(sum(r["fusion_ms"] for r in p5_records) / n_p5, 3) if n_p5 else 0.0,
                "avg_total_detection_ms": round(sum(r["total_detection_ms"] for r in p5_records) / n_p5, 2) if n_p5 else 0.0,
                "avg_frame_age_ms": met_summary.get("avg_frame_age_ms", 0.0),
                "avg_detections": round(sum(r["n_final"] for r in p5_records) / n_p5, 2) if n_p5 else 0.0,
                "e2e_fps": met_summary.get("inference_fps", 0.0),
            }
            all_summaries.append(summary_row)
            print()

    # ── Cross-Video Summary ─────────────────────────────────────────────────
    if all_summaries:
        print("=" * 70)
        print("PHASE 5 BENCHMARK SUMMARY")
        print("=" * 70)

        tot_captured = sum(r["captured"] for r in all_summaries)
        tot_consumed = sum(r["consumed"] for r in all_summaries)
        tot_replaced = sum(r["replaced"] for r in all_summaries)
        mean_prim = sum(r["avg_primary_ms"] for r in all_summaries) / len(all_summaries)
        mean_cands = sum(r["avg_candidates"] for r in all_summaries) / len(all_summaries)
        mean_rois = sum(r["avg_rois"] for r in all_summaries) / len(all_summaries)
        mean_redet = sum(r["avg_redetect_ms"] for r in all_summaries) / len(all_summaries)
        mean_fus = sum(r["avg_fusion_ms"] for r in all_summaries) / len(all_summaries)
        mean_tot_det = sum(r["avg_total_detection_ms"] for r in all_summaries) / len(all_summaries)
        mean_age = sum(r["avg_frame_age_ms"] for r in all_summaries) / len(all_summaries)
        mean_dets = sum(r["avg_detections"] for r in all_summaries) / len(all_summaries)
        mean_fps = sum(r["e2e_fps"] for r in all_summaries) / len(all_summaries)

        print(f"  Videos evaluated     : {len(all_summaries)}")
        print(f"  Total captured       : {tot_captured}")
        print(f"  Total consumed       : {tot_consumed}")
        print(f"  Total replaced       : {tot_replaced} ({tot_replaced / tot_captured * 100:.1f}% drop rate)")
        print(f"  Mean candidates/frame: {mean_cands:.2f}")
        print(f"  Mean ROIs/frame      : {mean_rois:.2f}")
        print(f"  Mean primary time    : {mean_prim:.2f} ms")
        print(f"  Mean redetect time   : {mean_redet:.2f} ms")
        print(f"  Mean fusion/NMS time : {mean_fus:.3f} ms")
        print(f"  Mean total detect ms : {mean_tot_det:.2f} ms")
        print(f"  Mean frame age       : {mean_age:.3f} ms")
        print(f"  Mean final detections: {mean_dets:.2f}")
        print(f"  Mean pipeline FPS    : {mean_fps:.2f}")
        print()

        summary_csv = os.path.join(PHASE5_RESULTS_DIR, "phase5_summary.csv")
        _save_csv(all_summaries, summary_csv)
        print(f"  Phase 5 Summary CSV  : {summary_csv}")

    print()
    print("Phase 5 pipeline execution complete.")


if __name__ == "__main__":
    main()
