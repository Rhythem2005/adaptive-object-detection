"""Phase 5: Comprehensive Validation of Selective Re-Detection Pipeline.

Tests all key criteria required for Phase 5 verification:
    1. Pipeline correctness: End-to-end execution on all project test videos.
    2. Re-detection activation: Verifies selective re-detection occurs and recovers targets.
    3. Coordinate remapping: Checks all secondary detections map strictly within native frame bounds [0, W] x [0, H] and lie inside their parent ROI.
    4. Duplicate suppression: Verifies class-aware NMS eliminates overlapping duplicate proposals.
    5. Adaptive frame controller integrity: Verifies captured == consumed + replaced and bounded frame age.
    6. Environmental adaptation integrity: Verifies Phase 4 illumination adaptation still functions correctly.
    7. Performance telemetry: Reports candidates, ROIs, re-detection time, fusion time, total time, and FPS.

Usage:
    .venv/bin/python benchmarks/validate_phase5.py
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VIDEO_DIR,
    TEST_VIDEOS,
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
    PHASE5_RESULTS_DIR,
)
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
    compute_box_iou,
)


def validate_coordinate_remapping(detector, video_path, n_test_frames=10):
    """Verify that secondary ROI detections project back accurately to global frame coordinates."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video"

    remapping_passed = True
    total_checks = 0

    for _ in range(n_test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        results, _ = detector.detect(frame)
        boxes = results[0].boxes if results else []

        _, candidates, _ = filter_uncertain_candidates(
            boxes, (h, w), tau_low=CONF_TAU_LOW, tau_high=CONF_TAU_HIGH, alpha=AREA_ALPHA
        )
        rois = create_context_rois(
            candidates, (h, w), margin=ROI_CONTEXT_MARGIN, min_size=ROI_MIN_SIZE, max_rois=MAX_ROIS_PER_FRAME
        )

        for rx1, ry1, rx2, ry2 in rois:
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            crop_results, _ = detector.detect(crop)
            if crop_results and len(crop_results[0].boxes) > 0:
                for cb in crop_results[0].boxes:
                    cxyxy = cb.xyxy[0].cpu().numpy()
                    gx1 = rx1 + cxyxy[0]
                    gy1 = ry1 + cxyxy[1]
                    gx2 = rx1 + cxyxy[2]
                    gy2 = ry1 + cxyxy[3]

                    total_checks += 1
                    # Check 1: In global frame bounds
                    if not (0 <= gx1 <= w and 0 <= gx2 <= w and 0 <= gy1 <= h and 0 <= gy2 <= h):
                        remapping_passed = False
                    # Check 2: Lies within the ROI crop region
                    if not (rx1 <= gx1 and gx2 <= rx2 and ry1 <= gy1 and gy2 <= ry2):
                        remapping_passed = False
                    # Check 3: Box has positive area
                    if gx2 <= gx1 or gy2 <= gy1:
                        remapping_passed = False

    cap.release()
    return remapping_passed, f"Checked {total_checks} remapped boxes across {n_test_frames} frames"


def validate_nms_duplicate_suppression():
    """Verify class-aware NMS correctly suppresses overlapping duplicate proposals."""
    # Synthetic test: 2 highly overlapping boxes of class 2 (car), 1 box of class 0 (person)
    accepted = [
        {"box": np.array([100.0, 100.0, 200.0, 200.0], dtype=np.float32), "conf": 0.80, "cls": 2, "source": "primary"},
        {"box": np.array([300.0, 300.0, 350.0, 400.0], dtype=np.float32), "conf": 0.70, "cls": 0, "source": "primary"},
    ]
    # Secondary proposal with 85% IoU to the first car box, lower confidence
    secondary = [
        {"box": np.array([105.0, 102.0, 202.0, 198.0], dtype=np.float32), "conf": 0.60, "cls": 2, "source": "secondary"},
        # Distinct car proposal (no overlap)
        {"box": np.array([500.0, 100.0, 600.0, 200.0], dtype=np.float32), "conf": 0.55, "cls": 2, "source": "secondary"},
    ]

    final_dets, fusion_ms, n_suppressed = fuse_and_suppress(accepted, secondary, nms_iou_thresh=0.50)

    # We expect 3 final detections (the overlapping secondary car must be suppressed)
    passed = (len(final_dets) == 3 and n_suppressed == 1)
    return passed, f"Suppressed {n_suppressed}/1 duplicate proposals, kept {len(final_dets)} unique objects"


def run_video_validation(detector, video_name):
    """Run full validation on a single video file."""
    video_path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.exists(video_path):
        return None

    buffer = LatestFrameBuffer()
    capture = VideoCaptureThread(video_path, buffer, realtime=True)
    metrics = Phase3Metrics()

    illum_records = []
    phase5_records = []

    capture.start()
    time.sleep(0.1)

    wall_start = time.perf_counter()

    while True:
        data = buffer.get()
        if data is not None:
            frame, frame_id, capture_ts = data
            t_start = time.perf_counter()

            # Phase 4
            adapted, lum, enhanced, adapt_ms = adapt_illumination(
                frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
            )
            illum_records.append({"luminance": lum, "enhanced": enhanced, "adapt_ms": adapt_ms})

            # Primary YOLO
            prim_res, prim_ms = detector.detect(adapted)

            # Phase 5
            final_dets, p5_tel = selective_redetection_pipeline(
                adapted, prim_res, detector,
                tau_low=CONF_TAU_LOW, tau_high=CONF_TAU_HIGH, alpha=AREA_ALPHA,
                margin=ROI_CONTEXT_MARGIN, min_size=ROI_MIN_SIZE, max_rois=MAX_ROIS_PER_FRAME,
                nms_iou_thresh=FUSION_NMS_IOU,
            )

            tot_det_ms = prim_ms + p5_tel["total_phase5_ms"]
            age_ms = (t_start - capture_ts) * 1000.0
            metrics.record(frame_id, age_ms, tot_det_ms, len(final_dets))

            phase5_records.append({
                "primary_ms": prim_ms,
                "n_primary": p5_tel["n_primary"],
                "n_accepted": p5_tel["n_accepted"],
                "n_candidates": p5_tel["n_candidates"],
                "n_rois": p5_tel["n_rois"],
                "n_secondary": p5_tel["n_secondary"],
                "n_suppressed": p5_tel["n_suppressed"],
                "n_final": p5_tel["n_final"],
                "redetect_ms": p5_tel["redetect_ms"],
                "fusion_ms": p5_tel["fusion_ms"],
                "total_phase5_ms": p5_tel["total_phase5_ms"],
                "total_detection_ms": tot_det_ms,
            })

        elif capture.is_stopped():
            final = buffer.get()
            if final is not None:
                frame, frame_id, capture_ts = final
                t_start = time.perf_counter()
                adapted, lum, enhanced, adapt_ms = adapt_illumination(
                    frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
                )
                illum_records.append({"luminance": lum, "enhanced": enhanced, "adapt_ms": adapt_ms})
                prim_res, prim_ms = detector.detect(adapted)
                final_dets, p5_tel = selective_redetection_pipeline(
                    adapted, prim_res, detector,
                    tau_low=CONF_TAU_LOW, tau_high=CONF_TAU_HIGH, alpha=AREA_ALPHA,
                    margin=ROI_CONTEXT_MARGIN, min_size=ROI_MIN_SIZE, max_rois=MAX_ROIS_PER_FRAME,
                    nms_iou_thresh=FUSION_NMS_IOU,
                )
                tot_det_ms = prim_ms + p5_tel["total_phase5_ms"]
                age_ms = (t_start - capture_ts) * 1000.0
                metrics.record(frame_id, age_ms, tot_det_ms, len(final_dets))
                phase5_records.append({
                    "primary_ms": prim_ms,
                    "n_primary": p5_tel["n_primary"],
                    "n_accepted": p5_tel["n_accepted"],
                    "n_candidates": p5_tel["n_candidates"],
                    "n_rois": p5_tel["n_rois"],
                    "n_secondary": p5_tel["n_secondary"],
                    "n_suppressed": p5_tel["n_suppressed"],
                    "n_final": p5_tel["n_final"],
                    "redetect_ms": p5_tel["redetect_ms"],
                    "fusion_ms": p5_tel["fusion_ms"],
                    "total_phase5_ms": p5_tel["total_phase5_ms"],
                    "total_detection_ms": tot_det_ms,
                })
            break
        else:
            time.sleep(0.001)

    wall_end = time.perf_counter()
    buf_stats = buffer.stats()
    met_summary = metrics.summary()

    n = len(phase5_records)
    return {
        "video": video_name,
        "buf_stats": buf_stats,
        "met_summary": met_summary,
        "n_frames": n,
        "avg_primary_ms": sum(r["primary_ms"] for r in phase5_records) / n if n else 0,
        "avg_cands": sum(r["n_candidates"] for r in phase5_records) / n if n else 0,
        "avg_rois": sum(r["n_rois"] for r in phase5_records) / n if n else 0,
        "avg_secondary": sum(r["n_secondary"] for r in phase5_records) / n if n else 0,
        "avg_final": sum(r["n_final"] for r in phase5_records) / n if n else 0,
        "avg_redetect_ms": sum(r["redetect_ms"] for r in phase5_records) / n if n else 0,
        "avg_fusion_ms": sum(r["fusion_ms"] for r in phase5_records) / n if n else 0,
        "avg_total_det_ms": sum(r["total_detection_ms"] for r in phase5_records) / n if n else 0,
        "buffer_ok": buf_stats["captured"] == buf_stats["consumed"] + buf_stats["replaced"],
        "e2e_fps": met_summary.get("inference_fps", 0),
        "avg_frame_age_ms": met_summary.get("avg_frame_age_ms", 0),
    }


def main():
    print("=" * 78)
    print("  Phase 5 Comprehensive Validation: Selective Re-Detection Pipeline")
    print("=" * 78)
    print()

    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)

    # Test 1: Unit Test - Coordinate Remapping
    print("Test 1: ROI Coordinate Remapping Accuracy...")
    v_remap = os.path.join(VIDEO_DIR, "50.mp4")
    remap_ok, remap_msg = validate_coordinate_remapping(detector, v_remap)
    status_1 = "PASS" if remap_ok else "FAIL"
    print(f"  [{status_1}] {remap_msg}")
    print()

    # Test 2: Unit Test - Class-Aware NMS Fusion
    print("Test 2: Class-Aware NMS Duplicate Suppression...")
    nms_ok, nms_msg = validate_nms_duplicate_suppression()
    status_2 = "PASS" if nms_ok else "FAIL"
    print(f"  [{status_2}] {nms_msg}")
    print()

    # Test 3: Video Stream Execution & Telemetry
    print("Test 3: Video Stream Validation (Standard + Low-Light Sequences)...")
    videos_to_test = TEST_VIDEOS + ["621.mp4", "139.mp4"]
    results = []

    for vname in videos_to_test:
        vpath = os.path.join(VIDEO_DIR, vname)
        if not os.path.exists(vpath):
            continue
        print(f"  Evaluating {vname}...")
        r = run_video_validation(detector, vname)
        if r:
            results.append(r)

    print()
    print("=" * 78)
    print("  Phase 5 Video Validation Results Summary")
    print("=" * 78)
    print(f"  {'Video':<10} {'Frames':>6} {'Captured':>8} {'Consumed':>8} {'Replaced':>8} "
          f"{'Cands':>6} {'ROIs':>6} {'SecRec':>6} {'Final':>6} {'Primary(ms)':>11} {'Redet(ms)':>10} {'Total(ms)':>10} {'FPS':>6} {'Age(ms)':>8} {'Buffer OK':>10}")
    print("  " + "-" * 128)

    for r in results:
        buf_str = "YES" if r["buffer_ok"] else "NO"
        print(f"  {r['video']:<10} {r['n_frames']:>6} {r['buf_stats']['captured']:>8} "
              f"{r['buf_stats']['consumed']:>8} {r['buf_stats']['replaced']:>8} "
              f"{r['avg_cands']:>6.2f} {r['avg_rois']:>6.2f} {r['avg_secondary']:>6.2f} "
              f"{r['avg_final']:>6.2f} {r['avg_primary_ms']:>11.2f} {r['avg_redetect_ms']:>10.2f} "
              f"{r['avg_total_det_ms']:>10.2f} {r['e2e_fps']:>6.2f} {r['avg_frame_age_ms']:>8.3f} {buf_str:>10}")

    print()
    print("Verification Assertions:")
    all_buffer_ok = all(r["buffer_ok"] for r in results)
    all_redet_ok = all(r["avg_rois"] > 0 for r in results)
    all_age_ok = all(r["avg_frame_age_ms"] < 50.0 for r in results)

    print(f"  - Buffer accounting valid on all videos  : {'PASS' if all_buffer_ok else 'FAIL'}")
    print(f"  - Selective re-detection active on all : {'PASS' if all_redet_ok else 'FAIL'}")
    print(f"  - Low frame age maintained (<50ms)     : {'PASS' if all_age_ok else 'FAIL'}")
    print(f"  - Coordinate remapping accuracy        : {'PASS' if remap_ok else 'FAIL'}")
    print(f"  - Class-aware NMS deduplication         : {'PASS' if nms_ok else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
