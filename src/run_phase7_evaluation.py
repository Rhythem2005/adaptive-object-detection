import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, VIDEO_DIR, ILLUMINATION_THRESHOLD,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, CONF_TAU_LOW, CONF_TAU_HIGH,
    AREA_ALPHA, ROI_CONTEXT_MARGIN, ROI_MIN_SIZE, MAX_ROIS_PER_FRAME,
    FUSION_NMS_IOU, TRACKER_IOU_THRESHOLD, TRACKER_MAX_AGE, TRACKER_MIN_HITS
)
from src.frame_manager import LatestFrameBuffer
from src.capture import VideoCaptureThread
from src.detector import YOLODetector
from src.illumination import adapt_illumination
from src.selective_redetection import selective_redetection_pipeline
from src.tracker import MultiObjectTracker, Track


def _load_evaluation_videos(csv_path):
    videos = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(row)
    return videos


def _find_video(video_entry):
    path = video_entry["path"]
    if os.path.exists(path):
        return path
        
    alt_path = os.path.join("data", "videos_by_condition", video_entry["condition"], video_entry["video"])
    if os.path.exists(alt_path):
        return alt_path
        
    alt = os.path.join(VIDEO_DIR, video_entry["video"])
    if os.path.exists(alt):
        return alt
    return None


def warmup_detector(detector):
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector.detect(dummy)
    dummy_crop = np.zeros((160, 160, 3), dtype=np.uint8)
    detector.detect(dummy_crop)


def run_configuration(detector, video_path, is_full=True):
    buffer = LatestFrameBuffer()
    capture = VideoCaptureThread(video_path, buffer, realtime=True)
    
    tracker = None
    if is_full:
        Track.reset_id_counter()
        tracker = MultiObjectTracker(
            iou_threshold=TRACKER_IOU_THRESHOLD,
            max_age=TRACKER_MAX_AGE,
            min_hits=TRACKER_MIN_HITS,
        )

    records = []
    capture.start()
    time.sleep(0.1)

    if capture.is_stopped() and buffer.get() is None:
        return None, None, 0.0

    wall_start = time.perf_counter()

    while True:
        data = buffer.get()
        if data is not None:
            _process_frame(data, detector, tracker, is_full, records)
        elif capture.is_stopped():
            final = buffer.get()
            if final is not None:
                _process_frame(final, detector, tracker, is_full, records)
            break
        else:
            time.sleep(0.001)

    wall_end = time.perf_counter()
    wall_s = wall_end - wall_start

    return buffer.stats(), records, wall_s


def _process_frame(data, detector, tracker, is_full, records):
    frame, frame_id, capture_ts = data
    
    if not is_full:
        # Baseline Pipeline
        results, det_ms = detector.detect(frame)
        e2e_ms = (time.perf_counter() - capture_ts) * 1000.0
        n_dets = len(results[0].boxes) if results and len(results) > 0 else 0
        records.append({
            "e2e_ms": e2e_ms,
            "det_ms": det_ms,
            "redetect_ms": 0.0,
            "track_ms": 0.0,
            "n_rois": 0,
            "redet_triggered": False,
            "n_dets": n_dets
        })
    else:
        # Full Adaptive Pipeline
        adapted_frame, _, _, _ = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )
        
        prim_res, prim_ms = detector.detect(adapted_frame)
        
        final_dets, p5_tel = selective_redetection_pipeline(
            adapted_frame, prim_res, detector,
            tau_low=CONF_TAU_LOW, tau_high=CONF_TAU_HIGH, alpha=AREA_ALPHA,
            margin=ROI_CONTEXT_MARGIN, min_size=ROI_MIN_SIZE,
            max_rois=MAX_ROIS_PER_FRAME, nms_iou_thresh=FUSION_NMS_IOU
        )
        
        active_tracks = tracker.update(final_dets, timestamp=capture_ts)
        e2e_ms = (time.perf_counter() - capture_ts) * 1000.0
        
        n_rois = p5_tel["n_rois"]
        
        records.append({
            "e2e_ms": e2e_ms,
            "det_ms": prim_ms,
            "redetect_ms": p5_tel["redetect_ms"],
            "track_ms": tracker.last_telemetry["tracking_ms"],
            "n_rois": n_rois,
            "redet_triggered": n_rois > 0,
            "n_dets": len(active_tracks)
        })


def aggregate_results(condition, config_name, video_results):
    """Aggregate per-video results into a single condition/config row."""
    tot_captured = 0
    tot_consumed = 0
    tot_replaced = 0
    tot_wall_s = 0.0
    
    all_e2e = []
    all_det = []
    all_redet = []
    all_track = []
    all_rois = []
    all_dets = []
    
    redet_frames = 0
    
    for (stats, records, wall_s) in video_results:
        tot_captured += stats["captured"]
        tot_consumed += stats["consumed"]
        tot_replaced += stats["replaced"]
        tot_wall_s += wall_s
        
        for r in records:
            all_e2e.append(r["e2e_ms"])
            all_det.append(r["det_ms"])
            all_redet.append(r["redetect_ms"])
            all_track.append(r["track_ms"])
            all_rois.append(r["n_rois"])
            all_dets.append(r["n_dets"])
            if r["redet_triggered"]:
                redet_frames += 1

    frames_processed = len(all_e2e)
    drop_rate = tot_replaced / tot_captured if tot_captured > 0 else 0.0
    effective_fps = frames_processed / tot_wall_s if tot_wall_s > 0 else 0.0
    
    tot_rois = sum(all_rois)
    
    if len(all_e2e) > 0:
        avg_e2e = np.mean(all_e2e)
        median_e2e = np.median(all_e2e)
        p95_e2e = np.percentile(all_e2e, 95)
        avg_det = np.mean(all_det)
        avg_redet = np.mean(all_redet)
        avg_track = np.mean(all_track)
        avg_dets = np.mean(all_dets)
    else:
        avg_e2e = median_e2e = p95_e2e = 0.0
        avg_det = avg_redet = avg_track = avg_dets = 0.0

    redet_rate = redet_frames / frames_processed if frames_processed > 0 else 0.0
    avg_rois_per_redet = tot_rois / redet_frames if redet_frames > 0 else 0.0

    return {
        "condition": condition,
        "configuration": config_name,
        "videos": len(video_results),
        "frames_received": tot_captured,
        "frames_processed": frames_processed,
        "frames_dropped": tot_replaced,
        "drop_rate": round(drop_rate, 4),
        "effective_fps": round(effective_fps, 2),
        "avg_end_to_end_ms": round(avg_e2e, 2),
        "median_end_to_end_ms": round(median_e2e, 2),
        "p95_end_to_end_ms": round(p95_e2e, 2),
        "avg_detector_ms": round(avg_det, 2),
        "avg_redetection_ms": round(avg_redet, 2) if config_name != "Baseline YOLOv8n" else None,
        "avg_tracking_ms": round(avg_track, 2) if config_name != "Baseline YOLOv8n" else None,
        "redetection_frames": redet_frames if config_name != "Baseline YOLOv8n" else None,
        "redetection_rate": round(redet_rate, 4) if config_name != "Baseline YOLOv8n" else None,
        "total_rois": tot_rois if config_name != "Baseline YOLOv8n" else None,
        "avg_rois_per_redetection_frame": round(avg_rois_per_redet, 2) if config_name != "Baseline YOLOv8n" else None,
        "avg_detections_per_frame": round(avg_dets, 2)
    }


def main():
    print("=" * 70)
    print("Phase 7 Benchmark: Evaluation & Ablation Infrastructure")
    print("=" * 70)
    
    csv_path = os.path.join("benchmarks", "phase6_evaluation_videos.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found")
        sys.exit(1)
        
    eval_videos = _load_evaluation_videos(csv_path)
    print(f"Loaded {len(eval_videos)} videos for evaluation.")
    
    detector = YOLODetector(MODEL_PATH, CONFIDENCE_THRESHOLD)
    print("Warming up detector...")
    warmup_detector(detector)
    print("Warmup complete.\n")
    
    # Group videos by condition
    videos_by_condition = {}
    for entry in eval_videos:
        cond = entry["condition"]
        if cond not in videos_by_condition:
            videos_by_condition[cond] = []
        videos_by_condition[cond].append(entry)
        
    configurations = [
        {"name": "Baseline YOLOv8n", "is_full": False},
        {"name": "Full Adaptive Vision", "is_full": True}
    ]
    
    final_results = []
    
    for config in configurations:
        config_name = config["name"]
        is_full = config["is_full"]
        
        print(f"\n>>> Running Configuration: {config_name}")
        
        for condition, entries in videos_by_condition.items():
            print(f"  --- Condition: {condition} ---")
            
            cond_results = []
            for entry in entries:
                video_path = _find_video(entry)
                if not video_path:
                    print(f"    [SKIP] {entry['video']} not found")
                    continue
                    
                print(f"    Evaluating {entry['video']} ... ", end="", flush=True)
                stats, records, wall_s = run_configuration(detector, video_path, is_full)
                if stats is None:
                    print("FAILED")
                else:
                    print(f"DONE ({stats['consumed']} processed)")
                    cond_results.append((stats, records, wall_s))
            
            if cond_results:
                agg_row = aggregate_results(condition, config_name, cond_results)
                final_results.append(agg_row)
                
    # Save CSV
    out_dir = os.path.join("benchmarks", "phase7")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "phase7_evaluation_results.csv")
    
    if final_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
            writer.writeheader()
            writer.writerows(final_results)
        print(f"\nSuccessfully wrote results to {out_csv}")
    else:
        print("\n[ERROR] No results were generated.")


if __name__ == "__main__":
    main()
