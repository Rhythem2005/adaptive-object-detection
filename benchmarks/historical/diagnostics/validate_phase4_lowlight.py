"""Phase 4 Validation on low-light videos — validates the CLAHE enhancement path."""
import os
import sys
import time
import csv

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ILLUMINATION_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    VIDEO_DIR,
)
from src.illumination import compute_luminance, apply_clahe, adapt_illumination


# Low-light test videos found in dataset
LOW_LIGHT_VIDEOS = ["621.mp4", "139.mp4"]
# Normal (adequately illuminated) for comparison
NORMAL_VIDEOS = ["50.mp4"]


def validate_video(video_path):
    """Full validation of a single video covering all 7 criteria."""
    video_name = os.path.basename(video_path)
    print(f"\n{'=' * 60}")
    print(f"  Validating: {video_name}")
    print(f"{'=' * 60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = {
        "video": video_name,
        "total_frames": 0,
        "low_light_frames": 0,
        "normal_frames": 0,
        "bypass_identity_ok": 0,
        "bypass_pixel_ok": 0,
        "enhanced_lum_increased": 0,
        "shape_ok": 0,
        "dtype_ok": 0,
        "range_ok": 0,
    }
    
    luminances = []
    enhanced_deltas = []
    normal_times_ms = []
    enhanced_times_ms = []
    samples = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        output, luminance, was_enhanced, adapt_ms = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )
        
        luminances.append(luminance)
        results["total_frames"] += 1
        
        # V1: Classification
        if was_enhanced:
            results["low_light_frames"] += 1
            enhanced_times_ms.append(adapt_ms)
            
            # V4: Before/after luminance
            lum_after = compute_luminance(output)
            delta = lum_after - luminance
            enhanced_deltas.append(delta)
            if delta >= 0:
                results["enhanced_lum_increased"] += 1
            
            if len(samples) < 5:
                samples.append({
                    "frame": frame_idx,
                    "before": round(luminance, 2),
                    "after": round(lum_after, 2),
                    "delta": round(delta, 2),
                    "adapt_ms": round(adapt_ms, 3),
                })
        else:
            results["normal_frames"] += 1
            normal_times_ms.append(adapt_ms)
            
            # V3: Bypass identity
            if output is frame:
                results["bypass_identity_ok"] += 1
            if np.array_equal(output, frame):
                results["bypass_pixel_ok"] += 1
        
        # V5: Output validity
        if output.shape == frame.shape:
            results["shape_ok"] += 1
        if output.dtype == np.uint8:
            results["dtype_ok"] += 1
        if 0 <= output.min() and output.max() <= 255:
            results["range_ok"] += 1
    
    cap.release()
    
    n = results["total_frames"]
    n_low = results["low_light_frames"]
    n_normal = results["normal_frames"]
    
    # Report
    print(f"\n  V1: Luminance Classification")
    print(f"    Total frames    : {n}")
    print(f"    LOW-LIGHT       : {n_low} ({n_low / n * 100:.1f}%)")
    print(f"    NORMAL          : {n_normal} ({n_normal / n * 100:.1f}%)")
    print(f"    Avg luminance   : {sum(luminances) / len(luminances):.2f}")
    print(f"    Min luminance   : {min(luminances):.2f}")
    print(f"    Max luminance   : {max(luminances):.2f}")
    
    print(f"\n  V2: Enhancement Trigger Rate")
    print(f"    {n_low}/{n} = {n_low / n * 100:.1f}% of frames enhanced")
    
    print(f"\n  V3: Normal Frame Bypass")
    if n_normal > 0:
        id_ok = results["bypass_identity_ok"] == n_normal
        px_ok = results["bypass_pixel_ok"] == n_normal
        print(f"    Object identity : {results['bypass_identity_ok']}/{n_normal} {'PASS' if id_ok else 'FAIL'}")
        print(f"    Pixel identical : {results['bypass_pixel_ok']}/{n_normal} {'PASS' if px_ok else 'FAIL'}")
    else:
        print(f"    (No normal frames — all frames are low-light)")
    
    print(f"\n  V4: CLAHE Enhancement (before/after)")
    if n_low > 0:
        print(f"    Enhanced frames : {n_low}")
        print(f"    Avg Δ luminance : {sum(enhanced_deltas) / len(enhanced_deltas):.2f}")
        print(f"    Min Δ luminance : {min(enhanced_deltas):.2f}")
        print(f"    Max Δ luminance : {max(enhanced_deltas):.2f}")
        print(f"\n    Sample enhanced frames:")
        print(f"    {'Frame':>6}  {'Before':>8}  {'After':>8}  {'Δ':>8}  {'Time(ms)':>10}")
        for s in samples:
            print(f"    {s['frame']:>6}  {s['before']:>8.2f}  {s['after']:>8.2f}  {s['delta']:>8.2f}  {s['adapt_ms']:>10.3f}")
    else:
        print(f"    (No low-light frames)")
    
    print(f"\n  V5: Output Validity")
    all_valid = (results["shape_ok"] == n and results["dtype_ok"] == n and results["range_ok"] == n)
    print(f"    Shape HxWx3     : {results['shape_ok']}/{n}")
    print(f"    dtype uint8     : {results['dtype_ok']}/{n}")
    print(f"    range [0,255]   : {results['range_ok']}/{n}")
    print(f"    {'PASS' if all_valid else 'FAIL'}")
    
    print(f"\n  V7: Processing Overhead")
    if normal_times_ms:
        print(f"    Normal frames   : {len(normal_times_ms)}, avg {sum(normal_times_ms)/len(normal_times_ms):.3f} ms")
    if enhanced_times_ms:
        print(f"    Enhanced frames : {len(enhanced_times_ms)}, avg {sum(enhanced_times_ms)/len(enhanced_times_ms):.3f} ms")
    all_times = normal_times_ms + enhanced_times_ms
    if all_times:
        print(f"    Overall         : avg {sum(all_times)/len(all_times):.3f} ms, max {max(all_times):.3f} ms")
    
    return results


def main():
    print("=" * 60)
    print("Phase 4: Low-Light Video Validation")
    print("=" * 60)
    print(f"Threshold τ_illum   : {ILLUMINATION_THRESHOLD} (paper-specified)")
    print(f"CLAHE clipLimit     : {CLAHE_CLIP_LIMIT} (default)")
    print(f"CLAHE tileGridSize  : {CLAHE_TILE_GRID_SIZE} (default)")
    
    import platform
    print(f"Platform            : {platform.platform()}")
    print(f"OpenCV              : {cv2.__version__}")
    
    all_videos = LOW_LIGHT_VIDEOS + NORMAL_VIDEOS
    all_results = []
    
    for vname in all_videos:
        vpath = os.path.join(VIDEO_DIR, vname)
        if not os.path.exists(vpath):
            print(f"\n  [SKIP] {vpath} not found")
            continue
        
        result = validate_video(vpath)
        if result:
            all_results.append(result)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("CROSS-VIDEO SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        n = r["total_frames"]
        n_low = r["low_light_frames"]
        print(f"  {r['video']:>12}: {n_low}/{n} enhanced ({n_low/n*100:.1f}%)")
    
    print()
    print("Validation complete.")


if __name__ == "__main__":
    main()
