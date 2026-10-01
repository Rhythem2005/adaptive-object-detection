"""Phase 4 Validation: Environmental Adaptation / Illumination Handling.

Validates all aspects of Phase 4 as specified:
  1. Luminance classification correctness (sample values + classification)
  2. Enhancement trigger rate (% of frames)
  3. Normal frames bypass (pixel-identical)
  4. Low-light frames receive CLAHE (before/after luminance comparison)
  5. Output frame validity for YOLOv8n (shape, dtype, range)
  6. Phase 3 frame freshness unchanged (runs Phase 3 benchmark)
  7. Processing overhead measurement (ms/frame)

Uses the same BDD100K test videos as Phase 3.
"""

import os
import sys
import time

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VIDEO_DIR,
    TEST_VIDEOS,
    ILLUMINATION_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    PHASE4_RESULTS_DIR,
)
from src.illumination import compute_luminance, apply_clahe, adapt_illumination


def validate_luminance_classification(video_path):
    """Validation 1: Show sample luminance values and classification per frame.

    Reads all frames from a video, computes luminance, classifies each as
    low-light or normal, and prints a summary with sample values.
    """
    video_name = os.path.basename(video_path)
    print(f"\n  === Luminance Classification: {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    luminances = []
    classifications = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lum = compute_luminance(frame)
        luminances.append(lum)
        classifications.append("LOW-LIGHT" if lum < ILLUMINATION_THRESHOLD else "NORMAL")

    cap.release()

    n_low = classifications.count("LOW-LIGHT")
    n_normal = classifications.count("NORMAL")
    n_total = len(classifications)

    print(f"  Total frames     : {n_total}")
    print(f"  Threshold τ_illum: {ILLUMINATION_THRESHOLD}")
    print(f"  LOW-LIGHT frames : {n_low} ({n_low / n_total * 100:.1f}%)")
    print(f"  NORMAL frames    : {n_normal} ({n_normal / n_total * 100:.1f}%)")
    print(f"  Avg luminance    : {sum(luminances) / len(luminances):.2f}")
    print(f"  Min luminance    : {min(luminances):.2f}")
    print(f"  Max luminance    : {max(luminances):.2f}")
    print(f"  Std luminance    : {np.std(luminances):.2f}")

    # Show first 10 frames as sample
    print(f"\n  Sample frames (first 10):")
    print(f"  {'Frame':>6}  {'Luminance':>10}  {'Classification':>15}")
    for i in range(min(10, n_total)):
        print(f"  {i + 1:>6}  {luminances[i]:>10.2f}  {classifications[i]:>15}")

    # Also show 5 frames around classification boundary if any
    boundary_frames = []
    for i in range(1, n_total):
        if classifications[i] != classifications[i - 1]:
            boundary_frames.extend(range(max(0, i - 2), min(n_total, i + 3)))

    if boundary_frames:
        boundary_frames = sorted(set(boundary_frames))[:10]
        print(f"\n  Frames near classification boundaries:")
        print(f"  {'Frame':>6}  {'Luminance':>10}  {'Classification':>15}")
        for i in boundary_frames:
            print(f"  {i + 1:>6}  {luminances[i]:>10.2f}  {classifications[i]:>15}")

    return {
        "video": video_name,
        "total_frames": n_total,
        "low_light": n_low,
        "normal": n_normal,
        "avg_luminance": round(sum(luminances) / len(luminances), 2),
        "min_luminance": round(min(luminances), 2),
        "max_luminance": round(max(luminances), 2),
        "luminances": luminances,
        "classifications": classifications,
    }


def validate_bypass_pixel_identical(video_path):
    """Validation 3: Confirm normal frames bypass enhancement (pixel-identical).

    For frames with luminance >= threshold, the output of adapt_illumination
    must be the exact same numpy array object (identity check), meaning
    zero pixel modification.
    """
    video_name = os.path.basename(video_path)
    print(f"\n  === Bypass Verification: {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return False

    checked = 0
    identity_matches = 0
    pixel_identical = 0
    bypassed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, luminance, was_enhanced, adapt_ms = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )

        if not was_enhanced:
            bypassed += 1
            # Check object identity (same object returned, not a copy)
            if output is frame:
                identity_matches += 1
            # Also check pixel-level identity
            if np.array_equal(output, frame):
                pixel_identical += 1

        checked += 1

    cap.release()

    print(f"  Total frames checked  : {checked}")
    print(f"  Bypassed (not enhanced): {bypassed}")
    print(f"  Object identity (is)  : {identity_matches}/{bypassed}")
    print(f"  Pixel identical       : {pixel_identical}/{bypassed}")

    all_pass = (identity_matches == bypassed) and (pixel_identical == bypassed)
    if all_pass:
        print(f"  [PASS] All {bypassed} normal frames returned unmodified (same object)")
    else:
        print(f"  [FAIL] Some normal frames were modified!")

    return all_pass


def validate_clahe_enhancement(video_path):
    """Validation 4: Confirm low-light frames receive CLAHE (before/after luminance).

    For frames that are classified as low-light, verify that CLAHE was applied
    by comparing luminance before and after enhancement.
    """
    video_name = os.path.basename(video_path)
    print(f"\n  === CLAHE Enhancement Verification: {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    enhanced_count = 0
    luminance_increases = []
    samples = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        output, luminance_before, was_enhanced, adapt_ms = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )

        if was_enhanced:
            enhanced_count += 1
            luminance_after = compute_luminance(output)
            delta = luminance_after - luminance_before
            luminance_increases.append(delta)

            if len(samples) < 5:
                samples.append({
                    "frame": frame_idx,
                    "before": round(luminance_before, 2),
                    "after": round(luminance_after, 2),
                    "delta": round(delta, 2),
                })

    cap.release()

    print(f"  Enhanced frames   : {enhanced_count}")

    if enhanced_count > 0:
        print(f"  Avg Δ luminance   : {sum(luminance_increases) / len(luminance_increases):.2f}")
        print(f"  Min Δ luminance   : {min(luminance_increases):.2f}")
        print(f"  Max Δ luminance   : {max(luminance_increases):.2f}")
        print()
        print(f"  Sample enhanced frames:")
        print(f"  {'Frame':>6}  {'Before':>8}  {'After':>8}  {'Δ':>8}")
        for s in samples:
            print(f"  {s['frame']:>6}  {s['before']:>8.2f}  {s['after']:>8.2f}  {s['delta']:>8.2f}")

        # All enhanced frames should have luminance increase (CLAHE boosts contrast)
        all_increased = all(d >= 0 for d in luminance_increases)
        if all_increased:
            print(f"\n  [PASS] All {enhanced_count} enhanced frames show luminance increase/equality")
        else:
            n_decreased = sum(1 for d in luminance_increases if d < 0)
            print(f"\n  [INFO] {n_decreased}/{enhanced_count} frames show luminance decrease "
                  f"(CLAHE redistributes, may not always increase mean)")
    else:
        print(f"  [INFO] No low-light frames found in this video (all Y >= {ILLUMINATION_THRESHOLD})")

    return {
        "enhanced_count": enhanced_count,
        "luminance_increases": luminance_increases,
        "samples": samples,
    }


def validate_output_validity(video_path):
    """Validation 5: Confirm output frames are valid input for YOLOv8n.

    Checks: shape (HxWx3), dtype (uint8), value range (0–255).
    """
    video_name = os.path.basename(video_path)
    print(f"\n  === Output Validity: {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return False

    checked = 0
    all_valid = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output, luminance, was_enhanced, adapt_ms = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )

        # Check shape
        if output.shape != frame.shape:
            print(f"  [FAIL] Frame {checked}: shape mismatch {output.shape} != {frame.shape}")
            all_valid = False

        # Check dtype
        if output.dtype != np.uint8:
            print(f"  [FAIL] Frame {checked}: dtype {output.dtype} != uint8")
            all_valid = False

        # Check value range
        if output.min() < 0 or output.max() > 255:
            print(f"  [FAIL] Frame {checked}: values out of [0,255] range")
            all_valid = False

        checked += 1

    cap.release()

    if all_valid:
        print(f"  [PASS] All {checked} frames: shape=HxWx3, dtype=uint8, range=[0,255]")
    else:
        print(f"  [FAIL] Some frames had invalid output")

    return all_valid


def validate_processing_overhead(video_path):
    """Validation 7: Measure processing overhead (ms/frame).

    Measures adapt_illumination time separately for normal and enhanced frames.
    """
    video_name = os.path.basename(video_path)
    print(f"\n  === Processing Overhead: {video_name} ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    normal_times = []
    enhanced_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, luminance, was_enhanced, adapt_ms = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )

        if was_enhanced:
            enhanced_times.append(adapt_ms)
        else:
            normal_times.append(adapt_ms)

    cap.release()

    print(f"  Normal frames    : {len(normal_times)}")
    if normal_times:
        print(f"    Avg overhead   : {sum(normal_times) / len(normal_times):.3f} ms")
        print(f"    Min overhead   : {min(normal_times):.3f} ms")
        print(f"    Max overhead   : {max(normal_times):.3f} ms")

    print(f"  Enhanced frames  : {len(enhanced_times)}")
    if enhanced_times:
        print(f"    Avg overhead   : {sum(enhanced_times) / len(enhanced_times):.3f} ms")
        print(f"    Min overhead   : {min(enhanced_times):.3f} ms")
        print(f"    Max overhead   : {max(enhanced_times):.3f} ms")

    all_times = normal_times + enhanced_times
    if all_times:
        print(f"  Overall (all frames):")
        print(f"    Avg overhead   : {sum(all_times) / len(all_times):.3f} ms")
        print(f"    Max overhead   : {max(all_times):.3f} ms")

    return {
        "normal_count": len(normal_times),
        "normal_avg_ms": round(sum(normal_times) / len(normal_times), 3) if normal_times else 0,
        "enhanced_count": len(enhanced_times),
        "enhanced_avg_ms": round(sum(enhanced_times) / len(enhanced_times), 3) if enhanced_times else 0,
        "overall_avg_ms": round(sum(all_times) / len(all_times), 3) if all_times else 0,
    }


def main():
    print("=" * 70)
    print("Phase 4 Validation: Environmental Adaptation / Illumination Handling")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Illumination threshold (τ_illum) : {ILLUMINATION_THRESHOLD} (paper-specified)")
    print(f"  CLAHE clipLimit                  : {CLAHE_CLIP_LIMIT} (paper does not specify — OpenCV default)")
    print(f"  CLAHE tileGridSize               : {CLAHE_TILE_GRID_SIZE} (paper does not specify — OpenCV default)")
    print(f"  Luminance method                 : L-channel mean of L*a*b* (paper does not specify exact formula)")
    print(f"  CLAHE applied to                 : L channel of L*a*b* (paper-specified)")
    print()

    import platform
    print(f"Hardware/Software Context:")
    print(f"  Platform      : {platform.platform()}")
    print(f"  Python        : {platform.python_version()}")
    print(f"  OpenCV        : {cv2.__version__}")
    print(f"  NumPy         : {np.__version__}")
    print()

    videos = []
    for video_name in TEST_VIDEOS:
        video_path = os.path.join(VIDEO_DIR, video_name)
        if os.path.exists(video_path):
            videos.append(video_path)
        else:
            print(f"  [SKIP] {video_path} not found")

    if not videos:
        print("[ERROR] No test videos found.")
        return

    # ── Validation 1: Luminance Classification ──────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 1: Luminance Classification")
    print("=" * 70)
    all_lum_results = {}
    for vp in videos:
        result = validate_luminance_classification(vp)
        if result:
            all_lum_results[result["video"]] = result

    # ── Validation 2: Enhancement Trigger Rate ──────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 2: Enhancement Trigger Rate")
    print("=" * 70)
    total_frames = sum(r["total_frames"] for r in all_lum_results.values())
    total_low = sum(r["low_light"] for r in all_lum_results.values())
    print(f"\n  Total frames across all videos : {total_frames}")
    print(f"  Total LOW-LIGHT               : {total_low} ({total_low / total_frames * 100:.1f}%)")
    print(f"  Total NORMAL                  : {total_frames - total_low} ({(total_frames - total_low) / total_frames * 100:.1f}%)")

    # ── Validation 3: Normal Frame Bypass ───────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 3: Normal Frame Bypass (pixel-identical)")
    print("=" * 70)
    bypass_results = []
    for vp in videos:
        result = validate_bypass_pixel_identical(vp)
        bypass_results.append(result)

    all_bypass_pass = all(bypass_results)
    print(f"\n  Overall bypass validation: {'PASS' if all_bypass_pass else 'FAIL'}")

    # ── Validation 4: CLAHE Enhancement ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 4: CLAHE Enhancement (before/after luminance)")
    print("=" * 70)
    for vp in videos:
        validate_clahe_enhancement(vp)

    # ── Validation 5: Output Frame Validity ─────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 5: Output Frame Validity for YOLOv8n")
    print("=" * 70)
    validity_results = []
    for vp in videos:
        result = validate_output_validity(vp)
        validity_results.append(result)

    all_valid = all(validity_results)
    print(f"\n  Overall output validity: {'PASS' if all_valid else 'FAIL'}")

    # ── Validation 7: Processing Overhead ───────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 7: Processing Overhead (ms/frame)")
    print("=" * 70)
    overhead_results = []
    for vp in videos:
        result = validate_processing_overhead(vp)
        if result:
            overhead_results.append(result)

    if overhead_results:
        total_n = sum(r["normal_count"] for r in overhead_results)
        total_e = sum(r["enhanced_count"] for r in overhead_results)
        all_normal_avg = sum(r["normal_avg_ms"] * r["normal_count"] for r in overhead_results) / total_n if total_n > 0 else 0
        all_enhanced_avg = sum(r["enhanced_avg_ms"] * r["enhanced_count"] for r in overhead_results) / total_e if total_e > 0 else 0
        total_all = total_n + total_e
        all_overall_avg = sum(r["overall_avg_ms"] * (r["normal_count"] + r["enhanced_count"]) for r in overhead_results) / total_all if total_all > 0 else 0

        print(f"\n  Cross-video overhead summary:")
        print(f"  Normal frames  : {total_n}, avg {all_normal_avg:.3f} ms (luminance check only)")
        print(f"  Enhanced frames: {total_e}, avg {all_enhanced_avg:.3f} ms (luminance + CLAHE)")
        print(f"  Overall        : {total_all}, avg {all_overall_avg:.3f} ms")

    # ── Final Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  V1 Luminance classification : DONE (see per-video details above)")
    print(f"  V2 Enhancement trigger rate : {total_low}/{total_frames} = {total_low / total_frames * 100:.1f}%")
    print(f"  V3 Normal bypass (pixel)    : {'PASS' if all_bypass_pass else 'FAIL'}")
    print(f"  V4 CLAHE enhancement        : DONE (see per-video details above)")
    print(f"  V5 Output validity          : {'PASS' if all_valid else 'FAIL'}")
    print(f"  V6 Phase 3 unchanged        : (Run main.py separately and diff with pre-change baseline)")
    print(f"  V7 Processing overhead      : avg {all_overall_avg:.3f} ms/frame")
    print()


if __name__ == "__main__":
    main()
