"""Phase 4 — CLAHE Delta Distribution + Luminance Flicker Analysis.

Task 2 of the pre-Phase-5 cleanup:
  1. Full per-frame luminance delta distribution for 621.mp4 and 139.mp4
     (mean, median, std, min, max, percentiles)
  2. Luminance flicker analysis for 139.mp4 — consecutive-frame classification
     flips at the τ=45 boundary that would cause visible brightness popping.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VIDEO_DIR,
    ILLUMINATION_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    PHASE4_RESULTS_DIR,
)
from src.illumination import compute_luminance, adapt_illumination


def analyze_delta_distribution(video_path):
    """Compute full per-frame luminance delta distribution."""
    video_name = os.path.basename(video_path)
    print(f"\n{'=' * 65}")
    print(f"  CLAHE Delta Distribution: {video_name}")
    print(f"{'=' * 65}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return None

    # Collect per-frame data
    frames_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        output, lum_before, was_enhanced, adapt_ms = adapt_illumination(
            frame, ILLUMINATION_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE
        )

        lum_after = compute_luminance(output) if was_enhanced else lum_before
        delta = lum_after - lum_before if was_enhanced else 0.0

        frames_data.append({
            "frame": frame_idx,
            "lum_before": lum_before,
            "lum_after": lum_after,
            "delta": delta,
            "enhanced": was_enhanced,
            "classification": "LOW-LIGHT" if was_enhanced else "NORMAL",
        })

    cap.release()

    n_total = len(frames_data)
    enhanced = [f for f in frames_data if f["enhanced"]]
    normal = [f for f in frames_data if not f["enhanced"]]
    n_enh = len(enhanced)
    n_norm = len(normal)

    print(f"\n  Total frames    : {n_total}")
    print(f"  Enhanced        : {n_enh} ({n_enh / n_total * 100:.1f}%)")
    print(f"  Normal (bypass) : {n_norm}")

    # --- Full luminance distribution (all frames) ---
    all_lums = [f["lum_before"] for f in frames_data]
    print(f"\n  Luminance Distribution (all frames, pre-enhancement):")
    print(f"    Mean    : {np.mean(all_lums):.2f}")
    print(f"    Median  : {np.median(all_lums):.2f}")
    print(f"    Std     : {np.std(all_lums):.2f}")
    print(f"    Min     : {np.min(all_lums):.2f}")
    print(f"    Max     : {np.max(all_lums):.2f}")
    print(f"    P5      : {np.percentile(all_lums, 5):.2f}")
    print(f"    P25     : {np.percentile(all_lums, 25):.2f}")
    print(f"    P75     : {np.percentile(all_lums, 75):.2f}")
    print(f"    P95     : {np.percentile(all_lums, 95):.2f}")

    # --- Delta distribution (enhanced frames only) ---
    if n_enh > 0:
        deltas = [f["delta"] for f in enhanced]
        lum_befores = [f["lum_before"] for f in enhanced]
        lum_afters = [f["lum_after"] for f in enhanced]

        print(f"\n  CLAHE Luminance Delta Distribution ({n_enh} enhanced frames):")
        print(f"    Δ Mean    : {np.mean(deltas):.2f}")
        print(f"    Δ Median  : {np.median(deltas):.2f}")
        print(f"    Δ Std     : {np.std(deltas):.2f}")
        print(f"    Δ Min     : {np.min(deltas):.2f}")
        print(f"    Δ Max     : {np.max(deltas):.2f}")
        print(f"    Δ P5      : {np.percentile(deltas, 5):.2f}")
        print(f"    Δ P25     : {np.percentile(deltas, 25):.2f}")
        print(f"    Δ P75     : {np.percentile(deltas, 75):.2f}")
        print(f"    Δ P95     : {np.percentile(deltas, 95):.2f}")

        print(f"\n  Pre-enhancement luminance ({n_enh} enhanced frames):")
        print(f"    Mean    : {np.mean(lum_befores):.2f}")
        print(f"    Min     : {np.min(lum_befores):.2f}")
        print(f"    Max     : {np.max(lum_befores):.2f}")

        print(f"\n  Post-enhancement luminance ({n_enh} enhanced frames):")
        print(f"    Mean    : {np.mean(lum_afters):.2f}")
        print(f"    Min     : {np.min(lum_afters):.2f}")
        print(f"    Max     : {np.max(lum_afters):.2f}")

        # Histogram buckets for deltas
        bins = np.arange(0, max(deltas) + 3, 2)
        hist, edges = np.histogram(deltas, bins=bins)
        print(f"\n  Δ Histogram (bucket width = 2.0):")
        for i in range(len(hist)):
            if hist[i] > 0:
                bar = "█" * min(hist[i], 50)
                print(f"    [{edges[i]:5.1f}, {edges[i + 1]:5.1f}): {hist[i]:>4}  {bar}")

    return frames_data


def analyze_flicker(frames_data, video_name):
    """Analyze consecutive-frame classification flips (luminance flicker).

    If a frame is classified as LOW-LIGHT and the next as NORMAL (or vice
    versa), the CLAHE enhancement toggles on/off between adjacent frames.
    This produces a visible brightness pop: the enhanced frame is ~20 lum
    brighter than the un-enhanced neighbor, despite similar raw luminance.
    """
    print(f"\n{'=' * 65}")
    print(f"  Luminance Flicker Analysis: {video_name}")
    print(f"{'=' * 65}")

    n = len(frames_data)
    flips = []

    for i in range(1, n):
        prev = frames_data[i - 1]
        curr = frames_data[i]
        if prev["classification"] != curr["classification"]:
            flips.append({
                "frame": curr["frame"],
                "prev_class": prev["classification"],
                "curr_class": curr["classification"],
                "prev_lum_before": round(prev["lum_before"], 2),
                "curr_lum_before": round(curr["lum_before"], 2),
                "prev_lum_after": round(prev["lum_after"], 2),
                "curr_lum_after": round(curr["lum_after"], 2),
                "raw_delta": round(abs(curr["lum_before"] - prev["lum_before"]), 2),
                "output_delta": round(abs(curr["lum_after"] - prev["lum_after"]), 2),
            })

    print(f"\n  Total frames      : {n}")
    print(f"  Classification flips: {len(flips)}")

    if not flips:
        print(f"  [OK] No flicker — all frames consistently classified")
        return flips

    # Quantify the brightness pop
    raw_deltas = [f["raw_delta"] for f in flips]
    output_deltas = [f["output_delta"] for f in flips]

    print(f"\n  Flicker characterization:")
    print(f"    Avg raw luminance difference at flip    : {np.mean(raw_deltas):.2f}")
    print(f"    Avg OUTPUT luminance difference at flip : {np.mean(output_deltas):.2f}")
    print(f"    Max OUTPUT luminance difference at flip : {np.max(output_deltas):.2f}")
    print()
    print(f"  Explanation: At a flip boundary, one frame gets CLAHE (+~20 lum)")
    print(f"  and its neighbor doesn't, so even though the raw luminance changes")
    print(f"  by only ~{np.mean(raw_deltas):.1f}, the output luminance changes by ~{np.mean(output_deltas):.1f}.")
    print()

    # Show every flip transition
    print(f"  All flip transitions:")
    print(f"  {'Frame':>6}  {'Transition':>22}  {'Raw Δ':>7}  {'Output Δ':>9}  "
          f"{'Prev Out':>9}  {'Curr Out':>9}")
    for f in flips:
        trans = f"{f['prev_class'][:3]}→{f['curr_class'][:3]}"
        print(f"  {f['frame']:>6}  {trans:>22}  {f['raw_delta']:>7.2f}  "
              f"{f['output_delta']:>9.2f}  {f['prev_lum_after']:>9.2f}  "
              f"{f['curr_lum_after']:>9.2f}")

    # Streak analysis — how long between flips?
    if len(flips) >= 2:
        gaps = [flips[i + 1]["frame"] - flips[i]["frame"] for i in range(len(flips) - 1)]
        print(f"\n  Streak lengths between flips:")
        print(f"    Min gap : {min(gaps)} frames")
        print(f"    Max gap : {max(gaps)} frames")
        print(f"    Avg gap : {np.mean(gaps):.1f} frames")

        # Check for rapid oscillation (flips within 2 frames)
        rapid = [g for g in gaps if g <= 2]
        if rapid:
            print(f"    Rapid flips (≤2 frame gap): {len(rapid)}")
            print(f"    [WARN] Rapid oscillation detected — potential visible flicker")
        else:
            print(f"    No rapid oscillation (all gaps > 2 frames)")

    # Quantify the practical impact
    print(f"\n  Practical impact assessment:")
    if np.mean(output_deltas) > 15:
        print(f"    [WARN] Mean output brightness jump of {np.mean(output_deltas):.1f} "
              f"at flip boundaries is perceptually significant.")
        print(f"    Consider: hysteresis band (e.g., trigger at Y<45, clear at Y>50)")
        print(f"    to prevent oscillation near the threshold.")
    else:
        print(f"    [OK] Output brightness jumps are modest ({np.mean(output_deltas):.1f})")

    return flips


def main():
    print("=" * 65)
    print("Phase 4 — CLAHE Delta Distribution + Flicker Analysis")
    print("=" * 65)
    print(f"Threshold τ_illum = {ILLUMINATION_THRESHOLD}")

    # 621.mp4: night driving (100% enhanced)
    v621 = os.path.join(VIDEO_DIR, "621.mp4")
    v139 = os.path.join(VIDEO_DIR, "139.mp4")

    results = {}

    if os.path.exists(v621):
        data_621 = analyze_delta_distribution(v621)
        results["621.mp4"] = data_621
    else:
        print(f"\n  [SKIP] 621.mp4 not found")

    if os.path.exists(v139):
        data_139 = analyze_delta_distribution(v139)
        results["139.mp4"] = data_139

        # Flicker analysis only relevant for 139.mp4 (mixed classification)
        flips = analyze_flicker(data_139, "139.mp4")
        results["139_flips"] = flips
    else:
        print(f"\n  [SKIP] 139.mp4 not found")

    # Save per-frame CSV for both videos
    import csv
    os.makedirs(PHASE4_RESULTS_DIR, exist_ok=True)

    for vname, data in results.items():
        if vname.endswith("_flips") or data is None:
            continue
        csv_path = os.path.join(PHASE4_RESULTS_DIR,
                                vname.replace(".mp4", "_delta_distribution.csv"))
        with open(csv_path, "w", newline="") as f:
            fields = ["frame", "lum_before", "lum_after", "delta",
                       "enhanced", "classification"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in data:
                writer.writerow({
                    "frame": row["frame"],
                    "lum_before": round(row["lum_before"], 2),
                    "lum_after": round(row["lum_after"], 2),
                    "delta": round(row["delta"], 2),
                    "enhanced": row["enhanced"],
                    "classification": row["classification"],
                })
        print(f"\n  Saved: {csv_path}")

    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()
