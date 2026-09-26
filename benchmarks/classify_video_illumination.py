"""Canonical Illumination Stratification Script for BDD100K Videos.

Measures perceptual lightness (L channel in L*a*b* color space, OpenCV 8-bit scale 0-255)
across evenly spaced frames for video clips in data/bdd100k/videos/.

Classification Rules:
    - Sample 30 frames evenly spaced across each video's full duration.
    - Measure mean L per frame and compute:
        * per-video mean_luminance
        * pct_frames_below_threshold (τ = 45, empirical threshold from Phase 4)
    - Classification Bins:
        * Day: < 10% of frames below τ=45
        * Evening/low-light: 10% - 90% of frames below τ=45 (mixed / transitional)
        * Night: > 90% of frames below τ=45
    - Borderline / Uncertain Rule:
        * If pct_frames_below_threshold is within 5 percentage points of a bin edge:
            - Near 10% edge: 5.0% <= pct <= 15.0% -> "uncertain"
            - Near 90% edge: 85.0% <= pct <= 95.0% -> "uncertain"
        * If a video is corrupt / 0 frames readable -> "uncertain"
    - Output format:
        video,condition,mean_luminance,pct_frames_below_threshold,frames_sampled,confidence
"""

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

VIDEO_DIR = "data/bdd100k/videos"
OUTPUT_CSV = "benchmarks/video_conditions.csv"
TAU_ILLUM = 45.0
NUM_SAMPLES = 30
TEST_5_VIDEOS = ["50.mp4", "544.mp4", "1600.mp4", "621.mp4", "139.mp4"]


def sample_video(fname):
    """Sample 30 frames sequentially across video and measure L-channel luminance."""
    vpath = os.path.join(VIDEO_DIR, fname)
    cap = cv2.VideoCapture(vpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return {
            "video": fname,
            "condition": "uncertain",
            "mean_luminance": 0.0,
            "pct_frames_below_threshold": 0.0,
            "frames_sampled": 0,
            "confidence": "measured",
            "reason": "Corrupt/unreadable video container (missing moov atom / 0 frames decoded)",
        }

    indices = set(np.linspace(0, total_frames - 1, NUM_SAMPLES, dtype=int))
    lums = []
    f_idx = 0

    while True:
        if f_idx in indices:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lums.append(float(np.mean(lab[:, :, 0])))
            if len(lums) >= len(indices):
                break
        else:
            if not cap.grab():
                break
        f_idx += 1

    cap.release()

    n = len(lums)
    if n == 0:
        return {
            "video": fname,
            "condition": "uncertain",
            "mean_luminance": 0.0,
            "pct_frames_below_threshold": 0.0,
            "frames_sampled": 0,
            "confidence": "measured",
            "reason": "No frames could be decoded",
        }

    mean_l = round(float(np.mean(lums)), 2)
    below_tau = sum(1 for x in lums if x < TAU_ILLUM)
    pct_below = round(float(below_tau / n * 100.0), 2)

    is_borderline_10 = 5.0 <= pct_below <= 15.0
    is_borderline_90 = 85.0 <= pct_below <= 95.0

    if is_borderline_10:
        condition = "uncertain"
        reason = f"Borderline Day / Evening/low-light (pct_below={pct_below:.2f}%, within 5% of 10% edge)"
    elif is_borderline_90:
        condition = "uncertain"
        reason = f"Borderline Evening/low-light / Night (pct_below={pct_below:.2f}%, within 5% of 90% edge)"
    elif pct_below < 10.0:
        condition = "Day"
        reason = f"Day (<10% below threshold: {pct_below:.2f}%)"
    elif pct_below > 90.0:
        condition = "Night"
        reason = f"Night (>90% below threshold: {pct_below:.2f}%)"
    else:
        condition = "Evening/low-light"
        reason = f"Evening/low-light (10%-90% below threshold: {pct_below:.2f}%)"

    return {
        "video": fname,
        "condition": condition,
        "mean_luminance": mean_l,
        "pct_frames_below_threshold": pct_below,
        "frames_sampled": n,
        "confidence": "measured",
        "reason": reason,
    }


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def main():
    parser = argparse.ArgumentParser(description="Classify BDD100K videos by illumination condition.")
    parser.add_argument(
        "--scope",
        choices=["all", "phases34"],
        default="all",
        help="Scope of videos: 'all' (1,432 videos in data/bdd100k/videos/) or 'phases34' (5 validation videos)",
    )
    args = parser.parse_args()

    if not os.path.exists(VIDEO_DIR):
        print(f"Error: {VIDEO_DIR} does not exist.")
        sys.exit(1)

    if args.scope == "phases34":
        files = TEST_5_VIDEOS
    else:
        files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")], key=natural_sort_key)

    print(f"Running illumination classification for {len(files)} video(s) (scope='{args.scope}')...")

    t0 = time.time()
    workers = min(8, len(files))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(sample_video, files))
    t1 = time.time()

    print(f"Sampling completed in {t1 - t0:.2f}s.")

    # Write output CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = [
        "video",
        "condition",
        "mean_luminance",
        "pct_frames_below_threshold",
        "frames_sampled",
        "confidence",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "video": r["video"],
                "condition": r["condition"],
                "mean_luminance": r["mean_luminance"],
                "pct_frames_below_threshold": r["pct_frames_below_threshold"],
                "frames_sampled": r["frames_sampled"],
                "confidence": r["confidence"],
            })

    print(f"Metadata CSV written to: {OUTPUT_CSV}")

    # Summary
    day_count = sum(1 for r in results if r["condition"] == "Day")
    eve_count = sum(1 for r in results if r["condition"] == "Evening/low-light")
    night_count = sum(1 for r in results if r["condition"] == "Night")
    unc_count = sum(1 for r in results if r["condition"] == "uncertain")

    print(f"Summary: Day={day_count}, Evening={eve_count}, Night={night_count}, Uncertain={unc_count}")


if __name__ == "__main__":
    main()
