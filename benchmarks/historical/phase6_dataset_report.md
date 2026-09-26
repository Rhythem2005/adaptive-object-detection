# Phase 6 Dataset Inventory & Evaluation Subset Report

**Date:** 2026-09-26  
**Scope:** Dataset inventory, data-quality validation, and deterministic selection of a 20-video evaluation subset for Phase 6 (Temporal Tracking).  
**Inventory Manifest:** [`benchmarks/dataset_inventory.csv`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/benchmarks/dataset_inventory.csv)  
**Evaluation Subset Manifest:** [`benchmarks/phase6_evaluation_videos.csv`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/benchmarks/phase6_evaluation_videos.csv)  

---

## 1. Dataset Scope & Nature

> [!IMPORTANT]
> **Dataset Classification Notice:** This dataset is a **BDD100K video subset/mirror** (derived from the Berkeley DeepDrive BDDA video release), containing **1,429 organized driving video sequences**. It is not the full monolithic 100,000-video BDD100K corpus. All evaluation numbers and inventory statistics reflect this specific local research subset.

---

## 2. Available Videos by Condition

The video corpus is physically partitioned under [`data/bdd100k/videos_by_condition/`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/) into three illumination conditions:

| Condition Directory | Video Count | Spatial Resolution | Dominant Frame Rates | Duration Range | Mean Duration | Total Frames |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| [`day/`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/day/) | 1,394 | 1280 × 720 (100%) | 29.97 FPS (979), 30.0 FPS (255), 59.94 FPS (133), 60.0 FPS (22) | 10.00s – 32.00s | 10.65s | 446,169 |
| [`evening_low_light/`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/evening_low_light/) | 5 | 1280 × 720 (100%) | 29.97 FPS (2), 30.0 FPS (2), 59.94 FPS (1) | 10.00s – 10.01s | 10.01s | 1,800 |
| [`night/`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/) | 30 | 1280 × 720 (100%) | 29.97 FPS (23), 30.0 FPS (5), 59.94 FPS (1), 60.0 FPS (1) | 10.00s – 21.02s | 10.51s | 9,750 |
| **Total Corpus** | **1,429** | **1280 × 720 (100%)** | — | **10.00s – 32.00s** | **10.64s** | **457,719** |

---

## 3. Phase 6 Evaluation Subset Selection

A representative 20-video evaluation subset was deterministically selected across the three conditions:
- **Day:** 5 videos
- **Evening/Low-Light:** All 5 available videos
- **Night:** 10 videos
- **Total Selected:** **20 videos** (10,319 total frames, 303.3 seconds of driving footage)

### 3.1 Selection Manifest

| Video | Condition | Resolution | FPS | Frames | Duration | Selection Rationale |
|---|---|:---:|:---:|:---:|:---:|---|
| [`50.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/day/50.mp4) | Day | 1280×720 | 29.97 | 510 | 17.02s | Core Phase 3–5 daytime urban baseline; dense vehicle traffic at standard 30 FPS. |
| [`544.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/day/544.mp4) | Day | 1280×720 | 29.97 | 300 | 10.01s | Core Phase 3–5 daytime suburban baseline; mixed pedestrian and cyclist presence. |
| [`1600.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/day/1600.mp4) | Day | 1280×720 | 30.00 | 480 | 16.02s | Core Phase 3–5 daytime highway/overpass baseline; open multilane vehicle traffic. |
| [`100.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/day/100.mp4) | Day | 1280×720 | 59.94 | 600 | 10.01s | High-frame-rate daytime sequence (60 FPS); tests high-frequency temporal tracking. |
| [`1127.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/day/1127.mp4) | Day | 1280×720 | 59.94 | 1,379 | 23.01s | Extended-duration, high-frame-rate clip; tests long-term tracking continuity and drift resistance. |
| [`513.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/evening_low_light/513.mp4) | Evening/Low-Light | 1280×720 | 30.00 | 300 | 10.00s | Early transitional low-light sequence with high ambient illumination (mean L=97.24, 20.0% low-light). |
| [`976.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/evening_low_light/976.mp4) | Evening/Low-Light | 1280×720 | 30.00 | 300 | 10.00s | Moderate dusk transitional scene with evening sky and shadow contrast (mean L=50.72, 23.3% low-light). |
| [`1348.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/evening_low_light/1348.mp4) | Evening/Low-Light | 1280×720 | 29.97 | 300 | 10.01s | Dense transitional dusk sequence under vehicle headlight illumination (mean L=42.26, 60.0% low-light). |
| [`1451.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/evening_low_light/1451.mp4) | Evening/Low-Light | 1280×720 | 29.97 | 300 | 10.01s | Transitional roadway sequence with fading daylight (mean L=55.22, 16.7% low-light). |
| [`1760.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/evening_low_light/1760.mp4) | Evening/Low-Light | 1280×720 | 59.94 | 600 | 10.01s | High-frame-rate transitional low-light sequence (60 FPS, mean L=38.72, 80.0% low-light). |
| [`621.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/621.mp4) | Night | 1280×720 | 29.97 | 300 | 10.01s | Core Phase 4–5 nighttime benchmark; dark roadway with vehicle headlights and streetlights (mean L=29.14). |
| [`139.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/139.mp4) | Night | 1280×720 | 29.97 | 300 | 10.01s | Core Phase 4–5 mixed/flicker benchmark; fluctuating illumination across underpasses (mean L=41.13). |
| [`105.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/105.mp4) | Night | 1280×720 | 29.97 | 300 | 10.01s | Standard nighttime urban driving sequence with street reflections (mean L=38.69, 100% low-light). |
| [`250.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/250.mp4) | Night | 1280×720 | 29.97 | 300 | 10.01s | Urban night scene with commercial lighting, mixed pedestrian and vehicle presence (mean L=40.67). |
| [`262.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/262.mp4) | Night | 1280×720 | 60.00 | 600 | 10.00s | High-frame-rate nighttime driving sequence (60.00 FPS, 600 frames, mean L=21.78, 100% low-light). |
| [`812.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/812.mp4) | Night | 1280×720 | 59.94 | 600 | 10.01s | High-frame-rate nighttime arterial road sequence (59.94 FPS, 600 frames, mean L=32.94). |
| [`1695.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/1695.mp4) | Night | 1280×720 | 29.97 | 300 | 10.01s | Night arterial roadway with oncoming vehicle headlight glare (mean L=52.62, 43.3% low-light). |
| [`1813.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/1813.mp4) | Night | 1280×720 | 29.97 | 630 | 21.02s | Longest duration nighttime sequence (21.02s, 630 frames, mean L=32.93); tests extended track retention. |
| [`1905.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/1905.mp4) | Night | 1280×720 | 29.97 | 300 | 10.01s | Low-ambient-light unlit roadway scene (mean L=20.28, 100% low-light); tests extreme contrast degradation. |
| [`1918.mp4`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/data/bdd100k/videos_by_condition/night/1918.mp4) | Night | 1280×720 | 29.97 | 420 | 14.01s | Extended-duration nighttime highway sequence (14.01s, 420 frames, mean L=23.14, 96.7% low-light). |

---

## 4. Selection Methodology

The selection process was designed to satisfy four core experimental criteria:

1. **Longitudinal Cross-Phase Consistency:**
   Retains all standard benchmark sequences evaluated in Phase 3 (lightweight detection), Phase 4 (illumination adaptation), and Phase 5 (selective re-detection): `50.mp4`, `544.mp4`, `1600.mp4`, `621.mp4`, and `139.mp4`. This ensures direct apples-to-apples performance comparisons across developmental phases.
2. **Temporal Frame-Rate Diversity:**
   Includes both standard video feeds (~30 FPS: 29.97/30.0) and high-temporal-resolution streams (~60 FPS: 59.94/60.0) across all conditions (`100.mp4`, `1127.mp4`, `1760.mp4`, `262.mp4`, `812.mp4`). This enables evaluating tracking Kalman filters and association gates across distinct inter-frame motion displacements.
3. **Temporal Horizon Span:**
   Includes clips ranging from standard 10-second clips (300 frames) up to extended sequences exceeding 20 seconds (`1127.mp4` with 1,379 frames; `1813.mp4` with 630 frames). This tests trajectory continuity, track fragmentation, and identity switch behavior over long horizons.
4. **Environmental Photometric Variation:**
   Spans broad luminance conditions from bright daytime ($\mu_L = 126.8$) down to severe low-light ($\mu_L = 20.3$), alongside mixed/flickering illumination transitions (`139.mp4`, `1348.mp4`, `1760.mp4`).

---

## 5. Data Quality Validation Results

All 1,429 videos in the active repository folders were systematically audited using OpenCV:

- **Decode Integrity:** 1,429 of 1,429 files (100%) successfully initialized and decoded valid image frames.
- **Dimensional Consistency:** 1,429 of 1,429 files (100%) strictly match 1280 × 720 pixels (720p HD).
- **FPS Metadata:** All files possess valid nonzero frame rates (ranging between 29.83 and 120.0 FPS, with 98.7% falling at ~30 FPS or ~60 FPS).
- **Duration & Frame Count:** All files have valid frame counts ($\ge 300$ frames) and durations ($\ge 10.0$ seconds).
- **Selected Subset Integrity:**
  - Duplicate entries: 0
  - Missing file paths: 0
  - File read failures: 0
