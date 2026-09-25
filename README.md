# Adaptive Vision

### Adaptive Real-Time Object Detection for Driving Videos Using Selective Re-Detection and Temporal Tracking

**GitHub Repository:** `adaptive-vision`  
**Project Category:** B.Tech Minor Project / Applied Computer Vision Research  
**Current Milestone:** Phase 5 Completed (Confidence/Uncertainty Filtering, Context Expansion, Localized Re-Detection & NMS Fusion)

---

## 1. Overview

**Adaptive Vision** is an applied computer vision research project engineering an adaptive, deterministic real-time object detection architecture specifically optimized for automotive driving video streams.

Autonomous driving systems and Advanced Driver Assistance Systems (ADAS) must balance strict real-time constraints (high frame rates, bounded latency, minimal frame staleness) against detection fidelity across dynamic roadway environments. Conventional object detection architectures evaluate every sequential video frame uniformly through deep neural networks at full resolution, causing computational bottlenecks, redundant inferences, and high latency during resource contention.

Adaptive Vision investigates a multi-stage hybrid pipeline:
1. **Asynchronous Video Ingestion:** Decouples video capture from inference via a thread-safe, bounded fresh-frame buffer (`capacity = 1`) that eliminates queuing delay and guarantees near-zero frame staleness.
2. **Environmental Adaptation:** Evaluates dynamic frame illumination (average luma) and applies real-time localized contrast adjustments (CLAHE/gamma) only when conditions demand it.
3. **Lightweight Primary Detection:** Deploys a fast primary detector (YOLOv8n) to establish candidate bounding boxes at ~18 ms latency.
4. **Confidence & Uncertainty Filtering:** Evaluates detection confidence and multi-class classification entropy/margin to identify low-confidence or ambiguous detections.
5. **Context-Expanded Localized Re-Detection:** Extracts regions of interest (ROIs) around ambiguous candidates with context expansion ($\beta = 1.0$, min size $64 \times 64$ px) to prevent convolutional feature collapse, running high-resolution secondary inference strictly over local patches.
6. **Class-Aware NMS Fusion:** Remaps local coordinates to the full frame via affine translation and merges primary and secondary proposals using class-aware Non-Maximum Suppression ($\text{IoU} = 0.50$).
7. **Comprehensive Telemetry:** Monitors per-frame capture-to-display latency, isolated stage runtimes, buffer drop rates, and object recovery deltas.

---

## 2. Research Objectives

1. **Eliminate Frame Latency Drift:** Mitigate FIFO queue buildup and frame staleness in streaming video by enforcing a bounded fresh-frame buffer policy.
2. **Empirical Ground-Truth Failure Analysis:** Quantify the scale-dependent error modes of lightweight models (e.g., small-object recall drop on BDD100K driving scenes).
3. **Adaptive Inference Allocation:** Trigger localized secondary inference dynamically on difficult or ambiguous regions rather than executing dense high-capacity models on entire frames.
4. **Prevent Context Collapse:** Maintain sufficient contextual surrounding margins ($\beta \ge 1.0$) during crop extraction so that localized detectors retain spatial cues for small/occluded objects.
5. **Deterministic Latency & Telemetry:** Maintain strict latency accounting across all pipeline stages to evaluate the real-world operational trade-offs of selective re-detection.

---

## 3. Pipeline Architecture

```text
VIDEO STREAM (H.264 / 720p @ ~30 FPS)
      │
      ▼
┌──────────────────────────────────────────────┐
│  Phase 2: Adaptive Frame Controller          │ [COMPLETED]
│  - Background capture thread                 │
│  - Bounded FreshFrameBuffer (capacity = 1)   │
│  - Zero queuing delay / bounded frame age    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 4: Environmental Adaptation           │ [COMPLETED]
│  - Average luma calculation (L = 0.299R...)  │
│  - Adaptive thresholding (T_low=45, T_high)  │
│  - Isolated CLAHE enhancement (~0.8-2.1 ms)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 3: Primary Lightweight YOLO (YOLOv8n) │ [COMPLETED]
│  - Full-frame high-speed inference (~18 ms)  │
│  - Initial bounding box & confidence scoring │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 5: Uncertainty Filtering & Extraction │ [COMPLETED]
│  - Confidence band: 0.25 <= conf < 0.65      │
│  - Multiclass margin uncertainty (diff<0.05) │
│  - Context expansion factor beta = 1.0       │
│  - Minimum crop constraint (>= 64x64 px)     │
│  - Bounded ROI budget (max K = 4 per frame)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 5: Localized Re-Detection & Fusion    │ [COMPLETED]
│  - High-resolution inference on ROI crops    │
│  - Affine local-to-global coordinate remap   │
│  - Class-aware NMS fusion (IoU = 0.50)       │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase 6: Temporal Tracking                  │ [PLANNED]
│  - Inter-frame track association (SORT/Byte) │
│  - Trajectory preservation & state recovery  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Telemetry & Profiling (FPS, Latency, Age)   │ [COMPLETED]
│  - Stage-by-stage runtime profiling          │
│  - Buffer replacement & drop rate metrics    │
│  - Final detection yield and recovery delta  │
└──────────────────────────────────────────────┘
```

---

## 4. Completed Implementation Phases

### Phase 1: Dataset Acquisition & Verification
- **Video Dataset:** Acquired 1,435 video clips from the **BDD100K / BDDA** dataset.
- **Stream Verification:** Formats verified as H.264/AVC, 1280×720 resolution, 29.97 FPS, 300 frames per sequence.
- **Repository Setup:** Configured project layout, dependencies, and strict `.gitignore` rules for media, weights, and caches.

### Phase 2: Adaptive Ingestion & Fresh-Frame Buffer Controller
- **Producer-Consumer Threading (`src/capture.py`, `src/frame_manager.py`):**
  - Dedicated capture thread ingests video frames continuously at native stream FPS.
  - Thread-safe synchronization using `threading.Lock` and `threading.Event`.
- **Bounded Buffer Policy:**
  - `FreshFrameBuffer(capacity=1)` stores only the most recently captured frame.
  - When inference takes longer than inter-frame arrival time ($>33.3\text{ ms}$), intermediate frames are dropped atomically in the buffer.
  - Guarantees capture-to-inference frame age remains bounded ($\sim 18\text{ ms}$) without accumulating FIFO latency lag.
  - Complete buffer accounting: $\text{captured} = \text{consumed} + \text{replaced}$ with zero unmanaged frame leaks.

### Phase 3: Primary Lightweight Detector & Empirical Baseline
- **Detector Implementation (`src/detector.py`):**
  - Integrated YOLOv8n running in isolated inference mode.
  - Full-frame inference latency: $\sim 17.8\text{ ms}$ on standard test clips ($\sim 55\text{ FPS}$ throughput).
- **Ground-Truth Baseline Evaluation on BDD100K Val ($N=425$ images):**
  - Evaluated plain YOLOv8n against 425 ground-truth annotated driving scenes across a **7-class automotive vocabulary**:
    `car`, `bus`, `truck`, `pedestrian`, `rider`, `bicycle`, `traffic light`.
  - *Stated Limitation:* `traffic sign` was excluded from the paper's original 8-class vocabulary because standard COCO-pretrained YOLOv8n does not have an equivalent class.
  - *Scale-Dependent Empirical Findings:*
    - **Small objects ($<32^2\text{ px}$):** 3,612 GT objects $\to$ **Recall = 5.84%** (3,401 missed).
    - **Medium objects ($32^2 \le \text{area} < 96^2$):** 2,217 GT objects $\to$ Recall = 44.65%.
    - **Large objects ($\ge 96^2$):** 986 GT objects $\to$ Recall = 80.53%.
  - This empirical discrepancy demonstrated that lightweight full-frame models miss over 94% of distant or small traffic participants, establishing the direct motivation for localized selective re-detection.

### Phase 4: Environmental Adaptation (Adaptive Illumination)
- **Dynamic Illumination Analysis (`src/illumination.py`):**
  - Computes per-frame average perceived luma:
    $$L = \frac{1}{W \cdot H}\sum (0.299R + 0.587G + 0.114B)$$
  - Evaluates against dual thresholds: $T_{\text{low}} = 45.0$ and $T_{\text{high}} = 210.0$.
- **Targeted Enhancement:**
  - Underexposed frames ($L < T_{\text{low}}$) undergo Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in LAB color space (`clip_limit = 2.0`, `tile_grid = (8, 8)`).
  - Overexposed frames ($L > T_{\text{high}}$) undergo dynamic gamma correction ($\gamma = 1.5$).
- **Isolated Telemetry:**
  - Evaluated strictly in isolation from detector compute, demonstrating $\sim 0.83 - 2.30\text{ ms}$ execution overhead.

### Phase 5: Confidence/Uncertainty Filtering, Context Expansion, Localized Re-Detection & NMS Fusion
- **Selective Re-Detection Module (`src/selective_redetection.py`):**
  - **Uncertainty Criteria:**
    - Primary candidate detection bounding box $[x_1, y_1, x_2, y_2]$ with confidence $c$.
    - Filter band: $\tau_{\text{low}} = 0.25 \le c < \tau_{\text{high}} = 0.65$.
    - Multi-class top-2 margin uncertainty: $(c_1 - c_2) < \alpha = 0.05$.
  - **Context Expansion & Crop Boundary Preservation:**
    - *Discovery:* Testing revealed that tight bounding box cropping ($\beta = 0.20$, $<64\text{ px}$) caused receptive field and context collapse in CNN feature extractors, dropping confidence to $0.00$.
    - *Resolution:* Configured context expansion factor $\beta = 1.0$ (expanding width and height by 100% on each side, doubling crop footprint) with boundary clamping and a minimum size constraint ($64 \times 64\text{ px}$). This boosted candidate re-detection confidence from $0.39 \to 0.74$.
  - **Bounded Execution Budget:**
    - Limits ROI count to $K = 4$ per frame, prioritizing the most uncertain candidates first.
  - **Affine Coordinate Remapping:**
    - Secondary inference is executed directly over the high-resolution localized ROI crops.
    - Remaps local crop coordinates $(u_1, v_1, u_2, v_2)$ back to the global frame:
      $$x_1^{\text{global}} = x_1^{\text{roi}} + u_1, \quad y_1^{\text{global}} = y_1^{\text{roi}} + v_1, \quad x_2^{\text{global}} = x_1^{\text{roi}} + u_2, \quad y_2^{\text{global}} = y_1^{\text{roi}} + v_2$$
  - **Class-Aware NMS Fusion:**
    - Retains primary high-confidence detections ($c \ge \tau_{\text{high}}$) and combines them with remapped secondary detections.
    - Applies class-aware Non-Maximum Suppression ($\text{IoU}_{\text{thresh}} = 0.50$) to suppress lower-confidence overlapping anchors and prevent duplicate proposals.
  - **Empirical Validation:**
    - Verified by unit tests in `benchmarks/validate_phase5.py` (Coordinate remapping: PASS, NMS suppression: PASS).
    - Achieved **+13.3% detection recovery** on driving sequences while maintaining fresh-frame latency under $19\text{ ms}$.

---

## 5. Empirical Benchmark Results

### 5.1 Ground-Truth Baseline Performance (BDD100K Val, $N=425$)

Evaluated plain YOLOv8n across $N=425$ driving validation scenes (confidence threshold = 0.25, IoU = 0.50):

| Metric | Overall | Small ($<32^2$) | Medium ($32^2 \le A < 96^2$) | Large ($\ge 96^2$) |
|---|:---:|:---:|:---:|:---:|
| **Ground-Truth Objects** | 6,815 | 3,612 | 2,217 | 986 |
| **True Positives (TP)** | 1,995 | 211 | 990 | 794 |
| **False Positives (FP)** | 816 | 123 | 381 | 193 |
| **False Negatives (FN)** | 4,820 | 3,401 | 1,227 | 192 |
| **Precision** | **70.97%** | 63.17% | 72.21% | 80.45% |
| **Recall** | **29.27%** | **5.84%** | **44.65%** | **80.53%** |
| **mAP@0.5** | **21.57%** | — | — | — |
| **Inference Time** | 17.51 ms | — | — | — |

#### Class Breakdown (7-Class Vocabulary):
| Category | GT Count | Detections | TP | FP | FN | Precision | Recall | AP@0.5 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Car** | 4,568 | 2,170 | 1,645 | 525 | 2,923 | 75.81% | 36.01% | 34.58% |
| **Person / Pedestrian** | 616 | 242 | 179 | 63 | 437 | 73.97% | 29.06% | 27.46% |
| **Truck** | 195 | 139 | 57 | 82 | 138 | 41.01% | 29.23% | 22.68% |
| **Bus** | 75 | 49 | 27 | 22 | 48 | 55.10% | 36.00% | 30.45% |
| **Traffic Light** | 1,286 | 202 | 80 | 122 | 1,206 | 39.60% | 6.22% | 4.73% |
| **Bicycle** | 75 | 9 | 7 | 2 | 68 | 77.78% | 9.33% | 9.53% |
| *Rider (sub-category)* | 35 | — | 11 | — | 24 | — | 31.43% | — |

---

### 5.2 Phase-by-Phase Video Stream Benchmark Comparison

Performance across BDD100K evaluation sequences (`50.mp4`, `544.mp4`, `1600.mp4`):

| Pipeline Phase | Primary Det (ms) | Re-Det (ms) | Fusion (ms) | Total Det Time (ms) | Frame Age (ms) | Avg Detections / Frame | Buffer Accounting |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 3 (Primary YOLO)** | 18.17 ms | — | — | 18.17 ms | 0.79 ms | 8.27 | 100% Verified |
| **Phase 4 (+ Illumination)** | 18.15 ms | — | — | 20.35 ms | 0.75 ms | 8.28 | 100% Verified |
| **Phase 5 (+ Selective Re-Det)** | **17.85 ms** | **54.12 ms** | **0.047 ms** | **72.13 ms** | **18.48 ms** | **9.37 (+13.3%)** | **100% Verified** |

#### Phase 5 Per-Video Detailed Telemetry:
| Video | Frames Cap. | Frames Proc. | Replaced | Drop Rate | Avg ROIs | Primary (ms) | Re-Det (ms) | Total (ms) | Frame Age (ms) | Detections |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **50.mp4** | 510 | 241 | 269 | 52.75% | 3.44 | 17.85 ms | 60.16 ms | 78.20 ms | 18.79 ms | 10.35 |
| **544.mp4** | 300 | 176 | 124 | 41.33% | 1.90 | 17.91 ms | 44.57 ms | 62.63 ms | 19.14 ms | 8.71 |
| **1600.mp4** | 480 | 233 | 247 | 51.46% | 2.64 | 17.78 ms | 57.64 ms | 75.57 ms | 17.51 ms | 9.04 |

---

## 6. Project Structure

```text
adaptive-vision/
├── .gitignore                          # Strict gitignore (weights, media, baseline images, logs)
├── README.md                           # Comprehensive documentation & benchmark reports
├── config.py                           # Centralized pipeline configuration & hyperparameters
├── main.py                             # Main live pipeline orchestration & telemetry entry point
├── requirements.txt                    # Project runtime dependencies
│
├── benchmarks/
│   ├── detection_baseline/             # Ground-truth evaluation outputs (CSV reports)
│   │   ├── baseline_overall.csv        # Overall GT precision, recall, mAP@0.5
│   │   ├── baseline_per_class.csv      # Per-class detection and AP metrics
│   │   ├── baseline_per_size.csv       # Small, medium, large recall breakdowns
│   │   ├── baseline_pedestrian_rider.csv
│   │   └── images/                     # 425 local BDD100K validation images (git-ignored)
│   ├── detection_baseline.py           # Ground-truth baseline evaluation script (N=425)
│   ├── phase3/                         # Phase 3 baseline benchmark telemetry CSVs
│   ├── phase4/                         # Phase 4 environmental adaptation telemetry CSVs
│   ├── phase5/                         # Phase 5 selective re-detection telemetry CSVs
│   ├── validate_phase4.py              # Phase 4 validation suite
│   ├── validate_phase4_delta.py        # Isolated illumination timing delta analysis
│   └── validate_phase5.py              # Phase 5 validation suite & unit tests
│
├── data/
│   ├── bdd100k/
│   │   └── videos/                     # 1,435 local BDD100K driving videos (git-ignored)
│   └── outputs/                        # Output visualizations and annotated video runs
│
├── logs/
│   └── .gitkeep                        # Execution logs
│
├── src/
│   ├── __init__.py                     # Package export declarations
│   ├── capture.py                      # Multi-threaded continuous frame capture worker
│   ├── detector.py                     # Primary YOLOv8n detector initialization and inference
│   ├── download_dataset.py             # Dataset acquisition helper
│   ├── frame_manager.py                # Thread-safe FreshFrameBuffer (capacity = 1)
│   ├── illumination.py                 # Environmental illumination estimation & CLAHE
│   ├── metrics.py                      # Frame age, latency, and throughput telemetry
│   └── selective_redetection.py        # Uncertainty filtering, ROI expansion, re-det & NMS
│
└── weights/
    └── yolov8n.pt                      # YOLOv8n model weights (git-ignored)
```

---

## 7. Setup & Execution

### 7.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/Rhythem2005/MINOR-PROJECT.git
cd MINOR-PROJECT

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Running the Adaptive Pipeline
To execute the live pipeline (Capture $\to$ Illumination $\to$ Primary YOLO $\to$ Uncertainty Filtering $\to$ Context Re-Detection $\to$ NMS Fusion $\to$ Telemetry):
```bash
python main.py
```

### 7.3 Running the Validation Suites
```bash
# Validate Phase 5 Selective Re-Detection (Unit tests + 5 Video benchmark)
python benchmarks/validate_phase5.py

# Validate Phase 4 Environmental Adaptation
python benchmarks/validate_phase4.py

# Run Ground-Truth Baseline Benchmark (N=425 BDD100K validation scenes)
python benchmarks/detection_baseline.py
```

---

## 8. Hyperparameter Configuration

Key parameters defined in [`config.py`](file:///Users/rhythemsabharwalgmail.com/Desktop/MINOR-PROJECT/config.py):

| Parameter | Default Value | Description |
|---|:---:|---|
| `FRAME_BUFFER_CAPACITY` | `1` | Capacity of the latest-frame buffer (prevents queue buildup) |
| `CONF_THRESHOLD` | `0.25` | Primary detector minimum confidence threshold |
| `IOU_THRESHOLD` | `0.45` | Primary detector NMS IoU threshold |
| `LOW_LIGHT_THRESH` | `45.0` | Perceived luma threshold triggering CLAHE enhancement |
| `HIGH_LIGHT_THRESH` | `210.0` | Perceived luma threshold triggering gamma correction |
| `RE_DETECT_CONF_LOW` | `0.25` | Minimum confidence for selective re-detection candidacy |
| `RE_DETECT_CONF_HIGH` | `0.65` | Upper confidence bound defining uncertain predictions |
| `RE_DETECT_CLASS_MARGIN`| `0.05` | Top-2 classification probability margin threshold |
| `RE_DETECT_CONTEXT_FACTOR`| `1.0` | Surrounding context expansion factor ($\beta$) around ROI |
| `RE_DETECT_MIN_SIZE` | `64` | Minimum ROI crop width/height (pixels) to avoid feature collapse |
| `RE_DETECT_MAX_ROIS` | `4` | Maximum ROI re-detection invocations per frame |
| `FUSION_IOU_THRESH` | `0.50` | Class-aware NMS IoU threshold for fusing re-detections |

---

## 9. Next Steps: Phase 6 (Temporal Tracking)

The final core algorithmic phase will integrate inter-frame **Temporal Tracking**:
- **Trajectory Association:** Maintain persistent object tracks across consecutive frames using lightweight motion estimation (e.g., Kalman filter / SORT / ByteTrack principles).
- **Reduced Re-Detection Duty Cycle:** Suppress redundant re-detection calls on established high-confidence tracks, reserving selective inference strictly for newly entering or actively deteriorating tracks.
- **Occlusion Handling:** Preserve object identities and bounding box estimates through momentary occlusions or missed detections.
- **End-to-End Evaluation:** Comprehensive ablation study comparing dense baseline detection vs. the complete Adaptive Vision pipeline across accuracy, latency, and power efficiency.

---

## 10. Repository Hygiene & Git Policies

To ensure a lightweight and clean version history:
- **No Large Media:** Video sequences under `data/` and `data/dev_videos/` are strictly git-ignored.
- **No Baseline Evaluation Images:** The 425 local validation scenes under `benchmarks/detection_baseline/images/` are git-ignored.
- **No Model Weights:** Model checkpoints in `weights/` (`*.pt`, `*.onnx`, etc.) remain local.
- **No Temporary Run Artifacts:** Generated debug frames and caches are excluded.
