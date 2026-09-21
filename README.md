# Adaptive Vision

### Adaptive Real-Time Object Detection for Driving Videos Using Selective Re-Detection and Temporal Tracking

**GitHub Repository:** `adaptive-vision`  
**Project Category:** B.Tech Minor Project / Applied Computer Vision Research  

---

## 1. Overview

**Adaptive Vision** is an academic research project focused on designing an adaptive, efficient real-time object detection architecture specifically engineered for automotive driving video streams. 

Autonomous driving systems and advanced driver-assistance systems (ADAS) must balance strict real-time processing constraints (high FPS, low latency) against high detection accuracy under dynamic environmental conditions. Standard object detection architectures process every sequential video frame through heavy deep neural networks, resulting in computational bottlenecks and redundant inferences.

Adaptive Vision investigates a hybrid approach that integrates:
- Real-time video ingestion with controlled frame synchronization.
- Lightweight primary object detection coupled with confidence and uncertainty evaluation.
- Selective re-detection triggers to invoke higher-capacity inference only when necessary.
- Inter-frame temporal tracking to preserve object identities with low computational overhead.
- Fine-grained telemetry to monitor latency, frame staleness, and processing throughput.

---

## 2. Research Objective

1. **Mitigate Frame-by-Frame Redundancy:** Eliminate unnecessary continuous deep neural network inference on visually redundant sequential frames.
2. **Adaptive Inference Allocation:** Dynamically trigger selective re-detection on ambiguous, low-confidence, or rapidly changing regions of interest.
3. **Temporal Consistency:** Maintain stable object trajectories using temporal tracking between detection cycles.
4. **Deterministic Latency & Telemetry:** Minimize end-to-end pipeline latency and frame age while maintaining frame throughput suitable for real-time driving scenarios.
5. **Empirical Benchmarking:** Evaluate trade-offs between processing throughput (FPS), latency (ms), and detection fidelity across real-world driving datasets.

---

## 3. Pipeline Architecture

The overall research pipeline design is structured into sequential stages:

```text
VIDEO STREAM
     ↓
ADAPTIVE FRAME CONTROLLER
     ↓
ENVIRONMENTAL ADAPTATION
     ↓
PRIMARY LIGHTWEIGHT YOLO
     ↓
CONFIDENCE / UNCERTAINTY FILTER
     ↓
SELECTIVE RE-DETECTION
     ↓
FUSION / NMS
     ↓
TEMPORAL TRACKING
     ↓
FPS / LATENCY / TELEMETRY
```

> **Note on Implementation Status:** This diagram represents the end-to-end research architecture. Only Phase 1 (Dataset Acquisition, Organization & Verification) is currently completed. Core algorithmic modules are structured as functional skeletons awaiting progressive implementation.

---

## 4. Current Progress

### Phase 1: Dataset Acquisition & Verification (Completed)
- [x] **Dataset Acquisition:** Acquired driving camera video data from the BDD100K / BDDA dataset.
- [x] **Local Data Organization:** Stored **1,435 raw video files** locally under `data/bdd100k/videos/`.
- [x] **Video Stream & Codec Verification:** Inspected and validated video stream specifications using media stream probes:
  - **Video Codec:** H.264 / AVC
  - **Resolution:** 1280 × 720 (720p HD)
  - **Frame Rate:** 29.97 FPS
  - **Duration:** 10.01 seconds per clip
  - **Frame Count:** 300 frames per video sequence
- [x] **Repository Hygiene & Git Governance:** Verified that all video files, virtual environments, temporary logs, and model artifacts are strictly excluded via `.gitignore`.
- [x] **Project Scaffolding:** Established modular package layout with initial interfaces for capture, inference, frame management, telemetry, and benchmarking.

### Upcoming Phases (In Progress / Planned)
- [ ] **Phase 2:** Continuous video ingestion & fresh-frame buffer controller (`src/capture.py`, `src/frame_manager.py`).
- [ ] **Phase 3:** Primary lightweight detector integration and baseline inference profiling (`src/detector.py`).
- [ ] **Phase 4:** Confidence / uncertainty filtering and selective re-detection triggering.
- [ ] **Phase 5:** Temporal tracking association and bounding box fusion.
- [ ] **Phase 6:** End-to-end benchmarking and latency/accuracy ablation studies (`benchmarks/run_benchmark.py`).

---

## 5. Dataset

The project utilizes camera video sequences from the **BDD100K / BDDA** (Berkeley DeepDrive) dataset, reflecting diverse real-world driving environments across varied weather conditions, times of day, and roadway types.

- **Local Storage:** 1,435 camera video files stored under `data/bdd100k/videos/`.
- **Git Handling Notice:** Due to dataset size, the raw video clips are **strictly local** and are **not committed to the GitHub repository**. All media formats (`*.mp4`, `*.avi`, `*.mov`, etc.) are tracked and excluded by `.gitignore`.

---

## 6. Dataset Setup

To set up the dataset locally:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Rhythem2005/adaptive-vision.git
   cd adaptive-vision
   ```

2. **Set Up Python Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Acquire Video Data:**
   The repository provides a download helper script using `kagglehub`:
   ```bash
   python src/download_dataset.py
   ```
   Organize or symlink the downloaded video clips into the designated project directory:
   ```bash
   data/bdd100k/videos/
   ```

4. **Verify Video Format:**
   Ensure video clips match the expected input standard (e.g., 720p H.264 @ ~30 FPS).

---

## 7. Project Structure

```text
adaptive-vision/
├── .gitignore               # Strict exclusion rules for datasets, weights, venvs, and logs
├── README.md                # Project documentation and research overview
├── config.py                # Centralized pipeline configuration constants (skeleton)
├── main.py                  # Main pipeline orchestration entry point (skeleton)
├── requirements.txt         # Project runtime dependencies
│
├── benchmarks/
│   └── run_benchmark.py     # Evaluation & benchmark execution runner (skeleton)
│
├── data/
│   ├── bdd100k/
│   │   └── videos/          # Local-only: 1,435 BDD100K driving videos (git-ignored)
│   ├── dev_videos/          # Local-only: Short development video samples (git-ignored)
│   └── outputs/             # Local-only: Generated output videos & predictions (git-ignored)
│
├── logs/
│   └── .gitkeep             # Preserves directory structure for runtime execution logs
│
├── src/
│   ├── __init__.py
│   ├── capture.py           # Frame stream ingestion from video/camera sources (skeleton)
│   ├── detector.py          # YOLO model loading and inference routines (skeleton)
│   ├── download_dataset.py  # Dataset download utility using kagglehub
│   ├── frame_manager.py     # Latest-frame buffer synchronization (capacity = 1) (skeleton)
│   └── metrics.py           # Latency, FPS, frame age, and throughput telemetry (skeleton)
│
└── weights/                 # Local directory for deep learning model weights (git-ignored)
```

---

## 8. Technology Stack & Dependencies

The project relies on standard computer vision and deep learning libraries:

| Component | Library / Tool | Role in Project |
|---|---|---|
| **Language** | Python 3 | Primary development language |
| **Object Detection** | `ultralytics` | YOLO model loading and inference |
| **Computer Vision** | `opencv-python` | Video decoding, frame capture, and image transforms |
| **Numerical Processing** | `numpy` | Array manipulations and matrix operations |
| **Data Analysis** | `pandas` | Telemetry logging, metric aggregation, and tabular reports |
| **Visualization** | `matplotlib` | Performance graphing and benchmark visualization |
| **Dataset Ingestion** | `kagglehub` | Dataset download facilitation |

---

## 9. Research Pipeline (Detailed Concept)

1. **Video Stream Ingestion (`src/capture.py`):**  
   Continuously reads frames from driving video feeds or pre-recorded BDD100K video files.
2. **Adaptive Frame Controller (`src/frame_manager.py`):**  
   Maintains a bounded latest-frame buffer (capacity = 1) to eliminate frame queuing delays and ensure the model processes only the freshest available frame.
3. **Environmental Adaptation:**  
   Evaluates illumination and scene dynamics to adapt detection parameters.
4. **Primary Lightweight YOLO (`src/detector.py`):**  
   Executes high-speed primary inference to establish initial spatial candidate detections.
5. **Confidence / Uncertainty Filter:**  
   Analyzes prediction entropy, bounding box variances, and class confidences to identify ambiguous detections.
6. **Selective Re-Detection:**  
   Selectively re-evaluates low-confidence crops or complex scene regions without incurring the cost of full-image dense re-inference.
7. **Fusion / NMS:**  
   Merges primary detections, re-detection refinements, and spatial overlaps into unified detections.
8. **Temporal Tracking:**  
   Propagates track states across un-detected or intermediate frames using motion prediction and association algorithms.
9. **FPS / Latency / Telemetry (`src/metrics.py`):**  
   Measures end-to-end processing latency, detector compute time, frame age, dropped frames, and effective FPS.

---

## 10. Evaluation Plan

Once the algorithmic stages are implemented, the framework will be systematically benchmarked on the local BDD100K driving dataset using `benchmarks/run_benchmark.py`:

- **Latency Metrics:** End-to-end pipeline latency (ms), frame age from capture to output, and inter-stage processing latency.
- **Throughput Metrics:** Sustained frames per second (FPS) and dropped frame percentage under varied load.
- **Detection Accuracy:** Mean Average Precision ($\text{mAP}_{50}$, $\text{mAP}_{50-95}$) evaluated on primary driving object categories (vehicles, pedestrians, cyclists, traffic signals).
- **Tracking Quality:** Trajectory consistency, ID switches, and tracking fragmentation metrics.
- **Comparative Baseline:** Standard dense per-frame inference vs. the proposed selective re-detection and temporal tracking architecture.

---

## 11. Git & Dataset Handling

To maintain a clean and lightweight repository:
- **No Large Media Files:** Video files under `data/` are strictly excluded from version control via `.gitignore`.
- **No Model Weights:** Model weight checkpoints (`*.pt`, `*.pth`, `*.onnx`, etc.) stored in `weights/` must remain local.
- **No Transient Outputs:** Generated logs and annotated videos (`data/outputs/`, `logs/*.log`) are ignored.

---

## 12. Future Work

- Implement multi-threaded producer-consumer frame buffering in `src/capture.py` and `src/frame_manager.py`.
- Benchmark baseline lightweight YOLO models on sample BDD100K sequences.
- Formulate criteria and heuristics for confidence-based selective re-detection.
- Integrate lightweight temporal tracking algorithms (e.g., Kalman filter / SORT variants).
- Complete automated evaluation scripts and performance profiling in `benchmarks/run_benchmark.py`.

---

## 13. License

A software license has not yet been assigned to this repository. All rights are reserved by the project author pending academic review.
