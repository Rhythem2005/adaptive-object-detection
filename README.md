# MINOR-PROJECT

A real-time video ingestion and object detection pipeline utilizing YOLO inference, single-frame buffering, and pipeline latency/throughput benchmarking.

## Project Structure

```text
MINOR-PROJECT/
│
├── data/
│   ├── dev_videos/              # Small road videos for development/testing
│   ├── bdd100k/                 # Final BDD100K benchmark videos
│   └── outputs/                 # Annotated/output videos
│
├── weights/                     # YOLO model weights
│
├── src/
│   ├── __init__.py
│   ├── capture.py               # Continuous video/frame ingestion
│   ├── frame_manager.py         # Latest-frame buffer, capacity = 1
│   ├── detector.py              # YOLO inference
│   └── metrics.py               # Latency, FPS, frame age, dropped frames
│
├── benchmarks/
│   └── run_benchmark.py         # Benchmark runner
│
├── logs/
│   └── .gitkeep
│
├── config.py                    # Project configuration/settings
├── main.py                      # Main pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Module Overview

- **`src/capture.py`**: Handles video capture from cameras or pre-recorded video files.
- **`src/frame_manager.py`**: Manages a single-frame buffer (capacity = 1) to ensure inference always operates on the freshest frame.
- **`src/detector.py`**: Loads YOLO model weights and runs object detection inference.
- **`src/metrics.py`**: Computes pipeline metrics (end-to-end latency, FPS, frame age, and dropped frames).
- **`benchmarks/run_benchmark.py`**: Runs evaluations across benchmark video datasets.
- **`config.py`**: Centralized configuration parameters and settings.
- **`main.py`**: Main application entry point orchestrating ingestion, detection, and metrics tracking.

## Getting Started

1. **Set up virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline**:
   ```bash
   python main.py
   ```
