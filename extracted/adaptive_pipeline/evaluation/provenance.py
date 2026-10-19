"""Input/settings provenance for new runs; no model or experiment is launched."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from ..shared.config import DATASET_NAMES


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_fresh_run(out_dir):
    for name in ("metrics.json", "frames.csv", "eval_stats.npz", "input_manifest.json"):
        if (Path(out_dir) / name).exists():
            raise FileExistsError(f"Refusing to overwrite an existing run: {out_dir}. Choose a new output directory.")


def validate_label(path):
    """Fail before inference on malformed labels; an empty file is valid."""
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = np.array([float(value) for value in line.split()])
        except ValueError as error:
            raise ValueError(f"Malformed label {path}:{line_number}") from error
        if row.shape != (5,) or not np.isfinite(row).all():
            raise ValueError(f"Invalid YOLO label {path}:{line_number}")
        cls, cx, cy, width, height = row
        if cls != int(cls) or not 0 <= cls < len(DATASET_NAMES) or \
                not ((0 <= row[1:]).all() and (row[1:] <= 1).all()) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid class or box coordinates {path}:{line_number}")


def input_manifest(sequences, max_frames=None, hash_images=False):
    records = []
    for sequence in sequences:
        selected = sequence.frames[:max_frames] if max_frames else sequence.frames
        for i, frame in enumerate(selected):
            image = Path(frame)
            stat = image.stat()
            record = {"video": sequence.name, "frame_index": i, "image": str(image.resolve()),
                      "image_size_bytes": stat.st_size, "image_mtime_ns": stat.st_mtime_ns}
            if hash_images:
                record["image_sha256"] = sha256_file(image)
            if getattr(sequence, "targets", None) is not None:
                record["image_id"] = sequence.image_ids[i] if getattr(sequence, "image_ids", None) is not None else i
                record["targets_sha256"] = hashlib.sha256(
                    np.asarray(sequence.targets[i], dtype="<f4").tobytes()).hexdigest()
                record["ignored_sha256"] = hashlib.sha256(np.asarray(
                    sequence.ignored[i] if sequence.ignored is not None else [], dtype="<f4").tobytes()).hexdigest()
            if sequence.labels is not None:
                label = Path(sequence.labels[i])
                if not label.is_file():
                    raise FileNotFoundError(f"Missing evaluation label: {label}")
                validate_label(label)
                record["label"] = str(label.resolve())
                record["label_sha256"] = sha256_file(label)
            records.append(record)
    if not records:
        raise ValueError("No evaluation frames found")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"frames": records, "sha256": hashlib.sha256(encoded).hexdigest(),
            "image_hashing": "sha256" if hash_images else "path_size_mtime",
            "label_hashing": "sha256; native targets hashed as little-endian float32 when present"}


def runtime_metadata(cfg, detector):
    packages = {}
    for name in ("ultralytics", "torch", "opencv-python", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    weights = Path(cfg.model_path)
    current_hash = sha256_file(weights) if weights.is_file() else None
    loaded_hash = getattr(detector, "checkpoint_sha256", current_hash)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages,
            "weights_sha256": loaded_hash,
            "checkpoint_file_changed_since_load": current_hash != loaded_hash,
            "resolved_device": str(getattr(getattr(detector, "model", None), "device", "unavailable")),
            "timing_scope": "pipeline processing; excludes decoding, scoring, display and provenance I/O"}
