"""Read-only adapter for the downloaded Argoverse-HD COCO-style export.

No labels/images are generated. Categories and sequence identity come from
the export; source filenames supply nanosecond timestamps (Argoverse API).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..adaptive_framing.sequence import Sequence
from .provenance import sha256_file

# Stock YOLO COCO indices, not COCO's sparse annotation category IDs.
COCO_NAMES = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
              "bus": 5, "truck": 7, "traffic_light": 9, "stop_sign": 11}


def _unique(rows, key, kind):
    result = {}
    for row in rows:
        value = row[key]
        if type(value) is not int or value < 0 or value in result:
            raise ValueError(f"Invalid/duplicate {kind} {key}: {value}")
        result[value] = row
    return result


def _path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_argoverse(root, verify_images=True):
    """Validate the whole export and return sequences, names, class_map, audit.

    Invalid boxes, unknown classes, missing/orphan/duplicate records, sequence
    gaps, non-monotonic timestamps and corrupt/dimension-mismatched images
    fail before model loading. Finite positive boxes crossing image boundaries
    are retained verbatim and explicitly reported, as in COCO pixel boxes.
    """
    root = Path(root).resolve()
    annotation_path, manifest_path = root / "annotations.json", root / "manifest.json"
    data = json.loads(annotation_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    categories = _unique(data["categories"], "id", "category")
    category_ids = sorted(categories)
    names = [categories[c]["name"] for c in category_ids]
    if len(names) != len(set(names)) or set(names) != set(COCO_NAMES):
        raise ValueError(f"Unsupported/duplicate category taxonomy: {names}")
    dense_id = {c: i for i, c in enumerate(category_ids)}
    class_map = {COCO_NAMES[categories[c]["name"]]: dense_id[c] for c in category_ids}
    if "coco_subset" in data and data["coco_subset"] != [COCO_NAMES[n] for n in names]:
        raise ValueError("coco_subset disagrees with named categories")
    if "coco_mapping" in data:
        if len(data["coco_mapping"]) != 80 or any(data["coco_mapping"][c] != category_ids[v]
                                                for c, v in class_map.items()):
            raise ValueError("coco_mapping disagrees with named categories")
    images = _unique(data["images"], "id", "image")
    entries = _unique(manifest, "image_id", "manifest image")
    if not images or set(images) != set(entries):
        raise ValueError("Annotation images and manifest must contain identical nonempty image IDs")
    annotations = _unique(data["annotations"], "id", "annotation")
    targets, ignored = defaultdict(list), defaultdict(list)
    counts, crowd_counts, ignore_counts = Counter(), Counter(), Counter()
    outside = []
    for ann in annotations.values():
        iid, cid = ann["image_id"], ann["category_id"]
        if type(iid) is not int or iid not in images or type(cid) is not int or cid not in categories:
            raise ValueError(f"Orphan image or unknown category in annotation {ann['id']}")
        bbox = np.asarray(ann["bbox"], dtype=float)
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox[2:] <= 0).any():
            raise ValueError(f"Invalid bbox in annotation {ann['id']}: {bbox}")
        if not np.isfinite(ann["area"]) or not np.isclose(ann["area"], bbox[2] * bbox[3]):
            raise ValueError(f"Inconsistent bbox area in annotation {ann['id']}")
        for flag in ("ignore", "iscrowd"):
            if ann.get(flag, False) not in (False, True, 0, 1):
                raise ValueError(f"Invalid {flag} in annotation {ann['id']}")
        x, y, w, h = bbox
        im = images[iid]
        if x < 0 or y < 0 or x + w > im["width"] or y + h > im["height"]:
            outside.append({"annotation_id": ann["id"], "image_id": iid, "bbox": bbox.tolist()})
        row = [dense_id[cid], x, y, x + w, y + h]
        name = categories[cid]["name"]
        counts[name] += 1
        if ann.get("iscrowd", False) or ann.get("ignore", False):
            ignored[iid].append(row + [int(bool(ann.get("iscrowd", False)))])
            if ann.get("iscrowd", False):
                crowd_counts[name] += 1
            else:
                ignore_counts[name] += 1
        else:
            targets[iid].append(row)

    groups = defaultdict(list)
    paths, sizes = set(), Counter()
    for iid, row in entries.items():
        im = images[iid]
        if row["export_path"] != im["file_name"]:
            raise ValueError(f"Manifest path disagreement for image {iid}")
        frame_path = _path(root, im["file_name"])
        if frame_path in paths:
            raise ValueError(f"Duplicate image path: {frame_path}")
        paths.add(frame_path)
        width, height = im["width"], im["height"]
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError(f"Invalid dimensions for image {iid}")
        if verify_images:
            frame = cv2.imread(str(frame_path))
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(f"Corrupt image/dimension disagreement: {frame_path}")
        sid = im["sid"]
        if type(sid) is not int or not 0 <= sid < len(data["seq_dirs"]) or \
                row["sequence"] != data["seq_dirs"][sid] or \
                data["sequences"][sid] != Path(row["sequence"]).parent.name:
            raise ValueError(f"Sequence identity disagreement for image {iid}")
        if type(row["frame_index"]) is not int or row["frame_index"] != im["fid"]:
            raise ValueError(f"Frame index disagreement for image {iid}")
        source_name = Path(row["source_path"]).name
        if source_name != im["name"] or source_name != im.get("original_name", source_name):
            raise ValueError(f"Source name disagreement for image {iid}")
        try:
            timestamp = int(Path(source_name).stem.rsplit("_", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Missing timestamp: {source_name}") from exc
        groups[sid].append((row["frame_index"], iid, str(frame_path), timestamp))
        sizes[f"{width}x{height}"] += 1
    actual_paths = {p.resolve() for p in (root / "images").rglob("*")
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    if actual_paths != paths:
        raise ValueError("Image files and annotation/manifest paths disagree")
    sequences, stats = [], []
    for sid, rows in sorted(groups.items()):
        rows.sort()
        if [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Noncontiguous/duplicate frame indices in sequence {sid}")
        dt = np.diff([r[3] for r in rows])
        if len(dt) and (dt <= 0).any():
            raise ValueError(f"Non-monotonic timestamps in sequence {sid}")
        # Reject temporal gaps before the existing frame-step tracker is used.
        if len(dt) and dt.max() > 1.5 * np.median(dt):
            raise ValueError(f"Timestamp gap in sequence {sid}")
        ids = [r[1] for r in rows]
        seq = Sequence(f"sequence_{sid:02d}", [r[2] for r in rows],
                       targets=[np.array(targets[i], np.float32).reshape(-1, 5) for i in ids],
                       ignored=[np.array(ignored[i], np.float32).reshape(-1, 6) for i in ids], image_ids=ids)
        sequences.append(seq)
        stats.append({"name": seq.name, "source_sequence": data["seq_dirs"][sid], "sid": sid,
                      "frames": len(ids), "regular_instances": sum(len(targets[i]) for i in ids),
                      "ignored_instances": sum(len(ignored[i]) for i in ids),
                      "empty_annotation_frames": sum(not targets[i] and not ignored[i] for i in ids),
                      "no_regular_target_frames": sum(not targets[i] for i in ids),
                      "timestamp_first_ns": rows[0][3], "timestamp_last_ns": rows[-1][3],
                      "delta_ns_min": int(dt.min()) if len(dt) else None,
                      "delta_ns_median": float(np.median(dt)) if len(dt) else None,
                      "delta_ns_max": int(dt.max()) if len(dt) else None,
                      "fps_from_timestamps": float(1e9 / np.median(dt)) if len(dt) else None})
    report = {"root": str(root), "images": len(images), "annotations": len(annotations),
              "regular_instances": sum(len(x) for x in targets.values()),
              "crowd_instances": sum(crowd_counts.values()), "explicit_ignore_instances": sum(ignore_counts.values()),
              "categories": data["categories"], "class_names": names, "coco_to_evaluation": class_map,
              "annotation_counts_by_class": dict(counts), "crowd_counts_by_class": dict(crowd_counts),
              "unsupported_classes": [], "image_sizes": dict(sizes), "images_decoded_for_validation": verify_images,
              "outside_image_boxes_retained": outside, "sequence_metadata_entries": len(data["sequences"]),
              "sequences": stats, "annotations_sha256": sha256_file(annotation_path),
              "manifest_sha256": sha256_file(manifest_path),
              "box_policy": "provided pixel xywh converted to xyxy; no clipping or rounding",
              "ignore_policy": "regular IoU-greedy matching first; unmatched class-compatible predictions ignored per IoU threshold using crowd IoA (many-to-one) or explicit-ignore IoU (one-to-one)",
              "metric_protocol": "repository DetectionEvaluator current AP; not official COCO/streaming AP"}
    return sequences, names, class_map, report
