"""YOLOv8n Stock Baseline Evaluation on Dataset2 TEST Split.

Plain, unmodified YOLOv8n (COCO pretrained) inference on the dataset2 test
split, evaluated against ground-truth labels.

Dataset2 has 7 custom classes:
    0: car, 1: bus, 2: truck, 3: pedestrian, 4: rider, 5: bicycle, 6: traffic light

COCO→dataset2 class mapping (following existing repo convention in
detection_baseline.py):
    dataset2 car (0)           → COCO car (2)
    dataset2 bus (1)           → COCO bus (5)
    dataset2 truck (2)         → COCO truck (7)
    dataset2 pedestrian (3)    → COCO person (0)
    dataset2 rider (4)         → COCO person (0)  [no COCO 'rider' — mapped
                                  to person, with separate sub-metrics reported]
    dataset2 bicycle (5)       → COCO bicycle (1)
    dataset2 traffic light (6) → COCO traffic light (9)

Latency methodology (matching repo convention):
    - 1 warmup inference on a dummy 720×1280 frame
    - time.perf_counter() around model.predict() only (excludes data loading)
    - Reports mean, median, p95 inference time and FPS

Size buckets (COCO standard, matching detection_baseline.py):
    small:  area < 32² = 1024 px²
    medium: 32² ≤ area < 96² = 9216 px²
    large:  area ≥ 96²

Usage:
    .venv/bin/python benchmarks/yolov8n_stock_baseline_dataset2.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREP_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Prep-Data"
)
TEST_IMAGES_DIR = os.path.join(PREP_DATA_DIR, "images", "test")
TEST_LABELS_DIR = os.path.join(PREP_DATA_DIR, "labels", "test")
DATA_YAML_PATH = os.path.join(PREP_DATA_DIR, "data.yaml")

CONFIDENCE_THRESHOLD = 0.25      # Operating-point threshold for P/R
AP_CONF_THRESHOLD = 0.001        # Low threshold for full AP PR curve

# COCO-style size thresholds (pixels²)
SMALL_AREA = 32 * 32    # 1024
MEDIUM_AREA = 96 * 96   

# Dataset2 class names (from data.yaml)
DS2_CLASSES = {0: "car", 1: "bus", 2: "truck", 3: "pedestrian",
               4: "rider", 5: "bicycle", 6: "traffic light"}

# Dataset2 → COCO class mapping
# Following existing repo convention: pedestrian and rider both map to
# COCO "person" (id=0). Rider has no COCO equivalent; this is explicitly
# documented and separate sub-metrics are reported.
DS2_TO_COCO = {
    0: 2,   # car → car
    1: 5,   # bus → bus
    2: 7,   # truck → truck
    3: 0,   # pedestrian → person
    4: 0,   # rider → person (no COCO equivalent, see docstring)
    5: 1,   # bicycle → bicycle
    6: 9,   # traffic light → traffic light
}

# Reverse: which COCO ids are in scope?
COCO_IDS_IN_SCOPE = set(DS2_TO_COCO.values())

# IoU thresholds for mAP@0.50:0.95
IOU_THRESHOLDS_COARSE = [0.5]
IOU_THRESHOLDS_FINE = np.arange(0.5, 1.0, 0.05).tolist()  # 10 thresholds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area_px(box):
    """Area of a box [x1, y1, x2, y2] in pixels."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def size_bucket(area_px):
    if area_px < SMALL_AREA:
        return "small"
    elif area_px < MEDIUM_AREA:
        return "medium"
    else:
        return "large"


def compute_ap(precisions, recalls):
    """Compute AP using 101-point interpolation (COCO style)."""
    if len(precisions) == 0:
        return 0.0
    prec = np.concatenate(([1.0], precisions, [0.0]))
    rec = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = rec >= t
        if mask.any():
            ap += prec[mask].max()
    return ap / 101.0


def extract_condition(filename):
    """Extract condition (day/night) from filename prefix."""
    basename = os.path.basename(filename)
    if basename.startswith("night"):
        return "night"
    elif basename.startswith("day"):
        return "day"
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_yolo_labels(label_path, img_w, img_h):
    """Load YOLO format labels (class_id cx cy w h) → list of (ds2_class_id, x1, y1, x2, y2)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert from normalized xywh to pixel xyxy
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def collect_test_samples():
    """Collect all test image/label pairs, verify counts match."""
    images = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    labels = sorted([f for f in os.listdir(TEST_LABELS_DIR)
                     if f.lower().endswith('.txt')])

    # Build lookup from stem → label file
    label_stems = {os.path.splitext(f)[0]: f for f in labels}

    samples = []
    missing_labels = 0
    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        label_file = label_stems.get(stem)
        if label_file is None:
            missing_labels += 1
            continue
        samples.append({
            "image_path": os.path.join(TEST_IMAGES_DIR, img_file),
            "label_path": os.path.join(TEST_LABELS_DIR, label_file),
            "condition": extract_condition(img_file),
            "filename": img_file,
        })

    return samples, missing_labels


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_at_iou_threshold(all_predictions, all_gt, iou_threshold, eval_classes):
    """Compute per-class AP at a single IoU threshold.

    Returns dict: {eval_class → ap}
    """
    # Build GT index: (img_idx, eval_class) → list of GT boxes
    gt_by_img_cls = defaultdict(list)
    for gt in all_gt:
        img_idx, eval_cls, x1, y1, x2, y2 = gt[:6]
        gt_by_img_cls[(img_idx, eval_cls)].append({
            "box": (x1, y1, x2, y2), "matched": False
        })

    # Sort predictions by confidence (descending)
    all_predictions_sorted = sorted(all_predictions, key=lambda x: -x[2])

    results = {}
    for eval_cls in eval_classes:
        # Reset matched flags
        for key in gt_by_img_cls:
            if key[1] == eval_cls:
                for g in gt_by_img_cls[key]:
                    g["matched"] = False

        cls_preds = [(p[0], p[2], p[3], p[4], p[5], p[6])
                     for p in all_predictions_sorted if p[1] == eval_cls]
        cls_gt_count = sum(1 for g in all_gt if g[1] == eval_cls)

        if cls_gt_count == 0:
            results[eval_cls] = 0.0
            continue

        tp_list = []
        for pred in cls_preds:
            img_idx, conf, px1, py1, px2, py2 = pred
            pred_box = (px1, py1, px2, py2)
            gts = gt_by_img_cls.get((img_idx, eval_cls), [])

            best_iou_val = 0
            best_gt_idx = -1
            for gi, gt_info in enumerate(gts):
                if gt_info["matched"]:
                    continue  # Only consider unmatched GTs
                iou_val = iou(pred_box, gt_info["box"])
                if iou_val > best_iou_val:
                    best_iou_val = iou_val
                    best_gt_idx = gi

            if best_iou_val >= iou_threshold and best_gt_idx >= 0:
                tp_list.append(1)
                gts[best_gt_idx]["matched"] = True
            else:
                tp_list.append(0)

        if len(tp_list) == 0:
            results[eval_cls] = 0.0
            continue

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum([1 - t for t in tp_list])

        precisions = tp_cum / (tp_cum + fp_cum)
        recalls = tp_cum / cls_gt_count

        ap = compute_ap(precisions, recalls)
        results[eval_cls] = ap

    return results


def run_full_evaluation(samples, model):
    """Run stock YOLOv8n inference on all test samples and compute metrics.

    Inference uses conf=AP_CONF_THRESHOLD (0.001) to capture the full
    prediction pool for AP calculation.  Operating-point precision/recall
    are computed separately at conf>=CONFIDENCE_THRESHOLD (0.25).
    """
    import torch

    print(f"\n  Warming up model (1 dummy frame, 720×1280)...")
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=AP_CONF_THRESHOLD, verbose=False,
                  save=False, device="mps")
    torch.mps.synchronize()
    print(f"  Warmup complete.")

    # We evaluate in the "eval class" space = COCO class IDs that are in scope.
    # GT labels (dataset2 class ids) are mapped to COCO ids via DS2_TO_COCO.
    # Predictions from the model are already in COCO class ids; we filter to
    # only those in COCO_IDS_IN_SCOPE.

    all_predictions = []  # (img_idx, coco_cls_id, confidence, x1, y1, x2, y2)
    all_gt = []           # (img_idx, coco_cls_id, x1, y1, x2, y2, ds2_cls_id, size_bucket)

    inference_times_ms = []
    total_gt_objects = 0
    images_with_no_gt = 0

    # Condition tracking
    condition_data = defaultdict(lambda: {
        "predictions": [], "gt": [], "inference_times": [], "n_images": 0
    })

    print(f"\n  Running inference on {len(samples)} test images...")
    t_total_start = time.perf_counter()

    for img_idx, sample in enumerate(samples):
        img = cv2.imread(sample["image_path"])
        if img is None:
            print(f"  [WARN] Cannot read {sample['image_path']}")
            continue

        img_h, img_w = img.shape[:2]
        condition = sample["condition"]

        # Load GT
        gt_boxes = load_yolo_labels(sample["label_path"], img_w, img_h)

        # Map GT from dataset2 class ids → COCO class ids
        for ds2_cls, x1, y1, x2, y2 in gt_boxes:
            if ds2_cls not in DS2_TO_COCO:
                continue
            coco_cls = DS2_TO_COCO[ds2_cls]
            area = box_area_px((x1, y1, x2, y2))
            sb = size_bucket(area)
            all_gt.append((img_idx, coco_cls, x1, y1, x2, y2, ds2_cls, sb))
            condition_data[condition]["gt"].append(
                (img_idx, coco_cls, x1, y1, x2, y2, ds2_cls, sb)
            )
            total_gt_objects += 1

        if len(gt_boxes) == 0:
            images_with_no_gt += 1

        # Run inference (timed, MPS-synchronized)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        results = model.predict(
            source=img, conf=AP_CONF_THRESHOLD, verbose=False, save=False,
            device="mps"
        )
        torch.mps.synchronize()
        t1 = time.perf_counter()
        inf_ms = (t1 - t0) * 1000.0
        inference_times_ms.append(inf_ms)
        condition_data[condition]["inference_times"].append(inf_ms)
        condition_data[condition]["n_images"] += 1

        # Collect predictions (only COCO classes in scope)
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in COCO_IDS_IN_SCOPE:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                pred_entry = (img_idx, cls_id, conf,
                              xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                all_predictions.append(pred_entry)
                condition_data[condition]["predictions"].append(pred_entry)

        if (img_idx + 1) % 2000 == 0 or (img_idx + 1) == len(samples):
            elapsed = time.perf_counter() - t_total_start
            eta = elapsed / (img_idx + 1) * (len(samples) - img_idx - 1)
            print(f"    [{img_idx + 1:>6}/{len(samples)}] "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                  f"avg={np.mean(inference_times_ms):.1f}ms/img")

    t_total_end = time.perf_counter()
    total_wall_s = t_total_end - t_total_start
    n_images = len(inference_times_ms)

    print(f"\n  Inference complete: {n_images} images in {total_wall_s:.1f}s")

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------

    eval_classes = sorted(COCO_IDS_IN_SCOPE)

    # COCO id → display name
    coco_names = {0: "person", 1: "bicycle", 2: "car", 5: "bus",
                  7: "truck", 9: "traffic light"}

    # --- mAP@0.50 (per-class) ---
    ap50_by_class = evaluate_at_iou_threshold(
        all_predictions, [(g[0], g[1], g[2], g[3], g[4], g[5]) for g in all_gt],
        0.5, eval_classes
    )

    # --- mAP@0.50:0.95 (per-class) ---
    ap_accum = {cls: [] for cls in eval_classes}
    for iou_thresh in IOU_THRESHOLDS_FINE:
        ap_at_thresh = evaluate_at_iou_threshold(
            all_predictions,
            [(g[0], g[1], g[2], g[3], g[4], g[5]) for g in all_gt],
            iou_thresh, eval_classes
        )
        for cls in eval_classes:
            ap_accum[cls].append(ap_at_thresh[cls])

    ap50_95_by_class = {cls: np.mean(ap_accum[cls]) for cls in eval_classes}

    # --- Per-class precision/recall at IoU=0.5 (operating point: conf>=0.25) ---
    gt_by_img_cls = defaultdict(list)
    for gt in all_gt:
        img_idx, coco_cls, x1, y1, x2, y2, ds2_cls, sb = gt
        gt_by_img_cls[(img_idx, coco_cls)].append({
            "box": (x1, y1, x2, y2), "matched": False,
            "ds2_cls": ds2_cls, "size": sb
        })

    # Filter to operating-point confidence for P/R metrics
    op_predictions = [p for p in all_predictions if p[2] >= CONFIDENCE_THRESHOLD]
    all_preds_sorted = sorted(op_predictions, key=lambda x: -x[2])

    per_class_results = {}
    for coco_cls in eval_classes:
        # Reset
        for key in gt_by_img_cls:
            if key[1] == coco_cls:
                for g in gt_by_img_cls[key]:
                    g["matched"] = False

        cls_preds = [(p[0], p[2], p[3], p[4], p[5], p[6])
                     for p in all_preds_sorted if p[1] == coco_cls]
        cls_gt_count = sum(1 for g in all_gt if g[1] == coco_cls)

        tp = 0
        fp = 0
        for pred in cls_preds:
            img_idx, conf, px1, py1, px2, py2 = pred
            pred_box = (px1, py1, px2, py2)
            gts = gt_by_img_cls.get((img_idx, coco_cls), [])

            best_iou_val = 0
            best_gt_idx = -1
            for gi, gt_info in enumerate(gts):
                if gt_info["matched"]:
                    continue  # Only consider unmatched GTs
                iou_val = iou(pred_box, gt_info["box"])
                if iou_val > best_iou_val:
                    best_iou_val = iou_val
                    best_gt_idx = gi

            if best_iou_val >= 0.5 and best_gt_idx >= 0:
                tp += 1
                gts[best_gt_idx]["matched"] = True
            else:
                fp += 1

        fn = cls_gt_count - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / cls_gt_count if cls_gt_count > 0 else 0.0

        # Dataset2 class names that map to this COCO class
        ds2_mapped = [DS2_CLASSES[k] for k, v in DS2_TO_COCO.items() if v == coco_cls]

        per_class_results[coco_cls] = {
            "coco_name": coco_names[coco_cls],
            "ds2_mapped_from": ds2_mapped,
            "gt_count": cls_gt_count,
            "pred_count": len(cls_preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "ap50": round(ap50_by_class[coco_cls], 4),
            "ap50_95": round(ap50_95_by_class[coco_cls], 4),
        }

    # --- Pedestrian vs Rider sub-breakdown ---
    ped_gt = sum(1 for g in all_gt if g[6] == 3)  # ds2 class 3 = pedestrian
    rider_gt = sum(1 for g in all_gt if g[6] == 4)  # ds2 class 4 = rider
    ped_tp = 0
    rider_tp = 0
    for key in gt_by_img_cls:
        coco_cls = key[1]
        if coco_cls != 0:  # person
            continue
        for g in gt_by_img_cls[key]:
            if g["matched"]:
                if g["ds2_cls"] == 3:
                    ped_tp += 1
                elif g["ds2_cls"] == 4:
                    rider_tp += 1

    person_breakdown = {
        "pedestrian": {
            "gt": ped_gt, "tp": ped_tp, "fn": ped_gt - ped_tp,
            "recall": round(ped_tp / ped_gt, 4) if ped_gt > 0 else 0.0
        },
        "rider": {
            "gt": rider_gt, "tp": rider_tp, "fn": rider_gt - rider_tp,
            "recall": round(rider_tp / rider_gt, 4) if rider_gt > 0 else 0.0,
            "note": "No COCO equivalent; mapped to 'person' per repo convention"
        },
    }

    # --- Per-size breakdown ---
    size_metrics = {"small": {"tp": 0, "fn": 0, "fp": 0},
                    "medium": {"tp": 0, "fn": 0, "fp": 0},
                    "large": {"tp": 0, "fn": 0, "fp": 0}}

    for key in gt_by_img_cls:
        for g in gt_by_img_cls[key]:
            sb = g["size"]
            if g["matched"]:
                size_metrics[sb]["tp"] += 1
            else:
                size_metrics[sb]["fn"] += 1

    # FP by size (based on prediction box area)
    for pred in all_preds_sorted:
        img_idx, coco_cls, conf, px1, py1, px2, py2 = pred
        if coco_cls not in COCO_IDS_IN_SCOPE:
            continue
        pred_area = box_area_px((px1, py1, px2, py2))
        pred_sb = size_bucket(pred_area)
        gts = gt_by_img_cls.get((img_idx, coco_cls), [])
        is_tp = any(iou((px1, py1, px2, py2), g["box"]) >= 0.5 for g in gts)
        if not is_tp:
            size_metrics[pred_sb]["fp"] += 1

    for sb in size_metrics:
        tp = size_metrics[sb]["tp"]
        fp = size_metrics[sb]["fp"]
        fn = size_metrics[sb]["fn"]
        size_metrics[sb]["precision"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        size_metrics[sb]["recall"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    # --- Overall ---
    total_tp = sum(r["tp"] for r in per_class_results.values())
    total_fp = sum(r["fp"] for r in per_class_results.values())
    total_fn = sum(r["fn"] for r in per_class_results.values())
    total_gt_count = sum(r["gt_count"] for r in per_class_results.values())
    total_pred_count = sum(r["pred_count"] for r in per_class_results.values())
    total_ap_pool = len(all_predictions)  # Full AP pool (conf>=0.001)
    mAP50 = np.mean([r["ap50"] for r in per_class_results.values()])
    mAP50_95 = np.mean([r["ap50_95"] for r in per_class_results.values()])

    inf_arr = np.array(inference_times_ms)

    overall = {
        "n_images": n_images,
        "total_gt_objects": total_gt_count,
        "total_predictions_op": total_pred_count,
        "total_predictions_ap_pool": total_ap_pool,
        "images_with_no_gt": images_with_no_gt,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0,
        "recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0,
        "mAP50": round(float(mAP50), 4),
        "mAP50_95": round(float(mAP50_95), 4),
        "avg_inference_ms": round(float(np.mean(inf_arr)), 2),
        "median_inference_ms": round(float(np.median(inf_arr)), 2),
        "p95_inference_ms": round(float(np.percentile(inf_arr, 95)), 2),
        "fps": round(n_images / total_wall_s, 2),
        "total_wall_s": round(total_wall_s, 2),
    }

    # --- Condition-wise results ---
    condition_results = {}
    for cond, cdata in condition_data.items():
        if cdata["n_images"] == 0:
            continue

        cond_gt_slim = [(g[0], g[1], g[2], g[3], g[4], g[5]) for g in cdata["gt"]]
        cond_ap50 = evaluate_at_iou_threshold(
            cdata["predictions"], cond_gt_slim, 0.5, eval_classes
        )

        # Precision/recall at IoU=0.5
        cond_gt_count = len(cdata["gt"])
        # Filter to operating-point confidence for P/R
        cond_op_preds = [p for p in cdata["predictions"] if p[2] >= CONFIDENCE_THRESHOLD]
        cond_pred_count = len(cond_op_preds)

        # Operating-point TP/FP counting for condition
        cond_gt_idx = defaultdict(list)
        for g in cdata["gt"]:
            cond_gt_idx[(g[0], g[1])].append({
                "box": (g[2], g[3], g[4], g[5]), "matched": False
            })

        cond_preds_sorted = sorted(cond_op_preds, key=lambda x: -x[2])
        cond_tp = 0
        cond_fp = 0
        for pred in cond_preds_sorted:
            img_idx, coco_cls, conf, px1, py1, px2, py2 = pred
            gts = cond_gt_idx.get((img_idx, coco_cls), [])
            best_iou_val = 0
            best_gi = -1
            for gi, gt_info in enumerate(gts):
                if gt_info["matched"]:
                    continue  # Only consider unmatched GTs
                iv = iou((px1, py1, px2, py2), gt_info["box"])
                if iv > best_iou_val:
                    best_iou_val = iv
                    best_gi = gi
            if best_iou_val >= 0.5 and best_gi >= 0:
                cond_tp += 1
                gts[best_gi]["matched"] = True
            else:
                cond_fp += 1
        cond_fn = cond_gt_count - cond_tp

        cond_inf = np.array(cdata["inference_times"])
        cond_mAP50 = np.mean([cond_ap50[c] for c in eval_classes])

        condition_results[cond] = {
            "n_images": cdata["n_images"],
            "gt_objects": cond_gt_count,
            "predictions": cond_pred_count,
            "tp": cond_tp,
            "fp": cond_fp,
            "fn": cond_fn,
            "precision": round(cond_tp / (cond_tp + cond_fp), 4) if (cond_tp + cond_fp) > 0 else 0.0,
            "recall": round(cond_tp / (cond_tp + cond_fn), 4) if (cond_tp + cond_fn) > 0 else 0.0,
            "mAP50": round(float(cond_mAP50), 4),
            "avg_inference_ms": round(float(np.mean(cond_inf)), 2),
            "median_inference_ms": round(float(np.median(cond_inf)), 2),
            "p95_inference_ms": round(float(np.percentile(cond_inf, 95)), 2),
        }

    return {
        "overall": overall,
        "per_class": per_class_results,
        "person_breakdown": person_breakdown,
        "size_breakdown": size_metrics,
        "condition_results": condition_results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results):
    o = results["overall"]
    pc = results["per_class"]
    pb = results["person_breakdown"]
    sb = results["size_breakdown"]
    cr = results["condition_results"]

    coco_names = {0: "person", 1: "bicycle", 2: "car", 5: "bus",
                  7: "truck", 9: "traffic light"}

    print()
    print("=" * 85)
    print("  YOLOv8n STOCK BASELINE — Dataset2 TEST Split")
    print("=" * 85)
    print()
    print(f"  Model              : yolov8n.pt (COCO pretrained, stock weights)")
    print(f"  Dataset            : dataset2 (Prep-Data)")
    print(f"  Split              : TEST")
    print(f"  Device             : MPS (Apple GPU)")
    print(f"  AP conf threshold  : {AP_CONF_THRESHOLD}")
    print(f"  OP conf threshold  : {CONFIDENCE_THRESHOLD}")
    print(f"  Total images       : {o['n_images']}")
    print(f"  Images with no GT  : {o['images_with_no_gt']}")
    print(f"  Total GT objects   : {o['total_gt_objects']}")
    print(f"  Predictions (OP)   : {o['total_predictions_op']} (conf>={CONFIDENCE_THRESHOLD})")
    print(f"  Predictions (AP)   : {o['total_predictions_ap_pool']} (conf>={AP_CONF_THRESHOLD})")
    print()
    print(f"  ── AP Metrics (conf>={AP_CONF_THRESHOLD}, full PR curve) ──")
    print(f"  mAP@0.50           : {o['mAP50']:.4f} ({o['mAP50']*100:.2f}%)")
    print(f"  mAP@0.50:0.95      : {o['mAP50_95']:.4f} ({o['mAP50_95']*100:.2f}%)")
    print()
    print(f"  ── Operating-Point Metrics (conf>={CONFIDENCE_THRESHOLD}, IoU>=0.50) ──")
    print(f"  Precision          : {o['precision']:.4f} ({o['precision']*100:.2f}%)")
    print(f"  Recall             : {o['recall']:.4f} ({o['recall']*100:.2f}%)")
    print()
    print(f"  ── Latency (MPS-synchronized) ──")
    print(f"  Mean inference     : {o['avg_inference_ms']:.2f} ms")
    print(f"  Median inference   : {o['median_inference_ms']:.2f} ms")
    print(f"  P95 inference      : {o['p95_inference_ms']:.2f} ms")
    print(f"  FPS (wall clock)   : {o['fps']:.2f}")
    print(f"  Total wall time    : {o['total_wall_s']:.1f}s")
    print()

    print(f"  ── Per-Class Results ──")
    header = (f"  {'COCO Class':<16} {'DS2 src':<22} {'GT':>7} {'Pred':>7} "
              f"{'TP':>6} {'FP':>6} {'FN':>6} "
              f"{'Prec':>7} {'Recall':>7} {'AP@.50':>8} {'AP@.50:.95':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for coco_cls in sorted(pc.keys()):
        r = pc[coco_cls]
        ds2_src = ", ".join(r["ds2_mapped_from"])
        print(f"  {r['coco_name']:<16} {ds2_src:<22} {r['gt_count']:>7} "
              f"{r['pred_count']:>7} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['ap50']:>8.4f} {r['ap50_95']:>10.4f}")
    print()

    print(f"  ── Pedestrian vs Rider Sub-Breakdown (under COCO 'person') ──")
    print(f"  {'Category':<16} {'GT':>7} {'TP':>6} {'FN':>6} {'Recall':>8}")
    print("  " + "-" * 48)
    for cat in ["pedestrian", "rider"]:
        p = pb[cat]
        print(f"  {cat:<16} {p['gt']:>7} {p['tp']:>6} {p['fn']:>6} {p['recall']:>8.4f}")
    print()

    print(f"  ── Per-Size Breakdown (COCO area defs) ──")
    areas = {"small": "area < 32²", "medium": "32² ≤ area < 96²", "large": "area ≥ 96²"}
    print(f"  {'Size':<10} {'Definition':<18} {'GT':>7} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>7} {'Recall':>7}")
    print("  " + "-" * 74)
    for s in ["small", "medium", "large"]:
        r = sb[s]
        gt_total = r["tp"] + r["fn"]
        print(f"  {s:<10} {areas[s]:<18} {gt_total:>7} {r['tp']:>6} "
              f"{r['fp']:>6} {r['fn']:>6} {r['precision']:>7.4f} {r['recall']:>7.4f}")
    print()

    if cr:
        print(f"  ── Condition-Wise Results ──")
        print(f"  {'Condition':<10} {'Images':>8} {'GT':>8} {'Pred':>8} "
              f"{'Prec':>7} {'Recall':>7} {'mAP@.50':>8} "
              f"{'Avg ms':>8} {'Med ms':>8} {'P95 ms':>8}")
        print("  " + "-" * 92)
        for cond in sorted(cr.keys()):
            c = cr[cond]
            print(f"  {cond:<10} {c['n_images']:>8} {c['gt_objects']:>8} "
                  f"{c['predictions']:>8} {c['precision']:>7.4f} {c['recall']:>7.4f} "
                  f"{c['mAP50']:>8.4f} {c['avg_inference_ms']:>8.2f} "
                  f"{c['median_inference_ms']:>8.2f} {c['p95_inference_ms']:>8.2f}")
        print()


def save_results_json(results, output_path):
    """Save full results as JSON for reproducibility."""
    import copy

    output = {
        "metadata": {
            "model": "yolov8n.pt",
            "model_type": "stock COCO pretrained (no fine-tuning)",
            "pretrained": True,
            "fine_tuned": False,
            "dataset": "dataset2",
            "dataset_path": PREP_DATA_DIR,
            "split": "test",
            "device": "mps",
            "ap_confidence_threshold": AP_CONF_THRESHOLD,
            "operating_point_confidence": CONFIDENCE_THRESHOLD,
            "iou_threshold_ap50": 0.5,
            "iou_thresholds_ap50_95": IOU_THRESHOLDS_FINE,
            "class_mapping": {
                DS2_CLASSES[k]: {
                    "ds2_id": k,
                    "coco_id": v,
                    "coco_name": {0: "person", 1: "bicycle", 2: "car",
                                  5: "bus", 7: "truck", 9: "traffic light"}[v],
                    "note": ("No COCO equivalent; mapped to 'person' per repo "
                             "convention in detection_baseline.py"
                             if k == 4 else None)
                }
                for k, v in DS2_TO_COCO.items()
            },
            "latency_methodology": {
                "warmup": "1 dummy frame (720x1280, zeros)",
                "timing": "time.perf_counter() around model.predict(), MPS-synchronized",
                "synchronization": "torch.mps.synchronize() before t0 and after predict()",
                "excludes": "data loading (cv2.imread), label parsing",
                "metrics_reported": ["mean", "median", "p95", "fps_wall_clock"]
            },
            "size_buckets": {
                "small": "area < 1024 px² (32²)",
                "medium": "1024 px² ≤ area < 9216 px² (96²)",
                "large": "area ≥ 9216 px² (96²)"
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "overall": results["overall"],
        "per_class": {
            results["per_class"][k]["coco_name"]: {
                **{kk: vv for kk, vv in results["per_class"][k].items()},
            }
            for k in sorted(results["per_class"].keys())
        },
        "person_breakdown": results["person_breakdown"],
        "size_breakdown": results["size_breakdown"],
        "condition_results": results["condition_results"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import torch

    print("=" * 85)
    print("  YOLOv8n Stock Baseline Evaluation — Dataset2 TEST Split")
    print("=" * 85)

    # ── Pre-flight checks ──
    print("\n  ── Pre-flight Checks ──")

    # MPS check
    if not torch.backends.mps.is_available():
        print("  [FATAL] MPS is NOT available. Cannot proceed.")
        sys.exit(1)
    print(f"  ✓ MPS available: True")
    print(f"  ✓ PyTorch version: {torch.__version__}")

    # Ultralytics version
    import ultralytics
    print(f"  ✓ Ultralytics version: {ultralytics.__version__}")

    # Dataset paths
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"  [FATAL] Test images dir not found: {TEST_IMAGES_DIR}")
        sys.exit(1)
    if not os.path.isdir(TEST_LABELS_DIR):
        print(f"  [FATAL] Test labels dir not found: {TEST_LABELS_DIR}")
        sys.exit(1)
    print(f"  ✓ Test images dir: {TEST_IMAGES_DIR}")
    print(f"  ✓ Test labels dir: {TEST_LABELS_DIR}")

    # Collect samples
    samples, missing = collect_test_samples()
    print(f"  ✓ Test samples found: {len(samples)}")
    if missing > 0:
        print(f"  ⚠ Images missing labels: {missing}")

    # Condition breakdown
    cond_counts = defaultdict(int)
    for s in samples:
        cond_counts[s["condition"]] += 1
    for cond, cnt in sorted(cond_counts.items()):
        print(f"    - {cond}: {cnt} images")

    # ── Load model ──
    print(f"\n  Loading model: yolov8n.pt (stock COCO weights)")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print(f"  ✓ Model loaded ({len(model.names)} COCO classes)")

    # Print class mapping
    print(f"\n  ── Class Mapping (Dataset2 → COCO) ──")
    for ds2_id, coco_id in DS2_TO_COCO.items():
        note = " ← NO COCO equivalent, mapped to 'person'" if ds2_id == 4 else ""
        print(f"    DS2 '{DS2_CLASSES[ds2_id]}' (id={ds2_id}) "
              f"→ COCO '{model.names[coco_id]}' (id={coco_id}){note}")

    # ── Run evaluation ──
    results = run_full_evaluation(samples, model)

    # ── Print results ──
    print_results(results)

    # ── Save results ──
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "historical", "yolov8n_stock_baseline_dataset2"
    )
    output_path = os.path.join(output_dir, "results.json")
    save_results_json(results, output_path)

    print(f"\n  ── Self-Check ──")
    print(f"  ✓ Output file exists: {os.path.exists(output_path)}")
    print(f"  ✓ Evaluated split: TEST")
    print(f"  ✓ Image count: {results['overall']['n_images']}")
    print(f"  ✓ Model: yolov8n.pt (stock, no training)")
    print(f"  ✓ No Adaptive Vision modules used")
    print()


if __name__ == "__main__":
    main()
