"""Phase 5 Pre-requisite: Detection-Quality Baseline on BDD100K Val.

Downloads a representative subset of the BDD100K 10K validation set
(from HuggingFace: dgural/bdd100k), converts ground-truth annotations
to YOLO format, runs YOLOv8n inference, and computes precision / recall /
mAP — overall and broken down by object size (small / medium / large,
using COCO size definitions).

The purpose is to establish a measurable baseline before Phase 5
(confidence-guided re-detection) claims to improve small-object recall.

Class mapping:
    BDD100K uses 13 object categories.  YOLOv8n is trained on COCO's
    80 classes.  Only the 8 BDD100K categories that have a clear COCO
    equivalent are evaluated:

        BDD100K "car"          →  COCO "car" (id 2)
        BDD100K "truck"        →  COCO "truck" (id 7)
        BDD100K "bus"          →  COCO "bus" (id 5)
        BDD100K "pedestrian"   →  COCO "person" (id 0)
        BDD100K "rider"        →  COCO "person" (id 0)
        BDD100K "bicycle"      →  COCO "bicycle" (id 1)
        BDD100K "motorcycle"   →  COCO "motorcycle" (id 3)
        BDD100K "traffic light" → COCO "traffic light" (id 9)

    BDD100K "traffic sign", "train", "other vehicle", "trailer",
    "other person" are excluded (no direct COCO match or too rare).

Size definitions (COCO standard, in pixels):
    small:  area < 32²  = 1024
    medium: 32² ≤ area < 96² = 9216
    large:  area ≥ 96²

Usage:
    .venv/bin/python benchmarks/detection_baseline.py [--n-images 500]
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, CONFIDENCE_THRESHOLD

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BDD100K label → COCO class id mapping (YOLOv8n class indices)
BDD_TO_COCO = {
    "car":           2,
    "truck":         7,
    "bus":           5,
    "pedestrian":    0,
    "rider":         0,   # closest COCO class
    "bicycle":       1,
    "motorcycle":    3,
    "traffic light":  9,
}

# COCO class names for the mapped subset
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
              5: "bus", 7: "truck", 9: "traffic light"}

# COCO-style size thresholds (pixels²)
SMALL_AREA  = 32 * 32    # 1024
MEDIUM_AREA = 96 * 96    # 9216

BASELINE_DIR = os.path.join("benchmarks", "detection_baseline")

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
    # Prepend sentinel values
    prec = np.concatenate(([1.0], precisions, [0.0]))
    rec = np.concatenate(([0.0], recalls, [1.0]))
    # Make precision monotonically decreasing
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = rec >= t
        if mask.any():
            ap += prec[mask].max()
    return ap / 101.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations(samples_json_path, n_images=None, seed=42):
    """Load BDD100K annotations from FiftyOne samples.json.

    Returns:
        list of dicts: [{filepath, gt_boxes: [(class_id, x1, y1, x2, y2)], ...}, ...]
    """
    with open(samples_json_path) as f:
        data = json.load(f)

    samples = data["samples"]
    rng = np.random.RandomState(seed)

    if n_images and n_images < len(samples):
        indices = rng.choice(len(samples), n_images, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    results = []
    skipped_cats = Counter()
    total_gt = 0
    mapped_gt = 0

    for s in samples:
        filepath = s["filepath"]  # relative: "data/xxxxx.jpg"
        w = s["metadata"]["width"]
        h = s["metadata"]["height"]

        gt_boxes = []
        dets = s.get("detections", {})
        if dets and dets.get("detections"):
            for d in dets["detections"]:
                label = d["label"]
                total_gt += 1
                if label not in BDD_TO_COCO:
                    skipped_cats[label] += 1
                    continue
                mapped_gt += 1
                coco_id = BDD_TO_COCO[label]
                # FiftyOne bbox: [x_rel, y_rel, w_rel, h_rel]
                bx, by, bw, bh = d["bounding_box"]
                x1 = bx * w
                y1 = by * h
                x2 = (bx + bw) * w
                y2 = (by + bh) * h
                gt_boxes.append((coco_id, x1, y1, x2, y2))

        results.append({
            "filepath": filepath,
            "gt_boxes": gt_boxes,
            "timeofday": s.get("timeofday", {}).get("label", "unknown")
                         if isinstance(s.get("timeofday"), dict)
                         else s.get("timeofday", "unknown"),
        })

    print(f"  Loaded {len(results)} images, {mapped_gt} mapped GT boxes "
          f"({total_gt} total, {total_gt - mapped_gt} skipped)")
    if skipped_cats:
        print(f"  Skipped categories: {dict(skipped_cats)}")

    return results


def download_images(samples, hf_repo="dgural/bdd100k", dest_dir=None):
    """Download images from HuggingFace Hub.

    Returns updated samples with absolute filepaths.
    """
    from huggingface_hub import hf_hub_download

    if dest_dir is None:
        dest_dir = os.path.join(BASELINE_DIR, "images")
    os.makedirs(dest_dir, exist_ok=True)

    updated = []
    for i, s in enumerate(samples):
        rel_path = s["filepath"]  # e.g., "data/b1c66a42-6f7d68ca.jpg"
        local_name = os.path.basename(rel_path)
        local_path = os.path.join(dest_dir, local_name)

        if not os.path.exists(local_path):
            # Download from HuggingFace
            cached = hf_hub_download(hf_repo, rel_path, repo_type="dataset")
            # Copy/link to our directory
            import shutil
            shutil.copy2(cached, local_path)

        s_copy = dict(s)
        s_copy["filepath"] = local_path
        updated.append(s_copy)

        if (i + 1) % 50 == 0:
            print(f"    Downloaded {i + 1}/{len(samples)} images...")

    print(f"  All {len(updated)} images ready at {dest_dir}")
    return updated


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(samples, model_path, conf_threshold, iou_threshold=0.5):
    """Run YOLOv8n on images and compute per-class, per-size metrics.

    Returns:
        dict with overall and per-class and per-size metrics.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Warm up
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=conf_threshold, verbose=False, save=False)

    # Accumulators: for each (class_id, size_bucket), track TP/FP/FN
    # We'll compute AP properly by collecting all predictions with scores
    all_predictions = []  # (image_idx, class_id, confidence, x1, y1, x2, y2)
    all_gt = []           # (image_idx, class_id, x1, y1, x2, y2, size_bucket)

    mapped_coco_ids = set(COCO_NAMES.keys())

    total_inference_ms = 0.0
    total_frames = 0

    for img_idx, sample in enumerate(samples):
        img = cv2.imread(sample["filepath"])
        if img is None:
            print(f"  [WARN] Cannot read {sample['filepath']}")
            continue

        total_frames += 1
        t0 = time.perf_counter()
        results = model.predict(
            source=img, conf=conf_threshold, verbose=False, save=False
        )
        t1 = time.perf_counter()
        total_inference_ms += (t1 - t0) * 1000.0

        # Collect ground truth
        for gt in sample["gt_boxes"]:
            cls_id, x1, y1, x2, y2 = gt
            area = box_area_px((x1, y1, x2, y2))
            sb = size_bucket(area)
            all_gt.append((img_idx, cls_id, x1, y1, x2, y2, sb))

        # Collect predictions (only for mapped classes)
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in mapped_coco_ids:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                all_predictions.append(
                    (img_idx, cls_id, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                )

        if (img_idx + 1) % 50 == 0:
            print(f"    Processed {img_idx + 1}/{len(samples)} images...")

    avg_inference = total_inference_ms / total_frames if total_frames > 0 else 0

    # -----------------------------------------------------------------------
    # Compute metrics per class and per size
    # -----------------------------------------------------------------------

    # Group GT by (image_idx, class_id)
    gt_by_img_cls = defaultdict(list)
    for gt in all_gt:
        img_idx, cls_id, x1, y1, x2, y2, sb = gt
        gt_by_img_cls[(img_idx, cls_id)].append({
            "box": (x1, y1, x2, y2), "matched": False, "size": sb
        })

    # Sort predictions by confidence (descending) for AP computation
    all_predictions.sort(key=lambda x: -x[2])

    # Per-class AP computation
    results_by_class = {}
    results_by_size = {"small": {"tp": 0, "fp": 0, "fn": 0},
                       "medium": {"tp": 0, "fp": 0, "fn": 0},
                       "large": {"tp": 0, "fp": 0, "fn": 0}}

    for cls_id in sorted(mapped_coco_ids):
        cls_preds = [(p[0], p[2], p[3], p[4], p[5], p[6])
                     for p in all_predictions if p[1] == cls_id]
        cls_gt_count = sum(1 for g in all_gt if g[1] == cls_id)

        if cls_gt_count == 0 and len(cls_preds) == 0:
            continue

        # Reset matched flags
        for key in gt_by_img_cls:
            if key[1] == cls_id:
                for g in gt_by_img_cls[key]:
                    g["matched"] = False

        tp_list = []
        fp_list = []

        for pred in cls_preds:
            img_idx, conf, px1, py1, px2, py2 = pred
            pred_box = (px1, py1, px2, py2)
            gts = gt_by_img_cls.get((img_idx, cls_id), [])

            best_iou = 0
            best_gt_idx = -1
            for gi, gt_info in enumerate(gts):
                iou_val = iou(pred_box, gt_info["box"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gi

            if best_iou >= iou_threshold and best_gt_idx >= 0 and not gts[best_gt_idx]["matched"]:
                tp_list.append(1)
                fp_list.append(0)
                gts[best_gt_idx]["matched"] = True
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)

        precisions = tp_cum / (tp_cum + fp_cum)
        recalls = tp_cum / cls_gt_count if cls_gt_count > 0 else tp_cum

        ap = compute_ap(precisions, recalls)
        total_tp = int(tp_cum[-1]) if len(tp_cum) > 0 else 0
        total_fp = int(fp_cum[-1]) if len(fp_cum) > 0 else 0
        fn = cls_gt_count - total_tp

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / cls_gt_count if cls_gt_count > 0 else 0

        results_by_class[cls_id] = {
            "name": COCO_NAMES[cls_id],
            "gt_count": cls_gt_count,
            "pred_count": len(cls_preds),
            "tp": total_tp,
            "fp": total_fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "ap50": round(ap, 4),
        }

    # Per-size metrics: count matched/unmatched GT by size
    for key in gt_by_img_cls:
        for g in gt_by_img_cls[key]:
            sb = g["size"]
            if g["matched"]:
                results_by_size[sb]["tp"] += 1
            else:
                results_by_size[sb]["fn"] += 1

    # FP by size — attribute FP to size of the prediction box
    fp_by_size = {"small": 0, "medium": 0, "large": 0}
    pred_idx = 0
    # Re-sort and re-process for FP size attribution
    for pred in all_predictions:
        img_idx, cls_id, conf, px1, py1, px2, py2 = pred
        if cls_id not in mapped_coco_ids:
            continue
        pred_area = box_area_px((px1, py1, px2, py2))
        pred_size = size_bucket(pred_area)
        gts = gt_by_img_cls.get((img_idx, cls_id), [])
        is_tp = False
        for g in gts:
            if iou((px1, py1, px2, py2), g["box"]) >= iou_threshold:
                is_tp = True
                break
        if not is_tp:
            fp_by_size[pred_size] += 1

    for sb in results_by_size:
        results_by_size[sb]["fp"] = fp_by_size[sb]
        tp = results_by_size[sb]["tp"]
        fp = results_by_size[sb]["fp"]
        fn = results_by_size[sb]["fn"]
        results_by_size[sb]["precision"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0
        results_by_size[sb]["recall"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0

    # Overall
    total_tp = sum(r["tp"] for r in results_by_class.values())
    total_fp = sum(r["fp"] for r in results_by_class.values())
    total_fn = sum(r["fn"] for r in results_by_class.values())
    total_gt_count = sum(r["gt_count"] for r in results_by_class.values())
    mAP50 = np.mean([r["ap50"] for r in results_by_class.values()]) if results_by_class else 0

    overall = {
        "n_images": total_frames,
        "total_gt": total_gt_count,
        "total_pred": len([p for p in all_predictions if p[1] in mapped_coco_ids]),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0,
        "recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0,
        "mAP50": round(mAP50, 4),
        "avg_inference_ms": round(avg_inference, 2),
    }

    return {
        "overall": overall,
        "by_class": results_by_class,
        "by_size": results_by_size,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results):
    overall = results["overall"]
    by_class = results["by_class"]
    by_size = results["by_size"]

    print()
    print("=" * 70)
    print("  YOLOv8n Detection Baseline on BDD100K Val")
    print("=" * 70)
    print()
    print(f"  Images evaluated   : {overall['n_images']}")
    print(f"  Total GT boxes     : {overall['total_gt']}")
    print(f"  Total predictions  : {overall['total_pred']}")
    print(f"  True positives     : {overall['total_tp']}")
    print(f"  False positives    : {overall['total_fp']}")
    print(f"  False negatives    : {overall['total_fn']}")
    print(f"  Precision          : {overall['precision']:.4f}")
    print(f"  Recall             : {overall['recall']:.4f}")
    print(f"  mAP@0.5            : {overall['mAP50']:.4f}")
    print(f"  Avg inference      : {overall['avg_inference_ms']:.2f} ms")
    print()

    print("  Per-Class Results (IoU≥0.5):")
    print(f"  {'Class':<16} {'GT':>6} {'Pred':>6} {'TP':>6} {'FP':>6} "
          f"{'FN':>6} {'Prec':>8} {'Recall':>8} {'AP@0.5':>8}")
    print("  " + "-" * 82)
    for cls_id in sorted(by_class.keys()):
        r = by_class[cls_id]
        print(f"  {r['name']:<16} {r['gt_count']:>6} {r['pred_count']:>6} "
              f"{r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['ap50']:>8.4f}")
    print()

    print("  Per-Size Results (COCO size definitions):")
    print(f"  {'Size':<10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8}")
    print("  " + "-" * 52)
    for sb in ["small", "medium", "large"]:
        r = by_size[sb]
        print(f"  {sb:<10} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f}")
    print()


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Overall
    overall_path = os.path.join(output_dir, "baseline_overall.csv")
    with open(overall_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results["overall"].keys())
        w.writeheader()
        w.writerow(results["overall"])

    # Per-class
    class_path = os.path.join(output_dir, "baseline_per_class.csv")
    with open(class_path, "w", newline="") as f:
        rows = [results["by_class"][k] for k in sorted(results["by_class"].keys())]
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Per-size
    size_path = os.path.join(output_dir, "baseline_per_size.csv")
    with open(size_path, "w", newline="") as f:
        fieldnames = ["size", "tp", "fp", "fn", "precision", "recall"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sb in ["small", "medium", "large"]:
            row = {"size": sb, **results["by_size"][sb]}
            w.writerow(row)

    print(f"  Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n detection baseline on BDD100K val")
    parser.add_argument("--n-images", type=int, default=500,
                        help="Number of validation images to evaluate (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 5 Pre-requisite: Detection-Quality Baseline")
    print("=" * 70)
    print()
    print(f"  Model: {MODEL_PATH}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Images to evaluate: {args.n_images}")
    print(f"  IoU threshold: 0.5")
    print()

    # Step 1: Load annotations
    samples_json = os.path.join(
        os.path.expanduser("~"),
        ".cache/huggingface/hub/datasets--dgural--bdd100k/"
        "snapshots/c2e7f266756bcd07b87f1a45a35937c8eac20241/samples.json"
    )

    if not os.path.exists(samples_json):
        print("  [INFO] Downloading BDD100K val annotations from HuggingFace...")
        from huggingface_hub import hf_hub_download
        samples_json = hf_hub_download("dgural/bdd100k", "samples.json",
                                        repo_type="dataset")

    print("  Loading annotations...")
    samples = load_annotations(samples_json, n_images=args.n_images, seed=args.seed)

    # Step 2: Download images
    print("  Downloading images...")
    samples = download_images(samples)

    # Step 3: Run evaluation
    print("  Running YOLOv8n inference + evaluation...")
    results = run_evaluation(samples, MODEL_PATH, CONFIDENCE_THRESHOLD)

    # Step 4: Print and save
    print_results(results)
    save_results(results, BASELINE_DIR)

    # Step 5: Flag mismatch
    print()
    print("  --- Dataset Mismatch Note ---")
    print("  Speed benchmarks (Phase 3/4) used: 50.mp4, 544.mp4, 1600.mp4")
    print("  CLAHE validation used: 621.mp4 (night), 139.mp4 (dusk)")
    print("  Detection baseline uses: BDD100K val images (10K official split)")
    print("  These are DIFFERENT data sources:")
    print("    - Videos: sequential frames from BDD100K video subset (Kaggle)")
    print("    - Val images: BDD100K's official 10K labeled still images")
    print("  The video dataset has no ground-truth detection labels, so")
    print("  the detection baseline necessarily uses a different data split.")
    print("  Both are from BDD100K and share the same domain (US road scenes),")
    print("  so the baseline is representative even though not frame-identical.")
    print()


if __name__ == "__main__":
    main()
