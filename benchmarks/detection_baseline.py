"""Phase 5 Pre-requisite: Detection-Quality Baseline on BDD100K Val (N=425).

Evaluates plain YOLOv8n against 425 BDD100K validation set images on disk,
scoped to the 7 road-scene categories defined in the research paper:
    car, bus, truck, pedestrian, rider, bicycle, traffic light.

Stated limitations / Class exclusions:
    - BDD100K "traffic sign" (1,472 annotations in N=425) is EXCLUDED because
      standard COCO-pretrained lightweight YOLO architectures (such as
      YOLOv8n) lack an equivalent class representation.
    - "motorcycle" (21 annotations) is EXCLUDED from the 7-class vocabulary.
    - "other vehicle", "trailer", "train" are EXCLUDED.

Rider handling:
    - BDD100K distinguishes "pedestrian" (walking) and "rider" (on bike/moto).
    - COCO class 0 is "person" (encompassing both walking pedestrians and riders).
    - In the standard 7-class evaluation, both "pedestrian" and "rider" map to
      COCO class 0 ("person").
    - This script also computes and reports sub-metrics for "pedestrian" vs "rider"
      separately so the effect of this mapping is fully transparent.

Size definitions (COCO standard, in pixels):
    small:  area < 32²  = 1024 px²
    medium: 32² ≤ area < 96² = 9216 px²
    large:  area ≥ 96²

Usage:
    .venv/bin/python benchmarks/detection_baseline.py [--n-images 425]
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

# 7-Class paper vocabulary mapping: BDD100K label -> COCO class id (YOLOv8n)
BDD_TO_COCO = {
    "car":           2,
    "truck":         7,
    "bus":           5,
    "pedestrian":    0,
    "rider":         0,   # mapped to person (flagged explicitly)
    "bicycle":       1,
    "traffic light": 9,
}

# Evaluated COCO classes (6 heads covering the 7 BDD categories)
COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    5: "bus",
    7: "truck",
    9: "traffic light",
}

# COCO-style size thresholds (pixels²)
SMALL_AREA  = 32 * 32    # 1024
MEDIUM_AREA = 96 * 96    # 9216

BASELINE_DIR = os.path.join("benchmarks", "detection_baseline")
DEFAULT_IMAGE_DIR = os.path.join(BASELINE_DIR, "images")

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotations(samples_json_path, image_dir=None, n_images=None, seed=42):
    """Load BDD100K annotations from FiftyOne samples.json for available images.

    Returns:
        list of dicts: [{filepath, gt_boxes: [(class_id, x1, y1, x2, y2, orig_label)], ...}, ...]
    """
    with open(samples_json_path) as f:
        data = json.load(f)

    samples = data["samples"]

    # Filter to images existing in image_dir if specified
    if image_dir and os.path.exists(image_dir):
        local_files = set(os.listdir(image_dir))
        matched = []
        for s in samples:
            fname = os.path.basename(s["filepath"])
            if fname in local_files:
                s_copy = dict(s)
                s_copy["filepath"] = os.path.join(image_dir, fname)
                matched.append(s_copy)
        samples = matched
        print(f"  Matched {len(samples)} local images in {image_dir}")

    if n_images and n_images < len(samples):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(samples), n_images, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    results = []
    skipped_cats = Counter()
    total_gt = 0
    mapped_gt = 0

    for s in samples:
        filepath = s["filepath"]
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
                bx, by, bw, bh = d["bounding_box"]
                x1 = bx * w
                y1 = by * h
                x2 = (bx + bw) * w
                y2 = (by + bh) * h
                gt_boxes.append((coco_id, x1, y1, x2, y2, label))

        results.append({
            "filepath": filepath,
            "gt_boxes": gt_boxes,
            "timeofday": s.get("timeofday", {}).get("label", "unknown")
                         if isinstance(s.get("timeofday"), dict)
                         else s.get("timeofday", "unknown"),
        })

    print(f"  Loaded {len(results)} images (N={len(results)})")
    print(f"  Mapped GT boxes   : {mapped_gt}")
    print(f"  Excluded GT boxes : {total_gt - mapped_gt} (Total: {total_gt})")
    print(f"  Exclusion breakdown:")
    for cat, count in skipped_cats.most_common():
        reason = "COCO lack equivalent class" if cat == "traffic sign" else "not in 7-class paper vocabulary"
        print(f"    - {cat:<15}: {count:>5} ({reason})")

    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(samples, model_path, conf_threshold, iou_threshold=0.5):
    """Run YOLOv8n on images and compute per-class, per-size, and sub-label metrics."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Warm up
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=conf_threshold, verbose=False, save=False)

    all_predictions = []  # (image_idx, class_id, confidence, x1, y1, x2, y2)
    all_gt = []           # (image_idx, class_id, x1, y1, x2, y2, size_bucket, orig_label)

    mapped_coco_ids = set(COCO_NAMES.keys())

    total_inference_ms = 0.0
    total_frames = 0

    print("  Running YOLOv8n inference...")
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
            cls_id, x1, y1, x2, y2, orig_label = gt
            area = box_area_px((x1, y1, x2, y2))
            sb = size_bucket(area)
            all_gt.append((img_idx, cls_id, x1, y1, x2, y2, sb, orig_label))

        # Collect predictions (only for mapped classes in the 7-class scope)
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

        if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(samples):
            print(f"    Evaluated {img_idx + 1}/{len(samples)} images...")

    avg_inference = total_inference_ms / total_frames if total_frames > 0 else 0

    # -----------------------------------------------------------------------
    # Compute metrics per class and per size
    # -----------------------------------------------------------------------

    gt_by_img_cls = defaultdict(list)
    for gt in all_gt:
        img_idx, cls_id, x1, y1, x2, y2, sb, orig_label = gt
        gt_by_img_cls[(img_idx, cls_id)].append({
            "box": (x1, y1, x2, y2), "matched": False, "size": sb, "orig_label": orig_label
        })

    all_predictions.sort(key=lambda x: -x[2])

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

    # Per-size metrics
    for key in gt_by_img_cls:
        for g in gt_by_img_cls[key]:
            sb = g["size"]
            if g["matched"]:
                results_by_size[sb]["tp"] += 1
            else:
                results_by_size[sb]["fn"] += 1

    fp_by_size = {"small": 0, "medium": 0, "large": 0}
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

    # Sub-breakdown for pedestrian vs rider under person class (cls_id 0)
    pedestrian_gt = 0
    pedestrian_tp = 0
    rider_gt = 0
    rider_tp = 0

    for key in gt_by_img_cls:
        if key[1] == 0:  # person
            for g in gt_by_img_cls[key]:
                if g["orig_label"] == "pedestrian":
                    pedestrian_gt += 1
                    if g["matched"]:
                        pedestrian_tp += 1
                elif g["orig_label"] == "rider":
                    rider_gt += 1
                    if g["matched"]:
                        rider_tp += 1

    person_breakdown = {
        "pedestrian": {
            "gt": pedestrian_gt,
            "tp": pedestrian_tp,
            "fn": pedestrian_gt - pedestrian_tp,
            "recall": round(pedestrian_tp / pedestrian_gt, 4) if pedestrian_gt > 0 else 0.0,
        },
        "rider": {
            "gt": rider_gt,
            "tp": rider_tp,
            "fn": rider_gt - rider_tp,
            "recall": round(rider_tp / rider_gt, 4) if rider_gt > 0 else 0.0,
        },
    }

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
        "person_breakdown": person_breakdown,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results):
    overall = results["overall"]
    by_class = results["by_class"]
    by_size = results["by_size"]
    pb = results["person_breakdown"]

    print()
    print("=" * 78)
    print("  YOLOv8n Ground-Truth Baseline Results (BDD100K Val, N=425)")
    print("=" * 78)
    print()
    print(f"  Dataset split      : BDD100K 10K val subset (N=425 images)")
    print(f"  Scope              : 7 categories (traffic sign excluded)")
    print(f"  Total GT boxes     : {overall['total_gt']}")
    print(f"  Total predictions  : {overall['total_pred']}")
    print(f"  True positives     : {overall['total_tp']}")
    print(f"  False positives    : {overall['total_fp']}")
    print(f"  False negatives    : {overall['total_fn']}")
    print(f"  Precision          : {overall['precision']:.4f} ({overall['precision']*100:.2f}%)")
    print(f"  Recall             : {overall['recall']:.4f} ({overall['recall']*100:.2f}%)")
    print(f"  mAP@0.5            : {overall['mAP50']:.4f} ({overall['mAP50']*100:.2f}%)")
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

    print("  Rider Mapping Analysis (under 'person' class):")
    print(f"  {'Category':<16} {'GT':>6} {'TP':>6} {'FN':>6} {'Recall':>8}")
    print("  " + "-" * 48)
    print(f"  {'pedestrian':<16} {pb['pedestrian']['gt']:>6} {pb['pedestrian']['tp']:>6} "
          f"{pb['pedestrian']['fn']:>6} {pb['pedestrian']['recall']:>8.4f}")
    print(f"  {'rider':<16} {pb['rider']['gt']:>6} {pb['rider']['tp']:>6} "
          f"{pb['rider']['fn']:>6} {pb['rider']['recall']:>8.4f}")
    print(f"  {'combined person':<16} {by_class[0]['gt_count']:>6} {by_class[0]['tp']:>6} "
          f"{by_class[0]['fn']:>6} {by_class[0]['recall']:>8.4f}")
    print()

    print("  Per-Size Results (COCO area definitions):")
    print(f"  {'Size Bucket':<14} {'Pixel Area':<16} {'GT (TP+FN)':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8}")
    print("  " + "-" * 82)
    areas = {"small": "area < 32²", "medium": "32² ≤ area < 96²", "large": "area ≥ 96²"}
    for sb in ["small", "medium", "large"]:
        r = by_size[sb]
        gt_size = r["tp"] + r["fn"]
        print(f"  {sb:<14} {areas[sb]:<16} {gt_size:>10} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
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

    # Rider breakdown
    pb_path = os.path.join(output_dir, "baseline_pedestrian_rider.csv")
    with open(pb_path, "w", newline="") as f:
        fieldnames = ["category", "gt", "tp", "fn", "recall"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cat in ["pedestrian", "rider"]:
            w.writerow({"category": cat, **results["person_breakdown"][cat]})

    print(f"  Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOv8n detection baseline on BDD100K val (N=425)")
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR,
                        help=f"Directory of validation images (default: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("--n-images", type=int, default=425,
                        help="Number of validation images to evaluate (default: 425)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection (default: 42)")
    args = parser.parse_args()

    print("=" * 78)
    print("Phase 5 Pre-requisite: Plain YOLOv8n Ground-Truth Baseline (N=425)")
    print("=" * 78)
    print()
    print(f"  Model               : {MODEL_PATH}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Target sample size  : N={args.n_images}")
    print(f"  Image directory     : {args.image_dir}")
    print(f"  IoU threshold       : 0.5")
    print()

    # Step 1: Locate annotations
    samples_json = os.path.join(
        os.path.expanduser("~"),
        ".cache/huggingface/hub/datasets--dgural--bdd100k/"
        "snapshots/c2e7f266756bcd07b87f1a45a35937c8eac20241/samples.json"
    )

    if not os.path.exists(samples_json):
        print(f"  [ERROR] Cannot find annotations at {samples_json}")
        sys.exit(1)

    print("  Loading annotations...")
    samples = load_annotations(samples_json, image_dir=args.image_dir, n_images=args.n_images, seed=args.seed)

    if len(samples) == 0:
        print(f"  [ERROR] No images found in {args.image_dir}")
        sys.exit(1)

    # Step 2: Run evaluation
    results = run_evaluation(samples, MODEL_PATH, CONFIDENCE_THRESHOLD)

    # Step 3: Print and save
    print_results(results)
    save_results(results, BASELINE_DIR)

    print()
    print("  --- Evaluation Scope Summary ---")
    print("  - Evaluated dataset: BDD100K 10K val split, N=425 still images on disk")
    print("  - 7 Categories evaluated: car, bus, truck, pedestrian, rider, bicycle, traffic light")
    print("  - Traffic sign (1,472 annotations) explicitly excluded (COCO model lacks class)")
    print("  - Rider (35 annotations) mapped to COCO person head; pedestrian (581) also mapped to person")
    print("  - Metrics established as ground truth baseline for Phase 5 small-object comparisons.")
    print()


if __name__ == "__main__":
    main()
