import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VAL_IMAGES_DIR = "Stage3-Smoke/images/val"
VAL_LABELS_DIR = "Stage3-Smoke/labels/val"

CONFIDENCE_THRESHOLD = 0.25      # Operating-point threshold for P/R
AP_CONF_THRESHOLD = 0.001        # Low threshold for full AP PR curve

# Dataset2 class names
DS2_CLASSES = {0: "car", 1: "bus", 2: "truck", 3: "pedestrian",
               4: "rider", 5: "bicycle", 6: "traffic light"}

# Dataset2 → COCO class mapping for STOCK model
DS2_TO_COCO = {
    0: 2,   # car → car
    1: 5,   # bus → bus
    2: 7,   # truck → truck
    3: 0,   # pedestrian → person
    4: 0,   # rider → person
    5: 1,   # bicycle → bicycle
    6: 9,   # traffic light → traffic light
}

IOU_THRESHOLDS_FINE = np.arange(0.5, 1.0, 0.05).tolist()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def compute_ap(precisions, recalls):
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

def load_yolo_labels(label_path, img_w, img_h):
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
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes

def collect_val_samples():
    images = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    labels = sorted([f for f in os.listdir(VAL_LABELS_DIR) if f.lower().endswith('.txt')])
    label_stems = {os.path.splitext(f)[0]: f for f in labels}
    samples = []
    for img_file in images:
        stem = os.path.splitext(img_file)[0]
        label_file = label_stems.get(stem)
        if label_file is None: continue
        samples.append({
            "image_path": os.path.join(VAL_IMAGES_DIR, img_file),
            "label_path": os.path.join(VAL_LABELS_DIR, label_file),
            "filename": img_file,
        })
    return samples

# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------
def evaluate_at_iou_threshold(all_predictions, all_gt, iou_threshold, eval_classes):
    gt_by_img_cls = defaultdict(list)
    for gt in all_gt:
        img_idx, eval_cls, x1, y1, x2, y2 = gt[:6]
        gt_by_img_cls[(img_idx, eval_cls)].append({"box": (x1, y1, x2, y2), "matched": False})

    all_predictions_sorted = sorted(all_predictions, key=lambda x: -x[2])
    results = {}
    for eval_cls in eval_classes:
        for key in gt_by_img_cls:
            if key[1] == eval_cls:
                for g in gt_by_img_cls[key]: g["matched"] = False

        cls_preds = [p for p in all_predictions_sorted if p[1] == eval_cls]
        cls_gt_count = sum(1 for g in all_gt if g[1] == eval_cls)

        if cls_gt_count == 0:
            results[eval_cls] = 0.0
            continue

        tp_list = []
        for pred in cls_preds:
            img_idx, _, conf, px1, py1, px2, py2 = pred
            gts = gt_by_img_cls.get((img_idx, eval_cls), [])
            best_iou_val = 0
            best_gt_idx = -1
            for gi, gt_info in enumerate(gts):
                if gt_info["matched"]: continue
                iou_val = iou((px1, py1, px2, py2), gt_info["box"])
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
        results[eval_cls] = compute_ap(precisions, recalls)

    return results

def run_evaluation(samples, model_path, is_stock):
    model = YOLO(model_path)
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, conf=AP_CONF_THRESHOLD, verbose=False, save=False, device="mps")
    torch.mps.synchronize()

    all_predictions = []
    all_gt = []
    inference_times_ms = []

    eval_classes = list(set(DS2_TO_COCO.values())) if is_stock else list(DS2_CLASSES.keys())

    for img_idx, sample in enumerate(samples):
        img = cv2.imread(sample["image_path"])
        img_h, img_w = img.shape[:2]
        
        gt_boxes = load_yolo_labels(sample["label_path"], img_w, img_h)
        for ds2_cls, x1, y1, x2, y2 in gt_boxes:
            if is_stock:
                if ds2_cls not in DS2_TO_COCO: continue
                eval_cls = DS2_TO_COCO[ds2_cls]
            else:
                eval_cls = ds2_cls
            all_gt.append((img_idx, eval_cls, x1, y1, x2, y2, ds2_cls))

        torch.mps.synchronize()
        t0 = time.perf_counter()
        results = model.predict(source=img, conf=AP_CONF_THRESHOLD, verbose=False, save=False, device="mps", imgsz=640)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        inference_times_ms.append((t1 - t0) * 1000.0)

        if (img_idx + 1) % 100 == 0:
            print(f"Processed {img_idx + 1}/{len(samples)} images... Avg Latency: {np.mean(inference_times_ms):.2f}ms")

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if is_stock and cls_id not in eval_classes: continue
                if not is_stock and cls_id not in eval_classes: continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                all_predictions.append((img_idx, cls_id, conf, xyxy[0], xyxy[1], xyxy[2], xyxy[3]))

    # --- Metrics ---
    ap50_by_class = evaluate_at_iou_threshold(all_predictions, all_gt, 0.5, eval_classes)
    
    ap_accum = {cls: [] for cls in eval_classes}
    for iou_thresh in IOU_THRESHOLDS_FINE:
        ap_at_thresh = evaluate_at_iou_threshold(all_predictions, all_gt, iou_thresh, eval_classes)
        for cls in eval_classes: ap_accum[cls].append(ap_at_thresh[cls])
    ap50_95_by_class = {cls: np.mean(ap_accum[cls]) for cls in eval_classes}

    op_predictions = [p for p in all_predictions if p[2] >= CONFIDENCE_THRESHOLD]
    all_preds_sorted = sorted(op_predictions, key=lambda x: -x[2])

    gt_by_img_cls = defaultdict(list)
    for gt in all_gt:
        img_idx, eval_cls, x1, y1, x2, y2, ds2_cls = gt
        gt_by_img_cls[(img_idx, eval_cls)].append({"box": (x1, y1, x2, y2), "matched": False, "ds2_cls": ds2_cls})

    per_class_results = {}
    for eval_cls in eval_classes:
        for key in gt_by_img_cls:
            if key[1] == eval_cls:
                for g in gt_by_img_cls[key]: g["matched"] = False

        cls_preds = [p for p in all_preds_sorted if p[1] == eval_cls]
        cls_gt_count = sum(1 for g in all_gt if g[1] == eval_cls)

        tp = fp = 0
        for pred in cls_preds:
            img_idx, _, conf, px1, py1, px2, py2 = pred
            gts = gt_by_img_cls.get((img_idx, eval_cls), [])
            best_iou_val, best_gt_idx = 0, -1
            for gi, gt_info in enumerate(gts):
                if gt_info["matched"]: continue
                iou_val = iou((px1, py1, px2, py2), gt_info["box"])
                if iou_val > best_iou_val:
                    best_iou_val, best_gt_idx = iou_val, gi
            if best_iou_val >= 0.5 and best_gt_idx >= 0:
                tp += 1
                gts[best_gt_idx]["matched"] = True
            else:
                fp += 1
        fn = cls_gt_count - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / cls_gt_count if cls_gt_count > 0 else 0.0

        per_class_results[eval_cls] = {
            "gt": cls_gt_count, "pred": len(cls_preds), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "ap50": round(ap50_by_class[eval_cls], 4), "ap50_95": round(ap50_95_by_class[eval_cls], 4),
        }

    total_tp = sum(r["tp"] for r in per_class_results.values())
    total_fp = sum(r["fp"] for r in per_class_results.values())
    total_fn = sum(r["fn"] for r in per_class_results.values())
    overall = {
        "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0,
        "recall": round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) > 0 else 0.0,
        "mAP50": round(float(np.mean(list(ap50_by_class.values()))), 4),
        "mAP50_95": round(float(np.mean(list(ap50_95_by_class.values()))), 4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "gt": sum(r["gt"] for r in per_class_results.values()),
        "pred_op": len(op_predictions),
        "pred_ap": len(all_predictions),
        "mean_latency": round(float(np.mean(inference_times_ms)), 2),
        "median_latency": round(float(np.median(inference_times_ms)), 2),
        "p95_latency": round(float(np.percentile(inference_times_ms, 95)), 2)
    }

    # Format output by DS2 class names
    final_class_results = {}
    for ds2_id, ds2_name in DS2_CLASSES.items():
        if is_stock:
            eval_cls = DS2_TO_COCO.get(ds2_id)
            if eval_cls is None:
                final_class_results[ds2_name] = None
                continue
            # For pedestrian and rider, they both map to person. We need sub-metrics.
            # So let's recompute them specifically for ds2_id
            sub_tp = sub_fn = sub_gt = 0
            for key in gt_by_img_cls:
                if key[1] == eval_cls:
                    for g in gt_by_img_cls[key]:
                        if g["ds2_cls"] == ds2_id:
                            sub_gt += 1
                            if g["matched"]: sub_tp += 1
            sub_fn = sub_gt - sub_tp
            sub_rec = sub_tp / sub_gt if sub_gt > 0 else 0.0
            # precision, ap are shared for the whole class, so we report N/A or shared
            # we will just assign the person APs, and put the exact TP/FN
            base = per_class_results[eval_cls]
            final_class_results[ds2_name] = {
                "gt": sub_gt, "tp": sub_tp, "fp": "N/A (Shared)", "fn": sub_fn,
                "precision": base["precision"], "recall": round(sub_rec, 4),
                "ap50": base["ap50"], "ap50_95": base["ap50_95"]
            }
        else:
            base = per_class_results[ds2_id]
            final_class_results[ds2_name] = {
                "gt": base["gt"], "tp": base["tp"], "fp": base["fp"], "fn": base["fn"],
                "precision": base["precision"], "recall": base["recall"],
                "ap50": base["ap50"], "ap50_95": base["ap50_95"]
            }

    return {"overall": overall, "per_class": final_class_results}

if __name__ == "__main__":
    samples = collect_val_samples()
    print(f"Collected {len(samples)} samples.")
    stock_res = run_evaluation(samples, "yolov8n.pt", True)
    print("Stock Done.")
    ft_res = run_evaluation(samples, "Stage3-Smoke/runs/smoke/weights/best.pt", False)
    print("FT Done.")
    with open("results_smoke_eval.json", "w") as f:
        json.dump({"stock": stock_res, "ft": ft_res}, f, indent=2)
    print("Saved results_smoke_eval.json")
