"""ROI re-detection.

Two uses of the same machinery:

1. LOCAL tier (new, default): on a non-keyframe, re-detect only around the
   few tracks the scheduler flagged (flow failure or uncertain + stale).
   This is a cheaper alternative to a full pass, i.e. it *saves* compute.

2. Keyframe refinement (the original Phase-5 idea, optional): on a FULL
   frame, re-detect small low-confidence boxes at higher effective
   resolution. This *adds* compute and changes keyframe outputs, so it is
   an ablation, not part of the cost-saving path.

Fixes vs. original:
  * candidates that the second pass does not recover are KEPT (original
    silently deleted them, and also deleted those beyond the ROI cap);
  * all crops go through ONE batched forward pass, not one call per ROI;
  * detections touching an interior crop edge are discarded (truncated
    objects otherwise produce partial boxes that survive NMS as FPs);
  * the conf_threshold argument was ignored; config values were not wired;
  * package-relative imports (original `import config` broke as a package).
"""
import numpy as np

from ..shared.boxes import EMPTY, iou_matrix, nms_classaware


def build_rois(boxes, frame_shape, margin, min_size):
    h, w = frame_shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, 4), np.int64)
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    ew = np.maximum(bw * (1 + 2 * margin), min_size)
    eh = np.maximum(bh * (1 + 2 * margin), min_size)
    rois = np.stack([cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2], 1)
    rois = np.round(rois).astype(np.int64)
    rois[:, [0, 2]] = np.clip(rois[:, [0, 2]], 0, w)
    rois[:, [1, 3]] = np.clip(rois[:, [1, 3]], 0, h)
    return rois[(rois[:, 2] - rois[:, 0] >= 8) & (rois[:, 3] - rois[:, 1] >= 8)]


def merge_rois(rois, merge_iou):
    rois = [list(r) for r in rois]
    changed = True
    while changed and len(rois) > 1:
        changed = False
        for i in range(len(rois)):
            for j in range(i + 1, len(rois)):
                if iou_matrix(np.array([rois[i]], np.float32), np.array([rois[j]], np.float32))[0, 0] >= merge_iou:
                    a, b = rois[i], rois[j]
                    rois[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    rois.pop(j)
                    changed = True
                    break
            if changed:
                break
    return np.array(rois, np.int64).reshape(-1, 4)


def redetect(detector, frame, rois, cfg, preprocess=None):
    """Batched detection on crops; returns global-coordinate (N,6), ms, n_crops."""
    if len(rois) == 0:
        return EMPTY.copy(), 0.0, 0
    H, W = frame.shape[:2]
    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in rois]
    if preprocess is not None:
        crops = [preprocess(c) for c in crops]
    det_list, ms = detector.detect_batch(crops, cfg.roi_imgsz)
    b = cfg.crop_border_px
    out = []
    for (x1, y1, x2, y2), d in zip(rois, det_list):
        if len(d) == 0:
            continue
        d = d.copy()
        d[:, [0, 2]] += x1
        d[:, [1, 3]] += y1
        touch = (((d[:, 0] <= x1 + b) & (x1 > 0)) | ((d[:, 1] <= y1 + b) & (y1 > 0)) |
                 ((d[:, 2] >= x2 - b) & (x2 < W)) | ((d[:, 3] >= y2 - b) & (y2 < H)))
        out.append(d[~touch])
    dets = np.concatenate(out) if out else EMPTY.copy()
    return nms_classaware(dets, cfg.fusion_nms_iou), ms, len(crops)


def refine_keyframe(detector, frame, dets, cfg, preprocess=None):
    """Original selective re-detection, corrected. Returns (dets, ms, n_crops)."""
    H, W = frame.shape[:2]
    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    cand = (dets[:, 4] >= cfg.tau_low) & (dets[:, 4] < cfg.tau_high) & (area <= cfg.refine_alpha * W * H)
    if not cand.any():
        return dets, 0.0, 0
    c = dets[cand]
    c = c[np.argsort(-c[:, 4])][: cfg.refine_max_rois]
    rois = merge_rois(build_rois(c[:, :4], frame.shape, cfg.roi_margin, cfg.roi_min_size), cfg.roi_merge_iou)
    sec, ms, n = redetect(detector, frame, rois, cfg, preprocess)
    fused = nms_classaware(np.concatenate([dets, sec]), cfg.fusion_nms_iou)
    return fused, ms, n
