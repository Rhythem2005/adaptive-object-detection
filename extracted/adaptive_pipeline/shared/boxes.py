"""Vectorised box utilities. Detections everywhere are float32 (N,6):
[x1, y1, x2, y2, conf, cls]."""
import numpy as np

EMPTY = np.zeros((0, 6), np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = a[:, None, :4].astype(np.float32)
    b = b[None, :, :4].astype(np.float32)
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms_classaware(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(dets) <= 1:
        return dets
    keep = []
    for c in np.unique(dets[:, 5]):
        idx = np.where(dets[:, 5] == c)[0]
        idx = idx[np.argsort(-dets[idx, 4], kind="stable")]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            ious = iou_matrix(dets[i:i + 1], dets[idx[1:]])[0]
            idx = idx[1:][ious < iou_thr]
    keep = np.array(keep)
    keep = keep[np.argsort(-dets[keep, 4], kind="stable")]
    return dets[keep]


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out
