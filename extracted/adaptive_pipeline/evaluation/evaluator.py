"""Detection metrics that mirror ultralytics' DetectionValidator.

Mirrors: match_predictions (greedy by IoU, class-aware, 10 IoU thresholds),
ap_per_class (101-point interpolated AP, P/R at max smoothed mean-F1).
Re-implemented rather than imported so the protocol is fixed in this repo
and does not shift with ultralytics versions.

Sanity requirement: running preset `baseline` through this evaluator must
reproduce the frozen numbers (mAP50 0.9843, mAP50-95 0.9749, P 0.9779,
R 0.9752) to within ~0.002. If it does not, the class map, imgsz, conf,
half precision or label conversion differ from the frozen run, and no
comparison is valid until that is fixed.

Per-image tags allow scoring subsets: by decision (FULL/LOCAL/SKIP) and
by staleness (frames since last full detection).
"""
import numpy as np

from ..shared.boxes import iou_matrix

IOUV = np.linspace(0.5, 0.95, 10)


def load_yolo_labels(path, w, h):
    try:
        a = np.loadtxt(path, ndmin=2, dtype=np.float32)
    except (OSError, ValueError):
        return np.zeros((0, 5), np.float32)
    if a.size == 0:
        return np.zeros((0, 5), np.float32)
    a = a[:, :5]
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return np.stack([a[:, 0], cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)


def match_predictions(pred_cls, true_cls, iou):
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou * (true_cls[:, None] == pred_cls[None, :])
    for i, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], i] = True
    return correct


def _smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


# AP_PROTOCOL must match the ultralytics version that produced the frozen baseline:
#   "current": ultralytics 8.4.x -- precision drops to 0 right after max recall
#   "legacy" : older 8.x         -- linear ramp from (r_max, p) to (1, 0); inflates AP
AP_PROTOCOL = "current"


def _compute_ap(recall, precision):
    if AP_PROTOCOL == "legacy":
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
    else:
        mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return trapz(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls, fixed_conf=0.25, eps=1e-16):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique)
    x = np.linspace(0, 1, 1000)
    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))
    for ci, c in enumerate(unique):
        m = pred_cls == c
        if m.sum() == 0 or nt[ci] == 0:
            continue
        fpc = (1 - tp[m]).cumsum(0)
        tpc = tp[m].cumsum(0)
        recall = tpc / (nt[ci] + eps)
        precision = tpc / (tpc + fpc)
        r_curve[ci] = np.interp(-x, -conf[m], recall[:, 0], left=0)
        p_curve[ci] = np.interp(-x, -conf[m], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = _compute_ap(recall[:, j], precision[:, j])
    f1 = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    k = int(_smooth(f1.mean(0), 0.1).argmax())
    kf = int(np.searchsorted(x, fixed_conf))
    return {
        "p": p_curve[:, k], "r": r_curve[:, k], "ap": ap, "classes": unique.astype(int),
        "best_conf": float(x[k]), "p_fixed": p_curve[:, kf], "r_fixed": r_curve[:, kf],
    }


class DetectionEvaluator:
    def __init__(self, names):
        self.names = names
        self.tp, self.conf, self.pcls, self.tcls, self.tags = [], [], [], [], []

    def add(self, preds, gts, tags=None):
        """preds (N,6) xyxy,conf,cls ; gts (M,5) cls,xyxy (pixels)."""
        n = len(preds)
        tcls = gts[:, 0].astype(int)
        if n == 0:
            tp = np.zeros((0, len(IOUV)), bool)
        elif len(gts) == 0:
            tp = np.zeros((n, len(IOUV)), bool)
        else:
            tp = match_predictions(preds[:, 5].astype(int), tcls, iou_matrix(gts[:, 1:5], preds[:, :4]))
        self.tp.append(tp)
        self.conf.append(preds[:, 4].astype(np.float32))
        self.pcls.append(preds[:, 5].astype(int))
        self.tcls.append(tcls)
        self.tags.append(tags or {})

    def save(self, path):
        """Per-image matching stats, so runs can be compared offline (paired bootstrap)."""
        n_pred = np.array([len(c) for c in self.conf])
        n_gt = np.array([len(t) for t in self.tcls])
        np.savez_compressed(
            path, tp=np.concatenate(self.tp) if self.tp else np.zeros((0, 10), bool),
            conf=np.concatenate(self.conf), pcls=np.concatenate(self.pcls), tcls=np.concatenate(self.tcls),
            n_pred=n_pred, n_gt=n_gt,
            video=np.array([t.get("video", "") for t in self.tags]),
            decision=np.array([t.get("decision", "") for t in self.tags]))

    def compute(self, where=None):
        idx = [i for i, t in enumerate(self.tags) if where is None or where(t)]
        if not idx:
            return None
        tcls = np.concatenate([self.tcls[i] for i in idx])
        if len(tcls) == 0:
            return None
        res = ap_per_class(np.concatenate([self.tp[i] for i in idx]),
                           np.concatenate([self.conf[i] for i in idx]),
                           np.concatenate([self.pcls[i] for i in idx]), tcls)
        ap = res["ap"]
        return {
            "images": len(idx), "instances": int(len(tcls)),
            "precision": float(res["p"].mean()), "recall": float(res["r"].mean()),
            "mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()),
            "precision@0.25": float(res["p_fixed"].mean()), "recall@0.25": float(res["r_fixed"].mean()),
            "best_f1_conf": res["best_conf"],
            "per_class_AP50-95": {self.names[c] if c < len(self.names) else str(c): float(ap[i].mean())
                                  for i, c in enumerate(res["classes"])},
        }


# ── paired bootstrap over videos ────────────────────────────────────────────
def _load_stats(path):
    z = np.load(path)
    po = np.concatenate([[0], np.cumsum(z["n_pred"])])
    go = np.concatenate([[0], np.cumsum(z["n_gt"])])
    vids = z["video"]
    per_video = {}
    for i, v in enumerate(vids):
        per_video.setdefault(v, []).append(i)
    return z, po, go, per_video


def _map_for_images(z, po, go, imgs):
    pi = np.concatenate([np.arange(po[i], po[i + 1]) for i in imgs])
    gi = np.concatenate([np.arange(go[i], go[i + 1]) for i in imgs])
    r = ap_per_class(z["tp"][pi], z["conf"][pi], z["pcls"][pi], z["tcls"][gi])
    return r["ap"][:, 0].mean(), r["ap"].mean()


def paired_bootstrap(stats_a, stats_b, n_boot=200, seed=0):
    """Delta (B - A) of mAP50 and mAP50-95 with 95% CI, resampling VIDEOS
    (frames within a video are not independent)."""
    za, poa, goa, va = _load_stats(stats_a)
    zb, pob, gob, vb = _load_stats(stats_b)
    vids = sorted(set(va) & set(vb))
    rng = np.random.default_rng(seed)
    point_a = _map_for_images(za, poa, goa, [i for v in vids for i in va[v]])
    point_b = _map_for_images(zb, pob, gob, [i for v in vids for i in vb[v]])
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(len(vids), len(vids), replace=True)
        ia = [i for k in s for i in va[vids[k]]]
        ib = [i for k in s for i in vb[vids[k]]]
        a = _map_for_images(za, poa, goa, ia)
        b = _map_for_images(zb, pob, gob, ib)
        deltas.append((b[0] - a[0], b[1] - a[1]))
    d = np.array(deltas)
    return {
        "videos": len(vids),
        "delta_mAP50": float(point_b[0] - point_a[0]),
        "delta_mAP50_ci95": [float(x) for x in np.percentile(d[:, 0], [2.5, 97.5])],
        "delta_mAP50-95": float(point_b[1] - point_a[1]),
        "delta_mAP50-95_ci95": [float(x) for x in np.percentile(d[:, 1], [2.5, 97.5])],
    }
