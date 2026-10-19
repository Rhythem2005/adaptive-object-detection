"""Optional bounded verification using existing OpenCV/NumPy operations.

Background alignment is used for novelty only. Raw LK motion already contains
camera motion, so the background transform is never added to boxes a second time.
Appearance similarity is evidence of change, not semantic object confirmation.
"""
import cv2
import numpy as np


class MotionCompensator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.key_transform = np.eye(3)
        self.valid = True
        self.inlier_fraction = 0.0

    def update(self, previous, current, boxes, scale):
        if previous is None or previous.shape != current.shape or not self.valid:
            self.valid = False
            return False
        mask = np.full(previous.shape, 255, np.uint8)
        h, w = previous.shape
        for b in boxes * scale:
            x1, y1 = np.maximum(np.floor(b[:2]).astype(int) - 2, 0)
            x2, y2 = np.minimum(np.ceil(b[2:]).astype(int) + 2, [w, h])
            mask[y1:y2, x1:x2] = 0
        points = cv2.goodFeaturesToTrack(previous, self.cfg.motion_max_points, 0.01, 5, mask=mask)
        if points is None or len(points) < self.cfg.motion_min_points:
            self.valid = False
            return False
        forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if forward is None or status is None:
            self.valid = False
            return False
        backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None)
        if backward is None or reverse_status is None:
            self.valid = False
            return False
        fb = np.linalg.norm((points - backward).reshape(-1, 2), axis=1)
        good = (status.ravel() > 0) & (reverse_status.ravel() > 0) & (fb <= self.cfg.flow_fb_max)
        if good.sum() < self.cfg.motion_min_points:
            self.valid = False
            return False
        transform, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC,
                                                        maxIters=200, confidence=0.99, refineIters=5)
        self.inlier_fraction = float(np.mean(inliers)) if inliers is not None else 0.0
        if transform is None or not np.isfinite(transform).all() or self.inlier_fraction < self.cfg.motion_min_inlier_fraction:
            self.valid = False
            return False
        size_change = np.sqrt(abs(np.linalg.det(transform[:, :2])))
        if not 0.7 <= size_change <= 1.4:
            self.valid = False
            return False
        affine = np.eye(3)
        affine[:2] = transform
        self.key_transform = affine @ self.key_transform
        return True

    def novelty(self, key_thumb, thumb, boxes, thumb_scale, flow_width):
        if not self.valid:
            return None
        ratio = thumb.shape[1] / float(flow_width)
        transform = self.key_transform[:2].copy()
        transform[:, 2] *= ratio
        shape = (thumb.shape[1], thumb.shape[0])
        aligned = cv2.warpAffine(key_thumb, transform, shape)
        coverage = cv2.warpAffine(np.ones_like(key_thumb), transform, shape, flags=cv2.INTER_NEAREST).astype(bool)
        if coverage.mean() < self.cfg.motion_min_coverage:
            self.valid = False
            return None
        h, w = thumb.shape
        for box in boxes * thumb_scale:
            x1, y1 = np.maximum(np.floor(box[:2]).astype(int) - 1, 0)
            x2, y2 = np.minimum(np.ceil(box[2:]).astype(int) + 1, [w, h])
            coverage[y1:y2, x1:x2] = False
        if coverage.sum() < 0.05 * coverage.size:
            return 0.0
        changed = cv2.absdiff(thumb, aligned) > self.cfg.novelty_pix_thr
        return float((changed & coverage).sum() / coverage.sum())


class AppearanceVerifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.references = {}

    def _patch(self, gray, box):
        h, w = gray.shape
        x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
        x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        n = self.cfg.appearance_patch_size
        patch = cv2.resize(gray[y1:y2, x1:x2], (n, n), interpolation=cv2.INTER_AREA).astype(float).ravel()
        patch -= patch.mean()
        norm = np.linalg.norm(patch)
        return patch / norm if norm > 1e-6 else None

    def anchor(self, gray, boxes, ids, scale, confirmed=None):
        active = set(map(int, ids))
        self.references = {i: p for i, p in self.references.items() if i in active}
        for index, (box, ident) in enumerate(zip(boxes * scale, ids)):
            if confirmed is None or confirmed[index]:
                self.references[int(ident)] = self._patch(gray, box)

    def verify(self, gray, boxes, ids, scale, quality):
        adjusted = np.asarray(quality, dtype=float).copy()
        checked = failed = 0
        for i in np.argsort(adjusted)[:self.cfg.appearance_max_tracks]:
            reference = self.references.get(int(ids[i]))
            patch = self._patch(gray, boxes[i] * scale)
            if reference is None or patch is None:
                continue  # A flat patch supplies no positive or negative evidence.
            checked += 1
            if float(reference @ patch) < self.cfg.appearance_min_correlation:
                adjusted[i] = min(adjusted[i], self.cfg.risk_flow_quality / 2)
                failed += 1
        return adjusted, checked, failed
