"""Bounded causal research adapters; no changes to production tracker defaults."""
from contextlib import contextmanager
import hashlib
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment

from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.shared.boxes import EMPTY, iou_matrix
from adaptive_pipeline.tracking.tracker import Tracker
from analysis.argoverse_hd_phase2 import replay as replay_module
from analysis.argoverse_hd_phase3.observe import install as observe_install

BASE_DIGEST = replay_module.state_digest


def corroborated(current, previous, threshold=.3):
    """One-to-one same-class legacy max-IoU matching; no score alteration."""
    accepted = np.zeros(len(current), bool)
    if not len(current) or not len(previous):
        return accepted
    overlap = iou_matrix(current[:, :4], previous[:, :4])
    same = current[:, 5, None] == previous[None, :, 5]
    overlap[~same] = 0
    a, b = linear_sum_assignment(-overlap)
    accepted[a[same[a, b] & (overlap[a, b] >= threshold)]] = True
    return accepted


class ConfirmedBirthMixin:
    """Store one generation of unmatched weak births, without LK/KF state."""
    def reset(self):
        super().reset()
        self.weak_previous = EMPTY.copy()
        self.weak_next = EMPTY.copy()
        self.weak_previous_frame = None
        self.research_frame = None

    def update_full(self, detections):
        self.weak_next = EMPTY.copy()
        super().update_full(detections)
        self.weak_previous = self.weak_next.copy()
        self.weak_previous_frame = self.research_frame

    def _spawn(self, detections):
        weak = (detections[:, 4] >= self.cfg.association_low_conf) & (detections[:, 4] < self.cfg.track_min_conf)
        self.weak_next = detections[weak].copy()
        previous = self.weak_previous
        # One FULL interval, at most two original frame indices. No SKIP cache.
        if self.weak_previous_frame is None or self.research_frame is None or not 0 < self.research_frame-self.weak_previous_frame <= 2:
            previous = EMPTY.copy()
        accepted = corroborated(detections[weak], previous, self.cfg.assoc_iou)
        keep = ~weak
        keep[weak] = accepted
        if hasattr(self, 'recorder'):
            self.recorder.observe(lambda: self.recorder.events.append({
                'type': 'weak_birth_gate', 'candidates': detections[weak].copy(),
                'accepted': accepted.copy(), 'previous': previous.copy(),
                'frame': self.research_frame}))
        super()._spawn(detections[keep])


def state_digest(tracker):
    base = BASE_DIGEST(tracker)
    if not isinstance(tracker, ConfirmedBirthMixin):
        return base
    h = hashlib.sha256(base.encode())
    h.update(tracker.weak_previous.tobytes())
    h.update(str(tracker.weak_previous.shape).encode())
    h.update(str(tracker.weak_previous_frame).encode())
    return h.hexdigest()


class ResearchPipeline(AdaptivePipeline):
    def reset(self):
        super().reset()
        self.emission_previous = EMPTY.copy()
        self.emission_frame = None

    def step(self, frame, frame_idx):
        self.tracker.research_frame = frame_idx
        output, row = super().step(frame, frame_idx)
        if not getattr(self, 'filter_full_weak', False) or row['decision'] != 'FULL':
            return output, row
        start = time.perf_counter()
        weak = (output[:, 4] >= self.cfg.association_low_conf) & (output[:, 4] < self.cfg.track_min_conf)
        previous = self.emission_previous
        if self.emission_frame is None or not 0 < frame_idx-self.emission_frame <= 2:
            previous = EMPTY.copy()
        accepted = corroborated(output[weak], previous, self.cfg.assoc_iou)
        keep = ~weak
        keep[weak] = accepted
        # Output-only intervention: all raw observations still update tracker.
        self.emission_previous = output[output[:, 4] >= self.cfg.association_low_conf].copy()
        self.emission_frame = frame_idx
        self.last_prediction_details = self.last_prediction_details[keep]
        row['n_out'] = int(keep.sum())
        row['t_total_ms'] += (time.perf_counter()-start)*1e3
        return output[keep], row


def make_pipeline(cfg, cache, spec, observed=False):
    pipe = ResearchPipeline(cfg, cache)
    pipe.filter_full_weak = bool(spec.get('filter_full_weak', False))
    if not observed and spec.get('confirmed_births'):
        cls = type('ConfirmedTracker', (ConfirmedBirthMixin, Tracker), {})
        pipe.tracker = cls(cfg)
        pipe.reset()
    return pipe


@contextmanager
def replay_adapters(spec):
    """Use the existing replay integrity/observation harness unchanged."""
    def factory(cfg, cache):
        return make_pipeline(cfg, cache, spec, observed=True)

    def install(pipe, recorder, forced=None):
        observe_install(pipe, recorder, forced)
        if spec.get('confirmed_births'):
            cls = type('ObservedConfirmedTracker', (ConfirmedBirthMixin, type(pipe.tracker)), {})
            pipe.tracker = cls(pipe.cfg, recorder)
            pipe.reset()

    with patch.object(replay_module, 'AdaptivePipeline', factory), \
         patch.object(replay_module, 'install_observers', install), \
         patch.object(replay_module, 'state_digest', state_digest):
        yield
