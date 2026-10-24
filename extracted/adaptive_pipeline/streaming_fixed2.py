"""Opt-in latest-frame Practical Fixed-2 adapter; frozen algorithms are delegated.

Scheduling counts consumed packets. Tracker prediction/aging retains source-frame
units, certified against original timestamps. Replay pacing never defines motion
units. No pending-frame buffer, detector cache, or waiting policy lives here.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .adaptive_framing.controller import FrameController, FULL
from .pipeline import AdaptivePipeline


@dataclass(frozen=True)
class SourceCadence:
    minimum_ns: int
    maximum_ns: int
    median_ns: float

    @classmethod
    def from_rows(cls, rows):
        if len(rows) < 2 or len({r['source_id'] for r in rows}) != 1:
            raise ValueError('One verified source and at least two timestamps required')
        intervals = []
        for a,b in zip(rows, rows[1:]):
            if b['frame_id']-a['frame_id'] != 1:
                raise ValueError('Cadence certification requires consecutive source frames')
            delta = b['source_timestamp_ns']-a['source_timestamp_ns']
            if delta <= 0:
                raise ValueError('Source timestamps must strictly increase')
            intervals.append(delta)
        return cls(min(intervals), max(intervals), float(np.median(intervals)))

    def consistent(self, gap, elapsed_ns):
        # Bounds come from actual source intervals, not an invented gap cutoff.
        return self.minimum_ns*gap <= elapsed_ns <= self.maximum_ns*gap


class _ConsumedController(FrameController):
    def __init__(self, cfg, context):
        self.context = context
        super().__init__(cfg)

    def tick(self, dt=1):
        super().tick(1)  # Dropped source IDs cannot advance the detector schedule.

    def early_full(self, lum):
        return self.context.force_reasons.copy()

    def decide(self, *args, **kwargs):
        if self.context.force_reasons:
            return FULL, self.context.force_reasons.copy(), [], dict(novelty=np.nan,fail_frac=np.nan,n_need=0)
        return super().decide(*args, **kwargs)


class _GuardedFlow:
    """Validate unchanged LK measurements before any tracker correction."""
    def __init__(self, original, context):
        self.original, self.context = original, context

    @property
    def diagnostics(self):
        return self.original.diagnostics

    def prepare(self, frame):
        return self.original.prepare(frame)

    def propagate(self, prev, gray, boxes, scale):
        c = self.context
        c.flow_attempted = 1
        visible = c.pipe.tracker.visible()
        try:
            new, ok, quality = self.original.propagate(prev, gray, boxes, scale)
        except cv2.error:
            c.force_reasons.append('flow_backend_error')
            return boxes.copy(), np.zeros(len(boxes),bool), np.zeros(len(boxes),np.float32)
        if new.shape != boxes.shape or ok.shape != (len(boxes),) or quality.shape != (len(boxes),):
            raise ValueError('Unexpected flow result schema')
        c.flow_failed_visible = int((visible & ~ok).sum())
        c.flow_valid_visible = int((visible & ok).sum())
        if c.flow_failed_visible:
            c.force_reasons.append('visible_flow_failure')
        valid_geometry = np.isfinite(new).all(axis=1) & (new[:,2:] > new[:,:2]).all(axis=1)
        if ((visible | ok) & (~valid_geometry | ~np.isfinite(quality))).any():
            c.force_reasons.append('invalid_flow_geometry')
        widths = boxes[:,2:]-boxes[:,:2]
        ratios = (new[:,2:]-new[:,:2])/np.maximum(widths,np.finfo(float).tiny)
        # The estimator clamps scale to [.8,1.25]. Equality uses only a small
        # floating-point roundoff tolerance, not a tuned motion threshold.
        eps = 256*np.finfo(np.float64).eps
        clipped = np.isclose(ratios,.8,rtol=0,atol=eps).any(axis=1) | np.isclose(ratios,1.25,rtol=0,atol=eps).any(axis=1)
        c.flow_scale_clipped = int((visible & ok & clipped).sum())
        if c.flow_scale_clipped:
            c.force_reasons.append('flow_scale_clipped')
        if c.force_reasons:
            # Do not partially correct state with a rejected propagation, then
            # associate a fresh detection against that suspect correction.
            return boxes.copy(), np.zeros(len(boxes),bool), np.zeros(len(boxes),np.float32)
        c.flow_accepted = 1
        return new, ok, quality


class StreamingFixed2:
    def __init__(self, cfg, detector, cadence):
        if not (cfg.policy=='fixed' and cfg.fixed_k==2 and cfg.propagation=='flow' and
                cfg.observation_anchoring and cfg.fixed_signal_fastpath):
            raise ValueError('Practical Fixed-2 configuration required')
        self.cfg, self.cadence = cfg, cadence
        self.force_reasons = []
        self.pipe = AdaptivePipeline(cfg, detector)
        self.pipe.controller = _ConsumedController(cfg, self)
        self.pipe.flow = _GuardedFlow(self.pipe.flow, self)
        self.pipe.reset()
        self.previous = None
        self.last_full_id = None
        self.consumed = 0
        self.flow_attempted = self.flow_accepted = 0
        self.flow_failed_visible = self.flow_valid_visible = self.flow_scale_clipped = 0

    @property
    def last_prediction_details(self):
        return self.pipe.last_prediction_details

    def step_packet(self, packet):
        previous = self.previous
        if previous is not None and (packet.source_id!=previous.source_id or
                packet.frame_id<=previous.frame_id or packet.source_timestamp_ns<=previous.source_timestamp_ns):
            raise ValueError('Invalid source identity or nonmonotonic source metadata')
        gap = packet.frame_id-previous.frame_id if previous else 1
        elapsed = packet.source_timestamp_ns-previous.source_timestamp_ns if previous else 0
        self.force_reasons = []
        self.flow_attempted = self.flow_accepted = 0
        self.flow_failed_visible = self.flow_valid_visible = self.flow_scale_clipped = 0
        controller = self.pipe.controller
        nominal_full = controller.since_full is None or controller.since_full+1 >= self.cfg.fixed_k
        reset_reasons = []
        if previous is not None:
            if not self.cadence.consistent(gap, elapsed):
                reset_reasons.append('uncertified_source_cadence')
            if packet.frame.shape != previous.frame.shape:
                reset_reasons.append('source_shape_change')
        # A policy ceiling inherited from k_max, not an LK convergence guarantee.
        if self.last_full_id is not None and packet.frame_id-self.last_full_id >= self.cfg.k_max:
            reset_reasons.append('source_staleness')
        tracker = self.pipe.tracker
        if any(not np.isfinite(getattr(tracker,key)).all() for key in ['X','P','geometry','observation','since_det','score_age']):
            reset_reasons.append('invalid_temporal_state')
        if reset_reasons:
            self.pipe.reset()
            self.force_reasons.extend(reset_reasons)
        elif not nominal_full and (not len(tracker) or not tracker.visible().any() or self.pipe.prev_gray is None):
            self.force_reasons.append('no_usable_visible_state')
        out, rec = self.pipe.step(packet.frame, packet.frame_id)
        if rec['decision'] not in ['FULL','SKIP']:
            raise AssertionError('ROI/local processing is outside this integration')
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite stream output')
        fallback = rec['decision']=='FULL' and not nominal_full
        if self.force_reasons and rec['decision']!='FULL':
            raise AssertionError('Rejected propagation must produce a fresh detection')
        if rec['decision']=='FULL':
            self.last_full_id = packet.frame_id
        self.consumed += 1
        rec.update(consumed_index=self.consumed-1,source_gap=gap,source_elapsed_ms=elapsed/1e6,
            temporal_dt_source_frames=gap,nominal_decision='FULL' if nominal_full else 'SKIP',
            forced_full=int(fallback),fallback_reasons='|'.join(self.force_reasons),
            gap_triggered_full=int(fallback and gap>1),state_reset=int(bool(reset_reasons)),
            state_reset_reasons='|'.join(reset_reasons),flow_attempted=self.flow_attempted,
            flow_accepted=self.flow_accepted,flow_failed_visible=self.flow_failed_visible,
            flow_valid_visible=self.flow_valid_visible,flow_scale_clipped=self.flow_scale_clipped,
            source_frames_since_full=packet.frame_id-self.last_full_id,
            max_track_source_age=float(self.pipe.tracker.since_det.max()) if len(self.pipe.tracker) else 0.)
        # Only metadata/shape are needed next time; retaining the previous input
        # image here would add an unnecessary full-resolution image lifetime.
        from types import SimpleNamespace
        self.previous = SimpleNamespace(source_id=packet.source_id,frame_id=packet.frame_id,
            source_timestamp_ns=packet.source_timestamp_ns,frame=SimpleNamespace(shape=packet.frame.shape))
        return out, rec
