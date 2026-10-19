"""Cache-only execution and observational wrappers. Never imports a model."""
import csv
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from adaptive_pipeline.adaptive_framing.controller import FrameController
from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.tracking.tracker import Tracker
from .common import archive, clean, dump, sha

STATE_FIELDS = ('X', 'P', 'ids', 'cls', 'conf_det', 'since_det', 'missed',
                'fail_streak', 'geometry', 'observation', 'flow_quality',
                'flow_innovation', 'score_age')


def state_digest(tracker):
    h = hashlib.sha256()
    for name in STATE_FIELDS:
        a = getattr(tracker, name)
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_rows(tracker):
    if not len(tracker):
        return np.zeros((0, 15))
    return np.column_stack([tracker.ids, tracker.cls, tracker.conf_det, tracker.since_det,
        tracker.missed, tracker.fail_streak, tracker.boxes(), tracker.flow_quality,
        tracker.flow_innovation, tracker.uncertainty(), tracker.score_age,
        np.diagonal(tracker.P, axis1=1, axis2=2)[:, :4].max(1)])


class Recorder:
    def begin(self, frame):
        self.frame = frame
        self.events = []
        self.flow_called = False
        self.flow_points = 0
        self.diag_wall_ns = self.diag_cpu_ns = 0
        self.shadow = None
        self.shadow_eligible = False

    def observe(self, callback):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        result = callback()
        self.diag_wall_ns += time.perf_counter_ns() - wall
        self.diag_cpu_ns += time.process_time_ns() - cpu
        return result


class CacheDetector:
    """Each lookup must be explicitly bound to the current validated image key."""
    def __init__(self, cfg, predictions, recorder):
        self.cfg, self.predictions, self.recorder = cfg, predictions, recorder
        allowed = set(cfg.class_map.values())
        for key, array in predictions.items():
            a = np.asarray(array)
            if a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all():
                raise ValueError(f'Invalid cache array: {key}')
            if ((a[:, 4]<0) | (a[:, 4]>1)).any() or (a[:, 2:4]<a[:, :2]).any():
                raise ValueError(f'Invalid cache score or extent: {key}')
            if (a[:, 5] != a[:, 5].astype(int)).any() or not set(a[:, 5].astype(int)).issubset(allowed):
                raise ValueError(f'Invalid cache taxonomy: {key}')
        self.key = None
        self.calls = 0
        self.pipe = None

    def bind(self, key):
        if key not in self.predictions:
            raise KeyError(f'Missing cached frame: {key}')
        self.key = key
        self.calls_this_frame = 0

    def detect(self, frame):
        if self.key is None:
            raise RuntimeError('Cache lookup without image binding')
        if self.calls_this_frame:
            raise RuntimeError('Multiple detector requests on one cached frame')
        self.calls_this_frame += 1
        self.calls += 1
        if self.pipe is not None:
            def shadow():
                self.recorder.shadow = self.pipe.tracker.outputs()[0].copy()
                # Fixed+anchored FULL takes an early path without normal flow.
                self.recorder.shadow_eligible = bool(self.cfg.policy == 'adaptive'
                    and self.recorder.frame['frame_index'] > 0)
                self.recorder.events.append({'type': 'pre_FULL', 'tracks': state_rows(self.pipe.tracker)})
            self.recorder.observe(shadow)
        return self.predictions[self.key].copy(), 0.0

    def detect_batch(self, *args, **kwargs):
        raise RuntimeError('ROI/model inference is forbidden in this cache experiment')

    def warmup(self, *args, **kwargs):
        raise RuntimeError('Model warmup is forbidden in this cache experiment')

    def full_gflops(self, h, w):
        s = self.cfg.imgsz / max(h, w)
        return self.cfg.gflops_640 * math.ceil(h*s/32)*32 * math.ceil(w*s/32)*32 / 640**2

    def roi_gflops(self, n):
        raise RuntimeError('ROI work is outside the approved cache experiment')


class ObservedTracker(Tracker):
    def __init__(self, cfg, recorder):
        self.recorder = recorder
        super().__init__(cfg)

    def update_full(self, dets):
        def before():
            self._input_rows = {tuple(map(float, row)): [] for row in dets}
            for i, row in enumerate(dets):
                self._input_rows[tuple(map(float, row))].append(i)
            self._before_ids = self.ids.copy()
            self.recorder.events.append({'type': 'FULL_input', 'n': len(dets),
                'admitted_high': int((dets[:, 4] >= self.cfg.track_min_conf).sum()),
                'low_pool': int(((dets[:, 4] >= self.cfg.association_low_conf)
                                  & (dets[:, 4] < self.cfg.track_min_conf)).sum())})
        self.recorder.observe(before)
        super().update_full(dets)
        self.recorder.observe(lambda: self.recorder.events.append({'type': 'post_FULL',
            'hidden_ids': self.ids[self.missed > 0].copy(),
            'deleted_ids': np.setdiff1d(self._before_ids, self.ids),
            'tracks': state_rows(self)}))

    def _confirm(self, matches, dets):
        def capture():
            for d, t in matches:
                self.recorder.events.append({'type': 'confirm', 'track_id': int(self.ids[t]),
                    'source_rows': self._input_rows[tuple(map(float, dets[d]))],
                    'detection': dets[d].copy(), 'previous_missed': int(self.missed[t]),
                    'previous_age': float(self.since_det[t]),
                    'low': bool(dets[d, 4] < self.cfg.track_min_conf)})
        self.recorder.observe(capture)
        super()._confirm(matches, dets)

    def _spawn(self, dets):
        def capture():
            for i, d in enumerate(dets):
                self.recorder.events.append({'type': 'spawn', 'track_id': int(self._next_id+i),
                    'source_rows': self._input_rows[tuple(map(float, d))], 'detection': d.copy()})
        self.recorder.observe(capture)
        super()._spawn(dets)


class ObservedController(FrameController):
    def __init__(self, cfg, recorder, forced=None):
        self.recorder, self.forced = recorder, forced
        super().__init__(cfg)

    def decide(self, *args, **kwargs):
        natural = super().decide(*args, **kwargs)
        def capture():
            tracker = args[3]
            self.recorder.events.append({'type': 'pre_decision', 'natural_decision': natural[0],
                'reasons': natural[1], 'signals': natural[3], 'tracks': state_rows(tracker),
                'flow_ok': np.asarray(args[4]).copy()})
        self.recorder.observe(capture)
        if self.forced is None:
            return natural
        row = self.recorder.frame
        action = 'FULL' if row['frame_index'] in self.forced[row['video']] else 'SKIP'
        return action, ['forced_schedule'] if action == 'FULL' else [], [], natural[3]


def install_observers(pipe, recorder, forced=None):
    pipe.tracker = ObservedTracker(pipe.cfg, recorder)
    pipe.controller = ObservedController(pipe.cfg, recorder, forced)
    original = pipe.flow.propagate
    def propagate(prev, gray, boxes, scale):
        result = original(prev, gray, boxes, scale)
        def capture():
            recorder.flow_called = True
            recorder.flow_points = len(boxes)*pipe.cfg.flow_grid**2
            recorder.events.append({'type': 'flow', 'track_ids': pipe.tracker.ids.copy(),
                'previous_boxes': boxes.copy(), 'measured_boxes': result[0].copy(),
                'ok': result[1].copy(), 'quality': result[2].copy(),
                'diagnostics': {k: np.asarray(v).copy() for k, v in pipe.flow.diagnostics.items()}})
        recorder.observe(capture)
        return result
    pipe.flow.propagate = propagate
    pipe.reset()


def replay(cfg, plan, dense, out, reference=None, forced=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'config.json', cfg.to_dict())
    recorder = Recorder(); cache = CacheDetector(cfg, dense, recorder)
    pipe = AdaptivePipeline(cfg, cache); cache.pipe = pipe
    install_observers(pipe, recorder, forced)
    old_arrays = archive(reference/'preds.npz') if reference else None
    if reference:
        with (reference/'frames.csv').open() as f:
            old_rows = list(csv.DictReader(f))
        old_details = archive(reference/'prediction_details.npz')
    rows, preds, details, states, shadows, digests = [], {}, {}, {}, {}, {}
    last_video = None
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    with gzip.open(out/'events.jsonl.gz', 'xt') as event_file:
        for index, frameinfo in enumerate(plan):
            if frameinfo['video'] != last_video:
                pipe.reset(); last_video = frameinfo['video']
                print(f"  {out.name}: {last_video}", flush=True)
            if sha(frameinfo['image']) != frameinfo['image_sha256']:
                raise ValueError(f'Image bytes changed: {frameinfo["key"]}')
            td = time.perf_counter_ns()
            image = cv2.imread(frameinfo['image'])
            decode_ms = (time.perf_counter_ns()-td)/1e6
            if image is None or image.shape != (1200, 1920, 3):
                raise ValueError(f'Unexpected image dimensions: {frameinfo["key"]}')
            recorder.begin(frameinfo); cache.bind(frameinfo['key'])
            tw, tc = time.perf_counter_ns(), time.process_time_ns()
            result, rec = pipe.step(image, frameinfo['frame_index'])
            step_wall, step_cpu = time.perf_counter_ns()-tw, time.process_time_ns()-tc
            key = frameinfo['key']
            if reference:
                if not np.array_equal(result, old_arrays[key]):
                    dump(out/'PARITY_FAILURE.json', {'frame': frameinfo, 'kind': 'prediction',
                        'new_shape': result.shape, 'old_shape': old_arrays[key].shape,
                        'max_abs_difference': float(np.abs(result-old_arrays[key]).max()) if result.shape==old_arrays[key].shape and result.size else None})
                    raise AssertionError(f'Prediction parity failed at {key}; dependent experiments prohibited')
                if rec['decision'] != old_rows[index]['decision'] or not np.array_equal(pipe.last_prediction_details, old_details[key]):
                    raise AssertionError(f'Decision/detail parity failed at {key}')
                for field in ['n_tracks', 'n_out', 'since_full']:
                    if rec[field] != int(old_rows[index][field]):
                        raise AssertionError(f'Telemetry parity failed: {key}/{field}')
            if forced and (rec['decision']=='FULL') != (frameinfo['frame_index'] in forced[last_video]):
                raise AssertionError('Forced schedule was not followed')
            preds[key] = result.copy(); details[key] = pipe.last_prediction_details.copy()
            states[key] = state_rows(pipe.tracker); digests[key] = state_digest(pipe.tracker)
            if recorder.shadow is not None and recorder.shadow_eligible:
                shadows[key] = recorder.shadow
            event_file.write(json.dumps(clean({'key': key, 'decision': rec['decision'], 'events': recorder.events}), separators=(',', ':'), allow_nan=False)+'\n')
            rec.update(video=last_video, image_id=frameinfo['image_id'], decode_ms=decode_ms,
                step_wall_with_diagnostics_ms=step_wall/1e6, step_cpu_with_diagnostics_ms=step_cpu/1e6,
                diagnostic_wall_ms=recorder.diag_wall_ns/1e6, diagnostic_cpu_ms=recorder.diag_cpu_ns/1e6,
                corrected_step_wall_ms=(step_wall-recorder.diag_wall_ns)/1e6,
                corrected_step_cpu_ms=(step_cpu-recorder.diag_cpu_ns)/1e6,
                flow_called=int(recorder.flow_called), flow_points=recorder.flow_points)
            rows.append(rec)
    np.savez_compressed(out/'preds.npz', **preds)
    np.savez_compressed(out/'prediction_details.npz', **details)
    np.savez_compressed(out/'track_states.npz', **states)
    np.savez_compressed(out/'pre_full_shadow.npz', **shadows)
    dump(out/'state_hashes.json', digests)
    with (out/'frames.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {'frames': len(rows), 'simulated_FULL_requests': cache.calls, 'new_YOLO_inferences': 0,
        'reference_bitwise_parity': bool(reference), 'reference': str(reference) if reference else None,
        'backend': {'images': 'OpenCV CPU', 'flow': 'OpenCV CPU sparse LK' if cfg.propagation=='flow' else 'not executed',
                    'tracker': 'NumPy/SciPy CPU', 'detector': 'saved NumPy cache; inference not executed'},
        'stage_wall_seconds_including_decode_hash_diagnostics_IO': time.perf_counter()-wall_start,
        'stage_process_seconds_including_decode_hash_diagnostics_IO': time.process_time()-cpu_start,
        'timing_scope': 'cache replay component times; no model, not deployed pipeline FPS',
        'mean_post_frame_tracks': float(np.mean([r['n_tracks'] for r in rows])),
        'max_post_frame_tracks': max(r['n_tracks'] for r in rows),
        'total_flow_point_evaluations_before_FB': sum(r['flow_points'] for r in rows),
        'mean_corrected_step_wall_ms': float(np.mean([r['corrected_step_wall_ms'] for r in rows])),
        'mean_corrected_step_cpu_ms': float(np.mean([r['corrected_step_cpu_ms'] for r in rows])),
        'mean_diagnostic_wall_ms': float(np.mean([r['diagnostic_wall_ms'] for r in rows])),
        'mean_decode_ms': float(np.mean([r['decode_ms'] for r in rows])),
        'flow_frames': sum(r['flow_called'] for r in rows),
        'state_columns': ['track_id','class','confidence','age','missed_FULL','flow_fail_streak',
                         'x1','y1','x2','y2','flow_quality','flow_innovation','uncertainty','score_age','max_position_variance']}
    dump(out/'replay_summary.json', summary)
    dump(out/'REPLAY_COMPLETE.json', {'predictions_sha256': sha(out/'preds.npz'), 'frames': len(rows),
        'reference_bitwise_parity': bool(reference), 'new_YOLO_inferences': 0})
    return summary
