"""Synthetic-only correctness checks: no dataset, videos, weights or real YOLO.

Run from the repository root:
  PYTHONPATH=extracted python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from adaptive_pipeline.adaptive_framing.budget import DetectorBudget, LocalCostGate
from adaptive_pipeline.adaptive_framing.controller import FrameController, FULL, LOCAL, SKIP
from adaptive_pipeline.evaluation.evaluator import DetectionEvaluator
from adaptive_pipeline.evaluation.evaluator import paired_bootstrap
from adaptive_pipeline.experiments.compare import verify_comparison
from adaptive_pipeline.evaluation.provenance import assert_fresh_run, input_manifest
from adaptive_pipeline.experiments.presets import PRESETS
from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.run import build_config
from adaptive_pipeline.shared.config import PipelineConfig
from adaptive_pipeline.tracking.flow import FlowPropagator
from adaptive_pipeline.tracking.tracker import Tracker
from adaptive_pipeline.tracking.verification import AppearanceVerifier, MotionCompensator


def detections(*boxes):
    return np.asarray(boxes, np.float32).reshape(-1, 6)


BOX = [100, 100, 140, 140, .9, 0]


class FakeDetector:
    def __init__(self, full=None, crops=None):
        self.full = full or [detections(BOX)]
        self.crops = crops or [detections([28, 28, 68, 68, .9, 0])]
        self.full_calls = self.roi_calls = 0

    def detect(self, frame):
        value = self.full[min(self.full_calls, len(self.full) - 1)]
        self.full_calls += 1
        return value.copy(), 10.0

    def detect_batch(self, crops, imgsz):
        self.roi_calls += 1
        return [self.crops[min(i, len(self.crops) - 1)].copy() for i in range(len(crops))], 2.0

    def full_gflops(self, h, w):
        return 5.22

    def roi_gflops(self, n):
        return n * 2.175


class FakeFlow:
    def __init__(self, dx=0, quality=1, ok=True):
        self.dx, self.quality, self.ok = dx, quality, ok
        self.calls = 0

    def prepare(self, frame):
        return np.zeros((64, 96), np.uint8), 96 / frame.shape[1]

    def propagate(self, prev, current, boxes, scale):
        self.calls += 1
        b = boxes.copy()
        b[:, [0, 2]] += self.dx
        return b, np.full(len(b), self.ok, bool), np.full(len(b), self.quality)


def pipeline(preset='adaptive_safe', **overrides):
    cfg = build_config(preset, []).with_overrides(**overrides)
    det = FakeDetector()
    pipe = AdaptivePipeline(cfg, det)
    pipe.flow = FakeFlow()
    return pipe, det


class TrackerTests(unittest.TestCase):
    def test_detector_observation_is_exact_after_prediction(self):
        t = Tracker(PipelineConfig(observation_anchoring=True))
        t.update_full(detections(BOX))
        t.predict(); t.tick()
        shifted = detections([104, 100, 144, 140, .8, 0])
        t.update_full(shifted)
        np.testing.assert_array_equal(t.boxes(), shifted[:, :4])
        np.testing.assert_allclose(t.X[0, 4:], [4, 0, 0, 0])
        t.predict()
        np.testing.assert_allclose(t.boxes()[0], [108, 100, 148, 140])

    def test_flow_measurement_is_not_smoothed_in_output(self):
        t = Tracker(PipelineConfig(observation_anchoring=True, quality_aware_flow=True))
        t.update_full(detections(BOX)); t.predict()
        b = np.array([[103, 100, 143, 140.]])
        t.apply_flow(b, np.array([True]), np.array([.9]))
        np.testing.assert_array_equal(t.boxes(), b)
        self.assertGreater(t.flow_innovation[0], 0)

    def test_quality_decay_retains_scores_but_failed_flow_penalizes_them(self):
        t = Tracker(PipelineConfig(quality_conf_decay=True, conf_decay=.93))
        t.update_full(detections(BOX))
        t.apply_flow(t.boxes(), np.array([True]), np.array([1.])); t.tick()
        healthy = t.outputs()[0][0, 4]
        self.assertGreater(healthy, .9 * .93)
        t.apply_flow(t.boxes(), np.array([False]), np.array([1.])); t.tick()
        self.assertEqual(t.flow_quality[0], 0)
        self.assertLess(t.outputs()[0][0, 4], healthy)

    def test_low_score_association_recovers_but_never_spawns(self):
        t = Tracker(PipelineConfig(low_score_association=True, observation_anchoring=True))
        t.update_full(detections(BOX)); ident = t.ids[0]
        t.tick()
        t.update_full(detections([102, 100, 142, 140, .2, 0], [400, 400, 440, 440, .2, 0]))
        self.assertEqual(len(t), 1); self.assertEqual(t.ids[0], ident)
        self.assertEqual(t.missed[0], 0)
        np.testing.assert_allclose(t.boxes()[0], [102, 100, 142, 140])

    def test_confirmed_local_geometry_and_bounded_births(self):
        t = Tracker(PipelineConfig(observation_anchoring=True, local_births=True, local_max_births=1))
        t.update_full(detections(BOX)); t.predict(); t.tick()
        d = detections([102, 100, 142, 140, .8, 0], [200, 100, 240, 140, .9, 0],
                       [300, 100, 340, 140, .7, 0])
        self.assertEqual(t.update_local(d, [0]), 0)
        self.assertEqual(len(t), 2)
        np.testing.assert_array_equal(t.boxes()[0], d[0, :4])

    def test_state_arrays_stay_aligned_on_deletion_and_reset(self):
        t = Tracker(PipelineConfig(observation_anchoring=True, max_missed_full=0))
        t.update_full(detections(BOX, [300, 300, 340, 340, .9, 0]))
        t.update_full(detections(BOX))
        for name in ['X', 'P', 'geometry', 'observation', 'score_age', 'flow_quality', 'flow_innovation']:
            self.assertEqual(len(getattr(t, name)), 1)
        t.reset(); self.assertEqual(len(t), 0); self.assertEqual(len(t.geometry), 0)

    def test_variable_trust_covariance_remains_symmetric_positive(self):
        t = Tracker(PipelineConfig(observation_anchoring=True, quality_aware_flow=True))
        t.update_full(detections(BOX))
        for i in range(12):
            t.predict(); b = t.boxes(); b[:, [0, 2]] += 1
            t.apply_flow(b, np.array([True]), np.array([.2 if i % 2 else .9])); t.tick()
            np.testing.assert_allclose(t.P, np.swapaxes(t.P, 1, 2), atol=1e-10)
            self.assertGreaterEqual(np.linalg.eigvalsh(t.P).min(), -1e-10)


class FlowTests(unittest.TestCase):
    def test_real_lk_on_generated_translation(self):
        rng = np.random.default_rng(9)
        prev = rng.integers(0, 255, (96, 128), dtype=np.uint8)
        cur = cv2.warpAffine(prev, np.float32([[1, 0, 2], [0, 1, 1]]), (128, 96))
        flow = FlowPropagator(PipelineConfig(quality_aware_flow=True))
        b, ok, quality = flow.propagate(prev, cur, np.array([[20., 20., 70., 70.]]), 1.)
        self.assertTrue(ok[0]); self.assertGreater(quality[0], 0)
        self.assertLessEqual(quality[0], flow.diagnostics['valid_fraction'][0])
        np.testing.assert_allclose(b[0], [22, 21, 72, 71], atol=.8)

    def test_no_points_and_lk_failure_are_safe(self):
        f = FlowPropagator(PipelineConfig(quality_aware_flow=True))
        gray = np.zeros((80, 80), np.uint8)
        with patch('cv2.calcOpticalFlowPyrLK', return_value=(None, None, None)):
            b, ok, q = f.propagate(gray, gray, np.array([[10., 10., 30., 30.]]), 1.)
        self.assertFalse(ok[0]); self.assertEqual(q[0], 0); self.assertTrue(np.isfinite(b).all())

    def test_corners_on_flat_image_do_not_claim_reliable_motion(self):
        f = FlowPropagator(PipelineConfig(flow_point_mode='corners', quality_aware_flow=True))
        gray = np.zeros((80, 80), np.uint8)
        _, ok, q = f.propagate(gray, gray, np.array([[10., 10., 50., 50.]]), 1.)
        self.assertFalse(ok[0]); self.assertEqual(q[0], 0)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = build_config('adaptive_safe', [])
        self.t = Tracker(self.cfg)
        self.t.update_full(detections([*BOX[:4], .4, 0]))
        self.t.P[:] = 0; self.t.since_det[:] = 2
        self.c = FrameController(self.cfg)
        self.thumb = np.zeros((90, 160), np.uint8)
        self.c.on_full(self.thumb, 100); self.c.since_full = 2

    def decide(self, **kw):
        return self.c.decide(self.thumb, 100, 1, self.t, np.ones(len(self.t), bool), **kw)

    def test_low_confidence_alone_does_not_force_full(self):
        self.assertEqual(self.decide()[0], SKIP)

    def test_failed_quality_refreshes_even_if_budget_is_exhausted(self):
        self.t.flow_quality[:] = 0
        self.assertEqual(self.decide(budget_allows_full=False)[0], FULL)

    def test_hard_deadline_ignores_budget(self):
        self.c.since_full = self.cfg.k_max
        self.assertEqual(self.decide(budget_allows_full=False)[0], FULL)

    def test_disable_local_still_refreshes_in_risk_mode(self):
        self.t.flow_quality[:] = .2
        action, reasons, _, _ = self.decide()
        self.assertEqual(action, FULL); self.assertIn('risk_refresh', reasons)

    def test_motion_failure_is_conservative(self):
        self.cfg.motion_compensation = True
        self.assertEqual(self.decide(scene_valid=False)[0], FULL)

    def test_budget_can_defer_only_a_soft_concern(self):
        self.t.flow_quality[:] = .6
        action, _, _, sig = self.decide(budget_allows_full=False)
        self.assertEqual(action, SKIP); self.assertEqual(sig['budget_deferred'], 1)


class BudgetTests(unittest.TestCase):
    def test_rolling_budget_uses_evicted_frame_and_actual_reference_cost(self):
        b = DetectorBudget(PipelineConfig(budget_window=2))
        self.assertEqual(b.record(5, 5), 1)
        self.assertEqual(b.record(0, 5), .5)
        self.assertEqual(b.projected_fraction(0, 10), 0)
        self.assertEqual(b.record(0, 10), 0)

    def test_local_gate_rejects_multiple_crops_and_bad_success(self):
        g = LocalCostGate(PipelineConfig())
        self.assertTrue(g.admit(1, 2.175, 5.22)[0])
        self.assertFalse(g.admit(2, 4.35, 5.22)[0])
        for _ in range(8): g.observe_local(5, False)
        self.assertFalse(g.admit(1, 2.175, 5.22)[0])

    def test_actual_slow_roi_can_close_the_gate(self):
        g = LocalCostGate(PipelineConfig(local_success_prior=.9))
        g.observe_full(10); g.observe_local(12, True)
        self.assertFalse(g.admit(1, 2.175, 5.22)[0])


class PipelineTests(unittest.TestCase):
    frame = np.zeros((720, 1280, 3), np.uint8)

    def test_skip_outputs_predictions_without_real_detector(self):
        p, d = pipeline(); p.step(self.frame, 0)
        p.flow = FakeFlow(dx=2)
        out, r = p.step(self.frame, 1)
        self.assertEqual(r['decision'], SKIP); self.assertEqual(d.full_calls, 1)
        np.testing.assert_allclose(out[0, :4], [102, 100, 142, 140])
        self.assertEqual(r['gflops'], 0)
        self.assertEqual(p.last_prediction_details.shape, (1, 4))
        self.assertEqual(p.last_prediction_details[0, 1], 1)

    def test_early_deadline_skips_flow_but_keeps_full_authoritative(self):
        p, d = pipeline(); p.step(self.frame, 0)
        out, r = p.step(self.frame, p.cfg.k_max)
        self.assertEqual(r['decision'], FULL); self.assertEqual(p.flow.calls, 0)
        np.testing.assert_array_equal(out, detections(BOX))
        self.assertTrue(r['budget_overrun'])

    def test_failed_local_escalates_and_accounts_both_passes(self):
        p, d = pipeline('adaptive_safe_local', detector_budget_fraction=1, dynamic_roi_margin=False)
        d.crops = [detections()]; p.step(self.frame, 0); p.flow = FakeFlow(quality=.3)
        out, r = p.step(self.frame, 1)
        self.assertEqual(r['decision'], FULL); self.assertIn('local_escalate', r['reasons'])
        self.assertEqual(d.roi_calls, 1); self.assertEqual(d.full_calls, 2)
        self.assertAlmostEqual(r['gflops'], 5.22 + 2.175)
        self.assertEqual(p.local_gate.attempts, 1)
        np.testing.assert_array_equal(out, detections(BOX))

    def test_successful_local_uses_detector_geometry(self):
        p, d = pipeline('adaptive_safe_local', detector_budget_fraction=1, dynamic_roi_margin=False)
        d.crops = [detections([30, 28, 70, 68, .8, 0])]
        p.step(self.frame, 0); p.flow = FakeFlow(quality=.3)
        out, r = p.step(self.frame, 1)
        self.assertEqual(r['decision'], LOCAL); self.assertEqual(d.full_calls, 1)
        np.testing.assert_array_equal(out[0, :4], [102, 100, 142, 140])
        self.assertEqual(p.local_gate.successes, 1)

    def test_cost_rejection_does_not_call_crop_detector(self):
        p, d = pipeline('adaptive_safe_local', detector_budget_fraction=1, dynamic_roi_margin=False)
        d.full = [detections(BOX, [400, 100, 440, 140, .9, 0])]
        p.step(self.frame, 0); p.flow = FakeFlow(quality=.3)
        _, r = p.step(self.frame, 1)
        self.assertEqual(r['decision'], FULL); self.assertIn('local_cost_limit', r['reasons'])
        self.assertEqual(d.roi_calls, 0)

    def test_baseline_refinement_is_not_bypassed(self):
        cfg = build_config('baseline_refine', [])
        d = FakeDetector(full=[detections([100, 100, 140, 140, .4, 0])])
        p = AdaptivePipeline(cfg, d); p.step(self.frame, 0)
        self.assertEqual(d.full_calls, 1); self.assertEqual(d.roi_calls, 1)

    def test_sequence_reset_removes_all_temporal_state(self):
        p, d = pipeline(); p.step(self.frame, 0); p.reset()
        self.assertEqual(len(p.tracker), 0); self.assertIsNone(p.prev_gray)
        self.assertEqual(len(p.budget.costs), 0); self.assertEqual(p.local_gate.attempts, 0)
        _, r = p.step(self.frame, 0); self.assertEqual(r['reasons'], 'init')

    def test_all_presets_construct_without_model_loading(self):
        for name in PRESETS:
            with self.subTest(preset=name): build_config(name, [])
        self.assertFalse(build_config('baseline', []).observation_anchoring)
        self.assertEqual(build_config('C_fullfallback', []).controller_mode, 'legacy')

    def test_camera_failure_is_logged_by_pipeline(self):
        p, _ = pipeline('adaptive_safe_camera'); p.step(self.frame, 0)
        _, r = p.step(self.frame, 1)
        self.assertEqual(r['decision'], FULL)
        self.assertIn('motion_unreliable', r['reasons'])

    def test_appearance_failure_reaches_refresh_decision(self):
        p, _ = pipeline('adaptive_safe_verify', use_novelty=False, use_illum_trigger=False)
        rng = np.random.default_rng(33)
        a = rng.integers(0, 255, self.frame.shape, dtype=np.uint8)
        class ImageFlow(FakeFlow):
            def prepare(self, frame):
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.0
        p.flow = ImageFlow(); p.step(a, 0)
        _, r = p.step(255 - a, 1)
        self.assertEqual(r['decision'], FULL)
        self.assertEqual(r['appearance_failed'], 1)


class OptionalVerificationTests(unittest.TestCase):
    def test_background_alignment_on_synthetic_translation(self):
        rng = np.random.default_rng(12)
        prev = rng.integers(0, 255, (120, 160), dtype=np.uint8)
        cur = cv2.warpAffine(prev, np.float32([[1, 0, 2], [0, 1, 1]]), (160, 120))
        m = MotionCompensator(PipelineConfig(motion_compensation=True, flow_width=160))
        self.assertTrue(m.update(prev, cur, np.zeros((0, 4)), 1))
        np.testing.assert_allclose(m.key_transform[:2, 2], [2, 1], atol=.4)
        self.assertLess(m.novelty(prev, cur, np.zeros((0, 4)), 1, 160), .05)

    def test_flat_background_does_not_claim_alignment(self):
        gray = np.zeros((96, 128), np.uint8)
        m = MotionCompensator(PipelineConfig())
        self.assertFalse(m.update(gray, gray, np.zeros((0, 4)), 1))

    def test_appearance_cap_and_changed_patch(self):
        cfg = PipelineConfig(appearance_verification=True, appearance_max_tracks=1)
        v = AppearanceVerifier(cfg)
        rng = np.random.default_rng(13)
        gray = rng.integers(0, 255, (80, 80), dtype=np.uint8)
        boxes = np.array([[5, 5, 30, 30], [40, 40, 70, 70]])
        ids = np.array([1, 2]); v.anchor(gray, boxes, ids, 1)
        q, checks, failures = v.verify(255 - gray, boxes, ids, 1, np.ones(2))
        self.assertEqual(checks, 1); self.assertEqual(failures, 1)
        self.assertEqual(np.sum(q < 1), 1)

    def test_flat_patch_is_unknown(self):
        v = AppearanceVerifier(PipelineConfig())
        gray = np.zeros((40, 40), np.uint8); boxes = np.array([[2, 2, 30, 30]])
        v.anchor(gray, boxes, [1], 1)
        q, n, _ = v.verify(gray, boxes, [1], 1, np.ones(1))
        self.assertEqual(n, 0); self.assertEqual(q[0], 1)


class EvaluationTests(unittest.TestCase):
    def test_macro_f1_is_mean_of_class_f1_not_f1_of_mean_p_r(self):
        e = DetectionEvaluator(['car', 'person'])
        gt = np.array([[0, 0, 0, 10, 10], [1, 20, 0, 30, 10], [1, 40, 0, 50, 10]], np.float32)
        pred = detections([0, 0, 10, 10, .9, 0], [60, 0, 70, 10, .8, 0], [20, 0, 30, 10, .7, 1])
        e.add(pred, gt); r = e.compute()
        self.assertAlmostEqual(r['macro_f1@0.25'], 2/3, places=5)
        self.assertNotAlmostEqual(r['macro_f1@0.25'], 2*r['precision@0.25']*r['recall@0.25']/(r['precision@0.25']+r['recall@0.25']))

    def test_provenance_detects_changed_labels_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            image, label = Path(d)/'synthetic.bin', Path(d)/'synthetic.txt'
            image.write_bytes(b'generated fixture'); label.write_text('0 .5 .5 .2 .2')
            seq = [SimpleNamespace(name='synthetic', frames=[str(image)], labels=[str(label)])]
            a = input_manifest(seq, hash_images=True)
            label.write_text('0 .5 .5 .3 .3')
            self.assertNotEqual(a['sha256'], input_manifest(seq, hash_images=True)['sha256'])
            (Path(d)/'metrics.json').write_text('{}')
            with self.assertRaises(FileExistsError): assert_fresh_run(d)
            label.unlink()
            with self.assertRaises(FileNotFoundError): input_manifest(seq)

    def test_malformed_labels_are_not_silently_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            image, label = Path(d)/'synthetic.bin', Path(d)/'synthetic.txt'
            image.write_bytes(b'synthetic fixture'); label.write_text('not a YOLO label')
            seq = [SimpleNamespace(name='synthetic', frames=[str(image)], labels=[str(label)])]
            with self.assertRaises(ValueError): input_manifest(seq)

    def test_invalid_new_configuration_fails_early(self):
        for kw in [{'risk_uncertainty': 0}, {'local_max_crops': 0}, {'controller_mode': 'unknown'},
                   {'controller_mode': 'risk', 'k_max': 1}, {'flow_r_max': .5}]:
            with self.subTest(config=kw), self.assertRaises(ValueError): PipelineConfig(**kw)

    def test_paired_statistics_reject_different_targets(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a.npz', Path(d)/'b.npz'
            kwargs = dict(n_pred=np.array([1]), n_gt=np.array([1]), video=np.array(['generated']),
                          tp=np.ones((1, 10), bool), conf=np.array([.9]), pcls=np.array([0]))
            np.savez(a, **kwargs, tcls=np.array([0]))
            np.savez(b, **kwargs, tcls=np.array([1]))
            with self.assertRaises(ValueError): paired_bootstrap(a, b, n_boot=2)

    def test_compare_refuses_protocol_and_weight_mismatches(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'; a.mkdir(); b.mkdir()
            run = dict(config=PipelineConfig().to_dict(), ap_protocol='current',
                       provenance={'weights_sha256': 'one', 'input_manifest_sha256': 'same'})
            (a/'metrics.json').write_text(json.dumps(run))
            run['provenance']['weights_sha256'] = 'two'
            (b/'metrics.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError): verify_comparison(a, b, 'current')
            run['provenance']['weights_sha256'] = 'one'; run['ap_protocol'] = 'legacy'
            (b/'metrics.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError): verify_comparison(a, b, 'current')


if __name__ == '__main__': unittest.main()
