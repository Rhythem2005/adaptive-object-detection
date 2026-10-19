"""Synthetic Phase 3 association/admission checks; no model or dataset."""
import unittest
from unittest.mock import patch

import numpy as np

from adaptive_pipeline.shared.config import PipelineConfig
from adaptive_pipeline.tracking.tracker import Tracker


def dets(*rows):
    return np.asarray(rows, np.float32).reshape(-1, 6)


def box(x=0, score=.9, cls=0):
    return [x, 0, x+20, 20, score, cls]


class Phase3TrackingTests(unittest.TestCase):
    def tracker(self, **kw):
        return Tracker(PipelineConfig(observation_anchoring=True,
            low_score_association=True, **kw))

    def test_invalid_opt_in_limits(self):
        for name in ['track_birth_conf', 'low_association_iou']:
            for value in [-.1, 1.1, float('nan')]:
                with self.assertRaises(ValueError):PipelineConfig(**{name:value})

    def test_separate_birth_gate_keeps_low_confirmation_and_no_duplicate_birth(self):
        t=self.tracker(track_birth_conf=.1)
        t.update_full(dets(box()))
        t.update_full(dets(box(score=.15), box(100, .12), box(200, .05)))
        self.assertEqual(list(t.ids), [1,2])
        np.testing.assert_allclose(t.conf_det, [.15,.12])
        self.assertEqual(t.cfg.track_min_conf,.25)

    def test_birth_gate_can_be_higher_than_association_gate(self):
        t=self.tracker(track_birth_conf=.7)
        t.update_full(dets(box(score=.6),box(100,.8)))
        self.assertEqual(len(t),1)
        t.update_full(dets(box(100,.4)))
        self.assertEqual(len(t),1)
        self.assertAlmostEqual(t.conf_det[0],.4,places=6)

    def test_low_birth_floor_below_low_association_pool(self):
        t=self.tracker(track_birth_conf=.05)
        t.update_full(dets(box(score=.08)))
        self.assertEqual(len(t),1)

    def test_visible_only_does_not_revive_hidden_but_high_scores_can(self):
        for visible_only in [False,True]:
            t=self.tracker(low_association_visible_only=visible_only)
            t.update_full(dets(box()));t.update_full(dets())
            t.update_full(dets(box(score=.15)))
            self.assertEqual(len(t),0 if visible_only else 1)
        t=self.tracker(low_association_visible_only=True)
        t.update_full(dets(box()));t.update_full(dets());t.update_full(dets(box()))
        self.assertEqual(list(t.ids),[1])
        self.assertEqual(t.missed[0],0)

    def test_excluded_hidden_tracks_still_accumulate_misses(self):
        t=self.tracker(low_association_visible_only=True)
        t.update_full(dets(box(),box(100)))
        t.update_full(dets(box(100)))
        t.update_full(dets(box(score=.15),box(100,.15)))
        self.assertEqual(list(t.ids),[2])
        self.assertEqual(t.missed[0],0)

    def test_low_iou_threshold_only_changes_second_stage(self):
        t=self.tracker(low_association_iou=.5)
        t.update_full(dets(box()))
        # 10px displacement of a20px box has IoU1/3.
        t.update_full(dets(box(10,.15)))
        self.assertEqual(t.missed[0],1)
        t.update_full(dets(box(10,.8)))
        self.assertEqual(list(t.ids),[1])
        self.assertEqual(t.missed[0],0)

    def test_gating_before_assignment_recovers_available_valid_pair(self):
        matrix=np.array([[.30,.29],[.29,0.]])
        for enabled,expected in [(False,0),(True,1)]:
            t=self.tracker(threshold_aware_assignment=enabled)
            t.update_full(dets(box(),box(100)))
            with patch('adaptive_pipeline.tracking.tracker.iou_matrix',return_value=matrix.copy()):
                matches,_,_=t._associate(dets(box(),box(100)))
            self.assertEqual(len(matches),expected)

    def test_threshold_aware_matching_objective_is_explicit(self):
        # Cardinality first can trade a strong pair for two weaker valid pairs.
        # This is a tested design choice, not a promised dataset improvement.
        t=self.tracker(threshold_aware_assignment=True)
        t.update_full(dets(box(),box(100)))
        with patch('adaptive_pipeline.tracking.tracker.iou_matrix',return_value=np.array([[.99,.31],[.31,0.]])):
            matches,_,_=t._associate(dets(box(),box(100)))
        self.assertEqual(set(matches),{(0,1),(1,0)})

    def test_class_and_empty_rectangular_matching(self):
        t=self.tracker(threshold_aware_assignment=True,low_association_iou=0)
        t.update_full(dets(box()))
        matches,ud,ut=t._associate(dets(box(cls=1)),iou_threshold=0)
        self.assertEqual((matches,ud,ut),([],[0],[0]))
        self.assertEqual(t._associate(dets()),([] ,[],[0]))
        matches,ud,ut=t._associate(dets(box(),box(100),box(200)))
        self.assertEqual(matches,[(0,0)]);self.assertEqual(ud,[1,2]);self.assertEqual(ut,[])

    def test_phase_harness_preserves_native_fixed_fast_path(self):
        from adaptive_pipeline.pipeline import AdaptivePipeline
        from analysis.argoverse_hd_phase2.replay import CacheDetector, Recorder, state_digest
        from analysis.argoverse_hd_phase3.observe import install
        from analysis.argoverse_hd_phase3.timing import force_fixed
        rng=np.random.default_rng(43)
        texture=rng.integers(0,255,(96,128,3),dtype=np.uint8)
        predictions={f'v/{i}':dets(box(20+i,.8 if i!=4 else .15)) for i in range(10)}
        a_cfg=PipelineConfig(policy='fixed',fixed_k=2,observation_anchoring=True,class_map={0:0},low_score_association=True)
        b_cfg=a_cfg.with_overrides(fixed_k=100)
        rec=Recorder();da=CacheDetector(a_cfg,predictions,rec);a=AdaptivePipeline(a_cfg,da)
        db=CacheDetector(b_cfg,predictions,rec);b=AdaptivePipeline(b_cfg,db);db.pipe=b
        install(b,rec,{'v':set(range(0,10,2))})
        dc=CacheDetector(b_cfg,predictions,Recorder());c=AdaptivePipeline(b_cfg,dc)
        current=[None];force_fixed(c,{'v':set(range(0,10,2))},current)
        for i in range(10):
            info={'video':'v','frame_index':i,'key':f'v/{i}'};current[0]=info;rec.begin(info)
            for d in [da,db,dc]:d.bind(info['key'])
            im=np.roll(texture,i,axis=1)
            ap,ar=a.step(im,i);bp,br=b.step(im,i);cp,cr=c.step(im,i)
            np.testing.assert_array_equal(ap,bp);np.testing.assert_array_equal(ap,cp)
            self.assertEqual(ar['decision'],br['decision']);self.assertEqual(ar['decision'],cr['decision'])
            self.assertEqual(state_digest(a.tracker),state_digest(b.tracker))
            self.assertEqual(state_digest(a.tracker),state_digest(c.tracker))


if __name__=='__main__':unittest.main()
