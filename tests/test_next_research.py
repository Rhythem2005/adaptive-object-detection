"""Synthetic causal admission, output policy and fast-path controls."""
import numpy as np
import unittest

from adaptive_pipeline.shared.config import PipelineConfig
from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.tracking.tracker import Tracker
from analysis.argoverse_hd_next_research.methods import (ConfirmedBirthMixin,
    ResearchPipeline, corroborated, state_digest)


def box(score=.15, cls=0, x=0):
    return np.array([[x, 0, x+30, 30, score, cls]], np.float32)


class Confirmed(ConfirmedBirthMixin, Tracker):
    pass


def test_matching_empty_class_and_one_to_one():
    assert not corroborated(box(), np.zeros((0,6))).any()
    assert not corroborated(box(), box(cls=1)).any()
    assert corroborated(np.vstack([box(), box()]), box()).sum() == 1
    assert not corroborated(box(x=100), box()).any()


def test_weak_births_causal_expiring_and_reset():
    c=PipelineConfig(track_birth_conf=.1, low_score_association=True)
    t=Confirmed(c)
    t.research_frame=0; t.update_full(box()); assert len(t)==0
    before=state_digest(t)
    t.research_frame=2; t.update_full(box()); assert len(t)==1
    assert np.isclose(t.conf_det[0], .15)  # never promote confidence
    t.reset(); assert len(t.weak_previous)==0
    t.research_frame=2; t.update_full(box()); assert len(t)==0
    t.research_frame=5; t.update_full(box()); assert len(t)==0
    assert state_digest(t)!=before
    t.reset();t.research_frame=0;t.update_full(box(.8));assert len(t)==1


def test_no_duplicate_low_birth_after_confirmation():
    t=Confirmed(PipelineConfig(track_birth_conf=.1, low_score_association=True))
    t.research_frame=0;t.update_full(box(.8))
    t.research_frame=2;t.update_full(box())
    assert len(t)==1 and not len(t.weak_previous)


class Cache:
    def __init__(self): self.value=box(.8)
    def detect(self, frame): return self.value.copy(),0.
    def full_gflops(self, *a): return 1.


def test_output_only_filter_resets_and_preserves_tracker():
    c=PipelineConfig(policy='fixed',fixed_k=2,propagation='hold',observation_anchoring=True)
    d=Cache();d.value=np.vstack([box(.15),box(.8,x=100),box(.01,x=200)])
    p=ResearchPipeline(c,d);p.filter_full_weak=True
    frame=np.zeros((60,240,3),np.uint8)
    out,_=p.step(frame,0);assert len(out)==2 and len(p.tracker)==1
    assert np.array_equal(out,d.value[[1,2]])
    out,_=p.step(frame,2);assert np.array_equal(out,d.value)
    p.reset();out,_=p.step(frame,0);assert len(out)==2


def test_fast_path_rejects_consumers():
    for overrides in [{'policy':'adaptive'}, {'use_clahe':True},
        {'keyframe_refine':True},{'motion_compensation':True},{'appearance_verification':True}]:
        args=dict(policy='fixed',fixed_signal_fastpath=True);args.update(overrides)
        with unittest.TestCase().assertRaises(ValueError):PipelineConfig(**args)


def test_fast_path_prediction_state_schedule_parity():
    c=PipelineConfig(policy='fixed',fixed_k=2,observation_anchoring=True,flow_width=120,
        flow_grid=3,conf_decay=.93,low_score_association=True)
    a=AdaptivePipeline(c,Cache());b=AdaptivePipeline(c.with_overrides(fixed_signal_fastpath=True),Cache())
    image=np.random.default_rng(0).integers(0,256,(120,240,3),np.uint8)
    for i in range(6):
        f=np.roll(image,i,axis=1);pa,ra=a.step(f,i);pb,rb=b.step(f,i)
        np.testing.assert_array_equal(pa,pb)
        np.testing.assert_array_equal(a.last_prediction_details,b.last_prediction_details)
        assert state_digest(a.tracker)==state_digest(b.tracker)
        assert ra['decision']==rb['decision']
    assert np.isnan(rb['lum']) and b.controller.key_thumb is None


class ResearchTests(unittest.TestCase):
    pass

for _name, _fn in list(globals().items()):
    if _name.startswith('test_') and callable(_fn):
        setattr(ResearchTests, _name, lambda self, fn=_fn: fn())
