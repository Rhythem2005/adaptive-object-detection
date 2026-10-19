"""Synthetic-only validation of cache/replay observations and scoring IDs."""
import unittest
import numpy as np

from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.shared.config import PipelineConfig
from adaptive_pipeline.evaluation.coco import prepare_ground_truth, prediction_records, make_coco, run_coco
from analysis.argoverse_hd_phase2.replay import (CacheDetector, Recorder, install_observers,
                                              state_digest)
from analysis.argoverse_hd_phase2.scoring import match_arrays, frame_counts


class Phase2HarnessTests(unittest.TestCase):
    def config(self, **kw):
        return PipelineConfig(policy='fixed', fixed_k=2, class_map={0:0}, **kw)

    def test_cache_binding_copy_and_no_extra_inference(self):
        cfg=self.config();r=Recorder();a=np.array([[2,2,20,20,.8,0]],np.float32)
        cache=CacheDetector(cfg, {'v/0':a},r)
        with self.assertRaises(RuntimeError):cache.detect(None)
        with self.assertRaises(KeyError):cache.bind('v/9')
        cache.bind('v/0');result,_=cache.detect(None);result[0,0]=90
        self.assertEqual(a[0,0],2)
        with self.assertRaises(RuntimeError):cache.detect(None)
        with self.assertRaises(RuntimeError):cache.detect_batch([])
        with self.assertRaises(RuntimeError):cache.warmup(None)

    def test_cache_taxonomy_finiteness_and_extents_fail(self):
        for row in [[0,0,10,10,.5,1],[0,0,10,10,float('nan'),0],[10,0,0,10,.5,0],[0,0,10,10,.5,.1]]:
            with self.assertRaises(ValueError):CacheDetector(self.config(),{'v/0':np.array([row])},Recorder())
        CacheDetector(self.config(),{'v/0':np.array([[0,0,10,0,.5,0]])},Recorder())

    def test_observers_do_not_change_predictions_decisions_or_states(self):
        rng=np.random.default_rng(20);texture=rng.integers(0,255,(96,128,3),dtype=np.uint8)
        predictions={f'v/{i}':np.array([[20+i,20,48+i,52,.8 if i!=4 else .15,0]],np.float32) for i in range(12)}
        for anchored,quality in [(False,False),(True,False),(True,True)]:
            cfg=self.config(observation_anchoring=anchored,quality_aware_flow=quality)
            r=Recorder();d1=CacheDetector(cfg,predictions,r);p1=AdaptivePipeline(cfg,d1)
            d2=CacheDetector(cfg,predictions,r);p2=AdaptivePipeline(cfg,d2);d2.pipe=p2
            install_observers(p2,r)
            for i in range(12):
                key=f'v/{i}';im=np.roll(texture,i,axis=1)
                r.begin({'frame_index':i,'video':'v','key':key});d1.bind(key);d2.bind(key)
                a,ra=p1.step(im,i);b,rb=p2.step(im,i)
                np.testing.assert_array_equal(a,b)
                self.assertEqual(ra['decision'],rb['decision'])
                self.assertEqual(state_digest(p1.tracker),state_digest(p2.tracker))
                if i==0:self.assertTrue(any(e['type']=='spawn' for e in r.events))

    def test_forced_schedule_keeps_common_adaptive_path(self):
        cfg=PipelineConfig(class_map={0:0},use_local=True,max_local_rois=0,observation_anchoring=True)
        predictions={f'v/{i}':np.array([[10,10,30,30,.9,0]],np.float32) for i in range(6)}
        r=Recorder();d=CacheDetector(cfg,predictions,r);p=AdaptivePipeline(cfg,d);d.pipe=p
        install_observers(p,r,{'v':{0,2,5}})
        for i in range(6):
            r.begin({'frame_index':i,'video':'v'});d.bind(f'v/{i}')
            _,row=p.step(np.zeros((64,64,3),np.uint8),i)
            self.assertEqual(row['decision'],'FULL' if i in {0,2,5} else 'SKIP')
            if i in {2,5}:self.assertTrue(r.shadow_eligible)
        self.assertEqual(d.calls,3)

    def test_official_identity_matches_crowds_and_confidence_counts(self):
        gt={'images':[{'id':0,'width':100,'height':100}], 'categories':[{'id':0,'name':'person'}],
            'annotations':[{'id':0,'image_id':0,'category_id':0,'bbox':[0,0,10,10],'area':100,'iscrowd':0},
             {'id':1,'image_id':0,'category_id':0,'bbox':[50,50,40,40],'area':1600,'iscrowd':1}]}
        norm,_=prepare_ground_truth(gt)
        p={'v/0':np.array([[55,55,60,60,.99,0],[0,0,10,10,.2,0],[20,20,30,30,.3,0]])}
        records,_=prediction_records({0:p['v/0']},{0:0});g,d=make_coco(norm,records);ev,_=run_coco(g,d)
        a=match_arrays(ev,records)
        self.assertTrue(a['ignored'][0].all());self.assertTrue(a['tp'][1].all())
        self.assertEqual(a['matched_gt'][1,0],1)
        self.assertAlmostEqual(a['gt_scores'][1,0],.2)
        self.assertEqual(a['gt_scores'][2,0],-1)
        counts=frame_counts(a,[{'key':'v/0'}],p)
        np.testing.assert_array_equal(counts['0.001'][0,0],[1,1])
        np.testing.assert_array_equal(counts['0.25'][0,0],[0,1])


if __name__=='__main__':unittest.main()
