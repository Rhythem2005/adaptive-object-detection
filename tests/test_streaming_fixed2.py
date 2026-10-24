"""Synthetic integration controls: no YOLO, real images, or dataset evaluation."""
from dataclasses import replace
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from adaptive_pipeline.adaptive_framing.capture import VideoCaptureThread
from adaptive_pipeline.adaptive_framing.frame_manager import LatestFrameBuffer,FramePacket
from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.shared.config import PipelineConfig
from adaptive_pipeline.run_stream import consume,check_practical_config
from adaptive_pipeline.streaming_fixed2 import StreamingFixed2,SourceCadence

PERIOD=33_300_000


def config():
    return PipelineConfig(device='cpu',policy='fixed',fixed_k=2,propagation='flow',
        observation_anchoring=True,fixed_signal_fastpath=True,flow_grid=3,
        low_score_association=True,conf_decay=.93,class_map={0:0,1:1,2:2,3:3,5:4,7:5,9:6,11:7})


def packet(fid,*,pts=None):
    now=time.perf_counter_ns()
    return FramePacket(np.zeros((120,160,3),np.uint8),'s',fid,
        315969629022515560+fid*PERIOD if pts is None else pts,now,now,now)


class Detector:
    def __init__(self):
        self.calls=0;self.value=np.array([[20,20,60,60,.8,2]],np.float32);self.callback=None
    def detect(self,frame):
        self.calls+=1
        if self.callback:self.callback()
        return self.value.copy(),0.
    def full_gflops(self,*args):return 1.


class Flow:
    def __init__(self):self.calls=0;self.mode='valid';self.callback=None;self.diagnostics={}
    def prepare(self,frame):return frame[:,:,0],1.
    def propagate(self,prev,gray,boxes,scale):
        self.calls+=1
        if self.callback:self.callback()
        if self.mode=='error':raise cv2.error('synthetic backend error')
        new=boxes.copy();ok=np.ones(len(boxes),bool)
        if self.mode=='failed':ok[:]=False
        if self.mode=='nan':new[0,0]=np.nan
        if self.mode=='clipped':new[:,2:]=new[:,:2]+(new[:,2:]-new[:,:2])*1.25
        if self.mode=='near_limit':new[:,2:]=new[:,:2]+(new[:,2:]-new[:,:2])*1.249
        return new,ok,ok.astype(np.float32)


def processor():
    d=Detector();p=StreamingFixed2(config(),d,SourceCadence(PERIOD,PERIOD,PERIOD))
    f=Flow();p.pipe.flow.original=f
    return p,d,f


class CadenceTests(unittest.TestCase):
    def test_certification_keeps_original_integer_clock(self):
        rows=[dict(frame_id=i,source_id='s',source_timestamp_ns=315969629022515560+i*PERIOD) for i in range(4)]
        c=SourceCadence.from_rows(rows)
        self.assertEqual(c.minimum_ns,PERIOD);self.assertTrue(c.consistent(3,3*PERIOD))
        rows[1]['frame_id']=4
        with self.assertRaises(ValueError):SourceCadence.from_rows(rows)

    def test_bad_clock_rejected(self):
        for rows in [[],[dict(frame_id=0,source_id='s',source_timestamp_ns=2),dict(frame_id=1,source_id='s',source_timestamp_ns=1)]]:
            with self.assertRaises(ValueError):SourceCadence.from_rows(rows)


class ScheduleTests(unittest.TestCase):
    def test_consecutive_full_flow_pattern(self):
        p,d,f=processor()
        decisions=[p.step_packet(packet(i))[1]['decision'] for i in range(6)]
        self.assertEqual(decisions,['FULL','SKIP']*3);self.assertEqual(d.calls,3);self.assertEqual(f.calls,3)

    def test_source_drops_do_not_advance_consumed_schedule(self):
        p,d,f=processor();ids=[0,3,4,7,10]
        rec=[p.step_packet(packet(i))[1] for i in ids]
        self.assertEqual([r['decision'] for r in rec],['FULL','SKIP','FULL','SKIP','FULL'])
        self.assertEqual([r['source_gap'] for r in rec],[1,3,1,3,3])
        self.assertEqual([r['consumed_index'] for r in rec],list(range(5)))
        self.assertEqual([r['frame'] for r in rec],ids)

    def test_age_and_confidence_follow_source_gap_not_consumption(self):
        p,d,f=processor();p.step_packet(packet(0));out,r=p.step_packet(packet(3))
        self.assertEqual(p.pipe.controller.since_full,1)
        self.assertEqual(p.pipe.tracker.since_det[0],3)
        self.assertAlmostEqual(out[0,4],.8*.93**3,places=6)
        self.assertEqual(r['source_elapsed_ms'],3*PERIOD/1e6)

    def test_pacing_clock_does_not_set_motion_units(self):
        a,_,_=processor();b,_,_=processor()
        for i in [0,2,3]:
            p=packet(i);q=replace(p,capture_ns=p.capture_ns-1_000_000_000)
            pa,ra=a.step_packet(p);pb,rb=b.step_packet(q)
            np.testing.assert_array_equal(pa,pb)
            np.testing.assert_array_equal(a.pipe.tracker.P,b.pipe.tracker.P)

    def test_valid_gapped_kalman_covariance_is_psd(self):
        p,_,_=processor()
        for i in [0,3,4,6,7]:
            p.step_packet(packet(i))
            self.assertGreaterEqual(np.linalg.eigvalsh(p.pipe.tracker.P).min(),-1e-10)

    def test_no_gap_valid_measurements_match_native_frozen_pipeline(self):
        p,d,f=processor();native=AdaptivePipeline(config(),Detector());native.flow=Flow()
        for i in range(8):
            frame=packet(i);a,ar=p.step_packet(frame);b,br=native.step(frame.frame,i)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(p.last_prediction_details,native.last_prediction_details)
            self.assertEqual(ar['decision'],br['decision'])
            for key in ['X','P','ids','since_det','geometry','observation','score_age']:
                np.testing.assert_array_equal(getattr(p.pipe.tracker,key),getattr(native.tracker,key))


class GuardTests(unittest.TestCase):
    def test_flow_failure_forces_full_before_measurement_correction(self):
        p,d,f=processor();p.step_packet(packet(0));f.mode='failed'
        with patch.object(p.pipe.tracker,'_kf_update',wraps=p.pipe.tracker._kf_update) as update:
            out,r=p.step_packet(packet(2))
        self.assertEqual(update.call_count,2)
        self.assertEqual(len(update.call_args_list[0].args[0]),0)  # flow update is a no-op
        self.assertEqual(update.call_args.args[2],1.)  # detector correction only
        self.assertEqual(r['decision'],'FULL');self.assertEqual(r['forced_full'],1)
        self.assertEqual(r['gap_triggered_full'],1)
        self.assertEqual(r['flow_attempted'],1);self.assertEqual(r['flow_accepted'],0)
        self.assertEqual(p.pipe.tracker.since_det[0],0)
        np.testing.assert_array_equal(out,d.value)

    def test_nonfinite_flow_is_rejected_without_corrupting_state(self):
        p,d,f=processor();p.step_packet(packet(0));f.mode='nan'
        _,r=p.step_packet(packet(1))
        self.assertIn('invalid_flow_geometry',r['fallback_reasons'])
        self.assertTrue(np.isfinite(p.pipe.tracker.X).all())

    def test_scale_saturation_forces_full(self):
        p,d,f=processor();p.step_packet(packet(0));f.mode='clipped'
        _,r=p.step_packet(packet(1))
        self.assertIn('flow_scale_clipped',r['fallback_reasons']);self.assertEqual(r['decision'],'FULL')

    def test_scale_near_but_not_at_limit_is_not_rejected(self):
        p,d,f=processor();p.step_packet(packet(0));f.mode='near_limit'
        _,r=p.step_packet(packet(1));self.assertEqual(r['decision'],'SKIP')

    def test_known_opencv_error_falls_back(self):
        p,d,f=processor();p.step_packet(packet(0));f.mode='error'
        _,r=p.step_packet(packet(1));self.assertIn('flow_backend_error',r['fallback_reasons'])

    def test_repeated_fallbacks_then_recovery_do_not_double_tick(self):
        p,d,f=processor();p.step_packet(packet(0));f.mode='failed'
        for i in [1,2,3]:
            _,r=p.step_packet(packet(i));self.assertEqual(r['forced_full'],1)
            self.assertEqual(r['consumed_index'],i);self.assertEqual(p.pipe.controller.since_full,0)
        f.mode='valid'
        self.assertEqual(p.step_packet(packet(4))[1]['decision'],'SKIP')
        self.assertEqual(p.step_packet(packet(5))[1]['decision'],'FULL')
        self.assertEqual(d.calls,5)

    def test_source_staleness_resets_and_refreshes_without_flow(self):
        p,d,f=processor();p.step_packet(packet(0))
        _,r=p.step_packet(packet(8))
        self.assertEqual(r['state_reset'],1);self.assertEqual(r['forced_full'],1)
        self.assertEqual(f.calls,0);self.assertIn('source_staleness',r['fallback_reasons'])
        self.assertEqual(p.step_packet(packet(9))[1]['decision'],'SKIP')

    def test_clock_discontinuity_reinitializes_instead_of_false_dt(self):
        p,d,f=processor();first=packet(0);p.step_packet(first)
        _,r=p.step_packet(packet(1,pts=first.source_timestamp_ns+2*PERIOD))
        self.assertIn('uncertified_source_cadence',r['state_reset_reasons'])
        self.assertEqual(r['decision'],'FULL');self.assertEqual(f.calls,0)

    def test_invalid_temporal_state_is_reset(self):
        p,d,f=processor();p.step_packet(packet(0));p.pipe.tracker.X[0,0]=np.nan
        _,r=p.step_packet(packet(1));self.assertIn('invalid_temporal_state',r['state_reset_reasons'])
        self.assertTrue(np.isfinite(p.pipe.tracker.X).all())

    def test_no_visible_state_forces_detector(self):
        p,d,f=processor();d.value=np.zeros((0,6),np.float32);p.step_packet(packet(0))
        _,r=p.step_packet(packet(1));self.assertIn('no_usable_visible_state',r['fallback_reasons'])
        self.assertEqual(d.calls,2);self.assertEqual(f.calls,0)

    def test_bad_source_metadata_fails_without_inference(self):
        p,d,f=processor();p.step_packet(packet(0))
        for invalid in [packet(0),replace(packet(1),source_id='other'),packet(1,pts=1)]:
            with self.assertRaises(ValueError):p.step_packet(invalid)
        self.assertEqual(d.calls,1);self.assertEqual(p.consumed,1)

    def test_scheduled_full_does_not_run_flow(self):
        p,d,f=processor();p.step_packet(packet(0));p.step_packet(packet(1));f.mode='error'
        _,r=p.step_packet(packet(2));self.assertEqual(r['decision'],'FULL');self.assertEqual(f.calls,1)

    def test_practical_config_guard(self):
        data=config().to_dict();data['class_map']={str(k):v for k,v in data['class_map'].items()}
        check_practical_config(data)
        data['flow_grid']=4
        with self.assertRaises(ValueError):check_practical_config(data)


class CaptureIntegrationTests(unittest.TestCase):
    def test_capture_during_both_detector_and_flow_keeps_only_newest(self):
        p,d,f=processor();b=LatestFrameBuffer()
        detect_entered=threading.Event();detect_release=threading.Event()
        flow_entered=threading.Event();flow_release=threading.Event()
        def during_detect():
            if d.calls==1:
                detect_entered.set()
                if not detect_release.wait(1):raise RuntimeError('Producer blocked by detection')
        def during_flow():
            flow_entered.set()
            if not flow_release.wait(1):raise RuntimeError('Producer blocked by flow')
        d.callback=during_detect;f.callback=during_flow
        class Capture(VideoCaptureThread):
            def start(self):return self._launch()
            def _loop(self):
                self.epoch_ns=time.perf_counter_ns();b.put_packet(packet(0))
                if not detect_entered.wait(1):raise RuntimeError('Missing detection')
                for i in [1,2,3]:b.put_packet(packet(i))
                detect_release.set()
                if not flow_entered.wait(1):raise RuntimeError('Missing flow')
                for i in [4,5,6]:b.put_packet(packet(i))
                flow_release.set()
        cap=Capture('fake',b)
        preds,details,rows,s=consume(cap,p,deadline_ns=time.perf_counter_ns()+int(3e9))
        self.assertEqual([r['frame_id'] for r in rows],[0,3,6])
        self.assertEqual([r['decision'] for r in rows],['FULL','SKIP','FULL'])
        self.assertEqual(s['buffer']['peak_pending'],1);self.assertEqual(s['buffer']['replaced'],4)
        self.assertEqual(s['buffer']['pending'],0);self.assertTrue(cap.is_stopped())
        for r in rows:self.assertEqual(r['source_timestamp_ns'],315969629022515560+r['frame_id']*PERIOD)


class EvaluationControls(unittest.TestCase):
    def test_time_alignment_uses_actual_flow_prediction(self):
        from analysis.latest_frame_fixed2_validation import held_outputs
        actual=np.array([[9,9,20,20,.7,2]],np.float32)
        rows=[dict(key='s/4',frame_id=4,decision='FULL',result_ready_ns=10,capture_ns=0),
              dict(key='s/6',frame_id=6,decision='SKIP',result_ready_ns=20,capture_ns=12)]
        events=[dict(frame_id=i,capture_ns=t) for i,t in enumerate([0,11,21])]
        predictions={'s/4':np.zeros((0,6),np.float32),'s/6':actual}
        arrays,assign=held_outputs(rows,events,predictions)
        self.assertEqual(len(arrays[0]),0);self.assertEqual(len(arrays[1]),0)
        self.assertIs(arrays[2],actual)
        self.assertEqual(assign[2]['output_decision'],'SKIP')
        self.assertEqual(assign[2]['last_full_source_frame_id'],4)

    def test_official_scoring_preserves_noncontiguous_image_ids(self):
        import tempfile
        from pathlib import Path
        from analysis.latest_frame_fixed2_validation import score_arrays
        gt=dict(images=[dict(id=77,fid=5,width=100,height=100)],categories=[dict(id=2,name='car')],
            annotations=[dict(id=1,image_id=77,category_id=2,bbox=[9,9,11,11],area=121,iscrowd=0)],info={})
        with tempfile.TemporaryDirectory() as tmp:
            result,matches=score_arrays({5:np.array([[9,9,20,20,.7,2]],np.float32)},gt,[5],Path(tmp)/'score')
        self.assertAlmostEqual(result['COCO']['AP'],1.)
        self.assertEqual(matches[5]['matched_gt25'],[1]);self.assertEqual(matches[5]['fp25'],0)

    @staticmethod
    def fixture():
        passes={};aligned={}
        for fps in [15,90]:
            for rep in [1,2]:
                for arm in ['B','C']:
                    c=arm=='C';name=f'r{rep}_{fps}_{arm}'
                    passes[name]=dict(valid=True,loop_cpu_s=6 if c else 10,measured_yolo_calls=150 if c else 200,
                        capture_to_result_ms=dict(p95=35,max=80),processed=300,buffer=dict(replaced=0))
                    aligned[name]=dict(metrics=dict(COCO=dict(AP=.2),fixed_conf25=dict(micro_recall=.5,counts=dict(FP=100))))
                    if fps==15 and not c:passes[name]['measured_yolo_calls']=300
        return passes,aligned

    def test_accuracy_gate_is_not_satisfied_by_detector_savings(self):
        from analysis.latest_frame_fixed2_validation import acceptance
        passes,aligned=self.fixture()
        self.assertTrue(all(r['all_passed'] for r in acceptance(passes,aligned).values()))
        aligned['r1_90_C']['metrics']['COCO']['AP']=.01
        gates=acceptance(passes,aligned)
        self.assertFalse(gates['90']['all_passed']);self.assertFalse(gates['90']['flags']['mean_aligned_AP'])

    def test_CPU_regression_gate_is_separate_from_call_ratio(self):
        from analysis.latest_frame_fixed2_validation import acceptance
        passes,aligned=self.fixture();passes['r1_90_C']['loop_cpu_s']=11
        gate=acceptance(passes,aligned)['90']
        self.assertTrue(gate['flags']['detector_savings']);self.assertFalse(gate['flags']['no_large_cpu_regression'])

    def test_normal_drop_fails_even_with_good_latency(self):
        from analysis.latest_frame_fixed2_validation import acceptance
        passes,aligned=self.fixture();passes['r2_15_C']['processed']=299;passes['r2_15_C']['buffer']['replaced']=1
        self.assertFalse(acceptance(passes,aligned)['15']['flags']['no_normal_overwrites'])


if __name__=='__main__':unittest.main()
