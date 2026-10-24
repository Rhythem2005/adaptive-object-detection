"""Synthetic E controls. No checkpoint, real image, scoring or pipeline replay."""
import copy
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import torch

from analysis.argoverse_hd_runtime_E.run import (
    BLOCKS, ForwardCounter, assert_archive_equal, full_frame_warmup, timed_step, validate_predictor)


class RuntimeControlsTests(unittest.TestCase):
    def test_installed_prediction_argument_schema_without_inference(self):
        from ultralytics.cfg import get_cfg
        args=get_cfg(overrides=dict(imgsz=640,conf=.001,iou=.7,max_det=300,rect=True))
        predictor=SimpleNamespace(args=args,device='cpu',model=SimpleNamespace(fp16=False))
        validate_predictor(predictor)
        args.max_det=301
        with self.assertRaises(AssertionError):validate_predictor(predictor)

    def test_position_balance(self):
        self.assertEqual(len(BLOCKS),5)
        for order in BLOCKS:self.assertEqual(set(order),set('ABCDE'))
        for col in zip(*BLOCKS):self.assertEqual(set(col),set('ABCDE'))

    def test_warmup_is_three_full_frames_no_roi(self):
        detector=Mock();image=np.zeros((4,5,3),np.uint8)
        full_frame_warmup(detector,image)
        self.assertEqual(detector.detect.call_count,3)
        self.assertIs(detector.detect.call_args.args[0],image)
        detector.detect_batch.assert_not_called()
        detector.warmup.assert_not_called()

    def test_counter_deepcopy_and_actual_device(self):
        model=torch.nn.Linear(2,2);counter=ForwardCounter(model)
        copy.deepcopy(model)(torch.ones(1,2))
        self.assertEqual(counter.count,1)
        self.assertEqual(counter.summary(),[dict(shape=[1,2],dtype='torch.float32',device='cpu',count=1)])
        counter.handle.remove()

    def test_exact_comparator_including_empty_and_bits(self):
        a={'x':np.zeros((0,6),np.float32),'y':np.array([0.],np.float32)}
        assert_archive_equal(a,{k:v.copy() for k,v in a.items()},'test')
        for bad in [{'x':a['x']},dict(a,y=np.array([-0.],np.float32)),
                    dict(a,y=np.array([0.],np.float64))]:
            with self.assertRaises(AssertionError):assert_archive_equal(a,bad,'test')

    def test_nested_scopes_and_no_hidden_model_call(self):
        pipe=Mock();pred=np.zeros((0,6),np.float32);detail=np.zeros((0,4))
        pipe.step.return_value=(pred,{'decision':'SKIP'})
        pipe.last_prediction_details=detail
        frame=np.zeros((1200,1920,3),np.uint8)
        row=dict(image='synthetic',key='s/0',video='s',image_id=2,frame_index=0)
        a,b,rec=timed_step(pipe,row,read=lambda _:frame)
        self.assertIs(a,pred);self.assertIs(b,detail)
        self.assertGreaterEqual(rec['decode_inclusive_ms'],rec['step_wall_ms'])
        self.assertAlmostEqual(rec['decode_ms']+rec['step_wall_ms'],rec['decode_inclusive_ms'],places=9)
        self.assertAlmostEqual(rec['decode_cpu_ms']+rec['step_cpu_ms'],rec['decode_inclusive_cpu_ms'],places=9)
        pipe.step.assert_called_once()
        self.assertIs(pipe.step.call_args.args[0],frame)

    def test_bad_decodes_stop_before_pipeline(self):
        row=dict(image='synthetic',key='s/0');pipe=Mock()
        for frame in [None,np.zeros((640,640,3),np.uint8),np.zeros((1200,1920,3),np.float32)]:
            with self.assertRaises(ValueError):timed_step(pipe,row,read=lambda _:frame)
        pipe.step.assert_not_called()


if __name__=='__main__':unittest.main()
