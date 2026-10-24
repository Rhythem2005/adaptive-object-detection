"""Synthetic observer parity only; never loads a checkpoint or dataset image."""
import unittest
import copy
import numpy as np
import torch
from ultralytics.utils import nms
from analysis.argoverse_hd_detector_D.observer import NativeNMSObserver, map_rows


def head(n=400, same_score=False):
    a=torch.zeros(1,84,n)
    for i in range(n):
        a[0,:4,i]=torch.tensor([10+(i%20)*30,10+(i//20)*30,8,8])
        a[0,4+i%80,i]=.9 if same_score else .9-i*.001
    return a


class ObserverTests(unittest.TestCase):
    def test_counter_survives_predictor_model_deepcopy(self):
        from analysis.argoverse_hd_detector_D.run import ForwardCounter
        model=torch.nn.Linear(2,2)
        counter=ForwardCounter(model)
        cloned=copy.deepcopy(model)
        cloned(torch.ones(1,2))
        self.assertEqual(counter.count,1)
        self.assertEqual(counter.shapes[0]['shape'],[1,2])
        counter.handle.remove()

    def compare(self,a,cap=300):
        first=a.clone();control=nms.non_max_suppression(first,conf_thres=.001,iou_thres=.7,max_det=cap)
        second=a.clone()
        with NativeNMSObserver() as observer:
            actual=nms.non_max_suppression(second,conf_thres=.001,iou_thres=.7,max_det=cap)
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(control[0],actual[0]))
        self.assertTrue(torch.equal(a,observer.raw))
        self.assertTrue(torch.equal(observer.survivors_letterbox[:cap],control[0]))
        return observer

    def test_over300_and_exact_indices(self):
        o=self.compare(head());self.assertEqual(len(o.survivors_letterbox),400)
        self.assertEqual(len(o.capped_letterbox),300)

    def test_empty(self):
        o=self.compare(torch.zeros(1,84,10));self.assertEqual(o.primitive_calls,0)

    def test_equal_score_ties_and_mixed_classes(self):
        self.compare(head(100,True),cap=50)

    def test_overlap_suppression(self):
        a=head(10);a[0,:4,:]=a[0,:4,0,None];a[0,4:,:]=0;a[0,4,:]=.8
        o=self.compare(a);self.assertEqual(len(o.survivors_letterbox),1)

    def test_mapping_and_coordinate_restoration(self):
        o=self.compare(head(20))
        original=o.original_coordinates((640,640),(1200,1920,3))
        mapped=map_rows(original,{0:0,1:1,2:2,3:3,5:4,7:5,9:6,11:7})
        self.assertTrue(np.isfinite(mapped).all())
        self.assertEqual(set(mapped[:,5]),set(range(8)))
        np.testing.assert_array_equal(map_rows(np.zeros((0,6)),{0:0}),np.zeros((0,6),np.float32))

    def test_reject_class_filter_without_mutating_input(self):
        a=head(10);before=a.clone()
        with self.assertRaises(ValueError):
            with NativeNMSObserver():nms.non_max_suppression(a,classes=[0])
        self.assertTrue(torch.equal(a,before))


if __name__=='__main__':unittest.main()
