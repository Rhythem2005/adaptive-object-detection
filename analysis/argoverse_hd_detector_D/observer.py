"""Observe native YOLOv8 NMS without changing inputs, backend, indices or outputs."""
from contextlib import ExitStack
import inspect
import sys
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.utils import nms, ops


class NativeNMSObserver:
    def __init__(self):
        self.calls = 0
        self.primitive_calls = 0

    def __enter__(self):
        self.stack = ExitStack()
        self.original = nms.non_max_suppression
        self.signature = inspect.signature(self.original)
        self.use_torchvision = 'torchvision' in sys.modules
        owner = sys.modules['torchvision'].ops if self.use_torchvision else nms.TorchNMS
        self.backend = 'torchvision.ops.nms' if self.use_torchvision else 'ultralytics.TorchNMS.nms'
        self.primitive = owner.nms
        self.stack.enter_context(patch.object(owner, 'nms', self.observe_primitive))
        self.stack.enter_context(patch.object(nms, 'non_max_suppression', self.observe_nms))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def observe_nms(self, prediction, *args, **kwargs):
        if self.calls:
            raise RuntimeError('Observer permits one single-image NMS only')
        bound = self.signature.bind(prediction, *args, **kwargs)
        bound.apply_defaults(); p = bound.arguments
        if p['classes'] is not None or p['agnostic'] or p['multi_label'] or len(p['labels']) or p['rotated'] or p['end2end'] or p['return_idxs']:
            raise ValueError('Unsupported diagnostic NMS settings')
        value = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        if value.device.type != 'cpu' or value.dtype != torch.float32 or value.ndim != 3 or value.shape[:2] != (1,84):
            raise ValueError(f'Expected single-image CPU FP32 80-class head; got {value.shape}/{value.dtype}/{value.device}')
        if ('torchvision' in sys.modules) != self.use_torchvision:
            raise RuntimeError('Import state changed NMS selection')
        # Native NMS mutates coordinate storage: preserve a separate pre-mutation clone.
        self.raw = value.detach().clone()
        self.params = {k:v for k,v in p.items() if k != 'prediction'}
        raw = self.raw.transpose(-1,-2)[0]
        eligible = self.raw[:,4:84].amax(1)[0] > p['conf_thres']
        raw = raw[eligible]
        boxes = ops.xywh2xyxy(raw[:,:4])
        conf, cls = raw[:,4:84].max(1,keepdim=True)
        selected = conf.view(-1) > p['conf_thres']
        self.rows = torch.cat((boxes,conf,cls.float()),1)[selected]
        if len(self.rows) > p['max_nms']:
            self.rows = self.rows[self.rows[:,4].argsort(descending=True)[:p['max_nms']]]
        self.calls += 1
        output = self.original(prediction,*args,**kwargs)
        if not isinstance(output,list) or len(output)!=1:
            raise RuntimeError('Unexpected NMS output')
        if not len(self.rows):
            if self.primitive_calls or len(output[0]):raise AssertionError('Invalid empty-case observation')
            self.indices = torch.zeros(0,dtype=torch.int64)
        elif self.primitive_calls != 1:
            raise AssertionError('Native primitive was not observed exactly once')
        self.capped_letterbox = output[0].detach().clone()
        self.survivors_letterbox = self.rows[self.indices].clone()
        if not torch.equal(self.survivors_letterbox[:p['max_det']],self.capped_letterbox):
            raise AssertionError('Native pre-cap reconstruction failed')
        return output  # The original list/tensors, with no replacement or modification.

    def observe_primitive(self, boxes, scores, threshold):
        self.primitive_calls += 1
        shifted = self.rows[:,:4] + self.rows[:,5:6]*self.params['max_wh']
        if not torch.equal(shifted,boxes) or not torch.equal(self.rows[:,4],scores):
            raise AssertionError('Raw-to-native candidate row alignment failed')
        if threshold != self.params['iou_thres']:
            raise AssertionError('NMS IoU mismatch')
        output = self.primitive(boxes,scores,threshold)
        self.indices = output.detach().clone()
        return output  # The native index tensor, unchanged, before native top-k slicing.

    def original_coordinates(self, tensor_shape, image_shape):
        rows = self.survivors_letterbox.clone()
        rows[:,:4] = ops.scale_boxes(tensor_shape, rows[:,:4], image_shape)
        return rows.cpu().numpy().copy()


def map_rows(rows, class_map):
    """Same final float32/LUT semantics as the original detector wrapper."""
    a = np.asarray(rows,dtype=np.float32)
    keep = np.array([int(c) in class_map for c in a[:,5]],bool)
    out = a[keep].copy()
    out[:,5] = [class_map[int(c)] for c in out[:,5]]
    return out
