"""Synthetic archive-schema checks. No real images/models or benchmark passes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval
from analysis.argoverse_hd_runtime_E.resume import tensor_fidelity


class TensorSchemaTests(unittest.TestCase):
    def test_legacy_axis_metadata_is_validated_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'current.npz';b=Path(tmp)/'archive.npz'
            tensors={k:np.zeros((2,3),np.float64) for k in ['precision','recall','scores']}
            np.savez(a,**tensors)
            params=COCOeval().params
            extra=dict(category_ids=np.arange(8,dtype=np.int64),image_ids=np.array([0,1],np.int64),
                iou_thresholds=params.iouThrs,recall_thresholds=params.recThrs)
            np.savez(b,**tensors,**extra)
            tensor_fidelity(a,b,[dict(image_id=1),dict(image_id=0)])
            extra['category_ids']=np.arange(8,dtype=np.int64)[::-1]
            np.savez(b,**tensors,**extra)
            with self.assertRaises(AssertionError):tensor_fidelity(a,b,[dict(image_id=0),dict(image_id=1)])

    def test_result_tensor_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'current.npz';b=Path(tmp)/'archive.npz'
            tensors={k:np.zeros((2,3),np.float64) for k in ['precision','recall','scores']}
            np.savez(a,**tensors);tensors['precision'][0,0]=1e-15;np.savez(b,**tensors)
            with self.assertRaises(AssertionError):tensor_fidelity(a,b,[])


if __name__=='__main__':unittest.main()
