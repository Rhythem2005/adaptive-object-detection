"""Synthetic COCO protocol/conversion tests. No images, weights or inference."""
import copy
import unittest

import numpy as np

from adaptive_pipeline.evaluation.coco import (prepare_ground_truth, prediction_records, make_coco,
    run_coco, coco_metrics, official_match_arrays, sampled_ap, cap_mask, fixed_operating_point)
from adaptive_pipeline.evaluation.evaluator import match_predictions, _compute_ap
from adaptive_pipeline.shared.boxes import iou_matrix


def dataset(annotations=None, images=1):
    return {'images':[{'id':i,'width':200,'height':200} for i in range(images)],
            'categories':[{'id':0,'name':'person'}],
            'annotations':annotations if annotations is not None else [
                {'id':0,'image_id':0,'category_id':0,'bbox':[0,0,10,10],'area':100,'iscrowd':0,'ignore':False}]}


def score(data, rows, max_dets=(1,10,100), normalize=True):
    gtdata = prepare_ground_truth(data)[0] if normalize else data
    records,_=prediction_records(rows,{0:0})
    gt,dt=make_coco(gtdata,records)
    ev,_=run_coco(gt,dt,max_dets=max_dets)
    return ev,coco_metrics(ev,max_dets[-1])


class CocoScoringTests(unittest.TestCase):
    def test_positive_annotation_ids_are_necessary_and_source_is_unchanged(self):
        data=dataset(); original=copy.deepcopy(data)
        rows={0:np.array([[0,0,10,10,.9,0]])}
        _,broken=score(data,rows,normalize=False)
        _,fixed=score(data,rows)
        self.assertEqual(broken['AP'],0.)
        self.assertAlmostEqual(fixed['AP'],1.)
        self.assertEqual(data,original)
        out,audit=prepare_ground_truth(data)
        self.assertEqual(out['annotations'][0]['id'],1)
        self.assertEqual(audit['annotation_id_map'],[{'source_id':0,'coco_id':1}])

    def test_xyxy_conversion_and_noncontiguous_category_mapping(self):
        records,audit=prediction_records({9:np.array([[10,20,35,60,.8,1]])},{1:7})
        self.assertEqual(records[0]['bbox'],[10.,20.,25.,40.])
        self.assertEqual(records[0]['category_id'],7)
        self.assertEqual(records[0]['image_id'],9)
        self.assertEqual(audit['predictions'],1)

    def test_crowds_are_many_to_one_prediction_area_ignored(self):
        data=dataset();data['annotations'].append({'id':2,'image_id':0,'category_id':0,'bbox':[20,20,100,100],
                                                   'area':10000,'iscrowd':1,'ignore':False})
        rows={0:np.array([[30,30,35,35,.99,0],[50,50,55,55,.98,0],[0,0,10,10,.8,0]])}
        ev,res=score(data,rows);tp,ignored,keep=official_match_arrays(ev,3)
        self.assertAlmostEqual(res['AP'],1.)
        self.assertTrue(ignored[:2].all());self.assertFalse(tp[:2].any());self.assertTrue(tp[2].all())

    def test_empty_annotated_frame_preserved_and_its_false_positive_counts(self):
        ev,res=score(dataset(images=2),{0:np.array([[0,0,10,10,.7,0]]),1:np.array([[50,50,60,60,.9,0]])})
        self.assertAlmostEqual(res['AP'],.5)
        point=fixed_operating_point(ev)
        self.assertEqual(point['counts'],{'targets':1,'TP':1,'FP':1})
        self.assertEqual(point['micro_recall'],1.)

    def test_no_predictions_and_no_target_categories_are_valid(self):
        _,res=score(dataset(),{0:np.zeros((0,6))})
        self.assertEqual(res['AP'],0.)
        _,res=score(dataset([]),{0:np.zeros((0,6))})
        self.assertIsNone(res['AP'])

    def test_zero_area_predictions_are_retained_and_false_positive(self):
        rows={0:np.array([[0,0,10,0,.99,0],[0,0,10,10,.8,0]])}
        ev,res=score(dataset(),rows)
        tp,ignored,keep=official_match_arrays(ev,2)
        self.assertFalse(tp[0].any());self.assertFalse(ignored[0].any());self.assertTrue(keep.all())
        self.assertAlmostEqual(res['AP'],.5)

    def test_invalid_predictions_categories_and_annotations_fail(self):
        for rows in [[[0,0,10,10,float('nan'),0]],[[10,0,0,10,.5,0]],[[0,0,10,10,.5,1]],[[0,0,10,10,.5,.5]]]:
            with self.assertRaises(ValueError):prediction_records({0:np.array(rows)},{0:0})
        bad=dataset();bad['annotations'][0]['ignore']=True
        with self.assertRaisesRegex(ValueError,'Ignore-only'):prepare_ground_truth(bad)
        bad=dataset();bad['annotations'][0]['image_id']=7
        with self.assertRaises(ValueError):prepare_ground_truth(bad)

    def test_outside_image_boxes_preserved_not_clipped(self):
        data=dataset();data['annotations'][0].update(bbox=[195,10,6,10],area=60)
        norm,audit=prepare_ground_truth(data)
        self.assertEqual(norm['annotations'][0]['bbox'],[195,10,6,10])
        self.assertEqual(audit['out_of_bounds_annotation_ids_retained'],[0])

    def test_standard_cap100_vs_auxiliary_cap300(self):
        fp=[[50,50,60,60,1-i*.001,0] for i in range(100)]
        rows={0:np.array(fp+[[0,0,10,10,.1,0]])}
        _,standard=score(dataset(),rows)
        _,aux=score(dataset(),rows,(1,10,300))
        self.assertEqual(standard['AP'],0.)
        self.assertAlmostEqual(aux['AP'],1/101)

    def test_cap_is_per_image_category_and_stable_for_ties(self):
        mask=cap_mask(np.array([.8,.8,.8,.8,.8]),np.array([0,0,1,1,0]),np.array([4,1]),1)
        np.testing.assert_array_equal(mask,[True,False,True,False,True])
        _,a=score(dataset(),{0:np.array([[50,50,60,60,.8,0],[0,0,10,10,.8,0]])})
        _,b=score(dataset(),{0:np.array([[0,0,10,10,.8,0],[50,50,60,60,.8,0]])})
        self.assertAlmostEqual(a['AP'],.5);self.assertAlmostEqual(b['AP'],1.)

    def test_official_score_greedy_matching_can_use_an_alternative_gt(self):
        data=dataset();data['annotations'].append({'id':1,'image_id':0,'category_id':0,'bbox':[4.5,0,10,10],'area':100,'iscrowd':0})
        rows=np.array([[0,0,10,10,.9,0],[2,0,12,10,.8,0]])
        ev,_=score(data,{0:rows});tp,_,_=official_match_arrays(ev,2)
        old=match_predictions(rows[:,5],np.array([0,0]),iou_matrix(np.array([[0,0,10,10],[4.5,0,14.5,10]]),rows[:,:4]))
        self.assertEqual(int(tp[:,0].sum()),2)
        self.assertEqual(int(old[:,0].sum()),1)

    def test_sampled_ap_parity_and_legacy_integration_difference(self):
        ev,res=score(dataset(images=2),{0:np.array([[0,0,10,10,.7,0]]),1:np.array([[40,40,50,50,.9,0]])})
        tp,ignored,keep=official_match_arrays(ev,2)
        result=sampled_ap(tp,np.array([.7,.9]),np.array([0,0]),np.array([0]),ignored,keep=keep)
        self.assertAlmostEqual(result['AP'],res['AP'])
        self.assertAlmostEqual(_compute_ap(np.array([1.]),np.array([1.])),.995)
        self.assertAlmostEqual(sampled_ap(np.ones((1,10),bool),np.array([.9]),np.array([0]),np.array([0]),np.zeros((1,10),bool))['AP'],1.)

    def test_area_metrics_and_independent_subset_scoring(self):
        data=dataset(images=2)
        rows={0:np.array([[0,0,10,10,.8,0]]),1:np.array([[20,20,30,30,.9,0]])}
        norm,_=prepare_ground_truth(data);records,_=prediction_records(rows,{0:0});gt,dt=make_coco(norm,records)
        subset,_=run_coco(gt,dt,image_ids=[0])
        res=coco_metrics(subset)
        self.assertAlmostEqual(res['AP'],1.)
        self.assertAlmostEqual(res['AP_small'],1.)
        self.assertIsNone(res['AP_large'])


if __name__=='__main__':unittest.main()
