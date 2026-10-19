"""Generated images and annotations only; no dataset or YOLO used."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from adaptive_pipeline.evaluation.argoverse import COCO_NAMES, load_argoverse
from adaptive_pipeline.evaluation.evaluator import DetectionEvaluator, match_ignored, IOUV
from adaptive_pipeline.evaluation.provenance import input_manifest
from adaptive_pipeline.run import run_sequences, build_config


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = {"categories": [{"id": i, "name": n} for i, n in enumerate(COCO_NAMES)],
                     "images": [], "annotations": [], "seq_dirs": ["val/log-a/ring_front_center", "val/log-b/ring_front_center"],
                     "sequences": ["log-a", "log-b"]}
        self.manifest = []
        for sid in range(2):
            for fid in range(2):
                iid = sid * 2 + fid
                name = f"ring_front_center_{100000000 + fid * 33300000}.jpg"
                path = f"images/val/sequence_{sid:02d}/{fid:06d}.jpg"
                p = self.root / path
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), np.zeros((30, 50, 3), np.uint8))
                self.data["images"].append(dict(id=iid, sid=sid, fid=fid, width=50, height=30, name=name, file_name=path))
                self.manifest.append(dict(image_id=iid, sequence=self.data["seq_dirs"][sid], frame_index=fid,
                                          source_path=f"/source/{name}", export_path=path))
        self.data["annotations"] = [dict(id=0, image_id=0, category_id=2, bbox=[10, 5, 20, 10], area=200,
                                          iscrowd=False, ignore=False),
                                     dict(id=1, image_id=1, category_id=0, bbox=[0, 0, 50, 30], area=1500,
                                          iscrowd=True, ignore=False)]

    def load(self):
        (self.root / "annotations.json").write_text(json.dumps(self.data))
        # Deliberately shuffled manifest: order must be recovered from indices.
        (self.root / "manifest.json").write_text(json.dumps(self.manifest[::-1]))
        return load_argoverse(self.root)

    def test_pixel_conversion_mapping_empty_frames_and_boundaries(self):
        seqs, names, mapping, report = self.load()
        self.assertEqual([len(s.frames) for s in seqs], [2, 2])
        self.assertEqual(seqs[0].image_ids, [0, 1])
        np.testing.assert_array_equal(seqs[0].targets[0], [[2, 10, 5, 30, 15]])
        self.assertEqual(seqs[0].targets[1].shape, (0, 5))
        self.assertEqual(seqs[1].targets[0].shape, (0, 5))
        self.assertEqual(mapping, {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 7: 5, 9: 6, 11: 7})
        self.assertEqual(report["sequences"][1]["empty_annotation_frames"], 2)
        self.assertEqual(report["crowd_instances"], 1)

    def test_unknown_category_fails(self):
        self.data["annotations"][0]["category_id"] = 99
        with self.assertRaisesRegex(ValueError, "unknown category"): self.load()

    def test_unsupported_taxonomy_fails(self):
        self.data["categories"][0]["name"] = "alien"
        with self.assertRaisesRegex(ValueError, "taxonomy"): self.load()

    def test_missing_or_extra_image_fails(self):
        (self.root / self.manifest[0]["export_path"]).unlink()
        with self.assertRaises(FileNotFoundError): self.load()

    def test_corrupt_image_and_dimensions_fail(self):
        self.data["images"][0]["height"] = 31
        with self.assertRaisesRegex(ValueError, "dimension disagreement"): self.load()

    def test_duplicate_and_orphan_ids_fail(self):
        self.data["annotations"].append(self.data["annotations"][0].copy())
        with self.assertRaisesRegex(ValueError, "duplicate"): self.load()
        self.data["annotations"].pop()
        self.data["annotations"][0]["image_id"] = 100
        with self.assertRaisesRegex(ValueError, "Orphan"): self.load()

    def test_manifest_disagreement_and_index_gaps_fail(self):
        self.manifest[1]["frame_index"] = 3
        with self.assertRaisesRegex(ValueError, "Frame index disagreement"): self.load()
        self.data["images"][1]["fid"] = 3
        with self.assertRaisesRegex(ValueError, "Noncontiguous"): self.load()

    def test_timestamp_and_sequence_disagreement_fail(self):
        self.manifest[1]["source_path"] = self.manifest[0]["source_path"]
        self.data["images"][1]["name"] = self.data["images"][0]["name"]
        with self.assertRaisesRegex(ValueError, "Non-monotonic"): self.load()
        self.manifest[1]["sequence"] = "val/wrong/ring_front_center"
        with self.assertRaisesRegex(ValueError, "Sequence identity"): self.load()

    def test_invalid_boxes_and_area_fail(self):
        for bbox, area in [([0, 0, 0, 10], 0), ([0, 0, float('nan'), 10], 100), ([0, 0, 10, 10], 90)]:
            self.data["annotations"][0].update(bbox=bbox, area=area)
            with self.assertRaises(ValueError): self.load()

    def test_out_of_bounds_preserved_and_reported(self):
        self.data["annotations"][0].update(bbox=[40, 5, 11, 10], area=110)
        seqs, _, _, report = self.load()
        self.assertEqual(seqs[0].targets[0][0, 3], 51)
        self.assertEqual(len(report["outside_image_boxes_retained"]), 1)

    def test_native_target_hash_changes(self):
        seqs, *_ = self.load()
        a = input_manifest(seqs, hash_images=True)
        seqs[0].targets[0][0, 1] += 1
        self.assertNotEqual(a["sha256"], input_manifest(seqs, hash_images=True)["sha256"])

    def test_runner_uses_native_targets_and_resets_sequences(self):
        seqs, names, mapping, _ = self.load()
        class Detector:
            def warmup(self, frame): pass
            def detect(self, frame): return np.zeros((0, 6), np.float32), 0.
            def detect_batch(self, crops, imgsz): raise AssertionError("No LOCAL expected")
            def full_gflops(self, h, w): return 1.
            def roi_gflops(self, n): return float(n)
        cfg = build_config("fixed2_093", []).with_overrides(class_map=mapping)
        res, _ = run_sequences(cfg, seqs, str(self.root / "result"), detector=Detector(), names=names)
        self.assertEqual(res["efficiency"]["frames"], 4)
        self.assertEqual(res["detection"]["images"], 4)
        self.assertEqual(res["detection"]["instances"], 1)
        import csv
        with (self.root / "result/frames.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["decision"] for r in rows if r["frame"] == "0"], ["FULL", "FULL"])
        self.assertEqual(set(res["detection_by_sequence"]), {"sequence_00", "sequence_01"})


class IgnoreTests(unittest.TestCase):
    def test_crowd_many_to_one_ioa_and_class_aware(self):
        preds = np.array([[1, 1, 2, 2, .9, 0], [4, 4, 5, 5, .8, 0], [1, 1, 2, 2, .7, 1]])
        crowd = np.array([[0, 0, 0, 10, 10, 1]])
        ignored = match_ignored(preds, np.zeros((3, 10), bool), crowd)
        self.assertTrue(ignored[:2].all())
        self.assertFalse(ignored[2].any())

    def test_regular_match_wins_and_ignore_is_threshold_specific(self):
        preds = np.array([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0]])
        correct = np.zeros((2, 10), bool); correct[0] = True
        crowd = np.array([[0, 0, 0, 6, 10, 1]])
        ignored = match_ignored(preds, correct, crowd)
        self.assertFalse(ignored[0].any())
        self.assertTrue(ignored[1, 0]); self.assertFalse(ignored[1, -1])

    def test_explicit_ignore_one_to_one(self):
        preds = np.array([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0]])
        ignored = match_ignored(preds, np.zeros((2, 10), bool), np.array([[0, 0, 0, 10, 10, 0]]))
        self.assertTrue(ignored[0].all()); self.assertFalse(ignored[1].any())

    def test_ignore_removed_from_ap_but_empty_frame_false_positive_counts(self):
        gt = np.array([[0, 10, 10, 20, 20]])
        tp = np.array([[10, 10, 20, 20, .8, 0]])
        fp = np.array([[0, 0, 1, 1, .9, 0]])
        ev = DetectionEvaluator(["person"])
        ev.add(np.concatenate([tp, fp]), gt, ignored_regions=np.array([[0, 0, 0, 2, 2, 1]]))
        reference = DetectionEvaluator(["person"]); reference.add(tp, gt)
        self.assertEqual(ev.compute()["mAP50-95"], reference.compute()["mAP50-95"])
        ev.add(fp, np.zeros((0, 5)))
        self.assertLess(ev.compute()["mAP50-95"], reference.compute()["mAP50-95"])


if __name__ == "__main__":
    unittest.main()
