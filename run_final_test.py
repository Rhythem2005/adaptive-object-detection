#!/usr/bin/env python3
"""
Final Test Set Evaluation for Adaptive Vision
Uses the frozen configuration derived from the validation tuning phase.
"""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent
EXTRACTED = PROJECT / "extracted"
sys.path.insert(0, str(EXTRACTED))

from adaptive_pipeline.adaptive_framing.sequence import discover_sequences
from adaptive_pipeline.detection.detector import YOLODetector
from adaptive_pipeline.shared.config import PipelineConfig
from adaptive_pipeline.run import run_sequences

TEST_IMAGES = str(PROJECT / "Prep-Data" / "images" / "test")
SEQ_REGEX = r"^(?P<video>.+?)_f(?P<frame>\d+)$"
OUT_ROOT = str(PROJECT / "runs" / "final_test")

FROZEN_CONFIG = dict(
    k_max=4,
    novelty_thr=0.10,
    conf_decay=0.93,
    k_uncertain=2,
    max_local_rois=5,
    fail_frac_thr=0.20,
    use_local=True
)

def main():
    print("=" * 72)
    print("  FINAL TEST SET EVALUATION - ADAPTIVE VISION")
    print("=" * 72)
    os.makedirs(OUT_ROOT, exist_ok=True)
    
    seqs = discover_sequences(TEST_IMAGES, seq_regex=SEQ_REGEX)
    total_frames = sum(len(s.frames) for s in seqs)
    print(f"Found {len(seqs)} test sequences, {total_frames} total frames.")
    
    cfg = PipelineConfig().with_overrides(**FROZEN_CONFIG)
    
    print("\nInitializing detector...")
    detector = YOLODetector(cfg)
    
    print("\nRunning pipeline on the full test set...")
    t0 = time.time()
    
    # We use run_sequences from the original pipeline's run.py
    result, _ = run_sequences(
        cfg=cfg,
        sequences=seqs,
        out_dir=OUT_ROOT,
        detector=detector,
        save_preds=True,
        log_every=20
    )
    
    elapsed = time.time() - t0
    
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    det = result.get("detection", {})
    eff = result.get("efficiency", {})
    
    print("\n--- FINAL TEST METRICS ---")
    print(f"mAP50:           {det.get('mAP50', 0):.4f}")
    print(f"mAP50-95:        {det.get('mAP50-95', 0):.4f}")
    print(f"Recall:          {det.get('recall', 0):.4f}")
    print(f"Precision:       {det.get('precision', 0):.4f}")
    print(f"Skip Rate:       {eff.get('detector_skip_rate', 0):.1%}")
    print(f"LOCAL Rate:      {eff.get('local_redetect_rate', 0):.1%}")
    
    gflops_frac = eff.get("gflops_fraction_of_baseline", 1.0)
    print(f"Compute Savings: {1 - gflops_frac:.1%} (GFLOPs ratio: {gflops_frac:.3f})")
    print(f"End-to-End FPS:  {eff.get('fps_end_to_end', 0):.1f}")
    print("========================================================================")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
