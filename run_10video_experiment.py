#!/usr/bin/env python3
"""
First 10-video Adaptive Vision experiment.

Selects 10 diverse test videos from Prep-Data/images/test, runs the
complete adaptive pipeline (FULL/LOCAL/SKIP scheduling, optical-flow
propagation, environmental adaptation, uncertainty-based re-detection,
temporal tracking, telemetry), evaluates against ground-truth labels,
and writes per-video + aggregate results to runs/10_video_adaptive/.

Usage:
    python run_10video_experiment.py
"""
import csv
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

# ── project root ──────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
EXTRACTED = PROJECT / "extracted"
sys.path.insert(0, str(EXTRACTED))

from adaptive_pipeline.adaptive_framing.sequence import discover_sequences
from adaptive_pipeline.detection.detector import YOLODetector
from adaptive_pipeline.evaluation.evaluator import DetectionEvaluator, load_yolo_labels
from adaptive_pipeline.experiments.presets import PRESETS
from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.shared.config import DATASET_NAMES, PipelineConfig
from adaptive_pipeline.telemetry.metrics import Telemetry
from adaptive_pipeline.run import STALE_BINS, _stale_bin

# ── constants ─────────────────────────────────────────────────────────────
IMAGES_DIR = str(PROJECT / "Prep-Data" / "images" / "test")
SEQ_REGEX  = r"^(?P<video>.+?)_f(?P<frame>\d+)$"
OUT_DIR    = str(PROJECT / "runs" / "10_video_adaptive")
PRESET     = "adaptive_full"

# 10 hand-picked videos: diverse frame counts; 3 night scenes, mix of long/medium
# This selection captures variety in scene type and length.
SELECTED_VIDEOS = [
    "night_817",   # 300 frames – night
    "night_1869",  # 300 frames – night
    "night_1695",  # 300 frames – night
    "day_966",     # 780 frames – long
    "day_18",      # 780 frames – long
    "day_974",     # 600 frames – medium-long
    "day_590",     # 600 frames – medium-long
    "day_1078",    # 390 frames – medium
    "day_810",     # 301 frames – edge case (301 ≠ 300)
    "day_1013",    # 300 frames – standard
]


def build_config():
    """Build the adaptive_full config (stock defaults, no tuning overrides)."""
    return PipelineConfig().with_overrides(**PRESETS[PRESET])


def select_sequences(all_seqs):
    """Filter to exactly the 10 chosen videos, preserving frame order."""
    by_name = {s.name: s for s in all_seqs}
    selected = []
    missing = []
    for v in SELECTED_VIDEOS:
        if v in by_name:
            selected.append(by_name[v])
        else:
            missing.append(v)
    if missing:
        raise RuntimeError(f"Missing videos: {missing}")
    return selected


def run_single_video(cfg, seq, detector, ev_all, tel_all):
    """Run the full adaptive pipeline on one video sequence.

    Returns (per_video_result_dict, n_frames).
    """
    pipe = AdaptivePipeline(cfg, detector)
    detector.cfg = cfg
    ev = DetectionEvaluator(DATASET_NAMES)
    tel = Telemetry()
    full_gf = None

    for fi, path in enumerate(seq.frames):
        import cv2
        td = time.perf_counter()
        frame = cv2.imread(path)
        decode_ms = (time.perf_counter() - td) * 1e3
        if frame is None:
            raise IOError(f"Cannot read frame: {path}")

        if full_gf is None:
            full_gf = detector.full_gflops(*frame.shape[:2])

        out, rec = pipe.step(frame, fi)
        rec["video"] = seq.name
        rec["decode_ms"] = decode_ms
        tel.add(rec)
        tel_all.add(rec)

        # Evaluate against ground truth
        if seq.labels is not None:
            h, w = frame.shape[:2]
            gts = load_yolo_labels(seq.labels[fi], w, h)
            tags = {"decision": rec["decision"],
                    "stale": _stale_bin(rec["since_full"]),
                    "video": seq.name}
            ev.add(out, gts, tags)
            ev_all.add(out, gts, tags)

    eff = tel.summary(full_gf)
    det_res = ev.compute() if ev.tp else None
    det_by_dec = {}
    if ev.tp:
        for d in ("FULL", "LOCAL", "SKIP"):
            r = ev.compute(lambda t, d=d: t["decision"] == d)
            if r is not None:
                det_by_dec[d] = r

    return {
        "video": seq.name,
        "n_frames": len(seq.frames),
        "detection": det_res,
        "detection_by_decision": det_by_dec,
        "efficiency": eff,
    }, len(seq.frames), tel


def sanity_check(per_video_results, selected_seqs, agg_eff, agg_det):
    """Run sanity checks and collect warnings."""
    warnings = []

    # 1. Check video count
    if len(per_video_results) != 10:
        warnings.append(f"Expected 10 videos, got {len(per_video_results)}")

    # 2. Check frame counts match
    total_frames_expected = sum(len(s.frames) for s in selected_seqs)
    total_frames_actual = sum(r["n_frames"] for r in per_video_results)
    if total_frames_expected != total_frames_actual:
        warnings.append(f"Frame count mismatch: expected {total_frames_expected}, got {total_frames_actual}")

    # 3. Check efficiency totals
    if agg_eff:
        eff_total = agg_eff.get("full_frames", 0) + agg_eff.get("local_frames", 0) + agg_eff.get("skip_frames", 0)
        if eff_total != agg_eff.get("frames", -1):
            warnings.append(f"FULL+LOCAL+SKIP={eff_total} ≠ total frames={agg_eff.get('frames')}")

    # 4. Per-video checks
    for r in per_video_results:
        eff = r.get("efficiency", {})
        if not eff:
            warnings.append(f"{r['video']}: missing efficiency data")
            continue

        # FULL+LOCAL+SKIP must sum to total
        f_total = eff.get("full_frames", 0) + eff.get("local_frames", 0) + eff.get("skip_frames", 0)
        if f_total != eff.get("frames", -1):
            warnings.append(f"{r['video']}: FULL+LOCAL+SKIP={f_total} ≠ frames={eff.get('frames')}")

        # No NaN in key metrics
        det = r.get("detection")
        if det:
            for key in ("mAP50", "mAP50-95", "precision", "recall"):
                val = det.get(key)
                if val is not None and (np.isnan(val) or np.isinf(val)):
                    warnings.append(f"{r['video']}: {key} is {val}")

        # FPS sanity
        fps = eff.get("fps_end_to_end")
        if fps is not None and (fps <= 0 or fps > 10000):
            warnings.append(f"{r['video']}: suspicious FPS={fps}")

        # detector_skip_rate in [0, 1]
        dsr = eff.get("detector_skip_rate")
        if dsr is not None and not (0 <= dsr <= 1):
            warnings.append(f"{r['video']}: detector_skip_rate={dsr} out of range")

    # 5. Per-video frame totals match aggregate
    if agg_eff:
        per_video_frame_sum = sum(r["n_frames"] for r in per_video_results)
        if per_video_frame_sum != agg_eff.get("frames", -1):
            warnings.append(f"Per-video frame sum ({per_video_frame_sum}) ≠ aggregate frames ({agg_eff.get('frames')})")

    # 6. Detection metrics aggregate check
    if agg_det:
        for key in ("mAP50", "mAP50-95", "precision", "recall"):
            val = agg_det.get(key)
            if val is not None and (np.isnan(val) or np.isinf(val)):
                warnings.append(f"Aggregate {key} is {val}")

    return warnings


def main():
    t_experiment_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("  ADAPTIVE VISION — 10-VIDEO EXPERIMENT")
    print("=" * 72)

    # ── PHASE 1: Data / Sequence Setup ────────────────────────────────────
    print("\n▸ PHASE 1: Discovering sequences...")
    all_seqs = discover_sequences(IMAGES_DIR, seq_regex=SEQ_REGEX)
    print(f"  Found {len(all_seqs)} total sequences, "
          f"{sum(len(s.frames) for s in all_seqs)} total frames")

    selected = select_sequences(all_seqs)
    total_frames = sum(len(s.frames) for s in selected)
    print(f"  Selected {len(selected)} videos, {total_frames} total frames:")
    for s in selected:
        print(f"    • {s.name:20s}  {len(s.frames):5d} frames  "
              f"(labels: {'✓' if s.labels else '✗'})")

    # Verify all label files exist
    missing_labels = 0
    for s in selected:
        if s.labels:
            for lp in s.labels:
                if not os.path.exists(lp):
                    missing_labels += 1
    if missing_labels:
        print(f"  ⚠ WARNING: {missing_labels} label files missing")

    # Save video manifest
    manifest = []
    for s in selected:
        manifest.append({
            "video": s.name,
            "n_frames": len(s.frames),
            "first_frame": os.path.basename(s.frames[0]),
            "last_frame": os.path.basename(s.frames[-1]),
            "has_labels": s.labels is not None,
        })
    with open(os.path.join(OUT_DIR, "video_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved to {OUT_DIR}/video_manifest.json")

    # ── PHASE 2: Adaptive Execution ───────────────────────────────────────
    print("\n▸ PHASE 2: Running adaptive pipeline (preset: {})...".format(PRESET))
    cfg = build_config()
    print(f"  Policy: {cfg.policy}, Propagation: {cfg.propagation}")
    print(f"  k_max={cfg.k_max}, novelty_thr={cfg.novelty_thr}, "
          f"use_local={cfg.use_local}, use_uncertainty={cfg.use_uncertainty}")
    print(f"  Model: {cfg.model_path}")

    # Initialize detector once (share across videos)
    detector = YOLODetector(cfg)

    # Warmup
    import cv2
    warmup_frame = cv2.imread(selected[0].frames[0])
    if warmup_frame is None:
        raise IOError(f"Cannot read warmup frame: {selected[0].frames[0]}")
    print("  Warming up detector...")
    detector.warmup(warmup_frame)
    print("  Warmup complete.")

    # Aggregate evaluator and telemetry
    ev_all = DetectionEvaluator(DATASET_NAMES)
    tel_all = Telemetry()

    per_video_results = []
    per_video_telemetries = []
    errors = []

    for i, seq in enumerate(selected):
        print(f"\n  [{i+1}/10] {seq.name} ({len(seq.frames)} frames)...", end="", flush=True)
        t0 = time.time()
        try:
            result, n_frames, vid_tel = run_single_video(cfg, seq, detector, ev_all, tel_all)
            elapsed = time.time() - t0
            eff = result.get("efficiency", {})
            det = result.get("detection", {})
            print(f"  done in {elapsed:.1f}s  "
                  f"FULL={eff.get('full_frames', '?')} LOCAL={eff.get('local_frames', '?')} "
                  f"SKIP={eff.get('skip_frames', '?')}  "
                  f"mAP50={det.get('mAP50', '?') if det else '?'}  "
                  f"FPS={eff.get('fps_end_to_end', '?'):.1f}" if isinstance(eff.get('fps_end_to_end'), (int, float)) else
                  f"  done in {elapsed:.1f}s")
            per_video_results.append(result)
            per_video_telemetries.append(vid_tel)

            # Save per-video CSV
            vid_out = os.path.join(OUT_DIR, "per_video", seq.name)
            os.makedirs(vid_out, exist_ok=True)
            vid_tel.to_csv(os.path.join(vid_out, "frames.csv"))
            with open(os.path.join(vid_out, "metrics.json"), "w") as f:
                json.dump(result, f, indent=2, default=float)

        except Exception as e:
            elapsed = time.time() - t0
            err_msg = f"{seq.name}: {type(e).__name__}: {e}"
            print(f"  FAILED after {elapsed:.1f}s: {err_msg}")
            errors.append(err_msg)
            traceback.print_exc()

    # ── PHASE 3: Evaluation ───────────────────────────────────────────────
    print("\n\n▸ PHASE 3: Computing aggregate metrics...")

    # Aggregate detection
    full_gf = detector.full_gflops(*warmup_frame.shape[:2])
    agg_eff = tel_all.summary(full_gf)
    agg_det = ev_all.compute() if ev_all.tp else None
    agg_det_by_dec = {}
    if ev_all.tp:
        for d in ("FULL", "LOCAL", "SKIP"):
            r = ev_all.compute(lambda t, d=d: t["decision"] == d)
            if r is not None:
                agg_det_by_dec[d] = r
    agg_det_by_stale = {}
    if ev_all.tp:
        for *_, b in STALE_BINS:
            r = ev_all.compute(lambda t, b=b: t["stale"] == b)
            if r is not None:
                agg_det_by_stale[b] = r

    aggregate_result = {
        "config": cfg.to_dict(),
        "preset": PRESET,
        "n_videos": len(per_video_results),
        "n_frames_total": sum(r["n_frames"] for r in per_video_results),
        "detection": agg_det,
        "detection_by_decision": agg_det_by_dec,
        "detection_by_staleness": agg_det_by_stale,
        "efficiency": agg_eff,
    }

    # ── PHASE 4: Save Results ─────────────────────────────────────────────
    print("\n▸ PHASE 4: Saving results...")

    with open(os.path.join(OUT_DIR, "aggregate_metrics.json"), "w") as f:
        json.dump(aggregate_result, f, indent=2, default=float)
    print(f"  Aggregate metrics → {OUT_DIR}/aggregate_metrics.json")

    tel_all.to_csv(os.path.join(OUT_DIR, "all_frames.csv"))
    print(f"  All-frames CSV → {OUT_DIR}/all_frames.csv")

    if ev_all.tp:
        ev_all.save(os.path.join(OUT_DIR, "eval_stats.npz"))
        print(f"  Eval stats → {OUT_DIR}/eval_stats.npz")

    # Per-video summary CSV
    summary_rows = []
    for r in per_video_results:
        eff = r.get("efficiency", {})
        det = r.get("detection", {})
        summary_rows.append({
            "video": r["video"],
            "n_frames": r["n_frames"],
            "full_frames": eff.get("full_frames", ""),
            "local_frames": eff.get("local_frames", ""),
            "skip_frames": eff.get("skip_frames", ""),
            "detector_skip_rate": eff.get("detector_skip_rate", ""),
            "local_redetect_rate": eff.get("local_redetect_rate", ""),
            "tracking_only_rate": eff.get("tracking_only_rate", ""),
            "full_detector_calls": eff.get("full_detector_calls", ""),
            "roi_crops_total": eff.get("roi_crops_total", ""),
            "latency_mean_ms": eff.get("latency_mean_ms", ""),
            "latency_p50_ms": eff.get("latency_p50_ms", ""),
            "latency_p95_ms": eff.get("latency_p95_ms", ""),
            "fps_end_to_end": eff.get("fps_end_to_end", ""),
            "gflops_per_frame": eff.get("gflops_per_frame", ""),
            "gflops_fraction_of_baseline": eff.get("gflops_fraction_of_baseline", ""),
            "precision": det.get("precision", "") if det else "",
            "recall": det.get("recall", "") if det else "",
            "mAP50": det.get("mAP50", "") if det else "",
            "mAP50-95": det.get("mAP50-95", "") if det else "",
            "overhead_ms_mean": eff.get("overhead_ms_mean", ""),
            "peak_rss_mb": eff.get("peak_rss_mb", ""),
            "trigger_counts": json.dumps(eff.get("trigger_counts", {})),
        })
    summary_csv_path = os.path.join(OUT_DIR, "per_video_summary.csv")
    if summary_rows:
        with open(summary_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
    print(f"  Per-video summary CSV → {summary_csv_path}")

    # ── PHASE 5: Sanity Check ─────────────────────────────────────────────
    print("\n▸ PHASE 5: Sanity checks...")
    warnings = sanity_check(per_video_results, selected, agg_eff, agg_det)
    if warnings:
        print("  ⚠ WARNINGS:")
        for w in warnings:
            print(f"    • {w}")
    else:
        print("  ✓ All sanity checks passed.")

    # ── FINAL REPORT ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_experiment_start
    print("\n" + "=" * 72)
    print("  FINAL REPORT")
    print("=" * 72)

    # 1. Videos used
    print("\n  1. VIDEOS USED:")
    for r in per_video_results:
        print(f"     • {r['video']:20s}  {r['n_frames']:5d} frames")

    # 2. Frames per video (already shown above)

    # 3. Pipeline success
    success = len(errors) == 0 and len(per_video_results) == 10
    print(f"\n  3. PIPELINE STATUS: {'✓ SUCCESS' if success else '✗ FAILED'}")
    if errors:
        print(f"     Errors ({len(errors)}):")
        for e in errors:
            print(f"       • {e}")

    # 4. Aggregate metrics
    print("\n  4. AGGREGATE DETECTION METRICS:")
    if agg_det:
        for k in ("precision", "recall", "mAP50", "mAP50-95"):
            print(f"     {k:15s}: {agg_det[k]:.4f}")
        if "per_class_AP50-95" in agg_det:
            print("     Per-class AP50-95:")
            for cls, ap in agg_det["per_class_AP50-95"].items():
                print(f"       {cls:20s}: {ap:.4f}")
    else:
        print("     No detection metrics available")

    print("\n  5. AGGREGATE EFFICIENCY:")
    if agg_eff:
        for k in ("frames", "full_frames", "local_frames", "skip_frames",
                   "full_detector_calls", "detector_skip_rate",
                   "local_redetect_rate", "tracking_only_rate",
                   "roi_crops_total", "roi_pass_rate", "roi_confirm_rate",
                   "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
                   "latency_max_ms", "fps_end_to_end",
                   "gflops_per_frame", "gflops_fraction_of_baseline",
                   "overhead_ms_mean", "peak_rss_mb",
                   "cpu_ms_per_frame"):
            val = agg_eff.get(k)
            if val is not None:
                if isinstance(val, float):
                    print(f"     {k:32s}: {val:.4f}")
                else:
                    print(f"     {k:32s}: {val}")
        if "trigger_counts" in agg_eff:
            print("     Trigger counts:")
            for trig, cnt in sorted(agg_eff["trigger_counts"].items(), key=lambda x: -x[1]):
                print(f"       {trig:25s}: {cnt}")

    # 6. FULL/LOCAL/SKIP distribution
    print("\n  6. FULL/LOCAL/SKIP DISTRIBUTION:")
    if agg_eff:
        total = agg_eff.get("frames", 1)
        for d in ("full_frames", "local_frames", "skip_frames"):
            v = agg_eff.get(d, 0)
            pct = 100.0 * v / total if total else 0
            print(f"     {d:15s}: {v:6d}  ({pct:5.1f}%)")

    # 7. Detector computation saved
    print("\n  7. DETECTOR COMPUTATION SAVED:")
    if agg_eff:
        dsr = agg_eff.get("detector_skip_rate", 0)
        gf_frac = agg_eff.get("gflops_fraction_of_baseline")
        print(f"     Detector skip rate:       {dsr:.2%}")
        if gf_frac is not None:
            print(f"     GFLOPs fraction of baseline: {gf_frac:.4f} ({(1-gf_frac):.1%} saved)")

    # 8. Detection by decision type
    print("\n  8. DETECTION QUALITY BY DECISION TYPE:")
    if agg_det_by_dec:
        for d in ("FULL", "LOCAL", "SKIP"):
            r = agg_det_by_dec.get(d)
            if r:
                print(f"     {d}: mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50-95']:.4f}  "
                      f"P={r['precision']:.4f}  R={r['recall']:.4f}  "
                      f"images={r['images']}  instances={r['instances']}")

    # 9. Errors/warnings
    print(f"\n  9. ERRORS: {len(errors)}, WARNINGS: {len(warnings)}")

    # 10. Output locations
    print(f"\n  10. RESULTS SAVED TO:")
    print(f"      {OUT_DIR}/")
    print(f"      ├── aggregate_metrics.json")
    print(f"      ├── all_frames.csv")
    print(f"      ├── eval_stats.npz")
    print(f"      ├── video_manifest.json")
    print(f"      ├── per_video_summary.csv")
    print(f"      └── per_video/")
    for r in per_video_results:
        print(f"          ├── {r['video']}/metrics.json")
        print(f"          └── {r['video']}/frames.csv")

    # 11. Readiness
    print(f"\n  TOTAL EXPERIMENT TIME: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    if success and len(warnings) == 0:
        print("  ✓ System is READY for a larger validation experiment.")
    elif success:
        print("  ⚠ Pipeline ran successfully but warnings should be reviewed before scaling.")
    else:
        print("  ✗ Pipeline encountered errors — NOT ready for larger experiments.")

    # Save the final summary as JSON
    final_summary = {
        "experiment": "10_video_adaptive",
        "preset": PRESET,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": elapsed_total,
        "n_videos": len(per_video_results),
        "n_frames_total": sum(r["n_frames"] for r in per_video_results),
        "success": success,
        "errors": errors,
        "warnings": warnings,
        "aggregate_detection": agg_det,
        "aggregate_efficiency": agg_eff,
        "per_video": [{
            "video": r["video"],
            "n_frames": r["n_frames"],
            "detection": r.get("detection"),
            "efficiency_summary": {
                k: r.get("efficiency", {}).get(k) for k in (
                    "full_frames", "local_frames", "skip_frames",
                    "detector_skip_rate", "fps_end_to_end", "gflops_per_frame",
                    "gflops_fraction_of_baseline", "latency_mean_ms",
                    "trigger_counts",
                )
            },
        } for r in per_video_results],
        "output_dir": OUT_DIR,
    }
    with open(os.path.join(OUT_DIR, "experiment_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2, default=float)
    print(f"\n  Full summary → {OUT_DIR}/experiment_summary.json")
    print("=" * 72)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
