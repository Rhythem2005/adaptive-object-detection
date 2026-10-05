#!/usr/bin/env python3
"""
Adaptive Vision — Validation Tuning & Pareto Analysis

Runs a controlled sweep of adaptive and fixed-K configurations on
validation data only.  Produces per-configuration metrics, SKIP failure
analysis, LOCAL analysis, and a Pareto recommendation.

Usage:
    python run_validation_tuning.py
"""
import csv, json, os, sys, time, traceback, warnings
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
EXTRACTED = PROJECT / "extracted"
sys.path.insert(0, str(EXTRACTED))

from adaptive_pipeline.adaptive_framing.sequence import discover_sequences
from adaptive_pipeline.detection.detector import YOLODetector
from adaptive_pipeline.evaluation.evaluator import DetectionEvaluator, load_yolo_labels, ap_per_class, IOUV
from adaptive_pipeline.experiments.presets import PRESETS
from adaptive_pipeline.pipeline import AdaptivePipeline
from adaptive_pipeline.shared.config import DATASET_NAMES, PipelineConfig
from adaptive_pipeline.telemetry.metrics import Telemetry
from adaptive_pipeline.run import STALE_BINS, _stale_bin

VAL_IMAGES = str(PROJECT / "Prep-Data" / "images" / "val")
SEQ_REGEX  = r"^(?P<video>.+?)_f(?P<frame>\d+)$"
OUT_ROOT   = str(PROJECT / "runs" / "validation_tuning")

# ── 15 validation videos: 3 night, diverse lengths ────────────────────────
SELECTED_VIDEOS = [
    # night
    "night_993",    # 300
    "night_1368",   # 300
    "night_1355",   # 300
    # long day
    "day_747",      # 840
    "day_782",      # 810
    "day_752",      # 780
    # medium day
    "day_858",      # 600
    "day_472",      # 600
    "day_1688",     # 600
    "day_1370",     # 600
    # shorter day
    "day_874",      # 480
    "day_690",      # 390
    "day_856",      # 360
    "day_962",      # 300
    "day_1060",     # 300
]

# ── Configurations to sweep ───────────────────────────────────────────────
# All parameter names are confirmed from PipelineConfig.
# [TUNE-ON-VAL] params: k_max, novelty_thr, fail_frac_thr, k_uncertain

ADAPTIVE_CONFIGS = {
    # A. Current defaults (aggressive skipping)
    "A_current": dict(),

    # B. Slightly more conservative: shorter staleness, lower novelty threshold
    "B_conservative": dict(
        k_max=5,
        novelty_thr=0.12,
        conf_decay=0.95,
        k_uncertain=2,
    ),

    # C. More conservative: even shorter staleness, broader LOCAL
    "C_more_conservative": dict(
        k_max=4,
        novelty_thr=0.10,
        conf_decay=0.93,
        k_uncertain=2,
        max_local_rois=5,
        fail_frac_thr=0.20,
    ),

    # D. Very conservative: frequent full passes
    "D_very_conservative": dict(
        k_max=3,
        novelty_thr=0.08,
        conf_decay=0.90,
        k_uncertain=1,
        max_local_rois=6,
        fail_frac_thr=0.15,
    ),

    # E. Conservative but with LOCAL disabled (FULL or SKIP only)
    "E_no_local": dict(
        k_max=4,
        novelty_thr=0.10,
        conf_decay=0.93,
        k_uncertain=2,
        use_local=False,
    ),

    # F. Current defaults but with slower confidence decay
    "F_slow_decay": dict(
        conf_decay=0.99,
    ),

    # G. k_max=4 only, rest defaults (isolate staleness effect)
    "G_kmax4": dict(
        k_max=4,
    ),

    # H. Very tight: k_max=2, aggressive triggers
    "H_very_tight": dict(
        k_max=2,
        novelty_thr=0.08,
        conf_decay=0.90,
        k_uncertain=1,
    ),
}

FIXED_K_CONFIGS = {
    "fixed2_flow":  dict(policy="fixed", fixed_k=2, propagation="flow"),
    "fixed3_flow":  dict(policy="fixed", fixed_k=3, propagation="flow"),
    "fixed4_flow":  dict(policy="fixed", fixed_k=4, propagation="flow"),
    "fixed6_flow":  dict(policy="fixed", fixed_k=6, propagation="flow"),
    "fixed8_flow":  dict(policy="fixed", fixed_k=8, propagation="flow"),
}


# ── helpers ───────────────────────────────────────────────────────────────
def build_cfg(overrides):
    return PipelineConfig().with_overrides(**overrides)


def run_config_on_sequences(cfg, sequences, out_dir, detector):
    """Run pipeline, collect per-frame records + evaluation."""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    detector.cfg = cfg
    pipe = AdaptivePipeline(cfg, detector)
    ev = DetectionEvaluator(DATASET_NAMES)
    tel = Telemetry()
    full_gf = detector.full_gflops(720, 1280)  # will be recomputed on first frame

    all_records = []
    first = True
    for seq in sequences:
        pipe.reset()
        for fi, path in enumerate(seq.frames):
            frame = cv2.imread(path)
            if frame is None:
                raise IOError(path)
            if first:
                full_gf = detector.full_gflops(*frame.shape[:2])
                first = False
            out, rec = pipe.step(frame, fi)
            rec["video"] = seq.name
            tel.add(rec)
            all_records.append(rec)
            if seq.labels is not None:
                h, w = frame.shape[:2]
                gts = load_yolo_labels(seq.labels[fi], w, h)
                tags = {"decision": rec["decision"],
                        "stale": _stale_bin(rec["since_full"]),
                        "video": seq.name}
                ev.add(out, gts, tags)

    eff = tel.summary(full_gf)
    det_all = ev.compute() if ev.tp else None
    det_by_dec = {}
    if ev.tp:
        for d in ("FULL", "LOCAL", "SKIP"):
            r = ev.compute(lambda t, d=d: t["decision"] == d)
            if r is not None:
                det_by_dec[d] = r
    det_by_stale = {}
    if ev.tp:
        for *_, b in STALE_BINS:
            r = ev.compute(lambda t, b=b: t["stale"] == b)
            if r is not None:
                det_by_stale[b] = r

    result = {
        "config": cfg.to_dict(),
        "detection": det_all,
        "detection_by_decision": det_by_dec,
        "detection_by_staleness": det_by_stale,
        "efficiency": eff,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    tel.to_csv(os.path.join(out_dir, "frames.csv"))
    if ev.tp:
        ev.save(os.path.join(out_dir, "eval_stats.npz"))
    return result, all_records, ev


def analyze_skip_degradation(all_records, ev):
    """Analyze SKIP quality by consecutive skip length, staleness, etc."""
    analysis = {}

    # Group frames by since_full (staleness)
    stale_groups = defaultdict(list)
    for i, rec in enumerate(all_records):
        sf = rec["since_full"]
        stale_groups[sf].append(i)

    # Quality vs staleness (binned)
    stale_quality = {}
    for sf in sorted(stale_groups.keys()):
        indices = stale_groups[sf]
        if not indices or not ev.tp:
            continue
        tp_cat = np.concatenate([ev.tp[i] for i in indices]) if indices else np.zeros((0, 10), bool)
        conf_cat = np.concatenate([ev.conf[i] for i in indices]) if indices else np.zeros(0)
        pcls_cat = np.concatenate([ev.pcls[i] for i in indices]) if indices else np.zeros(0, int)
        tcls_cat = np.concatenate([ev.tcls[i] for i in indices]) if indices else np.zeros(0, int)
        if len(tcls_cat) > 0 and len(tp_cat) > 0:
            try:
                r = ap_per_class(tp_cat, conf_cat, pcls_cat, tcls_cat)
                stale_quality[sf] = {
                    "n_frames": len(indices),
                    "n_gt": int(len(tcls_cat)),
                    "n_pred": int(len(pcls_cat)),
                    "mAP50": float(r["ap"][:, 0].mean()),
                    "mAP50-95": float(r["ap"].mean()),
                    "precision": float(r["p"].mean()),
                    "recall": float(r["r"].mean()),
                }
            except Exception:
                pass
    analysis["quality_by_staleness"] = stale_quality

    # Quality by decision type (already in results, but compute detailed)
    decision_stats = defaultdict(lambda: {"count": 0, "total_ms": 0, "gflops": 0})
    for rec in all_records:
        d = rec["decision"]
        decision_stats[d]["count"] += 1
        decision_stats[d]["total_ms"] += rec["t_total_ms"]
        decision_stats[d]["gflops"] += rec["gflops"]
    analysis["decision_stats"] = dict(decision_stats)

    # Trigger reason frequency
    trigger_counts = Counter()
    for rec in all_records:
        for r in filter(None, rec["reasons"].split("|")):
            trigger_counts[r] += 1
    analysis["trigger_counts"] = dict(trigger_counts)

    # LOCAL analysis
    local_records = [r for r in all_records if r["decision"] == "LOCAL"]
    local_escalated = [r for r in all_records if "local_escalate" in r["reasons"]]
    roi_pass_records = [r for r in all_records if r.get("roi_pass", 0) > 0]
    analysis["local"] = {
        "local_frames": len(local_records),
        "local_escalated_to_full": len(local_escalated),
        "roi_passes_total": len(roi_pass_records),
        "confirmation_rate": len(local_records) / max(1, len(roi_pass_records)),
        "escalation_rate": len(local_escalated) / max(1, len(roi_pass_records)),
        "mean_crops_on_local": float(np.mean([r["n_crops"] for r in local_records])) if local_records else 0,
        "mean_latency_local_ms": float(np.mean([r["t_total_ms"] for r in local_records])) if local_records else 0,
        "mean_latency_full_ms": float(np.mean([r["t_total_ms"] for r in all_records if r["decision"] == "FULL"])) if any(r["decision"] == "FULL" for r in all_records) else 0,
    }

    # Consecutive skip streak analysis
    streaks = []
    current_streak = 0
    for rec in all_records:
        if rec["decision"] == "SKIP":
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    if streaks:
        analysis["skip_streaks"] = {
            "count": len(streaks),
            "mean": float(np.mean(streaks)),
            "median": float(np.median(streaks)),
            "max": int(np.max(streaks)),
            "p90": float(np.percentile(streaks, 90)),
            "distribution": dict(Counter(streaks)),
        }

    # Track count over time
    track_counts = [rec["n_tracks"] for rec in all_records]
    if track_counts:
        analysis["track_stats"] = {
            "mean": float(np.mean(track_counts)),
            "max": int(np.max(track_counts)),
            "min": int(np.min(track_counts)),
        }

    return analysis


def main():
    t0 = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("=" * 72)
    print("  ADAPTIVE VISION — VALIDATION TUNING")
    print("=" * 72)

    # ── Phase 1: Select validation videos ─────────────────────────────────
    print("\n▸ PHASE 1: Discovering validation sequences...")
    all_seqs = discover_sequences(VAL_IMAGES, seq_regex=SEQ_REGEX)
    print(f"  Found {len(all_seqs)} total val sequences, "
          f"{sum(len(s.frames) for s in all_seqs)} total frames")

    by_name = {s.name: s for s in all_seqs}
    selected = []
    for v in SELECTED_VIDEOS:
        if v not in by_name:
            print(f"  ⚠ WARNING: video {v} not found, skipping")
            continue
        selected.append(by_name[v])

    total_frames = sum(len(s.frames) for s in selected)
    print(f"  Selected {len(selected)} videos, {total_frames} total frames:")
    for s in selected:
        print(f"    • {s.name:20s}  {len(s.frames):5d} frames")

    # Save manifest
    manifest = [{"video": s.name, "n_frames": len(s.frames)} for s in selected]
    with open(os.path.join(OUT_ROOT, "video_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Initialize detector (shared) ──────────────────────────────────────
    print("\n▸ Initializing detector...")
    import cv2
    base_cfg = PipelineConfig()
    detector = YOLODetector(base_cfg)
    warmup_frame = cv2.imread(selected[0].frames[0])
    detector.warmup(warmup_frame)
    print("  Detector warmed up.")

    # ── Phase 2 + 3: Run all configurations ──────────────────────────────
    all_configs = {}
    all_configs.update({f"adaptive_{k}": v for k, v in ADAPTIVE_CONFIGS.items()})
    all_configs.update(FIXED_K_CONFIGS)

    results = {}
    analyses = {}

    for i, (name, overrides) in enumerate(all_configs.items()):
        print(f"\n▸ [{i+1}/{len(all_configs)}] Running: {name}")
        cfg = build_cfg(overrides)
        print(f"  policy={cfg.policy} k_max={cfg.k_max} novelty_thr={cfg.novelty_thr} "
              f"conf_decay={cfg.conf_decay} k_uncertain={cfg.k_uncertain} use_local={cfg.use_local} "
              f"max_local_rois={cfg.max_local_rois}")

        out_dir = os.path.join(OUT_ROOT, name)
        t1 = time.time()

        try:
            result, records, ev = run_config_on_sequences(
                cfg, selected, out_dir, detector
            )
            elapsed = time.time() - t1

            eff = result.get("efficiency", {})
            det = result.get("detection", {})
            print(f"  Done in {elapsed:.0f}s  "
                  f"FULL={eff.get('full_frames','?')} LOCAL={eff.get('local_frames','?')} "
                  f"SKIP={eff.get('skip_frames','?')}  "
                  f"mAP50={det.get('mAP50','?'):.4f}  "
                  f"mAP50-95={det.get('mAP50-95','?'):.4f}  "
                  f"skip_rate={eff.get('detector_skip_rate','?'):.3f}  "
                  f"FPS={eff.get('fps_end_to_end','?'):.1f}" if isinstance(det, dict) else
                  f"  Done in {elapsed:.0f}s — no detection metrics")

            results[name] = result

            # Phase 5 analysis
            analysis = analyze_skip_degradation(records, ev)
            analyses[name] = analysis
            with open(os.path.join(out_dir, "analysis.json"), "w") as f:
                json.dump(analysis, f, indent=2, default=float)

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # ── Phase 6: Pareto Analysis ──────────────────────────────────────────
    print("\n\n▸ PHASE 6: Pareto Analysis")

    pareto_rows = []
    for name, res in results.items():
        if "error" in res:
            continue
        det = res.get("detection", {})
        eff = res.get("efficiency", {})
        det_full = (res.get("detection_by_decision", {}).get("FULL") or {})
        det_local = (res.get("detection_by_decision", {}).get("LOCAL") or {})
        det_skip = (res.get("detection_by_decision", {}).get("SKIP") or {})
        cfg = res.get("config", {})
        row = {
            "config": name,
            "policy": cfg.get("policy", "?"),
            "k_max": cfg.get("k_max", "?"),
            "novelty_thr": cfg.get("novelty_thr", "?"),
            "conf_decay": cfg.get("conf_decay", "?"),
            "k_uncertain": cfg.get("k_uncertain", "?"),
            "use_local": cfg.get("use_local", "?"),
            "max_local_rois": cfg.get("max_local_rois", "?"),
            "fail_frac_thr": cfg.get("fail_frac_thr", "?"),
            "frames": eff.get("frames", 0),
            "full_frames": eff.get("full_frames", 0),
            "local_frames": eff.get("local_frames", 0),
            "skip_frames": eff.get("skip_frames", 0),
            "full_calls": eff.get("full_detector_calls", 0),
            "skip_rate": eff.get("detector_skip_rate", 0),
            "local_rate": eff.get("local_redetect_rate", 0),
            "mAP50": det.get("mAP50", 0) if det else 0,
            "mAP50-95": det.get("mAP50-95", 0) if det else 0,
            "precision": det.get("precision", 0) if det else 0,
            "recall": det.get("recall", 0) if det else 0,
            "fps": eff.get("fps_end_to_end", 0),
            "latency_mean_ms": eff.get("latency_mean_ms", 0),
            "latency_p95_ms": eff.get("latency_p95_ms", 0),
            "gflops_per_frame": eff.get("gflops_per_frame", 0),
            "gflops_frac": eff.get("gflops_fraction_of_baseline", 0),
            "compute_saved": 1 - eff.get("gflops_fraction_of_baseline", 1) if eff.get("gflops_fraction_of_baseline") else 0,
            "FULL_mAP50": det_full.get("mAP50", "") if det_full else "",
            "LOCAL_mAP50": det_local.get("mAP50", "") if det_local else "",
            "SKIP_mAP50": det_skip.get("mAP50", "") if det_skip else "",
            "FULL_mAP50-95": det_full.get("mAP50-95", "") if det_full else "",
            "SKIP_mAP50-95": det_skip.get("mAP50-95", "") if det_skip else "",
            "roi_crops": eff.get("roi_crops_total", 0),
            "overhead_ms": eff.get("overhead_ms_mean", 0),
            "trigger_staleness": eff.get("trigger_counts", {}).get("staleness", 0),
            "trigger_novelty": eff.get("trigger_counts", {}).get("novelty", 0),
            "trigger_too_many": eff.get("trigger_counts", {}).get("too_many_refresh", 0),
            "trigger_local_esc": eff.get("trigger_counts", {}).get("local_escalate", 0),
        }
        pareto_rows.append(row)

    # Sort by mAP50-95 descending
    pareto_rows.sort(key=lambda r: -r["mAP50-95"])

    # Save Pareto table as CSV
    pareto_csv = os.path.join(OUT_ROOT, "pareto_table.csv")
    if pareto_rows:
        with open(pareto_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pareto_rows[0].keys())
            w.writeheader()
            w.writerows(pareto_rows)

    # Identify Pareto-optimal configs (non-dominated on mAP50-95 vs gflops_frac)
    pareto_optimal = []
    for r in pareto_rows:
        dominated = False
        for other in pareto_rows:
            if other["config"] == r["config"]:
                continue
            # other dominates r if: better quality AND less compute
            if other["mAP50-95"] >= r["mAP50-95"] and other["gflops_frac"] <= r["gflops_frac"]:
                if other["mAP50-95"] > r["mAP50-95"] or other["gflops_frac"] < r["gflops_frac"]:
                    dominated = True
                    break
        if not dominated:
            pareto_optimal.append(r["config"])

    # ── Print results ─────────────────────────────────────────────────────
    print("\n  PARETO TABLE (sorted by mAP50-95):")
    print(f"  {'Config':35s} {'Skip%':>6s} {'mAP50':>7s} {'mAP50-95':>9s} {'Recall':>7s} {'Prec':>7s} {'FPS':>6s} {'GFLOPs%':>8s} {'SKIP_mAP50':>10s} {'Pareto':>7s}")
    print("  " + "-" * 110)
    for r in pareto_rows:
        marker = " ★" if r["config"] in pareto_optimal else ""
        print(f"  {r['config']:35s} {r['skip_rate']:6.1%} {r['mAP50']:7.4f} {r['mAP50-95']:9.4f} "
              f"{r['recall']:7.4f} {r['precision']:7.4f} {r['fps']:6.1f} {r['gflops_frac']:8.3f} "
              f"{r.get('SKIP_mAP50',''):>10} {marker}")

    # ── Recommendation ────────────────────────────────────────────────────
    # Best balanced: highest mAP50-95 among configs with at least 15% compute savings
    balanced = [r for r in pareto_rows if r["compute_saved"] >= 0.15 and r["config"] in pareto_optimal]
    if balanced:
        rec = max(balanced, key=lambda r: r["mAP50-95"])
    else:
        rec = pareto_rows[0] if pareto_rows else None

    print(f"\n  RECOMMENDED CONFIG: {rec['config'] if rec else 'NONE'}")
    if rec:
        print(f"    mAP50={rec['mAP50']:.4f}  mAP50-95={rec['mAP50-95']:.4f}  "
              f"Recall={rec['recall']:.4f}  Precision={rec['precision']:.4f}")
        print(f"    Skip rate={rec['skip_rate']:.1%}  GFLOPs fraction={rec['gflops_frac']:.3f}  "
              f"Compute saved={rec['compute_saved']:.1%}")
        print(f"    FPS={rec['fps']:.1f}  Latency mean={rec['latency_mean_ms']:.1f}ms")

    # ── Save final summary ────────────────────────────────────────────────
    summary = {
        "experiment": "validation_tuning",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - t0,
        "n_videos": len(selected),
        "n_frames_total": total_frames,
        "videos": [s.name for s in selected],
        "configs_tested": list(all_configs.keys()),
        "pareto_table": pareto_rows,
        "pareto_optimal": pareto_optimal,
        "recommended": rec["config"] if rec else None,
        "recommended_metrics": rec if rec else None,
        "analyses": {name: a for name, a in analyses.items()},
    }
    with open(os.path.join(OUT_ROOT, "tuning_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    elapsed_total = time.time() - t0
    print(f"\n  Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  Results saved to: {OUT_ROOT}/")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
    sys.exit(main())
