"""Run one configuration over a split (or a video) and write results.

  python -m adaptive_pipeline.run --preset adaptive_full \
      --images data/test/images --out runs/test/adaptive_full

Outputs in --out: metrics.json (config + detection + efficiency),
frames.csv (one row per frame), optional preds.npz.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from .adaptive_framing.sequence import DEFAULT_SEQ_REGEX, discover_sequences, iter_video
from .detection.detector import YOLODetector
from .evaluation.evaluator import DetectionEvaluator, load_yolo_labels
from .experiments.presets import PRESETS
from .pipeline import AdaptivePipeline
from .shared.config import DATASET_NAMES, PipelineConfig
from .telemetry.metrics import Telemetry

STALE_BINS = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10 ** 9, "6+")]


def _stale_bin(s):
    for lo, hi, name in STALE_BINS:
        if lo <= s <= hi:
            return name
    return "6+"


def build_config(preset, overrides):
    cfg = PipelineConfig().with_overrides(**PRESETS[preset])
    kw = {}
    for item in overrides or []:
        k, v = item.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    return cfg.with_overrides(**kw) if kw else cfg


def run_sequences(cfg, sequences, out_dir, detector=None, max_frames=None, save_preds=False, log_every=20):
    os.makedirs(out_dir, exist_ok=True)
    detector = detector or YOLODetector(cfg)
    detector.cfg = cfg
    pipe = AdaptivePipeline(cfg, detector)
    ev = DetectionEvaluator(DATASET_NAMES)
    tel = Telemetry()
    preds = {}
    warmed = False
    full_gf = None
    t_start = time.time()
    for si, seq in enumerate(sequences):
        pipe.reset()
        frames = seq.frames[:max_frames] if max_frames else seq.frames
        for fi, path in enumerate(frames):
            td = time.perf_counter()
            frame = cv2.imread(path)
            decode_ms = (time.perf_counter() - td) * 1e3
            if frame is None:
                raise IOError(path)
            if not warmed:
                detector.warmup(frame)
                full_gf = detector.full_gflops(*frame.shape[:2])
                tel = Telemetry()
                warmed = True
            out, rec = pipe.step(frame, fi)
            rec["video"] = seq.name
            rec["decode_ms"] = decode_ms
            tel.add(rec)
            if seq.labels is not None:
                h, w = frame.shape[:2]
                gts = load_yolo_labels(seq.labels[fi], w, h)
                ev.add(out, gts, {"decision": rec["decision"], "stale": _stale_bin(rec["since_full"]),
                                  "video": seq.name})
            if save_preds:
                preds[f"{seq.name}/{fi}"] = out
        if log_every and (si + 1) % log_every == 0:
            print(f"  [{si + 1}/{len(sequences)}] videos  {time.time() - t_start:.0f}s", flush=True)

    from .evaluation import evaluator as _ev
    result = {"config": cfg.to_dict(), "ap_protocol": _ev.AP_PROTOCOL, "efficiency": tel.summary(full_gf)}
    if ev.tp:
        result["detection"] = ev.compute()
        result["detection_by_decision"] = {d: ev.compute(lambda t, d=d: t["decision"] == d) for d in ("FULL", "LOCAL", "SKIP")}
        result["detection_by_staleness"] = {b: ev.compute(lambda t, b=b: t["stale"] == b) for *_, b in STALE_BINS}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    tel.to_csv(os.path.join(out_dir, "frames.csv"))
    if ev.tp:
        ev.save(os.path.join(out_dir, "eval_stats.npz"))
    if save_preds:
        np.savez_compressed(os.path.join(out_dir, "preds.npz"), **preds)
    return result, detector


def run_realtime(cfg, video, out_dir):
    """Live demo: paced capture, latest-frame buffer. Deadline drops reported separately."""
    from .adaptive_framing.capture import VideoCaptureThread
    from .adaptive_framing.frame_manager import LatestFrameBuffer
    det = YOLODetector(cfg)
    pipe = AdaptivePipeline(cfg, det)
    tel = Telemetry()
    import cv2 as _cv2
    _c = _cv2.VideoCapture(video)
    ok, first = _c.read()
    _c.release()
    if ok:
        det.warmup(first)                 # exclude model load / first-call cost
    buf = LatestFrameBuffer()
    cap = VideoCaptureThread(video, buf, realtime=True).start()
    while True:
        item = buf.get()
        if item is None:
            if cap.is_stopped():
                break
            time.sleep(0.001)
            continue
        frame, fid, _ = item
        _, rec = pipe.step(frame, fid)
        tel.add(rec)
    s = tel.summary()
    s["buffer"] = buf.stats()     # 'replaced' = deadline drops (NOT scheduler skips)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "realtime.json"), "w") as f:
        json.dump(s, f, indent=2, default=float)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="adaptive_full", choices=sorted(PRESETS))
    ap.add_argument("--set", nargs="*", help="config overrides key=value (JSON values)")
    ap.add_argument("--images", help="split image dir (YOLO layout; labels via /images/->/labels/)")
    ap.add_argument("--labels", help="explicit labels dir mirroring --images")
    ap.add_argument("--seq-regex", default=DEFAULT_SEQ_REGEX)
    ap.add_argument("--video", help="single video file (no labels): efficiency only")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--max-videos", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--ap-protocol", choices=["current", "legacy"], default="current",
                    help="must match the ultralytics version of the frozen baseline run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from .evaluation import evaluator
    evaluator.AP_PROTOCOL = a.ap_protocol

    cfg = build_config(a.preset, a.set)
    if a.realtime:
        print(json.dumps(run_realtime(cfg, a.video, a.out), indent=2, default=float))
        return
    if a.video:
        from .adaptive_framing.sequence import Sequence
        import tempfile
        tmp = tempfile.mkdtemp()
        paths = []
        for i, f in iter_video(a.video):
            p = os.path.join(tmp, f"{i:06d}.png")
            cv2.imwrite(p, f)
            paths.append(p)
        seqs = [Sequence(os.path.basename(a.video), paths, None)]
    else:
        seqs = discover_sequences(a.images, a.labels, a.seq_regex)
        n = sum(len(s.frames) for s in seqs)
        print(f"Found {len(seqs)} sequences, {n} frames "
              f"(expected test: 122 / 43,772 -- if this differs, fix --seq-regex)")
        if a.max_videos:
            seqs = seqs[: a.max_videos]
    res, _ = run_sequences(cfg, seqs, a.out, max_frames=a.max_frames, save_preds=a.save_preds)
    print(json.dumps({k: res[k] for k in ("detection", "efficiency") if k in res}, indent=2, default=float))


if __name__ == "__main__":
    main()
