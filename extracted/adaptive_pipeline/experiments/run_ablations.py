"""Run a suite of presets on one split, reusing one loaded detector,
and write a comparison table (summary.csv / summary.md).

  python -m adaptive_pipeline.experiments.run_ablations --suite main \
      --images data/val/images --out runs/val        # tune here
  python -m adaptive_pipeline.experiments.run_ablations --suite main \
      --images data/test/images --out runs/test      # report here, once
"""
import argparse
import csv
import os

from ..adaptive_framing.sequence import DEFAULT_SEQ_REGEX, discover_sequences
from ..run import build_config, run_sequences
from .presets import PRESETS, SUITES

COLS = ["preset", "mAP50", "mAP50-95", "precision", "recall", "precision@0.25", "recall@0.25",
        "SKIP_mAP50-95", "detector_skip_rate", "local_redetect_rate", "roi_confirm_rate", "tracking_only_rate",
        "gflops_fraction_of_baseline", "latency_mean_ms", "latency_p95_ms", "fps_end_to_end",
        "cpu_ms_per_frame"]


def row(name, res):
    d, e = res.get("detection") or {}, res["efficiency"]
    skip = (res.get("detection_by_decision") or {}).get("SKIP") or {}
    r = {"preset": name, "SKIP_mAP50-95": skip.get("mAP50-95")}
    for k in COLS[1:]:
        if k in d:
            r[k] = d[k]
        elif k in e:
            r[k] = e[k]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main", choices=sorted(SUITES))
    ap.add_argument("--presets", nargs="*")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--seq-regex", default=DEFAULT_SEQ_REGEX)
    ap.add_argument("--max-videos", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--ap-protocol", choices=["current", "legacy"], default="current")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ..evaluation import evaluator
    evaluator.AP_PROTOCOL = a.ap_protocol

    seqs = discover_sequences(a.images, a.labels, a.seq_regex)[: a.max_videos]
    names = a.presets or SUITES[a.suite]
    rows, det = [], None
    for name in names:
        assert name in PRESETS, name
        print(f"== {name}")
        cfg = build_config(name, a.set)
        res, det = run_sequences(cfg, seqs, os.path.join(a.out, name), detector=det,
                                 max_frames=a.max_frames, log_every=0)
        rows.append(row(name, res))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("| " + " | ".join(COLS) + " |\n|" + "---|" * len(COLS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r.get(c)) for c in COLS) + " |\n")
    print(open(os.path.join(a.out, "summary.md")).read())


if __name__ == "__main__":
    main()
