"""Per-frame telemetry and efficiency summary.

The original Phase3Metrics only logged detector calls and reported
`inference_fps = n / sum(inference_ms)`, which ignores every other stage
and is undefined for frames that never reach the detector. Here every
processed frame is a record, and throughput is end-to-end.

Timing scope: t_total_ms covers everything the pipeline does for a frame
(signals, propagation, decision, detector/ROI, tracking). Image decoding is
recorded separately (decode_ms) and excluded, because it is identical for
all configurations and depends on storage, not on the method.
"""
import csv
import os
import resource
import time
from collections import Counter

import numpy as np


class Telemetry:
    def __init__(self):
        self.records = []
        self._cpu0 = time.process_time()
        self._wall0 = time.perf_counter()
        try:
            import torch
            self._cuda = torch.cuda.is_available()
            if self._cuda:
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._cuda = False

    def add(self, rec):
        self.records.append(rec)

    def summary(self, full_frame_gflops=None):
        r = self.records
        if not r:
            return {}
        n = len(r)
        dec = Counter(x["decision"] for x in r)
        tot = np.array([x["t_total_ms"] for x in r])
        full_t = np.array([x["t_total_ms"] for x in r if x["decision"] == "FULL"])
        skip_t = np.array([x["t_total_ms"] for x in r if x["decision"] == "SKIP"])
        local_t = np.array([x["t_total_ms"] for x in r if x["decision"] == "LOCAL"])
        reasons = Counter()
        for x in r:
            for k in filter(None, x["reasons"].split("|")):
                reasons[k] += 1
        gfl = float(sum(x["gflops"] for x in r))
        s = {
            "frames": n,
            "full_frames": dec["FULL"], "local_frames": dec["LOCAL"], "skip_frames": dec["SKIP"],
            "full_detector_calls": dec["FULL"],
            "detector_skip_rate": 1 - dec["FULL"] / n,          # frames without a full-frame pass
            "local_redetect_rate": dec["LOCAL"] / n,              # frames resolved by ROI pass alone
            "tracking_only_rate": dec["SKIP"] / n,              # no detector work at all
            "roi_crops_total": int(sum(x["n_crops"] for x in r)),
            # LOCAL tier effectiveness: ROI passes that avoided a full pass vs. escalated
            "roi_pass_rate": sum(x.get("roi_pass", 0) for x in r) / n,
            "roi_confirm_rate": (dec["LOCAL"] / max(1, sum(x.get("roi_pass", 0) for x in r))),
            "latency_mean_ms": float(tot.mean()),
            "latency_p50_ms": float(np.percentile(tot, 50)),
            "latency_p95_ms": float(np.percentile(tot, 95)),
            "latency_p99_ms": float(np.percentile(tot, 99)),
            "latency_max_ms": float(tot.max()),
            "fps_end_to_end": float(1000.0 * n / tot.sum()),
            "full_frame_ms_mean": float(full_t.mean()) if len(full_t) else None,
            "local_frame_ms_mean": float(local_t.mean()) if len(local_t) else None,
            "skip_frame_ms_mean": float(skip_t.mean()) if len(skip_t) else None,
            "detector_ms_mean_on_full": float(np.mean([x["t_detect_ms"] for x in r if x["decision"] == "FULL"])) if dec["FULL"] else None,
            "overhead_ms_mean": float(np.mean([x["t_signal_ms"] + x["t_propagate_ms"] + x["t_decide_ms"] + x["t_track_ms"] for x in r])),
            "gflops_per_frame": gfl / n,
            "trigger_counts": dict(reasons),
            "cpu_ms_per_frame": 1000.0 * (time.process_time() - self._cpu0) / n,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        }
        if full_frame_gflops:
            s["gflops_fraction_of_baseline"] = s["gflops_per_frame"] / full_frame_gflops
        if self._cuda:
            import torch
            s["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        return s

    def to_csv(self, path):
        if not self.records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keys = list(self.records[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.records)
