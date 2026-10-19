"""Additional non-mutating association observations and fixed-phase controls."""
from contextlib import contextmanager
from types import MethodType
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from adaptive_pipeline.shared.boxes import iou_matrix
from analysis.argoverse_hd_phase2 import replay as replay_module

ORIGINAL_INSTALL = replay_module.install_observers


def install(pipe, recorder, forced=None):
    ORIGINAL_INSTALL(pipe, recorder, forced)
    original = pipe.tracker._associate

    def associate(dets, track_indices=None, iou_threshold=None):
        result = original(dets, track_indices, iou_threshold)
        def capture():
            t = pipe.tracker
            tracks = np.arange(len(t)) if track_indices is None else np.asarray(track_indices, int)
            if not len(dets) or not len(tracks):
                return
            iou = iou_matrix(dets[:, :4], t.boxes()[tracks])
            same = dets[:, 5, None] == t.cls[tracks][None, :]
            iou[~same] = 0
            threshold = t.cfg.assoc_iou if iou_threshold is None else iou_threshold
            valid = same & (iou >= threshold)
            r,c = linear_sum_assignment(-iou)
            legacy = set(zip(r[valid[r,c]].tolist(), c[valid[r,c]].tolist()))
            r,c = linear_sum_assignment(np.where(valid, -(min(iou.shape)+1+iou), 0.))
            gated = set(zip(r[valid[r,c]].tolist(), c[valid[r,c]].tolist()))
            recorder.events.append({'type':'association_comparison',
                'stage':'low' if np.all(dets[:,4] < t.cfg.track_min_conf) else 'high',
                'threshold':threshold, 'legacy_matches':len(legacy), 'gated_matches':len(gated),
                'different_pairs':legacy != gated,
                'legacy_iou_sum':sum(float(iou[p]) for p in legacy),
                'gated_iou_sum':sum(float(iou[p]) for p in gated),
                'actual_pairs':[{'source_rows':t._input_rows[tuple(map(float,dets[d]))],
                                 'track_id':int(t.ids[track]), 'previous_missed':int(t.missed[track])}
                                for d,track in result[0]]})
        recorder.observe(capture)
        return result
    pipe.tracker._associate = associate
    if forced is not None:
        # All fixed-phase cells use the same early-FULL path as the native
        # anchored Fixed-2 control. No flow is added before scheduled FULLs.
        def early_full(self, lum):
            frame = recorder.frame
            return ['forced_schedule'] if frame['frame_index'] in forced[frame['video']] else []
        pipe.controller.early_full = MethodType(early_full, pipe.controller)


@contextmanager
def observations():
    with patch.object(replay_module, 'install_observers', install):
        yield
