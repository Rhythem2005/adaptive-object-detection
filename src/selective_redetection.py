"""Phase 5: Confidence/Uncertainty Filtering, Selective Re-Detection, and Fusion/NMS.

Implements the localized re-detection pipeline described in the research paper:

    "A Unified Lightweight YOLO Framework with Adaptive Frame Control and
     Selective Re-Detection for Real-Time Road-Scene Monitoring"
    — Kapoor, Bharti, Sabharwal, Bhatia

Paper-specified pipeline:
    1. Baseline Detection:
       Active frame evaluated by primary YOLOv8n detector.
    2. Uncertainty Filtering:
       Classify all primary proposals into three distinct pools:
       - Accepted Pool (D_accept): s_i >= τ_high (0.65) or Area > α * Area_frame
       - Candidate Pool (D_cand): τ_low (0.25) <= s_i < τ_high (0.65) and Area <= α * Area_frame (5%)
       - Rejected Pool (D_reject): s_i < τ_low (0.25) discarded as background noise
    3. Selective Context Expansion:
       Each candidate box is symmetrically expanded by a proportional margin β
       before patch extraction to preserve visual context and avoid boundary clipping.
    4. Localized Re-Detection:
       ROI patches extracted from native frame, resized to model dimensions, and
       evaluated in a secondary YOLO pass.
    5. Coordinate Remapping:
       Secondary detections projected back into global coordinate space.
    6. Bounding Box Fusion & NMS:
       Secondary proposals aggregated with high-confidence primary detections;
       class-aware Non-Maximum Suppression (NMS) deduplicates the unified pool.

Pipeline position:
    Adaptive Frame Controller (Phase 3)
            ↓
    Environmental Adaptation (Phase 4)
            ↓
    YOLOv8n Primary Inference
            ↓
    Confidence/Uncertainty Filtering (Phase 5)
            ↓
    Selective Re-Detection (Phase 5)
            ↓
    Bounding Box Fusion & NMS (Phase 5)
            ↓
    Downstream Output / Tracking
"""

import time
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np


def compute_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        float: IoU value in range [0.0, 1.0].
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def filter_uncertain_candidates(
    primary_boxes,
    frame_shape: Tuple[int, int],
    tau_low: float = 0.25,
    tau_high: float = 0.65,
    alpha: float = 0.05,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Partition primary YOLO detections into Accepted and Candidate pools.

    Paper:
        "Accepted Pool (D_accept): Detections with s_i >= τ_high (0.65) are treated
         as reliable and bypass secondary processing."
        "Candidate Pool (D_cand): Ambiguous detections that fall within the confidence
         band τ_low=0.25 to τ_high=0.65 and occupy an area smaller than α=5% of the
         total frame are routed to localized re-detection."
        "Rejected Pool (D_reject): Detections with s_i < τ_low are discarded as
         background noise."

    Args:
        primary_boxes: Ultralytics Results.boxes object or list of box items.
        frame_shape:   (H, W) of the active frame.
        tau_low:       Lower confidence threshold (paper: 0.25).
        tau_high:      Upper confidence threshold (paper: 0.65).
        alpha:         Area fraction threshold (paper: 0.05 = 5%).

    Returns:
        tuple: (accepted_list, candidate_list, stats_dict)
            - accepted_list:  list of dicts {"box": ndarray, "conf": float, "cls": int, "source": "primary"}
            - candidate_list: list of dicts {"box": ndarray, "conf": float, "cls": int}
            - stats_dict:     {"n_primary", "n_accepted", "n_candidates", "n_rejected"}
    """
    h, w = frame_shape[:2]
    frame_area = float(w * h)
    area_limit = alpha * frame_area

    accepted = []
    candidates = []
    n_rejected = 0
    n_primary = len(primary_boxes) if primary_boxes is not None else 0

    if primary_boxes is not None and len(primary_boxes) > 0:
        for box in primary_boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)

            box_w = max(0.0, xyxy[2] - xyxy[0])
            box_h = max(0.0, xyxy[3] - xyxy[1])
            box_area = box_w * box_h

            if conf < tau_low:
                # Discarded as noise
                n_rejected += 1
            elif conf >= tau_high or box_area > area_limit:
                # High confidence or large object (bypasses re-detection)
                accepted.append({
                    "box": xyxy,
                    "conf": conf,
                    "cls": cls_id,
                    "source": "primary",
                })
            else:
                # Ambiguous small-scale candidate (τ_low <= s < τ_high and Area <= α * Area_frame)
                candidates.append({
                    "box": xyxy,
                    "conf": conf,
                    "cls": cls_id,
                    "area": box_area,
                })

    stats = {
        "n_primary": n_primary,
        "n_accepted": len(accepted),
        "n_candidates": len(candidates),
        "n_rejected": n_rejected,
    }

    return accepted, candidates, stats


def create_context_rois(
    candidates: List[Dict[str, Any]],
    frame_shape: Tuple[int, int],
    margin: float = 1.0,
    min_size: int = 64,
    merge_iou: float = 0.30,
    max_rois: int = 4,
) -> List[List[int]]:
    """Expand candidate boxes to context ROIs, merge overlapping regions, and cap count.

    Paper:
        "To preserve visual context, each candidate box is symmetrically expanded
         by a fixed proportional margin before patch extraction."
        "The expanded region of interest (ROI) is extracted from the native-resolution
         source frame, bilinearly interpolated to the model's standard input
         dimensions, and passed through a secondary forward pass."

    Args:
        candidates:  List of candidate dicts with "box" and "conf".
        frame_shape: (H, W) of the active frame.
        margin:      Proportional expansion margin β. 1.0 expands each side by
                     1.0x of the box width/height (total 3x dimension).
        min_size:    Minimum pixel dimension for an ROI crop (prevents extreme zoom on tiny noise).
        merge_iou:   IoU overlap threshold above which two adjacent ROIs are merged.
        max_rois:    Maximum ROIs to evaluate per frame to preserve real-time streaming throughput.

    Returns:
        List of [rx1, ry1, rx2, ry2] integer pixel bounding boxes.
    """
    if not candidates:
        return []

    h, w = frame_shape[:2]
    raw_rois = []

    for c in candidates:
        box = c["box"]
        bw = float(box[2] - box[0])
        bh = float(box[3] - box[1])
        cx = float(box[0] + box[2]) / 2.0
        cy = float(box[1] + box[3]) / 2.0

        # Symmetrical proportional expansion with minimum size floor
        w_exp = max(bw * (1.0 + 2.0 * margin), float(min_size))
        h_exp = max(bh * (1.0 + 2.0 * margin), float(min_size))

        rx1 = max(0, int(round(cx - w_exp / 2.0)))
        ry1 = max(0, int(round(cy - h_exp / 2.0)))
        rx2 = min(w, int(round(cx + w_exp / 2.0)))
        ry2 = min(h, int(round(cy + h_exp / 2.0)))

        if rx2 > rx1 and ry2 > ry1:
            raw_rois.append([rx1, ry1, rx2, ry2, c["conf"]])

    if not raw_rois:
        return []

    # Merge overlapping candidate ROIs (prevents redundant crops covering the same scene patch)
    merged_rois = []
    for roi in raw_rois:
        box_roi = np.array(roi[:4], dtype=np.float32)
        merged = False
        for mroi in merged_rois:
            box_mroi = np.array(mroi[:4], dtype=np.float32)
            if compute_box_iou(box_roi, box_mroi) >= merge_iou:
                # Merge into union bounding box
                mroi[0] = min(mroi[0], roi[0])
                mroi[1] = min(mroi[1], roi[1])
                mroi[2] = max(mroi[2], roi[2])
                mroi[3] = max(mroi[3], roi[3])
                mroi[4] = max(mroi[4], roi[4])  # keep highest confidence
                merged = True
                break
        if not merged:
            merged_rois.append(list(roi))

    # Prioritize and cap ROIs to preserve real-time streaming constraints
    if len(merged_rois) > max_rois:
        merged_rois.sort(key=lambda r: -r[4])  # sort by confidence descending
        merged_rois = merged_rois[:max_rois]

    # Return pure coordinate lists [rx1, ry1, rx2, ry2]
    return [[r[0], r[1], r[2], r[3]] for r in merged_rois]


def redetect_rois(
    frame: np.ndarray,
    rois: List[List[int]],
    detector,
    conf_threshold: float = 0.25,
) -> Tuple[List[Dict[str, Any]], float]:
    """Execute localized YOLO inference on selected ROIs and remap to global coordinates.

    Paper:
        "Following localized inference, secondary detections are projected back
         into the global coordinate space through an affine transformation based
         on the crop's spatial offsets and scaling factors."

    Args:
        frame:          Native-resolution BGR frame.
        rois:           List of [rx1, ry1, rx2, ry2] crop coordinates.
        detector:       YOLODetector instance.
        conf_threshold: Detection confidence threshold on crops.

    Returns:
        tuple: (secondary_detections, redetect_ms)
            - secondary_detections: list of dicts {"box": ndarray, "conf": float, "cls": int, "source": "secondary"}
            - redetect_ms:          float wall-clock re-detection duration in milliseconds.
    """
    if not rois:
        return [], 0.0

    h, w = frame.shape[:2]
    secondary_detections = []

    t0 = time.perf_counter()

    for rx1, ry1, rx2, ry2 in rois:
        crop = frame[ry1:ry2, rx1:rx2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        # Single-crop forward pass through YOLO
        results, _ = detector.detect(crop)

        if results and len(results[0].boxes) > 0:
            for cbox in results[0].boxes:
                cconf = float(cbox.conf[0])
                ccls = int(cbox.cls[0])
                cxyxy = cbox.xyxy[0].cpu().numpy()

                # Remap local crop coordinates back to global frame coordinates
                gx1 = min(w, max(0, rx1 + cxyxy[0]))
                gy1 = min(h, max(0, ry1 + cxyxy[1]))
                gx2 = min(w, max(0, rx1 + cxyxy[2]))
                gy2 = min(h, max(0, ry1 + cxyxy[3]))

                if gx2 > gx1 and gy2 > gy1:
                    secondary_detections.append({
                        "box": np.array([gx1, gy1, gx2, gy2], dtype=np.float32),
                        "conf": cconf,
                        "cls": ccls,
                        "source": "secondary",
                    })

    t1 = time.perf_counter()
    redetect_ms = (t1 - t0) * 1000.0

    return secondary_detections, redetect_ms


def fuse_and_suppress(
    accepted_detections: List[Dict[str, Any]],
    secondary_detections: List[Dict[str, Any]],
    nms_iou_thresh: float = 0.50,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Aggregate primary accepted and secondary detections, applying class-aware NMS.

    Paper:
        "The newly recovered secondary proposals are aggregated with the high-confidence
         detections retained from the primary pass. Because a single object may be
         detected across both passes, class-aware Non-Maximum Suppression (NMS)
         deduplicates the aggregated pool. When spatial overlap between adjacent
         predictions exceeds an empirical intersection threshold, the pipeline
         retains the proposal exhibiting superior classification confidence and
         suppresses the redundant candidate."

    Args:
        accepted_detections:  Reliable primary detections.
        secondary_detections: Recovered detections from localized re-detection.
        nms_iou_thresh:       IoU suppression threshold (paper: empirical threshold).

    Returns:
        tuple: (final_detections, fusion_ms, n_suppressed)
            - final_detections: list of retained detection dicts.
            - fusion_ms:        float wall-clock fusion duration in milliseconds.
            - n_suppressed:     int count of duplicate proposals suppressed.
    """
    t0 = time.perf_counter()

    combined = list(accepted_detections) + list(secondary_detections)
    if not combined:
        t1 = time.perf_counter()
        return [], (t1 - t0) * 1000.0, 0

    final_detections = []
    n_suppressed = 0

    # Group and suppress per class (class-aware NMS)
    unique_classes = set(d["cls"] for d in combined)

    for cls_id in unique_classes:
        class_pool = [d for d in combined if d["cls"] == cls_id]
        # Sort descending by confidence
        class_pool.sort(key=lambda d: -d["conf"])

        while class_pool:
            best = class_pool.pop(0)
            final_detections.append(best)

            kept = []
            for other in class_pool:
                overlap = compute_box_iou(best["box"], other["box"])
                if overlap >= nms_iou_thresh:
                    n_suppressed += 1
                else:
                    kept.append(other)
            class_pool = kept

    t1 = time.perf_counter()
    fusion_ms = (t1 - t0) * 1000.0

    return final_detections, fusion_ms, n_suppressed


def selective_redetection_pipeline(
    frame: np.ndarray,
    primary_results,
    detector,
    tau_low: float = 0.25,
    tau_high: float = 0.65,
    alpha: float = 0.05,
    margin: float = 1.0,
    min_size: int = 64,
    merge_iou: float = 0.30,
    max_rois: int = 4,
    nms_iou_thresh: float = 0.50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Complete Phase 5 pipeline: Uncertainty Filter -> Selective Re-Detection -> Fusion/NMS.

    Args:
        frame:           Active frame (from Phase 4 adaptation).
        primary_results: Ultralytics Results object from initial forward pass.
        detector:        YOLODetector instance.
        tau_low:         Lower confidence boundary (0.25).
        tau_high:        Upper confidence boundary (0.65).
        alpha:           Area boundary fraction (0.05).
        margin:          Context expansion margin (1.0 = 3x box size).
        min_size:        Minimum ROI dimension (64 px).
        merge_iou:       Overlap threshold to merge candidate ROIs (0.30).
        max_rois:        Max ROIs per frame (4).
        nms_iou_thresh:  Class-aware NMS IoU threshold (0.50).

    Returns:
        tuple: (final_detections, telemetry_dict)
            - final_detections: consolidated deduplicated detections.
            - telemetry_dict:   detailed timing and proposal accounting metrics.
    """
    t_start = time.perf_counter()

    primary_boxes = primary_results[0].boxes if primary_results else []

    # Step 1: Uncertainty Filtering (Partition into D_accept and D_cand)
    accepted, candidates, filter_stats = filter_uncertain_candidates(
        primary_boxes,
        frame.shape,
        tau_low=tau_low,
        tau_high=tau_high,
        alpha=alpha,
    )

    # Step 2: Context Expansion and ROI Generation
    rois = create_context_rois(
        candidates,
        frame.shape,
        margin=margin,
        min_size=min_size,
        merge_iou=merge_iou,
        max_rois=max_rois,
    )

    # Step 3: Localized Re-Detection on ROIs
    secondary_detections, redetect_ms = redetect_rois(
        frame,
        rois,
        detector,
        conf_threshold=tau_low,
    )

    # Step 4: Bounding Box Fusion & Class-Aware NMS
    final_detections, fusion_ms, n_suppressed = fuse_and_suppress(
        accepted,
        secondary_detections,
        nms_iou_thresh=nms_iou_thresh,
    )

    t_end = time.perf_counter()
    total_phase5_ms = (t_end - t_start) * 1000.0

    telemetry = {
        "n_primary": filter_stats["n_primary"],
        "n_accepted": filter_stats["n_accepted"],
        "n_candidates": filter_stats["n_candidates"],
        "n_rejected": filter_stats["n_rejected"],
        "n_rois": len(rois),
        "n_secondary": len(secondary_detections),
        "n_suppressed": n_suppressed,
        "n_final": len(final_detections),
        "redetect_ms": round(redetect_ms, 3),
        "fusion_ms": round(fusion_ms, 3),
        "total_phase5_ms": round(total_phase5_ms, 3),
    }

    return final_detections, telemetry
