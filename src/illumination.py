"""Phase 4: Environmental Adaptation / Illumination Handling.

Implements low-light detection and CLAHE-based contrast enhancement as
described in the research paper:

    "A Unified Lightweight YOLO Framework with Adaptive Frame Control and
     Selective Re-Detection for Real-Time Road-Scene Monitoring"
    — Kapoor, Bharti, Sabharwal, Bhatia

Paper-specified behavior:
    - Monitor average frame luminance Y.
    - When Y < τ_illum (45, 8-bit scale), apply CLAHE to the L channel
      of the L*a*b* color space.
    - Adequately illuminated frames bypass enhancement entirely (no-op).

Paper does NOT specify:
    - The exact formula for computing average frame luminance Y.
      Implementation assumption: mean of the L channel after converting
      the BGR frame to L*a*b* color space.  The L channel in L*a*b*
      represents perceptual lightness and ranges 0–255 in OpenCV's 8-bit
      implementation, making it directly comparable to the paper's 8-bit
      threshold (τ_illum = 45).
    - CLAHE clipLimit — using OpenCV default: 2.0.
    - CLAHE tileGridSize — using OpenCV default: (8, 8).

Pipeline position:
    Adaptive Frame Controller → Environmental Adaptation → YOLOv8n
"""

import time

import cv2
import numpy as np


def compute_luminance(frame):
    """Compute average frame luminance Y from the L channel of L*a*b*.

    Paper: "we monitor the average frame luminance Y"
    Paper does NOT specify the exact formula — this implementation uses the
    mean of the L channel in CIE L*a*b* color space (OpenCV 8-bit, range
    0–255), which represents perceptual lightness and is directly
    comparable to the paper's 8-bit threshold.

    Args:
        frame: numpy array, BGR image (uint8, HxWx3).

    Returns:
        float: Mean L-channel value (0.0–255.0).
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    return float(np.mean(l_channel))


def apply_clahe(frame, clip_limit, tile_grid_size):
    """Apply CLAHE to the L channel of the frame in L*a*b* color space.

    Paper: "CLAHE applied specifically to the lightness channel in the
    L*a*b* color space"

    Uses OpenCV's cv2.createCLAHE to create the CLAHE instance.

    Args:
        frame:          numpy array, BGR image (uint8, HxWx3).
        clip_limit:     float, CLAHE clip limit.
                        Paper does not specify — using stated default.
        tile_grid_size: tuple (int, int), CLAHE tile grid size.
                        Paper does not specify — using stated default.

    Returns:
        numpy array: Enhanced BGR image (uint8, HxWx3), same shape/dtype.
    """
    # Convert BGR → L*a*b*
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Extract L channel and apply CLAHE
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # Replace L channel and convert back to BGR
    lab[:, :, 0] = l_enhanced
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def adapt_illumination(frame, threshold, clip_limit, tile_grid_size):
    """Environmental adaptation: conditionally enhance low-light frames.

    Paper: "When Y falls below an empirical low-light threshold
    (τ_illum = 45 on an 8-bit scale), the frame is routed through CLAHE"

    Adequately illuminated frames (Y >= threshold) bypass enhancement
    entirely — the original frame object is returned unmodified (no-op,
    not a pass-through transform).

    Args:
        frame:          numpy array, BGR image (uint8, HxWx3).
        threshold:      int, low-light threshold τ_illum (paper: 45).
        clip_limit:     float, CLAHE clipLimit.
        tile_grid_size: tuple (int, int), CLAHE tileGridSize.

    Returns:
        tuple: (output_frame, luminance, was_enhanced, adapt_ms)
            - output_frame:  numpy array, either original or CLAHE-enhanced.
            - luminance:     float, computed average luminance Y.
            - was_enhanced:  bool, True if CLAHE was applied.
            - adapt_ms:      float, total adaptation time in milliseconds
                             (includes luminance computation; 0 overhead
                             beyond luminance check when not enhanced).
    """
    t0 = time.perf_counter()

    luminance = compute_luminance(frame)

    if luminance < threshold:
        # Low-light: apply CLAHE enhancement
        output = apply_clahe(frame, clip_limit, tile_grid_size)
        was_enhanced = True
    else:
        # Adequately illuminated: bypass entirely (return same object)
        output = frame
        was_enhanced = False

    t1 = time.perf_counter()
    adapt_ms = (t1 - t0) * 1000.0

    return output, luminance, was_enhanced, adapt_ms
