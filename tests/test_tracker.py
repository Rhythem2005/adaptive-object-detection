"""Unit tests for Phase 6 tracker: src/tracker.py.

Tests cover:
    - BBox conversion (xyxy_to_cxcywh, cxcywh_to_xyxy, round-trip)
    - Kalman predict/update (state progression, convergence)
    - IoU computation (overlap, no overlap, identical, partial)
    - Hungarian matching (basic, class-aware, IoU gating)
    - Track creation and termination
    - ID persistence across frames
    - Missed-frame handling
    - Multi-track scenarios
    - Duplicate-ID prevention
    - Coordinate validity (non-negative width/height)
"""

import sys
import os
import importlib.util
import numpy as np

# Load src/tracker.py directly without triggering src/__init__.py
# (which imports ultralytics — may not be available in the test environment)
_tracker_path = os.path.join(os.path.dirname(__file__), "..", "src", "tracker.py")
_spec = importlib.util.spec_from_file_location("tracker", _tracker_path)
_tracker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tracker)

xyxy_to_cxcywh = _tracker.xyxy_to_cxcywh
cxcywh_to_xyxy = _tracker.cxcywh_to_xyxy
compute_iou = _tracker.compute_iou
compute_iou_matrix = _tracker.compute_iou_matrix
KalmanBoxTracker = _tracker.KalmanBoxTracker
Track = _tracker.Track
associate_detections_to_tracks = _tracker.associate_detections_to_tracks
MultiObjectTracker = _tracker.MultiObjectTracker


def _make_det(box, conf=0.8, cls=0):
    """Helper to create a detection dict."""
    return {"box": np.array(box, dtype=np.float32), "conf": conf, "cls": cls, "source": "primary"}


# ── Test 1: BBox Conversion xyxy → cxcywh ───────────────────────────────────

def test_xyxy_to_cxcywh():
    """Verify [x1,y1,x2,y2] → [cx,cy,w,h] conversion."""
    xyxy = np.array([10, 20, 50, 80], dtype=np.float64)
    cxcywh = xyxy_to_cxcywh(xyxy)
    assert np.allclose(cxcywh, [30, 50, 40, 60]), f"Expected [30,50,40,60], got {cxcywh}"
    print("  PASS: test_xyxy_to_cxcywh")


# ── Test 2: BBox Conversion cxcywh → xyxy ───────────────────────────────────

def test_cxcywh_to_xyxy():
    """Verify [cx,cy,w,h] → [x1,y1,x2,y2] conversion."""
    cxcywh = np.array([30, 50, 40, 60], dtype=np.float64)
    xyxy = cxcywh_to_xyxy(cxcywh)
    assert np.allclose(xyxy, [10, 20, 50, 80]), f"Expected [10,20,50,80], got {xyxy}"
    print("  PASS: test_cxcywh_to_xyxy")


# ── Test 3: Round-Trip Conversion ────────────────────────────────────────────

def test_roundtrip_conversion():
    """Verify xyxy → cxcywh → xyxy round-trip preserves values."""
    original = np.array([15.5, 25.3, 100.7, 200.1], dtype=np.float64)
    result = cxcywh_to_xyxy(xyxy_to_cxcywh(original))
    assert np.allclose(result, original, atol=1e-10), f"Round-trip failed: {original} → {result}"
    print("  PASS: test_roundtrip_conversion")


# ── Test 4: IoU — Identical Boxes ────────────────────────────────────────────

def test_iou_identical():
    """Two identical boxes should have IoU = 1.0."""
    box = np.array([10, 20, 50, 80], dtype=np.float64)
    iou = compute_iou(box, box)
    assert abs(iou - 1.0) < 1e-6, f"Expected IoU=1.0, got {iou}"
    print("  PASS: test_iou_identical")


# ── Test 5: IoU — No Overlap ────────────────────────────────────────────────

def test_iou_no_overlap():
    """Non-overlapping boxes should have IoU = 0.0."""
    a = np.array([0, 0, 10, 10], dtype=np.float64)
    b = np.array([20, 20, 30, 30], dtype=np.float64)
    iou = compute_iou(a, b)
    assert abs(iou) < 1e-6, f"Expected IoU=0.0, got {iou}"
    print("  PASS: test_iou_no_overlap")


# ── Test 6: IoU — Partial Overlap ────────────────────────────────────────────

def test_iou_partial():
    """Verify partial overlap IoU calculation."""
    a = np.array([0, 0, 20, 20], dtype=np.float64)
    b = np.array([10, 10, 30, 30], dtype=np.float64)
    # Intersection: [10,10,20,20] = 10*10 = 100
    # Union: 400 + 400 - 100 = 700
    # IoU: 100/700 ≈ 0.14286
    iou = compute_iou(a, b)
    assert abs(iou - 100.0 / 700.0) < 1e-6, f"Expected ~0.1429, got {iou}"
    print("  PASS: test_iou_partial")


# ── Test 7: IoU Matrix ──────────────────────────────────────────────────────

def test_iou_matrix():
    """Verify pairwise IoU matrix computation."""
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
    b = np.array([[5, 5, 15, 15]], dtype=np.float64)
    mat = compute_iou_matrix(a, b)
    assert mat.shape == (2, 1), f"Expected shape (2,1), got {mat.shape}"
    # a[0] ∩ b[0]: [5,5,10,10]=25, union=100+100-25=175 → 25/175
    expected_0 = 25.0 / 175.0
    assert abs(mat[0, 0] - expected_0) < 1e-6, f"Expected {expected_0}, got {mat[0, 0]}"
    # a[1] ∩ b[0]: no overlap
    assert abs(mat[1, 0]) < 1e-6, f"Expected 0.0, got {mat[1, 0]}"
    print("  PASS: test_iou_matrix")


# ── Test 8: Kalman Predict ───────────────────────────────────────────────────

def test_kalman_predict():
    """Verify Kalman predict moves state according to velocity * dt."""
    kf = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float64))
    # Manually set velocity: vx=10px/s, vy=5px/s
    kf.x[4] = 10.0   # vx
    kf.x[5] = 5.0    # vy
    kf.x[6] = 0.0    # vw
    kf.x[7] = 0.0    # vh

    pred = kf.predict(dt=1.0)  # 1 second
    # cx should move from 150 → 160, cy from 150 → 155
    cx, cy = kf.x[0], kf.x[1]
    assert abs(cx - 160.0) < 1e-3, f"Expected cx=160, got {cx}"
    assert abs(cy - 155.0) < 1e-3, f"Expected cy=155, got {cy}"
    print("  PASS: test_kalman_predict")


# ── Test 9: Kalman Update ────────────────────────────────────────────────────

def test_kalman_update():
    """Verify Kalman update pulls state toward measurement."""
    kf = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float64))
    kf.predict(dt=0.033)

    # Update with a slightly shifted box
    kf.update(np.array([105, 105, 205, 205], dtype=np.float64))

    # State should move toward the measurement
    cx, cy = kf.x[0], kf.x[1]
    # Should be between original (150) and measurement (155)
    assert 149.0 < cx < 156.0, f"cx={cx} not in expected range"
    assert 149.0 < cy < 156.0, f"cy={cy} not in expected range"
    print("  PASS: test_kalman_update")


# ── Test 10: Kalman Variable Δt ──────────────────────────────────────────────

def test_kalman_variable_dt():
    """Verify different Δt values produce different predictions."""
    kf1 = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float64))
    kf1.x[4] = 30.0  # vx = 30 px/s

    kf2 = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float64))
    kf2.x[4] = 30.0  # vx = 30 px/s

    kf1.predict(dt=0.033)  # ~30 FPS
    kf2.predict(dt=0.066)  # ~15 FPS

    # kf2 should have moved further
    assert kf2.x[0] > kf1.x[0], f"Larger dt should produce larger displacement: {kf2.x[0]} vs {kf1.x[0]}"
    print("  PASS: test_kalman_variable_dt")


# ── Test 11: Hungarian Matching — Basic ──────────────────────────────────────

def test_hungarian_basic():
    """Verify basic detection-to-track matching."""
    Track.reset_id_counter()
    dets = [_make_det([100, 100, 200, 200], cls=0)]
    tracks = [Track(_make_det([102, 102, 202, 202], cls=0))]

    matches, unmatched_d, unmatched_t = associate_detections_to_tracks(
        dets, tracks, iou_threshold=0.3
    )
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)}"
    assert len(unmatched_d) == 0, f"Expected 0 unmatched det, got {len(unmatched_d)}"
    assert len(unmatched_t) == 0, f"Expected 0 unmatched trk, got {len(unmatched_t)}"
    print("  PASS: test_hungarian_basic")


# ── Test 12: Hungarian Matching — IoU Gating ─────────────────────────────────

def test_hungarian_iou_gating():
    """Verify that matches with IoU below threshold are rejected."""
    Track.reset_id_counter()
    # Two boxes far apart — IoU should be ~0
    dets = [_make_det([0, 0, 10, 10], cls=0)]
    tracks = [Track(_make_det([500, 500, 510, 510], cls=0))]

    matches, unmatched_d, unmatched_t = associate_detections_to_tracks(
        dets, tracks, iou_threshold=0.3
    )
    assert len(matches) == 0, f"Expected 0 matches (IoU too low), got {len(matches)}"
    assert len(unmatched_d) == 1
    assert len(unmatched_t) == 1
    print("  PASS: test_hungarian_iou_gating")


# ── Test 13: Class-Aware Association ─────────────────────────────────────────

def test_class_aware_association():
    """Verify that detections of different classes don't match, even if overlapping."""
    Track.reset_id_counter()
    # Same bbox, different class
    dets = [_make_det([100, 100, 200, 200], cls=0)]
    tracks = [Track(_make_det([100, 100, 200, 200], cls=2))]  # Different class

    matches, unmatched_d, unmatched_t = associate_detections_to_tracks(
        dets, tracks, iou_threshold=0.3
    )
    assert len(matches) == 0, f"Expected 0 matches (class mismatch), got {len(matches)}"
    assert len(unmatched_d) == 1
    assert len(unmatched_t) == 1
    print("  PASS: test_class_aware_association")


# ── Test 14: Track Creation on Unmatched Detection ──────────────────────────

def test_track_creation():
    """Verify tracker creates new tracks for unmatched detections."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=30, min_hits=1)
    dets = [_make_det([100, 100, 200, 200], cls=0)]

    result = tracker.update(dets, timestamp=1.0)
    # With min_hits=1, should appear immediately
    assert len(tracker.tracks) == 1, f"Expected 1 track, got {len(tracker.tracks)}"
    assert len(result) == 1, f"Expected 1 output track (min_hits=1), got {len(result)}"
    print("  PASS: test_track_creation")


# ── Test 15: Track Termination After Max Age ─────────────────────────────────

def test_track_termination():
    """Verify tracks are terminated after max_age missed frames."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=3, min_hits=1)

    # Create track
    tracker.update([_make_det([100, 100, 200, 200])], timestamp=1.0)
    assert len(tracker.tracks) == 1

    # Miss for max_age+1 frames
    for i in range(4):
        tracker.update([], timestamp=1.0 + (i + 1) * 0.033)

    assert len(tracker.tracks) == 0, f"Expected 0 tracks after max_age, got {len(tracker.tracks)}"
    print("  PASS: test_track_termination")


# ── Test 16: ID Persistence Across Frames ────────────────────────────────────

def test_id_persistence():
    """Verify a track retains its ID across multiple frames when matched."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=30, min_hits=1)

    # Frame 1
    result1 = tracker.update([_make_det([100, 100, 200, 200])], timestamp=1.0)
    id1 = result1[0]["track_id"]

    # Frame 2 — slightly shifted
    result2 = tracker.update([_make_det([102, 102, 202, 202])], timestamp=1.033)
    id2 = result2[0]["track_id"]

    # Frame 3
    result3 = tracker.update([_make_det([105, 105, 205, 205])], timestamp=1.066)
    id3 = result3[0]["track_id"]

    assert id1 == id2 == id3, f"ID should persist: {id1}, {id2}, {id3}"
    print("  PASS: test_id_persistence")


# ── Test 17: Missed-Frame Handling ───────────────────────────────────────────

def test_missed_frame_handling():
    """Verify track survives missed frames and resumes on re-detection."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=10, min_hits=1)

    # Frame 1: create track
    result1 = tracker.update([_make_det([100, 100, 200, 200])], timestamp=1.0)
    tid = result1[0]["track_id"]

    # Frame 2-4: no detections (3 misses)
    for i in range(3):
        tracker.update([], timestamp=1.0 + (i + 1) * 0.033)

    assert len(tracker.tracks) == 1, "Track should survive 3 misses with max_age=10"

    # Frame 5: re-detection near predicted position
    result5 = tracker.update([_make_det([100, 100, 200, 200])], timestamp=1.133)

    # Track should still exist and have same ID
    assert len(tracker.tracks) >= 1
    # Check that the original track got updated (hits increased)
    found = False
    for t in tracker.tracks:
        if t.track_id == tid:
            found = True
            assert t.time_since_update == 0, f"Expected time_since_update=0 after match, got {t.time_since_update}"
    assert found, f"Original track {tid} should still exist"
    print("  PASS: test_missed_frame_handling")


# ── Test 18: Multi-Track Scenario ────────────────────────────────────────────

def test_multi_track():
    """Verify multiple objects are tracked independently."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=30, min_hits=1)

    dets = [
        _make_det([10, 10, 50, 50], cls=0),
        _make_det([200, 200, 300, 300], cls=2),
        _make_det([400, 400, 500, 500], cls=0),
    ]

    result = tracker.update(dets, timestamp=1.0)
    assert len(result) == 3, f"Expected 3 tracks, got {len(result)}"

    # Move each slightly
    dets2 = [
        _make_det([12, 12, 52, 52], cls=0),
        _make_det([202, 202, 302, 302], cls=2),
        _make_det([402, 402, 502, 502], cls=0),
    ]
    result2 = tracker.update(dets2, timestamp=1.033)
    assert len(result2) == 3, f"Expected 3 tracks still, got {len(result2)}"

    # All IDs should match frame 1
    ids1 = sorted([r["track_id"] for r in result])
    ids2 = sorted([r["track_id"] for r in result2])
    assert ids1 == ids2, f"IDs should persist: {ids1} vs {ids2}"
    print("  PASS: test_multi_track")


# ── Test 19: Duplicate-ID Prevention ─────────────────────────────────────────

def test_no_duplicate_ids():
    """Verify no two tracks ever share the same ID."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=2, min_hits=1)

    all_ids = set()

    # Create and terminate several tracks
    for i in range(5):
        x = i * 200
        dets = [_make_det([x, x, x + 50, x + 50])]
        result = tracker.update(dets, timestamp=float(i))
        for r in result:
            assert r["track_id"] not in all_ids or r["track_id"] in {r2["track_id"] for r2 in result}, \
                f"Duplicate ID {r['track_id']} found!"
            all_ids.add(r["track_id"])

    # Terminate all by sending empty frames
    for i in range(5, 10):
        tracker.update([], timestamp=float(i))

    # Create more
    for i in range(10, 15):
        x = i * 100
        result = tracker.update([_make_det([x, x, x + 50, x + 50])], timestamp=float(i))
        for r in result:
            old_len = len(all_ids)
            all_ids.add(r["track_id"])
            assert len(all_ids) > old_len, f"Duplicate ID {r['track_id']}!"

    print("  PASS: test_no_duplicate_ids")


# ── Test 20: Coordinate Validity ─────────────────────────────────────────────

def test_coordinate_validity():
    """Verify Kalman state always produces valid (positive w, h) boxes."""
    Track.reset_id_counter()
    kf = KalmanBoxTracker(np.array([50, 50, 60, 60], dtype=np.float64))

    # Run many predictions to check for degenerate boxes
    for i in range(100):
        pred = kf.predict(dt=0.033)
        w = pred[2] - pred[0]
        h = pred[3] - pred[1]
        assert w > 0, f"Width became non-positive at step {i}: {w}"
        assert h > 0, f"Height became non-positive at step {i}: {h}"

    print("  PASS: test_coordinate_validity")


# ── Test 21: min_hits Suppression ────────────────────────────────────────────

def test_min_hits_suppression():
    """Verify tracks are NOT reported until they reach min_hits."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(iou_threshold=0.3, max_age=30, min_hits=3)

    # Frame 1: track created but not confirmed
    result1 = tracker.update([_make_det([100, 100, 200, 200])], timestamp=1.0)
    assert len(result1) == 0, f"Should not report until min_hits=3, got {len(result1)} tracks"

    # Frame 2
    result2 = tracker.update([_make_det([102, 102, 202, 202])], timestamp=1.033)
    assert len(result2) == 0, f"Should not report at hits=2, got {len(result2)}"

    # Frame 3: now hits=3, should be confirmed
    result3 = tracker.update([_make_det([104, 104, 204, 204])], timestamp=1.066)
    assert len(result3) == 1, f"Should report at hits=3, got {len(result3)}"
    print("  PASS: test_min_hits_suppression")


# ── Test 22: Empty Detections ────────────────────────────────────────────────

def test_empty_detections():
    """Verify tracker handles empty detection list without errors."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker()
    result = tracker.update([], timestamp=1.0)
    assert result == [], f"Expected empty result, got {result}"
    print("  PASS: test_empty_detections")


# ── Test 23: Telemetry Output ────────────────────────────────────────────────

def test_telemetry_output():
    """Verify telemetry dict contains expected keys."""
    Track.reset_id_counter()
    tracker = MultiObjectTracker(min_hits=1)
    tracker.update([_make_det([100, 100, 200, 200])], timestamp=1.0)

    tel = tracker.last_telemetry
    expected_keys = {
        "dt_seconds", "n_detections", "n_tracks_before", "n_matched",
        "n_unmatched_det", "n_unmatched_trk", "n_created", "n_terminated",
        "n_active_tracks", "n_confirmed_output", "tracking_ms",
    }
    assert expected_keys.issubset(tel.keys()), f"Missing keys: {expected_keys - tel.keys()}"
    assert tel["tracking_ms"] >= 0
    print("  PASS: test_telemetry_output")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 Tracker Unit Tests")
    print("=" * 60)
    print()

    tests = [
        test_xyxy_to_cxcywh,
        test_cxcywh_to_xyxy,
        test_roundtrip_conversion,
        test_iou_identical,
        test_iou_no_overlap,
        test_iou_partial,
        test_iou_matrix,
        test_kalman_predict,
        test_kalman_update,
        test_kalman_variable_dt,
        test_hungarian_basic,
        test_hungarian_iou_gating,
        test_class_aware_association,
        test_track_creation,
        test_track_termination,
        test_id_persistence,
        test_missed_frame_handling,
        test_multi_track,
        test_no_duplicate_ids,
        test_coordinate_validity,
        test_min_hits_suppression,
        test_empty_detections,
        test_telemetry_output,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  FAIL: {test_fn.__name__} — {e}")

    print()
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if errors:
        print()
        print("Failures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
