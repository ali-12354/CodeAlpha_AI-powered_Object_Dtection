"""
Smoke test: validates imports, SORT tracker logic, and a synthetic
detection-tracking round-trip — no webcam or GPU required.
"""

import sys
import numpy as np


def test_sort_logic():
    print("─" * 50)
    print("TEST: SORT tracker logic")

    from sort_tracker import SORTTracker, iou_batch, bbox_to_z, z_to_bbox

    # IoU sanity checks
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10]])
    assert abs(iou_batch(a, b)[0, 0] - 1.0) < 1e-5, "IoU of identical boxes must be 1"

    c = np.array([[20, 20, 30, 30]])
    assert iou_batch(a, c)[0, 0] == 0.0, "Non-overlapping boxes must have IoU 0"

    # bbox ↔ state vector round-trip
    box = np.array([100.0, 200.0, 200.0, 300.0])
    z   = bbox_to_z(box)
    box2 = z_to_bbox(z)[0]
    assert np.allclose(box, box2, atol=1e-3), f"Round-trip failed: {box} → {box2}"

    # Tracker: feed 5 identical detections for the same object
    tracker = SORTTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    det = np.array([[50.0, 50.0, 150.0, 150.0, 0.9, 0.0]])

    ids_seen = set()
    for i in range(5):
        out = tracker.update(det)
        for row in out:
            ids_seen.add(int(row[4]))

    assert len(ids_seen) == 1, f"Expected 1 unique track ID, got {ids_seen}"
    print(f"  ✓ Single track ID assigned: #{list(ids_seen)[0]}")

    # Two separate objects → two distinct IDs
    tracker.reset()
    det2 = np.array([
        [  0.0,   0.0, 100.0, 100.0, 0.9, 0.0],
        [300.0, 300.0, 400.0, 400.0, 0.85, 1.0],
    ])
    all_ids = set()
    for _ in range(5):
        out = tracker.update(det2)
        for row in out:
            all_ids.add(int(row[4]))

    assert len(all_ids) == 2, f"Expected 2 distinct IDs, got {all_ids}"
    print(f"  ✓ Two distinct track IDs: {sorted(all_ids)}")

    print("TEST PASSED: SORT logic ✓")


def test_viz_utils():
    print("─" * 50)
    print("TEST: Visualization utilities")
    import cv2
    from viz_utils import (
        generate_color_palette, get_color,
        draw_bounding_box, draw_label, draw_trail, draw_hud, draw_controls_hint,
    )

    palette = generate_color_palette(50)
    assert len(palette) == 50
    for c in palette:
        assert len(c) == 3 and all(0 <= v <= 255 for v in c)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    draw_bounding_box(frame, 50, 50, 200, 200, (0, 255, 0))
    draw_label(frame, "#1 person 0.92", 50, 50, (0, 255, 0))
    draw_trail(frame, [(60, 60), (80, 80), (100, 90)], (0, 200, 255))
    draw_hud(frame, fps=29.5, num_tracks=3, num_detections=4,
             model_name="yolov8n.pt", source_label="webcam #0", paused=False)
    draw_controls_hint(frame)

    assert frame.sum() > 0, "Frame should have non-zero pixels after drawing"
    print("TEST PASSED: viz_utils ✓")


def test_detector_import():
    print("─" * 50)
    print("TEST: YOLO detector import")
    from main_tracker import YOLODetector
    # Just verify the class is importable; don't download weights in CI
    assert callable(YOLODetector), "YOLODetector must be callable"
    print("TEST PASSED: YOLODetector importable ✓")


def test_synthetic_pipeline():
    """
    Simulate a 10-frame pipeline without a real video or model.
    Injects synthetic detections directly into the tracker.
    """
    print("─" * 50)
    print("TEST: Synthetic 10-frame tracking pipeline")
    import cv2
    from sort_tracker import SORTTracker
    from viz_utils    import draw_bounding_box, draw_label, draw_trail, get_color

    tracker = SORTTracker(max_age=10, min_hits=1, iou_threshold=0.3)

    W, H = 640, 480
    all_track_ids = set()

    for frame_i in range(10):
        # Two moving objects
        x_offset = frame_i * 15
        dets = np.array([
            [50 + x_offset, 100, 150 + x_offset, 200, 0.9, 0.0],
            [300,           200, 400,             300, 0.8, 2.0],
        ], dtype=float)

        tracks = tracker.update(dets)
        assert tracks.ndim == 2 and tracks.shape[1] == 7, \
            f"Unexpected track shape: {tracks.shape}"

        for row in tracks:
            all_track_ids.add(int(row[4]))

        # Draw on a blank frame
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        for row in tracks:
            x1, y1, x2, y2 = (int(v) for v in row[:4])
            tid = int(row[4])
            draw_bounding_box(frame, x1, y1, x2, y2, get_color(tid))
            draw_label(frame, f"#{tid}", x1, y1, get_color(tid))

            trk_obj = next((t for t in tracker.trackers if t.id == tid), None)
            if trk_obj and len(trk_obj.trail) > 1:
                draw_trail(frame, trk_obj.trail, get_color(tid))

    assert len(all_track_ids) == 2, \
        f"Expected 2 unique track IDs over 10 frames, got {all_track_ids}"
    print(f"  ✓ Tracked IDs over 10 frames: {sorted(all_track_ids)}")
    print("TEST PASSED: synthetic pipeline ✓")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print(" Smoke Tests: Object Detection & Tracking")
    print("=" * 50)

    tests = [
        test_sort_logic,
        test_viz_utils,
        test_detector_import,
        test_synthetic_pipeline,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
    sys.exit(0 if passed == len(tests) else 1)
