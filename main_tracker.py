"""
Real-Time Object Detection and Tracking
========================================
Detects objects using YOLOv8 and tracks them with SORT
(Kalman Filter + Hungarian Algorithm).

Usage
-----
  # Webcam
  python main_tracker.py

  # Video file
  python main_tracker.py --source path/to/video.mp4

  # Specific webcam index
  python main_tracker.py --source 1

  # All options
  python main_tracker.py --source 0 --model yolov8n.pt --conf 0.4 \\
      --iou 0.45 --classes 0 2 --max-age 30 --min-hits 3 --save

Keyboard Controls (when window is open)
-----------------------------------------
  Q      – quit
  P      – pause / resume
  S      – save screenshot
  R      – reset all track IDs
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── local modules ──────────────────────────────────────────────
from sort_tracker import SORTTracker
from viz_utils   import (
    draw_bounding_box, draw_label, draw_trail,
    draw_hud, draw_controls_hint, get_color,
)


# ─────────────────────────────────────────────────────────────
# YOLO detector wrapper
# ─────────────────────────────────────────────────────────────

class YOLODetector:
    """Thin wrapper around Ultralytics YOLOv8."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO model: {model_name}  (device={device})")
        self.model      = YOLO(model_name)
        self.device     = device
        self.class_names: list[str] = self.model.names  # type: ignore

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.4,
        iou_threshold:  float = 0.45,
        classes: list[int] | None = None,
    ) -> np.ndarray:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        np.ndarray  shape (N, 6)  columns: [x1, y1, x2, y2, score, class_id]
        """
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes   = r.boxes.xyxy.cpu().numpy()
            scores  = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(boxes, scores, cls_ids):
                detections.append([*box, score, cls_id])

        return np.array(detections, dtype=float) if detections else np.empty((0, 6))


# ─────────────────────────────────────────────────────────────
# Frame processor
# ─────────────────────────────────────────────────────────────

class FrameProcessor:
    """Runs detection + tracking on one frame and annotates it."""

    def __init__(
        self,
        detector:    YOLODetector,
        tracker:     SORTTracker,
        conf:        float = 0.4,
        iou:         float = 0.45,
        classes:     list[int] | None = None,
        show_trail:  bool = True,
        show_conf:   bool = True,
    ):
        self.detector   = detector
        self.tracker    = tracker
        self.conf       = conf
        self.iou        = iou
        self.classes    = classes
        self.show_trail = show_trail
        self.show_conf  = show_conf

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Detect → Track → Annotate one frame.

        Returns
        -------
        annotated_frame, num_tracks, num_detections
        """
        out = frame.copy()

        # ── Detection ──────────────────────────────────────────
        detections = self.detector.detect(
            frame,
            conf_threshold=self.conf,
            iou_threshold=self.iou,
            classes=self.classes,
        )
        num_dets = len(detections)

        # ── Tracking ───────────────────────────────────────────
        tracks = self.tracker.update(detections)  # (M, 7)
        num_tracks = len(tracks)

        # ── Annotate ───────────────────────────────────────────
        for track in tracks:
            x1, y1, x2, y2 = (int(v) for v in track[:4])
            track_id  = int(track[4])
            class_id  = int(track[5])
            score     = float(track[6])

            color      = get_color(track_id)
            class_name = self.detector.class_names.get(class_id, f"cls{class_id}")

            # Bounding box
            draw_bounding_box(out, x1, y1, x2, y2, color)

            # Label
            label = f"#{track_id} {class_name}"
            if self.show_conf:
                label += f" {score:.2f}"
            draw_label(out, label, x1, y1, color)

            # Trail
            if self.show_trail:
                trk_obj = next(
                    (t for t in self.tracker.trackers if t.id == track_id), None
                )
                if trk_obj and len(trk_obj.trail) > 1:
                    draw_trail(out, trk_obj.trail, color)

        return out, num_tracks, num_dets


# ─────────────────────────────────────────────────────────────
# Video source helper
# ─────────────────────────────────────────────────────────────

def open_source(source: str) -> tuple[cv2.VideoCapture, str]:
    """
    Open a webcam index or video file path.
    Returns (capture, label).
    """
    # Try to parse as integer (webcam index)
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        label = f"Webcam #{idx}"
    except ValueError:
        cap = cv2.VideoCapture(source)
        label = Path(source).name

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    return cap, label


# ─────────────────────────────────────────────────────────────
# FPS meter
# ─────────────────────────────────────────────────────────────

class FPSMeter:
    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps   = 0.0
        self._t0    = time.perf_counter()

    def tick(self) -> float:
        t1        = time.perf_counter()
        inst_fps  = 1.0 / max(t1 - self._t0, 1e-6)
        self._fps = self._alpha * inst_fps + (1 - self._alpha) * self._fps
        self._t0  = t1
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # ── Setup ──────────────────────────────────────────────────
    detector = YOLODetector(model_name=args.model, device=args.device)

    tracker = SORTTracker(
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.track_iou,
    )

    processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes or None,
        show_trail=not args.no_trail,
        show_conf=not args.no_conf,
    )

    cap, src_label = open_source(args.source)

    # ── Optional video writer ───────────────────────────────────
    writer = None
    if args.save:
        out_path = args.output or "output_tracked.avi"
        w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (w, h))
        print(f"[INFO] Saving output to: {out_path}")

    fps_meter = FPSMeter()
    paused    = False
    frame_idx = 0

    window_name = "Object Detection + SORT Tracking  |  Q=Quit  P=Pause"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] Starting tracking loop … press Q to quit.")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of stream.")
                    break

                # ── Process frame ───────────────────────────────
                annotated, num_tracks, num_dets = processor.process(frame)

                # ── HUD ─────────────────────────────────────────
                fps = fps_meter.tick()
                draw_hud(
                    annotated, fps, num_tracks, num_dets,
                    model_name=args.model,
                    source_label=src_label,
                    paused=paused,
                )
                draw_controls_hint(annotated)

                if writer is not None:
                    writer.write(annotated)

                frame_idx += 1

            # ── Display ─────────────────────────────────────────
            if not args.headless:
                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:      # Q or Esc
                    print("[INFO] Quit requested.")
                    break

                elif key == ord("p"):                  # Pause toggle
                    paused = not paused
                    print(f"[INFO] {'Paused' if paused else 'Resumed'}")

                elif key == ord("s"):                  # Screenshot
                    ts   = time.strftime("%Y%m%d_%H%M%S")
                    path = f"screenshot_{ts}.jpg"
                    cv2.imwrite(path, annotated)
                    print(f"[INFO] Screenshot saved: {path}")

                elif key == ord("r"):                  # Reset IDs
                    tracker.reset()
                    print("[INFO] Track IDs reset.")
            else:
                # Headless mode: process N frames then exit
                if args.max_frames and frame_idx >= args.max_frames:
                    print(f"[INFO] Processed {frame_idx} frames (headless mode).")
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[INFO] Done. Total frames processed: {frame_idx}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Time Object Detection and SORT Tracking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source
    p.add_argument("--source",  default="0",
                   help="Webcam index (0,1,…) or path to a video file")

    # Detection
    p.add_argument("--model",   default="yolov8n.pt",
                   help="YOLO model weights (e.g. yolov8n.pt, yolov8s.pt)")
    p.add_argument("--conf",    type=float, default=0.40,
                   help="Detection confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="NMS IoU threshold for detection")
    p.add_argument("--classes", type=int,   nargs="+", default=None,
                   help="Filter by class IDs (e.g. --classes 0 2 for person+car)")
    p.add_argument("--device",  default="cpu",
                   help="Inference device: cpu | cuda | mps")

    # Tracking
    p.add_argument("--max-age",   type=int,   default=30,
                   help="Frames to keep a lost track alive")
    p.add_argument("--min-hits",  type=int,   default=3,
                   help="Frames before a new track is confirmed")
    p.add_argument("--track-iou", type=float, default=0.30,
                   help="IoU threshold for detection–track matching")

    # Visualisation
    p.add_argument("--no-trail", action="store_true",
                   help="Disable motion trail")
    p.add_argument("--no-conf",  action="store_true",
                   help="Hide confidence score on labels")

    # Output
    p.add_argument("--save",       action="store_true",
                   help="Save the annotated video to disk")
    p.add_argument("--output",     default="output_tracked.avi",
                   help="Output video file path (used with --save)")
    p.add_argument("--headless",   action="store_true",
                   help="Run without display (for servers / CI)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (useful in headless mode)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
