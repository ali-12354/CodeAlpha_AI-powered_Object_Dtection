# CodeAlpha_AI-powered_Object_Dtection
# 🎯 Real-Time Object Detection and Tracking

A Python pipeline for real-time multi-object detection and tracking using YOLOv8 and the SORT algorithm (Simple Online and Realtime Tracking).

## Features
- 🔍 **YOLOv8 detection** via Ultralytics — supports `n`, `s`, `m`, `l`, `x` weight variants
- 📦 **SORT tracker** built from scratch — Kalman Filter + Hungarian algorithm
- 🎨 **Rich visualization** — corner-accented bounding boxes, fading motion trails, per-ID color palette
- 📊 **Live HUD** — real-time FPS, track count, and detection count overlay
- 🎥 **Flexible input** — webcam index or any video file (MP4, AVI, etc.)
- 💾 **Optional video export** — save annotated output with `--save`
- ⌨️ **Interactive controls** — pause, screenshot, reset track IDs on the fly

## Tech Stack
`Python` · `OpenCV` · `YOLOv8 (Ultralytics)` · `FilterPy` · `SciPy` · `NumPy` · `PyTorch`

## Quick Start

```bash
pip install -r requirements.txt

# Webcam
python main_tracker.py

# Video file
python main_tracker.py --source video.mp4

# Filter to people & cars, save output
python main_tracker.py --source video.mp4 --classes 0 2 --save
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `P` | Pause / Resume |
| `S` | Save screenshot |
| `R` | Reset track IDs |

## Project Structure

```
├── main_tracker.py   # Entry point — CLI, video loop, pipeline
├── sort_tracker.py   # SORT: Kalman Filter + Hungarian algorithm
├── viz_utils.py      # Drawing helpers — boxes, labels, trails, HUD
├── test_tracker.py   # Smoke tests (no webcam or GPU required)
└── requirements.txt  # Dependencies
```

## How It Works

1. **Detect** — YOLOv8 runs inference on each frame, returning bounding boxes, class labels, and confidence scores.
2. **Track** — SORT propagates existing tracks via a Kalman Filter (constant-velocity model), then matches them to new detections using the Hungarian algorithm on an IoU cost matrix.
3. **Visualize** — Tracks are drawn with unique colors, motion trails, and ID-labeled badges in real time.
