"""
Visualization utilities for object detection and tracking.
"""

import colorsys
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────

def generate_color_palette(n: int = 100) -> list[tuple[int, int, int]]:
    """Generate N visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360           # golden-angle spacing
        sat = 0.75 + (i % 3) * 0.083        # slight saturation variation
        val = 0.85 + (i % 2) * 0.10
        r, g, b = colorsys.hsv_to_rgb(hue / 360, sat, val)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


COLOR_PALETTE = generate_color_palette(200)


def get_color(track_id: int) -> tuple[int, int, int]:
    return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

def draw_bounding_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    corner_ratio: float = 0.25,
) -> None:
    """Draw a stylised bounding box with corner accents."""
    cw = int((x2 - x1) * corner_ratio)
    ch = int((y2 - y1) * corner_ratio)

    # Thin full rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Thick corners
    t = thickness + 1
    # top-left
    cv2.line(frame, (x1, y1), (x1 + cw, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + ch), color, t)
    # top-right
    cv2.line(frame, (x2, y1), (x2 - cw, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + ch), color, t)
    # bottom-left
    cv2.line(frame, (x1, y2), (x1 + cw, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - ch), color, t)
    # bottom-right
    cv2.line(frame, (x2, y2), (x2 - cw, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - ch), color, t)


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draw a filled label badge above a bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Badge background
    bx1, by1 = x, y - th - 2 * padding
    bx2, by2 = x + tw + 2 * padding, y

    # Clamp to frame
    bx1 = max(bx1, 0); by1 = max(by1, 0)
    bx2 = min(bx2, frame.shape[1]); by2 = min(by2, frame.shape[0])

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, cv2.FILLED)

    # Contrast text colour (white or black)
    lum = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if lum > 127 else (255, 255, 255)

    cv2.putText(
        frame, text,
        (bx1 + padding, by2 - padding + baseline // 2 - 1),
        font, font_scale, text_color, thickness, cv2.LINE_AA,
    )


def draw_trail(
    frame: np.ndarray,
    trail: list[tuple[int, int]],
    color: tuple[int, int, int],
    max_thickness: int = 3,
) -> None:
    """Draw a fading motion trail."""
    n = len(trail)
    for i in range(1, n):
        alpha = i / n
        thickness = max(1, int(max_thickness * alpha))
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, trail[i - 1], trail[i], faded, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────

def draw_hud(
    frame: np.ndarray,
    fps: float,
    num_tracks: int,
    num_detections: int,
    model_name: str,
    source_label: str,
    paused: bool,
) -> None:
    """Draw an information overlay in the top-left corner."""
    lines = [
        f"Model : {model_name}",
        f"Source: {source_label}",
        f"FPS   : {fps:5.1f}",
        f"Tracks: {num_tracks:3d}",
        f"Dets  : {num_detections:3d}",
    ]
    if paused:
        lines.append("[ PAUSED ]")

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness  = 1
    padding    = 6
    line_h     = 18

    panel_w = 210
    panel_h = len(lines) * line_h + padding * 2

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        color = (0, 255, 100) if "PAUSED" in line else (200, 230, 200)
        cv2.putText(
            frame, line,
            (padding, padding + (i + 1) * line_h - 4),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )


def draw_controls_hint(frame: np.ndarray) -> None:
    """Draw keyboard shortcuts in the bottom-left corner."""
    lines = ["[Q] Quit  [P] Pause  [S] Screenshot  [R] Reset IDs"]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    h, w       = frame.shape[:2]
    padding    = 5
    line_h     = 14

    panel_h = len(lines) * line_h + padding * 2
    panel_w = 370

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, h - panel_h),
        (panel_w, h),
        (20, 20, 20),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (padding, h - panel_h + padding + (i + 1) * line_h),
            font, font_scale, (180, 180, 180), thickness, cv2.LINE_AA,
        )
