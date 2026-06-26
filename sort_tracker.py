"""
SORT: Simple Online and Realtime Tracking
Implementation of the SORT algorithm using Kalman Filter + Hungarian Algorithm.
Reference: Bewley et al., 2016 (https://arxiv.org/abs/1602.00763)
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────
# Bounding-box helpers
# ─────────────────────────────────────────────────────────────

def bbox_to_z(bbox):
    """Convert [x1,y1,x2,y2] → state vector [cx, cy, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # scale (area)
    r = w / float(h)   # aspect ratio
    return np.array([cx, cy, s, r]).reshape((4, 1))


def z_to_bbox(z, score=None):
    """Convert state vector [cx, cy, s, r] → [x1, y1, x2, y2, (score)]."""
    w = np.sqrt(abs(z[2] * z[3]))
    h = abs(z[2]) / w if w != 0 else 0
    x1 = z[0] - w / 2.0
    y1 = z[1] - h / 2.0
    x2 = z[0] + w / 2.0
    y2 = z[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


def iou_batch(bb_test, bb_gt):
    """
    Compute IoU between every pair of boxes.
    bb_test: (N, 4), bb_gt: (M, 4)  →  (N, M) IoU matrix.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[..., 2]   - bb_gt[..., 0])   * (bb_gt[..., 3]   - bb_gt[..., 1])
    union = area_test + area_gt - intersection

    return intersection / np.maximum(union, 1e-6)


def associate_detections(detections, trackers, iou_threshold=0.3):
    """
    Match detections to existing trackers using the Hungarian algorithm.

    Returns:
        matches         – (N, 2) array of [det_idx, trk_idx]
        unmatched_dets  – indices of unmatched detections
        unmatched_trks  – indices of unmatched trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty(0, dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    # Hungarian matching (maximise IoU → minimise cost = 1-IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.stack([row_ind, col_ind], axis=1)

    unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trks = [t for t in range(len(trackers))   if t not in matched_indices[:, 1]]

    # Filter low-IoU matches
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# ─────────────────────────────────────────────────────────────
# Kalman-based single-object tracker
# ─────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Represents one tracked object as a constant-velocity Kalman Filter.

    State vector: [cx, cy, s, r, vx, vy, vs]
    Measurement:  [cx, cy, s, r]
    """
    count = 0  # class-level ID counter

    def __init__(self, bbox, class_id=0, class_name="object"):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant-velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Covariance matrix — high uncertainty for unobserved velocity
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P          *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history   = []
        self.hits      = 0
        self.hit_streak = 0
        self.age       = 0

        self.class_id   = class_id
        self.class_name = class_name
        self.score      = 1.0

        # Trail (centre points) for visualisation
        self.trail: list[tuple[int, int]] = []

    def update(self, bbox, score=1.0):
        """Update tracker with a new matched detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.kf.update(bbox_to_z(bbox))

    def predict(self):
        """Advance state estimate by one time-step."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """Return current bounding-box estimate [x1, y1, x2, y2]."""
        return z_to_bbox(self.kf.x)[0]


# ─────────────────────────────────────────────────────────────
# Multi-object SORT tracker
# ─────────────────────────────────────────────────────────────

class SORTTracker:
    """
    SORT multi-object tracker.

    Args:
        max_age        – frames to keep a track alive without a detection match
        min_hits       – frames a track must be seen before being returned
        iou_threshold  – minimum IoU to consider a detection–track pair matched
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def reset(self):
        """Clear all tracks and reset ID counter."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def update(self, detections: np.ndarray):
        """
        Process one frame of detections.

        Args:
            detections: np.ndarray of shape (N, 6)
                        columns: [x1, y1, x2, y2, score, class_id]
                        Pass np.empty((0, 6)) when there are no detections.

        Returns:
            np.ndarray of shape (M, 7):
                [x1, y1, x2, y2, track_id, class_id, score]
        """
        self.frame_count += 1

        # ── 1. Predict new locations for all existing trackers ──
        trk_boxes = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_boxes.append(pos)

        for t in reversed(to_del):
            self.trackers.pop(t)

        trk_boxes = np.array(trk_boxes) if trk_boxes else np.empty((0, 4))

        # ── 2. Match detections to predictions ──
        det_boxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))
        matches, unmatched_dets, unmatched_trks = associate_detections(
            det_boxes, trk_boxes, self.iou_threshold
        )

        # ── 3. Update matched trackers ──
        for m in matches:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                score=float(detections[det_idx, 4]),
            )

        # ── 4. Create new trackers for unmatched detections ──
        for i in unmatched_dets:
            det = detections[i]
            class_id = int(det[5]) if len(det) > 5 else 0
            new_trk = KalmanBoxTracker(det[:4], class_id=class_id)
            new_trk.score = float(det[4])
            self.trackers.append(new_trk)

        # ── 5. Collect outputs and prune dead tracks ──
        results = []
        survivors = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                survivors.append(trk)
                if (trk.hit_streak >= self.min_hits or
                        self.frame_count <= self.min_hits):
                    bbox = trk.get_state()
                    # Update trail
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    trk.trail.append((cx, cy))
                    if len(trk.trail) > 30:
                        trk.trail.pop(0)

                    results.append(
                        [bbox[0], bbox[1], bbox[2], bbox[3],
                         trk.id, trk.class_id, trk.score]
                    )

        self.trackers = survivors

        return np.array(results) if results else np.empty((0, 7))
