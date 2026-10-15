"""Merge detections from overlapping views so each object is counted once."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .mosaic import Mosaic


@dataclass
class Detection:
    view: int
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in that view's pixels
    score: float = 1.0
    label: str = "object"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_canvas_box(mosaic: Mosaic, det: Detection) -> np.ndarray:
    """Project a view-space box into the canvas and return its axis-aligned bounds."""
    x1, y1, x2, y2 = det.box
    pts = mosaic.project_points(det.view, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])


def deduplicate(mosaic: Mosaic, detections: list[Detection], *, iou_thresh: float = 0.3) -> list[Detection]:
    """Greedy cross-view NMS in the shared canvas frame.

    Detections are projected into the mosaic, sorted by score, and a lower
    scoring box from a *different* view is dropped if it overlaps a kept box
    by more than ``iou_thresh``. Boxes from the same view are never merged
    (that is the per-view detector's job).
    """
    order = sorted(range(len(detections)), key=lambda i: -detections[i].score)
    canvas = [to_canvas_box(mosaic, d) for d in detections]
    kept: list[int] = []
    for i in order:
        dup = False
        for j in kept:
            if detections[j].view != detections[i].view and _iou(canvas[i], canvas[j]) > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(i)
    return [detections[i] for i in sorted(kept)]
