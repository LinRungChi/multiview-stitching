"""Generate a textured planar scene seen by several overlapping cameras.

Useful for tests and demos: ground-truth homographies and object positions
are known, so registration and de-duplication can be checked numerically.
"""
from __future__ import annotations
import numpy as np
import cv2


def _textured_scene(w: int, h: int, n_objects: int, rng: np.random.Generator):
    scene = np.full((h, w, 3), 40, np.uint8)
    # noise texture so feature detectors have something to lock onto
    noise = rng.integers(0, 60, (h // 8, w // 8, 3), np.uint8)
    scene = cv2.add(scene, cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC))
    for _ in range(300):
        p = rng.integers(0, [w, h]); c = tuple(int(v) for v in rng.integers(60, 200, 3))
        cv2.circle(scene, tuple(int(v) for v in p), int(rng.integers(2, 6)), c, -1)
    objects = []
    for _ in range(n_objects):
        cx, cy = rng.uniform(40, w - 40), rng.uniform(40, h - 40)
        a, b = rng.uniform(10, 16), rng.uniform(5, 8)
        ang = rng.uniform(0, 180)
        cv2.ellipse(scene, (int(cx), int(cy)), (int(a), int(b)), ang, 0, 360, (230, 230, 230), -1)
        objects.append((cx, cy, a))
    return scene, objects


def make_synthetic_views(*, scene_size=(1400, 900), view_size=(640, 480), grid=(2, 2), overlap=0.35,
                         n_objects=40, jitter_deg=4.0, seed=0):
    """Return ``(views, H_true, objects, scene)``.

    ``views[i]`` is what camera i sees; ``H_true[i]`` maps view-i pixels to
    scene pixels; ``objects`` is a list of (x, y, radius) in scene pixels.
    """
    rng = np.random.default_rng(seed)
    sw, sh = scene_size
    vw, vh = view_size
    scene, objects = _textured_scene(sw, sh, n_objects, rng)

    cols, rows = grid
    step_x = (sw - vw) / max(cols - 1, 1) if cols > 1 else 0
    step_y = (sh - vh) / max(rows - 1, 1) if rows > 1 else 0
    # shrink the step so neighbours overlap by `overlap` of the view width/height
    step_x = min(step_x, vw * (1 - overlap)); step_y = min(step_y, vh * (1 - overlap))
    x0 = (sw - (step_x * (cols - 1) + vw)) / 2
    y0 = (sh - (step_y * (rows - 1) + vh)) / 2

    views, H_true = [], []
    for r in range(rows):
        for c in range(cols):
            tx, ty = x0 + c * step_x, y0 + r * step_y
            ang = np.deg2rad(rng.uniform(-jitter_deg, jitter_deg))
            s = rng.uniform(0.95, 1.05)
            # scene <- view: rotate/scale about view centre, then translate
            R = cv2.getRotationMatrix2D((vw / 2, vh / 2), np.rad2deg(ang), s)
            A = np.vstack([R, [0, 0, 1]])
            A[0, 2] += tx; A[1, 2] += ty
            # mild perspective
            P = np.eye(3); P[2, 0] = rng.uniform(-1e-5, 1e-5); P[2, 1] = rng.uniform(-1e-5, 1e-5)
            H = A @ P
            view = cv2.warpPerspective(scene, np.linalg.inv(H), (vw, vh), flags=cv2.INTER_LINEAR)
            # per-camera brightness difference so blending is tested too
            view = cv2.convertScaleAbs(view, alpha=rng.uniform(0.9, 1.1), beta=rng.uniform(-10, 10))
            views.append(view); H_true.append(H)
    return views, H_true, objects, scene


def project_objects_to_view(H_view_to_scene: np.ndarray, objects, view_size):
    """Ground-truth boxes of scene objects as they appear in one view (fully inside only)."""
    vw, vh = view_size
    Hi = np.linalg.inv(H_view_to_scene)
    boxes = []
    for (x, y, r) in objects:
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hi).reshape(2)
        if 0 <= p[0] - r and p[0] + r <= vw and 0 <= p[1] - r and p[1] + r <= vh:
            boxes.append((p[0] - r, p[1] - r, p[0] + r, p[1] + r))
    return boxes
