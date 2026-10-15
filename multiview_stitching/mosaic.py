"""Chain pairwise homographies into one reference frame and render a mosaic."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2
from .registration import register_pair, PairRegistration


@dataclass
class Mosaic:
    """A set of images expressed in one common (reference) pixel frame."""
    H_to_ref: list[np.ndarray]           # per-image homography into the reference image frame
    offset: np.ndarray                   # translation applied so the mosaic has non-negative coords
    size: tuple[int, int]                # (width, height) of the mosaic canvas
    pair_stats: list[PairRegistration] = field(default_factory=list)

    def to_canvas(self, i: int) -> np.ndarray:
        """Homography mapping image ``i`` pixels -> mosaic canvas pixels."""
        T = np.array([[1, 0, self.offset[0]], [0, 1, self.offset[1]], [0, 0, 1]], float)
        return T @ self.H_to_ref[i]

    def project_points(self, i: int, pts: np.ndarray) -> np.ndarray:
        """Map (N,2) pixel points from image ``i`` into the canvas."""
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.to_canvas(i)).reshape(-1, 2)

    def render(self, images: list[np.ndarray], *, blend: str = "feather") -> np.ndarray:
        """Warp every image into the canvas and merge them.

        ``blend='feather'`` weights each pixel by distance to its image border,
        which hides seams in the overlap; ``blend='overwrite'`` simply pastes
        later images over earlier ones.
        """
        W, Hh = self.size
        acc = np.zeros((Hh, W, 3), np.float64)
        wsum = np.zeros((Hh, W), np.float64)
        for i, im in enumerate(images):
            im3 = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            Hc = self.to_canvas(i)
            warped = cv2.warpPerspective(im3, Hc, (W, Hh)).astype(np.float64)
            if blend == "feather":
                h, w = im3.shape[:2]
                base = np.ones((h, w), np.uint8)
                base[0, :] = base[-1, :] = base[:, 0] = base[:, -1] = 0
                dist = cv2.distanceTransform(base, cv2.DIST_L2, 3)
                wgt = cv2.warpPerspective(dist / dist.max(), Hc, (W, Hh))
            else:
                wgt = cv2.warpPerspective(np.ones(im3.shape[:2], np.float32), Hc, (W, Hh))
                acc[wgt > 0] = 0
                wsum[wgt > 0] = 0
            acc += warped * wgt[..., None]
            wsum += wgt
        out = acc / np.maximum(wsum, 1e-9)[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


def build_mosaic(images: list[np.ndarray], *, reference: int = 0, order: list[tuple[int, int]] | None = None,
                 **register_kw) -> Mosaic:
    """Register a list of images into one frame.

    ``order`` is a list of ``(src, dst)`` pairs describing which image is
    registered onto which; by default each image i>0 is registered onto i-1
    (a chain). ``dst`` must already be connected to the reference.
    """
    n = len(images)
    if order is None:
        order = [(i, i - 1) for i in range(1, n)]
    H_to_ref: list[np.ndarray | None] = [None] * n
    H_to_ref[reference] = np.eye(3)
    stats: list[PairRegistration] = []

    pending = list(order)
    while pending:
        progressed = False
        for src, dst in list(pending):
            if H_to_ref[dst] is None:
                continue
            reg = register_pair(images[src], images[dst], **register_kw)
            H_to_ref[src] = H_to_ref[dst] @ reg.H
            stats.append(reg)
            pending.remove((src, dst))
            progressed = True
        if not progressed:
            raise RuntimeError(f"cannot connect pairs {pending} to the reference image")
    if any(h is None for h in H_to_ref):
        raise RuntimeError("some images were never registered; check `order`")

    corners = []
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(c, H_to_ref[i]).reshape(-1, 2))
    allc = np.vstack(corners)
    mn, mx = allc.min(0), allc.max(0)
    offset = -np.floor(mn)
    size = tuple(int(v) for v in np.ceil(mx - np.floor(mn)))
    return Mosaic(H_to_ref=[h for h in H_to_ref], offset=offset, size=size, pair_stats=stats)
