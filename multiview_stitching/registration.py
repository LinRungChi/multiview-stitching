"""Pairwise feature matching and homography estimation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class PairRegistration:
    """Result of registering image ``src`` onto image ``dst``."""
    H: np.ndarray                 # 3x3 homography mapping src pixels -> dst pixels
    n_matches: int                # matches passed to RANSAC
    n_inliers: int                # matches consistent with H
    reproj_rmse: float            # RMSE of inlier reprojection (px)

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0


def match_features(img_a: np.ndarray, img_b: np.ndarray, *, detector: str = "sift",
                   ratio: float = 0.75, max_features: int = 4000):
    """Detect and match local features between two images.

    Returns ``(pts_a, pts_b)`` as float32 arrays of shape (N, 2) in pixel
    coordinates, after Lowe's ratio test.
    """
    gray_a = img_a if img_a.ndim == 2 else cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = img_b if img_b.ndim == 2 else cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    if detector == "sift":
        det = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    elif detector == "orb":
        det = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"unknown detector {detector!r}; use 'sift' or 'orb'")

    kp_a, des_a = det.detectAndCompute(gray_a, None)
    kp_b, des_b = det.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    matcher = cv2.BFMatcher(norm)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts_a, pts_b


def estimate_homography(pts_src: np.ndarray, pts_dst: np.ndarray, *,
                        ransac_thresh: float = 3.0, min_matches: int = 8) -> PairRegistration:
    """Robustly fit ``H`` such that ``dst ~ H @ src`` with RANSAC."""
    if len(pts_src) < min_matches:
        raise ValueError(f"need at least {min_matches} matches, got {len(pts_src)}")
    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("RANSAC failed to find a homography")
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pts_src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((proj - pts_dst[inl]) ** 2, axis=1)))) if inl.any() else float("nan")
    return PairRegistration(H=H, n_matches=int(len(pts_src)), n_inliers=int(inl.sum()), reproj_rmse=rmse)


def register_pair(img_src: np.ndarray, img_dst: np.ndarray, **kw) -> PairRegistration:
    """Convenience: match + estimate in one call."""
    detector = kw.pop("detector", "sift")
    a, b = match_features(img_src, img_dst, detector=detector)
    return estimate_homography(a, b, **kw)
