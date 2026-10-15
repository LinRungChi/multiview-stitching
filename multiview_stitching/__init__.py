"""Multi-view image registration, mosaicking and cross-view detection de-duplication.

A compact, dependency-light toolkit for the common case of several cameras
looking down at one planar-ish scene with overlapping fields of view.
"""
from .registration import match_features, estimate_homography, PairRegistration
from .mosaic import Mosaic, build_mosaic
from .dedup import Detection, deduplicate
from .synthetic import make_synthetic_views

__all__ = [
    "match_features", "estimate_homography", "PairRegistration",
    "Mosaic", "build_mosaic",
    "Detection", "deduplicate",
    "make_synthetic_views",
]
__version__ = "0.1.0"
