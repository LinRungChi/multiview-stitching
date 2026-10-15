import numpy as np
import cv2
from multiview_stitching import make_synthetic_views, build_mosaic, Detection, deduplicate
from multiview_stitching.registration import register_pair
from multiview_stitching.synthetic import project_objects_to_view


def _corner_error(H_est, H_true_src, H_true_dst, size):
    """Mean px error of mapping src corners with estimated vs true src->dst."""
    w, h = size
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    H_gt = np.linalg.inv(H_true_dst) @ H_true_src
    a = cv2.perspectiveTransform(c, H_est).reshape(-1, 2)
    b = cv2.perspectiveTransform(c, H_gt).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).mean())


def test_pairwise_registration_recovers_truth():
    views, H_true, _, _ = make_synthetic_views(grid=(2, 1), seed=1)
    reg = register_pair(views[1], views[0])
    assert reg.n_inliers >= 30
    assert reg.inlier_ratio > 0.5
    assert reg.reproj_rmse < 2.0
    assert _corner_error(reg.H, H_true[1], H_true[0], (640, 480)) < 3.0


def test_mosaic_chain_of_four_views():
    views, H_true, _, _ = make_synthetic_views(grid=(2, 2), seed=2)
    # 2x2 grid: register 1->0, 2->0, 3->1
    m = build_mosaic(views, order=[(1, 0), (2, 0), (3, 1)])
    assert len(m.pair_stats) == 3
    for i in range(4):
        err = _corner_error(m.H_to_ref[i], H_true[i], H_true[0], (640, 480))
        assert err < 4.0, f"view {i} corner error {err:.2f}px"
    canvas = m.render(views)
    assert canvas.shape[0] == m.size[1] and canvas.shape[1] == m.size[0]
    assert canvas.mean() > 20  # not blank


def test_dedup_counts_each_object_once():
    views, H_true, objects, _ = make_synthetic_views(grid=(2, 2), n_objects=40, seed=3)
    m = build_mosaic(views, order=[(1, 0), (2, 0), (3, 1)])
    dets = []
    for i in range(4):
        for b in project_objects_to_view(H_true[i], objects, (640, 480)):
            dets.append(Detection(view=i, box=b))
    # ground truth: number of distinct objects visible in at least one view
    seen = set()
    for i in range(4):
        for (x, y, r) in objects:
            p = cv2.perspectiveTransform(np.float32([[[x, y]]]), np.linalg.inv(H_true[i])).reshape(2)
            if r <= p[0] <= 640 - r and r <= p[1] <= 480 - r:
                seen.add((round(x, 3), round(y, 3)))
    merged = deduplicate(m, dets, iou_thresh=0.3)
    assert len(dets) > len(seen)            # duplicates really exist in the raw detections
    assert abs(len(merged) - len(seen)) <= 1  # and are removed by de-duplication
