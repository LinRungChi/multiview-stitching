"""End-to-end demo on synthetic data: stitch 4 overlapping views and de-duplicate detections.

    python examples/demo.py --out docs
"""
import argparse, pathlib
import numpy as np, cv2
from multiview_stitching import make_synthetic_views, build_mosaic, Detection, deduplicate
from multiview_stitching.synthetic import project_objects_to_view
from multiview_stitching.dedup import to_canvas_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    views, H_true, objects, scene = make_synthetic_views(grid=(2, 2), n_objects=40, seed=args.seed)
    m = build_mosaic(views, order=[(1, 0), (2, 0), (3, 1)])
    for k, s in enumerate(m.pair_stats):
        print(f"pair {k}: {s.n_inliers}/{s.n_matches} inliers, reproj RMSE {s.reproj_rmse:.2f} px")

    # input views side by side
    tiles = [cv2.resize(v, (320, 240)) for v in views]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out / "views.jpg"), grid)

    canvas = m.render(views)
    cv2.imwrite(str(out / "mosaic.jpg"), canvas)

    # simulated per-view detections (ground-truth boxes stand in for a detector)
    dets = [Detection(view=i, box=b) for i in range(4) for b in project_objects_to_view(H_true[i], objects, (640, 480))]
    merged = deduplicate(m, dets, iou_thresh=0.3)
    print(f"raw detections across views: {len(dets)}  ->  after de-duplication: {len(merged)}  (objects in scene: {len(objects)})")

    # left: every raw detection, coloured by source view (duplicates show as stacked boxes)
    # right: after de-duplication
    colors = [(0, 200, 255), (255, 120, 0), (255, 0, 200), (0, 255, 120)]
    raw = canvas.copy(); fin = canvas.copy()
    for d in dets:
        b = to_canvas_box(m, d).astype(int); pad = 3 + 4 * d.view
        cv2.rectangle(raw, (b[0] - pad, b[1] - pad), (b[2] + pad, b[3] + pad), colors[d.view], 2)
    for d in merged:
        b = to_canvas_box(m, d).astype(int)
        cv2.rectangle(fin, (b[0] - 3, b[1] - 3), (b[2] + 3, b[3] + 3), (0, 255, 0), 2)
    for im, txt in ((raw, f"raw: {len(dets)} boxes from 4 views"), (fin, f"after cross-view NMS: {len(merged)} objects")):
        cv2.putText(im, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(im, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    vis = np.hstack([raw, np.full((raw.shape[0], 8, 3), 255, np.uint8), fin])
    cv2.imwrite(str(out / "dedup.jpg"), vis)
    print("wrote", out / "views.jpg", out / "mosaic.jpg", out / "dedup.jpg")


if __name__ == "__main__":
    main()
