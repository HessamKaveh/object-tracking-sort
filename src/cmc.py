"""Camera-motion compensation (CMC): a global 2-D similarity transform between two processed frames.

Corners are tracked with pyramidal Lucas-Kanade on a half-resolution grey image, pixels inside the current
detections are ignored (people move on their own), and a rotation + uniform scale + translation is fitted
with RANSAC (as the sparse-optical-flow CMC of BoT-SORT). The tracker uses the 2x3 matrix to move every
Kalman state from the previous processed frame into the current one before predicting, so ego-motion of a
robot-mounted camera is not mistaken for people moving.

    python src/cmc.py --rates 30 15 10 6 3 2      # cache transforms for every stride the study needs
"""
import argparse
import os
import pickle

import cv2
import numpy as np

import mot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _grey(img, scale):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else g


def estimate(prev, cur, boxes=None, scale=0.5, max_corners=1000):
    """2x3 transform mapping pixel coordinates of `prev` to `cur` (full resolution). Identity on failure."""
    a, b = _grey(prev, scale), _grey(cur, scale)
    mask = np.full(a.shape, 255, np.uint8)
    if boxes is not None:
        for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4) * scale:
            mask[max(0, int(y1)):max(0, int(y2)), max(0, int(x1)):max(0, int(x2))] = 0
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=max_corners, qualityLevel=0.01, minDistance=1, blockSize=3,
                                 mask=mask)
    if p0 is None or len(p0) < 10:
        return IDENTITY.copy()
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=4)
    ok = st.reshape(-1) == 1
    if ok.sum() < 10:
        return IDENTITY.copy()
    A, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if A is None or inl.sum() < 10:
        return IDENTITY.copy()
    A[:, 2] /= scale
    return A


def cache_path(seq):
    return os.path.join(ROOT, "cache", f"cmc_{seq}.pkl")


def load(seq):
    """{(frame, stride): 2x3 transform from frame-stride to frame}; empty dict if not cached."""
    p = cache_path(seq)
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[30, 15, 10, 6, 3, 2])
    ap.add_argument("--half", default="val")
    ap.add_argument("--dets", default="public_frcnn", help="detections used to mask people")
    args = ap.parse_args()
    from run_benchmark import load_dets
    dets = load_dets(args.dets)
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    for seq in mot.SEQS:
        fps = mot.seq_info(seq)["fps"]
        strides = sorted({max(1, round(fps / hz)) for hz in args.rates})
        frames = list(mot.half_frames(seq, args.half))
        out = load(seq)
        imgs = {}
        for f in frames:
            imgs[f] = _grey(cv2.imread(mot.frame_path(seq, f)), 0.5)
        for k in strides:
            for f in frames:
                if (f, k) in out or f - k not in imgs:
                    continue
                boxes = dets[seq].get(f - k, (np.zeros((0, 4)), None))[0]
                A = estimate(imgs[f - k], imgs[f], boxes * 0.5, scale=1.0)
                A[:, 2] *= 2.0                                     # images were pre-scaled by 0.5
                out[(f, k)] = A
            print(seq, "stride", k, "done", flush=True)
        with open(cache_path(seq), "wb") as fh:
            pickle.dump(out, fh)


if __name__ == "__main__":
    main()
