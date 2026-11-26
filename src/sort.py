"""SORT (Bewley et al., ICIP 2016) and a BYTE-style two-stage association (Zhang et al., ECCV 2022).

Boxes are [x1, y1, x2, y2]. The Kalman state is [u, v, s, r, u', v', s'] (centre, area, aspect ratio
and their velocities; the aspect ratio is assumed constant), exactly as in the original SORT.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def _enclosing(a, b):
    """Broadcast helper: (union area, enclosing-box width, height, centre offsets) for a[:,None], b[None]."""
    a, b = a[:, None, :], b[None, :, :]
    ew = np.maximum(a[..., 2], b[..., 2]) - np.minimum(a[..., 0], b[..., 0])
    eh = np.maximum(a[..., 3], b[..., 3]) - np.minimum(a[..., 1], b[..., 1])
    du = (a[..., 0] + a[..., 2] - b[..., 0] - b[..., 2]) / 2
    dv = (a[..., 1] + a[..., 3] - b[..., 1] - b[..., 3]) / 2
    return ew, eh, du, dv


def giou_matrix(a, b):
    """Generalised IoU (Rezatofighi et al., CVPR 2019), in [-1, 1]. Still informative for disjoint boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iou = iou_matrix(a, b)
    ew, eh, _, _ = _enclosing(a, b)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    inter = iou * (area(a)[:, None] + area(b)[None, :]) / (1 + iou)
    union = area(a)[:, None] + area(b)[None, :] - inter
    enc = ew * eh + 1e-9
    return iou - (enc - union) / enc


def diou_matrix(a, b):
    """Distance-IoU (Zheng et al., AAAI 2020): IoU minus squared centre distance over the enclosing diagonal."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ew, eh, du, dv = _enclosing(a, b)
    return iou_matrix(a, b) - (du ** 2 + dv ** 2) / (ew ** 2 + eh ** 2 + 1e-9)


SIMILARITY = {"iou": iou_matrix, "giou": giou_matrix, "diou": diou_matrix}


def box_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def x_to_box(x):
    s, r = max(x[2], 1e-6), x[3]
    w = np.sqrt(s * r); h = s / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxTrack:
    """Constant-velocity Kalman filter on (u, v, s, r), written out in numpy."""

    F = np.eye(7); F[0, 4] = F[1, 5] = F[2, 6] = 1
    H = np.eye(4, 7)
    R = np.diag([1.0, 1.0, 10.0, 10.0])
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])

    def __init__(self, box, score, tid):
        self.x = np.zeros(7); self.x[:4] = box_to_z(box)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.id, self.score = tid, score
        self.hits, self.hit_streak, self.age, self.time_since_update = 1, 1, 0, 0

    def predict(self):
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_box(self.x)

    def warp(self, A):
        """Move the state into the next frame's pixel coordinates under a 2x3 similarity transform A
        (camera-motion compensation): centre and its velocity rotate/scale/shift, area scales by det."""
        L, t = A[:, :2], A[:, 2]
        sc = abs(np.linalg.det(L))
        self.x[:2] = L @ self.x[:2] + t
        self.x[4:6] = L @ self.x[4:6]
        self.x[2] *= sc; self.x[6] *= sc
        T = np.eye(7)
        T[:2, :2] = T[4:6, 4:6] = L
        T[2, 2] = T[6, 6] = sc
        self.P = T @ self.P @ T.T

    def update(self, box, score):
        z = box_to_z(box)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1; self.hit_streak += 1
        self.score = score

    @property
    def box(self):
        return x_to_box(self.x)


def associate(dets, preds, iou_thr, sim="iou"):
    """Hungarian matching on a box similarity ('iou', 'giou' or 'diou'); pairs below iou_thr are rejected.
    Returns matches (det_i, trk_j), unmatched dets, unmatched trks."""
    if len(dets) == 0 or len(preds) == 0:
        return [], list(range(len(dets))), list(range(len(preds)))
    iou = SIMILARITY[sim](dets, preds)
    r, c = linear_sum_assignment(-iou)
    matches = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thr]
    md, mt = {i for i, _ in matches}, {j for _, j in matches}
    return matches, [i for i in range(len(dets)) if i not in md], [j for j in range(len(preds)) if j not in mt]


class Tracker:
    """SORT tracker. With byte=True, low-score detections are used in a second association
    round to recover occluded tracks (BYTE), but they never start new tracks.

    sim selects the association similarity of the first round ('iou' as in SORT, or 'giou' / 'diou',
    which keep ranking candidate pairs when the predicted and detected boxes no longer overlap, e.g. at a
    low processing frame rate); byte_thr is the threshold of the second round."""

    def __init__(self, max_age=30, min_hits=3, iou_thr=0.3, det_thr=0.5, low_thr=0.1,
                 byte=False, new_track_thr=None, sim="iou", byte_thr=0.5):
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_thr
        self.sim, self.byte_thr = sim, byte_thr
        self.det_thr, self.low_thr, self.byte = det_thr, low_thr, byte
        self.new_track_thr = det_thr if new_track_thr is None else new_track_thr
        self.tracks, self.frame, self.next_id = [], 0, 1

    def update(self, boxes, scores, warp=None):
        """boxes (N,4) xyxy, scores (N,). warp: optional 2x3 camera-motion transform from the previous
        processed frame to this one. Returns array of [x1, y1, x2, y2, id, score]."""
        self.frame += 1
        if warp is not None:
            for t in self.tracks:
                t.warp(np.asarray(warp, float))
        boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float).reshape(-1)
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        keep = np.isfinite(preds).all(1)
        self.tracks = [t for t, k in zip(self.tracks, keep) if k]
        preds = preds[keep]

        high = scores >= self.det_thr
        low = (scores >= self.low_thr) & ~high if self.byte else np.zeros_like(high)
        hb, hs = boxes[high], scores[high]
        lb, ls = boxes[low], scores[low]

        # round 1: confident detections vs all tracks
        m1, ud, ut = associate(hb, preds, self.iou_thr, self.sim)
        for i, j in m1:
            self.tracks[j].update(hb[i], hs[i])

        # round 2 (BYTE): low-score detections vs still-unmatched tracks
        if self.byte and len(ut) and len(lb):
            m2, _, ut2 = associate(lb, preds[ut], self.byte_thr, self.sim)
            for i, jj in m2:
                self.tracks[ut[jj]].update(lb[i], ls[i])
            ut = [ut[jj] for jj in ut2]

        for i in ud:
            if hs[i] >= self.new_track_thr:
                self.tracks.append(KalmanBoxTrack(hb[i], hs[i], self.next_id)); self.next_id += 1

        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits):
                out.append([*t.box, t.id, t.score])
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(out).reshape(-1, 6)
