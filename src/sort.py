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


def associate(dets, preds, iou_thr):
    """Hungarian matching on IoU. Returns matches (det_i, trk_j), unmatched dets, unmatched trks."""
    if len(dets) == 0 or len(preds) == 0:
        return [], list(range(len(dets))), list(range(len(preds)))
    iou = iou_matrix(dets, preds)
    r, c = linear_sum_assignment(-iou)
    matches = [(i, j) for i, j in zip(r, c) if iou[i, j] >= iou_thr]
    md, mt = {i for i, _ in matches}, {j for _, j in matches}
    return matches, [i for i in range(len(dets)) if i not in md], [j for j in range(len(preds)) if j not in mt]


class Tracker:
    """SORT tracker. With byte=True, low-score detections are used in a second association
    round to recover occluded tracks (BYTE), but they never start new tracks."""

    def __init__(self, max_age=30, min_hits=3, iou_thr=0.3, det_thr=0.5, low_thr=0.1,
                 byte=False, new_track_thr=None):
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_thr
        self.det_thr, self.low_thr, self.byte = det_thr, low_thr, byte
        self.new_track_thr = det_thr if new_track_thr is None else new_track_thr
        self.tracks, self.frame, self.next_id = [], 0, 1

    def update(self, boxes, scores):
        """boxes (N,4) xyxy, scores (N,). Returns array of [x1, y1, x2, y2, id, score]."""
        self.frame += 1
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
        m1, ud, ut = associate(hb, preds, self.iou_thr)
        for i, j in m1:
            self.tracks[j].update(hb[i], hs[i])

        # round 2 (BYTE): low-score detections vs still-unmatched tracks
        if self.byte and len(ut) and len(lb):
            m2, _, ut2 = associate(lb, preds[ut], 0.5)
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
