"""CLEAR-MOT and identity metrics on MOT17 val-half, following the MOTChallenge preprocessing.

Per frame, tracker boxes are first matched (IoU >= 0.5, Hungarian) against *all* ground-truth boxes;
tracker boxes matched to distractor classes (static person, reflection, person on vehicle, distractor)
are discarded, then the remaining tracker boxes are scored against pedestrian ground truth with
consider-flag 1. This mirrors TrackEval's MotChallenge2DBox preprocessing.
"""
import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

import mot
from sort import iou_matrix

METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses",
           "mostly_tracked", "mostly_lost", "precision", "recall", "num_unique_objects"]


def accumulate(seq, results, frames):
    """results: dict frame -> array [x1, y1, x2, y2, id, score]."""
    gt = mot.load_gt(seq)
    acc = mm.MOTAccumulator(auto_id=False)
    for f in frames:
        g = gt[gt[:, 0] == f]
        gb = g[:, 2:6].copy(); gb[:, 2:] += gb[:, :2]
        t = results.get(f, np.zeros((0, 6)))
        tb, tid = t[:, :4], t[:, 4].astype(int)

        # remove tracker boxes that match distractor GT
        if len(t) and len(g):
            iou = iou_matrix(tb, gb)
            iou_m = np.where(iou >= 0.5 - 1e-9, iou, 0)
            r, c = linear_sum_assignment(-iou_m)
            drop = [i for i, j in zip(r, c) if iou_m[i, j] > 0 and int(g[j, 7]) in mot.DISTRACTORS]
            keep = np.setdiff1d(np.arange(len(t)), drop)
            tb, tid = tb[keep], tid[keep]

        valid = (g[:, 7] == mot.PEDESTRIAN) & (g[:, 6] != 0)
        gb, gid = gb[valid], g[valid, 1].astype(int)
        dist = 1 - iou_matrix(gb, tb)
        dist[dist > 0.5] = np.nan
        acc.update(gid, tid, dist, frameid=f)
    return acc


def summarize(accs, names):
    mh = mm.metrics.create()
    return mh.compute_many(accs, metrics=METRICS, names=names, generate_overall=True)
