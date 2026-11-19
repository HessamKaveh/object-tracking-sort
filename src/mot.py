"""MOT17 helpers: sequences, the train/val half split, ground truth and public detections.

Protocol (as in ByteTrack / FairMOT ablations): every MOT17 *train* sequence is cut in two; the first
half of the frames is used to fine-tune the detector and the second half ("val-half") for evaluation.
The images are identical across the DPM/FRCNN/SDP copies, so only the -FRCNN copies are used.
"""
import configparser
import os

import numpy as np

MOT_ROOT = os.environ.get("MOT17_ROOT", os.path.expanduser("~/datasets/portfolio/mot17/MOT17"))
SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]

PEDESTRIAN = 1
DISTRACTORS = {2, 7, 8, 12}  # person on vehicle, static person, distractor, reflection


def seq_dir(seq, det="FRCNN"):
    return os.path.join(MOT_ROOT, "train", f"{seq}-{det}")


def seq_info(seq):
    c = configparser.ConfigParser()
    c.read(os.path.join(seq_dir(seq), "seqinfo.ini"))
    s = c["Sequence"]
    return {"length": int(s["seqLength"]), "fps": float(s["frameRate"]),
            "w": int(s["imWidth"]), "h": int(s["imHeight"]), "imdir": s["imDir"], "ext": s["imExt"]}


def half_frames(seq, half):
    """Frame numbers (1-based) for half='train' (first half) or half='val' (second half)."""
    n = seq_info(seq)["length"]
    mid = n // 2 + 1
    return range(1, mid) if half == "train" else range(mid, n + 1)


def frame_path(seq, f):
    i = seq_info(seq)
    return os.path.join(seq_dir(seq), i["imdir"], f"{f:06d}{i['ext']}")


def load_gt(seq):
    """Rows: frame, id, x, y, w, h, consider, class, visibility."""
    return np.loadtxt(os.path.join(seq_dir(seq), "gt", "gt.txt"), delimiter=",")


def load_public_dets(seq, det="FRCNN"):
    """MOT17 public detections -> {frame: (boxes_xyxy, scores)}."""
    d = np.loadtxt(os.path.join(seq_dir(seq, det), "det", "det.txt"), delimiter=",", ndmin=2)
    out = {}
    for f in np.unique(d[:, 0]).astype(int):
        r = d[d[:, 0] == f]
        b = r[:, 2:6].copy(); b[:, 2:] += b[:, :2]
        out[f] = (b, r[:, 6])
    return out


def write_mot(path, rows):
    """rows: iterable of (frame, id, x1, y1, x2, y2, score) -> MOTChallenge txt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, x1, y1, x2, y2, s in rows:
            f.write(f"{int(fr)},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.3f},-1,-1,-1\n")
