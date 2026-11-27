"""Does appearance rescue a slow tracker? Motion vs appearance cues as the update rate drops.

Follow-up to src/framerate_study.py. Camera-motion compensation (CMC) restored identities down to ~6 Hz;
below that, people turn and cross between updates. Appearance does not depend on the frame rate, so does a
ReID embedding take over where motion fails, and is a zero-shot foundation model (DINOv2) good enough?

1. ``calibrate``  per embedder, the appearance threshold is set on the *train* half of every sequence: the
                  cosine distance below which only 1 % of *nearby different people* (centre within one body
                  height) fall. The rate at which the same person, 1 s later, is accepted is reported too.
2. ``cues``       detector-free diagnostic on val-half ground truth: for each person visible in two
                  consecutive processed frames, is the right person ranked first among all people in the
                  earlier frame by (a) constant-velocity IoU, (b) the same after CMC, (c) appearance?
3. ``sweep``      trackers at 30 to 2 Hz: CMC baseline, + appearance for each embedder, + appearance without
                  CMC (can appearance replace ego-motion compensation?), + a 3 s memory for re-identification.

    python src/reid.py embed --dets yolo11s_mot17 public_frcnn oracle_gt      # val-half embeddings
    python src/appearance_study.py embed_train                               # train-half GT embeddings
    python src/appearance_study.py calibrate
    python src/appearance_study.py cues
    python src/appearance_study.py why
    python src/appearance_study.py sweep --dets yolo11s_mot17 public_frcnn oracle_gt
    python src/appearance_study.py plot
"""
import argparse
import json
import math
import os
import pickle

import numpy as np

import cmc
import mot
import reid
from evaluate import accumulate, summarize
from framerate_study import MAX_AGE, MIN_HITS, RATES, oracle_dets, stride
from run_benchmark import ROOT, load_dets
from sort import Tracker, iou_matrix

OUT = os.path.join(ROOT, "results", "appearance")
FAR = 0.01            # false-accept rate among nearby different people used to set each threshold
LABEL = {"imagenet": "ImageNet ResNet-18", "dinov2": "DINOv2 ViT-S/14", "reid": "MOT17 ReID (cross-fold)"}


# ------------------------------------------------------------------------------------------- embeddings
def gt_boxes(seq, half):
    """{frame: (ids, boxes)} for considered pedestrians."""
    g = mot.load_gt(seq)
    g = g[(g[:, 7] == mot.PEDESTRIAN) & (g[:, 6] != 0)]
    out = {}
    for f in mot.half_frames(seq, half):
        r = g[g[:, 0] == f]
        b = r[:, 2:6].copy(); b[:, 2:] += b[:, :2]
        out[f] = (r[:, 1].astype(int), b)
    return out


def embed_train():
    """Embed train-half GT boxes with every embedder (for threshold calibration only)."""
    import cv2
    for kind in reid.EMBEDDERS:
        shared = None if kind == "reid" else reid.Embedder(kind)
        out = {}
        for seq in mot.SEQS:
            emb = reid.Embedder(kind, fold=reid.fold_of(seq)) if kind == "reid" else shared
            out[seq] = {}
            for f, (ids, b) in gt_boxes(seq, "train").items():
                if len(ids):
                    img = cv2.imread(mot.frame_path(seq, f))
                    out[seq][f] = emb(np.stack([reid.crop(img, x) for x in b]))
            print("train-half", kind, seq, flush=True)
        with open(os.path.join(reid.CACHE, f"emb_gt_train_{kind}.pkl"), "wb") as fh:
            pickle.dump(out, fh)


def pairs(seq, gt, k):
    """Consecutive processed frames (f-k, f): yields (ids0, boxes0, ids1, boxes1, f, f-k)."""
    for f in gt:
        if f - k in gt and len(gt[f][0]) and len(gt[f - k][0]):
            yield (*gt[f - k], *gt[f], f, f - k)


# -------------------------------------------------------------------------------------------- calibrate
def calibrate():
    res = {}
    for kind in reid.EMBEDDERS:
        with open(os.path.join(reid.CACHE, f"emb_gt_train_{kind}.pkl"), "rb") as fh:
            emb = pickle.load(fh)
        same, near_diff = [], []
        for seq in mot.SEQS:
            gt = gt_boxes(seq, "train")
            k = max(1, round(mot.seq_info(seq)["fps"]))          # 1 s apart
            for i0, b0, i1, b1, f, f0 in pairs(seq, gt, k):
                d = 1 - emb[seq][f] @ emb[seq][f0].T              # (n1, n0)
                c1, c0 = (b1[:, :2] + b1[:, 2:]) / 2, (b0[:, :2] + b0[:, 2:]) / 2
                near = np.linalg.norm(c1[:, None] - c0[None], axis=2) <= (b0[:, 3] - b0[:, 1])[None]
                eq = i1[:, None] == i0[None]
                same += list(d[eq]); near_diff += list(d[~eq & near])
        same, near_diff = np.array(same), np.array(near_diff)
        thr = float(np.quantile(near_diff, FAR))
        res[kind] = {"threshold": thr, "same_person_1s_accept_%": 100 * float((same <= thr).mean()),
                     "nearby_other_accept_%": 100 * FAR, "n_same": int(len(same)), "n_nearby_other": int(len(near_diff))}
        print(kind, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[kind].items()}, flush=True)
    with open(os.path.join(OUT, "calibration_train_half.json"), "w") as fh:
        json.dump(res, fh, indent=2)


# ------------------------------------------------------------------------------------------------- cues
def cues():
    """Rank-1 identity linking between consecutive processed frames, per cue, on val-half GT."""
    embs = {kind: reid.load_emb("oracle_gt", kind) for kind in reid.EMBEDDERS}
    out = {}
    for hz in RATES:
        hit = {c: 0 for c in ["motion", "motion + CMC", *reid.EMBEDDERS, "motion + CMC + ReID (sum)",
                              "CMC wrong, ReID right", "CMC right, ReID wrong"]}
        n = 0
        for seq in mot.SEQS:
            k = stride(seq, hz)
            warps = cmc.load(seq)
            gt = gt_boxes(seq, "val")
            for i0, b0, i1, b1, f, f0 in pairs(seq, gt, k):
                prev = gt.get(f0 - k)
                # constant-velocity prediction from f0-k (when the person was visible then)
                pred = b0.copy()
                if prev is not None:
                    pid = {p: j for j, p in enumerate(prev[0])}
                    for j, p in enumerate(i0):
                        if p in pid:
                            pred[j] = 2 * b0[j] - prev[1][pid[p]]
                A = warps.get((f, k))
                pred_c = pred.copy()
                if A is not None:                               # move f0 boxes (and velocity) into frame f
                    w = lambda bb: np.c_[bb[:, :2] @ A[:, :2].T + A[:, 2], bb[:, 2:] @ A[:, :2].T + A[:, 2]]
                    pred_c = w(b0)
                    if prev is not None and (f0, k) in warps:
                        A0 = warps[(f0, k)]
                        pp = {p: j for j, p in enumerate(prev[0])}
                        for j, p in enumerate(i0):
                            if p in pp:
                                bp = prev[1][pp[p]]
                                bp0 = np.r_[A0[:, :2] @ bp[:2] + A0[:, 2], A0[:, :2] @ bp[2:] + A0[:, 2]]
                                pred_c[j] = w((2 * b0[j] - bp0)[None])[0]
                common = np.isin(i1, i0)
                if not common.any():
                    continue
                truth = np.array([np.where(i0 == p)[0][0] for p in i1[common]])
                n += int(common.sum())
                ok = {}
                for name, P in (("motion", pred), ("motion + CMC", pred_c)):
                    s = iou_matrix(b1[common], P)
                    ok[name] = (s.argmax(1) == truth) & (s.max(1) > 0)
                for kind, e in embs.items():
                    s = e[seq][f][common] @ e[seq][f0].T
                    ok[kind] = s.argmax(1) == truth
                s = iou_matrix(b1[common], pred_c) + embs["reid"][seq][f][common] @ embs["reid"][seq][f0].T
                ok["motion + CMC + ReID (sum)"] = s.argmax(1) == truth
                ok["CMC wrong, ReID right"] = ~ok["motion + CMC"] & ok["reid"]
                ok["CMC right, ReID wrong"] = ok["motion + CMC"] & ~ok["reid"]
                for c, v in ok.items():
                    hit[c] += int(v.sum())
        out[hz] = {c: 100 * v / n for c, v in hit.items()}
        out[hz]["pairs"] = n
        print(hz, "Hz", {c: round(v, 1) for c, v in out[hz].items()}, flush=True)
    with open(os.path.join(OUT, "cue_linking_val_half.json"), "w") as fh:
        json.dump(out, fh, indent=2)


def why(hz=3):
    """Appearance rank-1 at `hz`, broken down by the later box's visibility and height (val-half GT)."""
    embs = {kind: reid.load_emb("oracle_gt", kind) for kind in reid.EMBEDDERS}
    rows = []
    for seq in mot.SEQS:
        g = mot.load_gt(seq)
        g = g[(g[:, 7] == mot.PEDESTRIAN) & (g[:, 6] != 0)]
        k = stride(seq, hz)
        by = {f: g[g[:, 0] == f] for f in mot.half_frames(seq, "val")}
        for f in by:
            if f - k not in by or not len(by[f]) or not len(by[f - k]):
                continue
            i0 = by[f - k][:, 1]
            for a, p in enumerate(by[f][:, 1]):
                if p in i0:
                    t = np.where(i0 == p)[0][0]
                    rows.append([by[f][a, 8], by[f][a, 5]] +
                                [float(np.argmax(embs[e][seq][f - k] @ embs[e][seq][f][a]) == t) for e in reid.EMBEDDERS])
    R = np.array(rows)
    out = {"rate_hz": hz, "visibility": {}, "height_px": {}}
    for key, col, bins in (("visibility", 0, [0, .25, .5, .75, 1.01]), ("height_px", 1, [0, 80, 150, 300, 1e4])):
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (R[:, col] >= lo) & (R[:, col] < hi)
            out[key][f"{lo:g}-{min(hi, 1) if key == 'visibility' else hi:g}"] = {
                "share_%": 100 * float(m.mean()), **{e: 100 * float(R[m, 2 + i].mean()) for i, e in enumerate(reid.EMBEDDERS)}}
    print(json.dumps(out, indent=1))
    with open(os.path.join(OUT, "appearance_by_visibility_3hz.json"), "w") as fh:
        json.dump(out, fh, indent=2)


# ------------------------------------------------------------------------------------------------ sweep
VARIANTS = {   # name: (appearance embedder or None, use CMC, memory in seconds)
    "CMC (motion only)": (None, True, 1.0),
    "CMC + ImageNet": ("imagenet", True, 1.0),
    "CMC + DINOv2": ("dinov2", True, 1.0),
    "CMC + MOT17 ReID": ("reid", True, 1.0),
    "MOT17 ReID, no CMC": ("reid", False, 1.0),
    "CMC + MOT17 ReID, 3 s memory": ("reid", True, 3.0),
}


def run_rate(dets, embs, name, hz, thr, seqs=mot.SEQS):
    kind, use_cmc, memory = VARIANTS[name]
    accs, names = [], []
    for seq in seqs:
        fps, frames = mot.seq_info(seq)["fps"], list(mot.half_frames(seq, "val"))
        k = stride(seq, hz)
        eff = fps / k
        warps = cmc.load(seq) if use_cmc else {}
        for off in range(k):
            sub = frames[off::k]
            trk = Tracker(max_age=max(1, math.ceil(MAX_AGE / 30 * eff * memory)),
                          min_hits=max(1, math.ceil(MIN_HITS / 30 * eff)), iou_thr=0.3, det_thr=0.5, low_thr=0.1,
                          new_track_thr=0.6, byte=True, app_thr=thr[kind]["threshold"] if kind else None)
            res = {}
            for i, f in enumerate(sub):
                b, s = dets[seq].get(f, (np.zeros((0, 4)), np.zeros(0)))
                fe = embs[kind][seq][f] if kind else None
                if fe is not None and fe.shape[0] and fe.shape[1] == 1:   # no confident box in this frame
                    fe = None
                res[f] = trk.update(b, s, warp=warps[(f, k)] if use_cmc and i else None, feats=fe)
            accs.append(accumulate(seq, res, sub)); names.append(f"{seq}@{off}")
    o = summarize(accs, names).loc["OVERALL"]
    n_gt = sum(int(a.events["Type"].isin(["MATCH", "SWITCH", "MISS"]).sum()) for a in accs)
    return {"MOTA": 100 * o.mota, "IDF1": 100 * o.idf1, "IDSW": int(o.num_switches), "FP": int(o.num_false_positives),
            "FN": int(o.num_misses), "Rec": 100 * o.recall, "Prec": 100 * o.precision, "GT": n_gt,
            "IDSW/1k": 1000 * int(o.num_switches) / max(1, n_gt)}


def sweep(det_names):
    thr = json.load(open(os.path.join(OUT, "calibration_train_half.json")))
    path = os.path.join(OUT, "sweep_val_half.json")
    table = json.load(open(path)) if os.path.exists(path) else {}
    split = {"static camera": ["MOT17-02", "MOT17-04", "MOT17-09"],
             "moving camera": ["MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"]}
    for dk in det_names:
        dets = oracle_dets() if dk == "oracle_gt" else load_dets(dk)
        embs = {kind: reid.load_emb(dk, kind) for kind in reid.EMBEDDERS}
        table[dk] = {}
        for name in VARIANTS:
            table[dk][name] = {}
            for hz in RATES:
                r = run_rate(dets, embs, name, hz, thr)
                table[dk][name][hz] = r
                print(dk, "|", name, "|", hz, "Hz", {k: round(v, 1) for k, v in r.items()}, flush=True)
        table[dk]["_by_camera"] = {g: {name: {hz: run_rate(dets, embs, name, hz, thr, seqs=s) for hz in RATES}
                                       for name in ("CMC (motion only)", "CMC + MOT17 ReID", "MOT17 ReID, no CMC")}
                                   for g, s in split.items()}
        with open(path, "w") as fh:
            json.dump(table, fh, indent=2)


# ------------------------------------------------------------------------------------------------- plot
def plot(dk="yolo11s_mot17"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cue = json.load(open(os.path.join(OUT, "cue_linking_val_half.json")))
    table = json.load(open(os.path.join(OUT, "sweep_val_half.json")))
    base = json.load(open(os.path.join(ROOT, "results", "framerate", "sweep_val_half.json")))
    x = RATES
    col = {"motion": "#9aa0a6", "motion + CMC": "#5f6368", "imagenet": "#e8a33d", "dinov2": "#3b82c4",
           "reid": "#d1495b", "motion + CMC + ReID (sum)": "#7a1f3d"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))
    a = ax[0]
    for c in ["motion", "motion + CMC", *reid.EMBEDDERS, "motion + CMC + ReID (sum)"]:
        a.plot(x, [cue[str(h)][c] for h in x], marker="o", ms=3, c=col[c],
               ls="-." if c.endswith("(sum)") else "--" if c.startswith("motion") else "-", label=LABEL.get(c, c))
    a.set_title("(a) GT: right person ranked first\nbetween processed frames", fontsize=10)
    a.set_ylabel("% of people"); a.legend(fontsize=8)
    style = {"CMC (motion only)": ("#5f6368", "-"), "CMC + ImageNet": (col["imagenet"], "-"),
             "CMC + DINOv2": (col["dinov2"], "-"), "CMC + MOT17 ReID": (col["reid"], "-"),
             "MOT17 ReID, no CMC": (col["reid"], ":"), "CMC + MOT17 ReID, 3 s memory": ("#7a1f3d", "--")}
    for a, d, t in ((ax[1], dk, "(b) IDF1, fine-tuned YOLO11s"), (ax[2], "oracle_gt", "(c) IDF1, oracle detector")):
        a.plot(x, [base[d]["SORT+BYTE"][str(h)]["IDF1"] for h in x], "o-", ms=3, c="#bdbdbd", label="stock SORT+BYTE")
        for name, (c, ls) in style.items():
            a.plot(x, [table[d][name][str(h)]["IDF1"] for h in x], marker="o", ms=3, c=c, ls=ls, label=name)
        a.set_title(t, fontsize=10)
    ax[1].legend(fontsize=7.5)
    for a in ax:
        a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off(); a.invert_xaxis()
        a.set_xlabel("tracker update rate (Hz)"); a.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(OUT, "appearance_vs_motion.png")
    fig.savefig(p, dpi=130)
    print("saved", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed_train", "calibrate", "cues", "why", "sweep", "plot"])
    ap.add_argument("--dets", nargs="+", default=["yolo11s_mot17", "public_frcnn", "oracle_gt"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"embed_train": embed_train, "calibrate": calibrate, "cues": cues, "why": why, "plot": plot,
     "sweep": lambda: sweep(a.dets)}[a.cmd]()


if __name__ == "__main__":
    main()
