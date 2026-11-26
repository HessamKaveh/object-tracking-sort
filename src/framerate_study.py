"""How slow can the perception loop get before SORT breaks? A processing-rate stress test on MOT17.

On a robot the detector rarely runs at the camera rate: it shares a GPU with other networks, runs on an
embedded board, or the control loop only asks for people a few times per second. This script keeps the
detections exactly as they are and only changes *how often the tracker sees them*: each sequence is
processed every k-th frame, where k = round(camera fps / target Hz), and scored on the processed frames
only (as in the low-frame-rate ablations of ByteTrack). Every phase offset 0..k-1 is run and the CLEAR-MOT
counts are pooled, so each target rate uses every frame once.

Three things are measured:

1. ``ceiling``: a detector-free diagnostic on ground truth. For every pedestrian visible in two consecutive
   processed frames, what fraction of the pairs have IoU < 0.3 between the box at t+k and (a) the box at t
   (a stationary prediction) or (b) a constant-velocity extrapolation from t-k and t? Those pairs cannot be
   associated by an IoU-gated tracker even with a perfect detector.
2. ``sweep``: SORT / SORT+BYTE at 30, 15, 10, 6, 3 and 2 Hz with several association variants:
   frame-count life-cycle parameters (stock SORT) versus time-aware ones (confirm within 0.1 s, delete after
   1 s), and the IoU gate versus GIoU / DIoU similarities that still rank boxes that no longer overlap.
3. ``tune``: thresholds of the GIoU / DIoU / low-IoU variants are chosen on the *train* half (public FRCNN
   detections) and then frozen for the val-half evaluation, so the val numbers are not tuned on.

    python src/framerate_study.py ceiling
    python src/framerate_study.py tune
    python src/framerate_study.py sweep --dets public_frcnn yolo11s_mot17
    python src/framerate_study.py plot
"""
import argparse
import json
import math
import os

import numpy as np

import cmc
import mot
from evaluate import accumulate, summarize
from run_benchmark import ROOT, load_dets
from sort import Tracker, iou_matrix

RATES = [30, 15, 10, 6, 3, 2]          # target processing rates (Hz)
OUT = os.path.join(ROOT, "results", "framerate")

# reference life-cycle at 30 Hz, from the main benchmark
MAX_AGE, MIN_HITS = 30, 3


def stride(seq, hz):
    return max(1, round(mot.seq_info(seq)["fps"] / hz))


def variant(name, k, fps, thr=None):
    """(Tracker kwargs, use camera-motion compensation) for an association variant at stride k."""
    thr = thr or {}
    eff = fps / k
    # stock SORT counts the frames it is *called* on; time-aware keeps the 30 Hz settings in seconds
    # (confirm within 0.1 s, delete after 1 s without a match)
    frames = dict(max_age=MAX_AGE, min_hits=MIN_HITS)
    timed = dict(max_age=max(1, math.ceil(MAX_AGE / 30 * eff)), min_hits=max(1, math.ceil(MIN_HITS / 30 * eff)))
    byte = dict(det_thr=0.5, low_thr=0.1, new_track_thr=0.6, byte=True)
    giou = dict(sim="giou", iou_thr=thr.get("giou", -0.3), byte_thr=thr.get("giou_byte", 0.2))
    table = {
        "SORT": (dict(**frames, iou_thr=0.3, det_thr=0.5), False),
        "SORT+BYTE": (dict(**frames, iou_thr=0.3, **byte), False),
        "+ time-aware": (dict(**timed, iou_thr=0.3, **byte), False),
        "+ time-aware + low IoU": (dict(**timed, iou_thr=thr.get("iou", 0.05), byte_thr=thr.get("iou_byte", 0.5),
                                        **byte), False),
        "+ time-aware + DIoU": (dict(**timed, sim="diou", iou_thr=thr.get("diou", 0.1),
                                     byte_thr=thr.get("diou_byte", 0.3), **byte), False),
        "+ time-aware + GIoU": (dict(**timed, **giou, **byte), False),
        "+ time-aware + CMC": (dict(**timed, iou_thr=0.3, **byte), True),
        "+ time-aware + CMC + GIoU": (dict(**timed, **giou, **byte), True),
    }
    return table[name]


VARIANTS = ["SORT", "SORT+BYTE", "+ time-aware", "+ time-aware + low IoU", "+ time-aware + DIoU",
            "+ time-aware + GIoU", "+ time-aware + CMC", "+ time-aware + CMC + GIoU"]
TUNED = {"iou": "+ time-aware + low IoU", "giou": "+ time-aware + GIoU", "diou": "+ time-aware + DIoU"}


def run_rate(dets, name, hz, half="val", thr=None, seqs=mot.SEQS):
    """Pool every phase offset of every sequence at one target rate -> overall + per-sequence metrics."""
    accs, names = [], []
    for seq in seqs:
        fps, frames = mot.seq_info(seq)["fps"], list(mot.half_frames(seq, half))
        k = stride(seq, hz)
        for off in range(k):
            sub = frames[off::k]
            kw, use_cmc = variant(name, k, fps, thr)
            warps = cmc.load(seq) if use_cmc else {}
            if use_cmc and not warps:
                raise FileNotFoundError(f"no CMC cache for {seq}: run python src/cmc.py first")
            trk = Tracker(**kw)
            res = {}
            for i, f in enumerate(sub):
                b, s = dets[seq].get(f, (np.zeros((0, 4)), np.zeros(0)))
                res[f] = trk.update(b, s, warp=warps[(f, k)] if use_cmc and i else None)
            accs.append(accumulate(seq, res, sub)); names.append(f"{seq}@{off}")
    summ = summarize(accs, names)
    o = summ.loc["OVERALL"]
    row = {"MOTA": 100 * o.mota, "IDF1": 100 * o.idf1, "IDSW": int(o.num_switches),
           "FP": int(o.num_false_positives), "FN": int(o.num_misses), "Rec": 100 * o.recall,
           "Prec": 100 * o.precision}
    # IDSW per 1000 GT boxes makes rates with fewer evaluated frames comparable
    n_gt = sum(int(a.events["Type"].isin(["MATCH", "SWITCH", "MISS"]).sum()) for a in accs)
    row["GT"] = n_gt
    row["IDSW/1k"] = 1000 * row["IDSW"] / max(1, n_gt)
    return row


# --------------------------------------------------------------------------------------------- ceiling
def ceiling(half="val"):
    out = {}
    for hz in RATES:
        stat = cv = n = 0
        for seq in mot.SEQS:
            gt = mot.load_gt(seq)
            gt = gt[(gt[:, 7] == mot.PEDESTRIAN) & (gt[:, 6] != 0)]
            k = stride(seq, hz)
            frames = set(mot.half_frames(seq, half))
            by = {}
            for r in gt:
                if int(r[0]) in frames:
                    b = r[2:6].copy(); b[2:] += b[:2]
                    by[(int(r[0]), int(r[1]))] = b
            for (f, tid), b1 in by.items():
                b0, bm = by.get((f - k, tid)), by.get((f - 2 * k, tid))
                if b0 is None:
                    continue
                n += 1
                stat += iou_matrix(b0[None], b1[None])[0, 0] < 0.3
                pred = b0 + (b0 - bm) if bm is not None else b0
                cv += iou_matrix(pred[None], b1[None])[0, 0] < 0.3
        out[hz] = {"pairs": n, "stationary_IoU<0.3": 100 * stat / n, "const_velocity_IoU<0.3": 100 * cv / n}
        print(hz, "Hz", {k: round(v, 2) for k, v in out[hz].items()}, flush=True)
    return out


# ------------------------------------------------------------------------------------------------ tune
def tune():
    """Grid-search thresholds on the train half (public FRCNN), objective = mean IDF1 over 10, 6, 3, 2 Hz."""
    dets = load_dets("public_frcnn")
    grids = {"iou": (TUNED["iou"], [0.05, 0.1, 0.2], [0.2, 0.3, 0.5]),
             "giou": (TUNED["giou"], [-0.5, -0.3, -0.1, 0.1], [-0.2, 0.0, 0.2]),
             "diou": (TUNED["diou"], [-0.3, -0.2, -0.1, 0.1], [-0.1, 0.1, 0.3])}
    best, log = {}, []
    for key, (name, g1, g2) in grids.items():
        scores = {}
        for t1 in g1:
            for t2 in g2:
                thr = {key: t1, f"{key}_byte": t2}
                r = [run_rate(dets, name, hz, half="train", thr=thr) for hz in (10, 6, 3, 2)]
                scores[(t1, t2)] = float(np.mean([x["IDF1"] for x in r]))
                log.append({"variant": name, "thr": t1, "byte_thr": t2, "mean_IDF1": scores[(t1, t2)],
                            "mean_MOTA": float(np.mean([x["MOTA"] for x in r]))})
                print(name, t1, t2, round(scores[(t1, t2)], 2), flush=True)
        (t1, t2) = max(scores, key=scores.get)
        best[key], best[f"{key}_byte"] = t1, t2
    with open(os.path.join(OUT, "tuned_thresholds.json"), "w") as f:
        json.dump({"selected_on": "MOT17 train half, public FRCNN, mean IDF1 over 10/6/3/2 Hz",
                   "thresholds": best, "grid": log}, f, indent=2)
    print("best", best)


def oracle_dets(half="val"):
    """Ground-truth pedestrian boxes as detections (score 1): isolates association errors from detector errors."""
    out = {}
    for seq in mot.SEQS:
        g = mot.load_gt(seq)
        g = g[(g[:, 7] == mot.PEDESTRIAN) & (g[:, 6] != 0)]
        out[seq] = {}
        for f in mot.half_frames(seq, half):
            r = g[g[:, 0] == f]
            b = r[:, 2:6].copy(); b[:, 2:] += b[:, :2]
            out[seq][f] = (b, np.ones(len(b)))
    return out


# ----------------------------------------------------------------------------------------------- sweep
def sweep(det_names):
    with open(os.path.join(OUT, "tuned_thresholds.json")) as f:
        thr = json.load(f)["thresholds"]
    path = os.path.join(OUT, "sweep_val_half.json")
    table = json.load(open(path)) if os.path.exists(path) else {}
    for dk in det_names:
        dets = oracle_dets() if dk == "oracle_gt" else load_dets(dk)
        table[dk] = {}
        for name in VARIANTS:
            table[dk][name] = {}
            for hz in RATES:
                r = run_rate(dets, name, hz, thr=thr)
                table[dk][name][hz] = r
                print(dk, "|", name, "|", hz, "Hz", {k: round(v, 1) for k, v in r.items()}, flush=True)
        # camera-motion split for the two main variants
        split = {"static camera": ["MOT17-02", "MOT17-04", "MOT17-09"],
                 "moving camera": ["MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"]}
        table[dk]["_by_camera"] = {
            g: {name: {hz: run_rate(dets, name, hz, thr=thr, seqs=s) for hz in RATES}
                for name in ("SORT+BYTE", "+ time-aware", "+ time-aware + GIoU", "+ time-aware + CMC",
                             "+ time-aware + CMC + GIoU")}
            for g, s in split.items()}
        with open(path, "w") as f:
            json.dump(table, f, indent=2)


# ------------------------------------------------------------------------------------------------ plot
def plot(dk="yolo11s_mot17"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = json.load(open(os.path.join(OUT, "sweep_val_half.json")))[dk]
    ceil = json.load(open(os.path.join(OUT, "gt_ceiling.json")))
    style = {"SORT+BYTE": ("#5f6368", "-", "stock SORT+BYTE"),
             "+ time-aware": ("#e8a33d", "-", "+ time-aware life cycle"),
             "+ time-aware + GIoU": ("#3b82c4", "-", "+ time-aware + GIoU"),
             "+ time-aware + CMC": ("#d1495b", "-", "+ time-aware + camera-motion comp."),
             "+ time-aware + CMC + GIoU": ("#7a1f3d", "--", "+ time-aware + CMC + GIoU")}
    x = RATES
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    a = ax[0]
    a.plot(x, [ceil[str(h)]["stationary_IoU<0.3"] for h in x], "o--", c="#9aa0a6", label="stationary prediction")
    a.plot(x, [ceil[str(h)]["const_velocity_IoU<0.3"] for h in x], "o-", c="#5f6368", label="constant velocity")
    a.set_title("(a) GT: same person, IoU < 0.3\nbetween processed frames", fontsize=10)
    a.set_ylabel("% of consecutive pairs"); a.legend(fontsize=8)
    for a, m, t in ((ax[1], "IDF1", "(b) IDF1, all 7 sequences"), (ax[2], "MOTA", "(c) MOTA, all 7 sequences")):
        for name, (c, ls, lab) in style.items():
            a.plot(x, [table[name][str(h)][m] for h in x], marker="o", ms=3, c=c, ls=ls, label=lab)
        a.set_title(t, fontsize=10)
    ax[1].legend(fontsize=7.5)
    a = ax[3]
    for g, ls in (("static camera", ":"), ("moving camera", "-")):
        for name in ("SORT+BYTE", "+ time-aware + CMC"):
            c = style[name][0]
            a.plot(x, [table["_by_camera"][g][name][str(h)]["IDF1"] for h in x], marker="o", ms=3, c=c, ls=ls,
                   label=f"{g}: {'stock' if name == 'SORT+BYTE' else '+ time-aware + CMC'}")
    a.set_title("(d) IDF1, static vs moving camera", fontsize=10); a.legend(fontsize=7.5)
    for a in ax:
        a.set_xscale("log"); a.set_xticks(x); a.set_xticklabels(x); a.minorticks_off(); a.invert_xaxis()
        a.set_xlabel("tracker update rate (Hz)"); a.grid(alpha=.3)
    fig.tight_layout()
    path = os.path.join(OUT, "framerate_stress.png")
    fig.savefig(path, dpi=130)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ceiling", "tune", "sweep", "plot"])
    ap.add_argument("--dets", nargs="+", default=["public_frcnn"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.cmd == "ceiling":
        with open(os.path.join(OUT, "gt_ceiling.json"), "w") as f:
            json.dump(ceiling(), f, indent=2)
    elif args.cmd == "tune":
        tune()
    elif args.cmd == "sweep":
        sweep(args.dets)
    else:
        plot()


if __name__ == "__main__":
    main()
