"""Run every detector x tracker combination on MOT17 val-half and write the results table.

Detectors : MOT17 public FRCNN detections, YOLO11s (COCO weights), YOLO11s fine-tuned on MOT17 train-half
Trackers  : SORT (Bewley 2016) and SORT + BYTE two-stage association (Zhang 2022)
"""
import json
import os
import pickle

import numpy as np

import mot
from evaluate import accumulate, summarize
from sort import Tracker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dets(name):
    if name == "public_frcnn":
        return {s: mot.load_public_dets(s) for s in mot.SEQS}
    with open(os.path.join(ROOT, "cache", f"dets_{name}.pkl"), "rb") as f:
        return pickle.load(f)


def run(dets, cfg, seq):
    trk = Tracker(**cfg)
    res, rows = {}, []
    for f in mot.half_frames(seq, "val"):
        b, s = dets[seq].get(f, (np.zeros((0, 4)), np.zeros(0)))
        o = trk.update(b, s)
        res[f] = o
        rows += [(f, r[4], *r[:4], r[5]) for r in o]
    return res, rows


def main():
    detectors = {"public_frcnn": "MOT17 public FRCNN",
                 "yolo11s_coco": "YOLO11s (COCO)",
                 "yolo11s_mot17": "YOLO11s fine-tuned on MOT17"}
    trackers = {
        "SORT": dict(max_age=30, min_hits=3, iou_thr=0.3, det_thr=0.5),
        "SORT + BYTE": dict(max_age=30, min_hits=3, iou_thr=0.3, det_thr=0.5, low_thr=0.1, byte=True,
                            new_track_thr=0.6),
    }
    table = {}
    for dk, dname in detectors.items():
        try:
            dets = load_dets(dk)
        except FileNotFoundError:
            print("skip", dk); continue
        for tname, cfg in trackers.items():
            accs = []
            for seq in mot.SEQS:
                res, rows = run(dets, cfg, seq)
                mot.write_mot(os.path.join(ROOT, "results", "tracks", dk, tname.replace(" + ", "_"),
                                           f"{seq}.txt"), rows)
                accs.append(accumulate(seq, res, mot.half_frames(seq, "val")))
            summ = summarize(accs, mot.SEQS)
            o = summ.loc["OVERALL"]
            key = f"{dname} | {tname}"
            table[key] = {"MOTA": 100 * o.mota, "IDF1": 100 * o.idf1, "IDSW": int(o.num_switches),
                          "FP": int(o.num_false_positives), "FN": int(o.num_misses),
                          "MT": int(o.mostly_tracked), "ML": int(o.mostly_lost),
                          "Prec": 100 * o.precision, "Rec": 100 * o.recall}
            if dk == "yolo11s_mot17" and tname == "SORT + BYTE":
                per_seq = {s: {"MOTA": 100 * summ.loc[s].mota, "IDF1": 100 * summ.loc[s].idf1,
                               "IDSW": int(summ.loc[s].num_switches)} for s in mot.SEQS}
                with open(os.path.join(ROOT, "results", "per_sequence_best.json"), "w") as f:
                    json.dump(per_seq, f, indent=2)
            print(key, {k: round(v, 1) for k, v in table[key].items()}, flush=True)
    with open(os.path.join(ROOT, "results", "mot17_val_half.json"), "w") as f:
        json.dump(table, f, indent=2)


if __name__ == "__main__":
    main()
