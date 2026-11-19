"""Run a YOLO11 person detector on the MOT17 val-half frames and cache the detections.

    python src/detect.py --weights yolo11s.pt --name yolo11s_coco
    python src/detect.py --weights runs/mot17_yolo11s/weights/best.pt --name yolo11s_mot17

Detections are stored with a low confidence floor (0.05) so that the BYTE second round can use them.
"""
import argparse
import os
import pickle

import numpy as np
from ultralytics import YOLO

import mot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=1088)
    ap.add_argument("--half", default="val")
    ap.add_argument("--person_class", type=int, default=0)
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = {}
    for seq in mot.SEQS:
        frames = list(mot.half_frames(seq, args.half))
        dets = {}
        for k in range(0, len(frames), 2):
            chunk = frames[k:k + 2]
            res = model.predict([mot.frame_path(seq, f) for f in chunk], imgsz=args.imgsz, conf=0.05,
                                iou=0.7, classes=[args.person_class], verbose=False, half=True)
            for f, r in zip(chunk, res):
                dets[f] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
        out[seq] = dets
        n = sum(len(v[1]) for v in dets.values())
        print(f"{seq}: {len(frames)} frames, {n} detections", flush=True)
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    with open(os.path.join(ROOT, "cache", f"dets_{args.name}.pkl"), "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
