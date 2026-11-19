"""Fine-tune YOLO11s (COCO-pretrained) as a single-class person detector on MOT17 train-half.

Builds a YOLO-format dataset (symlinked images + labels) from the first half of every MOT17 train
sequence; a held-out slice (last 10 % of each train-half) is used for model selection, so the
val-half used for tracking evaluation is never seen.
Ground-truth boxes of class pedestrian (1), person on vehicle (2) and static person (7) become "person".
"""
import argparse
import os

import mot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "mot17_yolo")


def build():
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATA, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATA, "labels", split), exist_ok=True)
    for seq in mot.SEQS:
        info, gt = mot.seq_info(seq), mot.load_gt(seq)
        frames = list(mot.half_frames(seq, "train"))
        cut = int(len(frames) * 0.9)
        for k, f in enumerate(frames):
            split = "train" if k < cut else "val"
            name = f"{seq}_{f:06d}"
            dst = os.path.join(DATA, "images", split, name + ".jpg")
            if not os.path.exists(dst):
                os.symlink(mot.frame_path(seq, f), dst)
            g = gt[(gt[:, 0] == f) & ((gt[:, 7] == 1) | (gt[:, 7] == 2) | (gt[:, 7] == 7))]
            with open(os.path.join(DATA, "labels", split, name + ".txt"), "w") as fh:
                for _, _, x, y, w, h, *_ in g:
                    x1, y1 = max(x, 0), max(y, 0)
                    x2, y2 = min(x + w, info["w"]), min(y + h, info["h"])
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    fh.write(f"0 {(x1 + x2) / 2 / info['w']:.6f} {(y1 + y2) / 2 / info['h']:.6f} "
                             f"{(x2 - x1) / info['w']:.6f} {(y2 - y1) / info['h']:.6f}\n")
    yaml = os.path.join(DATA, "mot17.yaml")
    with open(yaml, "w") as fh:
        fh.write(f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1088)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    from ultralytics import YOLO

    yaml = build()
    model = YOLO("yolo11s.pt")
    model.train(data=yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=6,
                project=os.path.join(ROOT, "runs"), name="mot17_yolo11s", exist_ok=True,
                close_mosaic=5, amp=True, seed=0, plots=True)


if __name__ == "__main__":
    main()
