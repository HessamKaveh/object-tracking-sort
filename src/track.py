"""Track people in any video (or a MOT17 sequence) with YOLO11 + SORT/BYTE and render the result.

    python src/track.py --source my_video.mp4 --weights runs/mot17_yolo11s/weights/best.pt
    python src/track.py --mot_seq MOT17-09 --weights ... --out results/demo.mp4 --gif results/demo.gif
    python src/track.py --source robot_cam.mp4 --weights ... --stride 10 --cmc   # 3 Hz on a 30 fps camera

--stride k runs detection + tracking on every k-th frame (a slow perception loop); --cmc adds camera-motion
compensation and the life-cycle parameters are then kept in seconds (see src/framerate_study.py);
--reid adds appearance matching with a ReID checkpoint from src/reid.py (see src/appearance_study.py).
"""
import argparse
import colorsys
import os

import cv2
from ultralytics import YOLO

import cmc
from sort import Tracker


def color(tid):
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def frames_from(args):
    if args.mot_seq:
        import mot
        for f in mot.half_frames(args.mot_seq, "val"):
            yield cv2.imread(mot.frame_path(args.mot_seq, f))
    else:
        cap = cv2.VideoCapture(args.source)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--mot_seq")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--person_class", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1088)
    ap.add_argument("--no_byte", action="store_true")
    ap.add_argument("--out", default="results/tracked.mp4")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--max_frames", type=int, default=10**9)
    ap.add_argument("--stride", type=int, default=1, help="process every k-th frame")
    ap.add_argument("--cmc", action="store_true", help="camera-motion compensation (moving camera)")
    ap.add_argument("--reid", default=None, help="ReID checkpoint, e.g. weights/reid_r18_embeds_foldA.pt")
    ap.add_argument("--app_thr", type=float, default=0.324, help="cosine-distance threshold (calibrated)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    s = args.stride
    trk = Tracker(max_age=max(1, -(-30 // s)), min_hits=max(1, -(-3 // s)), iou_thr=0.3, det_thr=0.5,
                  low_thr=0.1, byte=not args.no_byte, new_track_thr=0.6,
                  app_thr=args.app_thr if args.reid else None)
    embedder = None
    if args.reid:
        import numpy as np
        import reid
        embedder = reid.Embedder("reid", path=args.reid)
    writer, gif_frames, prev = None, [], None
    for k, frame in enumerate(frames_from(args)):
        if k >= args.max_frames:
            break
        if k % s:
            continue
        r = model.predict(frame, imgsz=args.imgsz, conf=0.05, classes=[args.person_class], verbose=False)[0]
        boxes, scores = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        warp = None
        if args.cmc:
            if prev is not None:
                warp = cmc.estimate(prev[0], frame, prev[1])
            prev = (frame.copy(), boxes)
        feats = None
        if embedder is not None:
            keep = scores >= 0.5
            feats = np.zeros((len(scores), 512), np.float32)
            if keep.any():
                feats[keep] = embedder(np.stack([reid.crop(frame, b) for b in boxes[keep]]))
        tracks = trk.update(boxes, scores, warp=warp, feats=feats)
        for x1, y1, x2, y2, tid, _ in tracks:
            c = color(int(tid))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
            cv2.putText(frame, str(int(tid)), (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        if writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        writer.write(frame)
        if args.gif and (k // s) % max(1, 3 // s) == 0:
            gif_frames.append(cv2.cvtColor(cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1]))),
                                           cv2.COLOR_BGR2RGB))
    writer.release()
    if args.gif:
        from PIL import Image
        ims = [Image.fromarray(f) for f in gif_frames]
        ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=120, loop=0, optimize=True)


if __name__ == "__main__":
    main()
