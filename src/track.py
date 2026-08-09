import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataset import download_sample_video
from detector import ObjectDetector
from sort import SORT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def track():
    print("Downloading/creating sample video...")
    video_path = download_sample_video()

    print("Initializing detector and tracker...")
    detector = ObjectDetector()
    sort_tracker = SORT(max_age=30, min_hits=1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(RESULTS_DIR, "tracked_output.mp4"), fourcc, fps, (width, height))

    frame_count = 0
    colors = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Detect objects
        boxes, scores = detector.detect(frame)

        # Update tracker
        tracks = sort_tracker.update(boxes)

        # Draw results
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            w, h = x2 - x1, y2 - y1
            x1, y1 = int(x1 - w/2), int(y1 - h/2)
            x2, y2 = x1 + int(w), y1 + int(h)

            if track_id not in colors:
                colors[track_id] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

            color = colors[track_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved tracked video to {os.path.join(RESULTS_DIR, 'tracked_output.mp4')}")


if __name__ == "__main__":
    track()
