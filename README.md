# Object Tracking with SORT

Multi-Object Tracker using Simple Online and Realtime Tracking (SORT) algorithm
combined with Faster R-CNN for detection.

**Data note:** when no input video is given, `src/dataset.py` creates a synthetic 10 s video of two moving coloured squares. It is a didactic implementation of the SORT pipeline, not an evaluated tracker.

## Pipeline
1. Detect objects in each frame with Faster R-CNN
2. Associate detections across frames using Hungarian algorithm
3. Track each object with Kalman Filter
4. Assign unique IDs to objects across the video

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/track.py
```

## Results

> On this synthetic video the COCO-trained Faster R-CNN detects no objects, so `results/tracked_output.mp4` shows no tracks. Run `src/track.py` on a real video (people, cars) to see SORT IDs.

## Author
Hessam Kaveh - Research Fellow, Italian Institute of Technology

## License

Released under the [MIT License](LICENSE).
