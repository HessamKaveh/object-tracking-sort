# Object Tracking with SORT

Multi-Object Tracker using Simple Online and Realtime Tracking (SORT) algorithm
combined with Faster R-CNN for detection.

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
- `results/tracked_output.mp4` — video with tracked objects and IDs

## Author
Hessam Kaveh - Research Fellow, Italian Institute of Technology
