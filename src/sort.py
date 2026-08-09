import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanTracker:
    """Kalman Filter برای tracking یک شی"""
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R = np.eye(4) * 10
        self.kf.P = np.eye(7) * 1000
        self.kf.Q = np.eye(7) * 0.01

        x, y, w, h = bbox
        self.kf.x = np.array([[x], [y], [w], [h], [0], [0], [0]])
        self.time_since_update = 0
        self.id = None

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].flatten()

    def update(self, bbox):
        x, y, w, h = bbox
        z = np.array([[x], [y], [w], [h]])
        self.kf.update(z)
        self.time_since_update = 0


class SORT:
    """Simple Online and Realtime Tracking"""
    def __init__(self, max_age=30, min_hits=3):
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections):
        self.frame_count += 1

        # Predict
        predictions = []
        for tracker in self.trackers:
            pred = tracker.predict()
            predictions.append(pred)

        # Association (Hungarian algorithm)
        if len(detections) > 0 and len(predictions) > 0:
            cost_matrix = np.zeros((len(detections), len(self.trackers)))
            for i, det in enumerate(detections):
                for j, pred in enumerate(predictions):
                    cost_matrix[i, j] = np.linalg.norm(det - pred)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 50:
                    self.trackers[j].update(detections[i])

        # Create new trackers for unmatched detections
        matched_idx = set(col_ind) if len(detections) > 0 and len(predictions) > 0 else set()
        for i, det in enumerate(detections):
            if i not in matched_idx:
                tracker = KalmanTracker(det)
                tracker.id = self.next_id
                self.next_id += 1
                self.trackers.append(tracker)

        # Remove old trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]

        # Return active tracks
        active_tracks = []
        for tracker in self.trackers:
            if tracker.time_since_update < self.max_age:
                bbox = tracker.kf.x[:4].flatten()
                active_tracks.append((tracker.id, bbox))

        return active_tracks
