# tracker.py
"""
Lightweight multi-object tracker for MVP:
- Maintains persistent IDs by centroid distance + short-term OpenCV trackers (for smoothing)
- Not a production-grade ReID; replace with DeepSORT/ByteTrack later.
"""

import numpy as np
import cv2
from collections import OrderedDict
import time
import uuid

class TrackedObject:
    def __init__(self, bbox, frame, object_id=None):
        self.id = object_id or str(uuid.uuid4())[:8]
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.last_seen = time.time()
        self.missed = 0
        # Initialize a short-term OpenCV tracker for smoothness
        self.cv_tracker = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        self.cv_tracker.init(frame, (x1, y1, w, h))

    def update_from_bbox(self, bbox, frame):
        self.bbox = bbox
        self.last_seen = time.time()
        self.missed = 0
        try:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            self.cv_tracker = cv2.TrackerCSRT_create()
            self.cv_tracker.init(frame, (x1, y1, w, h))
        except Exception:
            pass

    def predict_with_cv(self, frame):
        try:
            ok, box = self.cv_tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in box]
                self.bbox = [x, y, x + w, y + h]
        except Exception:
            pass

class CentroidTracker:
    def __init__(self, max_missed=10, max_distance=80):
        self.objects = OrderedDict()  # id -> TrackedObject
        self.max_missed = max_missed
        self.max_distance = max_distance

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2)//2, (y1 + y2)//2)

    def update(self, detections, frame):
        """
        detections: list of dict with 'bbox'
        frame: current frame
        returns: list of tracked objects
        """
        if len(self.objects) == 0:
            for det in detections:
                obj = TrackedObject(det["bbox"], frame)
                self.objects[obj.id] = obj
            return list(self.objects.values())

        # Predict current bounding boxes via cv trackers
        for obj in self.objects.values():
            obj.predict_with_cv(frame)

        detected_centroids = [self._centroid(d["bbox"]) for d in detections]
        object_ids = list(self.objects.keys())
        object_centroids = [self._centroid(self.objects[oid].bbox) for oid in object_ids]

        if len(detections) == 0:
            # no detections: increment missed counters
            for oid in object_ids:
                self.objects[oid].missed += 1
            # remove stale
            for oid in list(self.objects.keys()):
                if self.objects[oid].missed > self.max_missed:
                    del self.objects[oid]
            return list(self.objects.values())

        # compute distance matrix
        D = np.linalg.norm(np.array(object_centroids)[:, None] - np.array(detected_centroids)[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            det = detections[c]
            self.objects[oid].update_from_bbox(det["bbox"], frame)
            assigned_rows.add(r)
            assigned_cols.add(c)

        # unassigned detections -> new objects
        for i, det in enumerate(detections):
            if i not in assigned_cols:
                new_obj = TrackedObject(det["bbox"], frame)
                self.objects[new_obj.id] = new_obj

        # unassigned existing objects -> missed increment
        for i, oid in enumerate(object_ids):
            if i not in assigned_rows:
                if oid in self.objects:
                    self.objects[oid].missed += 1
                    if self.objects[oid].missed > self.max_missed:
                        del self.objects[oid]
        return list(self.objects.values())
