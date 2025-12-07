# analytics.py
"""
Aggregates counts, durations, heatmap grid and busiest-hour computations.
Simple in-memory aggregator; replace with DB/Timeseries in production.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
from datetime import datetime

class AnalyticsStore:
    def __init__(self, heatmap_bins=(32, 18), store_seconds=3600*24):
        # heatmap_bins: (width_cells, height_cells)
        self.heatmap_bins = heatmap_bins
        self.heatmap = np.zeros(heatmap_bins, dtype=np.int32)
        self.events = []  # append dict events
        self.active_sessions = {}  # id -> session start time
        # rolling aggregator for busiest hours (simple minute buckets)
        self.minute_buckets = defaultdict(int)
        self.store_seconds = store_seconds

    def record_detection(self, obj_id, bbox, frame_shape):
        # record presence for heatmap and counts
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        bx = int(cx / w * self.heatmap_bins[0])
        by = int(cy / h * self.heatmap_bins[1])
        bx = np.clip(bx, 0, self.heatmap_bins[0]-1)
        by = np.clip(by, 0, self.heatmap_bins[1]-1)
        self.heatmap[bx, by] += 1

        now = datetime.utcnow()
        minute_key = now.replace(second=0, microsecond=0)
        self.minute_buckets[minute_key] += 1
        # Clean old buckets
        self._cleanup_old()

    def event(self, event_type, payload):
        ev = {"ts": time.time(), "t": event_type, "p": payload}
        self.events.append(ev)

    def session_start(self, obj_id):
        if obj_id not in self.active_sessions:
            self.active_sessions[obj_id] = time.time()

    def session_end(self, obj_id):
        start = self.active_sessions.pop(obj_id, None)
        if start is not None:
            duration = time.time() - start
            self.event("session_end", {"id": obj_id, "duration": duration})

    def busiest_hours(self, top_n=6):
        # return the top minute buckets aggregated into hour buckets
        df = pd.Series(self.minute_buckets)
        if df.empty:
            return []
        # convert minute keys to hour (floor)
        df_hour = df.groupby(lambda t: t.replace(minute=0, second=0, microsecond=0)).sum()
        df_hour = df_hour.sort_values(ascending=False)
        return [(k.isoformat(), int(v)) for k, v in df_hour.head(top_n).items()]

    def get_heatmap(self):
        # return 2D heatmap array
        return self.heatmap.copy()

    def _cleanup_old(self):
        cutoff = time.time() - self.store_seconds
        # events cleanup
        self.events = [e for e in self.events if e["ts"] >= cutoff]
        # minute_buckets cleanup
        for k in list(self.minute_buckets.keys()):
            if k.timestamp() < cutoff:
                del self.minute_buckets[k]
