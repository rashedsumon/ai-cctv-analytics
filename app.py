# app.py
"""
Streamlit dashboard & real-time monitor for AI CCTV Analytics MVP.
Entrypoint for deployment on Streamlit Cloud.
"""

import streamlit as st
import cv2
import numpy as np
from detector import YOLODetector
from tracker import CentroidTracker
from analytics import AnalyticsStore
from anomaly_detector import AnomalyDetector
from utils import VideoStreamAsync
import time
import base64
from matplotlib import pyplot as plt

st.set_page_config(layout="wide", page_title="AI CCTV Analytics MVP")

# ---------------------
# Sidebar: Settings
# ---------------------
st.sidebar.title("Settings")
camera_sources = st.sidebar.text_area("Camera sources (one per line): IP or video file paths", value="0")
conf_threshold = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35)
model_choice = st.sidebar.selectbox("Detector model (local/shared)", ["yolov8n.pt", "yolov8s.pt"])
start_button = st.sidebar.button("Start monitoring")

# ---------------------
# Initialize components (singletons per Streamlit run)
# ---------------------
@st.cache_resource(show_spinner=False)
def init_detector(model_name, conf):
    return YOLODetector(model_name=model_name, conf=conf)

@st.cache_resource(show_spinner=False)
def init_tracker():
    return CentroidTracker()

@st.cache_resource(show_spinner=False)
def init_analytics():
    return AnalyticsStore()

@st.cache_resource(show_spinner=False)
def init_anomaly_detector():
    return AnomalyDetector()

detector = init_detector(model_choice, conf_threshold)
tracker = init_tracker()
analytics = init_analytics()
anomaly_detector = init_anomaly_detector()

# ---------------------
# UI layout
# ---------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Live Streams")
    stream_placeholders = st.container()
with col2:
    st.header("Analytics")
    busiest_placeholder = st.empty()
    heatmap_placeholder = st.empty()
    anomalies_placeholder = st.empty()
    stats_placeholder = st.empty()

# ---------------------
# Start cameras
# ---------------------
if start_button:
    srcs = [s.strip() for s in camera_sources.splitlines() if s.strip()]
    if len(srcs) == 0:
        st.error("Add at least one camera source in sidebar (0 for webcam).")
    else:
        streams = []
        for src in srcs:
            vs = VideoStreamAsync(src if src != "0" else 0)
            vs.start()
            streams.append((src, vs))
        st.success(f"Started {len(streams)} stream(s).")

        # Prepare placeholders for each stream
        placeholders = {}
        for i, (src, vs) in enumerate(streams):
            ph = stream_placeholders.empty()
            placeholders[src] = ph

        # main loop (simple implementation: will run while Streamlit session alive)
        try:
            while True:
                # process each camera once per loop
                for src, vs in streams:
                    ret, frame = vs.read()
                    if not ret or frame is None:
                        continue
                    # Resize for display & speed
                    display_frame = cv2.resize(frame, (960, 540))
                    detections = detector.detect(display_frame)

                    # Update tracker
                    tracked = tracker.update(detections, display_frame)

                    # draw boxes + ids
                    vis = display_frame.copy()
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis, f'{det["label"]}:{det["score"]:.2f}', (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    for obj in tracked:
                        x1, y1, x2, y2 = obj.bbox
                        cv2.putText(vis, f'ID:{obj.id}', (x1, y2 + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                        # analytics
                        analytics.record_detection(obj.id, obj.bbox, vis.shape)
                        analytics.session_start(obj.id)

                    # Convert vis BGR->RGB for Streamlit
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    placeholders[src].image(vis_rgb, caption=f"Camera: {src}", use_column_width=True)

                # update right column analytics
                busiest = analytics.busiest_hours(top_n=6)
                busiest_placeholder.subheader("Busiest Hours (UTC)")
                if busiest:
                    busiest_placeholder.table([{"hour": h, "count": c} for h, c in busiest])
                else:
                    busiest_placeholder.write("No data yet.")

                heatmap = analytics.get_heatmap()
                heatmap_placeholder.subheader("Heatmap (grid)")
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(heatmap.T, origin='lower', interpolation='nearest')
                ax.set_title("Heatmap")
                ax.axis('off')
                heatmap_placeholder.pyplot(fig)

                # anomalies (placeholder; anomaly detector not fully wired to POS by default)
                anomalies_placeholder.subheader("Anomalies")
                anomalies = []  # placeholder retrieval
                if anomalies:
                    anomalies_placeholder.write(anomalies)
                else:
                    anomalies_placeholder.write("No anomalies detected (MVP).")

                # stats
                stats_placeholder.subheader("Realtime stats")
                stats_placeholder.write({
                    "tracked_objects": len(tracker.objects),
                    "total_events": len(analytics.events)
                })

                # throttle the loop
                time.sleep(0.2)

        except Exception as e:
            st.error(f"Monitoring stopped due to error: {e}")
            for _, vs in streams:
                vs.stop()
        finally:
            for _, vs in streams:
                vs.stop()
else:
    st.info("Configure camera sources and click **Start monitoring** in the sidebar.")
