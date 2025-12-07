# AI CCTV Analytics MVP

## Overview
MVP system to integrate existing CCTV/IP cameras to:
- Count & track clients
- Track services (placeholder/service classifiers)
- Generate busiest hours and heatmaps
- Flag anomalies (rule-based) and provide fraud-alert hooks
- Streamlit dashboard for live view + analytics

## Quickstart (local / dev)
1. Clone repo
2. Create a Python 3.11 virtualenv
3. `pip install -r requirements.txt`
4. If you want sample dataset downloaded, run:
   `python data_loader.py`
5. Run Streamlit:
   `streamlit run app.py`

## Deploy to Streamlit Cloud
- Ensure `requirements.txt` is present and `app.py` entrypoint.
- Add any model assets to `assets/models/` (or download at startup in `app.py`).

## Notes
- Detector uses `ultralytics` YOLO; change to your preferred detection model if required.
- Tracking uses a stable centroid-based tracker + short-term CV trackers (simple and robust for MVP).
- Anomaly detection is rule-based — replace with statistical/ML models in further iterations.
- Integration with POS / booking / payroll is supported via adapter functions in `anomaly_detector.py`.

## Files
See repository tree in project root.

## Contact
For upgrades: add person re-identification, multi-camera association, and DB-backed event logging (Postgres/Timescale).
