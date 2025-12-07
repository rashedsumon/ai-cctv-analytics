# anomaly_detector.py
"""
Simple anomaly detector for MVP.
- Flags unreported services if detection indicates a service event but POS/booking isn't logged
- Flags early unregistered checkouts (if tracked person leaves without sign-off)
- Fraud hooks: compares activity logs vs. financial logs via adapter functions (stubs)
"""

from typing import Dict, Any, List
import time

class AnomalyDetector:
    def __init__(self, pos_adapter=None, booking_adapter=None, thresholds=None):
        """
        pos_adapter: a callable that takes event and returns matching POS transactions (stub)
        booking_adapter: callable to check bookings
        thresholds: dict for rule thresholds
        """
        self.pos_adapter = pos_adapter
        self.booking_adapter = booking_adapter
        self.thresholds = thresholds or {"no_pos_time_window": 300, "min_service_duration": 30}

    def check_service_event(self, event: Dict[str, Any]) -> List[Dict]:
        """
        event: { 'type': 'service', 'service_type': 'haircut', 'obj_id': 'abc', 'ts': timestamp }
        returns list of anomalies
        """
        anomalies = []
        ts = event.get("ts", time.time())
        if self.pos_adapter:
            # check POS for matching transaction in a small window
            pos_matches = self.pos_adapter(event, window_seconds=self.thresholds["no_pos_time_window"])
            if not pos_matches:
                anomalies.append({"t": "unreported_service", "detail": event})
        else:
            # if no pos_adapter configured, flag as info for manual check
            anomalies.append({"t": "unreported_service_no_pos_adapter", "detail": event})
        return anomalies

    def check_early_checkout(self, session_info: Dict[str, Any]) -> List[Dict]:
        """
        session_info: { 'id': obj_id, 'start_ts':, 'end_ts':, 'registered_checkout': bool }
        """
        anomalies = []
        if not session_info.get("registered_checkout", False) and session_info.get("end_ts"):
            anomalies.append({"t": "unregistered_checkout", "detail": session_info})
        return anomalies

    # Example adapter stubs:
    @staticmethod
    def pos_adapter_stub(event, window_seconds=300):
        """
        Replace with actual logic: query POS / DB for a transaction matching person/time/service.
        Returns list of matches (empty means no matching POS transactions).
        """
        return []
