# detector.py
"""
Simple detector wrapper using ultralytics YOLO.
Provides detect(frame) -> list of detections with bbox, score, class_id.
"""

from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt", device="cpu", conf=0.35):
        """
        model_name: "yolov8n.pt" (nano) for MVP. Replace with custom weights for production.
        device: "cpu" or "cuda"
        """
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        # send to device if needed (ultralytics handles device selection via environment)
        # self.model.to(device)

    def detect(self, frame):
        """
        Input: BGR frame (OpenCV)
        Output: list of detections: dict { 'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int, 'label': str }
        """
        # convert BGR->RGB
        img = frame[..., ::-1]
        results = self.model.predict(source=[img], imgsz=640, conf=self.conf, verbose=False)
        detections = []
        if len(results) == 0:
            return detections
        r = results[0]
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(conf),
                "class_id": int(cls),
                "label": self.model.names[int(cls)] if hasattr(self.model, "names") else str(int(cls))
            })
        return detections
