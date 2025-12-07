# utils.py
"""
Camera capture helpers for IP cameras or local video files
"""

import cv2
import threading
import time

class VideoStreamAsync:
    """
    Simple threaded video capture for IP streams or files.
    """
    def __init__(self, src=0):
        self.src = src
        self.capture = cv2.VideoCapture(self.src)
        self.ret, self.frame = self.capture.read()
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                # small sleep to avoid busy loop if connection broken
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.capture.release()
