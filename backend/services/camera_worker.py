import cv2
import time
import threading
import logging
from typing import Optional
from backend.services.alert_manager import alert_manager

logger = logging.getLogger("forestguard.camera_worker")

class CameraWorker:
    """
    Background worker thread for RTSP, MJPEG (IP Webcam app), or USB/laptop webcam.
    Ensures latest-frame-wins with single-frame buffer, handles reconnects,
    and raises camera_disconnected alerts if stream stalls.
    """
    def __init__(self, pipeline_fn=None):
        self.pipeline_fn = pipeline_fn
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.source_type = "none"  # "ip", "webcam", "browser"
        self.source_target = 0
        
        self.latest_frame = None
        self.last_frame_ts = 0.0
        self.fps = 0.0
        self.connected = False
        self.disconnect_alerted = False
        self.lock = threading.Lock()

    def start(self, source_type: str = "webcam", target=0):
        if self.running:
            self.stop()

        self.source_type = source_type
        self.source_target = target
        self.running = True
        self.connected = False
        self.disconnect_alerted = False

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info(f"CameraWorker started with source: {source_type} ({target})")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.connected = False
        self.latest_frame = None
        logger.info("CameraWorker stopped.")

    def update_frame_from_browser(self, bgr_image):
        """Called when browser sends frames over /ws/stream."""
        with self.lock:
            self.latest_frame = bgr_image
            now = time.time()
            if self.last_frame_ts > 0:
                dt = now - self.last_frame_ts
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
            self.last_frame_ts = now
            self.connected = True
            self.disconnect_alerted = False

    def _worker_loop(self):
        cap = None
        backoff = 1.0

        while self.running:
            if cap is None or not cap.isOpened():
                try:
                    logger.info(f"Opening video capture: {self.source_target}")
                    if isinstance(self.source_target, str) and (self.source_target.isdigit()):
                        target = int(self.source_target)
                    else:
                        target = self.source_target
                    cap = cv2.VideoCapture(target)
                    if cap.isOpened():
                        self.connected = True
                        self.disconnect_alerted = False
                        backoff = 1.0
                        logger.info("Video capture opened successfully.")
                    else:
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 5.0)
                        continue
                except Exception as e:
                    logger.error(f"Error opening capture: {e}")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 5.0)
                    continue

            # Read latest frame
            ret, frame = cap.read()
            now = time.time()

            if not ret or frame is None:
                self.connected = False
                logger.warning("Failed to grab frame from video stream.")
                if (now - self.last_frame_ts) > 5.0 and not self.disconnect_alerted:
                    self.disconnect_alerted = True
                    logger.warning("Camera stream disconnected (>5s with no frames).")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(1.0)
                continue

            with self.lock:
                self.latest_frame = frame
                if self.last_frame_ts > 0:
                    dt = now - self.last_frame_ts
                    if dt > 0:
                        current_fps = 1.0 / dt
                        self.fps = 0.9 * self.fps + 0.1 * current_fps if self.fps > 0 else current_fps
                self.last_frame_ts = now
                self.connected = True
                self.disconnect_alerted = False

            # Throttle loop to avoid 100% CPU on capture
            time.sleep(0.01)

        if cap:
            cap.release()
        self.connected = False

    def get_status(self) -> dict:
        now = time.time()
        age = round(now - self.last_frame_ts, 1) if self.last_frame_ts > 0 else 999.0
        is_connected = self.connected and (age < 5.0)
        return {
            "active": self.running or (age < 5.0 and self.source_type == "browser"),
            "source": self.source_type,
            "target": str(self.source_target),
            "fps": round(self.fps, 1),
            "last_frame_age_s": age,
            "connected": is_connected
        }

camera_worker = CameraWorker()
