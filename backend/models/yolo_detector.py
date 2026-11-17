import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("forestguard.yolo")

class YOLODetector:
    """
    Ultralytics YOLO Fire/Smoke Object Detector (CPU inference).
    Gracefully handles missing weights or missing dependencies.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None
        self.available = False
        self.last_inference_ms = 0.0
        self.classes = {0: "fire", 1: "smoke"}
        
        self.load_model()

    def load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            logger.warning(f"YOLO model file not found at {self.model_path}. Running in unavailable mode.")
            self.available = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.available = True
            logger.info(f"YOLO detector loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.available = False

    def predict(
        self,
        bgr_image,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 416
    ) -> list[dict]:
        """
        Runs object detection on a BGR image.
        Returns:
            list of dict: [{"cls": "fire"|"smoke", "conf": float, "xyxy": [int, int, int, int]}]
        """
        if not self.available or self.model is None:
            return []

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=bgr_image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device="cpu",
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = results[0].names.get(cls_id, f"class_{cls_id}").lower()
                    
                    # Map to standard fire/smoke if named differently
                    if "fire" in cls_name:
                        cls_name = "fire"
                    elif "smoke" in cls_name:
                        cls_name = "smoke"
                    
                    confidence = float(box.conf[0].item())
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    
                    detections.append({
                        "cls": cls_name,
                        "conf": round(confidence, 4),
                        "xyxy": xyxy
                    })

            self.last_inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return detections
        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}")
            return []
