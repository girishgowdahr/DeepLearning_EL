import json
import logging
from pathlib import Path
from backend.config import settings
from backend.models.cnn_classifier import CNNClassifier
from backend.models.yolo_detector import YOLODetector

logger = logging.getLogger("forestguard.models")

class ModelRegistry:
    def __init__(self):
        self.cnn_path = settings.MODEL_DIR / "cnn_mobilenetv2.onnx"
        self.yolo_path = settings.MODEL_DIR / "yolo_best.pt"
        if not self.yolo_path.exists():
            onnx_opt = settings.MODEL_DIR / "yolo_best.onnx"
            if onnx_opt.exists():
                self.yolo_path = onnx_opt

        self.cnn = CNNClassifier(self.cnn_path)
        self.yolo = YOLODetector(self.yolo_path)
        self.model_info_path = settings.MODEL_DIR / "MODEL_INFO.json"

    def reload(self):
        """Reload models if new weight files are dropped in."""
        logger.info("Reloading models from disk...")
        self.cnn.model_path = self.cnn_path
        self.cnn.load_model()
        self.yolo.model_path = self.yolo_path
        self.yolo.load_model()

    def get_status(self) -> dict:
        info = self.get_model_info()
        return {
            "cnn": {
                "name": "MobileNetV2 Classifier",
                "available": self.cnn.available,
                "loaded": self.cnn.available,
                "version": info.get("cnn", {}).get("version", "1.0.0"),
                "trained_by": info.get("cnn", {}).get("trained_by", "Pending / Not Trained Yet"),
                "last_inference_ms": self.cnn.last_inference_ms,
                "format": "ONNX (CPU)"
            },
            "yolo": {
                "name": "YOLO Nano Detector",
                "available": self.yolo.available,
                "loaded": self.yolo.available,
                "version": info.get("yolo", {}).get("version", "1.0.0"),
                "trained_by": info.get("yolo", {}).get("trained_by", "Pending / Not Trained Yet"),
                "last_inference_ms": self.yolo.last_inference_ms,
                "format": "PyTorch / ONNX (CPU)"
            }
        }

    def get_model_info(self) -> dict:
        if self.model_info_path.exists():
            try:
                with open(self.model_info_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read MODEL_INFO.json: {e}")
        return {
            "cnn": {
                "name": "MobileNetV2",
                "architecture": "MobileNetV2 Transfer Learning",
                "version": "1.0.0",
                "trained_by": "Not trained yet",
                "input_size": "224x224",
                "classes": ["fire", "no_fire"]
            },
            "yolo": {
                "name": "YOLO nano",
                "architecture": "Ultralytics YOLO nano",
                "version": "1.0.0",
                "trained_by": "Not trained yet",
                "input_size": "416x416",
                "classes": ["fire", "smoke"]
            }
        }

registry = ModelRegistry()
