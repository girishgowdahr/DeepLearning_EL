
import json
import logging

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
        """Reload models from disk."""
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
                "version": info["cnn"]["version"],
                "trained_by": info["cnn"]["trained_by"],
                "last_inference_ms": self.cnn.last_inference_ms,
                "format": "ONNX (CPU)"
            },
            "yolo": {
                "name": "YOLO Nano Detector",
                "available": self.yolo.available,
                "loaded": self.yolo.available,
                "version": info["yolo"]["version"],
                "trained_by": info["yolo"]["trained_by"],
                "last_inference_ms": self.yolo.last_inference_ms,
                "format": "PyTorch / ONNX (CPU)"
            }
        }

    def get_model_info(self) -> dict:
        info = {}

        if self.model_info_path.exists():
            try:
                with open(self.model_info_path, "r", encoding="utf-8") as f:
                    info = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Failed to read MODEL_INFO.json: {e}")

        cnn_trained = self.cnn_path.exists()
        yolo_trained = self.yolo_path.exists()

        cnn_info = info.get("cnn", {})
        yolo_info = info.get("yolo", {})

        info["cnn"] = {
            "name": cnn_info.get("name", "MobileNetV2"),
            "architecture": cnn_info.get(
                "architecture", "MobileNetV2 Transfer Learning"
            ),
            "version": cnn_info.get("version", "1.0.0"),
            "trained_by": cnn_info.get(
                "trained_by",
                "Trained - MobileNetV2 Transfer Learning"
                if cnn_trained else "Not trained"
            ),
            "input_size": cnn_info.get("input_size", "224x224"),
            "classes": cnn_info.get("classes", ["fire", "no_fire"]),
            "training_status": "Completed" if cnn_trained else "Pending"
        }

        info["yolo"] = {
            "name": yolo_info.get("name", "YOLO Nano"),
            "architecture": yolo_info.get(
                "architecture", "Ultralytics YOLO nano"
            ),
            "version": yolo_info.get("version", "1.0.0"),
            "trained_by": yolo_info.get(
                "trained_by",
                "Trained - YOLO Transfer Learning"
                if yolo_trained else "Not trained"
            ),
            "input_size": yolo_info.get("input_size", "640x640"),
            "classes": yolo_info.get("classes", ["smoke", "fire"]),
            "training_status": "Completed" if yolo_trained else "Pending"
        }

        return info


registry = ModelRegistry()
