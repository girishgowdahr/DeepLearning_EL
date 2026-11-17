import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("forestguard.cnn")

class CNNClassifier:
    """
    MobileNetV2 Fire/No-Fire Classifier using ONNX Runtime (CPU).
    Gracefully handles missing weights or missing dependencies.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.session = None
        self.available = False
        self.input_name = None
        self.output_name = None
        self.last_inference_ms = 0.0
        self.classes = ["fire", "no_fire"]
        
        # ImageNet normalization constants
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.load_model()

    def load_model(self):
        if not self.model_path or not Path(self.model_path).exists():
            logger.warning(f"CNN model file not found at {self.model_path}. Running in unavailable mode.")
            self.available = False
            return

        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.available = True
            logger.info(f"CNN model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            self.available = False

    def preprocess(self, bgr_image):
        import cv2
        import numpy as np

        # Resize to 224x224
        img = cv2.resize(bgr_image, (224, 224))
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert to float32 and scale to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet mean and std
        img = (img - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension: (1, 3, 224, 224)
        return np.expand_dims(img, axis=0)

    def predict(self, bgr_image) -> Optional[dict]:
        """
        Runs inference on a BGR image.
        Returns:
            {
                "label": "fire" | "no_fire",
                "confidence": float,
                "probs": {"fire": float, "no_fire": float},
                "latency_ms": float
            }
        """
        if not self.available or self.session is None:
            return None

        import numpy as np

        t0 = time.perf_counter()
        inp = self.preprocess(bgr_image)
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        logits = outputs[0][0]

        # Stable Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        latency = (time.perf_counter() - t0) * 1000.0
        self.last_inference_ms = latency

        prob_fire = float(probs[0])
        prob_no_fire = float(probs[1])

        label = "fire" if prob_fire >= prob_no_fire else "no_fire"
        confidence = prob_fire if label == "fire" else prob_no_fire

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probs": {
                "fire": round(prob_fire, 4),
                "no_fire": round(prob_no_fire, 4)
            },
            "latency_ms": round(latency, 2)
        }
