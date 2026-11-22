
import cv2
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from backend.config import settings
from backend.database import db
from backend.models.model_registry import registry
from backend.services.decision_engine import DecisionEngine, DecisionState
from backend.services.annotate import annotate_frame
from backend.services.alert_manager import alert_manager

logger = logging.getLogger("forestguard.pipeline")


class InferencePipeline:
    """
    Latest-frame-wins inference pipeline.
    Runs YOLO and MobileNetV2 CNN on incoming frames.
    Still images use immediate decisions.
    Live camera and video retain temporal confirmation.
    """

    def __init__(self, ws_broadcast_fn=None):
        self.ws_broadcast_fn = ws_broadcast_fn
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.decision_engine = DecisionEngine()

        # Performance and FPS tracking
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0
        self.frame_count = 0

        # Snapshot throttling
        self.last_snapshot_time = 0.0
        self.latest_annotated_frame = None

    def set_ws_broadcast(self, ws_broadcast_fn):
        self.ws_broadcast_fn = ws_broadcast_fn
        alert_manager.set_ws_broadcast(ws_broadcast_fn)

    def resize_for_inference(self, bgr_image, max_dim: int = 640):
        h, w = bgr_image.shape[:2]

        if max(h, w) <= max_dim:
            return bgr_image

        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(
            bgr_image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    async def process_frame(
        self,
        bgr_image,
        source: str = "webcam"
    ) -> dict:
        """
        Processes one image or video/camera frame.
        Image uploads use immediate decisions.
        Live sources use temporal confirmation.
        """

        t0 = time.perf_counter()
        now_ts = time.time()

        # 1. Resize input
        resized = self.resize_for_inference(
            bgr_image,
            max_dim=640
        )

        # 2. Run actual trained models
        yolo_dets = []

        if registry.yolo.available:
            yolo_dets = registry.yolo.predict(
                resized,
                conf=self.decision_engine.yolo_conf_fire,
                imgsz=640
            )

        cnn_res = None

        if registry.cnn.available:
            cnn_res = registry.cnn.predict(resized)

        # 3. Decision engine
        if source == "image_upload":

            # Single uploaded image: no temporal waiting
            candidate, reason = self.decision_engine.evaluate_frame(
                cnn_result=cnn_res,
                yolo_result=yolo_dets,
                cnn_available=registry.cnn.available,
                yolo_available=registry.yolo.available
            )

            status_map = {
                "FIRE": DecisionState.FIRE_CONFIRMED,
                "SMOKE": DecisionState.SMOKE_CONFIRMED,
                "POSSIBLE_FIRE": DecisionState.POSSIBLE_FIRE,
                "UNCERTAIN": DecisionState.UNCERTAIN,
                "NO_FIRE": DecisionState.NO_FIRE,
                DecisionState.MODEL_UNAVAILABLE:
                    DecisionState.MODEL_UNAVAILABLE
            }

            state = status_map.get(
                candidate,
                DecisionState.UNCERTAIN
            )

            should_alert = state in (
                DecisionState.FIRE_CONFIRMED,
                DecisionState.SMOKE_CONFIRMED,
                DecisionState.POSSIBLE_FIRE
            )

            decision = {
                "status": state,
                "candidate": candidate,
                "reason": reason,
                "counts": {candidate: 1},
                "window_size": 1,
                "event_active": should_alert,
                "active_event": state if should_alert else None,
                "should_alert": should_alert,
                "alert_type": (
                    "fire_detected"
                    if state == DecisionState.FIRE_CONFIRMED
                    else "smoke_detected"
                    if state == DecisionState.SMOKE_CONFIRMED
                    else "possible_fire"
                    if state == DecisionState.POSSIBLE_FIRE
                    else None
                ),
                "alert_message": reason if should_alert else None,
                "timestamp": now_ts
            }

        else:
            # Live camera and video retain temporal confirmation
            decision = self.decision_engine.update(
                cnn_result=cnn_res,
                yolo_result=yolo_dets,
                cnn_available=registry.cnn.available,
                yolo_available=registry.yolo.available
            )

        # 4. Calculate FPS
        dt = t0 - self.last_frame_time
        self.last_frame_time = t0

        if dt > 0:
            current_fps = 1.0 / dt

            self.fps = (
                0.9 * self.fps + 0.1 * current_fps
                if self.fps > 0
                else current_fps
            )

        # 5. Annotate frame
        annotated = annotate_frame(
            resized,
            decision_result=decision,
            yolo_detections=yolo_dets,
            cnn_result=cnn_res,
            fps=self.fps
        )

        self.latest_annotated_frame = annotated

        # 6. Threat detection and snapshots
        snapshot_rel_path = None
        state = decision["status"]

        is_threat = state in (
            DecisionState.FIRE_CONFIRMED,
            DecisionState.SMOKE_CONFIRMED,
            DecisionState.POSSIBLE_FIRE,
            DecisionState.UNCERTAIN
        )

        should_save_snapshot = False

        if (
            decision["should_alert"]
            or (
                is_threat
                and (now_ts - self.last_snapshot_time) >= 2.0
            )
        ):
            should_save_snapshot = True
            self.last_snapshot_time = now_ts

        if should_save_snapshot:
            filename = f"snap_{int(now_ts * 1000)}.jpg"
            snap_full_path = settings.SNAPSHOT_DIR / filename

            cv2.imwrite(
                str(snap_full_path),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            snapshot_rel_path = (
                f"/api/detections/snapshots/{filename}"
            )

        # 7. Database logging and counters
        db.increment_counter("frames_processed", 1)

        top_class = "none"
        top_conf = 0.0

        if yolo_dets:
            top = max(
                yolo_dets,
                key=lambda d: d.get("conf", 0.0)
            )

            top_class = top.get("cls", "none")
            top_conf = top.get("conf", 0.0)

            if top_class == "fire":
                db.increment_counter("fire_detections", 1)

            elif top_class == "smoke":
                db.increment_counter("smoke_detections", 1)

        elif (
            cnn_res
            and cnn_res.get("label") == "fire"
            and cnn_res.get("confidence", 0) >= 0.60
        ):
            top_class = "fire"
            top_conf = cnn_res.get("confidence", 0)

            db.increment_counter("fire_detections", 1)

        elif state == DecisionState.MODEL_UNAVAILABLE:
            top_class = "unavailable"
            top_conf = 0.0

        else:
            db.increment_counter("no_fire_classifications", 1)

        # 8. Alert handling
        alert_id = None

        if decision["should_alert"]:
            alert_record = await alert_manager.raise_alert(
                alert_type=(
                    decision["alert_type"] or "threat_detected"
                ),
                message=(
                    decision["alert_message"]
                    or decision["reason"]
                ),
                state=state,
                source=source,
                snapshot_path=snapshot_rel_path
            )

            alert_id = alert_record.get("id")

        # 9. Detection history
        if is_threat or should_save_snapshot:
            cnn_label = (
                cnn_res.get("label")
                if cnn_res
                else None
            )

            cnn_conf = (
                cnn_res.get("confidence", 0)
                if cnn_res
                else 0.0
            )

            db.insert_detection({
                "ts": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "state": state,
                "cnn_label": cnn_label,
                "cnn_conf": cnn_conf,
                "yolo": yolo_dets,
                "top_class": top_class,
                "top_conf": top_conf,
                "snapshot_path": snapshot_rel_path,
                "alert_id": alert_id
            })

        # 10. Calculate inference latency
        total_latency = (
            time.perf_counter() - t0
        ) * 1000.0

        # 11. Broadcast detection event
        event_payload = {
            "type": "detection",
            "payload": {
                "state": state,
                "candidate": decision["candidate"],
                "reason": decision["reason"],
                "source": source,
                "cnn": cnn_res,
                "yolo": yolo_dets,
                "fps": round(self.fps, 1),
                "latency_ms": round(total_latency, 1),
                "timestamp": now_ts
            }
        }

        if self.ws_broadcast_fn:
            try:
                await self.ws_broadcast_fn(event_payload)

            except Exception as e:
                logger.error(
                    f"Error broadcasting detection event: {e}"
                )

        # 12. Return response
        return {
            "annotated_frame": annotated,
            "decision": decision,
            "yolo": yolo_dets,
            "cnn": cnn_res,
            "fps": round(self.fps, 1),
            "latency_ms": round(total_latency, 1),
            "snapshot_path": snapshot_rel_path
        }


pipeline = InferencePipeline()
