import cv2
import json
import time
import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.pipeline import pipeline
from backend.services.camera_worker import camera_worker
from backend.services.alert_manager import alert_manager

logger = logging.getLogger("forestguard.ws")

router = APIRouter()

# Active dashboard event subscriber connections
active_event_connections: list[WebSocket] = []

async def broadcast_event(event_dict: dict):
    """Broadcasts a JSON event to all connected dashboard clients."""
    disconnected = []
    for ws in active_event_connections:
        try:
            await ws.send_json(event_dict)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in active_event_connections:
            active_event_connections.remove(ws)

# Wire broadcast into pipeline and alert_manager
pipeline.set_ws_broadcast(broadcast_event)

@router.websocket("/ws/events")
async def websocket_events_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_event_connections.append(websocket)
    logger.info(f"Dashboard client connected to /ws/events (Total: {len(active_event_connections)})")
    try:
        while True:
            # Keep-alive ping/pong
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        logger.info("Dashboard client disconnected from /ws/events")
    finally:
        if websocket in active_event_connections:
            active_event_connections.remove(websocket)

@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Phone camera WebSocket stream.
    Receives binary JPEG frames with backpressure:
    1. Replies with JSON text message (detection, cnn, yolo, decision, fps)
    2. Replies with binary annotated JPEG frame
    """
    await websocket.accept()
    camera_worker.source_type = "browser"
    camera_worker.connected = True
    camera_worker.disconnect_alerted = False
    logger.info("Phone camera connected to /ws/stream")

    last_received_ts = time.time()

    try:
        while True:
            # Receive frame data (bytes)
            data = await websocket.receive_bytes()
            now = time.time()
            last_received_ts = now

            # Decode JPEG
            nparr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr is None:
                continue

            camera_worker.update_frame_from_browser(bgr)

            # Process frame through inference pipeline
            result = await pipeline.process_frame(bgr, source="phone_camera")

            # 1. Send JSON metadata
            meta_payload = {
                "ts": now,
                "fps": result["fps"],
                "decision": result["decision"],
                "cnn": result["cnn"],
                "yolo": result["yolo"],
                "latency_ms": result["latency_ms"]
            }
            await websocket.send_text(json.dumps(meta_payload))

            # 2. Send binary annotated JPEG frame
            ret, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        logger.info("Phone camera disconnected from /ws/stream")
    except Exception as e:
        logger.error(f"Error in /ws/stream loop: {e}")
    finally:
        camera_worker.connected = False
        # Alert if unexpectedly disconnected after streaming
        if (time.time() - last_received_ts) > 5.0:
            await alert_manager.raise_alert(
                alert_type="camera_disconnected",
                message="Phone camera stream disconnected.",
                state="UNCERTAIN",
                source="phone_camera"
            )
