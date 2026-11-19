import asyncio
import cv2
import json
import time
import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.pipeline import pipeline
from backend.services.camera_worker import camera_worker
from backend.services.alert_manager import alert_manager
from backend.services.security import is_remote, verify_phone_token
from backend.api.routes_camera import get_phone_config

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

# --- /ws/stream limits -------------------------------------------------------------
AUTH_TIMEOUT_S = 5.0          # remote clients must authenticate within this time
STALE_ALERT_S = 5.0           # no frames for this long -> one camera_disconnected alert
IDLE_CLOSE_S = 30.0           # no frames for this long -> server closes the socket
MAX_FRAME_BYTES = 1_000_000   # reject frames larger than ~1 MB
MAX_PIXELS = 12_000_000       # decompression-bomb guard after decoding
MIN_FRAME_INTERVAL_S = 0.10   # server-side cap of ~10 FPS per phone
MAX_CONSECUTIVE_ERRORS = 20
WATCHDOG_TICK_S = 1.0         # how often the stall watchdog wakes up
CLEAN_CLOSE_CODES = (1000, 1001)  # Stop button / tab closed

_session_counter = 0
_current_session = 0
_current_socket: WebSocket | None = None


async def _announce_camera(connected: bool, reason: str = ""):
    """Tell dashboards the phone camera connected/disconnected (informational event)."""
    await broadcast_event({
        "type": "camera",
        "payload": {
            "source": "phone_camera",
            "connected": connected,
            "reason": reason,
            "timestamp": time.time(),
        },
    })


async def _authenticate(websocket: WebSocket, remote: bool):
    """
    Returns (ok, first_frame_bytes_or_None, did_handshake).
    Remote clients (tunnel / LAN) MUST send {"type":"auth","token":...} first.
    Local clients (laptop, same machine, no forwarding headers) may skip auth and
    send a binary frame directly, which keeps scripts/test_webcam_pipeline.py working.
    The token is never logged and never placed in the URL.
    """
    try:
        msg = await asyncio.wait_for(websocket.receive(), timeout=AUTH_TIMEOUT_S if remote else 30.0)
    except asyncio.TimeoutError:
        await websocket.close(code=4401)
        return False, None, False
    if msg.get("type") == "websocket.disconnect":
        return False, None, False

    if msg.get("text") is not None:
        try:
            data = json.loads(msg["text"])
        except Exception:
            data = {}
        if data.get("type") == "auth":
            if verify_phone_token(data.get("token")):
                return True, None, True
            await websocket.close(code=4401)
            return False, None, False
        if remote:
            await websocket.close(code=4401)
            return False, None, False
        return True, None, False  # local client sent some other text first

    # binary first message
    if remote:
        await websocket.close(code=4401)
        return False, None, False
    return True, msg.get("bytes"), False


@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Phone camera WebSocket stream.

    Handshake (remote phone): client sends {"type":"auth","token":"..."}; the server
    answers {"type":"auth_ok","config":{...}}. Local clients may skip the handshake.
    Then, per frame (with ack-based backpressure):
      client -> binary JPEG
      server -> JSON text (decision, cnn, yolo, fps, latency)  then  binary annotated JPEG
    Invalid / oversize frames get a JSON {"type":"error", ...} reply instead (also an ack).
    Only one phone session is active at a time: a newer connection replaces the older one.
    """
    global _session_counter, _current_session, _current_socket

    remote = is_remote(websocket.scope)
    await websocket.accept()

    ok, first_frame, handshake = await _authenticate(websocket, remote)
    if not ok:
        logger.warning("Phone stream rejected (authentication failed or timed out).")
        return

    # Single active session: replace any previous phone connection
    _session_counter += 1
    my_session = _session_counter
    previous = _current_socket
    _current_session = my_session
    _current_socket = websocket
    if previous is not None:
        try:
            await previous.close(code=4409)  # replaced by a newer session
        except Exception:
            pass

    camera_worker.source_type = "browser"
    camera_worker.connected = True
    camera_worker.disconnect_alerted = False
    logger.info(f"Phone camera connected to /ws/stream (session {my_session}, remote={remote})")

    if handshake:  # clients that skipped the (local-only) handshake keep the original protocol
        try:
            await websocket.send_text(json.dumps({"type": "auth_ok", "config": get_phone_config()}))
        except Exception:
            return
    await _announce_camera(True, "phone connected")

    last_received_ts = time.time()
    last_processed_ts = 0.0
    stale_alerted = False
    consecutive_errors = 0
    clean_close = False
    recv_task: asyncio.Future | None = None
    pending_first = first_frame

    async def handle_frame(data: bytes):
        nonlocal last_processed_ts, consecutive_errors
        if len(data) > MAX_FRAME_BYTES:
            await websocket.send_text(json.dumps({"type": "error", "error": "frame_too_large"}))
            return
        # Server-side pacing (never queue; just wait out the minimum interval)
        wait = MIN_FRAME_INTERVAL_S - (time.time() - last_processed_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            await websocket.send_text(json.dumps({"type": "error", "error": "invalid_frame"}))
            return
        if bgr.shape[0] * bgr.shape[1] > MAX_PIXELS:
            await websocket.send_text(json.dumps({"type": "error", "error": "frame_too_large"}))
            return

        last_processed_ts = time.time()
        camera_worker.update_frame_from_browser(bgr)
        result = await pipeline.process_frame(bgr, source="phone_camera")
        consecutive_errors = 0

        await websocket.send_text(json.dumps({
            "ts": last_processed_ts,
            "fps": result["fps"],
            "decision": result["decision"],
            "cnn": result["cnn"],
            "yolo": result["yolo"],
            "latency_ms": result["latency_ms"],
        }))
        ret, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            await websocket.send_bytes(buffer.tobytes())

    try:
        while True:
            if pending_first is not None:
                data, pending_first = pending_first, None
                msg = {"type": "websocket.receive", "bytes": data}
            else:
                if recv_task is None:
                    recv_task = asyncio.ensure_future(websocket.receive())
                done, _ = await asyncio.wait({recv_task}, timeout=WATCHDOG_TICK_S)
                now = time.time()
                if not done:
                    # Watchdog: no data for a while
                    idle = now - last_received_ts
                    if idle > STALE_ALERT_S and not stale_alerted:
                        stale_alerted = True
                        camera_worker.connected = False
                        logger.warning(f"Phone stream stalled ({idle:.1f}s without frames).")
                        await alert_manager.raise_alert(
                            alert_type="camera_disconnected",
                            message="Phone camera stopped sending frames.",
                            state="UNCERTAIN",
                            source="phone_camera",
                        )
                        await _announce_camera(False, "no frames received")
                    if idle > IDLE_CLOSE_S:
                        logger.warning("Closing idle phone stream.")
                        await websocket.close(code=4408)
                        return
                    continue
                msg = recv_task.result()
                recv_task = None

            if msg.get("type") == "websocket.disconnect":
                code = msg.get("code", 1005)
                clean_close = code in CLEAN_CLOSE_CODES
                break

            if msg.get("text") is not None:
                if msg["text"] == "ping":
                    await websocket.send_text("pong")
                continue

            data = msg.get("bytes")
            if not data:
                continue

            last_received_ts = time.time()
            if stale_alerted:
                stale_alerted = False
                camera_worker.connected = True
                camera_worker.disconnect_alerted = False
                await _announce_camera(True, "frames resumed")

            try:
                await handle_frame(data)
            except WebSocketDisconnect:
                raise
            except Exception as e:  # never let one bad frame kill the stream
                consecutive_errors += 1
                logger.error(f"Frame processing error ({consecutive_errors}): {type(e).__name__}: {e}")
                try:
                    await websocket.send_text(json.dumps({"type": "error", "error": "processing_error"}))
                except Exception:
                    break
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    await websocket.close(code=1011)
                    break

    except WebSocketDisconnect as e:
        clean_close = getattr(e, "code", 1005) in CLEAN_CLOSE_CODES
    except Exception as e:
        logger.error(f"Error in /ws/stream loop: {type(e).__name__}: {e}")
    finally:
        if recv_task is not None and not recv_task.done():
            recv_task.cancel()
        # Only the newest session may update shared camera state / raise alerts
        if my_session == _current_session:
            _current_socket = None
            camera_worker.connected = False
            if not clean_close and not stale_alerted:
                await alert_manager.raise_alert(
                    alert_type="camera_disconnected",
                    message="Phone camera stream disconnected unexpectedly.",
                    state="UNCERTAIN",
                    source="phone_camera",
                )
            await _announce_camera(False, "stopped" if clean_close else "connection lost")
        logger.info(f"Phone camera session {my_session} ended (clean={clean_close})")
