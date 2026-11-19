import cv2
import time
import asyncio
from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from backend.services.camera_worker import camera_worker
from backend.services.pipeline import pipeline
from backend.services.security import is_remote, verify_phone_token
from backend.database import db

router = APIRouter()


def get_phone_config() -> dict:
    """Streaming parameters the phone page should use (clamped to safe ranges)."""
    saved = db.get_all_settings()

    def clamp(value, lo, hi, default):
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return default

    return {
        "send_width": int(clamp(saved.get("send_width", 640), 320, 960, 640)),
        "jpeg_quality": round(clamp(saved.get("jpeg_quality", 0.6), 0.3, 0.9, 0.6), 2),
        "fps_cap": int(clamp(saved.get("inference_fps_cap", 5), 1, 10, 5)),
    }


@router.get("/phone/config")
def phone_config(request: Request, x_fg_token: str | None = Header(default=None)):
    """Minimal, token-protected config for the phone page (the only /api route remote devices may call)."""
    if is_remote(request.scope) and not verify_phone_token(x_fg_token):
        raise HTTPException(status_code=401, detail="Invalid or missing access token.")
    return get_phone_config()

class CameraStartRequest(BaseModel):
    source: str = "webcam"  # "ip", "webcam", "browser"
    url: Optional[str] = None
    index: Optional[int] = 0

@router.post("/camera/start")
def start_camera(req: CameraStartRequest):
    if req.source == "browser":
        camera_worker.source_type = "browser"
        camera_worker.connected = True
        return {"success": True, "message": "Phone browser camera stream selected."}
    elif req.source == "ip":
        if not req.url:
            raise HTTPException(status_code=400, detail="URL is required for IP camera source.")
        camera_worker.start(source_type="ip", target=req.url)
        return {"success": True, "message": f"IP Camera stream started from {req.url}"}
    elif req.source == "webcam":
        camera_worker.start(source_type="webcam", target=req.index or 0)
        return {"success": True, "message": f"Local webcam started on index {req.index or 0}"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown camera source: {req.source}")

@router.post("/camera/stop")
def stop_camera():
    camera_worker.stop()
    return {"success": True, "message": "Camera stopped"}

@router.get("/camera/status")
def get_camera_status():
    return camera_worker.get_status()

async def _mjpeg_stream_generator():
    """Yields multipart/x-mixed-replace MJPEG frames for dashboard <img>."""
    while True:
        # If camera worker has a live frame from RTSP/webcam, run pipeline on it
        if camera_worker.running and camera_worker.latest_frame is not None:
            with camera_worker.lock:
                frame_to_process = camera_worker.latest_frame.copy()
            res = await pipeline.process_frame(frame_to_process, source=camera_worker.source_type)
            ann = res["annotated_frame"]
        elif pipeline.latest_annotated_frame is not None:
            ann = pipeline.latest_annotated_frame
        else:
            # Standby blank frame with "Awaiting Stream"
            import numpy as np
            ann = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                ann,
                "Awaiting Live Stream...",
                (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (150, 150, 150),
                2,
                cv2.LINE_AA
            )

        ret, buffer = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        
        # Stream at ~10-15 FPS display rate
        await asyncio.sleep(0.08)

@router.get("/stream/annotated")
def stream_annotated():
    return StreamingResponse(
        _mjpeg_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
