from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.database import db
from backend.services.pipeline import pipeline

router = APIRouter()

DEFAULT_SETTINGS = {
    "yolo_conf_fire": 0.40,
    "yolo_conf_smoke": 0.45,
    "cnn_threshold": 0.60,
    "cnn_strong_threshold": 0.90,
    "window_size": 6,
    "confirm_frames": 3,
    "clear_seconds": 10.0,
    "cooldown_seconds": 30.0,
    "sound_enabled": True,
    "send_width": 640,
    "jpeg_quality": 0.6,
    "telegram_enabled": False,
    "esp32_enabled": False
}

class SettingsUpdateRequest(BaseModel):
    yolo_conf_fire: Optional[float] = None
    yolo_conf_smoke: Optional[float] = None
    cnn_threshold: Optional[float] = None
    cnn_strong_threshold: Optional[float] = None
    window_size: Optional[int] = None
    confirm_frames: Optional[int] = None
    clear_seconds: Optional[float] = None
    cooldown_seconds: Optional[float] = None
    sound_enabled: Optional[bool] = None
    send_width: Optional[int] = None
    jpeg_quality: Optional[float] = None

@router.get("/settings")
def get_settings():
    saved = db.get_all_settings()
    merged = {**DEFAULT_SETTINGS, **saved}
    return merged

@router.put("/settings")
def update_settings(req: SettingsUpdateRequest):
    current = get_settings()
    updates = req.model_dump(exclude_unset=True)
    for k, v in updates.items():
        if v is not None:
            current[k] = v
            db.set_setting(k, v)

    # Apply live to pipeline decision engine
    if "yolo_conf_fire" in updates and updates["yolo_conf_fire"] is not None:
        pipeline.decision_engine.yolo_conf_fire = updates["yolo_conf_fire"]
    if "yolo_conf_smoke" in updates and updates["yolo_conf_smoke"] is not None:
        pipeline.decision_engine.yolo_conf_smoke = updates["yolo_conf_smoke"]
    if "cnn_threshold" in updates and updates["cnn_threshold"] is not None:
        pipeline.decision_engine.cnn_threshold = updates["cnn_threshold"]
    if "cnn_strong_threshold" in updates and updates["cnn_strong_threshold"] is not None:
        pipeline.decision_engine.cnn_strong_threshold = updates["cnn_strong_threshold"]
    if "window_size" in updates and updates["window_size"] is not None:
        pipeline.decision_engine.window_size = updates["window_size"]
    if "confirm_frames" in updates and updates["confirm_frames"] is not None:
        pipeline.decision_engine.confirm_frames = updates["confirm_frames"]
    if "clear_seconds" in updates and updates["clear_seconds"] is not None:
        pipeline.decision_engine.clear_seconds = updates["clear_seconds"]
    if "cooldown_seconds" in updates and updates["cooldown_seconds"] is not None:
        pipeline.decision_engine.cooldown_seconds = updates["cooldown_seconds"]

    return {"success": True, "settings": current}
