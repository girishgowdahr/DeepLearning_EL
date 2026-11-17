import json
import socket
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from backend.config import settings
from backend.database import db
from backend.models.model_registry import registry
from backend.services.alert_manager import alert_manager

router = APIRouter()

@router.get("/detections")
def get_detections(limit: int = 50, offset: int = 0, source: str = None, state: str = None):
    items = db.get_detections(limit=limit, offset=offset, source=source, state=state)
    return {
        "detections": items,
        "count": len(items),
        "limit": limit,
        "offset": offset
    }

@router.get("/detections/snapshots/{filename}")
def get_snapshot(filename: str):
    snap_path = settings.SNAPSHOT_DIR / filename
    if not snap_path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return FileResponse(str(snap_path), media_type="image/jpeg")

@router.get("/alerts")
def get_alerts(limit: int = 50, offset: int = 0, acknowledged: bool = None):
    alerts = db.get_alerts(limit=limit, offset=offset, acknowledged=acknowledged)
    return {
        "alerts": alerts,
        "count": len(alerts)
    }

@router.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    success = db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    return {"success": True, "alert_id": alert_id}

@router.post("/alerts/test")
async def trigger_test_alert():
    """Triggers a safe test alert for sound and notification verification."""
    alert = await alert_manager.raise_test_alert()
    return {"success": True, "alert": alert}

@router.get("/analytics")
def get_analytics():
    counters = db.get_all_counters()
    
    # Read real metrics from results/metrics/*.json if available
    metrics_dir = settings.RESULTS_DIR / "metrics"
    real_metrics = {}
    if metrics_dir.exists():
        for file in metrics_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    real_metrics[file.stem] = json.load(f)
            except Exception:
                pass

    recent_detections = db.get_detections(limit=20)
    
    return {
        "counters": counters,
        "recent_detections": recent_detections,
        "model_performance_metrics": real_metrics
    }

@router.get("/models/status")
def get_models_status():
    return registry.get_status()

@router.get("/models/info")
def get_models_info():
    return registry.get_model_info()

@router.post("/models/reload")
def reload_models():
    registry.reload()
    return {"success": True, "status": registry.get_status()}

def _get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@router.get("/system/network")
def get_network_info():
    lan_ip = _get_lan_ip()
    tunnel_file = settings.BASE_DIR / "run" / "tunnel_url.txt"
    tunnel_url = None
    if tunnel_file.exists():
        try:
            with open(tunnel_file, "r", encoding="utf-8") as f:
                tunnel_url = f.read().strip()
        except Exception:
            pass

    phone_url = f"{tunnel_url}/camera" if tunnel_url else f"https://{lan_ip}:8000/camera"

    return {
        "lan_ip": lan_ip,
        "port": settings.PORT,
        "tunnel_url": tunnel_url,
        "phone_camera_url": phone_url,
        "is_tunnel_active": tunnel_url is not None
    }
