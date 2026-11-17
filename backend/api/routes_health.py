import time
from fastapi import APIRouter
from backend.config import settings

router = APIRouter()
start_time = time.time()

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "uptime_seconds": round(time.time() - start_time, 1)
    }
