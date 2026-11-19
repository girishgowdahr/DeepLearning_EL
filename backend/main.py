import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.config import settings
from backend.database.db import init_db
from backend.models.model_registry import registry
from backend.services.security import RemoteAccessGuard
from backend.api import routes_health, routes_detect, routes_camera, routes_data, routes_settings, ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("forestguard.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ForestGuard AI Backend...")
    init_db()
    status = registry.get_status()
    logger.info(f"Model status on startup: CNN available={status['cnn']['available']}, YOLO available={status['yolo']['available']}")
    yield
    # Shutdown
    logger.info("Shutting down ForestGuard AI Backend...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS Middleware: same-origin by default (the dashboard and the phone page are both
# served from this app, and Vite dev proxies /api and /ws). Only explicit origins from
# CORS_ORIGINS are allowed; no wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Remote-access guard (added last = outermost). Requests arriving through a public
# tunnel or from another device are limited to the phone page and the token-protected
# /ws/stream; the dashboard, API, history, snapshots and docs stay laptop-only.
app.add_middleware(RemoteAccessGuard)

# Include API Routers
app.include_router(routes_health.router, prefix="/api", tags=["Health"])
app.include_router(routes_detect.router, prefix="/api", tags=["Detection"])
app.include_router(routes_camera.router, prefix="/api", tags=["Camera"])
app.include_router(routes_data.router, prefix="/api", tags=["Data & Analytics"])
app.include_router(routes_settings.router, prefix="/api", tags=["Settings"])
app.include_router(ws.router, tags=["WebSockets"])

# Mount results directory for serving evaluation graphs & metrics
if settings.RESULTS_DIR.exists():
    app.mount("/results", StaticFiles(directory=str(settings.RESULTS_DIR)), name="results")

# Mount built frontend if available (in demo mode)
frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
    
    _frontend_root = frontend_dist.resolve()

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        # Unknown API / WS / results paths are real 404s, not the SPA shell
        if full_path.startswith(("api/", "ws/", "results/")):
            raise HTTPException(status_code=404, detail="Not found")
        # Only serve files that resolve inside frontend/dist (blocks ../ traversal)
        candidate = (_frontend_root / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(_frontend_root):
            return FileResponse(str(candidate))
        return FileResponse(str(_frontend_root / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
