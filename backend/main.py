import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.config import settings
from backend.database.db import init_db
from backend.models.model_registry import registry
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

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits localhost Vite dev and tunneled domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(routes_health.router, prefix="/api", tags=["Health"])
app.include_router(routes_detect.router, prefix="/api", tags=["Detection"])
app.include_router(routes_camera.router, prefix="/api", tags=["Camera"])
app.include_router(routes_data.router, prefix="/api", tags=["Data & Analytics"])
app.include_router(routes_settings.router, prefix="/api", tags=["Settings"])
app.include_router(ws.router, tags=["WebSockets"])

# Mount built frontend if available (in demo mode)
frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        # Don't intercept API or WS routes
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return None
        file_candidate = frontend_dist / full_path
        if file_candidate.is_file():
            return FileResponse(str(file_candidate))
        return FileResponse(str(frontend_dist / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
