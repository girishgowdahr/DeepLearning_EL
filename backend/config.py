import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory: ForestGuard-AI/
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
env_file = BASE_DIR / "backend" / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv(BASE_DIR / ".env")

class Settings:
    PROJECT_NAME: str = "ForestGuard AI"
    VERSION: str = "1.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    BASE_DIR: Path = BASE_DIR
    BACKEND_DIR: Path = BASE_DIR / "backend"
    DATA_DIR: Path = BASE_DIR / "backend" / "data"
    DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "backend/data/forestguard.db")
    MODEL_DIR: Path = BASE_DIR / os.getenv("MODEL_DIR", "model_weights")
    SNAPSHOT_DIR: Path = BASE_DIR / os.getenv("SNAPSHOT_DIR", "backend/data/snapshots")
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "backend/data/uploads")
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    CORS_ORIGINS: list[str] = [
        origin.strip() 
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",")
        if origin.strip()
    ]
    
    # Alert stubs (disabled by default)
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    ESP32_BUZZER_URL: str = os.getenv("ESP32_BUZZER_URL", "")

settings = Settings()

# Ensure runtime directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(settings.RESULTS_DIR / "graphs").mkdir(parents=True, exist_ok=True)
(settings.RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(settings.RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
