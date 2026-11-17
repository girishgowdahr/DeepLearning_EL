import logging
from backend.config import settings
from backend.services.notifiers.base import BaseNotifier

logger = logging.getLogger("forestguard.notifier.esp32")

class Esp32BuzzerNotifier(BaseNotifier):
    """
    MODULAR STUB (Future Enhancement).
    Disabled by default. In this version, does NOT send any hardware commands.
    
    To enable in future physical IoT deployments:
    1. Flash the ESP32 sketch provided in docs/esp32_buzzer.md to your microcontroller.
    2. Set ESP32_BUZZER_URL=http://<esp32-ip>/buzz?ms=3000 in backend/.env
    3. Set enabled = True
    """
    name = "esp32_buzzer"
    enabled = False  # Explicitly disabled by default for academic prototype

    def __init__(self):
        self.buzzer_url = settings.ESP32_BUZZER_URL

    async def send(self, alert: dict) -> bool:
        if not self.enabled:
            logger.debug("ESP32 buzzer notifier is disabled (stub). Skipping.")
            return False

        if not self.buzzer_url:
            logger.warning("ESP32 notifier enabled but ESP32_BUZZER_URL is empty.")
            return False

        # In a future release with active hardware:
        # async with httpx.AsyncClient() as client:
        #     await client.get(self.buzzer_url, timeout=2.0)
        logger.info(f"[STUB] ESP32 buzzer trigger would be sent: {self.buzzer_url}")
        return True
