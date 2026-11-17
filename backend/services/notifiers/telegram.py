import logging
from backend.config import settings
from backend.services.notifiers.base import BaseNotifier

logger = logging.getLogger("forestguard.notifier.telegram")

class TelegramNotifier(BaseNotifier):
    """
    MODULAR STUB (Future Enhancement).
    Disabled by default. In this version, does NOT send any network requests.
    
    To enable in future deployments:
    1. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in backend/.env
    2. Set enabled = True
    3. Call the Telegram Bot API: https://api.telegram.org/bot<TOKEN>/sendPhoto
    """
    name = "telegram"
    enabled = False  # Explicitly disabled by default for academic prototype

    def __init__(self):
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

    async def send(self, alert: dict) -> bool:
        if not self.enabled:
            logger.debug("Telegram notifier is disabled (stub). Skipping.")
            return False

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram notifier enabled but TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not configured.")
            return False

        # In a future release with active hardware/bot:
        # payload = {
        #     "chat_id": self.chat_id,
        #     "caption": f"🚨 [FORESTGUARD ALERT] {alert.get('type')}: {alert.get('message')}"
        # }
        # async with httpx.AsyncClient() as client:
        #     await client.post(f"https://api.telegram.org/bot{self.bot_token}/sendMessage", data=payload)
        logger.info(f"[STUB] Telegram alert would be sent: {alert.get('type')}")
        return True
