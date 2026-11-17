import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from backend.database import db
from backend.services.notifiers.base import BaseNotifier
from backend.services.notifiers.dashboard import DashboardNotifier
from backend.services.notifiers.telegram import TelegramNotifier
from backend.services.notifiers.esp32_buzzer import Esp32BuzzerNotifier

logger = logging.getLogger("forestguard.alert_manager")

class AlertManager:
    def __init__(self, ws_broadcast_fn=None):
        self.dashboard_notifier = DashboardNotifier(ws_broadcast_fn)
        self.telegram_notifier = TelegramNotifier()
        self.esp32_notifier = Esp32BuzzerNotifier()
        
        self.notifiers: list[BaseNotifier] = [
            self.dashboard_notifier,
            self.telegram_notifier,
            self.esp32_notifier
        ]

    def set_ws_broadcast(self, ws_broadcast_fn):
        self.dashboard_notifier.ws_broadcast_fn = ws_broadcast_fn

    async def raise_alert(
        self,
        alert_type: str,
        message: str,
        state: str = "FIRE_CONFIRMED",
        source: str = "unknown",
        snapshot_path: Optional[str] = None,
        is_test: bool = False
    ) -> dict:
        """
        Creates, stores, and dispatches an alert.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        alert_record = {
            "ts": now_iso,
            "type": alert_type,
            "message": message,
            "state": state,
            "source": source,
            "snapshot_path": snapshot_path,
            "acknowledged": 0,
            "is_test": 1 if is_test else 0
        }

        # Store in SQLite
        alert_id = db.insert_alert(alert_record)
        alert_record["id"] = alert_id

        # Update counter if not a test alert
        if not is_test:
            db.increment_counter("alerts_generated", 1)

        # Dispatch across all active notifiers
        for notifier in self.notifiers:
            if notifier.enabled:
                try:
                    await notifier.send(alert_record)
                except Exception as e:
                    logger.error(f"Notifier {notifier.name} error: {e}")

        logger.info(f"Alert #{alert_id} [{alert_type}] dispatched: {message}")
        return alert_record

    async def raise_test_alert(self) -> dict:
        """Fires a safe, clearly-labeled test alert for sound and notification testing."""
        return await self.raise_alert(
            alert_type="test",
            message="TEST ALERT: Verification of browser siren and dashboard notification system.",
            state="FIRE_CONFIRMED",
            source="manual_test",
            snapshot_path=None,
            is_test=True
        )

alert_manager = AlertManager()
