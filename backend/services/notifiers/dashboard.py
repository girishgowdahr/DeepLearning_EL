import logging
from backend.services.notifiers.base import BaseNotifier

logger = logging.getLogger("forestguard.notifier.dashboard")

class DashboardNotifier(BaseNotifier):
    """
    Active Dashboard Notifier. Broadcasts alerts over WebSocket
    to connected React clients and triggers the Web Audio siren.
    """
    name = "dashboard"
    enabled = True

    def __init__(self, ws_broadcast_fn=None):
        self.ws_broadcast_fn = ws_broadcast_fn

    async def send(self, alert: dict) -> bool:
        logger.info(f"DashboardNotifier dispatching alert: {alert.get('type')} - {alert.get('message')}")
        if self.ws_broadcast_fn:
            try:
                await self.ws_broadcast_fn({
                    "type": "alert",
                    "payload": alert
                })
                return True
            except Exception as e:
                logger.error(f"Failed to broadcast alert over WS: {e}")
        return False
