from abc import ABC, abstractmethod

class BaseNotifier(ABC):
    """Abstract base class for alert notification channels."""
    name: str = "base"
    enabled: bool = False

    @abstractmethod
    async def send(self, alert: dict) -> bool:
        """Dispatches an alert dictionary."""
        pass
