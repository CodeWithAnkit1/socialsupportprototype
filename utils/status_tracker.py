from typing import Optional
from utils.logger import get_logger

logger = get_logger("status_tracker")

class StatusTracker:
    _current_status: Optional[str] = None
    
    @classmethod
    def set_status(cls, status: str):
        cls._current_status = status
        logger.info(f"Status updated: {status}")
    
    @classmethod
    def get_status(cls) -> Optional[str]:
        return cls._current_status