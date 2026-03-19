import time
import logging

logger = logging.getLogger(__name__)

class HealthMonitor:
    def __init__(self):
        self.last_heartbeat = time.time()
        self.error_count = 0
        self.start_time = time.time()

    def heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = time.time()

    def record_error(self):
        """Increment the error count"""
        self.error_count += 1
        logger.warning(f"[HEALTH] Error recorded. Current count: {self.error_count}")

    def is_healthy(self):
        """
        Check if the system is healthy based on:
        1. Last heartbeat within 60 seconds
        2. Error count below 10
        """
        now = time.time()
        heartbeat_ok = (now - self.last_heartbeat) < 60
        errors_ok = self.error_count < 10
        
        healthy = heartbeat_ok and errors_ok
        
        if not healthy:
            if not heartbeat_ok:
                logger.error(f"[HEALTH] System Unhealthy: Heartbeat stale ({now - self.last_heartbeat:.1f}s)")
            if not errors_ok:
                logger.error(f"[HEALTH] System Unhealthy: Too many errors ({self.error_count})")
                
        return healthy

    def get_status(self):
        """Get detailed health status"""
        return {
            "healthy": self.is_healthy(),
            "uptime": time.time() - self.start_time,
            "last_heartbeat_ago": time.time() - self.last_heartbeat,
            "error_count": self.error_count
        }
