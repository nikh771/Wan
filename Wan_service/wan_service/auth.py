"""Re-export auth dependency (prefer wan_service.utils.auth)."""

from wan_service.utils.auth import require_worker_auth

__all__ = ["require_worker_auth"]
