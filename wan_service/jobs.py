"""Re-export job manager (prefer wan_service.apps.jobs.services.manager)."""

from wan_service.apps.jobs.services.manager import JobManager, JobRecord, job_manager

__all__ = ["JobManager", "JobRecord", "job_manager"]
