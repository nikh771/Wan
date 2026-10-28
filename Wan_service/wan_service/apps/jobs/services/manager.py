"""In-memory job queue with single-GPU concurrency."""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wan_service import config as service_config
from wan_service.apps.jobs.services.inference import run_wan_inference
from wan_service.core.runtime import assert_inference_available
from wan_service.utils.ffmpeg import trim_video_to_duration, validate_output_mp4

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"
    progress: int | None = None
    error: str | None = None
    prompt: str = ""
    duration_seconds: float | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    video_path: Path | None = None
    work_dir: Path | None = None

    def to_dict(self, *, base_url: str = "") -> dict[str, Any]:
        video_url = None
        if self.status == "completed" and self.video_path:
            video_url = f"{base_url.rstrip('/')}/jobs/{self.job_id}/video"
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "video_url": video_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._active: str | None = None
        service_config.JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def create_job(self, *, prompt: str, duration_seconds: float | None) -> JobRecord:
        with self._lock:
            if len(self._queue) + (1 if self._active else 0) >= service_config.MAX_QUEUE_SIZE:
                raise RuntimeError("Job queue is full. Try again later.")
            job_id = f"wan_{uuid.uuid4().hex[:16]}"
            work_dir = service_config.JOBS_DIR / job_id
            work_dir.mkdir(parents=True, exist_ok=True)
            record = JobRecord(
                job_id=job_id,
                status="pending",
                prompt=prompt,
                duration_seconds=duration_seconds,
                work_dir=work_dir,
            )
            self._jobs[job_id] = record
        return record

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._queue.append(job_id)
        threading.Thread(target=self._drain_queue, daemon=True).start()

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def _drain_queue(self) -> None:
        with self._lock:
            if self._active or not self._queue:
                return
            job_id = self._queue.pop(0)
            self._active = job_id
        try:
            self._run_job(job_id)
        finally:
            with self._lock:
                self._active = None
            self._drain_queue()

    def _run_job(self, job_id: str) -> None:
        record = self._jobs.get(job_id)
        if not record or not record.work_dir:
            return
        record.status = "running"
        record.progress = 10
        record.updated_at = _utc_now()
        image_path = record.work_dir / "input.png"
        raw_out = record.work_dir / "output_raw.mp4"
        final_out = record.work_dir / "output.mp4"
        try:
            assert_inference_available()
            if not image_path.exists():
                raise FileNotFoundError("Input image missing for job.")
            record.progress = 20
            run_wan_inference(image_path=image_path, prompt=record.prompt, output_mp4=raw_out)
            record.progress = 85
            if record.duration_seconds and record.duration_seconds > 0:
                trim_video_to_duration(raw_out, final_out, float(record.duration_seconds))
            else:
                shutil.copy2(raw_out, final_out)
            validate_output_mp4(final_out)
            record.video_path = final_out
            record.status = "completed"
            record.progress = 100
            record.error = None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            record.status = "failed"
            record.error = str(exc)[:2000]
            record.progress = None
            self._cleanup_failed(record)
        record.updated_at = _utc_now()

    @staticmethod
    def _cleanup_failed(record: JobRecord) -> None:
        if not record.work_dir:
            return
        for name in ("input.png", "output_raw.mp4"):
            path = record.work_dir / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass


job_manager = JobManager()
