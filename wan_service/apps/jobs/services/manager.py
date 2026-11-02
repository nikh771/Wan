"""In-memory job queue with single-GPU concurrency."""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wan_service import config as service_config
from wan_service.apps.jobs.services.engine_dispatch import assert_engine_available, run_engine
from wan_service.apps.jobs.services.progress_parser import parse_wan_generate_line
from wan_service.engines.types import EngineType, GenerationJobParams
from wan_service.utils.ffmpeg import trim_video_to_duration, validate_output_mp4

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"
    progress: int | None = None
    stage: str = "queued"
    message: str | None = None
    inference_step: int | None = None
    inference_total_steps: int | None = None
    error: str | None = None
    prompt: str = ""
    engine_type: EngineType = "wan"
    engine_requested: EngineType = "wan"
    engine_used: str | None = None
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int = 24
    generation_metadata: dict | None = None
    duration_seconds: float | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    video_path: Path | None = None
    work_dir: Path | None = None
    recent_logs: deque[str] = field(default_factory=lambda: deque(maxlen=40))

    def to_dict(self, *, base_url: str = "") -> dict[str, Any]:
        video_url = None
        progress_url = None
        base = base_url.rstrip("/")
        if base:
            progress_url = f"{base}/jobs/{self.job_id}/progress"
        if self.status == "completed" and self.video_path:
            video_url = f"{base}/jobs/{self.job_id}/video"
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "inference_step": self.inference_step,
            "inference_total_steps": self.inference_total_steps,
            "recent_logs": list(self.recent_logs),
            "error": self.error,
            "engine_type": self.engine_type,
            "engine_requested": self.engine_requested,
            "engine_used": self.engine_used,
            "generation_metadata": self.generation_metadata,
            "video_url": video_url,
            "progress_url": progress_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_progress_dict(self, *, base_url: str = "") -> dict[str, Any]:
        """Minimal payload for polling a single job by id."""
        video_url = None
        base = base_url.rstrip("/")
        if self.status == "completed" and self.video_path and base:
            video_url = f"{base}/jobs/{self.job_id}/video"
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "inference_step": self.inference_step,
            "inference_total_steps": self.inference_total_steps,
            "error": self.error,
            "video_url": video_url,
            "done": self.status in ("completed", "failed"),
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._active: str | None = None
        service_config.JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        *,
        prompt: str,
        duration_seconds: float | None,
        engine_type: EngineType = "wan",
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> JobRecord:
        with self._lock:
            if len(self._queue) + (1 if self._active else 0) >= service_config.MAX_QUEUE_SIZE:
                raise RuntimeError("Job queue is full. Try again later.")
            job_id = f"{engine_type}_{uuid.uuid4().hex[:16]}"
            work_dir = service_config.JOBS_DIR / job_id
            work_dir.mkdir(parents=True, exist_ok=True)
            record = JobRecord(
                job_id=job_id,
                status="pending",
                stage="queued",
                message="Waiting in queue…",
                progress=0,
                prompt=prompt,
                engine_type=engine_type,
                engine_requested=engine_type,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                fps=fps or 24,
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

    def _apply_log_line(self, record: JobRecord, line: str) -> None:
        record.recent_logs.append(line)
        parsed = parse_wan_generate_line(line)
        if parsed is not None:
            record.stage = parsed.stage
            record.message = parsed.message
            record.progress = parsed.progress
            record.inference_step = parsed.current_step
            record.inference_total_steps = parsed.total_steps
        elif line.startswith("[storyboard]"):
            record.stage = "rendering"
            record.message = line.replace("[storyboard]", "").strip() or "Rendering…"
            record.progress = max(record.progress or 0, 40)
        elif line.startswith("[ltx]") or line.startswith("[wan_service]"):
            record.stage = "booting"
            if "Waiting for warm worker" in line:
                record.message = "Waiting for startup model load (avoiding duplicate GPU load)…"
                record.progress = max(record.progress or 0, 10)
            elif "Warm worker" in line:
                record.message = "Running on warm worker (models already loaded)…"
                record.progress = max(record.progress or 0, 30)
            elif "LTX" in line:
                record.message = "Starting LTX worker…"
                record.progress = max(record.progress or 0, 18)
            else:
                record.message = "Starting inference subprocess…"
                record.progress = max(record.progress or 0, 12)
        record.updated_at = _utc_now()

    def _run_job(self, job_id: str) -> None:
        record = self._jobs.get(job_id)
        if not record or not record.work_dir:
            return
        with self._lock:
            record.status = "running"
            record.stage = "booting"
            record.message = "Preparing job…"
            record.progress = 8
            record.updated_at = _utc_now()
        image_path = record.work_dir / "input.png"
        raw_out = record.work_dir / "output_raw.mp4"
        final_out = record.work_dir / "output.mp4"

        def on_log_line(line: str) -> None:
            with self._lock:
                self._apply_log_line(record, line)

        try:
            assert_engine_available(record.engine_type)
            if not image_path.exists():
                raise FileNotFoundError("Input image missing for job.")
            with self._lock:
                record.progress = max(record.progress or 0, 15)
                record.message = f"Running {record.engine_type} engine…"
                record.updated_at = _utc_now()
            params = GenerationJobParams(
                job_id=job_id,
                image_path=image_path,
                prompt=record.prompt,
                output_mp4=raw_out,
                duration_seconds=record.duration_seconds,
                negative_prompt=record.negative_prompt,
                width=record.width,
                height=record.height,
                fps=record.fps,
                on_log_line=on_log_line,
            )
            meta = run_engine(params, record.engine_type)
            with self._lock:
                record.engine_used = str(meta.get("engine_used", record.engine_type))
                record.generation_metadata = meta
            with self._lock:
                record.stage = "postprocessing"
                record.message = "Finalizing video (trim / validate)…"
                record.progress = 92
                record.updated_at = _utc_now()
            if record.duration_seconds and record.duration_seconds > 0:
                trim_video_to_duration(raw_out, final_out, float(record.duration_seconds))
            else:
                shutil.copy2(raw_out, final_out)
            validate_output_mp4(final_out)
            with self._lock:
                record.video_path = final_out
                record.status = "completed"
                record.stage = "completed"
                record.message = "Video ready"
                record.progress = 100
                record.error = None
                record.updated_at = _utc_now()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            with self._lock:
                record.status = "failed"
                record.stage = "failed"
                record.message = "Job failed"
                record.error = str(exc)[:2000]
                record.progress = None
                record.updated_at = _utc_now()
            self._cleanup_failed(record)

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
