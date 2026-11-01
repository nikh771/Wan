"""Manage a long-lived Wan TI2V subprocess (models loaded once)."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wan_service.apps.jobs.services.inference import (
    _inference_env,
    resolve_gpu_only_inference,
)
from wan_service.core.config import get_settings

logger = logging.getLogger(__name__)

_WORKER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "inference_worker_process.py"
)


class PersistentInferenceWorker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._init_error: str | None = None
        self._job_done = threading.Event()
        self._job_error: str | None = None
        self._active_job_id: str | None = None
        self._on_log_line: Callable[[str], None] | None = None
        self._started = False

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._process is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    @property
    def is_loading(self) -> bool:
        return (
            self._started
            and not self.is_ready
            and self._init_error is None
            and self._process is not None
            and self._process.poll() is None
        )

    def wait_until_ready(self, *, timeout_sec: float) -> None:
        """Block until warm worker is ready or failed (avoids a second GPU model load)."""
        if self.is_ready:
            return
        if self._init_error:
            raise RuntimeError(self._init_error)
        if not self._started:
            raise RuntimeError("Persistent inference worker was not started")
        if not self._ready.wait(timeout=timeout_sec):
            raise RuntimeError(
                f"Timed out waiting for persistent worker ({timeout_sec:.0f}s). "
                "Check /health persistent_worker_ready and retry."
            )
        if self._init_error:
            raise RuntimeError(self._init_error)
        if not self.is_ready:
            raise RuntimeError("Persistent inference worker is not ready")

    def start(self, *, load_timeout_sec: float) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        cfg = get_settings()
        if cfg.resolved_inference_backend() != "linux_cuda":
            self._init_error = "Persistent worker requires linux_cuda"
            return
        if not _WORKER_SCRIPT.is_file():
            self._init_error = f"Worker script missing: {_WORKER_SCRIPT}"
            return
        if not cfg.wan_ckpt_dir.is_dir():
            self._init_error = f"Checkpoint dir missing: {cfg.wan_ckpt_dir}"
            return

        gpu_only, gpu_reason = resolve_gpu_only_inference()
        repo = cfg.wan_repo_dir
        env = _inference_env(repo=repo, gpu_only=gpu_only)

        cmd = [cfg.WAN_PYTHON, "-u", str(_WORKER_SCRIPT)]
        logger.info(
            "Starting persistent inference worker (gpu_only=%s, %s)",
            gpu_only,
            gpu_reason,
        )
        self._process = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        assert self._process.stdin is not None
        init_payload = {
            "type": "init",
            "ckpt_dir": str(cfg.wan_ckpt_dir.resolve()),
            "gpu_only": gpu_only,
            "convert_model_dtype": True,
            "device_id": 0,
        }
        self._process.stdin.write(json.dumps(init_payload) + "\n")
        self._process.stdin.flush()

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

        if not self._ready.wait(timeout=load_timeout_sec):
            self._init_error = (
                f"Worker did not become ready within {load_timeout_sec:.0f}s"
            )
            self.stop()
            return

        if self._init_error:
            self.stop()

    def stop(self) -> None:
        with self._lock:
            proc = self._process
            self._process = None
        if proc is None:
            return
        try:
            if proc.stdin and proc.poll() is None:
                proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                proc.stdin.flush()
        except OSError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._ready.clear()
        self._started = False

    def _read_stdout(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self._forward_log(line)
                    continue
                self._handle_event(event)
            else:
                self._forward_log(line)

        if proc.poll() is not None and proc.returncode not in (0, None):
            if not self._ready.is_set():
                self._init_error = f"Worker exited during init (code {proc.returncode})"
            elif self._active_job_id:
                self._job_error = f"Worker exited unexpectedly (code {proc.returncode})"
                self._job_done.set()

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "ready":
            logger.info("Persistent inference worker ready (models loaded)")
            self._ready.set()
            return
        if kind == "done":
            if event.get("job_id") == self._active_job_id:
                self._job_done.set()
            return
        if kind == "error":
            msg = event.get("message") or "Worker error"
            if not self._ready.is_set():
                self._init_error = msg
                self._ready.set()
            elif event.get("job_id") == self._active_job_id:
                self._job_error = msg
                self._job_done.set()
            return
        if kind == "log":
            self._forward_log(str(event.get("line", "")))
            return

    def _forward_log(self, line: str) -> None:
        if not line:
            return
        logger.info("[wan worker] %s", line)
        if self._on_log_line:
            self._on_log_line(line)

    def run_job(
        self,
        *,
        job_id: str,
        image_path: Path,
        prompt: str,
        output_mp4: Path,
        size: str,
        sample_steps: int | None,
        frame_num: int | None,
        on_log_line: Callable[[str], None] | None = None,
        job_timeout_sec: float = 3600.0,
    ) -> None:
        if not self.is_ready:
            raise RuntimeError(
                self._init_error or "Persistent inference worker is not ready"
            )
        proc = self._process
        if proc is None or proc.stdin is None:
            raise RuntimeError("Worker process is not running")

        gpu_only, _ = resolve_gpu_only_inference()
        payload: dict[str, Any] = {
            "type": "generate",
            "job_id": job_id,
            "image_path": str(image_path.resolve()),
            "prompt": prompt,
            "output_mp4": str(output_mp4.resolve()),
            "size": size,
            "gpu_only": gpu_only,
        }
        if sample_steps is not None:
            payload["sample_steps"] = sample_steps
        if frame_num is not None:
            payload["frame_num"] = frame_num

        with self._lock:
            self._job_done.clear()
            self._job_error = None
            self._active_job_id = job_id
            self._on_log_line = on_log_line

        if on_log_line:
            on_log_line(
                f"[wan_service] Warm worker job {job_id} (models already loaded)"
            )

        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        if not self._job_done.wait(timeout=job_timeout_sec):
            raise RuntimeError(f"Inference timed out after {job_timeout_sec:.0f}s")

        if self._job_error:
            raise RuntimeError(self._job_error)

        if not output_mp4.exists() or output_mp4.stat().st_size < 1000:
            raise RuntimeError("Worker did not produce a valid MP4 file.")


_worker: PersistentInferenceWorker | None = None
_worker_lock = threading.Lock()


def get_persistent_worker() -> PersistentInferenceWorker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = PersistentInferenceWorker()
        return _worker


def shutdown_persistent_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None
