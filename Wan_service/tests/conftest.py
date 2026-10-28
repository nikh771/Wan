"""Pytest fixtures for Wan_service (no GPU, no model download)."""

from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

# Configure worker paths before wan_service modules are imported.
os.environ.setdefault("WAN_WORKER_API_KEY", "pytest-api-key")
os.environ.setdefault("WAN_RUNTIME_MODE", "mac_dev")
os.environ.setdefault("WAN_JOBS_DIR", tempfile.mkdtemp(prefix="wan_service_jobs_"))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wan_service import config as service_config
from wan_service.apps.jobs.services import manager as manager_mod
from wan_service import main as main_mod
from wan_service.apps.jobs.services.manager import JobManager


@pytest.fixture()
def jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "wan_jobs"
    path.mkdir()
    monkeypatch.setattr(service_config, "JOBS_DIR", path)
    monkeypatch.setattr(service_config, "WORKER_API_KEY", "pytest-api-key")
    mgr = JobManager()
    monkeypatch.setattr(manager_mod, "job_manager", mgr)
    import wan_service.apps.jobs.api.routes as routes_mod

    monkeypatch.setattr(routes_mod, "job_manager", mgr)
    return path


@pytest.fixture()
def client(jobs_dir: Path) -> TestClient:
    return TestClient(main_mod.app)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer pytest-api-key"}


@pytest.fixture()
def panel_png() -> tuple[str, bytes, str]:
    img = Image.new("RGB", (128, 192), color=(90, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ("panel.png", buf.getvalue(), "image/png")


@pytest.fixture()
def inference_ready(monkeypatch: pytest.MonkeyPatch):
    """Allow job routes and worker to proceed without CUDA or checkpoints."""

    from wan_service.core.runtime import (
        CudaProbe,
        InferenceReadiness,
        MpsProbe,
    )

    fake = InferenceReadiness(
        ready=True,
        detail="test override",
        public_reason="test override",
        runtime_mode="linux_cuda",
        checkpoint_ready=True,
        generate_script_ready=True,
        ffmpeg_ready=True,
        wan_python_ready=True,
        inference_dependencies_ready=True,
        cuda=CudaProbe(
            available=True,
            detail="test cuda",
            vram_gb=24.0,
            vram_sufficient=True,
        ),
        mps=MpsProbe(available=False, detail="test"),
    )

    monkeypatch.setattr(
        "wan_service.apps.jobs.api.routes.evaluate_inference_readiness",
        lambda: fake,
    )
    monkeypatch.setattr(
        "wan_service.apps.jobs.services.manager.assert_inference_available",
        lambda: None,
    )


def wait_for_job_status(
    client: TestClient,
    job_id: str,
    headers: dict[str, str],
    *,
    want: set[str],
    timeout: float = 5.0,
) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last.get("status") in want:
            return last
        time.sleep(0.05)
    return last
