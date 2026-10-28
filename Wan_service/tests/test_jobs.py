from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import wait_for_job_status


def test_jobs_require_auth(client: TestClient, panel_png: tuple[str, bytes, str]) -> None:
    name, data, mime = panel_png
    resp = client.post(
        "/jobs",
        data={"prompt": "anime scene", "duration_seconds": "5"},
        files={"image": (name, data, mime)},
    )
    assert resp.status_code == 401


def test_create_job_rejected_when_inference_not_ready(
    client: TestClient,
    auth_headers: dict[str, str],
    panel_png: tuple[str, bytes, str],
    jobs_dir: Path,
) -> None:
    before = list(jobs_dir.iterdir())
    name, data, mime = panel_png
    resp = client.post(
        "/jobs",
        data={"prompt": "anime rooftop dusk", "duration_seconds": "5"},
        files={"image": (name, data, mime)},
        headers=auth_headers,
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "mac_dev" in detail or "CUDA" in detail or "MPS" in detail
    assert list(jobs_dir.iterdir()) == before


def test_create_job_pending_and_video_not_ready(
    client: TestClient,
    auth_headers: dict[str, str],
    panel_png: tuple[str, bytes, str],
    inference_ready: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _block_inference(**kwargs: object) -> None:
        raise RuntimeError("inference blocked in test")

    monkeypatch.setattr(
        "wan_service.apps.jobs.services.manager.run_wan_inference",
        _block_inference,
    )

    name, data, mime = panel_png
    resp = client.post(
        "/jobs",
        data={"prompt": "anime rooftop dusk", "duration_seconds": "5"},
        files={"image": (name, data, mime)},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"].startswith("wan_")
    assert body["status"] in {"pending", "running", "failed"}

    job_id = body["job_id"]
    final = wait_for_job_status(client, job_id, auth_headers, want={"failed"})
    assert final["status"] == "failed"
    assert final.get("error")

    video_resp = client.get(f"/jobs/{job_id}/video", headers=auth_headers)
    assert video_resp.status_code == 404


def test_get_job_not_found(client: TestClient, auth_headers: dict[str, str]) -> None:
    resp = client.get("/jobs/wan_does_not_exist", headers=auth_headers)
    assert resp.status_code == 404


def test_create_job_rejects_empty_prompt(
    client: TestClient,
    auth_headers: dict[str, str],
    panel_png: tuple[str, bytes, str],
    inference_ready: None,
) -> None:
    name, data, mime = panel_png
    resp = client.post(
        "/jobs",
        data={"prompt": "   ", "duration_seconds": "5"},
        files={"image": (name, data, mime)},
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_select_wan_size_landscape_and_portrait(tmp_path: Path) -> None:
    from PIL import Image

    from wan_service.apps.jobs.services.inference import select_wan_size

    landscape = tmp_path / "land.png"
    Image.new("RGB", (200, 100), color=(1, 2, 3)).save(landscape)
    assert select_wan_size(landscape) == "1280*704"

    portrait = tmp_path / "port.png"
    Image.new("RGB", (100, 200), color=(1, 2, 3)).save(portrait)
    assert select_wan_size(portrait) == "704*1280"
