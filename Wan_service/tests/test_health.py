from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_mac_dev_reports_separate_cuda_and_mps(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "Wan2.2-TI2V-5B"
    assert body["runtime_mode"] == "mac_dev"
    assert body["api_available"] is True
    assert body["service_ready"] is True
    assert body["inference_ready"] is False
    assert "API/test-only" in (body["inference_unavailable_reason"] or "")
    assert body["upstream_ti2v5b_mps_supported"] is False
    assert "ffmpeg_ready" in body
    assert "cuda_available" in body
    assert "mps_available" in body
    assert body["gpu_available"] is body["cuda_available"]
    assert body["status"] == "degraded"
