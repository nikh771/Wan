"""Fetch or decode input images for i2v jobs."""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

import httpx
from fastapi import HTTPException

_DATA_URL_RE = re.compile(r"^data:image/(?P<fmt>png|jpeg|jpg|webp);base64,(?P<data>.+)$", re.I)


def _decode_base64_image(raw: str) -> tuple[bytes, str]:
    text = raw.strip()
    content_type = "image/png"
    if text.startswith("data:"):
        match = _DATA_URL_RE.match(text)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid image_base64 data URL")
        content_type = f"image/{match.group('fmt').lower()}"
        if content_type == "image/jpg":
            content_type = "image/jpeg"
        text = match.group("data")
    try:
        data = base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image_base64") from exc
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Empty image_base64")
    return data, content_type


async def fetch_image_url(url: str, *, max_bytes: int) -> tuple[bytes, str]:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="image_url must be http(s)")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            resp = await client.get(url)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"image_url fetch failed: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"image_url returned HTTP {resp.status_code}")
    data = resp.content
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Image too large")
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Empty image from image_url")
    content_type = (resp.headers.get("content-type") or "image/png").split(";")[0].strip().lower()
    if content_type not in ("image/png", "image/jpeg", "image/webp"):
        content_type = "image/png"
    return data, content_type


def decode_base64_image(raw: str) -> tuple[bytes, str]:
    return _decode_base64_image(raw)


def write_job_input_image(work_dir: Path, data: bytes, content_type: str) -> None:
    import shutil

    suffix = ".png" if "png" in content_type else ".jpg" if "jpeg" in content_type else ".webp"
    input_path = work_dir / f"input{suffix}"
    input_path.write_bytes(data)
    target = work_dir / "input.png"
    if input_path != target:
        shutil.copy2(input_path, target)