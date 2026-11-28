"""Request bodies for POST /video/generate (upstream video plan / storyboard)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from video_generation.modes import normalize_video_mode, video_mode_label


class VideoGenerateRequest(BaseModel):
    """Shot-level generation request from the orchestration service."""

    mode: str = Field(..., description="t2v | i2v (or text_to_video | image_to_video)")
    engine_type: str = Field(default="wan", description="wan | ltx | cogvideox | hunyuan")
    prompt: str
    duration_seconds: float = Field(default=5.0, gt=0, le=60)

    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    seed: int | None = None

    # Correlation (echoed on job status / logs)
    shot_id: str | None = None
    request_id: str | None = None
    storyboard_id: str | None = None
    scene_id: str | None = None

    # Upstream planning fields (stored on job; prompt is already final for inference)
    image_prompt: str | None = None
    motion_prompt: str | None = None
    camera_prompt: str | None = None

    # I2V without multipart: provide one of these
    image_url: str | None = None
    image_base64: str | None = None

    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @field_validator("mode", "engine_type", mode="before")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        return (value or "").strip()

    @field_validator("duration_seconds", mode="before")
    @classmethod
    def coerce_duration(cls, value: Any) -> Any:
        if value is None or value == "":
            return 5.0
        if isinstance(value, str):
            return float(value.strip())
        return value

    @field_validator("width", "height", "fps", "seed", mode="before")
    @classmethod
    def coerce_optional_int(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return int(float(value.strip()))
        return value

    @model_validator(mode="after")
    def _i2v_needs_image_source(self) -> VideoGenerateRequest:
        from video_generation.modes import normalize_video_mode

        try:
            video_mode = normalize_video_mode(self.mode)
        except ValueError:
            return self
        if video_mode != "image_to_video":
            return self
        if not self.image_url and not self.image_base64:
            # Multipart file may still be supplied by the HTTP layer.
            return self
        return self

    def correlation_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        try:
            video_mode = normalize_video_mode(self.mode)
            meta["mode"] = video_mode
            meta["mode_label"] = video_mode_label(video_mode)
        except ValueError:
            pass
        for key in (
            "shot_id",
            "request_id",
            "storyboard_id",
            "scene_id",
            "image_prompt",
            "motion_prompt",
            "camera_prompt",
        ):
            val = getattr(self, key, None)
            if val is not None and str(val).strip():
                meta[key] = str(val).strip()
        if self.extra:
            meta["extra"] = self.extra
        return meta