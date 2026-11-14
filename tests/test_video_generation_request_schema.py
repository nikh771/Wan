"""Upstream JSON shot payload validation."""

from __future__ import annotations

from video_generation.schemas import VideoGenerateRequest


def test_shot_payload_minimal_t2v_cogvideox() -> None:
    body = VideoGenerateRequest.model_validate(
        {
            "mode": "t2v",
            "engine_type": "cogvideox",
            "prompt": "A cinematic action shot.",
            "duration_seconds": 5.0,
            "shot_id": "shot_001",
            "request_id": "vplan_abc",
            "negative_prompt": "static pose",
            "width": 720,
            "height": 480,
            "fps": 8,
            "seed": 12345,
        }
    )
    meta = body.correlation_metadata()
    assert meta["shot_id"] == "shot_001"
    assert meta["request_id"] == "vplan_abc"
    assert body.engine_type == "cogvideox"


def test_planning_fields_stored_in_metadata() -> None:
    body = VideoGenerateRequest.model_validate(
        {
            "mode": "t2v",
            "engine_type": "ltx",
            "prompt": "main",
            "motion_prompt": "motion",
            "camera_prompt": "camera",
            "image_prompt": "keyframe",
            "storyboard_id": "sb_x",
            "scene_id": "scene_001",
        }
    )
    meta = body.correlation_metadata()
    assert meta["motion_prompt"] == "motion"
    assert meta["storyboard_id"] == "sb_x"