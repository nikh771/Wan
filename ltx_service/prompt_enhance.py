"""Pre-job LTX prompt enhancement via Ollama (CPU-friendly) + cinematic fallback.

Runs in the API process before the LTX GPU worker starts. Uses keep_alive=0 so
Ollama unloads the model and does not hold VRAM for generation.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Mode = Literal["text_to_video", "image_to_video"]

DEFAULT_NEGATIVE = (
    "still image, frozen frame, no motion, static pose, slideshow, nearly still, "
    "subtle motion, micro movement, idle, posed photograph, "
    "worst quality, inconsistent motion, blurry, jittery, distorted, watermark"
)

T2V_SYSTEM = """You rewrite prompts for text-to-video. Output ONE paragraph only.
REQUIRE large, obvious motion a viewer notices in the first second — big head turns, walking, clear camera pans/push-ins.
FORBIDDEN words: subtle, slight, gentle, soft, micro, tiny, barely, slowly drifting.
Name concrete actions with angles/distances (e.g. "turns 90 degrees", "walks toward camera", "camera pans left across the room").
Keep under 100 words. No quotes, labels, or preamble."""

I2V_SYSTEM = """You rewrite prompts for image-to-video. The first frame is a still photo; later frames MUST clearly change.
REQUIRE large, obvious motion visible immediately: big head/body turns, walking, arm gestures, strong camera move (push-in, orbit, pan).
FORBIDDEN words: subtle, slight, gentle, soft, micro, tiny, barely, idle.
Keep identity/appearance consistent with the still. State timing in seconds with BIG actions
(e.g. "over 3 seconds the subject turns a full 90 degrees toward camera while the camera pushes in").
Keep under 100 words. Output the enhanced prompt only — no quotes, labels, or preamble."""

_WEAK_MOTION = re.compile(
    r"\b(subtle|slight|gently|gentle|softly|soft|micro|tiny|barely|faint|idle)\b",
    flags=re.I,
)


@dataclass(frozen=True)
class PromptEnhanceResult:
    prompt: str
    negative_prompt: str
    source: str  # "ollama" | "fallback" | "passthrough"
    original_prompt: str


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(enhanced prompt|prompt)\s*:\s*", "", text, flags=re.I)
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    return " ".join(text.split())


def _force_strong_motion(prompt: str, *, mode: Mode, duration_seconds: float) -> str:
    """Strip weak language and append an explicit large-motion directive."""
    text = _WEAK_MOTION.sub("clear", _strip_wrapping(prompt))
    seconds = max(2, min(8, int(round(duration_seconds))))
    if mode == "image_to_video":
        mandate = (
            f" Over {seconds} seconds the subject performs a large unmistakable action "
            f"(full head/body turn toward camera or clear walk), clothing and hair swing "
            f"with the movement, and the camera pushes in strongly; continuous high-energy "
            f"motion from the first frame, never nearly still."
        )
    else:
        mandate = (
            f" Over {seconds} seconds show large unmistakable motion with a strong camera "
            f"move; continuous high-energy action, never nearly still."
        )
    if "unmistakable" in text.lower() and "never nearly still" in text.lower():
        return text
    return f"{text.rstrip('.')}." + mandate


def cinematic_fallback(prompt: str, *, mode: Mode, duration_seconds: float) -> str:
    """Deterministic large-motion rewrite when Ollama is unavailable."""
    base = _strip_wrapping(prompt) or "The subject begins moving"
    seconds = max(2, min(8, int(round(duration_seconds))))
    if mode == "image_to_video":
        return _force_strong_motion(
            f"{base}. The subject turns a full 90 degrees toward the camera over "
            f"{seconds} seconds while taking a clear step forward, arms shift, hair and "
            f"fabric swing, camera pushes in hard with continuous motion",
            mode=mode,
            duration_seconds=duration_seconds,
        )
    return _force_strong_motion(
        f"{base}. Wide camera pan while the subject walks across frame over {seconds} "
        f"seconds with clear continuous motion",
        mode=mode,
        duration_seconds=duration_seconds,
    )


def _ollama_generate(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    timeout_sec: float,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.55,
            "num_predict": 200,
            "num_gpu": 0,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = (body.get("response") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty prompt")
    return _strip_wrapping(text)


def ollama_available(base_url: str, timeout_sec: float = 2.0) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001
        return False


def enhance_ltx_prompt(
    prompt: str,
    *,
    mode: Mode = "image_to_video",
    duration_seconds: float = 5.0,
    negative_prompt: str | None = None,
    enabled: bool = True,
    ollama_base_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.2:3b",
    ollama_timeout_sec: float = 60.0,
) -> PromptEnhanceResult:
    original = (prompt or "").strip()
    neg = (negative_prompt or "").strip() or DEFAULT_NEGATIVE
    if not enabled:
        return PromptEnhanceResult(
            prompt=_force_strong_motion(
                original, mode=mode, duration_seconds=duration_seconds
            ),
            negative_prompt=neg,
            source="passthrough",
            original_prompt=original,
        )

    system = I2V_SYSTEM if mode == "image_to_video" else T2V_SYSTEM
    user = (
        f"Mode: {mode}\n"
        f"Duration seconds: {duration_seconds}\n"
        f"User prompt: {original}\n"
        f"Remember: LARGE obvious motion only. No subtle/gentle language."
    )

    if ollama_available(ollama_base_url):
        try:
            enhanced = _ollama_generate(
                base_url=ollama_base_url,
                model=ollama_model,
                system=system,
                user=user,
                timeout_sec=ollama_timeout_sec,
            )
            if len(enhanced) < 8:
                raise RuntimeError("enhanced prompt too short")
            enhanced = _force_strong_motion(
                enhanced, mode=mode, duration_seconds=duration_seconds
            )
            logger.info(
                "LTX prompt enhanced via Ollama (%s): %s -> %s",
                ollama_model,
                original[:80],
                enhanced[:160],
            )
            return PromptEnhanceResult(
                prompt=enhanced,
                negative_prompt=neg,
                source="ollama",
                original_prompt=original,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama prompt enhance failed (%s); using fallback", exc)

    enhanced = cinematic_fallback(
        original, mode=mode, duration_seconds=duration_seconds
    )
    logger.info(
        "LTX prompt enhanced via fallback: %s -> %s",
        original[:80],
        enhanced[:160],
    )
    return PromptEnhanceResult(
        prompt=enhanced,
        negative_prompt=neg,
        source="fallback",
        original_prompt=original,
    )
