"""Map Wan generate.py subprocess log lines to job progress for the API/UI."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LOG_PREFIX = re.compile(r"^\[[^\]]+\]\s+\w+:\s*")
_TQDM_RE = re.compile(r"(\d+)\s*%.*?(\d+)\s*/\s*(\d+)")
_TQDM_FRACTION_RE = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*\[")


@dataclass(frozen=True)
class InferenceProgress:
    stage: str
    message: str
    progress: int
    current_step: int | None = None
    total_steps: int | None = None


def _clean_line(line: str) -> str:
    text = line.strip()
    match = _LOG_PREFIX.match(text)
    if match:
        return text[match.end() :].strip()
    return text


def _parse_tqdm(text: str) -> tuple[int, int, int] | None:
    match = _TQDM_RE.search(text)
    if match:
        pct = int(match.group(1))
        current = int(match.group(2))
        total = max(int(match.group(3)), 1)
        return pct, current, total
    frac = _TQDM_FRACTION_RE.search(text)
    if frac:
        current = int(frac.group(1))
        total = max(int(frac.group(2)), 1)
        pct = int(100 * current / total)
        return pct, current, total
    return None


def parse_wan_generate_line(line: str) -> InferenceProgress | None:
    """Return progress update if the log line is recognized."""
    text = _clean_line(line)
    if not text:
        return None

    lowered = text.lower()

    if "creating wan" in lowered or "creating wanmodel" in lowered:
        return InferenceProgress(
            stage="loading_models",
            message="Loading T5, VAE, and diffusion weights…",
            progress=22,
        )

    if "generation job args" in lowered or "generation model config" in lowered:
        return InferenceProgress(
            stage="loading_models",
            message="Initializing generation pipeline…",
            progress=18,
        )

    if "input prompt" in lowered or "input image" in lowered:
        return InferenceProgress(
            stage="preprocessing",
            message="Reading prompt and input image…",
            progress=28,
        )

    if "generating video" in lowered:
        return InferenceProgress(
            stage="preprocessing",
            message="Encoding text and image (VAE / T5)…",
            progress=32,
        )

    if "extending prompt" in lowered:
        return InferenceProgress(
            stage="preprocessing",
            message="Extending prompt…",
            progress=26,
        )

    if "saving generated video" in lowered:
        return InferenceProgress(
            stage="saving",
            message="Writing video file…",
            progress=88,
        )

    if text == "Finished." or lowered.endswith("finished."):
        return InferenceProgress(
            stage="done",
            message="Inference finished",
            progress=90,
        )

    tqdm = _parse_tqdm(text)
    if tqdm is None:
        return None

    pct, current, total = tqdm
    mapped = 35 + int(50 * current / total)
    mapped = min(max(mapped, 35), 85)
    return InferenceProgress(
        stage="diffusion",
        message=f"Generating frames — step {current}/{total} ({pct}%)",
        progress=mapped,
        current_step=current,
        total_steps=total,
    )
