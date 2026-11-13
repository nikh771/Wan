"""API-side LTX runner: spawns isolated LTX_PYTHON worker (never loads LTX in API process).

Jobs longer than LTX_CHUNK_SECONDS (default 10s) are queued as sequential chunks.
Each chunk after the first is image-to-video conditioned on the previous chunk's last
frame, then all chunks are concatenated into the final MP4.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

from core.types import GenerationJobParams
from ltx_service.chunking import (
    chunk_duration_seconds,
    plan_ltx_chunks,
)
from ltx_service.config import (
    calculate_ltx_frame_count,
    normalize_ltx_resolution,
)
from ltx_service.prompt_enhance import enhance_ltx_prompt
from ltx_service.settings import get_ltx_settings

logger = logging.getLogger(__name__)

_WORKER = Path(__file__).resolve().parent / "worker_main.py"


def _ltx_env() -> dict[str, str]:
    cfg = get_ltx_settings()
    env = os.environ.copy()
    roots = [str(cfg.project_root), str(cfg.ltx_repo_dir)]
    existing = env.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["LTX_REPO_DIR"] = str(cfg.ltx_repo_dir)
    env["LTX_MODEL_PATH"] = str(cfg.ltx_model_path)
    env["LTX_PIPELINE_CONFIG"] = cfg.resolved_ltx_pipeline_config()
    env["CUDA_VISIBLE_DEVICES"] = cfg.resolved_cuda_device()
    env["PYTHONUNBUFFERED"] = "1"
    env["LTX_CONDITIONING_STRENGTH"] = str(cfg.LTX_CONDITIONING_STRENGTH)
    env["LTX_IMAGE_COND_NOISE_SCALE"] = str(cfg.LTX_IMAGE_COND_NOISE_SCALE)
    env["LTX_DEFAULT_NEGATIVE_PROMPT"] = cfg.LTX_DEFAULT_NEGATIVE_PROMPT
    env["LTX_MOTION_AMPLIFY"] = str(cfg.LTX_MOTION_AMPLIFY)
    return env


def _stream_output(
    process: subprocess.Popen[str],
    on_log_line: Callable[[str], None] | None,
) -> str:
    assert process.stdout is not None
    tail: deque[str] = deque(maxlen=50)

    def _reader() -> None:
        for line in process.stdout:
            s = line.rstrip()
            tail.append(s)
            logger.info("[ltx] %s", s)
            if on_log_line:
                on_log_line(s)

    threading.Thread(target=_reader, daemon=True).start()
    process.wait()
    return "\n".join(tail)


def _parse_worker_ok(tail: str) -> dict:
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line.startswith("{") and '"status"' in line:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ok":
                return payload.get("result") or {}
            if payload.get("status") == "error":
                raise RuntimeError(payload.get("message") or "LTX error")
    return {}


def _run_worker_chunk(
    *,
    mode: str,
    prompt: str,
    negative_prompt: str,
    output_mp4: Path,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    seed: int,
    image_path: Path | None,
    on_log_line: Callable[[str], None] | None,
    conditioning_strength: float | None = None,
    image_cond_noise_scale: float | None = None,
) -> dict:
    cfg = get_ltx_settings()
    cmd = [
        cfg.resolved_ltx_python(),
        "-u",
        str(_WORKER),
        "--mode",
        mode,
        "--prompt",
        prompt,
        "--negative-prompt",
        negative_prompt,
        "--output",
        str(output_mp4.resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--duration-seconds",
        str(duration_seconds),
        "--seed",
        str(seed),
    ]
    if mode == "image_to_video":
        if image_path is None or not image_path.is_file():
            raise ValueError("image_to_video requires a valid input image")
        cmd.extend(["--image", str(image_path.resolve())])

    env = _ltx_env()
    if conditioning_strength is not None:
        env["LTX_CONDITIONING_STRENGTH"] = str(conditioning_strength)
    if image_cond_noise_scale is not None:
        env["LTX_IMAGE_COND_NOISE_SCALE"] = str(image_cond_noise_scale)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg.project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    tail = _stream_output(proc, on_log_line)
    if proc.returncode != 0:
        raise RuntimeError(f"LTX worker exit {proc.returncode}\n{tail[-2000:]}")
    return _parse_worker_ok(tail)


def _maybe_amplify(
    path: Path,
    *,
    mode: str,
    on_log_line: Callable[[str], None] | None,
    allow: bool = True,
) -> float | None:
    cfg = get_ltx_settings()
    if not allow or mode != "image_to_video" or cfg.LTX_MOTION_AMPLIFY <= 1.01:
        return None
    if not path.is_file():
        return None
    from ltx_service.motion_boost import amplify_mp4_motion

    amplify_mp4_motion(path, scale=float(cfg.LTX_MOTION_AMPLIFY))
    if on_log_line:
        on_log_line(f"[ltx_service] motion amplify scale={cfg.LTX_MOTION_AMPLIFY}")
    return float(cfg.LTX_MOTION_AMPLIFY)


def _trim_chunk_freeze(
    path: Path,
    *,
    is_last_chunk: bool,
    on_log_line: Callable[[str], None] | None,
) -> int:
    """Drop held last-frame tails before bridge extract / concat."""
    cfg = get_ltx_settings()
    if not cfg.LTX_TRIM_TRAILING_FREEZE:
        return 0
    from ltx_service.trim_freeze import trim_trailing_freeze_mp4

    # Non-final chunks: shave settle zone so the bridge isn't a frozen pose.
    # Final chunk: lighter trim so the video doesn't end on a held last frame.
    min_trim = (
        int(cfg.LTX_FINAL_MIN_TRIM_FRAMES)
        if is_last_chunk
        else int(cfg.LTX_CHUNK_MIN_TRIM_FRAMES)
    )
    removed = trim_trailing_freeze_mp4(
        path,
        diff_threshold=float(cfg.LTX_TRIM_DIFF_THRESHOLD),
        min_keep_ratio=float(cfg.LTX_TRIM_MIN_KEEP_RATIO),
        min_trim_frames=min_trim,
        max_trim_frames=int(cfg.LTX_TRIM_MAX_FRAMES),
    )
    if removed and on_log_line:
        on_log_line(f"[ltx_service] trimmed trailing freeze frames={removed} from {path.name}")
    return removed


def run_ltx_generation(params: GenerationJobParams) -> dict:
    cfg = get_ltx_settings()
    if not _WORKER.is_file():
        raise FileNotFoundError(f"LTX worker missing: {_WORKER}")

    duration = float(
        params.duration_seconds
        if params.duration_seconds is not None
        else cfg.LTX_DEFAULT_DURATION_SECONDS
    )
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    # Soft floor so very short clips still produce a valid LTX frame count.
    duration = max(1.0, duration)

    width = params.width or 704
    height = params.height or 384
    fps = int(params.fps or cfg.LTX_DEFAULT_FPS or 24)
    mode = (params.mode or params.metadata.get("mode") or "image_to_video").strip()
    if mode not in ("text_to_video", "image_to_video"):
        mode = "image_to_video"
    seed = int(params.seed if params.seed is not None else params.metadata.get("seed", -1))
    if seed < 0:
        seed = 0

    plan = plan_ltx_chunks(
        duration,
        fps=fps,
        chunk_seconds=float(cfg.LTX_CHUNK_SECONDS),
    )
    res = normalize_ltx_resolution(width, height)

    enhanced = enhance_ltx_prompt(
        params.prompt,
        mode=mode,  # type: ignore[arg-type]
        duration_seconds=duration,
        negative_prompt=params.negative_prompt or cfg.LTX_DEFAULT_NEGATIVE_PROMPT,
        enabled=cfg.LTX_PROMPT_ENHANCE,
        ollama_base_url=cfg.LTX_OLLAMA_BASE_URL,
        ollama_model=cfg.LTX_OLLAMA_MODEL,
        ollama_timeout_sec=cfg.LTX_OLLAMA_TIMEOUT_SEC,
    )
    if params.on_log_line:
        params.on_log_line(
            f"[ltx_service] prompt enhance source={enhanced.source} "
            f"original={enhanced.original_prompt[:100]!r}"
        )
        if enhanced.prompt != enhanced.original_prompt:
            params.on_log_line(f"[ltx_service] enhanced prompt={enhanced.prompt[:240]!r}")

    work_dir = params.output_mp4.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---------- single pass (≤ chunk seconds) ----------
    if plan.single_pass:
        frames = calculate_ltx_frame_count(duration, fps)
        if params.on_log_line:
            params.on_log_line(
                f"[ltx_service] LTX single-pass {mode} | {res.width}x{res.height} "
                f"frames={frames.actual_frames} (~{frames.actual_duration_seconds:.2f}s) "
                f"cond_strength={cfg.LTX_CONDITIONING_STRENGTH} "
                f"cond_noise={cfg.LTX_IMAGE_COND_NOISE_SCALE}"
            )
        if mode == "image_to_video" and params.image_path is None:
            raise ValueError("image_to_video requires an input image")

        result = _run_worker_chunk(
            mode=mode,
            prompt=enhanced.prompt,
            negative_prompt=enhanced.negative_prompt,
            output_mp4=params.output_mp4,
            width=res.width,
            height=res.height,
            fps=fps,
            duration_seconds=duration,
            seed=seed,
            image_path=params.image_path if mode == "image_to_video" else None,
            on_log_line=params.on_log_line,
        )
        amplify = _maybe_amplify(
            params.output_mp4, mode=mode, on_log_line=params.on_log_line
        )
        trimmed = _trim_chunk_freeze(
            params.output_mp4, is_last_chunk=True, on_log_line=params.on_log_line
        )
        result.setdefault("prompt_enhance_source", enhanced.source)
        result.setdefault("original_prompt", enhanced.original_prompt)
        result.setdefault("enhanced_prompt", enhanced.prompt)
        result["chunks"] = 1
        result["chunked"] = False
        if amplify is not None:
            result["motion_amplify"] = amplify
        if trimmed:
            result["trimmed_trailing_frames"] = trimmed
        return result

    # ---------- queued chunks (> chunk seconds) ----------
    if params.on_log_line:
        params.on_log_line(
            f"[ltx_service] LTX queued chunks={plan.num_chunks} "
            f"× ≤{plan.chunk_seconds:.0f}s (target {duration:.2f}s, "
            f"last-frame continuity) | {res.width}x{res.height} @ {fps}fps"
        )

    chunk_paths: list[Path] = []
    chunk_metas: list[dict] = []
    # Continuity reference: user image for I2V chunk 0; last frame of prior chunk thereafter.
    bridge: Path | None = params.image_path if mode == "image_to_video" else None

    for i in range(plan.num_chunks):
        chunk_dur = chunk_duration_seconds(plan, i)
        chunk_out = work_dir / f"ltx_chunk_{i:02d}.mp4"
        if i == 0:
            chunk_mode = mode
            chunk_image = bridge
        else:
            chunk_mode = "image_to_video"
            chunk_image = bridge
            if chunk_image is None or not chunk_image.is_file():
                raise RuntimeError(
                    f"Missing bridge frame for LTX chunk {i + 1}/{plan.num_chunks}"
                )

        if chunk_mode == "image_to_video" and (
            chunk_image is None or not chunk_image.is_file()
        ):
            raise ValueError("image_to_video chunk requires a reference image")

        # Same cinematic prompt for every chunk — quality stays consistent.
        chunk_prompt = enhanced.prompt
        chunk_seed = seed + i
        is_continuation = i > 0
        # Continuation: lock harder to previous last frame (less noise/drift).
        cond_strength = (
            float(cfg.LTX_BRIDGE_CONDITIONING_STRENGTH)
            if is_continuation
            else float(cfg.LTX_CONDITIONING_STRENGTH)
        )
        cond_noise = (
            float(cfg.LTX_BRIDGE_IMAGE_COND_NOISE_SCALE)
            if is_continuation
            else float(cfg.LTX_IMAGE_COND_NOISE_SCALE)
        )
        # Amplify only on chunk 0 when allowed — amplify on bridge chunks ruins quality.
        allow_amplify = (not is_continuation) or bool(cfg.LTX_AMPLIFY_CONTINUATION_CHUNKS)

        if params.on_log_line:
            ref = "none"
            if chunk_image is not None:
                ref = "prev_last_frame" if is_continuation else "image"
            params.on_log_line(
                f"[ltx_service] Generating chunk {i + 1}/{plan.num_chunks} "
                f"mode={chunk_mode} duration≈{chunk_dur:.2f}s ref={ref} "
                f"cond_strength={cond_strength} cond_noise={cond_noise} "
                f"amplify={'on' if allow_amplify and chunk_mode == 'image_to_video' else 'off'}"
            )

        meta = _run_worker_chunk(
            mode=chunk_mode,
            prompt=chunk_prompt,
            negative_prompt=enhanced.negative_prompt,
            output_mp4=chunk_out,
            width=res.width,
            height=res.height,
            fps=fps,
            duration_seconds=chunk_dur,
            seed=chunk_seed,
            image_path=chunk_image,
            on_log_line=params.on_log_line,
            conditioning_strength=cond_strength if chunk_mode == "image_to_video" else None,
            image_cond_noise_scale=cond_noise if chunk_mode == "image_to_video" else None,
        )
        amplify = _maybe_amplify(
            chunk_out,
            mode=chunk_mode,
            on_log_line=params.on_log_line,
            allow=allow_amplify,
        )
        if amplify is not None:
            meta["motion_amplify"] = amplify
        # Trim freeze BEFORE bridge extract so the next chunk continues from
        # a live frame, not a held last pose.
        trimmed = _trim_chunk_freeze(
            chunk_out,
            is_last_chunk=(i == plan.num_chunks - 1),
            on_log_line=params.on_log_line,
        )
        if trimmed:
            meta["trimmed_trailing_frames"] = trimmed
        meta["conditioning_strength"] = cond_strength if chunk_mode == "image_to_video" else None
        meta["image_cond_noise_scale"] = cond_noise if chunk_mode == "image_to_video" else None
        meta["prompt"] = chunk_prompt
        chunk_paths.append(chunk_out)
        chunk_metas.append(meta)

        if i < plan.num_chunks - 1:
            from wan_service.utils.ffmpeg import extract_last_frame

            bridge = work_dir / f"ltx_bridge_{i + 1:02d}.png"
            extract_last_frame(chunk_out, bridge)
            if params.on_log_line:
                params.on_log_line(
                    f"[ltx_service] Bridge last-frame for chunk {i + 2}: {bridge.name}"
                )

    from wan_service.utils.ffmpeg import concat_mp4s

    if params.on_log_line:
        params.on_log_line(
            f"[ltx_service] Concatenating {len(chunk_paths)} chunks → {params.output_mp4.name}"
        )
    concat_mp4s(
        chunk_paths,
        params.output_mp4,
        skip_first_frame_after_first=True,
        fps=fps,
    )

    return {
        "engine": "ltx",
        "mode": mode,
        "chunks": plan.num_chunks,
        "chunked": True,
        "chunk_seconds": plan.chunk_seconds,
        "chunk_files": [p.name for p in chunk_paths],
        "frames_per_chunk": plan.frames_per_chunk,
        "last_chunk_frames": plan.last_chunk_frames,
        "duration_seconds": duration,
        "width": res.width,
        "height": res.height,
        "fps": fps,
        "seed": seed,
        "status": "completed",
        "output_path": str(params.output_mp4),
        "prompt_enhance_source": enhanced.source,
        "original_prompt": enhanced.original_prompt,
        "enhanced_prompt": enhanced.prompt,
        "chunk_results": chunk_metas,
    }
