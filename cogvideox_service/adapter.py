"""
CogVideoXAdapter — runs only inside COGVIDEOX_PYTHON (isolated from Wan API process).

Lazy load / unload around each generate() call; does not import Diffusers at FastAPI startup.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cogvideox_service.config import (
    COGVIDEOX_HF_ID,
    COGVIDEOX_MODEL_ID,
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_WIDTH,
    LICENSE_METADATA,
    calculate_cogvideox_duration,
    calculate_cogvideox_frame_count,
    model_available,
    normalize_cogvideox_resolution,
)
from cogvideox_service.gpu_state import GpuState, write_gpu_state

logger = logging.getLogger(__name__)


@dataclass
class CogVideoXGenerateRequest:
    prompt: str
    output_mp4: Path
    mode: str = "text_to_video"
    negative_prompt: str = ""
    image_path: Path | None = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    num_frames: int | None = DEFAULT_NUM_FRAMES
    duration_seconds: float | None = None
    steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int = 12345
    memory_mode: str = "memory"


@dataclass
class CogVideoXBenchmark:
    engine: str = "cogvideox"
    model: str = COGVIDEOX_MODEL_ID
    mode: str = ""
    gpu: str | None = None
    cuda: str | None = None
    pytorch: str | None = None
    diffusers: str | None = None
    vram_total_gb: float | None = None
    vram_free_before_gb: float | None = None
    vram_peak_gb: float | None = None
    load_time_seconds: float | None = None
    generation_time_seconds: float | None = None
    total_time_seconds: float | None = None
    frames: int = 0
    width: int = 0
    height: int = 0
    fps: int = DEFAULT_FPS
    duration_seconds: float = 0.0
    steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    seed: int = 0
    memory_mode: str = "memory"
    output_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class CogVideoXAdapter:
    """T2V adapter for zai-org/CogVideoX-2b via Diffusers CogVideoXPipeline."""

    def __init__(
        self,
        *,
        model_path: Path,
        hf_id: str = COGVIDEOX_HF_ID,
        dtype_name: str = "float16",
        device_index: int = 0,
    ) -> None:
        self.model_path = model_path.resolve()
        self.hf_id = hf_id
        self.dtype_name = dtype_name
        self.device_index = device_index
        self._pipe: Any = None
        self._loaded = False
        self._memory_mode: str | None = None

    @staticmethod
    def get_gpu_status() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "gpu": None,
                "vram_total_gb": None,
                "vram_allocated_gb": None,
                "vram_reserved_gb": None,
                "vram_free_gb": None,
                "cuda_version": None,
                "pytorch": torch.__version__,
            }
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory
        reserved = torch.cuda.memory_reserved(device)
        allocated = torch.cuda.memory_allocated(device)
        free = total - reserved
        return {
            "cuda_available": True,
            "gpu": torch.cuda.get_device_name(device),
            "vram_total_gb": round(total / (1024**3), 2),
            "vram_allocated_gb": round(allocated / (1024**3), 2),
            "vram_reserved_gb": round(reserved / (1024**3), 2),
            "vram_free_gb": round(free / (1024**3), 2),
            "cuda_version": torch.version.cuda,
            "pytorch": torch.__version__,
        }

    def health_check(self) -> dict[str, Any]:
        gpu = self.get_gpu_status()
        available = model_available(self.model_path)
        ready = available and gpu["cuda_available"]
        payload = {
            "engine": "cogvideox",
            "status": "ready" if ready else "unavailable",
            "model": COGVIDEOX_MODEL_ID,
            "hf_id": self.hf_id,
            "loaded": self._loaded,
            "cuda_available": gpu["cuda_available"],
            "gpu": gpu.get("gpu"),
            "model_path": str(self.model_path),
            "model_available": available,
            "capabilities": self.get_capabilities(),
            "license": LICENSE_METADATA,
        }
        if self._loaded:
            payload["vram_allocated_gb"] = gpu.get("vram_allocated_gb")
            payload["vram_reserved_gb"] = gpu.get("vram_reserved_gb")
            payload["vram_total_gb"] = gpu.get("vram_total_gb")
        return payload

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "modes": ["text_to_video"],
            "modes_planned": ["image_to_video"],
            "i2v_note": (
                "image_to_video requires a separate CogVideoX I2V checkpoint "
                "and CogVideoXImageToVideoPipeline — not part of this POC."
            ),
            "type": "video",
            "license": "Apache-2.0",
        }

    def _resolve_dtype(self):
        import torch

        name = (self.dtype_name or "float16").lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if name in ("fp32", "float32"):
            return torch.float32
        return torch.float16

    def load(self, *, memory_mode: str = "memory") -> float:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. CogVideoX-2B requires an NVIDIA GPU; "
                "refusing silent CPU fallback."
            )
        if not model_available(self.model_path):
            raise FileNotFoundError(
                f"CogVideoX model missing or incomplete at {self.model_path}. "
                f"Download explicitly: bash scripts/download_cogvideox_2b.sh "
                f"(HF id {self.hf_id})."
            )

        if self._loaded and self._pipe is not None and self._memory_mode == memory_mode:
            return 0.0

        if self._loaded:
            self.unload()

        gpu = self.get_gpu_status()
        try:
            import diffusers

            diffusers_ver = getattr(diffusers, "__version__", "unknown")
        except ImportError as exc:
            raise RuntimeError("diffusers is not installed in COGVIDEOX_PYTHON") from exc

        logger.info(
            "CogVideoX initialization:\n"
            "GPU: %s\n"
            "VRAM: %s GB total / %s GB free\n"
            "CUDA: %s\n"
            "PyTorch: %s\n"
            "Diffusers: %s\n"
            "Model: %s (%s)\n"
            "Memory mode: %s",
            gpu.get("gpu"),
            gpu.get("vram_total_gb"),
            gpu.get("vram_free_gb"),
            gpu.get("cuda_version"),
            gpu.get("pytorch"),
            diffusers_ver,
            COGVIDEOX_MODEL_ID,
            self.model_path,
            memory_mode,
        )

        write_gpu_state(GpuState.COGVIDEOX_LOADED, owner="cogvideox", detail="loading")
        t0 = time.perf_counter()
        from diffusers import CogVideoXPipeline

        dtype = self._resolve_dtype()
        pipe = CogVideoXPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )

        mode = (memory_mode or "memory").strip().lower()
        if mode == "normal":
            pipe = pipe.to(f"cuda:{self.device_index}")
            logger.info("CogVideoX memory mode=normal (full GPU, no CPU offload)")
        else:
            # Recommended for ~24GB cards: ~11–12GB with offload + VAE tiling.
            pipe.enable_model_cpu_offload(device=f"cuda:{self.device_index}")
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
                logger.info(
                    "CogVideoX memory mode=memory "
                    "(enable_model_cpu_offload + vae.enable_tiling)"
                )
            else:
                logger.info(
                    "CogVideoX memory mode=memory (enable_model_cpu_offload; "
                    "VAE tiling unavailable)"
                )

        self._pipe = pipe
        self._loaded = True
        self._memory_mode = mode
        elapsed = time.perf_counter() - t0
        write_gpu_state(GpuState.COGVIDEOX_LOADED, owner="cogvideox", detail="ready")
        logger.info("CogVideoX loaded in %.2fs | mode=%s", elapsed, mode)
        return elapsed

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
        self._memory_mode = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:  # noqa: BLE001
                    pass
            gc.collect()
        except ImportError:
            pass
        write_gpu_state(GpuState.IDLE, owner=None, detail="cogvideox_unloaded")
        logger.info("CogVideoX unloaded; CUDA cache cleared")

    def generate(self, request: CogVideoXGenerateRequest) -> tuple[dict[str, Any], CogVideoXBenchmark]:
        if request.mode != "text_to_video":
            raise ValueError(
                f"Unsupported CogVideoX mode for POC: {request.mode}. "
                "Only text_to_video is implemented. "
                "I2V requires a separate CogVideoX I2V checkpoint "
                "(CogVideoXImageToVideoPipeline) — do not fake with FFmpeg."
            )
        if not request.prompt or not request.prompt.strip():
            raise ValueError("prompt is required")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. CogVideoX-2B requires an NVIDIA GPU; "
                "refusing silent CPU fallback."
            )

        try:
            import diffusers
        except ImportError as exc:
            raise RuntimeError("diffusers is not installed in COGVIDEOX_PYTHON") from exc

        res = normalize_cogvideox_resolution(request.width, request.height)
        frame_plan = calculate_cogvideox_frame_count(
            num_frames=request.num_frames,
            duration_seconds=request.duration_seconds,
            fps=request.fps,
        )
        if frame_plan.actual_frames != frame_plan.requested_frames:
            logger.info(
                "CogVideoX frame alignment: requested=%s → actual=%s (8N+1 rule)",
                frame_plan.requested_frames,
                frame_plan.actual_frames,
            )
        if res.width != res.requested_width or res.height != res.requested_height:
            logger.info(
                "CogVideoX resolution alignment: requested=%sx%s → actual=%sx%s",
                res.requested_width,
                res.requested_height,
                res.width,
                res.height,
            )

        seed = int(request.seed)
        if seed < 0:
            seed = int(time.time()) % 2_000_000_000

        gpu_before = self.get_gpu_status()
        bench = CogVideoXBenchmark(
            mode=request.mode,
            gpu=gpu_before.get("gpu"),
            cuda=gpu_before.get("cuda_version"),
            pytorch=gpu_before.get("pytorch"),
            diffusers=getattr(diffusers, "__version__", None),
            vram_total_gb=gpu_before.get("vram_total_gb"),
            vram_free_before_gb=gpu_before.get("vram_free_gb"),
            frames=frame_plan.actual_frames,
            width=res.width,
            height=res.height,
            fps=request.fps,
            duration_seconds=round(frame_plan.actual_duration_seconds, 4),
            steps=int(request.steps),
            guidance_scale=float(request.guidance_scale),
            seed=seed,
            memory_mode=request.memory_mode,
            output_path=str(request.output_mp4),
        )

        total_t0 = time.perf_counter()
        load_t = self.load(memory_mode=request.memory_mode)
        bench.load_time_seconds = load_t
        assert self._pipe is not None

        write_gpu_state(GpuState.GENERATING, owner="cogvideox", detail="t2v")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        generator = torch.Generator(device="cuda").manual_seed(seed)
        logger.info(
            "CogVideoX generate | model=%s mode=%s size=%sx%s frames=%s fps=%s "
            "steps=%s guidance=%s seed=%s memory_mode=%s",
            COGVIDEOX_MODEL_ID,
            request.mode,
            res.width,
            res.height,
            frame_plan.actual_frames,
            request.fps,
            request.steps,
            request.guidance_scale,
            seed,
            request.memory_mode,
        )

        t0 = time.perf_counter()
        # Official Diffusers CogVideoXPipeline T2V call.
        result = self._pipe(
            prompt=request.prompt.strip(),
            negative_prompt=(request.negative_prompt or "").strip() or None,
            height=res.height,
            width=res.width,
            num_frames=frame_plan.actual_frames,
            num_inference_steps=int(request.steps),
            guidance_scale=float(request.guidance_scale),
            generator=generator,
        )
        frames = result.frames[0]
        bench.generation_time_seconds = time.perf_counter() - t0

        request.output_mp4.parent.mkdir(parents=True, exist_ok=True)
        from diffusers.utils import export_to_video

        export_to_video(frames, str(request.output_mp4), fps=request.fps)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            bench.vram_peak_gb = round(peak, 2)

        bench.total_time_seconds = round(time.perf_counter() - total_t0, 3)
        write_gpu_state(GpuState.COGVIDEOX_LOADED, owner="cogvideox", detail="idle_after_generate")

        meta = {
            "engine": "cogvideox",
            "status": "completed",
            "model": COGVIDEOX_MODEL_ID,
            "hf_id": self.hf_id,
            "mode": request.mode,
            "output_path": str(request.output_mp4),
            "width": res.width,
            "height": res.height,
            "requested_width": res.requested_width,
            "requested_height": res.requested_height,
            "fps": request.fps,
            "frames": frame_plan.actual_frames,
            "requested_frames": frame_plan.requested_frames,
            "duration_seconds": round(
                calculate_cogvideox_duration(frame_plan.actual_frames, request.fps), 4
            ),
            "steps": int(request.steps),
            "guidance_scale": float(request.guidance_scale),
            "seed": seed,
            "memory_mode": request.memory_mode,
            "load_time_seconds": bench.load_time_seconds,
            "generation_time_seconds": bench.generation_time_seconds,
            "total_time_seconds": bench.total_time_seconds,
            "vram_peak_gb": bench.vram_peak_gb,
            "gpu": bench.gpu,
            "license": LICENSE_METADATA,
        }
        return meta, bench


# Interface stub for a future I2V adapter (separate checkpoint required).
class CogVideoXImageToVideoAdapter:
    """Placeholder — do not use until a CogVideoX I2V checkpoint is integrated."""

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "modes": ["image_to_video"],
            "implemented": False,
            "requires": "CogVideoXImageToVideoPipeline + dedicated I2V weights",
        }

    def generate(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "CogVideoX I2V is not part of this POC. Use a dedicated I2V checkpoint "
            "with CogVideoXImageToVideoPipeline — do not fake with zoom/pan/FFmpeg."
        )
