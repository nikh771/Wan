"""
LTXVideoEngine — runs only inside LTX_PYTHON (isolated from Wan API process).

Lazy load / unload around each generate() call; does not import at FastAPI startup.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltx_service.config import (
    DEFAULT_PIPELINE_REL,
    LTX_CHECKPOINT_FILENAME,
    LTX_MODEL_ID,
    LtxFramePlan,
    LtxResolution,
    calculate_ltx_frame_count,
    materialize_pipeline_config,
    normalize_ltx_resolution,
)

logger = logging.getLogger(__name__)


@dataclass
class LtxGenerateRequest:
    prompt: str
    output_mp4: Path
    mode: str = "image_to_video"
    negative_prompt: str = ""
    image_path: Path | None = None
    duration_seconds: float = 5.0
    width: int = 704
    height: int = 384
    fps: int = 24
    seed: int = -1


@dataclass
class LtxBenchmark:
    engine: str = LTX_MODEL_ID
    model: str = LTX_CHECKPOINT_FILENAME
    mode: str = ""
    gpu: str | None = None
    vram_before_gb: float | None = None
    vram_peak_gb: float | None = None
    load_time_seconds: float | None = None
    generation_time_seconds: float | None = None
    frames: int = 0
    width: int = 0
    height: int = 0
    fps: int = 24
    seed: int = 0
    output_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class LTXVideoEngine:
    def __init__(
        self,
        *,
        repo_dir: Path,
        model_path: Path,
        pipeline_rel: str = DEFAULT_PIPELINE_REL,
    ) -> None:
        self.repo_dir = repo_dir.resolve()
        self.model_path = model_path.resolve()
        self.pipeline_rel = pipeline_rel
        self._loaded = False
        self._resolved_pipeline: Path | None = None

    @staticmethod
    def get_gpu_status() -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "gpu": None,
                "vram_total_gb": None,
                "vram_allocated_gb": None,
            }
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        return {
            "cuda_available": True,
            "gpu": torch.cuda.get_device_name(device),
            "vram_total_gb": round(props.total_memory / (1024**3), 2),
            "vram_allocated_gb": round(
                torch.cuda.memory_allocated(device) / (1024**3), 2
            ),
        }

    def health_check(self) -> dict[str, Any]:
        installed = (self.repo_dir / "inference.py").is_file()
        model_ok = self.model_path.is_file()
        gpu = self.get_gpu_status()
        ready = installed and model_ok and gpu["cuda_available"]
        return {
            "engine": "ltx",
            "model": LTX_MODEL_ID,
            "checkpoint": LTX_CHECKPOINT_FILENAME,
            "installed": installed,
            "model_available": model_ok,
            "model_path": str(self.model_path),
            "cuda_available": gpu["cuda_available"],
            "gpu": gpu.get("gpu"),
            "status": "ready" if ready else "unavailable",
        }

    def get_capabilities(self) -> list[str]:
        return ["text_to_video", "image_to_video"]

    def load(self) -> float:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"LTX checkpoint missing: {self.model_path}. "
                f"Place {LTX_CHECKPOINT_FILENAME} under ltx_service/models/ltx/"
            )
        t0 = time.perf_counter()
        self._resolved_pipeline = materialize_pipeline_config(
            self.repo_dir,
            self.pipeline_rel,
            self.model_path,
        )
        self._loaded = True
        elapsed = time.perf_counter() - t0
        logger.info(
            "LTX pipeline config ready in %.2fs | checkpoint=%s",
            elapsed,
            self.model_path.name,
        )
        return elapsed

    def unload(self) -> None:
        self._loaded = False
        self._resolved_pipeline = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, request: LtxGenerateRequest) -> tuple[dict[str, Any], LtxBenchmark]:
        if request.mode not in ("text_to_video", "image_to_video"):
            raise ValueError(f"Unsupported LTX mode: {request.mode}")
        if request.mode == "image_to_video":
            if request.image_path is None or not request.image_path.is_file():
                raise FileNotFoundError("image_to_video requires a valid input image")

        res = normalize_ltx_resolution(request.width, request.height)
        frame_plan = calculate_ltx_frame_count(request.duration_seconds, request.fps)
        seed = request.seed if request.seed >= 0 else int(time.time()) % 2_000_000_000

        bench = LtxBenchmark(
            mode=request.mode,
            gpu=self.get_gpu_status().get("gpu"),
            vram_before_gb=self.get_gpu_status().get("vram_allocated_gb"),
            frames=frame_plan.actual_frames,
            width=res.width,
            height=res.height,
            fps=request.fps,
            seed=seed,
            output_path=str(request.output_mp4),
        )

        load_t = self.load() if not self._loaded else 0.0
        bench.load_time_seconds = load_t

        assert self._resolved_pipeline is not None
        from ltx_video.inference import InferenceConfig, infer

        out_dir = request.output_mp4.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        conditioning_paths = None
        conditioning_start = None
        if request.mode == "image_to_video" and request.image_path:
            conditioning_paths = [str(request.image_path.resolve())]
            conditioning_start = [0]

        config = InferenceConfig(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            output_path=str(out_dir),
            pipeline_config=str(self._resolved_pipeline),
            seed=seed,
            height=res.height,
            width=res.width,
            num_frames=frame_plan.actual_frames,
            frame_rate=request.fps,
            conditioning_media_paths=conditioning_paths,
            conditioning_start_frames=conditioning_start,
            offload_to_cpu=False,
        )

        logger.info(
            "LTX generate | model=%s mode=%s frames=%s (%s@%sfps) size=%sx%s gpu=%s",
            LTX_CHECKPOINT_FILENAME,
            request.mode,
            frame_plan.actual_frames,
            round(frame_plan.actual_duration_seconds, 3),
            request.fps,
            res.width,
            res.height,
            bench.gpu,
        )

        t0 = time.perf_counter()
        infer(config=config)
        bench.generation_time_seconds = time.perf_counter() - t0

        mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("LTX infer() did not write an MP4 file.")
        if mp4s[0].resolve() != request.output_mp4.resolve():
            shutil.move(str(mp4s[0]), str(request.output_mp4))

        gpu_after = self.get_gpu_status()
        bench.vram_peak_gb = gpu_after.get("vram_allocated_gb")

        meta = {
            "engine": "ltx",
            "model": LTX_MODEL_ID,
            "checkpoint": LTX_CHECKPOINT_FILENAME,
            "mode": request.mode,
            "output_path": str(request.output_mp4),
            "width": res.width,
            "height": res.height,
            "requested_width": res.requested_width,
            "requested_height": res.requested_height,
            "fps": request.fps,
            "frames": frame_plan.actual_frames,
            "requested_frames": frame_plan.requested_frames,
            "duration_seconds": round(frame_plan.actual_duration_seconds, 4),
            "seed": seed,
            "status": "completed",
            "pipeline_config": str(self._resolved_pipeline),
        }
        return meta, bench
