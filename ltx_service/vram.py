"""VRAM-safe LTX pipeline loading for ≤24GB GPUs.

Upstream create_ltx_video_pipeline() moves the PixArt T5 encoder to CUDA in fp32
after transformer+VAE are already resident, which OOMs on 24GB cards. This module
patches that factory before infer() runs.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PATCHED = False
_DEVICE_PROP_PATCHED = False


def gpu_total_memory_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def should_offload_to_cpu(*, force: bool | None = None) -> bool:
    """Match upstream infer() gate: offload when flag set and VRAM < 30GB."""
    if force is False:
        return False
    if force is True:
        return gpu_total_memory_gb() < 30 or gpu_total_memory_gb() == 0
    # Auto: offload on consumer/pro 24GB-class cards.
    return 0 < gpu_total_memory_gb() < 30


def _install_forced_execution_device_property() -> None:
    """Allow pipelines to pin Diffusers' .device when T5 stays on CPU.

    Diffusers resolves pipeline.device from the first nn.Module (text_encoder). With T5
    on CPU, __call__ would build CPU latents while infer() passes a CUDA generator.
    """
    global _DEVICE_PROP_PATCHED
    if _DEVICE_PROP_PATCHED:
        return

    import torch
    from diffusers import DiffusionPipeline

    original = DiffusionPipeline.device.fget

    def device(self) -> torch.device:  # noqa: ANN001
        forced = getattr(self, "_forced_execution_device", None)
        if forced is not None:
            return torch.device(forced)
        return original(self)

    DiffusionPipeline.device = property(device)  # type: ignore[method-assign, assignment]
    _DEVICE_PROP_PATCHED = True


def install_vram_safe_pipeline_factory() -> None:
    """Replace ltx_video.inference.create_ltx_video_pipeline with a low-VRAM variant."""
    global _PATCHED
    if _PATCHED:
        return

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        T5EncoderModel,
        T5Tokenizer,
    )
    from safetensors import safe_open
    import json
    import os
    from pathlib import Path

    import ltx_video.inference as inf
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )
    from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from ltx_video.schedulers.rf import RectifiedFlowScheduler

    def create_ltx_video_pipeline(
        ckpt_path: str,
        precision: str,
        text_encoder_model_name_or_path: str,
        sampler: Optional[str] = None,
        device: Optional[str] = None,
        enhance_prompt: bool = False,
        prompt_enhancer_image_caption_model_name_or_path: Optional[str] = None,
        prompt_enhancer_llm_model_name_or_path: Optional[str] = None,
    ) -> LTXVideoPipeline:
        ckpt_path = Path(ckpt_path)
        assert os.path.exists(ckpt_path), (
            f"Ckpt path provided (--ckpt_path) {ckpt_path} does not exist"
        )

        with safe_open(ckpt_path, framework="pt") as f:
            metadata = f.metadata()
            config_str = metadata.get("config")
            configs = json.loads(config_str)
            allowed_inference_steps = configs.get("allowed_inference_steps", None)

        low_vram = should_offload_to_cpu()
        if low_vram and enhance_prompt:
            logger.warning(
                "Disabling LTX prompt enhancement on %.1fGB GPU (Florence/Llama need extra VRAM)",
                gpu_total_memory_gb(),
            )
            enhance_prompt = False

        vae = CausalVideoAutoencoder.from_pretrained(ckpt_path)
        transformer = inf.create_transformer(ckpt_path, precision)

        if sampler == "from_checkpoint" or not sampler:
            scheduler = RectifiedFlowScheduler.from_pretrained(ckpt_path)
        else:
            scheduler = RectifiedFlowScheduler(
                sampler=(
                    "Uniform" if sampler.lower() == "uniform" else "LinearQuadratic"
                )
            )

        # bf16 on load — upstream loads fp32 then .to(device) which OOMs on 24GB.
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        patchifier = SymmetricPatchifier(patch_size=1)
        tokenizer = T5Tokenizer.from_pretrained(
            text_encoder_model_name_or_path, subfolder="tokenizer"
        )

        transformer = transformer.to(device)
        vae = vae.to(device)
        vae = vae.to(torch.bfloat16)
        text_encoder = text_encoder.to(torch.bfloat16)

        if low_vram:
            # Keep T5 on CPU; pipeline.encode_prompt moves it when offload_to_cpu is set.
            text_encoder = text_encoder.to("cpu")
            logger.info(
                "LTX low-VRAM mode: T5 text encoder on CPU (GPU total≈%.1fGB)",
                gpu_total_memory_gb(),
            )
        else:
            text_encoder = text_encoder.to(device)

        if enhance_prompt:
            prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
                prompt_enhancer_image_caption_model_name_or_path,
                trust_remote_code=True,
            )
            prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
                prompt_enhancer_image_caption_model_name_or_path,
                trust_remote_code=True,
            )
            prompt_enhancer_llm_model = AutoModelForCausalLM.from_pretrained(
                prompt_enhancer_llm_model_name_or_path,
                torch_dtype="bfloat16",
            )
            prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained(
                prompt_enhancer_llm_model_name_or_path,
            )
        else:
            prompt_enhancer_image_caption_model = None
            prompt_enhancer_image_caption_processor = None
            prompt_enhancer_llm_model = None
            prompt_enhancer_llm_tokenizer = None

        pipeline = LTXVideoPipeline(
            transformer=transformer,
            patchifier=patchifier,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae,
            prompt_enhancer_image_caption_model=prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor=prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model=prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer=prompt_enhancer_llm_tokenizer,
            allowed_inference_steps=allowed_inference_steps,
        )
        if low_vram:
            # Avoid pipeline.to(device) pulling T5 back onto CUDA.
            if pipeline.transformer is not None:
                pipeline.transformer.to(device)
            if pipeline.vae is not None:
                pipeline.vae.to(device)
            # Pin execution device to CUDA so latents/generator agree while T5 stays on CPU.
            _install_forced_execution_device_property()
            exec_device = torch.device(device) if device is not None else torch.device("cuda")
            object.__setattr__(pipeline, "_forced_execution_device", exec_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            pipeline = pipeline.to(device)
        return pipeline

    inf.create_ltx_video_pipeline = create_ltx_video_pipeline  # type: ignore[assignment]
    _PATCHED = True
    logger.info("Installed VRAM-safe LTX create_ltx_video_pipeline patch")
