"""Entry point for COGVIDEOX_PYTHON subprocess (load → generate → unload)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _model_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    raw = os.environ.get("COGVIDEOX_MODEL_PATH", "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (root / p).resolve()
    return (root / "cogvideox_service" / "models" / "cogvideox" / "CogVideoX-2b").resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text_to_video", "image_to_video"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--memory-mode", choices=("memory", "normal"), default="memory")
    args = parser.parse_args()

    model = _model_path()
    hf_id = os.environ.get("COGVIDEOX_HF_ID", "zai-org/CogVideoX-2b")
    dtype = os.environ.get("COGVIDEOX_DTYPE", "float16")
    device = int(os.environ.get("WAN_CUDA_DEVICE", os.environ.get("CUDA_VISIBLE_DEVICES", "0")) or "0")
    # When CUDA_VISIBLE_DEVICES remaps, always use logical device 0 inside the process.
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        device = 0

    from cogvideox_service.adapter import CogVideoXAdapter, CogVideoXGenerateRequest

    engine = CogVideoXAdapter(
        model_path=model,
        hf_id=hf_id,
        dtype_name=dtype,
        device_index=device,
    )

    try:
        duration = args.duration_seconds if args.duration_seconds and args.duration_seconds > 0 else None
        request = CogVideoXGenerateRequest(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_mp4=Path(args.output),
            mode=args.mode,
            image_path=Path(args.image) if args.image else None,
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.num_frames if duration is None else None,
            duration_seconds=duration,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            memory_mode=args.memory_mode,
        )
        meta, bench = engine.generate(request)
        result = {**meta, "benchmark": bench.__dict__}
        print(json.dumps({"status": "ok", "result": result}), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "message": str(exc)}), flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        engine.unload()


if __name__ == "__main__":
    raise SystemExit(main())
