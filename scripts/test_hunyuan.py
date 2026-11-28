#!/usr/bin/env python3
"""Smoke test HunyuanVideo-1.5 via engine runner (no FastAPI)."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from core.types import GenerationJobParams
from hunyuan_service.runner import run_hunyuan_generation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--mode", choices=("text_to_video", "image_to_video"), default="text_to_video")
    parser.add_argument("--image", default="")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out = Path(tempfile.gettempdir()) / "hunyuan_smoke.mp4"
    image = Path(args.image) if args.image else None
    meta = run_hunyuan_generation(
        GenerationJobParams(
            job_id="smoke",
            prompt=args.prompt,
            output_mp4=out,
            image_path=image,
            duration_seconds=args.duration,
            mode=args.mode,
            seed=args.seed,
        )
    )
    print("OK:", out, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())