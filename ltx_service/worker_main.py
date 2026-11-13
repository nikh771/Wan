"""Entry point for LTX_PYTHON subprocess (load → generate → unload)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    repo = Path(os.environ.get("LTX_REPO_DIR", "")).resolve()
    if not repo.is_dir():
        repo = (root / "ltx_service" / "third_party" / "LTX-Video").resolve()
    model_raw = os.environ.get("LTX_MODEL_PATH", "").strip()
    if model_raw:
        model = Path(model_raw).resolve()
    else:
        model = (
            root
            / "ltx_service"
            / "models"
            / "ltx"
            / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
        ).resolve()
    return repo, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text_to_video", "image_to_video"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    repo, model = _paths()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    pipeline_rel = os.environ.get(
        "LTX_PIPELINE_CONFIG", "configs/ltxv-2b-0.9.6-distilled.yaml"
    )

    from ltx_service.adapter import LTXVideoEngine, LtxGenerateRequest

    engine = LTXVideoEngine(
        repo_dir=repo,
        model_path=model,
        pipeline_rel=pipeline_rel,
    )

    try:
        request = LtxGenerateRequest(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_mp4=Path(args.output),
            mode=args.mode,
            image_path=Path(args.image) if args.image else None,
            duration_seconds=args.duration_seconds,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
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
