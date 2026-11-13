#!/usr/bin/env python3
"""Manual LTX 2B distilled test (runs worker in LTX_PYTHON)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"'))


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Test LTX 2B distilled 0.9.6")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default="")
    parser.add_argument("--mode", choices=("text_to_video", "image_to_video"), default="")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--negative-prompt", default="")
    args = parser.parse_args()

    mode = args.mode or ("image_to_video" if args.image else "text_to_video")
    if mode == "image_to_video" and not args.image:
        print("image_to_video requires --image", file=sys.stderr)
        return 2

    ltx_py = os.environ.get("LTX_PYTHON", "ltx_service/engines/ltx-env/bin/python")
    if not Path(ltx_py).is_absolute():
        ltx_py = str(ROOT / ltx_py)
    worker = ROOT / "ltx_service" / "worker_main.py"
    out_dir = ROOT / "ltx_service" / "outputs" / "ltx"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_mp4 = out_dir / f"ltx_{mode}_{stamp}.mp4"

    env = os.environ.copy()
    ltx_repo = ROOT / "ltx_service" / "third_party" / "LTX-Video"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ltx_repo)])
    env.setdefault("LTX_REPO_DIR", str(ltx_repo))
    env.setdefault(
        "LTX_MODEL_PATH",
        str(
            ROOT
            / "ltx_service"
            / "models"
            / "ltx"
            / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
        ),
    )
    env.setdefault("LTX_PIPELINE_CONFIG", "configs/ltxv-2b-0.9.6-distilled.yaml")

    cmd = [
        ltx_py,
        "-u",
        str(worker),
        "--mode",
        mode,
        "--prompt",
        args.prompt,
        "--output",
        str(out_mp4),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--duration-seconds",
        str(args.duration),
        "--seed",
        str(args.seed),
    ]
    if args.negative_prompt:
        cmd.extend(["--negative-prompt", args.negative_prompt])
    if args.image:
        cmd.extend(["--image", args.image])

    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), check=False)
    if proc.returncode != 0:
        return proc.returncode
    print(f"Output: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
