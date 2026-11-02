"""
LTX-Video 2B I2V worker (run with LTX_PYTHON in an isolated venv).

Clones/setup: scripts/runpod_remote_setup_ltx.sh
Official CLI: {LTX_REPO_DIR}/inference.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def _repo_dir() -> Path:
    raw = os.environ.get("LTX_REPO_DIR", "").strip()
    if raw:
        return Path(raw).resolve()
    return (Path(__file__).resolve().parents[3] / "third_party" / "LTX-Video").resolve()


def _pipeline_config(repo: Path) -> str:
    cfg = os.environ.get("LTX_PIPELINE_CONFIG", "").strip()
    if cfg:
        path = Path(cfg)
        if not path.is_absolute():
            path = repo / cfg
        return str(path)
    default = repo / "configs" / "ltxv-2b-0.9.6-dev.yaml"
    return str(default)


def _align_num_frames(n: int) -> int:
    """LTX expects num_frames divisible by 8 + 1 (e.g. 9, 17, 25)."""
    if n < 9:
        return 9
    return ((n - 1) // 8) * 8 + 1


def _run_cli(args: argparse.Namespace) -> dict:
    repo = _repo_dir()
    inference_py = repo / "inference.py"
    if not inference_py.is_file():
        raise FileNotFoundError(
            f"LTX inference.py not found at {inference_py}. "
            "Clone https://github.com/Lightricks/LTX-Video and set LTX_REPO_DIR."
        )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_path.parent
    num_frames = _align_num_frames(int(args.num_frames))
    seed = int(args.seed) if args.seed >= 0 else 42
    pipeline = _pipeline_config(repo)

    cmd = [
        sys.executable,
        str(inference_py),
        "--prompt",
        args.prompt,
        "--conditioning_media_paths",
        str(Path(args.image).resolve()),
        "--conditioning_start_frames",
        "0",
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_frames",
        str(num_frames),
        "--seed",
        str(seed),
        "--pipeline_config",
        pipeline,
        "--output_path",
        str(out_dir),
    ]
    if args.negative_prompt:
        cmd.extend(["--negative_prompt", args.negative_prompt])

    print(f"[ltx] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-1500:]
        raise RuntimeError(f"LTX inference.py failed (exit {result.returncode}): {tail}")

    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise RuntimeError("LTX inference did not produce an MP4 in the output directory.")
    if mp4s[0].resolve() != out_path.resolve():
        shutil.move(str(mp4s[0]), str(out_path))

    return {
        "engine": "ltx",
        "output": str(out_path),
        "width": int(args.width),
        "height": int(args.height),
        "num_frames": num_frames,
        "pipeline_config": pipeline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LTX-Video I2V worker")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration-seconds", type=float, default=2.0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args()

    if args.num_frames <= 0:
        args.num_frames = max(9, int(round(args.duration_seconds * args.fps)))

    try:
        meta = _run_cli(args)
        print(json.dumps({"status": "ok", "metadata": meta}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "message": str(exc)}))
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
