#!/usr/bin/env python3
"""Manual CogVideoX-2B T2V benchmark (runs worker in COGVIDEOX_PYTHON).

Separate service smoke test — same role as scripts/test_ltx.py for LTX.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROMPT = (
    "The character slowly raises the kunai in his right hand and moves it into a "
    "defensive position. His wrist rotates naturally and his fingers tighten around "
    "the handle. His shoulders shift slightly. His hair and scarf move gently in the "
    "wind. A few leaves fall through the background. The camera remains mostly "
    "stationary with a subtle cinematic push-in. Smooth natural character motion, "
    "stable composition, detailed anime rendering."
)


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


def _resolve_python() -> str:
    raw = os.environ.get("COGVIDEOX_PYTHON", "").strip()
    if not raw:
        dedicated = ROOT / "cogvideox_service" / "engines" / "cogvideox-env" / "bin" / "python"
        if dedicated.is_file():
            return str(dedicated)
        ltx = ROOT / "ltx_service" / "engines" / "ltx-env" / "bin" / "python"
        if ltx.is_file():
            return str(ltx)
        return str(ROOT / "venv" / "bin" / "python")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Test CogVideoX-2B (separate service)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--memory-mode", choices=("memory", "normal"), default="memory")
    parser.add_argument("--negative-prompt", default="")
    args = parser.parse_args()

    py = _resolve_python()
    worker = ROOT / "cogvideox_service" / "worker_main.py"
    out_dir = ROOT / "cogvideox_service" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_mp4 = out_dir / f"cogvideox_t2v_{stamp}.mp4"

    model_path = os.environ.get(
        "COGVIDEOX_MODEL_PATH",
        "cogvideox_service/models/cogvideox/CogVideoX-2b",
    )
    if not Path(model_path).is_absolute():
        model_abs = str(ROOT / model_path)
    else:
        model_abs = model_path

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["COGVIDEOX_MODEL_PATH"] = model_abs
    env.setdefault("COGVIDEOX_HF_ID", "zai-org/CogVideoX-2b")
    env.setdefault("COGVIDEOX_DTYPE", "float16")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    cmd = [
        py,
        "-u",
        str(worker),
        "--mode",
        "text_to_video",
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
        "--num-frames",
        str(args.num_frames),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--memory-mode",
        args.memory_mode,
    ]
    if args.negative_prompt:
        cmd.extend(["--negative-prompt", args.negative_prompt])

    print("=== CogVideoX-2B benchmark (cogvideox_service) ===", flush=True)
    print(f"Python: {py}", flush=True)
    print(f"Model:  {model_abs}", flush=True)
    print(f"Output: {out_mp4}", flush=True)
    print("Running:", " ".join(cmd[:6]), "...", flush=True)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT), check=False, capture_output=True, text=True)
    total = time.perf_counter() - t0
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        print(f"FAILED exit={proc.returncode} total={total:.1f}s", file=sys.stderr)
        return proc.returncode

    # Parse last JSON ok line for metrics
    result = {}
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"status"' in line:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ok":
                result = payload.get("result") or {}
                break

    bench = result.get("benchmark") or {}
    print("\n=== Summary ===")
    print("CogVideoX-2B")
    print(f"GPU: {bench.get('gpu') or result.get('gpu')}")
    print(f"CUDA: {bench.get('cuda')}")
    print(f"PyTorch: {bench.get('pytorch')}")
    print(f"Diffusers: {bench.get('diffusers')}")
    print(f"Resolution: {result.get('width')}x{result.get('height')}")
    print(f"Frames: {result.get('frames')}")
    print(f"FPS: {result.get('fps')}")
    print(f"Duration: {result.get('duration_seconds')} sec")
    print(f"Steps: {result.get('steps')}")
    print(f"Guidance: {result.get('guidance_scale')}")
    print(f"Seed: {result.get('seed')}")
    print(f"Memory mode: {result.get('memory_mode')}")
    print(f"Load time: {result.get('load_time_seconds')} s")
    print(f"Generation time: {result.get('generation_time_seconds')} s")
    print(f"Total time: {result.get('total_time_seconds') or round(total, 3)} s")
    print(f"Peak VRAM: {result.get('vram_peak_gb')} GB")
    print(f"Output: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
