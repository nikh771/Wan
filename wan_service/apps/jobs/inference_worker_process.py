"""
Long-lived TI2V-5B worker: load models once, accept JSON-line jobs on stdin.

Started by wan_service (WAN_PYTHON subprocess). Protocol:
  → {"type":"init", ...}
  ← {"type":"log","line":"..."}  (optional, human-readable)
  ← {"type":"ready"}
  → {"type":"generate","job_id","image_path","prompt","output_mp4","size", ...}
  ← log lines + tqdm (plain text) on stdout
  ← {"type":"done","job_id":"..."} or {"type":"error","message":"..."}
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from PIL import Image


def _emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def _read_init() -> dict[str, Any]:
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("Worker stdin closed before init")
    msg = json.loads(line)
    if msg.get("type") != "init":
        raise RuntimeError(f"Expected init message, got {msg.get('type')!r}")
    return msg


def _load_pipeline(init: dict[str, Any]):
    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["ti2v-5B"]
    ckpt_dir = init["ckpt_dir"]
    gpu_only = bool(init.get("gpu_only", True))
    logging.info("Creating WanTI2V pipeline.")
    pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=int(init.get("device_id", 0)),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=not gpu_only,
        init_on_cpu=not gpu_only,
        convert_model_dtype=bool(init.get("convert_model_dtype", True)),
    )
    return pipeline, cfg


def _run_generate(pipeline, cfg, msg: dict[str, Any]) -> None:
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS
    from wan.utils.utils import save_video

    size_key = msg["size"]
    if size_key not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size {size_key!r}")

    img = Image.open(msg["image_path"]).convert("RGB")
    output_mp4 = Path(msg["output_mp4"])
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    frame_num = msg.get("frame_num")
    if frame_num is None:
        frame_num = cfg.frame_num
    sampling_steps = msg.get("sample_steps")
    if sampling_steps is None:
        sampling_steps = cfg.sample_steps

    logging.info("Generating video ...")
    video = pipeline.generate(
        msg["prompt"],
        img=img,
        size=SIZE_CONFIGS[size_key],
        max_area=MAX_AREA_CONFIGS[size_key],
        frame_num=int(frame_num),
        shift=cfg.sample_shift,
        sample_solver="unipc",
        sampling_steps=int(sampling_steps),
        guide_scale=cfg.sample_guide_scale,
        seed=int(msg.get("seed", -1)),
        offload_model=not bool(msg.get("gpu_only", True)),
    )
    if video is None:
        raise RuntimeError("Pipeline returned no video tensor")

    logging.info("Saving generated video to %s", output_mp4)
    save_video(
        tensor=video[None],
        save_file=str(output_mp4),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info("Finished.")


def main() -> int:
    _configure_logging()
    try:
        init = _read_init()
        pipeline, cfg = _load_pipeline(init)
        _emit({"type": "ready"})
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "error", "message": f"Init failed: {exc}"})
        traceback.print_exc(file=sys.stderr)
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Ignoring non-JSON stdin line")
            continue

        if msg.get("type") == "shutdown":
            _emit({"type": "shutdown_ack"})
            return 0

        if msg.get("type") != "generate":
            _emit({"type": "error", "message": f"Unknown message type {msg.get('type')!r}"})
            continue

        job_id = msg.get("job_id", "")
        try:
            msg["gpu_only"] = init.get("gpu_only", True)
            _run_generate(pipeline, cfg, msg)
            _emit({"type": "done", "job_id": job_id})
        except Exception as exc:  # noqa: BLE001
            _emit({"type": "error", "job_id": job_id, "message": str(exc)})
            traceback.print_exc(file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
