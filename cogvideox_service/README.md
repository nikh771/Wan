# CogVideoX service — CogVideoX-2B (T2V POC)

All CogVideoX code lives under `cogvideox_service/`. The FastAPI app in `wan_service/`
only imports this package for job dispatch and health routes.

**Model:** [`zai-org/CogVideoX-2b`](https://huggingface.co/zai-org/CogVideoX-2b)  
**Pipeline:** Diffusers `CogVideoXPipeline` (text-to-video)  
**License:** Apache-2.0 (2B only — do not conflate with CogVideoX-5B)

## Layout

```text
cogvideox_service/models/cogvideox/CogVideoX-2b/   # HF Diffusers snapshot
cogvideox_service/engines/cogvideox-env/           # isolated venv (recommended)
cogvideox_service/outputs/                         # manual test MP4s
```

## Isolation

- API uses `COGVIDEOX_PYTHON` subprocess only (`worker_main.py`).
- Wan `venv` and LTX env are **not** modified for CogVideoX weights loading.
- Do **not** load Wan and CogVideoX on the GPU at the same time.
- Keep `WAN_PERSISTENT_INFERENCE=false` when swapping engines on one GPU.

On disk-constrained hosts, `COGVIDEOX_PYTHON` may point at the existing LTX env
(compatible torch/diffusers) instead of installing a second CUDA stack.

## Setup

```bash
bash scripts/runpod_remote_setup.sh cogvideox
# or:
bash scripts/runpod_remote_setup_cogvideox.sh
```

Download is **explicit** (never at FastAPI import/startup):

```bash
bash scripts/download_cogvideox_2b.sh
```

## Manual test

```bash
python scripts/test_cogvideox.py
```

Default config matches the official CogVideoX-2B examples:

| Setting | Value |
|---------|-------|
| width × height | 720 × 480 |
| frames | 49 |
| fps | 8 |
| duration | 6.125 s |
| steps | 50 |
| guidance_scale | 6 |
| seed | 12345 |
| memory_mode | memory (CPU offload + VAE tiling) |

## Health

`GET /video/engines/cogvideox/health`  
(also `/api/v1/video/engines/cogvideox/health`)

## Job API

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -F "engine_type=cogvideox" \
  -F "task=text_to_video" \
  -F "prompt=Slow cinematic head turn." \
  -F "duration_seconds=6.125" \
  -F "width=720" -F "height=480" -F "fps=8" \
  -F "seed=12345"
```

## I2V (not in this POC)

`CogVideoXImageToVideoAdapter` is a stub only. Real I2V requires a **separate**
I2V checkpoint and `CogVideoXImageToVideoPipeline`. Do not fake with zoom/pan/FFmpeg.

## License metadata

See `cogvideox_service/license.json`.
