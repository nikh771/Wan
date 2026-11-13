# LTX service — LTX-Video 2B distilled (v0.9.6)

All LTX code lives under `ltx_service/`. The FastAPI app in `wan_service/` only imports
this package for job dispatch and health routes.

**Checkpoint:** `ltxv-2b-0.9.6-distilled-04-25.safetensors`  
**Config:** `configs/ltxv-2b-0.9.6-distilled.yaml` (8 steps, guidance 1, BF16)

## Layout

```text
ltx_service/models/ltx/ltxv-2b-0.9.6-distilled-04-25.safetensors
ltx_service/engines/ltx-env/       # isolated venv (see scripts/runpod_remote_setup_ltx.sh)
ltx_service/third_party/LTX-Video/ # official repo clone
ltx_service/outputs/ltx/           # manual test MP4s
```

## Isolation

- API uses `LTX_PYTHON` subprocess only (`worker_main.py`).
- Wan `venv` and `requirements-gpu.txt` are **not** modified for LTX.

## Manual test

```bash
source ltx_service/engines/ltx-env/bin/activate
python scripts/test_ltx.py --prompt "..." --duration 5
```

## Health

`GET /video/engines/ltx/health` (also under `/api/v1/video/engines/ltx/health`)
