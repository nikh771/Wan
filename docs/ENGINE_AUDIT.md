# Multi-engine implementation audit (Phase 1)

## Existing components (reuse)

| Area | Location | Notes |
|------|----------|--------|
| Create job API | `wan_service/apps/jobs/api/routes.py` | Multipart image + prompt + duration |
| Job queue | `wan_service/apps/jobs/services/manager.py` | Single active job, progress, logs |
| Wan inference | `wan_service/apps/jobs/services/inference.py` | Unchanged subprocess / warm worker |
| Progress parsing | `wan_service/apps/jobs/services/progress_parser.py` | Wan log lines |
| Post-process | `wan_service/utils/ffmpeg.py` | Trim + validate MP4 |
| Auth | `wan_service/utils/auth.py` | Bearer token |
| Health | `wan_service/core/runtime.py` | Wan CUDA readiness |
| Config | `core/config.py`, `wan_service/settings.py`, `ltx_service/settings.py` | `.env` driven |
| UI loader | `wan_service/static/job_loader.html` | Poll job status |

No Page Analysis service lives in this repo; prompts are passed on create-job.

## New components (this phase)

- `engine_type` on create-job (`wan` \| `ltx` \| `cogvideox`)
- `wan_service/engine_service.py` — thin wrapper over `run_wan_inference`
- `ltx_service/` — LTX engine (config, adapter, runner, health, `engine_service.py`)
- `cogvideox_service/` — CogVideoX engine (same pattern)
- `core/engine_dispatch.py` — readiness + dispatch
- `ltx_service/worker_main.py` — LTX_PYTHON subprocess entry (official repo on `PYTHONPATH`)
- Split requirements + RunPod setup scripts per engine

Deferred (later phases): hybrid router, per-shot jobs, fallback chain, video normalizer for merger.

## Files modified (minimal Wan touch)

- `routes.py`, `manager.py`, `config.py`, `runtime.py` (health engines)
- **Not modified:** `inference.py`, `persistent_worker.py`, `generate.py`, Wan subprocess flags

## Dependency conflicts

| Engine | Environment | Risk |
|--------|-------------|------|
| Wan | `venv` + `requirements-gpu.txt` + flash-attn | Heavy torch stack |
| LTX | `ltx_service/engines/ltx-env` + `ltx_service/requirements-gpu.txt` | Different torch/diffusers pins — **do not merge into Wan venv** |
| CogVideoX | `cogvideox_service/engines/cogvideox-env` | Isolated env like LTX |

Do not load Wan and LTX weights in the same process.

## GPU memory

- Wan TI2V-5B: ~14–24+ GB VRAM typical
- LTX 2B: lighter but still GPU; run in separate process, unload after job
- CogVideoX 2B: memory-offload mode on ~24GB cards
