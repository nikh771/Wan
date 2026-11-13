# Multi-engine video service

FastAPI job API with pluggable engines:

| Package | Engine | Role |
|---------|--------|------|
| `wan_service/` | **Wan** 2.2 TI2V-5B | GPU image-to-video (default) |
| `ltx_service/` | **LTX** 2B distilled | GPU text/image-to-video (isolated venv) |
| `storyboard_service/` | **Storyboard** | CPU pan/zoom (no diffusion) |

Shared wiring lives in `core/` (config, types, dispatch). One API process routes jobs by `engine_type`.

## Architecture

```text
./
├── core/                    # config, types, engine_dispatch, logging
├── wan_service/             # FastAPI app + Wan assets
│   ├── third_party/Wan2.2/
│   ├── models/Wan2.2-TI2V-5B/
│   └── apps/                # jobs API, inference, UI
├── ltx_service/
│   ├── third_party/LTX-Video/
│   ├── models/ltx/
│   └── engines/ltx-env/     # optional isolated Python env
├── storyboard_service/      # renderer + engine_service
├── scripts/                 # setup & download helpers
├── tests/
└── wan_jobs/                # job workspace (configurable)
```

## Prerequisites

- **GPU hosts (Wan, LTX):** Linux, NVIDIA drivers, CUDA-capable PyTorch, FFmpeg/ffprobe.
- **Storyboard-only:** Linux or macOS with FFmpeg; no GPU required.
- Python 3.10+ (3.11 recommended for GPU stacks).
- Git.
- Disk: ~32 GB for Wan TI2V-5B weights; additional space for LTX (~single safetensors) and job outputs.

Wan TI2V-5B is documented for 720p @ 24 FPS; plan for ~24 GB VRAM for comfortable full-GPU use (smaller GPUs can use offload settings in `.env`).

---

## 1. Clone the repository

```bash
git clone -b dev <YOUR_REPOSITORY_URL>
cd Wan
```

Replace `<YOUR_REPOSITORY_URL>` and directory name as needed.

---

## 2. Core setup (required for all engines)

Core steps install the **API virtual environment**, shared dependencies, and `.env` values every engine relies on. Install **engine-specific** stacks in sections 3–5 only for the engines you use.

### 2.1 API virtual environment

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

Activate `venv` whenever you run the API or storyboard jobs.

### 2.2 Install API Python dependencies

For a **minimal API + storyboard** host (no Wan/LTX torch in this venv):

```bash
python -m pip install -r requirements-storyboard.txt
```

For a **Wan GPU worker** (API + Wan inference in one venv), install CUDA PyTorch first, then GPU requirements:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps --only-binary=flash-attn -r requirements-gpu-flash.txt
```

The root `requirements.txt` remains the canonical manifest for local/dev installs that mirror production.

Verify CUDA (Wan/LTX GPU hosts):

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

**Automated core + Wan on RunPod:** `bash scripts/runpod_remote_setup.sh wan`  
**Dispatcher:** `bash scripts/runpod_remote_setup.sh [wan|ltx|storyboard]`

### 2.3 Core configuration (`.env`)

```bash
cp .env.example .env
chmod 600 .env
```

Generate a secure API key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it as `WAN_WORKER_API_KEY` in `.env`. Never commit `.env`.

**Shared / API variables** (all engines):

| Variable | Purpose |
|----------|---------|
| `APP_NAME`, `APP_VERSION`, `LOG_LEVEL` | Service metadata |
| `WAN_WORKER_HOST`, `WAN_WORKER_PORT` | Uvicorn bind |
| `WAN_WORKER_API_KEY` | Bearer auth for job routes |
| `WAN_JOBS_DIR` | Job input/output workspace |
| `WAN_MAX_QUEUE_SIZE`, `WAN_MAX_IMAGE_BYTES`, `WAN_MAX_PROMPT_CHARS` | Limits |
| `FFMPEG_BINARY`, `FFPROBE_BINARY` | Post-process & storyboard |
| `STORYBOARD_DEFAULT_FPS` | Default FPS when omitted |
| `PYTORCH_CUDA_ALLOC_CONF` | Optional CUDA allocator hint |

Engine-specific variables are documented in sections 3–5. Full template: `.env.example`.

### 2.4 System tools

```bash
ffmpeg -version
ffprobe -version
```

Install via your package manager if missing (e.g. `apt-get install ffmpeg` on Debian/Ubuntu).

### 2.5 Start the API

```bash
source venv/bin/activate
uvicorn wan_service.main:app --host 0.0.0.0 --port 8080
```

**Health (aggregated):**

```bash
curl -s http://127.0.0.1:8080/health | python -m json.tool
```

The payload includes `inference_ready` (Wan), `engine_wan_ready`, `engine_ltx_ready`, and `engine_storyboard_ready` plus reasons when an engine is unavailable.

**Job loader UI:** `http://127.0.0.1:8080/ui/job_loader.html` (or `/` redirect). Use `WAN_WORKER_API_KEY` in the form.

---

## 3. Wan service installation

Wan code and weights live under `wan_service/`. The API calls `generate.py` in a subprocess (or a persistent worker when `WAN_PERSISTENT_INFERENCE=true`).

### 3.1 Layout

```text
wan_service/third_party/Wan2.2/     # vendored upstream (in Git)
wan_service/models/Wan2.2-TI2V-5B/  # Hugging Face weights (not in Git)
```

Verify vendored source after clone:

```bash
test -f wan_service/third_party/Wan2.2/generate.py && echo OK
```

Do **not** re-clone upstream over `wan_service/third_party/Wan2.2` on each deploy — see `wan_service/third_party/Wan2.2/VENDORED.md`.

### 3.2 Wan environment variables

```dotenv
WAN_RUNTIME_MODE=linux_cuda
WAN_INFERENCE_BACKEND=linux_cuda
WAN_REPO_DIR=wan_service/third_party/Wan2.2
WAN_CKPT_DIR=wan_service/models/Wan2.2-TI2V-5B
WAN_GENERATE_SCRIPT=wan_service/third_party/Wan2.2/generate.py
WAN_PYTHON=venv/bin/python
WAN_CUDA_DEVICE=0
WAN_PERSISTENT_INFERENCE=true
WAN_WORKER_LOAD_TIMEOUT_SEC=900
# Optional: WAN_INFERENCE_SAMPLE_STEPS, WAN_INFERENCE_FRAME_NUM, GPU-only flags
```

### 3.3 Download TI2V-5B weights

```bash
source venv/bin/activate
python -m pip install "huggingface_hub==0.36.2"
bash scripts/download_ti2v_5b_weights.sh
# Or: hf download Wan-AI/Wan2.2-TI2V-5B --local-dir wan_service/models/Wan2.2-TI2V-5B
```

```bash
ls -lah wan_service/models/Wan2.2-TI2V-5B/
```

### 3.4 Setup script

```bash
bash scripts/runpod_remote_setup.sh wan
```

### 3.5 Verify Wan

- `/health` → `inference_ready: true`, `engine_wan_ready: true`
- Create job (default `engine_type=wan`):

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -F "image=@input.png;type=image/png" \
  -F "prompt=A cinematic scene with gentle natural movement" \
  -F "duration_seconds=5"
```

When `WAN_PERSISTENT_INFERENCE=true`, wait for `persistent_worker_ready` in `/health` before expecting fastest job turnaround.

---

## 4. LTX service installation

LTX runs in a **separate Python environment** so its PyTorch stack does not conflict with Wan.

### 4.1 Layout

```text
ltx_service/third_party/LTX-Video/   # git clone
ltx_service/models/ltx/                # ltxv-2b-0.9.6-distilled-04-25.safetensors
ltx_service/engines/ltx-env/           # isolated venv (recommended)
```

### 4.2 LTX environment variables

```dotenv
LTX_PYTHON=ltx_service/engines/ltx-env/bin/python
LTX_ENV_DIR=ltx_service/engines/ltx-env
LTX_REPO_DIR=ltx_service/third_party/LTX-Video
LTX_MODEL_PATH=ltx_service/models/ltx/ltxv-2b-0.9.6-distilled-04-25.safetensors
LTX_PIPELINE_CONFIG=configs/ltxv-2b-0.9.6-distilled.yaml
```

Keep `WAN_PERSISTENT_INFERENCE=false` on hosts that swap Wan/LTX unless you manage GPU memory explicitly.

### 4.3 Setup script

```bash
bash scripts/runpod_remote_setup.sh ltx
```

This creates `ltx_service/engines/ltx-env`, clones [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video), installs `pip install -e ".[inference]"`, and downloads the distilled checkpoint via `scripts/download_ltx_2b_distilled.sh`.

### 4.4 Manual smoke test

```bash
python scripts/test_ltx.py \
  --prompt "A character slowly turns toward the camera." \
  --duration 5 --width 704 --height 384 --fps 24
```

Outputs: `ltx_service/outputs/ltx/`.

### 4.5 Verify LTX

```bash
curl -s -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  http://127.0.0.1:8080/api/v1/video/engines/ltx/health | python -m json.tool
```

**T2V job:**

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -F "engine_type=ltx" \
  -F "ltx_mode=text_to_video" \
  -F "prompt=Slow cinematic head turn." \
  -F "duration_seconds=5" -F "width=704" -F "height=384" -F "fps=24"
```

**I2V job:** add `-F "ltx_mode=image_to_video"` and `-F "image=@panel.png"`.

Details: `ltx_service/README.md`.

---

## 5. Storyboard service installation

CPU-only pan/zoom; no model weights. Uses the **same API `venv`** as long as storyboard deps are installed.

### 5.1 Dependencies

```bash
source venv/bin/activate
python -m pip install -r requirements-storyboard.txt
# or: python -m pip install -r storyboard_service/requirements.txt
```

### 5.2 Setup script

```bash
bash scripts/runpod_remote_setup.sh storyboard
```

### 5.3 Verify storyboard

`/health` → `engine_storyboard_ready: true`.

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -F "engine_type=storyboard" \
  -F "image=@panel.png;type=image/png" \
  -F "prompt=ignored for stage-1 motion" \
  -F "duration_seconds=5"
```

Details: `storyboard_service/README.md`.

---

## 6. Using the job API

All engines share the same endpoints. Set `engine_type` to `wan` (default), `ltx`, or `storyboard`.

| Endpoint | Description |
|----------|-------------|
| `POST /jobs` | Create job (multipart image + prompt) |
| `GET /jobs/{id}` | Status and logs |
| `GET /jobs/{id}/progress` | Lightweight poll (`?api_key=` supported) |
| `GET /jobs/{id}/video` | Download MP4 when `completed` |

**Progress poll:**

```bash
curl -s "http://127.0.0.1:8080/jobs/JOB_ID/progress?api_key=$WAN_WORKER_API_KEY" \
  | python -m json.tool
```

**Download video:**

```bash
curl -fSL http://127.0.0.1:8080/jobs/JOB_ID/video \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -o output.mp4
```

Invalid job IDs return `revoke_polling: true` in error JSON — stop polling.

---

## 7. Development workflow

1. Edit code on your branch; keep engine code inside `wan_service/`, `ltx_service/`, or `storyboard_service/`.
2. Update the relevant requirements file (`requirements.txt`, `requirements-gpu.txt`, `ltx_service/requirements-gpu.txt`, `storyboard_service/requirements.txt`).
3. Push and pull on the GPU host (or use Remote SSH).
4. Re-run only the setup sections for engines you changed.
5. `curl /health` and run a short job per engine.

Do not commit `.env`, checkpoints, or `wan_jobs/` artifacts.

---

## 8. Repository and dependency policy

| Path | Contents |
|------|----------|
| `requirements.txt` | Canonical Python deps (dev) |
| `requirements-gpu.txt` | Wan GPU API/worker venv |
| `requirements-storyboard.txt` | API + OpenCV for storyboard |
| `ltx_service/requirements-gpu.txt` | Notes for LTX isolated env |
| `wan_service/third_party/Wan2.2/` | Upstream source (vendored) |
| `wan_service/models/` | Wan weights |
| `ltx_service/models/` | LTX weights |
| `core/` | Shared configuration and dispatch |

Weights and secrets stay on the deployment machine; they are not stored in Git.

---

## License and attribution

- [Wan 2.2](https://github.com/Wan-Video/Wan2.2) — review upstream license and model terms.
- [LTX-Video](https://github.com/Lightricks/LTX-Video) — review upstream license for LTX deployments.
