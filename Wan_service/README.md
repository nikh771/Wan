# Wan_service

Self-hosted **[Wan 2.2](https://github.com/Wan-Video/Wan2.2)** stack: vendored inference code under `third_party/Wan2.2` (with Mac MPS / Linux CUDA patches) plus a **FastAPI worker** for TI2V-5B image-to-video jobs. No third-party video APIs.

Upstream paper: [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314). Weights: [Hugging Face Wan-AI](https://huggingface.co/Wan-AI/) · [ModelScope](https://modelscope.cn/organization/Wan-AI).

---

## Repository layout

```text
Wan_service/
├── wan_service/          # FastAPI app (core/, apps/, utils/, main.py)
├── third_party/Wan2.2/ # Vendored Wan source + generate.py (patched)
├── models/               # Local checkpoints (gitignored; e.g. Wan2.2-TI2V-5B)
├── scripts/              # Weight download, MPS/TI2V smoke helpers
├── tests/
├── docs/                 # MPS port notes (ti2v_5b_mps_port_plan.md)
├── requirements.txt      # Single dependency file (service + all Wan stacks)
├── .env.example
└── Dockerfile
```

---

## Dependencies

One file: **`requirements.txt`** (FastAPI worker, pytest, Wan core, Speech-to-Video, and Animate packages).

| Platform | Install order |
|----------|----------------|
| **Linux NVIDIA** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` then `pip install -r requirements.txt` |
| **macOS** | `pip install torch torchvision torchaudio` then `pip install -r requirements.txt` (`flash_attn` is skipped on macOS) |

If `flash_attn` fails on Linux, install other packages first and install `flash_attn` last.

Passing API tests or starting uvicorn does **not** mean inference works. Check `GET /health` → `inference_ready`.

---

## Runtime modes

| `WAN_RUNTIME_MODE` | Purpose |
|--------------------|---------|
| `mac_dev` | API + tests on Apple Silicon; **`inference_ready` is always false** (API/test-only) |
| `linux_cuda` | Production inference on Linux + NVIDIA CUDA |

Set in `.env`, or leave empty to auto-detect (Darwin → `mac_dev`, else `linux_cuda`).

**Experimental TI2V on Apple MPS:** see [`docs/ti2v_5b_mps_port_plan.md`](docs/ti2v_5b_mps_port_plan.md). Probe (no weights): `python scripts/mps_probe_ops.py`.

---

## Configuration

Copy `.env.example` to `.env`. Never commit `.env`.

| Variable | Typical value | Notes |
|----------|---------------|--------|
| `WAN_WORKER_API_KEY` | (secret) | Bearer token for job endpoints |
| `WAN_REPO_DIR` | `third_party/Wan2.2` | Wan source root |
| `WAN_CKPT_DIR` | `Wan2.2-TI2V-5B` or `models/Wan2.2-TI2V-5B` | Checkpoints outside Git |
| `WAN_GENERATE_SCRIPT` | (empty) | Defaults to `generate.py` under `WAN_REPO_DIR` |
| `WAN_PYTHON` | `.venv/bin/python` | Interpreter used for inference subprocess |
| `WAN_JOBS_DIR` | `wan_jobs` | Job artifacts (gitignored) |
| `WAN_INFERENCE_BACKEND` | `mac_mps` / `linux_cuda` | Passed to patched `generate.py` |

---

## Mac development (Apple Silicon)

```bash
cd /path/to/Wan_service
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
cp .env.example .env
# WAN_WORKER_API_KEY=...
# WAN_RUNTIME_MODE=mac_dev

uvicorn wan_service.main:app --host 127.0.0.1 --port 8080
```

```bash
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool
pytest -q
```

Expect `runtime_mode: mac_dev`, `api_available: true`, `inference_ready: false`. `POST /jobs` returns **503** in this mode.

---

## Linux NVIDIA inference

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Download TI2V-5B weights (manual)

```bash
chmod +x scripts/download_ti2v_5b_weights.sh
./scripts/download_ti2v_5b_weights.sh
# Optional: WAN_CKPT_DIR=models/Wan2.2-TI2V-5B ./scripts/download_ti2v_5b_weights.sh
```

| Model (this service) | Hugging Face | ModelScope |
|----------------------|--------------|------------|
| **TI2V-5B** | [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | [Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) |

Other Wan 2.2 variants (T2V-A14B, I2V-A14B, S2V, Animate) are supported by vendored `generate.py`; see [upstream docs](https://github.com/Wan-Video/Wan2.2).

### Run the worker

```env
WAN_RUNTIME_MODE=linux_cuda
WAN_WORKER_API_KEY=your_secret
WAN_REPO_DIR=third_party/Wan2.2
WAN_CKPT_DIR=Wan2.2-TI2V-5B
WAN_PYTHON=.venv/bin/python
WAN_MIN_VRAM_GB=14
```

Readiness checks (in order): Linux host, `generate.py`, `WAN_PYTHON`, inference packages in **`WAN_PYTHON`** (API never imports `flash_attn` directly), checkpoints, FFmpeg, CUDA, VRAM heuristic.

```bash
uvicorn wan_service.main:app --host 0.0.0.0 --port 8080
```

### Docker

Build on a CUDA host. Image installs `requirements.txt`; **mount** checkpoint dir at runtime (`WAN_CKPT_DIR`).

---

## HTTP API

| Method | Path | Auth |
|--------|------|------|
| GET | `/health` | No |
| POST | `/jobs` | Bearer `WAN_WORKER_API_KEY` |
| GET | `/jobs/{job_id}` | Bearer |
| GET | `/jobs/{job_id}/video` | Bearer |

---

## Direct CLI (vendored `generate.py`)

From `Wan_service` with checkpoints on disk:

```bash
cd third_party/Wan2.2
# Text + image → video (TI2V-5B, 720P)
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ../../Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --image examples/i2v_input.JPG \
  --prompt "Your prompt here."
```

Portrait preset: `704*1280`. Set `WAN_INFERENCE_BACKEND=mac_mps` or `linux_cuda` in the environment when using patched backends.

---

## Upstream regression tests (optional)

Wan’s multi-model GPU test script lives under `third_party/Wan2.2/tests/`. Put checkpoints in a folder and run:

```bash
cd third_party/Wan2.2
bash ./tests/test.sh <local model dir> <gpu count>
```

---

## Security

Never commit `.env`, API keys, `models/` / checkpoint dirs, `wan_jobs/`, or generated videos.
