# Wan_service

Standalone FastAPI microservice for **self-hosted Wan 2.2 TI2V-5B** image-to-video generation.

Uses the official [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) `generate.py` CLI (no third-party video APIs).

Layout: `core/` (config, runtime), `apps/` (health, jobs), `utils/`, `routers.py`, `main.py`.

## Dependencies (two layers)

| File | Install where | Contents |
|------|----------------|----------|
| `requirements.txt` | **macOS + Linux** | FastAPI, config, jobs, Pillow, FFmpeg hooks, tests — **no torch, no flash_attn** |
| `requirements-linux-cuda.txt` | **Linux NVIDIA only** | `-r requirements.txt` + upstream `third_party/Wan2.2/requirements.txt` (includes **flash_attn**) |

**Do not** run `pip install -r requirements-linux-cuda.txt` on macOS.  
**Do not** run `pip install -r third_party/Wan2.2/requirements.txt` on macOS.

Passing API tests or starting uvicorn does **not** mean inference works. Check `/health` → `inference_ready`.

## Runtime modes

| Mode | Purpose |
|------|---------|
| `mac_dev` | API + tests on Apple Silicon; **`inference_ready` is always false** (API/test-only) |
| `linux_cuda` | Production inference on Linux + NVIDIA CUDA |

Set `WAN_RUNTIME_MODE` in `.env`, or auto-detect (Darwin → `mac_dev`, else `linux_cuda`).

**Experimental Wan on Apple MPS (TI2V-5B):** see [`docs/ti2v_5b_mps_port_plan.md`](docs/ti2v_5b_mps_port_plan.md).  
Upstream is CUDA-oriented today; `/health` must not report `inference_ready` until an MPS smoke test passes with real weights.  
Operator probe (no weights): `python scripts/mps_probe_ops.py`.

---

## Mac API development (Apple Silicon)

```bash
cd /path/to/Wan_service
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Wan 2.2 source is vendored at third_party/Wan2.2 (includes Mac/CUDA compatibility patches).

cp .env.example .env
# WAN_WORKER_API_KEY=...
# WAN_RUNTIME_MODE=mac_dev

uvicorn wan_service.main:app --host 127.0.0.1 --port 8080
```

Health:

```bash
curl -sS http://127.0.0.1:8080/health | python3 -m json.tool
```

Expect `runtime_mode: mac_dev`, `api_available: true`, `inference_ready: false`, and a reason mentioning **API/test-only**.  
`POST /jobs` returns **503** in this mode.

Tests:

```bash
pytest -q
```

(`pytest.ini` sets `pythonpath = .` — no `PYTHONPATH` needed.)

Optional: install PyTorch with MPS only if you want `/health` to report `mps_available`; it does **not** enable inference in `mac_dev`.

---

## Linux NVIDIA inference

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-linux-cuda.txt
```

Wan 2.2 source is included at `third_party/Wan2.2`. Download weights (**manual**, not automatic):

```bash
chmod +x scripts/download_ti2v_5b_weights.sh
./scripts/download_ti2v_5b_weights.sh
```

`.env` example:

```env
WAN_RUNTIME_MODE=linux_cuda
WAN_WORKER_API_KEY=your_secret
WAN_REPO_DIR=third_party/Wan2.2
WAN_CKPT_DIR=Wan2.2-TI2V-5B
# or WAN_CKPT_DIR=models/Wan2.2-TI2V-5B (weights stay out of Git; use models/ locally)
WAN_PYTHON=.venv/bin/python
WAN_MIN_VRAM_GB=14
```

Before accepting jobs, the service checks (in order): Linux host, `generate.py`, `WAN_PYTHON`, inference packages in **`WAN_PYTHON`** (via subprocess — **flash_attn is never imported by the API process**), checkpoints, FFmpeg, CUDA, VRAM heuristic.

```bash
uvicorn wan_service.main:app --host 0.0.0.0 --port 8080
```

Tune `WAN_MIN_VRAM_GB` (default `14`). Passing the heuristic does **not** guarantee generation succeeds.

### Docker

Build on a CUDA host; image installs upstream Wan deps from the cloned repo plus `requirements.txt`. Mount weights at runtime.

---

## API

| Method | Path | Auth |
|--------|------|------|
| GET | `/health` | No |
| POST | `/jobs` | Bearer `WAN_WORKER_API_KEY` |
| GET | `/jobs/{job_id}` | Bearer |
| GET | `/jobs/{job_id}/video` | Bearer |

## Security

Never commit `.env`, API keys, checkpoints, or `wan_jobs/` outputs.
