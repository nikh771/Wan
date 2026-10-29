# Wan Service

A FastAPI-based video generation service powered by [Wan 2.2 TI2V-5B](https://github.com/Wan-Video/Wan2.2), supporting image-to-video and text-to-video generation through a REST API.

## Architecture

The project separates the API service, upstream Wan source code, and model checkpoints.

```text
./                              # project root (after clone)
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── wan_service/                  # FastAPI application
├── third_party/
│   └── Wan2.2/                   # Wan 2.2 source (vendored in this repo)
├── models/
│   └── Wan2.2-TI2V-5B/           # Model checkpoints (not in Git)
├── scripts/                      # Setup and download scripts
├── tests/
└── wan_jobs/                     # Job inputs and outputs
```

## Prerequisites

- Linux machine with an NVIDIA GPU for CUDA inference.
- Python 3.11 or a compatible version supported by the dependencies.
- Git.
- NVIDIA drivers and a compatible CUDA-enabled PyTorch installation.
- FFmpeg and FFprobe.
- Sufficient disk space for approximately 32 GB of model checkpoints, Python dependencies, and generated videos.

Wan 2.2 TI2V-5B supports text-to-video and image-to-video generation. The official project documents 720p generation at 24 FPS and recommends at least 24 GB of GPU VRAM for its standard 720p configuration.

## 1. Clone Wan Service

Clone your project repository and switch to the development branch:

```bash
git clone -b dev <YOUR_REPOSITORY_URL>
cd Wan
```

Replace `<YOUR_REPOSITORY_URL>` with your Git repository URL and `Wan` with your clone directory name if different.

## 2. Create the Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
```

Activate the virtual environment whenever you work with the service.

## 3. Install Python Dependencies

The root `requirements.txt` is the **canonical Python dependency manifest** for this project.

For Linux NVIDIA deployment, install a compatible CUDA-enabled PyTorch build first. For example, the official PyTorch CUDA 12.8 wheel index can be used when compatible with your driver and environment:

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Then install the project's dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

The GPU availability check should return `True` on a correctly configured CUDA host.

Some optional packages, including native CUDA extensions, may require additional build configuration. Resolve any installation errors before proceeding.

## 4. Configure Environment Variables

Create a local environment file from the template:

```bash
cp .env.example .env
chmod 600 .env
```

Configure the appropriate values in `.env`:

```dotenv
APP_NAME="Wan 2.2 GPU Service"
APP_VERSION=0.1.0
LOG_LEVEL=INFO

WAN_RUNTIME_MODE=linux_cuda
WAN_INFERENCE_BACKEND=linux_cuda

WAN_REPO_DIR=third_party/Wan2.2
WAN_CKPT_DIR=models/Wan2.2-TI2V-5B
WAN_GENERATE_SCRIPT=third_party/Wan2.2/generate.py
WAN_PYTHON=venv/bin/python

WAN_WORKER_HOST=127.0.0.1
WAN_WORKER_PORT=8080
WAN_WORKER_API_KEY=REPLACE_WITH_A_RANDOM_SECRET

WAN_JOBS_DIR=wan_jobs
WAN_MAX_QUEUE_SIZE=8
WAN_MAX_IMAGE_BYTES=10485760
WAN_MAX_PROMPT_CHARS=8000

FFMPEG_BINARY=ffmpeg
FFPROBE_BINARY=ffprobe

WAN_SIZE_LANDSCAPE=1280*704
WAN_SIZE_PORTRAIT=704*1280
WAN_MIN_VRAM_GB=14.0

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Generate a secure API key on the deployment machine:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Replace the placeholder with the generated value. Never commit `.env` or expose the API key.

The default resolution above matches the standard TI2V-5B landscape configuration. Reduce it to a supported lower resolution if your GPU cannot handle the standard workload. The memory allocator setting can help with fragmentation but does not guarantee that inference will fit in VRAM.

## 5. Install the Wan 2.2 Source Code

Wan 2.2 source code is separate from Python dependencies and model weights.

The upstream source repository is:

https://github.com/Wan-Video/Wan2.2

Ensure the source is available in:

```text
third_party/Wan2.2/
```

This repository **vendors** Wan 2.2 under that path. After clone, verify the entry point:

```bash
test -f third_party/Wan2.2/generate.py \
  && echo "Wan 2.2 source found"
```

If the directory is missing (for example in a minimal checkout), clone upstream without overwriting an existing tree:

```bash
mkdir -p third_party

git clone https://github.com/Wan-Video/Wan2.2.git \
  third_party/Wan2.2
```

Record the upstream source revision for reproducibility:

```bash
git -C third_party/Wan2.2 rev-parse HEAD
```

For this service, use the root `requirements.txt` as the canonical dependency manifest rather than installing upstream `third_party/Wan2.2/requirements*.txt` separately.

## 6. Download the Wan 2.2 TI2V-5B Model

Model checkpoints are downloaded separately from the source repository.

Install the Hugging Face Hub CLI:

```bash
python -m pip install "huggingface_hub==0.36.2"
```

Download the official TI2V-5B checkpoints:

```bash
unset HF_HUB_ENABLE_HF_TRANSFER
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /Wan/models/Wan2.2-TI2V-5B
```

The download is approximately 32 GB. Ensure that the machine has sufficient disk space and network connectivity.

Verify the checkpoint directory:

```bash
ls -lah models/Wan2.2-TI2V-5B/
```

The model directory should contain the configuration, VAE, text encoder, and diffusion model checkpoint files required by the upstream implementation.

**Do not commit model weights to Git.**

## 7. Install System Tools

Ensure FFmpeg and FFprobe are installed:

```bash
ffmpeg -version
ffprobe -version
```

Install them using the system package manager if missing. Python packages such as `imageio-ffmpeg` do not necessarily replace every system-level FFmpeg requirement.

## 8. Start the FastAPI Service

From the project root, activate the virtual environment and run:

```bash
source venv/bin/activate

uvicorn wan_service.main:app \
  --host 0.0.0.0 \
  --port 8080
```

Check service health:

```bash
curl http://127.0.0.1:8080/health
```

The health response reports API availability, inference readiness, CUDA availability, checkpoint detection, and other runtime checks.

A healthy API does not guarantee that every generation request will fit into available GPU memory.

## 9. Generate a Video

The job endpoint accepts an input image, a prompt, and a duration.

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -F "image=@input.png;type=image/png" \
  -F "prompt=A cinematic scene with gentle natural movement" \
  -F "duration_seconds=5"
```

The response contains a job ID.

### Check job status

Replace `JOB_ID` with the returned ID:

```bash
curl http://127.0.0.1:8080/jobs/JOB_ID \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY"
```

### Retrieve the generated video

```bash
curl -fSL http://127.0.0.1:8080/jobs/JOB_ID/video \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -o output.mp4
```

Generated job files are stored under the configured jobs directory. Exact output paths depend on the service implementation and job status.

## 10. Development Workflow

The recommended development workflow is:

1. Edit application code and update `requirements.txt` on the development branch.
2. Commit and push changes to GitHub.
3. Pull the changes onto the GPU machine, or use Cursor Remote SSH to edit and debug the remote code directly.
4. Install any newly added Python dependencies from the root requirements file.
5. Verify `/health`.
6. Run a short inference test before larger generation jobs.

Keep model weights and `.env` on the GPU machine; neither should be committed to Git.

## Repository and Dependency Policy

- `requirements.txt` at the project root is the canonical Python dependency manifest.
- `third_party/Wan2.2/` contains upstream source code, not model weights.
- `models/Wan2.2-TI2V-5B/` contains downloaded model checkpoints.
- `.env` contains deployment-specific configuration and secrets.
- `wan_jobs/` contains temporary inputs and generated artifacts.
- System packages, GPU drivers, CUDA compatibility, source checkouts, and model weights are managed separately from Python package installation.

## License and Attribution

Wan 2.2 is maintained by [Wan-Video](https://github.com/Wan-Video/Wan2.2). Review the upstream license and model terms before redistributing the source, weights, or deploying the service commercially.
