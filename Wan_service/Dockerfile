# Wan 2.2 TI2V-5B GPU service — build on a CUDA host (e.g. RunPod).
# Model weights are NOT downloaded in this image; mount WAN_CKPT_DIR at runtime.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY wan_service /app/wan_service
COPY third_party/Wan2.2 /app/third_party/Wan2.2
RUN pip install --no-cache-dir -r /app/requirements.txt

ENV WAN_RUNTIME_MODE=linux_cuda
ENV WAN_REPO_DIR=/app/third_party/Wan2.2
ENV WAN_CKPT_DIR=/workspace/Wan2.2-TI2V-5B
ENV WAN_JOBS_DIR=/workspace/wan_jobs
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "wan_service.main:app", "--host", "0.0.0.0", "--port", "8080"]
