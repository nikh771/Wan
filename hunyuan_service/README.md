# HunyuanVideo-1.5 (MS2 engine)

Isolated like LTX / CogVideoX: dedicated venv, checkpoints under `hunyuan_service/models/`, upstream code under `hunyuan_service/third_party/HunyuanVideo-1.5/`.

## Setup (GPU host)

```bash
bash scripts/runpod_remote_setup.sh hunyuan
```

## Health

```bash
curl -s "http://127.0.0.1:8080/health?engine_type=hunyuan"
```

## Jobs

```bash
curl -X POST http://127.0.0.1:8080/video/generate \
  -H "Authorization: Bearer $WAN_WORKER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mode":"i2v","engine_type":"hunyuan","prompt":"...","duration_seconds":5,"image_url":"https://..."}'
```

Defaults (see `.env.example`): `480p`, offloading on, prompt rewrite **off**, I2V step-distill on (8 steps).

Manual smoke:

```bash
python scripts/test_hunyuan.py --prompt "A slow cinematic head turn."
```