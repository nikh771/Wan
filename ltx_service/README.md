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

## Prefetch text encoder

LTX loads T5 from `PixArt-alpha/PixArt-XL-2-1024-MS` (`text_encoder` + `tokenizer`). Setup runs this automatically; re-run if a job fails with missing `pytorch_model.bin` / `model.safetensors`:

```bash
bash scripts/download_ltx_pixart_text_encoder.sh
```

## VRAM (24GB cards)

On GPUs with under 30GB VRAM (e.g. RTX A5000), the worker keeps the T5 encoder on CPU (`offload_to_cpu`) and disables Florence/Llama prompt enhancement so create+generate fit in memory. Keep `WAN_PERSISTENT_INFERENCE=false` when swapping Wan/LTX on one GPU.

## I2V motion (distilled 2B)

- `LTX_CONDITIONING_STRENGTH=0.7`, `LTX_IMAGE_COND_NOISE_SCALE=0.5`
- `LTX_MOTION_AMPLIFY=2.4` post-boosts deviation from the first frame so motion reads clearly (set `1.0` to disable)
- Send final prompts from your upstream service; this API does not rewrite prompts.

## Long jobs (chunk queue)

If `duration_seconds` exceeds `LTX_CHUNK_SECONDS` (default **10**):
1. Split into ≤10s chunks (LTX frame rule `8N+1`)
2. Chunk 0 uses the request mode/image
3. Each later chunk is **image_to_video** conditioned on the **last frame** of the previous chunk
4. Concatenate chunks (drop duplicate bridge frame) into the final MP4

Logs show `LTX queued chunks=N` then `Generating chunk i/N` and `Concatenating`.

Each chunk’s trailing near-static hold is trimmed before the bridge frame is taken / before concat (`LTX_TRIM_TRAILING_FREEZE`), so the final video does not freeze on the last pose at chunk seams or the ending.

Continuation chunks (1..N):
- Reuse the **same enhanced prompt** as chunk 0
- Condition on the **previous chunk’s last frame**
- Use stronger bridge lock (`LTX_BRIDGE_CONDITIONING_STRENGTH=0.88`, lower noise)
- **Do not** apply motion-amplify (it was degrading quality vs chunk 0)

## Manual test

```bash
source ltx_service/engines/ltx-env/bin/activate
python scripts/test_ltx.py --prompt "..." --duration 5
```

## Health

`GET /video/engines/ltx/health` (also under `/api/v1/video/engines/ltx/health`)
