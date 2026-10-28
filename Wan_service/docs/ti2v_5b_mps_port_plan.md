# Wan 2.2 TI2V-5B — Apple MPS port investigation

**Scope:** Official `ti2v-5B` path only (`generate.py` → `WanTI2V` → TI2V-5B weights).  
**Goal:** Experimental local inference on Apple Silicon (M3, 16 GB unified memory).  
**Non-goals:** Different model, cloud GPU, paid APIs, or claiming readiness before a real smoke test.

**Service policy (unchanged until smoke test passes):**  
`WAN_RUNTIME_MODE=mac_dev` stays API/test-only; `/health` must keep `inference_ready=false` until an automated MPS smoke test succeeds end-to-end with real weights.

---

## 1. TI2V-5B call graph (single-process, default CLI)

| Step | File | Symbol / notes |
|------|------|----------------|
| CLI entry | `third_party/Wan2.2/generate.py` | `generate()` → `elif "ti2v" in args.task` |
| Pipeline | `wan/textimage2video.py` | `WanTI2V`, `generate()` → `i2v()` or `t2v()` |
| Config | `wan/configs/wan_ti2v_5B.py` | `ti2v_5B` (+ `shared_config.py`: `param_dtype=bfloat16`, T5 bf16) |
| Text encoder | `wan/modules/t5.py` | `T5EncoderModel` (service uses `--t5_cpu`) |
| VAE | `wan/modules/vae2_2.py` | `Wan2_2_VAE` encode/decode |
| DiT | `wan/modules/model.py` | `WanModel.from_pretrained(ckpt_dir)` |
| Attention | `wan/modules/attention.py` | **`flash_attention()`** (not the SDPA wrapper) |
| Schedulers | `wan/utils/fm_solvers_unipc.py`, `fm_solvers.py` | UniPC / DPM++ (already has **MPS float64 guard**) |
| Helpers | `wan/utils/utils.py` | `best_output_size`, `masks_like`, `save_video` |
| Tokenizer | `wan/modules/tokenizers.py` | HF `AutoTokenizer` |

**Not on default TI2V-5B single-GPU path:** `distributed/*`, FSDP, sequence parallel, `flash_attn` Python package (if a CUDA-only path were forced), `prompt_extend`, animate/s2v/A14B pipelines.

**Default service subprocess flags (unchanged):**  
`--task ti2v-5B --offload_model True --convert_model_dtype --t5_cpu`

---

## 2. CUDA-specific usage (by file)

### `generate.py`

| Mechanism | Lines (approx.) | MPS note |
|-----------|-----------------|----------|
| `torch.cuda.set_device` | 327 | Multi-GPU only; skip on MPS |
| `dist.init_process_group(backend="nccl")` | 328–332 | Not used in default single-process |
| `torch.cuda.synchronize()` | 565 | Replace with device-agnostic sync helper |
| `device_id` → `cuda:{id}` via `WanTI2V` | 408, 430 | **Must become selectable device** |

### `wan/textimage2video.py` (`WanTI2V`)

| Mechanism | Occurrences | MPS note |
|-----------|-------------|----------|
| `self.device = torch.device(f"cuda:{device_id}")` | 75 | **Blocker** — needs `mps` / `cpu` / `cuda` resolver |
| `torch.cuda.amp as amp` | 13 | Import unused in this file; autocast uses `torch.amp.autocast('cuda', …)` |
| `torch.amp.autocast('cuda', dtype=param_dtype)` | 330, 522 | Use `device_type` from runtime (`mps` / `cuda`) |
| `torch.cuda.empty_cache()` | 365, 565, 583, 587, 606, 615 | **No MPS equivalent** — use `gc.collect()` + optional `torch.mps.synchronize()` |
| `torch.cuda.synchronize()` | 398, 407, 605, 615 | Map to `torch.mps.synchronize()` on MPS |
| `torch.Generator(device=self.device)` | 296, 486 | Supported on MPS in recent PyTorch |
| `dist.barrier()` | 149, 408, 616 | No-op when dist not initialized |

### `wan/modules/model.py` (`WanModel` — TI2V-5B DiT)

| Mechanism | MPS note |
|-----------|----------|
| `@torch.amp.autocast('cuda', enabled=False)` on RoPE | Autocast device tag should follow runtime device |
| `rope_apply` / `sinusoidal_embedding_1d` use **float64** | **High risk on MPS** (no float64); solvers already downgrade MPS float64 → float32 |
| `flash_attention()` in self/cross-attn | **Hard blocker** — asserts `q.device.type == 'cuda'` |
| `@torch.amp.autocast('cuda', dtype=float32)` in blocks | Switch to dynamic `device_type` |

### `wan/modules/attention.py`

| Mechanism | MPS note |
|-----------|----------|
| `flash_attn` / `flash_attn_interface` | **Must not install on Mac** |
| `flash_attention()` | CUDA assert; primary path for `WanModel` |
| `attention()` SDPA fallback | Exists but **unused by TI2V DiT** — viable MPS path if wired with same shapes/dtypes |

**Semantics note:** SDPA without varlen padding masks may differ slightly from `flash_attn_varlen_func` when `k_lens` / padded sequences are used. TI2V uses `k_lens=seq_lens`; must compare outputs on CUDA (flash vs SDPA) before trusting MPS.

### `wan/modules/vae2_2.py` (`Wan2_2_VAE`)

| Mechanism | MPS note |
|-----------|----------|
| `import torch.cuda.amp as amp` | Use `torch.amp.autocast(device_type=…)` |
| `Wan2_2_VAE(..., device="cuda")` default | Pass `mps` explicitly |
| `Upsample(..., mode="nearest-exact")` | Verify on MPS (generally supported; test encode/decode) |
| 3D conv / resample stack | Operator-level testing required |

### `wan/modules/t5.py`

| Mechanism | MPS note |
|-----------|----------|
| Default arg `device=torch.cuda.current_device()` | **Breaks without CUDA** if caller omits device; TI2V passes `cpu` + `--t5_cpu` |
| With `--t5_cpu` | T5 runs on CPU; only context tensors move to `self.device` — **good for 16 GB** |

### `wan/utils/fm_solvers*.py`

| Mechanism | MPS note |
|-----------|----------|
| MPS float64 → float32 for timesteps/sigmas | Already upstream — keep |

---

## 3. Operators / features likely unsupported or risky on MPS

| Item | Severity | Notes |
|------|----------|-------|
| `flash_attn` | **Blocker** | CUDA-only; do not install on Mac |
| `flash_attention()` CUDA assert | **Blocker** | Must route to SDPA (or custom) on MPS |
| RoPE / sinusoid **float64** in `model.py` | **High** | Force float32/complex on MPS (match solver workaround) |
| Global **`bfloat16`** (`param_dtype`, T5 dtype) | **Medium** | MPS bf16 support is incomplete vs CUDA; may need fp16 for DiT/VAE on MPS |
| `torch.amp.autocast('cuda', …)` | **Medium** | Wrong device type on MPS — use `'mps'` |
| `torch.cuda.empty_cache()` | **Low** | Replace with GC + MPS sync; affects perf only |
| NCCL / `torch.distributed` | **N/A** | Default single-process TI2V |
| Very long seq (720p × 121 frames) SDPA memory | **High** | May OOM on 16 GB even if ops run |

---

## 4. Attention strategy without `flash_attn` (Mac)

Upstream already implements **`attention()` → `scaled_dot_product_attention`** when flash is absent, but **`WanModel` imports `flash_attention` directly**.

**Proposed compatibility layer (preserve CUDA path):**

```
wan_service/experimental/mps/attention_backend.py   # new, in Wan_service only first
  - resolve_attention(q,k,v,...) 
  - if device.type == "cuda" and flash available → call upstream flash_attention
  - elif device.type in ("cuda","mps","cpu") → SDPA with same layout as attention() fallback
  - never import flash_attn on Darwin
```

**Stage-1 patch surface (upstream, minimal forks):**  
Only replace `from .attention import flash_attention` usage in `model.py` with injectable backend, **or** monkeypatch `wan.modules.attention.flash_attention` at process start when `WAN_INFERENCE_DEVICE=mps`.

**Validation:** On a CUDA machine (later), compare flash vs SDPA outputs for one TI2V step; on Mac, all-close is not required before first smoke, but max diff thresholds should be documented.

---

## 5. Memory on M3 16 GB (expectations — not promises)

| Component | Rough order | With default service flags |
|-----------|-------------|----------------------------|
| DiT ~5B weights | ~10 GB @ bf16 | `offload_model` + `convert_model_dtype` moves DiT off GPU between phases |
| T5-UMT5-XXL | ~10+ GB @ bf16 | **`--t5_cpu`** keeps off GPU (critical) |
| VAE | ~1–2 GB | Stays on device for encode/decode windows |
| Latents + activations (720p, 121 frames, 50 steps) | **Large** | seq_len ~ tens of thousands; attention is O(seq²) per layer |

**Bottlenecks:** DiT peak when on GPU; VAE encode/decode at 720p; scheduler + latent buffers; unified memory pressure from macOS + apps.

**Safe optimizations to explore (after port):**

- Keep **`t5_cpu`** (already default in service).
- Keep **`offload_model True`**; extend with explicit **`torch.mps.synchronize()`** between phases.
- Reduce **`frame_num`** (e.g. 49 / 81) and/or **`max_area`** for smoke tests.
- Try **fp16** DiT/VAE on MPS if bf16 ops fail or are slow.
- Lower **`sample_steps`** only for smoke (quality tradeoff).

**Do not promise** full 720p@121 frames @50 steps on 16 GB until measured with weights.

---

## 6. Wan_service integration (health & jobs)

| Requirement | Plan |
|-------------|------|
| API contract unchanged | Keep existing routes; optional env `WAN_INFERENCE_DEVICE=cuda\|mps` for subprocess only |
| `/health` | Add `experimental_mps_enabled`, `mps_smoke_test_passed` (false until scripted test passes) |
| `inference_ready` | **Only true after:** weights present + MPS deps probe + **successful smoke** (short T2V or tiny I2V), not when API starts |
| Mac deps | New `requirements-mps-experimental.txt` (no flash_attn): torch+MPS, transformers, diffusers, etc. — **separate from** `requirements-linux-cuda.txt` |
| Subprocess | Still invoke upstream `generate.py` **or** thin wrapper that sets `PYTHONPATH` + patches before import |

---

## 7. Staged implementation plan

### Stage 0 — Documentation & probes (current)

- [x] This report.
- [ ] Add `scripts/mps_probe_ops.py` (no weights): MPS SDPA bf16/fp16, `Generator`, small conv, RoPE float32 rewrite micro-test.

**Smallest test:** `python scripts/mps_probe_ops.py` → all checks pass on M3.

### Stage 1 — Device abstraction (upstream-facing, CUDA preserved)

- Add `wan_service/experimental/mps/device.py`: `resolve_device()`, `autocast_device_type()`, `sync()`, `empty_cache()`.
- Patch **only** `textimage2video.py` via forked copy under `third_party/Wan2.2-mps/` **or** env-gated monkeypatch in wrapper script (preferred: wrapper so CUDA tree stays pristine).

**Smallest test:** Instantiate `WanTI2V` with dummy tiny random weights on CPU/MPS (no ckpt) — construction only.

### Stage 2 — Attention backend

- Wire SDPA path; block flash import on Darwin.
- CUDA regression: existing Linux CUDA path unchanged.

**Smallest test:** One forward pass of `WanModel` with random weights, batch 1, tiny `seq_len`, compare CUDA flash vs SDPA max diff.

### Stage 3 — RoPE / dtype fixes

- Float64 → float32 on MPS in RoPE helpers (mirror `fm_solvers` pattern).
- VAE autocast device-agnostic.

**Smallest test:** `vae.encode` / `vae.decode` on 64×64 dummy image on MPS.

### Stage 4 — End-to-end with weights (manual download)

- User runs `download_ti2v_5b_weights.sh`.
- Smoke: **low resolution**, **few frames** (e.g. 704×384, 17 frames, 4 steps), `--t5_cpu --offload_model True`.

**Smallest test:** `generate.py` produces non-empty MP4; service `inference_ready` gated on scripted smoke.

### Stage 5 — Service wiring

- Optional `WAN_RUNTIME_MODE=mac_dev` + `WAN_EXPERIMENTAL_MPS=1` launches wrapper.
- `/health` exposes smoke status; jobs 503 until Stage 4 passes.

**Smallest test:** Existing pytest suite + new `test_mps_readiness_not_ready_without_smoke`.

---

## 8. Tests to add in `Wan_service` (no weights, no GPU in CI)

| Test | Asserts |
|------|---------|
| `test_mac_dev_inference_never_ready_without_smoke` | `inference_ready` false; reason mentions API/test-only or smoke |
| `test_linux_cuda_still_requires_cuda` | Unchanged behavior |
| `test_mps_attention_never_imports_flash_attn` | Import `wan_service.experimental.mps` without flash_attn in sys.modules |
| `test_readiness_includes_wan_python_deps` | Already partially covered |

---

## 9. Explicit “cannot run on MPS today” list

Without port work, **these upstream pieces prevent TI2V-5B on MPS:**

1. Hard-coded **`cuda:{id}`** device in `WanTI2V`.
2. **`flash_attention()` CUDA assertion** (DiT never uses SDPA fallback).
3. **`torch.amp.autocast('cuda', …)`** throughout DiT/VAE path.
4. **`torch.cuda.*` cache/sync** calls.
5. **`generate.py` final `torch.cuda.synchronize()`**.
6. **RoPE float64** paths on a backend without float64.
7. **Operational:** 16 GB may be insufficient for default 720p/121f/50-step even after op fixes.

---

## 10. Next code change (when approved)

1. Add `docs/` (this file) + `scripts/mps_probe_ops.py` + empty `wan_service/experimental/mps/` package skeleton.  
2. Do **not** flip `UPSTREAM_TI2V5B_MPS_SUPPORTED` or `/health inference_ready` until Stage 4 smoke passes on hardware with weights.
