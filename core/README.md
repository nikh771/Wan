# Core (shared)

Cross-engine pieces used by `wan_service` (API) and `ltx_service`:

| Module | Purpose |
|--------|---------|
| `paths.py` | `REPO_ROOT`, service roots |
| `app_settings.py` | Worker API, jobs dir, FFmpeg, queue limits |
| `config.py` | `Settings` = app + Wan (`get_settings()`) |
| `types.py` | `GenerationJobParams`, `EngineType` |
| `logging.py` | `configure_logging`, `get_logger` |
| `engine_dispatch.py` | `run_engine`, readiness, `normalize_engine` |

Engine-specific env:

- **Wan:** `wan_service/settings.py`
- **LTX:** `ltx_service/settings.py`
