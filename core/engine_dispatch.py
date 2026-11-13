"""Route generation jobs to Wan, LTX, or CogVideoX."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from core.config import get_settings
from core.types import EngineType, GenerationJobParams
from cogvideox_service.engine_service import run_cogvideox_engine
from ltx_service.engine_service import run_ltx_engine
from wan_service.core.runtime import evaluate_inference_readiness
from wan_service.engine_service import run_wan_engine


@dataclass(frozen=True)
class EngineReadiness:
    ready: bool
    public_reason: str


def normalize_engine(value: str) -> EngineType:
    engine = (value or "wan").strip().lower()
    if engine in ("wan", "ltx", "cogvideox"):
        return engine  # type: ignore[return-value]
    raise ValueError(
        f"Unsupported engine_type {value!r}. Use wan, ltx, or cogvideox."
    )


def _ltx_python_executable() -> Path | None:
    from ltx_service.settings import get_ltx_settings

    cfg = get_ltx_settings()
    raw = cfg.resolved_ltx_python()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cfg.project_root / candidate
    if candidate.is_file():
        return candidate
    found = shutil.which(raw)
    return Path(found) if found else None


def _cogvideox_python_executable() -> Path | None:
    from cogvideox_service.settings import get_cogvideox_settings

    cfg = get_cogvideox_settings()
    raw = cfg.resolved_cogvideox_python()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cfg.project_root / candidate
    if candidate.is_file():
        return candidate
    found = shutil.which(raw)
    return Path(found) if found else None


def evaluate_engine_readiness(engine: EngineType) -> EngineReadiness:
    if engine == "wan":
        state = evaluate_inference_readiness()
        return EngineReadiness(state.ready, state.public_reason)

    if engine == "cogvideox":
        py = _cogvideox_python_executable()
        if py is None:
            from cogvideox_service.settings import get_cogvideox_settings

            cfg = get_cogvideox_settings()
            return EngineReadiness(
                False,
                f"COGVIDEOX_PYTHON not found: {cfg.resolved_cogvideox_python()!r}",
            )
        from cogvideox_service.health import cogvideox_health_payload

        health = cogvideox_health_payload()
        if health.get("status") == "ready":
            return EngineReadiness(True, "CogVideoX-2B ready.")
        return EngineReadiness(
            False,
            health.get("detail") or "CogVideoX engine is not ready.",
        )

    ltx_py = _ltx_python_executable()
    if ltx_py is None:
        from ltx_service.settings import get_ltx_settings

        cfg = get_ltx_settings()
        return EngineReadiness(
            False,
            f"LTX_PYTHON not found: {cfg.resolved_ltx_python()!r}",
        )
    from ltx_service.health import ltx_health_payload

    health = ltx_health_payload()
    if health.get("status") == "ready":
        return EngineReadiness(True, "LTX 2B distilled ready.")
    return EngineReadiness(
        False,
        health.get("detail") or "LTX engine is not ready.",
    )


def assert_engine_available(engine: EngineType) -> None:
    readiness = evaluate_engine_readiness(engine)
    if not readiness.ready:
        raise RuntimeError(readiness.public_reason)


def run_engine(params: GenerationJobParams, engine: EngineType) -> dict:
    _ = get_settings()  # ensure .env loaded before subprocess engines
    if engine == "wan":
        meta = run_wan_engine(params)
        if isinstance(meta, dict):
            meta.setdefault("engine_used", "wan")
            return meta
        return {"engine": "wan", "engine_used": "wan"}
    if engine == "cogvideox":
        meta = run_cogvideox_engine(params)
        meta["engine_used"] = "cogvideox"
        return meta
    meta = run_ltx_engine(params)
    meta["engine_used"] = "ltx"
    return meta
