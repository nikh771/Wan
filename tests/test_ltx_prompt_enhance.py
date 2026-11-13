"""Unit tests for LTX Ollama/fallback prompt enhancement."""

from __future__ import annotations

from ltx_service.prompt_enhance import (
    DEFAULT_NEGATIVE,
    cinematic_fallback,
    enhance_ltx_prompt,
)


def test_cinematic_fallback_i2v_adds_motion() -> None:
    out = cinematic_fallback("a woman portrait", mode="image_to_video", duration_seconds=5)
    assert "seconds" in out.lower()
    assert "90" in out or "turn" in out.lower()
    assert "unmistakable" in out.lower() or "pushes in" in out.lower()
    assert "subtle" not in out.lower()
    assert len(out) > len("a woman portrait")


def test_enhance_fallback_when_ollama_down(monkeypatch) -> None:
    monkeypatch.setattr(
        "ltx_service.prompt_enhance.ollama_available",
        lambda *_a, **_k: False,
    )
    result = enhance_ltx_prompt(
        "slow head turn",
        mode="image_to_video",
        duration_seconds=5,
        enabled=True,
    )
    assert result.source == "fallback"
    assert result.original_prompt == "slow head turn"
    assert "turn" in result.prompt.lower() or "motion" in result.prompt.lower()
    assert "never nearly still" in result.prompt.lower()
    assert result.negative_prompt == DEFAULT_NEGATIVE


def test_enhance_passthrough_when_disabled() -> None:
    result = enhance_ltx_prompt("keep me", mode="text_to_video", enabled=False)
    assert result.source == "passthrough"
    assert "keep me" in result.prompt
    assert "unmistakable" in result.prompt.lower() or "never nearly still" in result.prompt.lower()


def test_enhance_uses_ollama_when_available(monkeypatch) -> None:
    monkeypatch.setattr(
        "ltx_service.prompt_enhance.ollama_available",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "ltx_service.prompt_enhance._ollama_generate",
        lambda **_k: "The subject turns 90 degrees toward camera with continuous motion.",
    )
    result = enhance_ltx_prompt("turn", mode="image_to_video", enabled=True)
    assert result.source == "ollama"
    assert "90" in result.prompt or "turn" in result.prompt.lower()
    assert "never nearly still" in result.prompt.lower()
