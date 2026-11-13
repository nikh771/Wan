"""Engine routing and validation tests."""

from __future__ import annotations

import pytest

from core.engine_dispatch import normalize_engine


def test_normalize_engine_wan_default() -> None:
    assert normalize_engine("wan") == "wan"
    assert normalize_engine("  WAN ") == "wan"


def test_normalize_engine_ltx_storyboard() -> None:
    assert normalize_engine("ltx") == "ltx"
    assert normalize_engine("storyboard") == "storyboard"


def test_normalize_engine_rejects_hybrid_for_now() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_engine("hybrid")
