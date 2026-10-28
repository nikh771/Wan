#!/usr/bin/env python3
"""Stage-0 MPS operator probes (no Wan weights). Run on Apple Silicon only."""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    if not torch.backends.mps.is_available():
        print("MPS not available — run on Apple Silicon with a PyTorch MPS build.")
        return 1

    device = torch.device("mps")
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"OK  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")

    check(
        "Generator",
        lambda: torch.randn(4, device=device, generator=torch.Generator(device=device)),
    )

    def sdpa_fp16() -> None:
        q = k = v = torch.randn(1, 8, 128, 64, device=device, dtype=torch.float16)
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    check("SDPA fp16", sdpa_fp16)

    def rope_fp32() -> None:
        x = torch.randn(100, 24, 128, device=device, dtype=torch.float32)
        freqs = torch.randn(100, 64, device=device, dtype=torch.float32)
        _ = torch.view_as_complex(
            torch.stack([x[..., :64], x[..., 64:128]], dim=-1).reshape(100, 24, 64, 2)
        )

    check("view_as_complex fp32", rope_fp32)

    def conv3d() -> None:
        m = torch.nn.Conv3d(3, 8, 3, padding=1).to(device=device, dtype=torch.float16)
        t = torch.randn(1, 3, 4, 32, 32, device=device, dtype=torch.float16)
        m(t)

    check("Conv3d fp16", conv3d)

    if failures:
        print("\nSome probes failed (see above). Review docs/ti2v_5b_mps_port_plan.md.")
        return 2
    print("\nAll Stage-0 MPS probes passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
