# Wan 2.2 vs CogVideoX-2B benchmark workflow

Same GPU, sequential runs (never load both at once). Keep
`WAN_PERSISTENT_INFERENCE=false`.

## Shared prompt

```text
The character slowly raises the kunai in his right hand and moves it into a defensive position. His wrist rotates naturally and his fingers tighten around the handle. His shoulders shift slightly. His hair and scarf move gently in the wind. A few leaves fall through the background. The camera remains mostly stationary with a subtle cinematic push-in. Smooth natural character motion, stable composition, detailed anime rendering.
```

## CogVideoX-2B (separate service)

```bash
python scripts/test_cogvideox.py --seed 12345
```

Defaults: 720×480, 49 frames, 8 FPS (~6.125s), 50 steps, guidance 6.

## Wan 2.2

Use the existing Wan job/API or generate path — do not modify Wan code.
Aim for similar duration and resolution when comparing.

## Record

| Metric | Wan 2.2 | CogVideoX-2B |
|--------|---------|--------------|
| Generation time | | |
| Peak VRAM | | |
| Motion quality | | |
| Prompt adherence | | |
| Temporal consistency | | |
| Visual quality | | |
| Character consistency | | |

Do not claim a winner without filled benchmark rows.
