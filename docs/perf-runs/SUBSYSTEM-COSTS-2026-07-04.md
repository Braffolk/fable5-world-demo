# Subsystem cost map — dpr2, world scene (2026-07-04)

Everything measured as **whole-frame gpuWall A/B** (subsystem off vs on, same
boot), the only trustworthy number — per-pass GPU spans overlap and oversum.
Machine-rested numbers; cross-session drift is ~2 ms, so compare columns, not
against numbers from other days.

## The headline frame (user's walk, eye pose, grass on)

| | gpuWall p50 (ms) |
|---|---|
| **Walking 8 m/s (the real experience)** | **36.7** (~30 fps) |
| Standing still, same pose | 27.7 |

## Where the walking 36.7 ms goes

| subsystem | ms | how measured | notes |
|---|---|---|---|
| **Shadows while MOVING** | **~8.0** | `nanshadow=0` walking 28.8 vs 36.7 | Standing still shadows cost only **0.8** — almost the whole bill is the clipmap **strip re-raster machinery** that fires when the camera moves. Resolution-independent. |
| **SW-raster pixel loop** | **≥7.8** | rdbg ladder (default − rdbg2) | The per-pixel scanline walk of visible sliver triangles. Oblique pose: **18.4**. Per-TRIANGLE setup is free (rdbg2−rdbg1 ≈ 0). |
| **Post stack** | **5.5** | `postmin=1` | Splits (overlapping, don't sum): AO **3.7**, clouds+bounce+contact **2.3**, TAA **1.2**, bloom **0.8**. Oblique total only 3.5. |
| **Grass (ray lane)** | **4.4** | in-session `setGrass(0/1)` | Eye pose. The ≤2 ms law (#55) is still unmet. |
| **Lit resolve + shadow sampling** | **2.1** | `pure=1` minus `postmin=1` | Shading of the vis-buffer + PCSS sample. Oblique 1.6. |
| **Cut churn (LOD/cull while moving)** | **~2.6** | `freeze=1` walking 34.1 vs 36.7 | |
| **Other moving inflation (non-shadow)** | **~1.1** | still 27.7 vs walking-noshadow 28.8 | |
| **Floor (raster fixed costs, vis clears, HW pass, sky, terrain…)** | **~6.6** | remainder to the pure floor | Pure geometry floor (`pure=1`): eye **15.7**, oblique **28.8**. |

## Still poses (gpuWall p50, grass-off \| grass-on)

| pose | off | on |
|---|---|---|
| eye (forest walk) | 23.3 | 27.7 |
| hill | 22.8 | 26.3 |
| oblique (worst) | 33.9 | 37.9 |
| aerial | 28.9 | 32.0 |

## The scaling law (why the upscaler is THE lever)

Frame ≈ **A + B·pixels**, R²≈1: eye default A≈4.2 ms, B≈3.2 ms/Mpx
(oblique B≈5.0). So:

| resolution | eye default (ms) |
|---|---|
| dpr2 (5.94 Mpx) | 23.3 |
| dpr1.42 (2.99 Mpx) | 14.2 |
| dpr1 (1.49 Mpx) | **9.0 — inside the 11-12.5 target** |

visTris also scales with resolution (τ is device-px): 9.5M @dpr2 → 4.7M @dpr1.

## ?rscale results (TSR upscaler, shipped 7a4c207, session 2 baseline)

| eye still, grass-on | p50 (ms) | live HUD fps |
|---|---|---|
| default (rscale=1) | 28.9 | 37 |
| rscale=0.7 | 23.1 | 44 |
| rscale=0.5 | **14.6** | **72** |

Oblique: 39.2 → 26.8 → 16.5. Walking scales by 0.71× / 0.53×.
Stills: `scratchpad/shots/rs-eye-{100,070,050}.png` — 0.7 near-native,
0.5 softer grass mid-field, no artifacts.

## Measured dead (don't relitigate)

- τ coarsening (`loderr`, default is 3): halving visTris → only −0.7 ms; mid
  values WORSE (aggregate ladder grows geometry → overdraw).
- `shmaxlv` strip-budget deferral: null/negative (batches the same work).
- `shtau`: saturates at −1.3 ms by 3.
- `trihzb` per-tri HZB reject: −13% fragments but only −0.5 ms (porous canopy).
- Resolve shading + sky + scene children: ~0.5 total.

Full detail + instruments: `2026-07-04-90fps-arc.md`.
