# Bog v7 — pool depth + micro/macro quality (consolidated cook cycle)

Orchestrator/Fable design brief. ONE implementer (opus, xhigh) owns ONE end-to-end cycle:
edit synth → run gates → freeze float → re-pack (`network_preview.py`) → re-verify → render
ground-level QA PNGs + real-WebGPU boot (grass=0) → return numbers + PNG paths. The
orchestrator judges the ground-level PNGs HONESTLY (hillshade fooled us 3×) and iterates.

This supersedes v6 with a **v7** preregistration amendment (bind v6 + v2 + v1 + terminated
marked-form; record exact recipe/hashes/thresholds/seed — the §9.10.1 amendment pattern).
Research-only, non-latest, non-Estonia-transfer. Honest labeling, NOT a quality ceiling.

EMBED THIS VERBATIM in the implementer's prompt (it does NOT inherit CLAUDE.md/memory):

> When a result disappoints, do NOT start varying your approach inside the problem. Go up one
> level first: put on trial what you filed under "given" — the params, the METRIC/gate, the
> upstream setup that generates the task. Is the context itself producing the failure? If the
> generating setup is changeable at acceptable cost, change THAT; the problem dissolves. Only
> if the context is genuinely fixed and sound, drop back down and solve it where it sits. The
> trigger is the negative result, NOT your confidence. You may NOT conclude "hard/undoable"
> until a premise-audit has fired and cleared the setup (params, metric, upstream) as sound.

## Why v6 reads wrong at ground level (root causes, grounded in the code)

1. **No genuine sub-0.5 m surface content.** `synth.py` draws ALL fields (fine, coarse,
   rough) on the 0.25 m `FIELD_PITCH_M` lattice, then `fields.fft_interp_cellcentered`
   band-limited-interpolates to 6 cm. That is exact for a band-limited field ⇒ between 0.25 m
   nodes the surface is smooth by construction. At a 6 cm eye view it MUST read under-detailed.
2. **One microform size.** `fine_field` is a single Matérn (`fine_range_m=0.7`) ⇒ one
   characteristic size, "lacks size variation."
3. **Roughness is tiny AND band-limited.** `roughness_sigma_m=0.006` (6 mm) and drawn at
   0.25 m ⇒ can't add detectable fine texture.
4. **Single-wavelength macro.** `coarse_field` is ONE anisotropic Gaussian (60/22 m) at ONE
   global `_flow_angle` scalar ⇒ parallel stripes, one wavelength, "very repetitive."
5. **Fragile gates block iteration.** `run.evaluate` measures `fractions` + `pool_coupling`
   on the 128 m core, SMALLER than the ~60 m coarse wavelength (~2 λ) ⇒ any new realization
   trips them on coarse PHASE alone (the native-redraw experiment: hollow 0.42→0.67, failed).
   `pool_coupling` also uses `median(height[authority])`, which mis-fires on hollow-skewed
   surfaces. Prereg v6 records both as known-followups. FIX THESE FIRST or you fight a
   mis-scoped metric (the "go up a level" rule applies to the gate itself).

## Task order (all in ONE synth change, then one pack/verify/QA)

### A. Gate-appropriateness FIRST (prerequisite; `run.py`, `gates.py`, `synth.py`)
- Measure `fractions` and `pool_coupling` over the **full work-grid authority** (`result.authority`
  / `result.relief_work` / `result.latent_std` over the whole 512 m grid — already computed),
  not just `result.relief_core`/`core_slice`. Also report the distribution over several tiled
  128 m windows across the work grid (mean ± spread) as evidence the number is phase-stable.
- `synth.class_fractions`: compute over the full-grid authority (percentiles of `latent_std`
  over `authority`, not `core_auth`).
- `gates.pool_coupling`: replace `med = median(height[authority])` with a FIXED hollow-band
  test — a margin cell is "low" if its detrended relief < −BREAK_M (or relief < 0); the
  hummock-over-pool test uses relief > +BREAK_M, NOT `med + 0.06`. Robust to a hollow skew.
- Keep the SAFETY gates measured on the core exactly as now (edge taper, off-authority/water
  residual — those are core-storage properties). Do NOT loosen safety.
- Re-run v6 params through the rescoped gates FIRST and record they still pass (sanity anchor)
  before changing the recipe.

### B. Micro: real detectable texture + size variation (`fields.py`, `synth.py`)
- **Multi-scale fine field.** Change `fine_field` (or add `multiscale_fine_field`) so its
  Fourier amplitude is `sqrt(Σ w_i · Matérn_density(range_i, nu))` over a small set of ranges
  giving a 0.3–3 m size distribution — start `[(0.4,0.5),(1.2,1.0),(2.5,0.35)]` (range_m,
  weight), nu≈1.5–2. One white stream, one field, deterministic; broadens the microform PSD
  (helps "no single-wavelength", keeps the size distribution honest). Re-tune so the
  detrended dominant spacing stays in DOMINANT_SPACING=[0.8,3] and first-zero in [0.3,0.9].
- **Native-6 cm roughness (THE detectability fix).** Draw the roughness channel at
  `OUTPUT_PITCH_M` (6 cm) — NOT at FIELD_PITCH_M then interpolate — so it carries genuine
  sub-0.5 m content. It is additive (`height_rel += roughness_sigma_m*rough`) and does NOT
  enter the latent, so it CANNOT disturb ordering/transiogram/fraction gates. Raise
  `roughness_sigma_m` 0.006 → ~0.018–0.025 m; keep `roughness_corner_m`≈0.5–0.9,
  `roughness_slope_exp`≈1.8 (PSD ~ k^−3.6). WATCH the anti_artifact PSD slope gate
  [−4.6,−3.3]: too much high-freq flattens the slope past −3.3. Tune σ to the largest value
  that keeps slope ≤ −3.3 with margin AND reads as detectable texture at ground level.
  (Native-6 cm roughness needs its own world-PRF stream at 6 cm; keep it world-keyed and
  determinate. Determinism of fine+coarse latent is UNCHANGED — still 0.25 m draw + interp.)

### C. Macro: break single-wavelength repetition (`fields.py`, `synth.py`)
- **Multi-octave coarse.** Superpose 2–3 anisotropic Gaussian octaves at different
  wavelengths, e.g. `(major,minor,weight)` = `[(90,34,1.0),(55,20,0.7),(32,12,0.45)]`, all
  oriented major-perpendicular-to-flow. Broadens the ridge-hollow spectrum, kills the single
  spike (helps periodicity gate), and yields a range of ridge spacings instead of one. Keep
  `coarse_half_autocorr_m ≥ COARSE_SCALE_MIN_M=8` and re-standardize to unit variance.
- If the macro still reads as parallel stripes at ground level, add a LIGHT spatially-varying
  orientation: a smooth low-frequency domain warp of the coarse field (displacement from a
  very-long-correlation field, amplitude a few m) so ridges braid/curve. Apply the warp at
  6 cm after interp; keep it deterministic. Only escalate to this if the octave sum alone
  isn't enough — do not over-engineer.

### D. Pool depth carve — CONFIRMED: carve the synth float (case B = surface-independent)
Root cause (measured): `cook_waterbed` (layers_cook.py:286) carves LOD0 ONLY ("Pilot LOD0
only"); the bog preview stands on uncarved LOD-2/-1 fine cores and `_corrected_base`
(network_preview.py:252-253) re-derives the LOD0 authority as `box_mean4` of the uncarved
fine ⇒ pool bed = pure DTM at every LOD. Data confirms flat: under-pool bed is only +0.06–0.08
m off the surrounding ring (< the ±0.10 m natural microrelief; ~5% of a real 1.5 m carve).

Water-surface source MEASURED = **INDEPENDENT of terrain height** (like #104 rivers):
- Rendered water Y = `field.fieldWaterY(q)` from a SEPARATE `water` HeightLevel plane
  (`WaterMaterial.ts:156`, `TerrainField.ts:93-94,448-452`), NOT the `height` layer.
- `waterY` for a Laugas = per-polygon flat median DTM level (`water.py:206-213 _polygon_level`)
  ≈ 57.4–57.7 m ≈ the uncarved bed ⇒ column thickness ≈ 0 ⇒ painted-on. Depth is Beer-Lambert
  on (surface − rendered bed); `NaniteGrass.ts:308-310` already does `waterY.sub(h).max(0)`.
- The bog preview only corrects HEIGHT, so it INHERITS the base's `water` layer unchanged ⇒
  carving the bed DOWN leaves `waterY` put and opens a real water column. CONFIRMED works.

Implement in the synth float (NOT a packer/LOD0 post-pass — that would be overwritten by the
box-mean re-derivation):
- Compute the Laugas bed wedge with `process/water.bed_depth_field` over the SAME haloed
  work window the synth uses (`work_bbox`, 192 m halo, texel = OUTPUT_PITCH_M), then crop to
  the core — so a Laugas clipped at the 128 m core edge doesn't get an over-deep shore-shelf
  (measured caveat). Verified callable at 6 cm: t=0.0625 → (8192²-scale) depth.max 1.466 m,
  Laugas tyyp40 → 1.5 m plateau, 10 m shelf `d=dmax·(2s−s²)`, `s=clip(d_shore/10,0,1)`; `stack`
  arg is unused inside, no DEM dependency. Per-type depth comes from E_202 `tyyp`.
- Replace `synth.py:211-212` `relief_work[open_water]=0.0` with
  `relief_work[open_water] = -depth[open_water]` (subtract the wedge from the baseline, exactly
  mirroring `layers_cook.py:310` `h - depth`). Since terrain h ≈ waterY at the pool, `h−depth`
  ≈ `waterY−depth`. `masks.open_water` (synth.py:164, native 6 cm over the work grid) is the
  exact cell set. The wedge tapers to 0 at the shoreline, so no cliff on the water side; keep
  the land-side `water_taper` so land relief still fades to 0 at the boundary.
- SAFETY gate "open-water relief-free (0 residual)" MUST be REINTERPRETED, not violated: it
  forbids FAKE microform BUMPS in water and keeps the water SURFACE reference exact. A
  monotone physics bed-carve DOWN is the legitimate #104 depth model the user asked for. New
  gate: NO positive/microform relief in water; the water cells carry ONLY a monotone,
  non-positive Laugas wedge (0 ≥ relief ≥ −dmax, 0 at shore, deepening inward); waterY
  untouched. Update `run.evaluate` safety accordingly (water_residual → assert relief ≤ 0 in
  water and == the expected wedge, not == 0).
- QA: in the ground-level boot, confirm the water surface visibly floats above the carved bed
  (a real depth column + Beer-Lambert darkening), and that hummocks/shore sit above waterY.

## Gates to pass (v7)
All v6 gates (rescoped per A), plus the ground-level verdict. Concretely: safety (core edge +
land off-authority residual 0; water = monotone bed wedge per D, no positive relief);
amplitude p50∈[0.15,0.25], p95∈[0.28,0.40]; transiogram hollow↔hummock≈0 monotone;
fractions within 0.08 of Ilyasov target MEASURED OVER FULL AUTHORITY; anti_artifact slope
∈[−4.6,−3.3], dominant spacing∈[0.8,3], periodicity≤8; two_scale first-zero∈[0.3,0.9] +
coarse≥8 m; pool_coupling (rescoped) margins low + no hummock over pools; pool bed shows
depth. ⭐ GROUND-LEVEL QA IS THE REAL GATE: ≥2 low-oblique ~1–2 m eye-height look-across
renders + a before(v6)/after(v7) pair; judge whether it reads as a bumpy organic bog with
detectable fine texture, varied microform sizes, non-repetitive rolling macro, and pools with
visible depth. Top-down hillshade is NOT sufficient.

## Reuse / pipeline
Reuse the pool-bearing site + base (mire `etak-component-0004071135`, base manifest
`7aa4d523…`), the existing packer `network_preview.py` + verifier `network_preview_verify.py`
(swap only the synth float, key `core_relief_00625m`, shape 2048² at 6 cm), `hydrology.py`,
`masks.py`, `prf.py`. Python via `uv run`. After freeze+pack+verify, boot with
`npx tsx tools/boot-smoke.ts` at a ground-level exact-position URL, grass=0. Report manifest
sha + exact-position URL + QA PNG paths + the full gate table.
