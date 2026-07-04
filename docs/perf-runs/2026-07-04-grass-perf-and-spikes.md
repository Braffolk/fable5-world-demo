# 2026-07-04 — grass ray perf pass + fly-through microspikes

## Context
User isolation (live retina, cross-boot): grass-on = "STABLE 25 fps" (quantum
lock); no-water/no-grass = spikes mainly flying fast through dense forest
(geo↔voxel switching suspicion); +water = walking spikes. This file = the
grass perf pass (SOLVED) + the spike attribution.

## Grass regression: CONFIRMED then KILLED (dpr2 eye, gpuWall A-B-A, grass8-frame.mjs)
Baseline: ON 48.7 / OFF 23.3 → grass +25.4 ms, c.grassRay 23.1 of it.
48.7 ms → 120Hz 50 ms quantum = the user's "stable 25 fps". Hypothesis held.

Feature A/B (kRay ms @dpr2 eye): raysetup floor 0.13; skeleton (shear+sway+
tilt+bomb+L2 off) 11.2; shear −4.5; tilt/bomb/sway −7.1; L2 −1.8; bakn5 −3.2;
rayend80 −0.5 (march DISTANCE ~free — election depth caps rays; the burn was
per-step ALU × rays).

### Shipped lever A — per-texel FIELD bake (733f91f)
kRay re-derived every smooth field per STEP (3× value noise = 12 hashes +
swirl/wind/arc trig ≈ 12 ms). Now baked per texel per frame in kGuideBake
(which rebakes for wind anyway; guide pass still 0.26 ms) into two rgba16f
StorageTextures, HW-bilinear at the ray's world pos:
  T1 = (Slx, Slz, Swx, Swz) lean + wind quad-term (time folded at bake)
  T2 = (ca, sa, a1x, a1z) swirl cos/sin (renormalized per step — bombI must
       be the exact bombF inverse) + static arc; L2 arc = −a1.
⚠️ grid law: piecewise-BILINEAR sampled at pos is continuous across space —
the 0.84 m quilt class was piecewise-CONSTANT fields. Pond/eye dpr2 shots
match the accepted look. kRay 23.1 → 12.9.

### Shipped lever B — quad march (d7c9e19)
March cost ∝ rays × steps; retina's full-res dispatch was the multiplier.
?grassquad (auto 2 at dpr ≥ 1.75): ONE quad-center ray per 2×2 pixels, hit
fanned to the quad. Election atomics stay per-PIXEL → true-geometry edges
keep pixel crispness; scene early-out takes the quad's FARTHEST bound;
only grass-over-background silhouettes quantize (≈ dpr1 grass, TRAA-softened).
kRay 12.9 → 3.7. Shots: eye/pond match accepted look; pond 34 → 66 fps.

### State after both
dpr2 eye whole frame: ON 27.2 / OFF 23.2 → grass +4.0 ms (was +25.4).
dpr1.5 eye: ON 24.0 / OFF 15.4 → +8.6 (Q=1 default; was +14.5); with
?grassquad=2: 18.7 → +3.3. Quad auto-threshold 1.75 keeps dpr1.5 full-res —
dropping to ~1.4 is a user quality call (booked #55). ≤2 ms law still unmet;
the LIVE regression (retina 25 fps lock) is what this pass killed.

## Microspikes (fly-through) — instrument
spike9-fly.mjs: fast fly (35 m/s, terrain+4 m) through the eye-pose forest,
EVERY frame isolated via __laas.measureFrames({frames:1}) → per-frame gpuWall
+ per-pass spans + counters + cpuSubmit; spike frames (top decile) print pass
deltas vs run median. Legs: default vs ?freeze=1 (cut churn vs streaming
split), water legs later.

### Leg results
- default: p50 40.8 / p90 48.4 / max 49.8; spikes cluster at FIXED path
  locations d76 / d89-90 / d102-104 m.
- ?freeze=1: p50 40.6 / p90 47.9 — IDENTICAL, same locations (d75/80/90/92/
  102-104). ⇒ NOT LOD-cut churn (the user's geo↔voxel switching hypothesis
  does not explain the GPU-side spikes; freeze pins the cut).
- Spike Δpasses: "compute" aggregate +7-8 ms while the only named risers are
  FIXED-SIZE clears (nanClipClear1 +3-4.5, nanVisClear +1-3). A clear can't
  do more work ⇒ its measured span absorbs a QUEUE STALL — smells like
  streaming uploads (copies aren't pass-timed) or shadow strip re-raster.
- Two-lap: LAP1 ≈ LAP0 (p50 40.1 vs 40.5, same spots) ⇒ NOT streaming.
- ?ablate=shadows: spike signature GONE (max ~33 vs ~50); residual smaller
  spikes at dense walls = main-view coverage (visTris +3.5M, render +7-8).
- grass=0: shadow spikes unchanged (shC1 +32-35k) ⇒ grass-independent ✓ (user).
- Counters on spike frames: nanite.shTotal +46-51k (shC0 +7k, shC1 +31-35k,
  shC2 +13-20k), shRaster gains a COARSE level bit. The batched filter+strip-
  raster chain lands in the "compute" aggregate (+7-8) and the fixed-size
  nanClipClear* spans absorb the queue serialization.

### VERDICT: fly-through spikes = shadow clipmap strip re-raster through dense
caster walls. NOT geo↔voxel switching (freeze identical), NOT streaming (lap 2
identical). Root cause one level up: the shared shadow cut ran at FLAT τ=1
with NONE of the camera path's distance-LOD coarsening (the beautification
lodWarp never reached casters) — a strip crossing a tree wall carries 30-50k
leaf-level caster clusters.

### P11 fix (this session): caster-LOD warp for the shared cut
NaniteClipCull/NaniteShadowClip: wire tau/lodNear/lodPow/simBandD into the
shared cut's buildNaniteCull. A DISTANCE warp is shared-cut-legal (identical
across levels; only per-level τ would break the shared traverse). Knobs:
?shtau (1) ?shtaunear (40 m plateau — contact shadows exact) ?shtauband (120 —
τ doubles 160 m out) ?shtaupow (1); escape ?shtauband=0. Composes with P10
ring-snap (warpDist is the snapped distance ⇒ no LOD-age). Gate: fly A/B +
shadow-look stills (deliberate shadow change — user eyeball pending).
