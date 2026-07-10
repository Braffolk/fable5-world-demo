# Optimisations for Real-Time Volumetric Cloudscapes (arXiv:1609.05344, 2016)

Source: https://arxiv.org/pdf/1609.05344
Authors: Alastair Toft, Huw Bowles, Daniel Zimmermann (Studio Gobo).

## The headline result (THE numbers)
A pixel-shader volume raymarch reduced to **~1/16 the number of steps** with visually similar
output, via **analytical transmittance integration + per-ray random jittered start offset + TAA**.

**Draw-call timings, 1920×1080, GTX 1080** (Figure 3):
| Resolution, method            | Draw time (ms) |
|-------------------------------|----------------|
| Full, 128 steps               | **297.7**      |
| Half, 128 steps               | 128.0          |
| Half, 8 steps                 | 2.3            |
| Half, 8 steps, **jitter**     | 7.5            |
| Half, 8 steps, jitter, **TAA**| 7.5            |
| Quarter, 8 steps, jitter, TAA | 2.4            |

So: 128→8 steps is the 16× cut; **297.7ms → 7.5ms** at half-res (the figure-1 comparison pair,
though note that's full-128 vs half-8-jitter-TAA). Jitter adds ~5ms (2.3→7.5) purely from **texture
cache misses** — that's the one cost the paper flags as its main downside.

## Method (the two tricks that let you drop steps)

### 3.1 Analytical integration (the enabler for fewer steps)
Standard raymarch updates color as `S = T·L` (S scattering, T transmittance, L lighting), which
**assumes transmittance is constant over the step** — an inaccuracy that makes brightness
**depend on step length**, so you normally can't take big steps without visible artifacts.

Fix: analytically integrate transmittance across the step, holding only **density** constant over
the step:
```
S = T0 · ∫₀ᴰ e^(−ρ·α·x) dx  =  T0 · (1 − e^(−ρ·α·D)) / (ρ·α)
```
(ρ density, α absorption, D step distance). This removes the step-length dependence → you can use
**large steps / few steps** without the brightness artifact. (Hillaire 2016 also notes this.)

### 3.2 Jitter + TAA (recover the detail few steps lost)
- Few steps loses cloud structure between samples. Counter it with a **random per-pixel offset to
  the raymarch start position**, so different depths get sampled each frame → recovers structure but
  produces a **very noisy** image.
- Apply **TAA** (Playdead/INSIDE-style temporal reprojection AA) to accumulate samples over frames
  and resolve the noise to acceptable levels, at very little perf cost.
- ⚠️ The jitter **must be per-frame-varying**; an identical fixed offset avoids the cache-miss
  penalty but then TAA can't improve it.

## Context (Previous Work — the two Horizon Zero Dawn optimisations they build on)
1. **Adaptive step size**: take longer steps and only sample the cheap base cloud shape when density
   is below a threshold (skip expensive detail in empty/thin regions).
2. **Half-resolution + reproject**: render clouds at 1/4 or 1/2 res, update only a fraction of pixels
   per frame (4×4 pattern), reproject the rest from previous frames.

## Conclusion / downside
1/16 steps, potentially fast enough for real-time game clouds. **Main open problem = the jittered
offset destroys texture-cache locality** (the 2.3→7.5ms jump), flagged as future work.

---

## MAP TO OUR kGrassRay march
Our grass is a **per-pixel analytic raycast, +14.5ms**. This paper is the playbook for cutting a
per-pixel volume march's step count — directly applicable if we STAY in the raymarch lane:
1. **Analytical integration → fewer steps for free.** If our march accumulates coverage/opacity with
   a step-length-dependent approximation, we're forced into many small steps. Deriving a closed-form
   per-step integral (as here) lets us take **big steps** without the banding/brightness artifact —
   the single biggest step-count lever. Check whether kGrassRay's opacity accumulation is
   step-length-dependent; if so, this is the fix.
2. **Jitter + TAA → 16× step reduction.** We already run TAA/TSR. Add a **per-pixel per-frame random
   start offset** to the grass march and let temporal accumulation resolve the noise — the same
   1/16 multiplier. This is the highest-leverage single change for the ray lane.
3. **Watch the cache-miss tax.** Their jitter cost +5ms from texture-cache thrash. Our per-blade
   hash/field fetches would take the same hit — mitigate by keeping the per-step fetch analytic /
   from a small resident LUT (ties to Babylon's zero-dependent-fetch analytic surface).
4. **Adaptive step / empty-space skip** (their prior-work #1): big steps where the grass field
   density is below threshold (bare ground, gaps), fine steps only near blades — cuts the steps our
   march wastes on empty ground.

**Verdict seed:** If we keep the per-pixel march, jitter+TAA+analytic-integration is the credible
path to ≤2ms (16× headroom vs the +14.5ms). But note this optimises clouds — a genuinely volumetric
medium. Grass is thin discrete blades (better matched by GoT per-blade generate or Babylon analytic
corridor); the raymarch lane may be spending its budget emulating a volume that isn't there.
