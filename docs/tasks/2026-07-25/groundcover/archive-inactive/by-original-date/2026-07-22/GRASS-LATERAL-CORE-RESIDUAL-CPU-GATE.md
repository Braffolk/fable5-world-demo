# Lateral-core + joint-residual CPU gate

Date: 2026-07-22  
Status: **PARKED — strict core is constructive, but the fitted two-layer transfer is RED; one remaining joint-fit question is isolated**  
Runtime/shader changes: **none**

## Binding correction — 2026-07-23

Sections 1–7 below are retained as historical evidence only. Their headline
retention figures (including `65.18%`) are **not valid evidence for the selected
representation**. That analyzer allowed per-primitive movable axial intervals
and a horizontal/centreline factorization. A class-E query has one true global
height coordinate and a family-global interval; silently rotating or translating
that interval per primitive changes the live chart and invalidates the claimed
fixed-cost compilation.

The corrected mathematical representation tested here is:

```text
finite compact periodic phase arcs Gamma_f
extruded through one true global height interval I_f
under A_beta(q,h) = (q_x + beta_x h, h, q_z + beta_z h)
+ one jointly compiled, plane-free analytic residual
```

`Gamma_f` need not be the boundary of a filled 2D mask. An open arc times a
closed height interval is an ordinary two-sided ruled ribbon with boundary
rulings at its endpoints, not a filled solid with axial caps. For a live ray
clipped to `[t_a,t_b]`, projected phase speed `s_p > 0`, initial phase `q_a`,
and projected direction `omega`, exact first contact is

```text
E = {ell in [0, s_p (t_b-t_a)] : q_a + ell omega in Gamma_f}
t_hit = t_a + min(E) / s_p
```

when `E` is non-empty. This requires no inside/outside state, successor search,
loop, candidate list, owner loop, or extra sampled dimension. Endpoints,
crossings, and coincident contacts use frozen categorical tie rules. The
projected-axis pole is the exact membership case `q_a in Gamma_f ? t_a : MISS`;
camera-on-plant is outside the exterior-view contract.

The corrected compiler gate has produced a non-redundant, continuously
height-certified arc network for the accepted Calamagrostis source under the
same fixed field/read budget. After attempt 1 exposed an optimistic-selection
mismatch, the strict emitted-path F4 selection was recomputed and frozen at:

```text
data/work/groundcover-relay-arc-network/
  37b0cf1d33f632b5/6219c2153f9523c7/9917249acc582ac1/
```

Its `45 degree` compiler variant has:

- `315` foliage arc edges over two disjoint global-height fields, exact
  sheared sheet area `0.026105 m^2`;
- `64` purple/brown reproductive arc edges over two disjoint global-height
  fields, exact sheared sheet area `0.012754 m^2`;
- fixed-record purple colour max-channel p95 `0.125` in the upper field and
  `0.027` in the lower field;
- cream plume content assigned to the bounded residual because no crisp
  continuously height-certified arc survived for that class.

Those figures prove constructive feasibility only. They do **not** prove image
quality and are not an implementation green light. The active terminal gate is
one common-perspective exterior ray-bundle comparison of the real accepted mesh
against the compiled arc field plus the actual `K_live <= 4` joint residual.
Reference and candidate use the same anisotropic pixel footprint at
`1,2,4,8,16,32 m`, including cardinal, low-oblique, and millimetre-translation
sequences. Fine world-space discrepancies may become tolerable only when their
projected footprint becomes subpixel; coherent fixed-angular streaks, wedges,
silhouette displacement, colour loss, or shimmer remain failures at every
distance.

Runtime implementation remains unauthorized until that exterior ray gate is
green. If the complete gate is red, there is one premise-audit/fix rerun; a
second red becomes a written mathematical/representation blocker rather than
runtime tuning.

### Exterior attempt 1 — RED, setup mismatch diagnosed

The first common-perspective run is preserved at:

```text
data/work/groundcover-relay-exterior-ray-gate/
  37b0cf1d33f632b5/5fe323311c5c8a0d/c0041091ce148828/
```

It is decisively red, not threshold-close. Representative results were:

- `18 deg`, `1 m`: silhouette IoU `0.935`, RGB p95 `0.414`;
- `5 deg`, `1 m`: silhouette IoU `1.000`, RGB p95 `0.574`;
- `18 deg`, `32 m`: silhouette IoU `0.662`, RGB p95 `0.457`;
- purple/cream RGB errors remain roughly `0.4–0.6`, and categorical core
  first-visible matches are near zero.

The candidate becomes a nearly uniform dark soft field while the identically
filtered source retains green and violet plant structure. The failure does not
yet refute the representation because the implementation did not match the
selected mathematics in three binding ways:

1. `F4` came from an optimistic selector rather than the strict continuous-
   height emitted-path certificate;
2. the residual was fitted to full top-down first-hit transfer using only
   horizontal modes, rather than fitting the complete positive
   core-plus-residual hybrid directly against full source transfer over
   exterior rays and partial cutoffs;
3. the `24x18` contiguous pixel window covered only one tiny periodic phase at
   the near distance and included no millimetre translation sequences, so its
   global metrics could not certify the community.

The one permitted diagnosis/fix rerun therefore uses strict emitted-path F4,
fits the complete positive core-plus-`K_total<=4` residual hybrid directly
against full source exterior/partial-cutoff transfer, stratifies the full
periodic phase domain, adds millimetre translations, and evaluates the complete
`1,2,4,8,16,32 m` law. The offline tessellation used by this CPU comparison is
only a truth-query implementation of the mathematical arc sheets; it is not a
runtime mesh proposal.

### Exterior attempt 2 — fitted candidate RED; representation verdict parked at one exact remaining question

The corrected run is preserved at:

```text
data/work/groundcover-relay-exterior-ray-gate/
  37b0cf1d33f632b5/9d993b97802dc47d/a8ca42a9b2901f1b/
```

It uses the strict `379`-edge/F4 network, both declared Tier-1 affine
populations on both sides of every comparison, `K=2` residual modes per layer
(`K_total=4` after expansion), the exact shared perspective ray bundle at
`1,2,4,8,16,32 m`, full periodic-phase stratification, `1/2.5/4.5 mm` camera
translations, exact horizontal rays with a finite scene cutoff, and `4x4` to
`8x8` quadrature convergence. It does not touch a shader or runtime path.

The fitted candidate fails by margins that cannot be called a slightly strict
threshold:

- near-vertical phase-stratified silhouette IoU is only `0.278–0.306`; the
  candidate is essentially opaque everywhere while the identically filtered
  two-layer source covers about `0.30–0.34` of the phase samples;
- RGB max-channel p95 is `0.421–0.553` across the tested production views,
  versus the `0.15` target;
- the largest top-down connected wrong region is `66.7–72.2%` of the phase
  grid;
- the directly fitted partial-prefix alpha error has p50 `0.383`, p95
  effectively `1.0`; RGB p95 is `0.429`;
- one exact-horizontal source/core air origin at normalized slab height
  `u=7/16` has no source or core event for `4 m` in any of the four cardinal
  directions, in either transformed layer. Every corresponding fitted
  candidate ray has alpha `1` and an olive premultiplied colour near
  `(0.25,0.24,0.16)` instead of transparent black;
- `8x8` confirms the serious low-angle core mismatch rather than removing a
  `1/16` quadrature atom. At `1 degree`, the paired core-early-by-more-than-
  `5 cm` fraction remains about `0.09–0.14`; the all-event positive core/source
  CDF excess reaches `1.0` near and `0.406` far.

Distance was treated correctly: reference and candidate used the identical
physical pixel bundle at each range. Microscopic source colour varies between
`4x4` and `8x8`, as expected, but the candidate's opacity error is coherent
and effectively unchanged. The result therefore does not reject distant
detail merely for being distant, nor excuse a constant-screen/full-coverage
artifact.

#### Exact lower bound for the fitted candidate

For each layer the certified density is

```text
kappa_l(x) = w(y) [a0 + 2 Re(c1 exp(i k1.x) + c2 exp(i k2.x))]
w(u) = 4u(1-u),                 kappa bracket >= 0.46075846498.
```

The selected two modes both have local horizontal index `(mx,mz)=(2,0)`;
their different `my` values only change the combined phase/amplitude with
height. At the certified air origin `u=7/16`, `w=63/64`. Nonnegativity alone,
without ray quadrature or a rendered image, gives for a horizontal segment of
length `L=4 m` through both layers

```text
tau >= L sum_l w kappa_min
    = 4 * 2 * (63/64) * 0.46075846498
    = 3.62847291172,

alpha = 1-exp(-tau) >= 0.97344329210.
```

The exact repeated source and the strict core are both misses there, so their
target alpha is zero. This proves that **the fitted coefficient set** is RED
independently of filtering resolution, distance, colour fitting, or any
renderer detail.

There is also a structural zero-set warning for this selected frequency
family. At fixed height each layer is a nonnegative one-dimensional stripe
polynomial in its rotated local `x`. If it integrates to zero on any open
horizontal segment with nonzero local-`x` velocity, analyticity and
nonnegativity force that fixed-height stripe polynomial to be identically
zero. The four cardinal source-air segments include such directions for both
affine layers. Exact transparency is therefore incompatible with retaining
any residual transfer elsewhere on that same height plane for the selected
frequency family.

#### Why this is not yet a coefficient-independent rejection of every K2+K2 residual

The premise audit found one remaining mismatch before the broader
representation can be called refuted. Attempt 2 optimized one layer's final
hybrid transfer and then applied that fitted density to both affine layers.
The specification calls for coefficients selected against the **joint final
two-layer union**.

More importantly, a legal two-mode vertical notch is an explicit
counterexample to any claimed general horizontal/top-down contradiction:

```text
P(y) = a [1-cos(lambda(y-y0))]^2
     = a [3/2 - 2 cos(u) + 1/2 cos(2u)].
```

It is nonnegative, uses only two non-constant vertical modes, vanishes on the
whole plane `y=y0`, remains positive elsewhere, and its finite-bundle error
near the notch is quartic in `delta y`. Thus one measured empty horizontal
height does not force a general K2 layer to abandon top-down transfer. A
coefficient-independent lower bound would need positive-measure empty
constraints over multiple heights/phases plus the frozen vertical bandlimit,
for example a valid Remez/Turan bound. The present artifact does not supply
that premise.

Running another heuristic fit would violate the two-real-failure limit and
would not answer the remaining question. The track is therefore parked, not
silently declared impossible and not sent to the shader.

**Objective resume condition:** a globally or near-globally optimal direct
minimax fit of the final two-layer `K2+K2` hybrid—not per-layer marginals—over
the finite allowed reciprocal-frequency catalogue, with positivity and
`0<=j<=kappa` enforced, exact-horizontal air queries included in the fit, and
all source/core event-adjacent partial cutoffs scored. It must pass the common-
footprint silhouette/RGB/connected-region/depth/translation gates. A fresh
heuristic correlation ranking is not a resume condition. Alternatively,
positive-measure multi-height/phase empty constraints plus a frozen vertical
bandlimit may support a coefficient-independent theorem that closes the
remaining zero-set/coverage tradeoff without another run.

Artifact bindings:

- source mesh SHA-256:
  `37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0`;
- strict network SHA-256:
  `1b376238658e3e5290d19b9a590c300108191d162bdf211fb0d18005e4e6860a`;
- metrics SHA-256:
  `89383363b6d4827c009c1e6e742ed8d30a6f10a81e276a2f97e3ebe9ab094f71`;
- tool SHA-256:
  `a8ca42a9b2901f1b8a440082a522d99e03fed8341fc5ae199e1bb85d1aa60402`.

Historical superseded analysis follows.

## 1. Question and result

This gate measures whether the accepted Calamagrostis and Sphagnum sources
have a plausible fixed-cost compilation into:

```text
F cap-free source-contained lateral fields
+ one joint quadratic-window RGB/extinction residual
+ K_live <= 4 total Fourier modes after both anti-tiling layers
```

The answer is not yet a green light for runtime work.

- The new representation is no longer killed by the old endpoint/cap-plane
  theorem. It can retain a substantial exact lateral subset with `F<=4`, and
  its omitted detail decreases under a projected-feature pre-gate rather
  than being charged forever at source resolution.
- The tall plant still leaves major multi-pixel near-field content outside the
  core. The most useful two-layer point (`F=4`, `5 mm`) retains `65.18%` of
  eligible lateral area, including `62.26%` foliage and `69.23%` reproductive
  area, but only `3.09%` of the small-area crisp structural class selected by
  this greedy catalogue.
- The one-layer `F=8`, `10 mm` point raises retention to `69.80%` eligible
  lateral area and `50.20%` crisp structure, but still delegates visible
  near-field content to the residual and sacrifices the second geometric
  anti-tiling population.
- A deliberately optimistic one-dimensional marginal fit says four total
  modes can match cumulative omitted height mass/colour to roughly `3–7%` for
  the useful candidates. This is encouraging, but it is not the required
  three-dimensional arbitrary-ray, arbitrary-cutoff alpha/RGB gate.

Therefore the current result is **AMBER**, not RED and not GREEN. The next
required result is a filtered exterior-ray transfer fit of the one joint 3D
field. Implementing the runtime before that measurement would be guessing.

## 2. Source and compiler binding

The analyzer consumes the deterministic primitive recipes and verifies the
accepted sources before measuring them.

- Production Calamagrostis: `2,049,985` vertices, `2,171,134` triangles,
  source mesh SHA-256
  `37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0`.
- Sphagnum: `25,397` vertices, `41,184` triangles, binary mesh SHA-256
  `ec8612e197dcc9a801c6146059cf8f85b7549ef33ac189252e91c8efbf01caa7`.
- No source or packed asset is modified.

The Calamagrostis compiler exposes `396,541` cap-free body atoms. They cover
`74.65%` of total source surface area before field selection. Hubs, plume
hairs, support sheets, root/terminal fans, and true axial caps are never
credited to the crisp core.

Each atom is assigned only when a common field axis and interval fit its
source sweep within the declared tolerance. The reported crisp proxy is
optimistic: it credits the exact source triangles after compatibility, so a
failure cannot be blamed on mask-raster or atlas sampling quality. Field
selection balances foliage, crisp structure, reproductive landmarks, and
moss rather than maximizing raw triangle area alone.

## 3. Tall-grass Pareto

Percentages below are fractions of the cap-free lateral candidate set, not of
the complete mesh. `Visible omitted` is a conservative pre-gate using a
`55 degree / 2160 px` angular
footprint and requires projected feature width `>=0.25 px`, projected span
`>=1 px`, and projected area `>=0.25 px^2` in the worst face-on orientation.
It is not an exact anisotropic pixel filter or image-space reconstruction.

| candidate | reads | retained | foliage | reproductive | crisp structure | visible omitted at 1 m | 8 m | 16 m | 32 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F=4`, 2 layers, 2.5 mm | 9 | 60.50% | 47.69% | 69.23% | 9.83% | 37.83% | 20.43% | 15.99% | 12.76% |
| `F=4`, 2 layers, 5 mm | 9 | 65.18% | 62.26% | 69.23% | 3.09% | 33.16% | 15.76% | 11.31% | 7.95% |
| `F=4`, 2 layers, 10 mm | 9 | 61.10% | 53.02% | 66.89% | 22.08% | 37.23% | 20.04% | 15.62% | 11.47% |
| `F=8`, 1 layer, 10 mm | 9 | 69.80% | 72.45% | 69.23% | 50.20% | 28.51% | 10.70% | 8.75% | 5.38% |

This confirms the user's distance observation quantitatively. Fine omissions
do fall away with distance; they are not charged indefinitely. It also shows
why distance cannot excuse the whole problem: at one to four metres, many
omitted patches remain multi-pixel, and some long recognition structures
remain visible much farther away.

The non-monotonic semantic percentages are real catalogue trade-offs, not a
claim that `10 mm` is worse geometry than `5 mm`: with only four fields, the
balanced greedy objective chooses a different finite catalogue. The complete
Pareto and selected fields are retained in `metrics.json`.

## 4. Joint residual bound

The fit follows the revised residual contract:

```text
w(u) = 4u(1-u)
one joint RGB/extinction field
K_live in {0,1,2,3,4} total after both layers
```

There is no hidden mode multiplier by species, colour, or height stratum. The
same selected frequencies fit extinction and all RGB channels. The gate is
still deliberately more permissive than production: it fits only the vertical
marginal and applies an offline pointwise positivity/`j<=kappa` projection.

At `K_live=4`:

- `F=4`, `5 mm`: cumulative omitted-mass sup error `8.12%`; cumulative
  max-channel colour sup error `7.48%`.
- `F=8`, `10 mm`: cumulative omitted-mass sup error `5.89%`; cumulative
  max-channel colour sup error `5.92%`.
- Sphagnum `F=4`: cumulative mass and colour sup error `1.69%`.

These are **height-cutoff marginal bounds**, not arbitrary live-ray alpha/RGB
errors. They show that the quadratic support window and four total modes are
not immediately refuted by vertical organization. They cannot certify local
tuft placement, silhouette coverage, coloured-head persistence, or a cutoff
at an oblique opaque-core hit.

## 5. Ribbon-factorization premise audit

For one authored ribbon quad, write its endpoint centres as `C0,C1`, endpoint
unit side vectors as `e0,e1`, and half-widths as `w0,w1`. Both
`appendBlade` and `appendLanceolateSurface` emit the two triangles

```text
(L0,L1,R0), (R0,L1,R1),     Li=Ci+wi ei, Ri=Ci-wi ei.
```

For any retained half-width `a<=min(w0,w1)`, the source trapezoid contains the
piecewise-affine ruled central strip `C(s)+t e(s)`, `|t|<=a`. The candidate
strip `C(s)+t e0` differs by at most

```text
a max_s ||e(s)-e0|| <= a ||e1-e0||.
```

The analyzer therefore uses `capacity=2 min(w0,w1)` and charges
`a||e1-e0||` as the side-frame-drift bound. Width taper itself is not an
error: choosing the minimum endpoint width keeps the central strip inside the
source trapezoid.

That same candidate parallelogram has two valid affine factorizations:

1. width extrusion: field axis `e0`, interval along `t`, mask boundary along
   `C0--C1`;
2. centreline extrusion: field axis `C1-C0`, interval along `s`, mask boundary
   across `[-a,a]e0`.

These alternatives never duplicate source area. Each atom categorically uses
whichever factorization best matches one field. Root fans and longitudinal or
axial caps remain excluded under both.

The original width-only premise was valid as a tolerance-bounded central
strip but incomplete. Adding the centreline factorization materially improves
the result:

- `F=4`, `5 mm`: `58.65% -> 65.18%` retained eligible area and
  `40.05% -> 33.16%` conservative visible omission at 1 m;
- `F=8`, `10 mm`: `63.29% -> 69.80%` retention and
  `35.47% -> 28.51%` conservative visible omission at 1 m.

The far-distance improvement is smaller because much of the extra recovered
area already falls below the projected-feature pre-gate. The full width-only
ablation remains beside the dual-factorization result in `metrics.json`.

## 6. Exact remaining gate

Before implementation, fit the one joint 3D field to pixel-footprint-filtered
reference rays for the selected `F=4` candidates and the `F=8` ablation:

1. reference coverage must be converted to finite optical depth only where
   filtered coverage is below the frozen residual maximum; opaque landmarks
   must remain categorical core failures rather than silently clamped fuzz;
2. use the actual quadratic window and charge the actual expanded
   `K_live<=4` after both layers;
3. score full and partial intervals ending before/behind opaque core hits;
4. score filtered alpha and premultiplied RGB at the exact world pixel
   footprints for `1, 2, 4, 8, 16, 32 m`;
5. retain the outside-start entry and inside-after-soft-cap lateral-exit
   orientations categorically; never emit or test a crisp cap event;
6. report silhouette IoU, largest connected multi-pixel omission, purple/cream
   head colour persistence, and millimetric translation stability.

If this gate passes for `F<=4`, runtime transcription is justified. If only
`F=8` passes, the one-layer geometry-variance loss must be surfaced to the
user. If the joint field cannot cover the measured multi-pixel omissions with
four total modes, return to the representation/quality decision; do not add
runtime reads, modes, direction cells, or a march.

## 7. Reproduction and artifacts

Command:

```text
node --import tsx tools/groundcover-bake/analyze-lateral-core-analytic-residual.ts
```

Artifact:

```text
data/work/groundcover-lateral-core-residual-gate/
  fa388df0604b80cc/50bc3ef7946bc8ea/
```

- metrics SHA-256:
  `36109d8e7f9eff6dcefd94b11b0f76371ec29b12e7587b543a10d6e3ec24a556`
- source-set SHA-256:
  `fa388df0604b80cc2c029e2e49557c55f6ba8e522449bc707abffc82a51f86e3`
- recipe SHA-256:
  `50bc3ef7946bc8eaacac0f8f7401c7a7a58ef33e3b00e80d08e39409d05e2f00`
- analyzer SHA-256:
  `8393ab25caa9123bf41808fa215624752686291fca9e0a5102a9b39891c9ebdf`

Focused source tests pass `11/11`; the analyzer passes a focused TypeScript
check and `git diff --check`.
