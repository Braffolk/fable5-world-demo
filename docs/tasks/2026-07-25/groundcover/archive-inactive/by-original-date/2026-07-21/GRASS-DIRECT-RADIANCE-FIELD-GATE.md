# Direct premultiplied radiance-field gate

Date: 2026-07-22  
Decision: **reject as the grass representation; preserve the low-angle lattice
result and the radiance/depth separation**  
Scope: actual checked-in Calamagrostis GCRP/v4, offline analytic truth and bake
simulation only; no runtime or shader file changed.

## Inspectable outcome

A literal interpretation of Sannikov's preserved video description was tested:
the repeating texture stores the exterior radiance field itself. The exact live
ray supplies fixed-top-plane phase and direction. Four surrounding angular
nodes are read, each with ordinary periodic hardware-equivalent spatial
bilinear filtering, and only premultiplied authored RGB and coverage are
bilinearly combined. Depth, owner, normal, and reconstructed position never
participate in colour.

This removes the current widening-triangle mechanism **by construction**: the
method never decodes a canonical hit and moves it to another point. It does not
make the image correct. Under the existing resident byte cap it replaces
geometric stretching with broad wrong-view cross-fades, ghost/lost silhouettes,
washed botanical colour, and millimetric-path class changes. The strongest
memory-neutral lattice has `0.85167` aggregate silhouette IoU but `0.21020`
median and `0.54714` p95 composited max-channel RGB error. Its connected
high-error region covers up to `59.51%` of a frame. This is not bounded one-
texel silhouette blur.

Low-angle rows are nevertheless decisively necessary. Adding exact 5- and
1-degree rows raises 5-degree IoU from `0.87240` to `0.99479` and 1-degree IoU
from `0.83854` to `1.0`; panicle retention rises from `0.39867/0.42373` to
`0.94020/1.0`. Those nearly saturated coverage scores are not radiance
correctness: 5- and 1-degree RGB p95 errors remain `0.49280` and `0.52031`.

## Exact image field

For descending live ray `X(t)=o+td`, intersect the fixed botanical top plane
`y=H`:

\[
q=o+\frac{H-o_y}{d_y}d.
\]

The exterior radiance field is

\[
L(q,d)=(A(q,d)c(q,d),A(q,d)),
\]

where `c` is exact authored first-hit RGB and `A` is binary point coverage at
the bake lattice. For neighbouring azimuth/elevation nodes `d_ij`, let `B_q`
be periodic bilinear filtering in phase and `w_ij(d)` the four fixed angular
weights. The tested reconstruction is simply

\[
\widehat L(q,d)=\sum_{i=0}^1\sum_{j=0}^1
 w_{ij}(d)\,B_q[L(\cdot,d_{ij})].
\]

Premultiplication makes the colour/coverage interpolation algebraically valid
as an image filter. It does not make the discontinuous 4D visibility signal
band-limited. A thin blade that exists in one neighbouring view and not another
becomes a fractional ghost; two different first surfaces become a colour
cross-fade. Because output remains at the original screen ray, this cannot
produce a relocated geometric wedge, but a connected wrong-radiance sector can
still occupy most of the image.

The fixed top-plane chart is origin-invariant for exterior rays. Exactly
horizontal rays have `d_y=0` and no finite `q`; the finite gate ends at one
degree. A camera inside the cover needs pointed-line origin/successor state and
is not represented by this exterior field.

## Memory-neutral lattices and fixed live cost

Every allocation is conservatively charged the complete current acceptance
footprint of 12 bytes per guarded phase/direction cell, even though the tested
colour query only needs premultiplied RGBA. The hard cap is `51,121,152` bytes.

| allocation | elevation nodes | phase | directions | charged bytes | cap left |
|---|---|---:|---:|---:|---:|
| current | `15,35,55,75` | 256 | 64 | 51,121,152 | 0 |
| low-angle | `5,15,35,60,85` | 224 | 80 | 49,032,960 | 2,088,192 |
| balanced | `1,5,15,35,60,85` | 208 | 96 | 50,803,200 | 317,952 |
| balanced + pole | `1,5,15,35,60,85` plus shared `90` | 207 | 97 | 50,844,684 | 276,468 |

The shared vertical pole is one direction, not sixteen duplicated azimuths.
It is the strongest tested sphere lattice and prevents a knowingly inferior
near-vertical design from deciding the gate.

Prospective colour cost is four spatially filtered RGBA reads, approximately
28 scalar lerp FMAs, one top-plane intersection, fixed direction-coordinate
math, and four atlas addresses. There is one filterable binding and no runtime
loop, march, traversal, candidate list, owner/normal/depth read, relift, pass,
barrier, distance-dependent work, or per-species query. The CPU BVH, loops, and
sixteen point contributors exist only to bake and validate those four filtered
reads.

## Actual-source gate

Immutable source:

- `src/assets/groundcover/calamagrostis-canescens.gcrp`;
- GCRP/v4, `118,663,280` bytes;
- `2,171,134` decoded triangles on the actual 0.52 m periodic tile;
- SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.

Each allocation was tested on exact analytic first-hit images at top-down,
18, 10, 5, and 1 degrees. Every row has 768 pinhole rays. Numbered contact
sheets show exact authored-colour composite, direct four-read prediction,
error class, and predicted coverage.

### Aggregate comparison

| allocation | silhouette IoU | RGB p50 / p95 | partial coverage | panicle retained | largest connected error region |
|---|---:|---:|---:|---:|---:|
| current | 0.74010 | 0.24976 / 0.65537 | 57.66% | 41.03% | 76.30% frame |
| low-angle | 0.85146 | 0.21389 / 0.55762 | 44.11% | 74.45% | 62.63% frame |
| balanced | 0.85376 | 0.21048 / 0.54507 | 42.68% | 78.51% | 54.56% frame |
| balanced + pole | 0.85167 | 0.21020 / 0.54714 | 44.01% | 79.67% | 59.51% frame |

The direct field improves the exact failure family from geometric wedges to a
screen-space filter, but its residual is not small or local. In the strongest
allocation:

| view | IoU | RGB p50 / p95 | partial coverage | panicle retained | largest error region |
|---|---:|---:|---:|---:|---:|
| 90 degrees | 0.26556 | 0.02933 / 0.33272 | 61.20% | 76.00% | 0.78% |
| 18 degrees | 0.68541 | 0.24574 / 0.59786 | 84.24% | 51.47% | 46.88% |
| 10 degrees | 0.90601 | 0.26162 / 0.63881 | 55.86% | 42.96% | 59.51% |
| 5 degrees | 0.99479 | 0.24161 / 0.49280 | 18.62% | 94.02% | 30.47% |
| 1 degree | 1.00000 | 0.22895 / 0.52031 | 0.13% | 100.00% | 25.65% |

The exact shared 90-degree node still has only `0.26556` IoU. This isolates a
second problem from angular spacing: 2.512 mm phase sampling plus bilinear
filtering crosses the extremely thin top-view coverage boundaries. At grazing
angles coverage saturates because almost every ray hits something, while the
first surface's authored appearance is still usually wrong.

### Camera-path stability

Four frames are translated laterally by `0/1/2.5/4.5 mm`; 192 stable screen
locations per view are compared. For the strongest allocation, p95 predicted
RGB change in excess of truth is `0.09182/0.15416/0.10588/0.13824/0.22885`
at `90/18/10/5/1` degrees. Even when the truth family is unchanged, predicted
family changes occur in `20.97/45.15/41.75/26.39/36.12%` of comparisons.
Thus filtering smooths some true high-frequency changes but introduces its own
wrong-view colour/class shimmer; it is not a stable botanical image field at
this lattice density.

## Depth and visibility are a separate rejection

No depth was used to form the reported image. Four separately scored elections
show what remains if the radiance result later needs real occlusion/depth:

- alpha-conditional mean drop;
- the dominant angular filtered tap's conditional drop;
- the dominant categorical phase/direction corner;
- a deliberately optimistic offline oracle choosing the strongest hit among
  all sixteen point contributors.

For the strongest lattice their same-live-ray aggregate position p95 errors
are `6.083/6.310/6.869/6.737 m`. At 5 degrees the conditional mean is
`7.165 m` p95; at 1 degree it is `7.310 m`. Even the sixteen-point hit-aware
oracle is not an event correspondence law. These numbers must not be used to
discredit the direct image separately: they establish that good binary alpha
at grazing views does not supply a correct surface for scene visibility,
intersection, normal, or world depth.

Writing only terrain/carrier depth would turn the accepted-looking low-angle
alpha into a textured plane. Arithmetic depth moments create an intermediate
sheet. Categorical depth chooses an unrelated first event. None is accepted.

## Provenance and publication boundary

The literal repeating-radiance-field interpretation is motivated by the
preserved Sannikov video description in
`docs/deep-research/grass/SANNIKOV-2024-YOUTUBE-COMMENTS.md`: many repeated
copies use one repeating hardware-sampled field at copy-count-independent
cost. Sannikov does not publish the later interpolation/election rule. No rule
here is attributed to him beyond that recorded description.

Four-dimensional exterior light-field resampling is prior-art context from
Levoy and Hanrahan, *Light Field Rendering* (SIGGRAPH 1996), and spatial/
angular sampling is context from Chai et al., *Plenoptic Sampling* (SIGGRAPH
2000), already recorded in the central source ledger. The premultiplied-only
gate, exact capped allocations, shared vertical pole, actual-Calamagrostis
truth, temporal/class metrics, radiance-versus-depth separation, and negative
result are LAAS derivation and measurement.

## Reproduction and artifacts

Run:

```sh
node --import tsx tools/groundcover-bake/analyze-direct-radiance-field.ts
```

- Tool SHA-256:
  `d10b02a19123abf8c809eb1428922be09fc1a7514dded238af059b6091add495`.
- Recipe SHA-256:
  `83b63ea1cb677fa428ba2a92e380cbe8e6f38b8beac6b74c274599792a41818e`.
- Artifact root:
  `data/work/groundcover-direct-radiance-field/2ed57f59d86e8376/83b63ea1cb677fa4/`.
- `metrics.json` SHA-256:
  `ced4601918f74892fea4bc7228feabfa1709c6632cc4434c40006d3075030f71`.
- `index.json` SHA-256:
  `f2287b18ec74f78818c386ff1b41ce19a154a77e18e225dabab2505cbaf0fa1f`.
- Strongest contact sheet:
  `qa/004-balanced-p207-a16-e6-plus-pole.png`, SHA-256
  `49efb79f479897e5c1d523a95f5a17ffdc593f5c35624f4e1bbbfede6eeeaafe`.

The machine index binds every numbered PNG to its dimensions, hash, and panel
interpretation.

## Decision

Reject direct four-read multilinear premultiplied radiance as the complete
grass representation under the current byte cap. It genuinely removes event-
relift stretching, so preserve that distinction; it fails because the sampled
4D visibility/appearance field is not band-limited enough, not because a depth
decoder was attached to it.

Preserve the exact low-angle result: a future successful fixed-read method must
include support at approximately 5 degrees, and approximately 1 degree if that
finite grazing view remains in contract. A shared vertical pole is the correct
non-wasteful top-view node. Resume this track only with a new interpolation or
prefilter invariant that bounds connected colour/coverage error on the actual
thin geometry and separately supplies a correct pointed-line event for depth.
Do not respond by adding runtime samples, marching, loops, candidates, passes,
per-species fields, or memory. Multiple overlapping species and moss remain one
offline-unioned marked community before any successful field is baked.
