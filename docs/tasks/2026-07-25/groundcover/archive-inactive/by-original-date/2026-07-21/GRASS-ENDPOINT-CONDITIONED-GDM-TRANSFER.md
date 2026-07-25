# Endpoint-conditioned GDM transfer gate

Status: **NO-GO: direct 4D lattice and grouped rank-eight fit both rejected**, 2026-07-22. No runtime shader implements either layout.

This note evaluates only the new fractional marked-transfer target. It does not
revive interpolation of opaque owner ids or first-hit depths.

The current live projected/categorical reconstruction is a **hard visual
rejection**. The latest user comparison still shows the same stretching and
wrong perspective, with no improvement. Nothing in this note reinterprets
that output as acceptable or uses it as evidence for the new field; only a new
held-out endpoint-transfer camera path can change that verdict.

## 1. Actual baseline

The accepted Calamagrostis GCRP/v4 header declares:

- `256 x 256` interior phase samples and one wrapped gutter on every side,
  hence `258 x 258` stored samples per direction;
- `16` azimuths and elevations `15/35/55/75` degrees, hence `64`
  directions;
- an `8 x 8` atlas;
- one eight-byte geometric `RGBA16F` runtime record. The standalone authored
  colour transcode adds one four-byte `RGBA8` record.

There are exactly

\[
64(258)^2=4,260,096
\]

stored records. The live geometry and colour allocations are therefore

\[
4,260,096(8+4)=51,121,152\ \text{bytes}.
\]

The `118,663,280`-byte source container is not the resident default: its owner
and packed source-geometry tables are load-time authoring data unless the
explicit owner diagnostic is enabled.

The default standalone shader evaluates two fixed anti-tiling transforms. Each
depth candidate performs four angular-corner `RGBA16F` filtered samples. After
election it performs four more `RGBA16F` samples for the normal and four
`RGBA8` samples for authored colour. Thus the accepted comparison point is:

- `16` hardware filtered sample instructions per covered output pixel;
- `64` physical texel fetches because each instruction is spatially bilinear;
- approximately `448` uncached payload bytes (`12*4*8 + 4*4*4`), before
  cache-line and texture-compression effects.

## 2. Transfer record

For a finite segment of length `L`, store the complete offline-composed union
of every participating plant and cover type as

\[
(A,m,C,N),\qquad A=1-T,\qquad m=M/L,
\]

where

\[
M=\int_{[0,L)} s\,dA(s),\quad
C=\int_{[0,L)} c(s)\,dA(s),\quad
N=\int_{[0,L)} n_f(s,d)\,dA(s).
\]

`C` and `N` are premultiplied linear moments. Misses are exactly zero. The
half-open endpoint reserves the opaque scene event at `L` for the background
resolve, so it cannot be counted once in `C` and again as `scene`. A source
cell boundary inside the segment uses the fixed half-open ownership convention
from Section 4: the following segment owns its origin. `n_f` is face-forwarded
to `-d` before accumulation. The
representative conditional distance, when one is needed, is

\[
\mu=\frac{M}{A}=L\frac{m}{A}.
\]

For adjacent segments, with the second segment's moment measured from its own
start,

\[
\begin{aligned}
A_{12}&=A_1+(1-A_1)A_2,\\
M_{12}&=M_1+(1-A_1)(M_2+L_1A_2),\\
C_{12}&=C_1+(1-A_1)C_2,\\
N_{12}&=N_1+(1-A_1)N_2.
\end{aligned}
\]

This avoids the long-range cancellation of the rejected transmittance-tail
`Q` parameterisation. It does not make a boundary-to-boundary record
origin-aware.

## 3. Best literal dense 4D packing

One filterable twelve-byte record is sufficient:

- `RGBA16F`: `(A,m,C.r,C.g)`, eight bytes;
- `RGBA8_UNORM`: `(C.b,0.5N.x+0.5,0.5N.y+0.5,0.5N.z+0.5)`, four bytes.

The affine normal encoding remains linearly filterable: interpolating
`0.5N+0.5` and decoding `2x-1` returns the interpolated signed normal moment.
Its eight-bit precision is only a hypothesis for low-mass plume samples and
would still require an actual-source normal-error gate.

At the current `64 x 258 x 258` lattice this packing is exactly
`51,121,152` bytes. It leaves no byte for a five-degree row, horizontal row,
reverse directions, vertical poles, mip levels, or a distinct opaque
structural carrier.

The least-bad all-angle direct array uses signed elevation rings

```text
-75 -55 -35 -15 -5 0 +5 +15 +35 +55 +75 degrees
```

at sixteen azimuths plus the two vertical poles: `178` directions. A
`154 x 154` stored tile (only `152 x 152` interior samples with one gutter)
occupies

\[
178(154)^2(12)=50,657,376\ \text{bytes}.
\]

That retains only `35.3%` of the current phase samples and increases the
Calamagrostis phase pitch from about `2.03 mm` to `3.42 mm`. It still has no
mips. A downward-only `0/5/15/35/55/75` lattice plus one vertical pole can use
`97 x 209 x 209 x 12 = 50,844,684` bytes, but it cannot serve upward
camera-inside rays.

Direction records use bilinear spatial filtering followed by linear mixing of
the four angular-corner transfer moments. Each row first reconstructs
`M_i=L_i m_i`; only the linear quantities `(A,M,C,N)` may be angularly mixed.
This is a mixture of prefiltered ray measures, not an interpolated opaque hit.

If the two current anti-tiling transforms remain, election can read only the
`RGBA16F` record for both candidates and retain one blended `vec4` per
candidate. The elected candidate then reads `RGBA8` once at four angular
corners. That is `12` filtered instructions, `48` physical texels and about
`320` uncached bytes, below the present `16/64/~448` traffic. It carries two
additional blended `vec4` candidate records through election, so register
pressure may rise even though traffic falls; generated Metal and occupancy
would have to decide that question.

## 4. Why a boundary chart does not make the direct field origin-aware

For a canonical half-open cell and an inside origin `o`, define the positive
time to each forward face and choose

\[
L_1=\min(t_x,t_y,t_z,L_{scene}).
\]

Ties use one fixed axis priority. The first segment is `[0,L_1)`; an event
exactly on the exit face belongs to the following segment. An X/Z exit wraps
to the opposite half-open face. A Y exit terminates the botanical slab. For
`d_y=0`, no Y exit is invented; exact horizontal rays use the real finite
scene/horizon endpoint.

This gives deterministic cell ownership, but it does not supply the transfer
from `o` to the first face. A boundary-to-boundary record begins before `o`.
Full-cell composition supplies only the later segments. The missing first
partial segment depends on the origin's longitudinal position on the line, a
genuine fifth coordinate.

The loss is information-theoretic even with `(A,M,C,N)`. For example, a
boundary segment can contain two transfer measures with identical total mass,
first moment, colour and normal but different mass before and after an inside
origin. Their whole-segment records are identical and their forward suffixes
are different. No fixed ALU transform of the whole record can right-censor it.

Adding only two longitudinal strata already forces the all-angle dense layout
from `152` to about `107` interior phase samples. A dense origin coordinate at
the current phase resolution is multi-gigabyte. Exact horizontal rays make the
same defect especially clear: the first X/Z cell exit is finite, but the
camera-to-exit partial segment is not a boundary-anchored query.

Therefore the literal dense 4D layout is **NO-GO** for the declared arbitrary
camera-inside contract. A five-degree row repairs neither this missing
coordinate nor exact horizontal support. This is independent of shader code.

## 5. Stronger GDM-grouped rank-eight candidate

The one resource-feasible follow-up is the full pointed endpoint-conditioned
field

\[
F(u,v,y,\phi,\theta,L),
\]

where `(u,v)` are periodic source phase, `y` is the physical inside-origin
height, `(phi,theta)` is the full-sphere direction and `L` is the actual
scene-clipped finite endpoint. Exterior queries first anchor to the algebraic
slab entry. Inside queries use the physical camera origin. Horizontal queries
use `theta=0` and their real endpoint; there is no division by `sin(theta)`.

Use the GDM-style grouping

\[
z_r=W_r(u,v,\phi)E_r(y,\theta,L),\quad r=1\ldots8,
\qquad o=Bz+b,
\]

with one fixed `8 x 8` output head for `(A,m,C,N)`. This grouping couples
azimuth to horizontal phase and elevation/endpoint to origin height; it is
materially different from a fully separated spatial-volume times direction
basis.

The bounded head is frozen before fitting. For raw
`o=(a,q,c_r,c_g,c_b,n_x,n_y,n_z)`, use

\[
\begin{aligned}
A&=\operatorname{saturate}(a),\\
m&=A\operatorname{saturate}(q),\\
C&=A\operatorname{saturate}(c),\\
N&=A\,n/\max(1,\lVert n\rVert).
\end{aligned}
\]

At `L=0` the record is selected to exact zero. The gate reports raw
out-of-domain and saturation rates; clamps may enforce a physical payload but
may not hide an underfit. Representative distance is
`mu=L*m/max(A,epsilon)` only on `A>0`.

One concrete resident layout is:

| field | format and dimensions | complete mip bytes | filtered samples |
|---|---|---:|---:|
| `W[0:4]` | `RGBA16F`, `212 x 212 x 48` 2D array, spatial mips | 22,993,920 | 2 (two azimuth layers) |
| `W[4:8]` | same | 22,993,920 | 2 |
| `E[0:4]` | `RGBA16F`, `64 x 193 x 20` 3D, full mips | 2,256,384 | 1 |
| `E[4:8]` | same | 2,256,384 | 1 |
| **total** |  | **50,500,608** | **6** |

The `193` signed-elevation nodes include exact `-90/0/+90` degrees. The `48`
azimuth layers wrap every `7.5` degrees. The `L` coordinate must use a declared
near-resolving nonlinear map; twenty uniform samples over `155 m` would be
categorically inadequate. The map and held-out endpoint errors are part of the
fit gate, not a shader tuning parameter.

A 2D array is required for `W`: its mip chain reduces phase only and retains
all 48 azimuth layers. A `212 x 212 x 48` 3D texture would incorrectly halve
angular resolution whenever spatial LOD increases. Per query the array uses
four bilinear W samples (`16` texels) and the two E textures use two trilinear
samples (`16` texels): `6` instructions, `32` physical texels, approximately
`256` uncached bytes. The elementwise product costs eight multiplies and the
dense output head costs 64 FMAs. Bounded output transforms, endpoint mapping,
normal decode and lighting remain comfortably below the stated 256-FMA
ceiling, but special-function and register cost still require generated-shader
inspection.

The resident mip bytes are exact, but “mip-correct” would overstate the
sampling result. A compute shader has no implicit fragment derivatives; it
must use an explicit level or explicit gradients. A 3D mip of
`E(y,theta,L)` also reduces all three axes together, so it can blur endpoint or
elevation merely because another axis has a large footprint. The fit gate must
freeze that LOD/gradient rule and evaluate quantised, hardware-filter-shaped
tables under it. LOD-zero metrics alone do not validate the distant view.
Independently filtering `W` and `E` is also only an approximation to filtering
their product under a correlated screen-ray footprint.

The four sampled `vec4`s, two rank vectors and two output vectors create a
roughly 24-scalar live working set before coordinates. This may reduce
occupancy; the byte/read proof does not predict occupancy.

The intended runtime is one field query, not the current two transformed
copies and not one query per species. Species, moss and anti-tiling variation
are composed into the offline community. Retaining both current transforms
would double this candidate to `12` filtered instructions, `64` texels and
about `512` uncached bytes, exceeding the present byte traffic; that variant is
rejected.

The field is evaluated in the existing grass compute dispatch with
`L=min(tScene,tSlab,tHorizon)`. One `RGBA16F` transfer target is **not** a
complete replacement for the current `grassRayNrm`: `C+A` consumes four
channels but resolve still needs the representative distance and normal to
light the cover at the foreground interaction rather than at the opaque
background.

A logically zero-net replacement is instead:

| screen target | payload | bytes/pixel | access |
|---|---|---:|---|
| `RGBA8_UNORM` | `C.rgb,A` | 4 | compute storage write, resolve exact-pixel load |
| `R32UINT` | f16 representative `mu` + 2x8-bit oct normal | 4 | compute storage write, resolve exact-pixel load |
| **total** |  | **8** | |

This is the same logical `8*width*height` bytes as the existing single
`RGBA16F grassRayNrm`. `r32uint` supports storage writes and exact integer
texture loads in WebGPU; it is not filtered. The normal is face-forwarded
before oct encoding. `mu` is ignored on `A=0`, and an incoherent near-zero
normal moment uses the declared terrain/up fallback before packing.

Resolve gains one screen-texture read (two exact loads replace its current one)
and composites `C_lit+(1-A)scene` in the existing resolve pass. The factor
query removes the current angular owner/depth-normal/authored-colour sampling,
so the added screen load does not erase its texture-traffic reduction. There
is no new pass, dispatch, march, candidate list, barrier, or species loop.

Zero-net screen bytes do not imply zero-net shading work. The opaque-winner
path shades either scene or grass once. Fractional composition must obtain the
opaque scene radiance and also light the ground-cover material moment at `mu`;
dynamic sun/shadow/GI can therefore add a second lighting evaluation on covered
pixels unless a separately justified approximation is used. Atlas-sample
savings may offset it, but only a generated shader/trace can establish that.

`RGBA8` precision is not accepted by algebra alone. Premultiplied pale plume
values below one UNORM step can lose chroma or disappear. The runtime-shaped
gate must compare direct premultiplied storage with the alternative exact-pixel
packing `(conditional C/A).rgb,A` followed by multiplication after load, and
must report sparse-plume colour/coverage error. No dither or stochastic
coverage is silently permitted.

Bindings change from one profile plus one colour texture to four sampled
factor textures. Access is read-only and spatially coherent. Synchronisation
and dispatch shape do not change. The grass compute uses two screen storage
targets instead of one; resolve uses two exact screen loads instead of one.
That is zero-net logical screen bytes, not zero-net binding count, and must be
checked against the generated compute/fragment resource layouts.

### 5.1 Mathematical/interface closure audit

Within the canonical flat periodic slab, the variables do name one finite
pointed ray. Explicitly,

\[
A=(T_xu,y,T_zv),\qquad
d=(\cos\theta\cos\phi,\sin\theta,\cos\theta\sin\phi),\qquad
R=\{A+td:0\le t<L\}.
\]

Periodic `(u,v)` representatives name translated copies of identical source
geometry. At `theta=+/-90 degrees`, all `phi` values are the same vertical ray
and must be identified by the pole constraint; at `L=0`, direction is
irrelevant. Apart from those required quotient identifications, there is no
missing longitudinal/origin coordinate. Inside, exterior-after-entry, and
exact-grazing rays are therefore uniquely specified in this source domain.

Unlike the rejected cell-exit head, this factorisation places endpoint `L`
inside `E(y,theta,L)` before multiplication by the phase/azimuth factor. It can
therefore represent a boundary-state-dependent length response; it does not
force the mixed `(phase,L)` cross-difference to zero. Rank eight may still
underfit that response, but the endpoint coordinate is present rather than
added after the factorisation. This removes the earlier finite-horizon
*structural* blocker and makes the fit falsifiable.

It does **not** mathematically prevent the reported visual failure. Linear
interpolation happens in eight latent factors, not in the physical transfer
measure. If rank eight cannot follow a fast phase/direction visibility change,
the head can decode a broad ghost opacity between two valid rays. In a grazing
camera path that angular support projects over a world width proportional to
range, so it can still appear as a widening triangle, translucent sheet, or
wrong-view band even though there is no discrete nearest-direction selector.
Output clamps can turn the same underfit into flat plateaus. Therefore
continuity/no-Voronoi-selection is not an acceptance proof: held-out
low-oblique paths must explicitly test screen/world error growth against
distance, and any recurrence of those shapes rejects rank eight without rank,
view, or read growth.

The pointed coordinate domain is complete only with the following clipping:

- inside origin: start at the physical camera origin;
- exterior origin: analytically intersect the finite botanical slab, start at
  its first entry, and subtract that entry distance from the opaque endpoint;
- terminate at the minimum of opaque-scene distance, slab exit, and declared
  finite horizon;
- exact horizontal inside rays use `theta=0` and the finite scene/horizon
  length; exact vertical rays use the slab exit, without a `sin(theta)` divide.

A ray with no opaque scene winner still needs a declared sky/far endpoint and
an existing sky-pass composite. Otherwise upward camera-inside silhouettes are
outside the candidate despite the full-sphere table. Adding the same overlay
loads to an existing sky pass need not add a pass, but it is a separate binding
and correctness gate.

The quotient seams are also explicit fit identities. `u,v` and the 48 azimuth
layers wrap periodically, including interpolation from layer 47 to layer 0.
At `theta=+/-90 degrees`, azimuth is undefined: predictions and their limiting
values must be independent of `phi` while retaining the correct `(u,v)`
dependence. Duplicate-pole and both-side seam paths belong in held-out QA, not
only the training loss. `L=0` must decode the exact zero record, and for fixed
pointed rays increasing `L` must obey monotone opacity and right-censoring
identities.

The composite `C+(1-A)scene` is exact only if `C` is premultiplied outgoing
radiance for the current lighting. The stored source field instead naturally
contains authored colour/material moments. Resolve can shade `C/A` using the
packed representative depth and normal, then premultiply by `A`, but that is a
single-representative lighting approximation when illumination varies through
the segment. Dynamic sun/shadow/GI cannot be pre-baked into `C`; the fit must
not label material-moment accuracy as exact radiance accuracy.

Keeping the opaque scene election intact and compositing in its existing
resolve removes the previous fractional-alpha interface blocker for colour.
It does not create a foreground depth layer: TAA reprojection, GTAO, later
transparent surfaces, and other depth consumers still see the opaque
background depth. The packed `mu` is available for ground-cover lighting and
motion work, but one depth cannot exactly represent a fractional depth
distribution. This downstream limitation must be evaluated at runtime rather
than hidden by writing an opaque mean surface.

Finally, the six coordinates are sufficient for the canonical straight-ray,
periodic-slab source. A terrain-following deformation is covered only if the
world-to-source map is proved line-preserving over the queried segment. If a
straight world ray becomes a curved source-space path depending on terrain
variation, `(u,v,y,phi,theta,L)` alone is not sufficient; the terrain transform
must be part of the truth generator/fit contract rather than an after-fit warp.
The same applies to live wind or other non-rigid source deformation: it needs a
proved analytic transport of this static field or an explicit state coordinate;
otherwise the six inputs no longer uniquely determine the deformed segment.

## 6. Decision and falsifiers

The direct dense 4D alternative is rejected. The grouped rank-eight layout is
**GO only for one offline actual-source fit**, not for shader integration. It
passes the coordinate/resource algebra but has unresolved quality risks which
cannot be decided from dimensions:

1. rank eight may not capture the phase-dependent endpoint crossings in `L`;
2. 48 azimuth layers and the nonlinear 20-sample endpoint axis may still miss
   the near-field angular/longitudinal bandwidth;
3. fitting independent transfer moments can violate monotonic endpoint growth
   and inside-origin right-censoring unless those coupled path identities are
   hard training/evaluation gates;
4. pole, periodic, and endpoint seams can still break after table sampling and
   quantisation;
5. `RGBA8` colour/coverage, f16 `mu`, and oct8 normal packing can erase or
   mislight sparse pale plume; and
6. arbitrary terrain deformation, sky rays without an opaque winner, and
   downstream fractional-depth consumers remain explicit integration gates.

The fit is rejected without rank/read growth if it shows any widening fan,
wrong-view sheet, distance band, panicle loss, non-monotone `A(L)`, boundary
discontinuity, or persistent-line censoring failure. Passing it would authorise
one measured shader integration; it would not itself establish visual
acceptance.

## 7. Actual-source rank-eight result: rejected before table sampling

The exact isolated production Calamagrostis source was traced into a
six-coordinate pointed-segment truth set. Each footprint uses four deterministic
phase samples inside one `212`-sample phase texel. Direction, physical origin
height, and endpoint are shared; every interpolated source normal is
face-forwarded so `dot(n,-d)>=0` before premultiplied accumulation. Outputs are
linear-premultiplied `(A,m,C,N)`, segment ownership is `[0,L)`, and the endpoint
coordinate is `q=log1p(10L)/log1p(1551)`.

Training excludes contiguous intervals in both phase axes, azimuth, origin
height, signed elevation, and endpoint. Separate truth covers exact horizontal,
exact vertical, camera-inside shifted-origin censoring, endpoint sequences, and
the preserved 17-frame/18-degree camera path. Endpoint and censoring gates use
physical `A` and `M=Lm`, never `m` by itself. Exact vertical truth is paired
under different azimuth labels and has zero measured phi disagreement.

Before interpreting the neural fit, an optimistic oracle arranged `65,536`
exact records as `256` fixed `W=(u,v,phi)` rows by `256` fixed
`E=(y,theta,L)` columns and fit independent best rank-eight SVDs to `A` and
`M`. This is strictly easier than shared W/E factors plus one head. It already
fails: rank eight retains only `59.99%` of A energy, coverage IoU is `0.4193`,
A p95 error is `0.2445`, M p95 error is `0.2015 m`, conditional-distance p95
is `3.391 m`, and plume recall is `0.9210`. The raw optimum also produces
19,302 A values outside `[0,1]`, 13,199 negative M values, and 24,541 samples
with `M>LA`; no clamp was used to hide this rank failure.

The frozen continuous model confirms the lower bound rather than repairing it.
On general held-out data, coverage IoU is `0.2655`, A/M p95 errors are
`0.3374/0.1998 m`, conditional-distance p95 is `6.764 m`, premultiplied colour
RMSE is `0.2770`, normal p95 error is `150.2 degrees`, and plume recall is
`0.4341`. On the exact 18-degree path, IoU is `0.4639`, conditional-distance
p95 is `4.543 m`, and the reconstruction visibly becomes a dark blurred sheet
with pale panicles largely removed. Censor A/M change errors are
`0.1323/0.1263 m` at p95; exact-vertical predicted phi invariance has a `0.708`
maximum error. Bounded output transforms pass A/M/C/N physicality but cannot
restore missing rank.

Decision: **reject and park rank eight.** The continuous fit fails, so the
`212x212x48` W and `64x193x20` E runtime tables were deliberately not sampled,
quantized, or filtered. No rank, read, architecture, memory, loop, march, pass,
binding, shader, or resolve change followed. Artifacts and numbered QA are at
`data/work/groundcover-endpoint-gdm/8cd69c2a6c61043c2861cad3a5ec06fee6dd4346a6722fd01fcc43468071ce5d/0526e98735b745f3074212ac07e1ff8d6e5c456061283b0d443e897955362207/`.
