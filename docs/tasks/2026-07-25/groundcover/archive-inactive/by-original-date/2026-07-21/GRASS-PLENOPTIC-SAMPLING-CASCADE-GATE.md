# Fixed plenoptic sampling cascade gate

Date: 2026-07-22  
Decision: **reject and park after one actual-source attempt**  
Scope: offline mathematical/source gate only; no runtime asset or shader was
changed.

## Inspectable outcome

A six-level, one-read plenoptic cascade was given almost the whole resident
profile budget and tested against periodic Calamagrostis directional
microbundles. It
does not repair the widening fans or wrong perspective. Even before record
quantisation, it produces coherent wrong-event regions, removes panicle mass,
and changes representative botanical class under millimetric camera motion.
It also has no camera-inside successor coordinate and its fixed two-Y-plane
phase chart is singular at the exact horizontal.

This is not a GI radiance cascade, and no radiance-cascade claim is borrowed.
It is a bounded LAAS test of a conventional two-plane plenoptic sampling idea:
spend less phase resolution and more angular resolution as the pixel footprint
grows, while retaining one selected complete record read.

## 1. Fixed line coordinates and the error that must be bounded

For a non-horizontal oriented line `X(t)=o+t d`, intersect the two fixed
botanical planes `y=H` and `y=H-b`:

\[
q_H=o+\frac{H-o_y}{d_y}d,\qquad
q_{H-b}=o+\frac{H-b-o_y}{d_y}d.
\]

The periodic phase is `q_H,xz`, and the direction coordinate can equivalently
be written as the finite slope

\[
s=\frac{q_{H-b,xz}-q_{H,xz}}{b}=-\frac{d_{xz}}{d_y}.
\]

Replacing `o` by `o+lambda d` leaves both intersections unchanged. Thus this
is an exact origin-invariant exterior line address; it has no moving carrier
or camera-centred shell.

If a level has phase pitch `Delta q` and angular covering radius
`Delta theta`, then at range `r` even an ideal decoded sample has transverse
address uncertainty bounded below by the two independent terms

\[
\epsilon(r)\lesssim \frac{\sqrt 2}{2}\Delta q
                 +2r\sin\frac{\Delta\theta}{2}.
\]

For a pinhole pixel with half-angle `gamma`, its footprint grows as

\[
\rho(r)=r\tan\gamma.
\]

Trading phase for angle across levels can make `Delta q` follow `rho`, but it
does not make the angular requirement disappear: requiring the angular term
to remain within one pixel footprint gives, to first order,
`Delta theta <= gamma`, independent of distance. Cone-prefiltering changes the
signal bandwidth and appearance, not the geometric displacement of the line
address.

The hemisphere covering lower bound makes the cap conflict explicit. If `D`
direction cells have covering radius `theta`, their spherical caps must at
least cover area `2 pi`, hence

\[
D\,2\pi(1-\cos\theta)\ge2\pi,
\qquad
\theta\ge\arccos(1-1/D).
\]

The tested far level has 1,600 directions. Its analytic parameter-cell radius
is `0.039764 rad`; at the measured grazing top-plane range it contributes
`8.705 m` median and `13.564 m` p95 to `r Delta theta`. The finest 255-square
phase level can afford only 16 directions; its angular term is already
`0.277 m` at the 35-degree near view. This is the mathematical origin of the
connected widening sectors. More range levels redistribute the same
space-angle product; they do not remove it.

## 2. Complete record and resident allocation

The gate grants every selected cell an optimistic eight-byte complete record:

- separately prefiltered premultiplied authored radiance and coverage;
- panicle/vegetative marked mass;
- one actual conditional median surface event for depth and botanical class.

The evaluator uses floating oracle values before a concrete eight-byte
quantiser, so packing error cannot explain the rejection. A feasible packed
form would devote 32 bits to premultiplied colour/coverage/mark and 32 bits to
representative depth/normal/class. Geometry and appearance always come from
one selected record; unrelated scalar depths are never averaged.

| level | phase interior | angular grid | directions | stored records |
|---:|---:|---:|---:|---:|
| 0 | 255² | 4² | 16 | 1,056,784 |
| 1 | 160² | 7² | 49 | 1,285,956 |
| 2 | 96² | 11² | 121 | 1,162,084 |
| 3 | 56² | 17² | 289 | 972,196 |
| 4 | 32² | 26² | 676 | 781,456 |
| 5 | 18² | 40² | 1,600 | 640,000 |

One wrapped texel gutter is included on every phase level. Total storage is
`47,187,808` bytes, leaving `3,933,344` bytes below the hard `51,121,152`-byte
cap. The proposed live operation is fixed level selection, fixed plane
intersection, analytic phase/direction quantisation, and one complete-record
read. There is no live loop, march, traversal, candidate list, species
multiplier, geometry proxy, extra pass, or distance-dependent iteration.

Two-level interpolation was not attempted after the one-read end-to-end
failure. Arithmetic interpolation of neighbouring marked events cannot supply
an absent owner or successor and would not change the angular covering bound;
the track limit does not authorize another source attempt.

## 3. Actual Calamagrostis directional-microbundle gate

Immutable source:

```text
src/assets/groundcover/calamagrostis-canescens.gcrp
SHA-256 2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c
```

The evaluator traces the decoded-u16 mesh analytically through periodic tile
copies. It compares 3x3 common-top directional microbundles with independently
prefiltered cascade-cell records. The centre line is an exact camera line, but
the eight surrounding rays pivot at its top-plane intersection rather than at
the camera apex. This deliberately removes the additional top-plane phase
spread of a real pinhole cone, so the measured result is an optimistic
directional stress test, not a publication claim about exact pinhole-cone
filtering. The mathematical line-address bound and the horizontal/inside
contract failures do not depend on this simplification. The camera grid is
`12x8`; each of the seven
elevations has four frames translated by `0/1/2.5/4.5 mm`. The ordinary pixel
angle is `55 degrees / 2160 px`; the 0.1-degree row narrows its vertical sample
pitch only enough to keep the complete grid below the horizon. There are 2,688
camera pixels and 578 distinct optimistic oracle records.

The rows are `0.1/1/5/15/35/75/90` degrees. Their median top-plane ranges are
approximately `219/22.6/4.58/1.54/0.697/0.414/0.400 m`.

Aggregate failure:

| metric | measured |
|---|---:|
| representative-event position p50 | `1.7377 m` |
| representative-event position p95 | `84.3435 m` |
| hit/miss disagreement | `21.7634%` |
| coverage absolute error p95 | `0.6667` |
| premultiplied-radiance error p95 | `0.6708` |
| panicle-mass absolute error p95 | `0.7778` |
| temporal radiance excess p95 | `0.1085` |
| unforced class changes with stable truth | `426/1100 = 38.7273%` |

The failure is not confined to the deliberately difficult 0.1-degree row:

| elevation | position p50 / p95 | connected >=25 cm region p95 |
|---:|---:|---:|
| 1° | `6.155 / 19.446 m` | `24.955 m` |
| 5° | `2.032 / 8.305 m` | `10.212 m` |
| 15° | `1.696 / 2.879 m` | `14.125 m` |
| 35° | `0.021 / 0.864 m` | `0.726 m` |
| 75° | `0.141 / 0.161 m` | `1.109 m` |
| 90° | `0.457 / 0.532 m` | `1.991 m` |

The numbered contact sheet shows exact-mesh directional-bundle authored colour, cascade
prediction, and representative-event error. Broad constant categories replace
the microbundle structure, while connected orange/red regions reproduce the
wrong-view bands rather than decorrelated noise.

## 4. Contract assessment

- **Near crisp geometry:** fail. Even 75 and 90 degrees have `0.161 m` and
  `0.532 m` p95 representative-event error, far above the 2 cm gate.
- **Distant appearance:** fail. At 0.1 degrees, radiance p95 is `0.564`,
  panicle-mass p95 is `0.556`, and representative-event p95 is `117.523 m`.
  This is not accepted as a distant-only radiance fallback.
- **Coherent fans/range-locked bands:** fail. Connected wrong-event regions
  grow to 10--25 m at ordinary low-oblique elevations; shared categorical
  records form broad view/range bands in the QA image.
- **Temporal stability:** fail. More than 38% of comparisons whose truth class
  stays fixed acquire an unforced predicted-class change under millimetric
  camera motion.
- **Terrain depth:** a representative height can write depth on the exact live
  ray, but the record frequently belongs to another event. A radiance-only
  variant would instead collapse back to a textured terrain/carrier plane.
- **Camera inside:** fail by representation. An exterior bundle record has no
  pointed-line origin phase or ordered successor.
- **Exact horizontal:** fail by chart. A horizontal line is parallel to both Y
  reference planes, so `q_H` is at infinity. A side chart would consume another
  complete representation and still needs chart-coherent event identity.
- **Overlapping species and moss:** the one-read contract can contain one
  offline-unioned marked ecological community, never one live field per
  species. Unioning adds occlusion/successor entropy; it cannot rescue this
  already failed single-species carrier.

## 5. Provenance and claim boundary

Primary-source ancestry:

- Levoy and Hanrahan, [*Light Field Rendering*](https://graphics.stanford.edu/papers/light/),
  SIGGRAPH 1996: a static unobstructed-space light field is a 4D function and
  can be rendered by resampling stored views.
- Chai, Tong, Chan, and Shum,
  [*Plenoptic Sampling*](https://doi.org/10.1145/344779.344932), SIGGRAPH 2000:
  plenoptic sampling rates can be analysed through spatial/angular spectral
  support and scene depth range.
- Sannikov supplies only the repeating precomputed-field/O(1)-copy objective
  and the exact vertical-extrusion lift documented in the central source
  ledger.

The six-level allocation, fixed-line derivation, `r Delta theta`/phase budget,
complete marked record, actual-Calamagrostis cone test, fan metric, and negative
result are **LAAS derivations and measurements**. The cited papers do not claim
that this cascade solves thin grass visibility, camera-inside successors, or
overlapping groundcover. This track is also unrelated to GI radiance cascades.

## 6. Reproduction and immutable artifacts

```text
node --import tsx tools/groundcover-bake/analyze-plenoptic-sampling-cascade.ts
```

- Tool SHA-256:
  `1bfc55ac563a0465750247cbe58f06580820f6356c368982b95605af30475f19`
- Recipe SHA-256:
  `51129ab7d93cd865a5ceec90d268ed0a445f0c116f693bbcd58f8bd6ee450d9c`
- Artifact root:
  `data/work/groundcover-plenoptic-cascade/2ed57f59d86e8376/51129ab7d93cd865/`
- `metrics.json` SHA-256:
  `56755e7dce9e942c12240637451cc870c6481e71079cdae1820b3bc4f8cd988e`
- `index.json` SHA-256:
  `f0b00bb8de9af1613e016286823308a53ad4587c76ce8db9be7a45072bb7022f`
- `qa/001-truth-prediction-event-error.png` SHA-256:
  `0b10ab465c963a7a7e08c5a35d4c647b5a80fef62f4ba9b268ab0641731aa464`

## 7. Park/resume decision

Reject this plenoptic cascade as both the full carrier and a distant fallback.
Do not integrate it, add reads, blend levels, enlarge the cap, or hide the
horizontal/inside failures. Preserve the error derivation and actual-source
artifacts as a lower bound. Resume only if a new representation supplies a
path-independent marked event/successor law whose angular support is not paid
as a dense independent direction lattice.
