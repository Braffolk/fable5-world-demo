# GBR4: joint exterior-line 4D brick codec

**Date:** 2026-07-23  
**Status:** v1 premise-audited and superseded by periodic-cap v2 below  
**Domain:** cameras outside the complete carrier box; fade after entry into it

## 0. Binding periodic premise correction

The first `GBR4/v1` minimal cook used the segment inside one `0.52 m` tile
box. That is exact for a finite, non-repeated soup contained by that box, but
**not for the infinitely repeated Calamagrostis community**. A ray can miss
the first tile segment and hit a later periodic copy. One v1 query stops at
the first tile exit, so the periodic v1 asset and its
`GREEN_CONTAINER_MINIMAL_COARSE` label are superseded and must not be bound by
runtime.

The periodic production address is instead one oriented line through the
vertical periodic slab:

`(capChart, entryPhaseX, entryPhaseZ, compactSlopeX, compactSlopeZ)`.

`capChart` is discrete top/downward or bottom/upward. The four continuous
coordinates are jointly bricked exactly as below. One query traces all later
periodic tiles up to the fixed visibility horizon or the opposite vertical
slab boundary. This is the correct one-query reduction for an exterior camera
above/below the complete infinite-periodic cover envelope.

The finite-soup face-pair derivation remains valid background for genuinely
finite non-periodic assets, but it is not the active Calamagrostis container.

## 1. Finite-soup address derivation (v1 background only)

For a convex carrier box, every non-tangent exterior camera ray that crosses
the box has one ordered entry face and one ordered exit face. The oriented
line inside the box is determined exactly by

`(entryFace, entryUV, exitFace, exitUV)`.

There are six faces and thirty ordered unequal face pairs. Within one pair the
continuous address is four-dimensional. Translating the exterior camera along
the same ray does not change this address. No view-direction bin, billboard,
shell height, or camera-inside state is present.

Face numbering and UV orientation are frozen:

| id | face | UV |
|---:|---|---|
| 0 | `x=min` | `(y,z)` |
| 1 | `x=max` | `(y,z)` |
| 2 | `y=min` | `(x,z)` |
| 3 | `y=max` | `(x,z)` |
| 4 | `z=min` | `(x,y)` |
| 5 | `z=max` | `(x,y)` |

Coordinates are normalized by the carrier bounds. Exact edge/corner ties use
the lowest face id among equal slab times. A same-face tangent has zero
interior measure and returns MISS/fade.

## 2. 4D blocks

Each scale level has `R` cells per coordinate and a fixed block edge `B`
cells. `R` is divisible by `B`; there are `(R/B)^4` descriptors per face
pair. A descriptor is one `u16` codebook id.

One codeword stores `(B+1)^4` shared boundary samples. The extra sample on
every axis makes adjacent blocks agree at a common node and allows local
four-dimensional interpolation without a second descriptor. Production VQ
must include boundary error in its objective and pass an explicit seam gate.
The minimal cook is lossless at block level: every block owns a codeword, so
it tests the container/reconstruction contract without claiming VQ quality.

Within one codeword, local coordinates `(l0,l1,l2,l3)` map to a 2D brick tile:

`x = l0 + (B+1) * l2`  
`y = l1 + (B+1) * l3`.

Thus a codeword tile is `(B+1)^2` square. Codeword tiles are packed into one
ordinary 2D atlas.

## 3. Stored filtered record

Every 4D node contains two prefiltered strata. For each stratum:

- premultiplied linear RGB + coverage in `RGBA16F`;
- categorical representative depth as `unorm16` of the live entry-to-exit
  segment length;
- categorical representative normal as octahedral `unorm8x2`.

The two color/coverage strata are separate `RGBA16F` atlases. The categorical
atlas is `RGBA16Uint`:

`[depth0, oct0x | oct0y<<8, depth1, oct1x | oct1y<<8]`.

Offline subrays are clustered into front/back depth strata. Premultiplied
color and coverage are integrated. Depth/normal are never numerically blended
across unrelated surface events: each stratum stores the actual subray hit
nearest its filtered conditional mean.

Each scale level is independently cooked from shared-origin pinhole pixels.
The live ray Jacobian chooses/interpolates levels; ordinary averaging of
nonlinear records is forbidden. The minimal asset has one `4 m` standoff,
`60 degrees / 1920` pixel level with `2x2` quadrature. It is structurally
loadable but not a complete distance cascade.

## 4. Fixed runtime reconstruction

1. Analytic ray-box slabs produce entry/exit face and UV (`0` texture reads).
2. A `6x6` constant table maps the ordered face pair to its ordinal.
3. Integer block coordinates load one `u16` codebook descriptor (`1` read).
4. For each stratum, hardware bilinear filtering handles local axes 0/1 at
   the four floor/ceil combinations of axes 2/3; fixed bilerp combines them
   (`4` filtered reads per stratum, `8` total).
5. The categorical node is the maximum 4D corner-weight node, equivalently
   round-to-nearest on each local axis; exact `0.5` ties choose the lower node.
   One `RGBA16Uint` load returns both depth/normal strata (`1` read).

Total: **10 logical reads** for a two-stratum BRICK lane. There is one
descriptor and no candidate list, loop, march, owner traversal, or runtime
geometry. Addressing, four bilerp weights, two color bilerp accumulations,
categorical decode and depth reconstruction are budgeted below about `150`
scalar ALU/FMA excluding the common analytic slab calculation. Control/Tier-1
layer reads are outside this community-terminal count.

Color interpolation is continuous inside and across lossless block borders.
Categorical depth/normal changes only at deterministic Voronoi midplanes. A
production VQ publication must separately gate excess output change under
`1..4.5 mm` camera translations.

## 5. Binary container (`GBR4/v1`, finite-soup background only)

All integers are little-endian. Sections are 256-byte aligned. Header size is
512 bytes.

| offset | type | field |
|---:|---|---|
| 0 | char[4] | `GBR4` |
| 4 | u32 | version `1` |
| 8 | u32 | header bytes `512` |
| 12 | u32 | profile id |
| 16 | u32 | face count `6` |
| 20 | u32 | ordered pair count `30` |
| 24 | u32 | continuous dimensions `4` |
| 28 | u32 | cells per dimension `R` |
| 32 | u32 | block cell edge `B` |
| 36 | u32 | block sample edge `B+1` |
| 40 | u32 | blocks per dimension `R/B` |
| 44 | u32 | scale count |
| 48 | u32 | descriptor format `1 = u16` |
| 52..72 | u32 pairs | descriptor, pair-table, scale-table offsets/bytes |
| 76..96 | u32 pairs | front color, back color, categorical offsets/bytes |
| 100,104 | u32 | atlas width, height |
| 108,112 | u32 | codeword tile width, height |
| 116 | u32 | codebook count |
| 120 | u32 | maximum descriptor id (`65535`) |
| 124 | u32 | flags: premul, two-strata, categorical, periodic-XZ |
| 128..140 | f32x4 | tile origin X/Z and size X/Z |
| 144..164 | f32x6 | carrier min/max XYZ |
| 168..199 | bytes32 | source SHA-256 |
| 200..231 | bytes32 | recipe SHA-256 |
| 232 | f32 | first-level pixel angle |
| 236 | f32 | first-level camera standoff metres |
| 240 | u32 | quadrature side |
| 244 | u32 | depth encoding `1 = segment unorm16` |
| 248 | u32 | face tie rule `1 = lowest id` |
| 252 | u32 | coordinate convention `1 = table above` |

Pair-table entries are 16 bytes: `entryFace:u8`, `exitFace:u8`, two reserved
bytes, `descriptorBase:u32`, `blockCount:u32`, four reserved bytes. Pair order
is lexicographic entry-major, excluding equal faces.

The first scale-table entry is 64 bytes. It freezes standoff, pixel angle,
descriptor first/count, codebook first/count, atlas dimensions, and reserved
future fitted-level fields. A future cascade adds independently fitted scale
entries and sections under a new version if offsets cannot remain global.

Descriptor linear order inside a scale is

`pair, block3, block2, block1, block0` (block0 fastest).

Atlas texels are row-major. `RGBA16F` and `RGBA16Uint` both consume 8 bytes.

## 6. Memory scaling

Let `C` be codewords and `S=(B+1)^4`. The three atlases cost

`C * S * (8 + 8 + 8) = 24 C S` bytes,

plus `2 * 30 * (R/B)^4` descriptor bytes and small tables/alignment.

The minimal real cook uses `R=6`, `B=3`, `C=480`, `S=256`:

- codeword payload: `2,949,120` bytes before rectangular-atlas slack;
- descriptors: `960` bytes;
- complete container: about `3 MiB`.

This is a container proof, not the quality setting. A production-resolution
asset must fit/validate block VQ or direct quantized bricks under the project
memory cap. Failure is reported; it is not repaired by runtime search.

## 7. Generality and remaining gate

The address/reconstruction applies to any finite opaque marked triangle soup
contained by a convex box. Periodic XZ copies are an optional cook flag, not a
mathematical dependency. Alpha-cut source geometry must first be diced into
equivalent opaque tokens or represented by an explicitly filtered source.

The minimal cook proves only: real-mesh truth, deterministic bytes, exact
loader layout, fixed reads, and lossless block seams at coarse resolution.
It does **not** prove Calamagrostis visual fidelity. Production publication
still requires adequate `R/C`, all footprint levels, cap/side/corner and exact
horizontal tests, held-out image/depth/normal/translation gates, and measured
simultaneous residency.

## 8. Active periodic-cap container (`GBR4/v2`)

### 8.1 Address

Chart `0` is the top plane with inward direction `d_y<0`; chart `1` is the
bottom plane with `d_y>0`. Coordinates 0/1 are periodic X/Z phase. Coordinates
2/3 compact the two tangent slopes. For inward magnitude `c=|d_y|`:

`s_x = d_x / (c + |d_x|)`, `s_z = d_z / (c + |d_z|)`, then store
`0.5*s+0.5`.

The inverse is `d_x/c=s_x/(1-|s_x|)` and likewise for Z, followed by
normalization and the chart's Y sign. It covers the complete inward
hemisphere; coordinate boundaries are the grazing limit. Exact grazing on a
guarded top/bottom plane is transparent. Horizontal views at a finite control
edge belong to that control-boundary representation; an exterior camera whose
height lies inside the infinite slab is already inside the permitted fade
envelope.

For each pinhole subray, the cook intersects the same cap plane independently
and invokes periodic first-hit truth for `155 m`, clipped by the opposite
vertical geometry boundary. A first-tile miss followed by any later periodic
hit inside that interval is therefore in the one stored query.

### 8.2 v2 header differences

The header remains 512 bytes and sections remain 256-byte aligned. Fields
not listed retain the v1 offsets.

| offset | type | v2 field |
|---:|---|---|
| 4 | u32 | version `2` |
| 16 | u32 | cap chart count `2` |
| 20 | u32 | address mode `2 = periodic cap phase+slope` |
| 60,64 | u32 | chart-table offset/bytes |
| 116 | u32 | codebook count |
| 124 | u32 | premul/two-strata/categorical/periodic flags |
| 232 | f32 | pixel angle |
| 236 | f32 | standoff metres |
| 240 | u32 | quadrature side |
| 244 | f32 | forward periodic horizon metres (`155`) |
| 248 | u32 | slope chart enum `2 = rational compact slopes` |
| 252 | u32 | cap convention enum `2` |

Chart-table entries are 16 bytes: `chartId:u8`, `inwardYSign:i8`, reserved
u16, `descriptorBase:u32`, `blockCount:u32`, reserved u32. Descriptor order is
`chart, block3, block2, block1, block0`.

Depth is v2 `unorm16` of the fixed 155 m forward horizon, so the categorical
world point is `capEntry + direction * (155 * depthUnorm)`. Misses are
distinguished by zero stratum coverage, never by depth.

The fixed brick decode stays one descriptor + eight filtered color reads +
one categorical read = **10 logical reads**. No loop, march, candidate, later
tile query, or runtime geometry is added.

### 8.3 Minimal v2 cook result

The corrected real-source cook is
`tools/groundcover-bake/cook-gbr4-periodic-cap-assets.ts`. Its content-addressed
report is:

`data/work/groundcover-gbr4-periodic-cap-assets/2ed57f59d86e8376/e29ac380de380f51/report.json`.

The stable v2 asset is
`src/assets/groundcover/calamagrostis-canescens.gbr4`, `222,464` bytes,
SHA-256 `573fb81057e217643addc4a8d4993ff6fcd65ef0bf9837962deedbd13a9c002a`.

The cook traced `9,800` non-grazing shared-origin subrays. `5,118` hit, and
**4,154 of those hits occurred only after the ray had left its entry periodic
tile**. Maximum sampled first-hit distance was `3.5123 m`, or more than six
tile widths. These 4,154 concrete counterexamples are absent from v1 and are
contained in the same one-query v2 node; they empirically exercise the exact
premise correction rather than merely asserting it.

This is `GREEN_PERIODIC_CAP_CONTAINER_MINIMAL_COARSE`, not visual production
acceptance. Moving finite-patch side-entry charts, production R/VQ, fitted
footprint levels, the held-out visual/translation gate, and runtime binding
remain open.

### 8.4 Runtime conformance result

The experimental v2 container was subsequently bound to the isolated
Calamagrostis runtime before the preceding open gates had passed. The complete
producer-to-resolve audit is recorded in
`GRASS-GBR4-RUNTIME-CONFORMANCE-AUDIT.md`.

Its verdict is **RED**: the ten-read brick decoder is mechanically consistent,
but the active path is not an implementation of the exterior boundary-transfer
model. In particular it lacks the carrier/finite-patch fields, fixed terrain
chart and winner-root correction, certified regular-versus-mixed record
semantics, and live footprint family required by the governing mathematics.
The v2 asset remains a container proof and may not be treated as visually
accepted grass.
