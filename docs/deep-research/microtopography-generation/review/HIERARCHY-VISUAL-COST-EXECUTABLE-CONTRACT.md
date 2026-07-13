# Hierarchy, Visual, And Cost Executable Contract

**Purpose:** exact normative corrections for H6-H8 of
`FINAL-SPEC-CRITIQUE.md`. This document is an input to the replacement spec; it
does not authorize implementation by itself.

**Code verified:** `asset-gen/config/base.toml`, `assetgen/grid.py`,
`assetgen/height_geom.py`, `cook/encode.py`, `cook/micro_hierarchy.py`,
`cook/micro_synth_cook.py`, `micro_verify.py`, `cook/chunkio.py`, `release.py`,
`src/core/Params.ts`, `src/core/FlyCamera.ts`, `src/debug/TerrainScene.ts`,
`src/nanite/world/TerrainField.ts`, and `tools/shoot.ts` on 2026-07-13.

## 1. H6: Exact Grid, Hierarchy, Quantization, And Seam Contract

### 1.1 Coordinate oracle

The following values and equations are frozen:

- CRS is L-EST97 / EPSG:3301. East is `+gameX`; south is `+gameZ`.
- World anchor is `A_E = 368640 m`, `A_N = 6635520 m`.
- The only negative-LOD coordinate oracle is
  `assetgen.height_geom.sample_center_en_units`; do not extend the integer-only
  helpers in `grid.py` to negative LODs independently.
- Geometry units are `U = 32 units/m`. Thus `A_Eu = 11796480` and
  `A_Nu = 212336640`.
- `R = 2048` is the core resolution and every payload is `(R + 1)^2 = 2049^2`.
- For physical height LOD `l in [-2,4]`, texel size in units is
  `T_l = 32 * 4^l` for `l >= 0`, `T_-1 = 8`, and `T_-2 = 2`.
- Chunk footprint is `F_l = R * T_l` units.
- Chunk `(l,cx,cz)` has north-west origin
  `O_E = A_Eu + cx*F_l`, `O_N = A_Nu - cz*F_l`.
- Payload sample `(row,col)`, including `row=R` or `col=R` apron samples, is
  exactly:

  ```text
  E_u = O_E + col*T_l + T_l/2
  N_u = O_N - row*T_l - T_l/2
  ```

  `T_l/2` is integral for every supported rung. Convert to meters only by
  division by `32` after this integer calculation.

The physical lattice remains:

| LOD | Texel | Chunk footprint | Payload |
|---:|---:|---:|---:|
| `-2` | `0.0625 m` | `128 m` | `2049 x 2049` |
| `-1` | `0.25 m` | `512 m` | `2049 x 2049` |
| `0` | `1 m` | `2048 m` | `2049 x 2049` |
| `1` | `4 m` | `8192 m` | `2049 x 2049` |
| `2` | `16 m` | `32768 m` | `2049 x 2049` |
| `3` | `64 m` | `131072 m` | `2049 x 2049` |
| `4` | `256 m` | `524288 m` | `2049 x 2049` |

Parent division is mathematical floor division, including for negative chunk
coordinates:

```text
parent(l,cx,cz) = (l+1, floor(cx/4), floor(cz/4))
child(l,cx,cz,dx,dz) = (l-1, 4*cx+dx, 4*cz+dz), dx,dz in {0,1,2,3}
```

In Python this is `cx // 4`, not truncation toward zero.

### 1.2 Parent reducer and read dependency

Each parent is made from browser-equivalent decoded immediate children, never
from the pre-quantized master and never by re-reading the raw DTM.

For parent `q=(l+1,qx,qz)`, the complete child-file read set is:

```text
D(q) = {(l, 4*qx+i, 4*qz+j) | i,j in {0,1,2,3,4}}
```

The north-west `4 x 4` subset supplies 16 child cores. The `i=4` column,
`j=4` row, and `(4,4)` corner are the nine east/south/southeast hierarchy-support
chunks needed for the parent apron. `assemble_parent_source_memmap` crops cores
and the first four support samples; it must not duplicate the last core row or
column.

Reduction is the existing fixed `4 x 4` area mean:

```text
parent[r,c] = mean_float64(child_mosaic[4*r:4*r+4, 4*c:4*c+4])
```

for all `r,c in [0,2048]`, using the existing reshape and `mean(axis=(1,3),
dtype=float64)` order in `box_mean4_striped`. Stripe size may change memory use
but must not change bytes.

### 1.3 Complete changed-core and apron closure through LOD4

Track core changes separately from files repacked only because their apron
depends on a changed neighbor.

For a set of changed-core chunk keys `C_l`, define:

```text
W(C_l)  = {(l,cx-1,cz)   | (l,cx,cz) in C_l}
N(C_l)  = {(l,cx,cz-1)   | (l,cx,cz) in C_l}
NW(C_l) = {(l,cx-1,cz-1) | (l,cx,cz) in C_l}
R_l     = C_l union W(C_l) union N(C_l) union NW(C_l)
C_l+1   = {parent(c) | c in C_l}
```

`R_l` is intersected with all published, inherited, or transient chunks that
exist in the transaction's coverage. A west chunk consumes the changed chunk's
first column as its east apron; a north chunk consumes its first row; a
northwest chunk consumes its north-west sample as the southeast apron corner.
At a coverage edge, an absent runtime chunk is not invented, but any inherited
published neighbor is replaced rather than left with a stale apron.

The planner must materialize and record all of these sets:

| Rung | Changed cores | Files replaced/repacked | Parent-core propagation |
|---:|---|---|---|
| `-2` | `C_-2` from the accepted master | `R_-2` | `C_-1 = parent(C_-2)` |
| `-1` | `C_-1` | `R_-1` | `C_0 = parent(C_-1)` |
| `0` | `C_0` | `R_0` | `C_1 = parent(C_0)` |
| `1` | `C_1` | `R_1` | `C_2 = parent(C_1)` |
| `2` | `C_2` | `R_2` | `C_3 = parent(C_2)` |
| `3` | `C_3` | `R_3` | `C_4 = parent(C_3)` |
| `4` | `C_4` | `R_4` | none |

For every whole-file rederive at rung `l+1`, record the full 25-key `D(q)` read
set and decoded artifact hashes. Apron-only repacks do not create new changed
cores at the next rung. The release transaction fails closed if any `R_l`,
`D(q)`, corrected parent, or dependency hash is absent.

### 1.4 Quantization and qoffset

Use `encode_quant16_checked` and its exact float32 wire qscale. For source values
`h`, decoded float32 values `d`, and

```text
s = float(float32(requested_qscale))
epsilon32 = 2 * abs(spacing(float32(max(abs(d)))))
```

every artifact must satisfy:

```text
max(abs(float64(d) - float64(h))) <= s/2 + epsilon32
```

No nonfinite sample, u16 clipping, or overflow is allowed. The gate is evaluated
inline before writing and independently after reading the file. The existing
qscales are `0.002 m` at LOD `-2`, `0.005 m` at `-1`, `0.01 m` at `0..2`,
`0.25 m` at `3`, and `1 m` at `4` unless a separately reviewed encoded-quality
trial changes them.

The current shared fine qoffset is permitted only for the bounded one-parent
proof: the 16 published LOD `-2` children plus nine east/south/southeast support
chunks. The current implementation reads the corresponding `640 x 640` LOD0
authority domain and selects
`qoffset_fine = float(floor(min(domain_authority) - 2.0))`; the integer is exactly
representable as float32 at Estonia elevations, and the checked encoder must
still prove every generated value fits. At `0.002 m`, a u16 domain spans only
`131.07 m`; therefore this policy is not a national policy. National publication
is blocked until a separate review approves deterministic qoffset-domain
ownership, boundary placement, transition handling, and decoded value/normal
tolerances. Per-chunk implicit qoffsets are not an approved substitute.

### 1.5 Decoded seam gates

The verifier operates on decoded float32 artifacts at every changed rung.

1. **Value identity:** for east neighbors `A,B`, require
   `A[:,2048] == B[:,0]`; for north/south neighbors `A,B`, require
   `A[2048,:] == B[0,:]`; require the four-way corner identity. In the bounded
   shared-qoffset proof these are bit-exact, so tolerance is zero.
2. **Gradient identity:** assemble the same `3 x 3` decoded world-sample
   neighborhood independently from each side of every seam sample. Reproduce
   the runtime central stencil from `TerrainField.cdTaps`:
   `dx=h_left-h_right`, `dz=h_north-h_south`, and
   `g=(dx/(2*t), dz/(2*t))`. Require the two independently assembled gradients
   to be bit-identical for the shared-qoffset proof.
3. **Normal identity:** reproduce `TerrainField.normalSlope4` as
   `normalize(dx,2*t,dz)`. Require zero angular difference for the shared-qoffset
   proof and record maximum angle in degrees.
4. **Bilinear continuity:** build the two decoded dependency assemblies
   independently, then evaluate `planeBilerp` across the two-texel strip centered
   on the shared sample at 17 positions `u=-1+k/8`, `k=0..16`. Require identical
   values at every world coordinate.

Do not incorrectly require the west and east one-sided slopes of a curved
surface to be equal. The gate compares the same world stencil reconstructed from
both dependency paths; natural curvature remains legal. A future multi-qoffset
policy must derive its nonzero value, gradient, and angular bounds from its two
half-step errors and then pass the visual gate; it may not inherit zero-tolerance
claims from the local proof.

## 2. H7: Bound And Executable Visual Acceptance

### 2.1 Reference identity and permitted use

The mandatory visual reference is:

```text
path: reference/suur-taevaskoda1.jpg
dimensions: 1920 x 1278 pixels
sha256: f031ef585dd82779c8e3c8a8059d15aacce11d2e6c886364bee3eb7c2eac382e
```

Any hash or dimension mismatch aborts review. The image is a morphology and
composition reference, not registered elevation truth: it has no usable camera
pose or ground-control metadata. Its vertical and undercut sandstone walls are
outside a single-valued heightfield's representational scope. The packed-height
review may claim the representable top surface, bank, floodplain, shore, and
submerged-bed behavior; it may not pass itself by pretending to reproduce the
wall topology.

### 2.2 Frozen review manifest

Before blind-test inference, create and hash one `visual-review-v1.json` that
contains:

- corrected-only, previous accepted owner, and finalist immutable manifest
  hashes; exact asset-server URLs; source/evidence snapshot hash;
- the reference path, dimensions, and hash above;
- geographically disjoint blind sites and their regime labels;
- every absolute camera pose, 30 Hz path sample, FOV, viewport, time-of-day,
  render flag, expected layer, and capture checksum;
- reviewer roster hash, review ID, trial order seed, questions, and statistical
  rule from Section 2.5.

The finalist may be rendered only once on the blind sites. A candidate changed
after inspection receives a new identity and a new untouched blind set.

### 2.3 Exact Estonia capture state

Every beauty capture uses the real production path:

```text
scene=world
src=estonia
seed=1
freeze=1
mschart=0
hud=0
dpr=1
shadowclipres=896
FOV=55 degrees
viewport=1920 x 1080 CSS pixels, deviceScaleFactor=1
primary T=12.5; light checks T=8.5 and T=17.5
```

The URL must contain the exact candidate `dataurl`. Do not pass flags that
disable or replace terrain, materials, water, trees, plants, understory, debris,
or grass. Boot each immutable manifest in a fresh real-WebGPU Chromium context,
capture all page/console/WebGPU diagnostics, wait through cloud bake and ready,
settle 48 frames, and align `stats.frame % 1024 == 0` before a still. Any boot
diagnostic prohibited by root `AGENTS.md` invalidates that candidate's panel.

Capture a second geometry diagnostic at the same pose with
`nanite=1&nanitedbg=cluster`. It is evidence only; the beauty image remains the
acceptance view. Full-production vegetation and materials must also be shown in
the beauty capture.

### 2.4 Fixed Taevaskoda cameras and paths

Let `O=(x0,z0)=(311123.082,190723.435)`. From the corrected-only control, define
once and freeze:

```text
G(x,z) = max(control_heightAt(x,z), control_waterAt(x,z) + 0.05)
L = (x0, G(x0,z0)+0.50, z0)
yaw(camera,L) = atan2(-(L.x-camera.x), -(L.z-camera.z))
pitch(camera,L) = atan2(L.y-camera.y, hypot(L.x-camera.x,L.z-camera.z))
```

Resolve the following to absolute poses from the control and store those numbers
in `visual-review-v1.json`; every candidate uses the identical absolute poses.

| ID | `(x,z)` | `y` | Aim |
|---|---|---|---|
| `S0-establish` | `(x0+96,z0+96)` | `G+40.0` | `L` |
| `S1-near-front` | `(x0+24,z0+28)` | `G+1.70` | `L` |
| `S2-near-west` | `(x0-28,z0+18)` | `G+1.70` | `L` |
| `S3-near-east` | `(x0+34,z0-16)` | `G+1.70` | `L` |
| `S4-grazing` | `(x0-18,z0-32)` | `G+1.70` | `L` |

The mandatory lossless 30 Hz paths are:

- `P1-lateral`: 361 inclusive linear samples over 12 s from
  `(x0-48,z0+24)` to `(x0+48,z0+24)`, control-ground-relative `y=G+1.70`,
  aimed at `L` each sample.
- `P2-dolly`: 481 inclusive linear samples over 16 s from
  `(x0+10,z0+16)` to `(x0+10,z0+144)`, control-ground-relative `y=G+2.00`,
  aimed at `L` each sample. This is the real terrain-DAG/availability survival
  check.
- `P3-orbit`: 361 inclusive samples over 12 s at radius `30 m`, angle `0..pi`,
  control-ground-relative `y=G+1.70`, aimed at `L` each sample.
- `P4-light`: hold the absolute `S1` pose for 301 samples over 10 s and call
  `setTimeOfDay(8.5 + 9*i/300)` before sample `i`.

Record lossless PNG frames or FFV1, not temporally lossy review video. The same
suite is generated for every held-out site by translating this local camera rig
to its frozen site anchor. Taevaskoda is mandatory but not sufficient; the blind
set must also contain ordinary sites and transition/water sites for the claimed
regime.

### 2.5 Blind-review procedure and statistic

1. Candidate authors choose one finalist using development sites only.
2. Use nine independent reviewers who did not author a candidate and have not
   seen the blind outputs. The user is a separate final acceptance authority and
   is not counted among the nine.
3. For each control independently, show synchronized finalist/control beauty,
   cluster, and motion panels at native size. Do not expose filenames, methods,
   manifests, or control identity.
4. Determine left/right per panel by the low bit of
   `SHA256(review_id || reviewer_id || control_id || panel_id)` and store the
   generated order before display. Randomize the order of the two control blocks
   by the next bit.
5. Reviewers may replay but may not zoom, scrub frame-by-frame before their first
   answer, or confer. Record one final `finalist`, `control`, or `tie` preference
   per control after all fixed panels. Record the hard-gate answers separately.
6. A tie is a non-win. The finalist must receive at least eight of nine reviewer
   wins against corrected-only and at least eight of nine against the previous
   accepted owner.

For each comparison, `8/9` gives a one-sided Clopper-Pearson lower bound of
`0.5175034851` at `alpha=0.025`, above chance. The two primary comparisons use
Bonferroni/Holm family-wise `alpha=0.05`; equivalently the exact one-sided
binomial tail for `>=8/9` is `0.01953125`, adjusted to `0.0390625`. Camera frames
are correlated evidence and must not be miscounted as independent samples.
Explicit user acceptance is mandatory and may reject any statistical pass.

### 2.6 Hard visible-geometry failures

Each reviewer answers every item `pass/fail` after seeing the fixed beauty,
cluster, still-light, and motion panels. Every item requires at least `8/9`
reviewer passes, and any objective invariant failure below fails immediately.

- **Clearly visible:** added detail is plainly visible at native size in the
  ground-level views without zoom. “Technically nonzero” or barely perceptible
  detail fails.
- **Geometric:** coherent forms survive the cluster view, camera translation,
  silhouettes, terrain contacts, and parallax. Detail visible only through a
  material, analytic normal, screen-space effect, or static lighting fails.
- **Retained by the real DAG:** the same named forms remain identifiable through
  `P2` after settled streaming/LOD transitions; a fine packed field simplified
  away by cluster construction fails.
- **Moving-light response:** shadows and shading rotate over fixed world-space
  forms during `P4`; forms may not crawl with the light or disappear when the
  light direction changes.
- **Physical character and variance:** the result contains connected,
  scale-appropriate morphology for the claimed soil/substrate/process regime,
  not random bumpiness, generic roughness, repeated atoms, sine waves, quilt
  phases, or one-meter-grid remnants.
- **Water is not terrain noise:** visible water remains the `waterY` surface.
  Static rough relief in water, relief inferred from optical water appearance,
  or a shore/bed solution incompatible with `watercover` fails.
- **Whole-surface agreement:** terrain, water, materials, trees/plants, grass,
  collision/probes, and contact shadows agree. Floating flat grass, vegetation
  islands, missing inherited layers, or mismatched shore grounding fails.
- **No storage boundary:** no chunk, apron, coverage, regime, qoffset-domain, or
  LOD boundary is visible in a still or any motion path.

Objective immediate failures are: missing layer, failed WebGPU boot, decoded
seam/normal failure, parent-closure failure, visible static water displacement,
candidate detail absent in `nanitedbg=cluster`, or a reproducible pop/crack along
`P2`. Metrics and reviewer preference cannot waive them.

## 3. H8: Exact Storage, Entropy, Compute, And Serving Accounting

### 3.1 Exact raw formulas

For every height chunk:

```text
samples             = 2049^2 = 4,198,401
u16 payload bytes   = 2 * 4,198,401 = 8,396,802
LAC header bytes    = 56
raw file bytes      = 8,396,858
core-only bytes     = 2 * 2048^2 = 8,388,608
apron bytes         = 2 * (2049^2 - 2048^2) = 8,194
```

The apron is 4,097 duplicated samples, `0.0975848%` of stored samples. A v2 index
record is exactly `struct <biiIQ> = 21 bytes` per indexed chunk. Serialized
manifest, content-store object, filesystem, database, and CDN metadata overhead
must be measured; none is included in the 56-byte header.

One parent-closed publication is 16 LOD `-2` files plus one LOD `-1` file:

```text
17 * 8,396,858 = 142,746,586 bytes = 136.13 MiB raw
index increment = 17 * 21 = 357 bytes
published core area = 16 * 128^2 = 262,144 m2 = 0.262144 km2
```

For a rectangular `P_x by P_z` LOD `-1` publication:

```text
N_-1 = P_x * P_z
N_-2 = 16 * P_x * P_z
N_negative = 17 * P_x * P_z
raw_chunk_bytes = N_negative * 8,396,858
index_bytes = N_negative * 21
outer LOD -2 hierarchy support = 4*P_x + 4*P_z + 1 chunks
```

The last formula counts only the unique external east column, south row, and
southeast corner when internal support is already published. Model halos,
overlap windows, corrected LOD `-1` support for LOD0, and higher-rung read closure
are additional and must be reported as unique sets per rung.

### 3.2 Audited exact coverage cases

| Case | LOD `-2` | LOD `-1` | Published chunks | Raw chunk files | v2 index increment |
|---|---:|---:|---:|---:|---:|
| One parent | `16` | `1` | `17` | `142,746,586 B` | `357 B` |
| `16.384 km` square (`32 x 32` parents) | `16,384` | `1,024` | `17,408` | `146,172,504,064 B` (`146.17 GB`) | `365,568 B` |
| Current country rectangle (`744 x 520` parents) | `6,190,080` | `386,880` | `6,576,960` | `55,225,799,191,680 B` (`55.23 TB`) | `138,116,160 B` |

The current rectangular AOI is `380,928 m x 266,240 m = 101,418.27072 km2`,
not Estonia's land area. Its fine grid is `2,976 x 2,080` LOD `-2` chunks and
`744 x 520` LOD `-1` chunks. The aligned pilot needs 257 external LOD `-2`
edge-support chunks if no adjacent fine coverage exists; the country rectangle
needs 5,057. These are transient cook inputs, not published objects.

An idealized `45,000 km2` land mask with no chunk closure, apron duplication,
headers, or indexes gives only a lower-order sample bound:

```text
LOD -2: 45e9 / 0.0625^2 = 11.52e12 samples = 23.04 TB u16
LOD -1: 45e9 / 0.25^2   = 0.72e12 samples = 1.44 TB u16
total ideal core-only bound = 24.48 TB
```

Do not mix this ideal land bound with the current rectangular release count.
Parent-closed land masking could reduce the rectangle, but its exact mask,
coast/water policy, islands, closure, and indexes must be planned first.

### 3.3 Rejected-LUKE diagnostic, not a forecast

The locally materialized rejected LUKE/quilt field measured:

- 16 published LOD `-2` files: `44,012,099 bytes`;
- one published LOD `-1` file: `3,169,314 bytes`;
- published total: `47,181,413 bytes`;
- nine transient LOD `-2` support files: `23,708,353 bytes`.

The `47,181,413 / 142,746,586 = 0.3305256842` ratio diagnoses that rejected
smooth field only. Blindly scaling it yields `48,313,766,912 bytes` for the
aligned pilot and `18,253,545,061,440 bytes` for the country rectangle; neither
is an accepted-quality estimate. Rich morphology may have materially higher
entropy. The old approximately `8.3 TB` country figure is unsafe because it
combined an idealized land-area sample count with a smooth rejected-output
compression assumption, omitted the actual 101,418 km2 rectangular closure,
and did not fully expose aprons, indexes, transient/scratch, replication, and
serving.

### 3.4 Mandatory percentile definitions

For sorted observations `x[1..n]`, report nearest-rank
`Q_p = x[ceil(p*n)]` for `p in {0.50,0.95,0.99}`, plus `n`, arithmetic mean,
minimum, and maximum. Do not silently use interpolated quantiles.

For packed output, an observation is one actual immutable chunk file including
its 56-byte header. Report both file bytes and
`8*(file_bytes-56)/4,198,401` compressed payload bits/sample. Stratify by LOD and
the exact regime/transition-mixture signature recorded by the recipe; also
report the all-chunk distribution. A p99 is labeled `insufficient-n` for
`n < 1000`. Before national approval, each material regime needs at least 1,000
accepted-quality chunk observations or all available chunks if its entire
approved coverage is smaller. Capacity totals use exact sums or the mean times
an explicit count, never p50 times count.

For compute, bin published cores into anchor-aligned `1 km x 1 km` accounting
cells. Apportion overlapping window work by output-core area, sum all duplicated
work into each cell, and report p50/p95/p99 wall seconds, CPU core-seconds, GPU
seconds, source bytes read, scratch bytes written, and output bytes per km2 by
regime. Training is reported separately and is never hidden by amortizing it
over a hypothetical national area.

### 3.5 Required cost ledger

Every bakeoff candidate and accepted release must emit a machine-readable ledger
with these exact categories:

| Category | Mandatory measurements |
|---|---|
| Identity | recipe, code, source, target, condition, checkpoint/model, environment, quantization, codec, and coverage hashes |
| Coverage | ideal land area; requested polygon; chunk-snapped rectangle/set; core km2; land/water km2; published chunks by rung; inherited/replaced chunks; model, hierarchy, and publication support sets |
| Training | GPU/accelerator type and count; GPU-hours; wall hours; CPU core-hours; peak host/GPU memory; dataset read bytes; checkpoints and retained checkpoint bytes; failed/retried runs |
| Inference | p50/p95/p99 and total GPU/CPU/wall time per km2 and regime; windows; batch; sampler steps; evaluated area; accepted output area |
| Support/overlap | `M_model = evaluated model area / unique output-core area`; unique halo area; overlap visits/sample; hierarchy-support chunks by rung; total duplicated compute and IO |
| Cook phases | source correction, condition reads, synthesis, fusion, composition, quantize/decode, hierarchy, compression, verification, index/manifest, and upload measured separately |
| Reliability | checkpoint interval; retry count and reason; duplicate work after retry; failed/abstained area; partial artifacts discarded; resume wall time |
| Local storage | source cache; targets; model/checkpoints; canonical masters; decoded memmaps; transient hierarchy support; compression staging; verification copies; peak scratch high-water; final unique objects |
| Packed entropy | p50/p95/p99/mean/max file bytes and bits/sample by LOD and regime; exact total object bytes; chunk count; 21-byte index total; manifest bytes |
| Retention | number of immutable builds; content dedup saved bytes; non-deduplicated bytes; backup/replication factor; replicated physical bytes; retention and garbage-collection cadence |
| Network | source download; upload; replication transfer; CDN origin fill; cache fill; cold/warm runtime egress; request count; failed/retried transfer bytes |
| Recook | expected source/model cadence; affected-area fraction by update class; partial and full recook GPU/CPU hours, scratch high-water, new unique bytes, upload, invalidated parents/aprons, and retained-old-build bytes |

Serving measurements use the exact existing demand logic and immutable release.
For fixed cold-cache boot, `P1`, `P2`, `P3`, and a preregistered 10 km traversal,
report p50/p95/p99 over independent sessions for: bytes/session, requests/session,
bytes per 10-second window, request latency, decode latency, and cache-hit ratio.
Use at least 100 sessions for p95 and 1,000 for a reported p99; otherwise label
the percentile `insufficient-n`. Separate origin egress, CDN-to-client egress,
and browser cache hits. Record bytes by layer and physical LOD so coarse/base
traffic cannot hide negative-rung demand.

National production remains blocked until the user explicitly accepts the
chosen coverage geometry, measured accepted-quality p50/p95/p99 entropy, total
unique and replicated bytes, cook/recook cost, and cold/warm serving ledger.
Quality is evaluated first; cost may break ties between quality-equivalent
survivors but may not justify visibly worse terrain.
