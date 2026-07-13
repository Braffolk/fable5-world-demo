# Ahja Hydraulic Topology Qualification

**Date:** 2026-07-13
**Scope:** ETAK centreline `etak_id=2356024`, its complete graph reach, the
Taevaskoda Stage 1 publication and repair-support domains, retained 2019 ALS,
local DTM, and the Jägala protected-discontinuity regression.
**Decision:** **the complete `2288.237233708688 m` Ahja feature is one
topologically unbranched, mapped-discontinuity-free confluence-to-confluence
reach. It needs no hydraulic split. It is not, however, authorized for one
full-line PAVA/Gaussian solve: the retained three-tile 2019 evidence has a
`31 m` unsupported span at stations `40..70`, inside Stage 1. The solver must
keep topology qualification separate from observation qualification.**

## 1. Executable decision

The following are the only defensible current states:

| property | decision |
|---|---|
| exact graph feature | `E_203_vooluveekogu_j.etak_id=2356024` |
| geometric station interval | `[0, 2288.237233708688] m` |
| unbranched interior | **true** |
| mapped owned level jump in the interior | **none found** |
| topologically discontinuity-free | **true** |
| observation mask | footbridge `E_505...j.etak_id=4477659` at `s=386.497376540326 m` |
| full-line 2019 evidence continuity | **false**, `31 m` unsupported at integer stations `40..70` |
| current full-line PAVA/Gaussian | **forbidden** |
| downstream evidence segment | integer stations `[71,2286]` pass the existing numerical gate; do not infer authority outside the explicit segment |
| Stage 1 publication interval on this line | `[35.719463803822, 675.343212781498] m` |
| Stage 1 repair-collar interval on this line | `[26.995569766588, 683.355805201327] m` |
| Stage 1 currently unsupported publication slice | `[35.719463803822,71) m` |

`discontinuity_free_reach=True` may be recorded in a **topology artifact** for
this exact ETAK feature. It must not be accepted as a caller-supplied promise
that bypasses the observation gate. If the current implementation has only one
boolean admission gate, it must remain false for the full solve until the two
facts are represented separately.

There is no scientific basis for declaring the `40..70 m` gap a waterfall or
for inventing a hydraulic split there. Six direct 2019 class-9 returns occur
inside that interval, at heights `38.18..38.44 m`, and the supported stations on
both sides are compatible with one continuous river surface. Six sparse points
do not satisfy the frozen four-points-per-three-metre-station and maximum-gap
contract. The correct current result is **missing evidence**, not a jump and not
permission to interpolate.

The short upstream candidate `[0,39] m` is also rejected for publication. It
passes the mechanical missing-span test but infers reverse flow and is strongly
contaminated: stations `31..33` have Huber locations `40.7491, 41.9190,
41.8059 m`, while nearby low returns are about `38.2..38.4 m`. Splitting at the
evidence gap and fitting that fragment would convert contamination into a false
hydraulic reach.

## 2. Frozen inputs and station convention

All vector results below come from the local whole-country
`ETAK_EESTI_GPKG.gpkg`, SHA-256
`e19084c4bff0a6096a7dd38cefa12402bdce98d59c73d5b35c30dda60682f5f9`,
extraction date `2026-07-04`. Geometry was forced to 2D for topology; stored ETAK
Z was inspected only as a discontinuity trigger, never treated as water-height
authority.

Station zero is the first coordinate of `2356024`,
`E 679525.7210000008, N 6445090.844999999`; the end is
`E 680780.8090000004, N 6444886.079`. The line is simple, has `158` vertices,
and is covered for its entire length by Ahja area polygon
`E_203_vooluveekogu_a.etak_id=1962675`.

The retained 2019 campaign is the ordered spatial union of:

| tile | file SHA-256 | decoded points | class 9 in full Ahja polygon | class 9 in reach qualification polygon |
|---|---|---:|---:|---:|
| `445679` | `02b98066adfe1247c0cc4c2ce59e7ec1711063f094e7a1cd29229e66d5cd6bf6` | 4,473,016 | 7,102 | 1,438 |
| `444679` | `d927ac75d56ac66c90eb17cf7d8bbcd382da641b56feaefdd94bc6cdb66c3fcd` | 4,546,012 | 16,846 | 16,846 |
| `444680` | `672a337dd9da433225cbc580cdda94e9b44ec0db9bdc662f212cc95fe95f6204` | 4,912,710 | 34,016 | 29,327 |

Class-9 overlap returns were excluded. The three retained files contribute
`47,611` eligible returns inside the bound reach qualification polygon. The
replay used exact line projection,
one-metre stations, a centred `3 m` interval, at least four samples, the frozen
Huber/Hampel rules, and did not pool another epoch.

### 2.1 Mandatory reach qualification polygon

The full ETAK Ahja polygon is **forbidden** as the point-to-station qualification
polygon. It extends well beyond both graph endpoints. Projecting every point in
it to the finite line clamps `5,664` off-reach northern returns to `s=0` and
`4,689` off-reach eastern returns to `s=L`. Those are not endpoint observations.

Construct the general cross-section-capped polygon deterministically:

1. force the source water polygon and selected simple centreline to 2D;
2. at each graph endpoint, use the unit tangent of the first, respectively last,
   non-zero centreline segment; its perpendicular through the endpoint is the
   endpoint cross-section line;
3. for current polygon bounds define
   `R=hypot(max_x-min_x,max_y-min_y)+1.0 m` and intersect the polygon with the
   cross-section segment from `endpoint-R*normal` to `endpoint+R*normal`; if the
   result has multiple chord components, select the unique chord whose distance
   to the endpoint is at most `tau=1e-6 m`;
4. extend only that local chord beyond both polygon-boundary ends by
   `epsilon=max(1e-6 m,1e-9*R)` along its unit direction and split the current
   polygon with it; do **not** intersect with an infinite half-plane, which can
   cut a later bend of a meandering reach;
5. after each endpoint split, retain the unique polygon component that covers
   the centreline midpoint;
6. require `centreline.difference(component).length == 0` and each endpoint's
   component distance to be no more than `tau`; reject every observation whose
   projected station is exactly `0` or `L`. This strict endpoint rejection is
   what prevents numerical boundary tolerance from reintroducing clamped
   returns.

For `2356024`, the selected local chords are:

- start: `LINESTRING (679512.4736295912 6445091.740588254,
  679540.3587340222 6445089.855416234)`, length
  `27.948755297494 m`;
- end: `LINESTRING (680793.7931623477 6444887.458375538,
  680772.2272830025 6444885.167319278)`, length
  `21.687233357898 m`.

The retained component has area `43,494.60172928659 m2`, bounds
`[679510.5520000011, 6444444.48, 680866.5869999975,
6445091.740588254]`, and its intersection with the selected centreline is
exactly `2288.237233708688 m`. Normalize the polygon, encode 2D little-endian WKB
without SRID, and bind SHA-256
`0e44e8389e17c87c46886248a6220c5d9d150329e159bc393e6f1a8d587ff458`
as the qualification-polygon identity. The capped replay has zero class-9 return
projecting exactly to either endpoint.

This construction is graph- and geometry-derived. It applies to any simple
unbranched reach and contains no Ahja coordinate or ID special case. A reach for
which either local chord or retained component is non-unique abstains rather
than choosing a polygon by area.

## 3. Exact water graph

The feature is already bounded by graph events; neither storage tiles nor the
Stage 1 rectangle define its hydraulic endpoints.

### Upstream node: `s=0`

- Ahja continuation `etak_id=2357065`, `kkr_kood=VEE1047200`, intersects at the
  exact endpoint.
- Unnamed `Oja`, `etak_id=2357066`, width class `1-2 m`, intersects at the same
  endpoint.

### Downstream node: `s=2288.237233708688`

- Ahja continuation `etak_id=7079045`, `kkr_kood=VEE1047200`, intersects at the
  exact endpoint.
- Unnamed `Kraav`, `etak_id=7072014`, width class `4-6 m`, intersects at the same
  endpoint.

No other flowing-water centreline intersects the open interval. No second
flowing-water polygon overlaps it: the entire feature is inside Ahja polygon
`1962675`. There is no geometry gap, self-intersection, island branch, secondary
axis, or internal confluence. Thus the smallest natural graph reach is the whole
feature, and its two confluences are the correct topology boundaries.

## 4. Structures and masks

### Owned level jumps

No `E_205_hudrotehniline_rajatis_j` feature occurs in the feature's 100 m search
window. In particular, no `Pais` intersects the reach. No
`E_206_truup_j` feature intersects it. The nearest culvert,
`etak_id=7065768`, is `57.726446277565 m` from the downstream endpoint; the
other queried culverts are farther away.

No vector evidence therefore owns a dam, weir, fall, or culvert level jump on
the open reach. An owned jump must split the profile; none is present here.

### Footbridge observation mask

`E_505_liikluskorralduslik_rajatis_j.etak_id=4477659`, type `Purre`, crosses the
centreline at:

- station `386.497376540326 m`;
- `E 679727.0729999989, N 6444890.802999999`;
- total mapped structure length `32.122526525821 m`;
- exact length inside the Ahja polygon `25.370307425590 m`.

The coincident path is `E_501_tee_j.etak_id=5050643`, type `Rada`. The bridge
deck can contaminate ALS labels, so its 2D structure footprint plus the declared
source-position uncertainty is an observation mask applied **before** station
aggregation. It is not a water-level jump and does not split PAVA. The 2019
class-9 medians around `s=376..397` remain near `37.94..37.78 m` with ample
support, so masking the crossing creates only a short fillable absence.

A mapped bridge area, `E_505_liikluskorralduslik_rajatis_ka.etak_id=1957310`,
is `8.669939233840 m` from the downstream endpoint. It belongs to the next graph
node/reach, not the open interval of `2356024`.

### Bank escarpments

Seven ETAK slope/escarpment lines lie within 30 m of the centreline. None
intersects the Ahja polygon, and none owns a transverse hydraulic jump:

| `E_102` ID | type | shoreline escarpment | nearest station m | centreline distance m |
|---:|---|---|---:|---:|
| 1829130 | `Looduslik järsak` | yes | 55.216277 | 13.745818 |
| 1829131 | `Looduslik järsak` | yes | 559.576198 | 11.880455 |
| 1829191 | `Nõlv` | yes | 795.186965 | 17.284508 |
| 1829192 | `Nõlv` | yes | 1240.485541 | 13.636392 |
| 1829193 | `Nõlv` | yes | 1425.631298 | 28.901485 |
| 1826743 | `Looduslik järsak` | yes | 1641.663144 | 15.596734 |
| 1826742 | `Nõlv` | yes | 2068.095192 | 16.031994 |

These are protected bank/cliff constraints. They forbid smoothing dry terrain
through the bank, but they do not split the water longitudinal profile.

## 5. DTM and direct-water changepoints

Nearest-cell sampling of the local 1 m DTM along the complete centreline ranges
from `36.384377` to `44.706875 m`; inside the Stage 1 repair collar it ranges
from `37.434998` to `44.706875 m`. A robust ten-metre-before/ten-metre-after
median comparison finds apparent steps of:

| station m | DTM median step m |
|---:|---:|
| 471 | -5.220 |
| 513 | +4.551 |
| 534 | -5.836 |
| 558 | -1.889 |
| 669 | -1.033 |

Those values are not five waterfalls. Direct 2019 class-9 medians remain
continuous: `37.63 m` at stations `450` and `470`, `37.61 m` at `500`,
`37.58 m` at `519` and `540`, `37.56 m` at `560`, and `37.46 m` at `669`.
The already qualified primary-tile profile has no rise after smoothing and a
maximum one-metre fall of only `0.038276 m`. The DTM changepoints are the same
interpolation/terrain contamination class as the known false river bridge and
must be marked interpolation-invalid, not converted to owned hydraulic jumps.

Across all three 2019 files, `2181/2289` integer stations have a raw Huber
observation. The full replay abstains with
`longest unsupported station span is 31.0 m`. The decisive run is `s=40..70`:
only six direct returns occur over that interval, though their `38.18..38.44 m`
heights support continuity qualitatively. Additional smaller raw gaps are at
`25..28` (`4 m`), `86..89` (`4 m`), `128..142` (`15 m`), `148..153`
(`6 m`), `157..159` (`3 m`), `166..168` (`3 m`), and downstream runs no longer
than `10 m`. The downstream integer segment `[71,2286]` passes the current gate
with forward orientation and a `15 m` maximum gap; applying the footbridge mask
does not change that maximum.

This produces a general evidence rule:

1. qualified direct water returns outrank interpolated DTM inside mapped water;
2. a DTM step contradicted by direct water is an interpolation defect;
3. a direct-water step supported on both sides is a possible owned jump and
   requires a split/feature audit;
4. an unsupported interval is abstention, not license to select whichever of
   smooth interpolation or raw DTM looks convenient.

## 6. Minimal Stage 1 solve

The Stage 1 rectangle intersects the line as follows:

| domain | continuous line interval m | length m |
|---|---:|---:|
| publication `[679424,679936] x [6444544,6445056]` | `[35.719463803822,675.343212781498]` | 639.623749 |
| 8 m repair collar `[679416,679944] x [6444536,6445064]` | `[26.995569766588,683.355805201327]` | 656.360235 |

The minimum currently defensible **published** profile is the intersection of
the qualified downstream evidence segment with these domains:

- publication: `[71,675.343212781498] m`;
- repair support: `[71,683.355805201327] m`.

The upstream publication slice `[35.719463803822,71) m` and repair-only slice
`[26.995569766588,35.719463803822)` remain unresolved. They may not be filled
from the reverse-oriented contaminated `[0,39]` fit, raw DTM, endpoint extension,
or a full-line isotonic pool. If Stage 1 release semantics require every mapped
water sample in the publication rectangle to be corrected, the release must
abstain as a whole until that slice has qualified evidence. If release semantics
permit an explicit no-change mask, the boundary must remain outside the rendered
wet surface or be independently proven invisible; it may not create a height
seam inside the river.

The full natural graph reach remains the preferred eventual solve domain because
its endpoints are real confluences. It becomes eligible only after the sparse
interval is resolved by a reviewed general evidence model or additional legally
usable observations. Merely raising `maximum_missing_span_m` from `20` to `31`
for this site is forbidden.

## 7. Jägala protected-discontinuity regression

### What ETAK actually contains

Jägala waterfall is **not** represented as a typed `Juga` or as an `E_205`
hydraulic structure. The national `E_205` vocabulary contains `Paadisild`,
`Pais`, and `Muul`; no `E_205` or `E_206` feature occurs in the inspected
waterfall window.

The relevant ETAK objects are:

- Jägala main centreline `E_203...j.etak_id=2268068`, length
  `427.209533005425 m`;
- Jägala area polygon `E_203...a.etak_id=8409382`, which covers the complete
  centreline;
- shoreline natural escarpment `E_102_nolv_j.etak_id=5348375`, which intersects
  the water polygon for `7.587617546171 m` and is `27.739875460825 m` from the
  centreline at station `105.233688530443 m`.

The centreline's stored Z drops from `26.958 m` at `s=83.904 m` to `19.819 m`
at `s=121.935 m`: `7.139 m` over `38.031 m`. The local national 10 m DTM
independently exposes the same discontinuity at its scale, from `26.732 m` near
`s=90 m` to `19.281 m` near `s=120 m`, a `7.451 m` drop over `30 m`. Stored
centreline Z and DTM are trigger evidence, not sufficient geometry for rebuilding
the waterfall face.

### General rule, not a Jägala exception

The national solver must perform a blind protected-discontinuity gate before
PAVA:

1. split unconditionally at graph confluences and at an intersecting typed owned
   jump such as `E_205` `Pais`;
2. mask bridge/culvert footprints as observation occlusion unless a separate
   structure record owns a level jump;
3. test robust upstream/downstream levels for every abrupt elevation candidate
   over a bounded physical window, using qualified direct water first and raster
   or stored-Z evidence only as a conservative trigger;
4. when a large trigger is corroborated, or direct evidence cannot disprove it,
   create two independent reach solves and an intervening
   `protected_discontinuity` abstention interval; no PAVA pool, linear gap fill,
   endpoint extension, or Gaussian kernel may cross that interval;
5. when qualified direct water continuously contradicts a DTM-only trigger, as
   on Ahja, invalidate the DTM interpolation rather than preserving a false jump.

The Jägala regression fixture must be selected from the national layers by these
feature/evidence predicates, never by a coordinate or `etak_id` allowlist. It
passes only if the `~7 m` drop is protected from cross-jump smoothing, the two
sides are solved independently or abstain, and the unknown waterfall geometry is
left untouched. It fails if the output is one monotone ramp, if the Gaussian
kernel spans the fall, or if the drop is moved or erased. This regression protects
waterfalls, dams, weirs, and comparable rapids generally while still allowing
direct evidence to remove false DTM bridges such as Ahja.

## 8. Final authorization

**Topology authorization:** record `2356024` as one unbranched,
mapped-discontinuity-free graph reach. Do not split it at the footbridge, bank
escarpments, tile boundaries, DTM spikes, or Stage 1 rectangle edges.

**Solver authorization:** do not fit or publish one profile over the full
`0..2288.237 m` line with the current evidence. The three-tile 2019 campaign
fails the frozen continuity gate at `40..70 m`. The downstream interval beginning
at station `71 m` is numerically eligible after the footbridge observation mask,
but it covers only the downstream part of Stage 1. The unresolved upstream
publication slice must remain explicit abstention unless a general, reviewed
evidence method closes it.

**National safety authorization:** no national flowing-water activation until
the blind Jägala fixture returns a protected discontinuity and demonstrates that
neither isotonic fitting nor Gaussian smoothing crosses it.
