# Taevaskoda ALS Water Qualification

**Status:** final evidence audit
**Date:** 2026-07-13
**Decision:** **allow a bounded, cook-side Ahja water-surface and shoreline-contact repair from the 2019 class-9 ALS observations; abstain from ALS-derived bathymetry, unsupported cliff-bank reconstruction, multi-epoch averaging, and literal raw-DTM-only profiling.**

## 1. Question and answer

This audit asks whether the eight retained official ALS epochs over tile `444679`
contain qualified evidence for repairing the false terrain bridge in the mapped
Ahja river near `E 679788.35, N 6444811.45`, and whether the flowing-water gate in
the Stage 1 method contract can be satisfied.

The answer is deliberately split by evidence type:

1. **The raw DTM-only replay does not qualify.** Its coherent interpolated bridge
   survives the `11 m` Hampel test. At the anomaly station the accepted DTM Huber
   value is `45.5919 m`, while the independently observed 2019 ALS water profile
   is `37.5852 m`. Subsequent isotonic regression spreads part of the error rather
   than identifying it.
2. **The 2019 class-9 ALS water observations do qualify a water-surface profile.**
   They support `485/493` one-metre stations, have a longest missing span of only
   `3 m`, provide an unambiguous downstream orientation, agree with both observed
   banks wherever both banks are sampled, and are independently corroborated by
   several other epochs around the anomaly.
3. **Class 2 is not water-height authority.** Its semantics change between
   campaigns, it is absent inside the mapped river in most later epochs, and it
   contains cliff/TIN heights up to `59.57 m` in the anomaly neighbourhood.
4. **The cliff-side bank remains partially unobserved.** At the anomaly, 2019
   supplies a low opposite-bank contact but no class-2 contact on the cliff side
   within `3 m` of the ETAK shore. Sparse returns at roughly `63-64 m` farther up
   that side establish a steep protected feature, not an interpolatable bank
   face. Water-surface repair is allowed there; bank-shape reconstruction
   abstains.
5. **No epoch supplies bathymetry.** The only permitted submerged surface is the
   specification's confidence-zero `0.10 m`
   `unknown_bathymetry_render_envelope`.

This is a general source-qualification decision, not a coordinate patch. The same
rules apply to any mapped flowing reach with equivalent classified support.

## 2. Frozen inputs

### 2.1 Manifests and software

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| retained ALS manifest | 6,354 | `255efbc76461303f8d68eee1707ecca8f8c2a0b7a7f15d8ce6dc2f6da1a75097` |
| deterministic ALS inventory | 55,263 | `5304331ae4f10d1b108ed7bd8ff76076d25b61f6f450f5a0820e245e28bcc8fc` |
| `ETAK_EESTI_GPKG.gpkg` | 3,588,767,744 | `e19084c4bff0a6096a7dd38cefa12402bdce98d59c73d5b35c30dda60682f5f9` |
| `54472_dtm_1m.tif` | 81,545,553 | `a70ddfeeec08f278201c6479205ca661e247b1ec6b0151597390aa1c726d5239` |
| Stage 1 method contract used for the replay | - | `e5070c5d9819251425669ed87bf6d1e953020f36e16cf12edae9ae92ca59bb52` |
| `asset-gen/uv.lock` | - | `b95dc19be540e7c52a39b822f321c238a3181933b25ac32cee34850ca440ff64` |

The retained manifest declares `morphologyTarget=false`; this report preserves
that boundary. The locked analysis environment used Python through `uv run` with
`laspy 2.7.0`, `lazrs 0.8.1`, `Shapely 2.1.2`, `pyogrio 0.13.0`,
`rasterio 1.4.4`, and `SciPy 1.17.1`.

### 2.2 ALS files

| Year/type | Points | Bytes | SHA-256 |
|---|---:|---:|---|
| 2011 `tava` | 708,508 | 3,756,671 | `73cd7645b44b53f014c7f334abe6365fd8de6f98256f8d6bc90e066c8cf053b8` |
| 2015 `tava` | 616,006 | 3,419,408 | `7bfa5a6454e4c08bb39357c891175c8a7c4e077126659a8306632d2839fbfc88` |
| 2016 `madal` | 3,396,568 | 16,120,145 | `9f5dc05e615087f91428d604f027cc608e0f9aaf3bbffaf80dbbe4316649832f` |
| 2017 `mets` | 1,477,035 | 12,242,358 | `59a5c1f79ac485a60dfcb2f1c2ac884b0c6cd5a8dbcc33ee96344837bd31d442` |
| 2019 `tava` | 4,546,012 | 41,979,068 | `d927ac75d56ac66c90eb17cf7d8bbcd382da641b56feaefdd94bc6cdb66c3fcd` |
| 2021 `mets` | 1,591,686 | 20,618,192 | `7997b7f1fbca2367ef7bd6627fe568cc4d24cc4dd872ed5c23f24c4138f220e3` |
| 2023 `tava` | 5,234,018 | 68,066,160 | `9c50c123f14841c717d0d123d2d08061b87a51baf6d806ef269fabf1036a9fb7` |
| 2024 `mets` | 1,871,839 | 24,243,549 | `cfc156b246933c079221c2ec16d29ed7634a14e3b770a035eb7eedf719b49770` |

All eight files use centimetre `X/Y/Z` scales and EPSG:3301 plus EVRF2007/EH2000
height semantics. The inventory and retained hashes above bind the complete LAS
headers, classification counts, point-source identifiers, flags, and source HTTP
provenance.

## 3. Exact reach geometry

The selected ETAK area is `etak_id=1962675`, `kkr_kood=VEE1047200`,
`nimetus="Ahja jõgi"`. Its no-SRID WKB SHA-256 is
`d6b85bc27ed012480963218310ea07e8a32b5b06a64298415f7b094e143b9dca`.
The selected principal centreline is `etak_id=2356024`, with the same KKR code and
name; its no-SRID WKB SHA-256 is
`d4d4e2d059cbc88d4268fb67135bb29e4aab50cc6d0ed26b42e79dd38f4bdbff`.
The centreline's stored Z coordinates were ignored as height evidence.

Within retained tile bounds `[679000, 6444000, 680000, 6445000)`:

- the clipped Ahja polygon area is `9,737.679078682082 m2`;
- the clipped centreline is one unbranched `LineString`, length
  `492.827633307022 m`;
- station zero is the north tile intersection at
  `E 679709.6561895024, N 6445000.0`;
- the last clipped point is `E 680000.0, N 6444847.505658351`;
- the anomaly projects to station `272.2340975329349 m`;
- the anomaly is `5.0224004505 m` from the centreline and
  `9.1125033013 m` inside the mapped water boundary.

The polygon is continuous and the principal line does not branch inside this
tile. The unrelated ditch centreline returned by the window query was excluded by
its null KKR code and `tyyp=Kraav`.

## 4. Reproduction method

Each LAZ was decoded sequentially with `laspy.open(...).chunk_iterator(1_000_000)`.
No rasterization or point thinning was used. Point-in-polygon tests used the exact
clipped ETAK polygon. A point exactly on the mathematical polygon boundary is not
counted as an interior observation; no conclusion depends on this zero-area case.

The flowing-profile replay follows the reviewed contract as closely as the ALS
source permits:

1. project every in-polygon point onto the clipped principal centreline;
2. create stations `0..492 m` inclusive;
3. at each station collect points with along-line distance at most `1.5 m`, giving
   the contract's `3 m` station interval;
4. require at least four points;
5. compute the contract's deterministic Huber location: median initialization,
   scale `max(1.4826*MAD, 0.02 m)`, cutoff `1.345`, tolerance `1e-8 m`, at most
   50 iterations;
6. apply an `11`-station Hampel window, truncated only at the two evidence edges,
   and reject beyond `3*1.4826*MAD`;
7. measure the longest consecutive unsupported-or-rejected station run;
8. compute Huber locations over the first and last `21 m`, compare their
   difference with `max(0.05 m, 2*max(endpoint raw MAD))`;
9. for the accepted candidate, linearly fill only the short accepted gaps, apply
   deterministic non-increasing pool-adjacent-violators, then a positive Gaussian
   with `sigma=4 m`, truncation `4 sigma`, and nearest endpoint extension.

This is a **qualification replay**, not permission to silently relabel the
contract's DTM observations as ALS. Section 8 states the required source-hierarchy
clarification.

For bank plausibility, class-2 points were selected outside the ETAK polygon but
within `3 m` of it, projected to the local centreline normal, split by side, and
aggregated by the same `3 m` along-line station interval. Both Huber height and the
per-bin tenth percentile were tested. These are plausibility checks, not a bank
surface solver.

## 5. Class support and station results

`C9 tile` is the file's complete class-9 count. `C9 river` and `C2 river` are exact
ETAK-interior counts. Density is over the `9,737.6791 m2` clipped polygon. `Raw`
means stations meeting the four-point rule before Hampel; `accepted` is after
Hampel.

| Epoch | C2 river | C9 tile | C9 river | C9 captured by polygon | C9 density `/m2` | Raw / 493 | Accepted / 493 | Longest gap | Profile decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2011 | 331 | 0 | 0 | n/a | 0 | 0 | 0 | n/a | reject: no class 9 |
| 2015 | 15 | 558 | 437 | 78.32% | 0.0449 | 122 | 121 | 74 m | reject |
| 2016 | 411 | 5,415 | 4,979 | 91.95% | 0.5113 | 462 | 437 | 12 m | corroboration only |
| 2017 | 0 | 4,052 | 3,950 | 97.48% | 0.4056 | 478 | 465 | 4 m | corroboration only |
| **2019** | **0** | **17,160** | **16,846** | **98.17%** | **1.7300** | **492** | **485** | **3 m** | **accept as authority** |
| 2021 | 1 | 2,223 | 2,035 | 91.54% | 0.2090 | 334 | 321 | 37 m | reject |
| 2023 | 0 | 4,488 | 4,068 | 90.64% | 0.4178 | 407 | 401 | 25 m | reject |
| 2024 | 0 | 2,977 | 2,752 | 92.44% | 0.2826 | 398 | 385 | **20 m** | formal continuity pass; corroboration only |

The method contract rejects spans **longer than** `20 m`; therefore 2024 is not a
formal continuity failure. It is not selected as authority because only four
accepted stations support each `21 m` endpoint window and coherent high-label
contamination remains. Similarly, 2016 and 2017 pass the formal continuity rule
but are weaker authorities than 2019.

Raw class labels are not sufficient validity. In-polygon class-9 maxima are
`60.89, 54.11, 43.16, 53.05, 60.49, 60.75, 54.42 m` for 2015 through 2024.
These heights are impossible as one water sheet with the low returns near
`37-39 m`; they are rejected bank/cliff/registration contamination.

Class 2 is even less stable as water evidence. It contributes `331` in-river
points in 2011 and `411` in 2016, only `15` in 2015, one in 2021, and zero in the
other four epochs. In the `25 m` anomaly neighbourhood, 2016 class 2 reaches
`59.57 m`. Class 2 remains legal ground/bank evidence after geometric and
surface-semantic qualification, but is **rejected for water-height estimation**.

## 6. Accepted 2019 profile

The only station with fewer than four 2019 class-9 observations is station `64`.
Hampel rejects stations `[74, 75, 239, 240, 241, 254, 281]`. The remaining
`485` stations form a sequence with a maximum gap of `3 m`.

The first and last `21 m` Huber locations are respectively `38.125614 m` and
`37.410750 m`. Their difference is `0.714864 m`. Endpoint raw MADs are
`0.067146 m` and `0.002008 m`, so the orientation threshold is `0.134293 m`.
The line direction from the north tile boundary toward the east tile boundary is
therefore unambiguously downstream for this snapshot.

After non-increasing PAVA and the specified positive `sigma=4 m` convolution:

- output range: `38.418046 .. 37.406493 m`;
- total clipped-reach fall: `1.011553 m`, average `0.002052 m/m`;
- maximum one-metre fall after smoothing: `0.038276 m`;
- rises after smoothing: none;
- accepted-station residual to the final curve: median `-0.000293 m`, RMSE
  `0.055036 m`, 95th absolute `0.110901 m`, maximum absolute `0.470968 m`.

The maximum residual is why raw class-9 values must not be interpolated directly.
The isotonic/Gaussian contract is doing substantive robust work, not merely
formatting an already clean label stream.

At the anomaly station, 2019 supplies `174` class-9 observations in the `3 m`
interval. Their Huber location is `37.584122 m`; the accepted station value is
`37.584146 m`, and the final profile is `37.585156 m`.

## 7. Independent corroboration and disagreement

### 7.1 Exact anomaly station

| Epoch | Accepted semantic class | Points in 3 m interval | Huber height |
|---|---|---:|---:|
| 2011 | class 2, corroboration only | 1 | `37.4400 m` |
| 2015 | none | 0 | n/a |
| 2016 | class 9 | 37 | `37.9142 m` |
| 2017 | class 9 | 41 | `37.5547 m` |
| **2019** | **class 9 authority** | **174** | **`37.5841 m`** |
| 2021 | class 9, incomplete reach | 45 | `37.4650 m` |
| 2023 | class 9, incomplete reach | 15 | `37.5400 m` |
| 2024 | class 9 corroboration | 52 | `37.5457 m` |

These are separate hydrological snapshots and must not be averaged. Their value is
that six independent class-9 campaigns place water around `37.46-37.91 m`, while
the raw DTM puts the anomaly station around `45.6 m`.

Across all common accepted stations, after subtracting one robust epoch offset:

- 2016 minus 2019: offset `+0.3040 m`, residual MAD `0.0313 m`, 95th absolute
  residual `0.8916 m`;
- 2017 minus 2019: offset `-0.0298 m`, residual MAD `0.0195 m`, 95th absolute
  residual `0.4827 m`;
- 2024 minus 2019: offset `-0.0199 m`, residual MAD `0.0408 m`, 95th absolute
  residual `1.1536 m`.

The low core MADs corroborate the longitudinal shape. The large tails show
localized, spatially coherent classification or bank contamination and forbid
naive epoch pooling.

### 7.2 Flightline and overlap evidence

All `16,846` accepted-candidate 2019 in-river class-9 points have
`point_source_id=19317`; none is marked overlap. The second tile source
`19318` contributes no in-river class-9 point. Thus 2019 has no within-epoch
duplicate-flightline validation. Its authority depends on strong internal support
plus cross-epoch corroboration, which must be recorded as a limitation.

The 2023 river sample does contain two sources: `2,848` class-9 points from
`23341`, `1,220` from `23342`, and `1,940` overlap-flagged points. Where both
source-specific station profiles have at least four points, they share `171`
stations. The robust source offset is only `0.0230 m` and residual MAD is
`0.0777 m`, but the 95th absolute residual is `2.2625 m` and the maximum is
`2.6626 m`; the combined profile also has a `25 m` gap. The overlap therefore
exposes localized contamination rather than rescuing 2023 as profile authority.

## 8. Raw DTM gate failure

There are `9,735` valid DTM cell centres inside the clipped ETAK polygon. Their
height distribution is not a water surface: median `38.057 m`, 75th percentile
`38.621 m`, 95th `45.881 m`, 99th `53.625 m`, maximum `60.378 m`.

A literal replay of the same station algorithm over those DTM cells yields all
`493` raw stations, rejects only `16` with Hampel, and accepts `477`. The coherent
bridge at the anomaly is not rejected:

- anomaly-centred `3 m` DTM Huber: `45.774526 m`;
- station `272` accepted DTM value: `45.591937 m`;
- 2019 ALS final profile at the same station: `37.585156 m`;
- difference before any legitimate hydrological offset: about `8.01 m`.

PAVA cannot classify a coherent invalid plateau. It spreads the conflict and the
DTM-derived smoothed profile is `38.921408 m` at the anomaly, still `1.336252 m`
above the qualified 2019 profile.

Therefore the phrase **valid DTM samples** in the method contract requires an
explicit, evidence-bound interpolation-validity decision. For this reach, the
2019 class-9 sequence is the higher-priority direct observation. Either:

1. use it directly as the water-surface observation sequence, preserving all
   station gates; or
2. use it to mark contradictory mapped-water DTM cells as interpolation-invalid,
   then estimate only from the remaining DTM observations.

Option 1 is simpler and loses no justified information. Quietly running the
existing DTM through Hampel is forbidden by this result.

## 9. Shore and bank qualification

ETAK is accepted as plan geometry, not height truth. Its geometric alignment is
strongly corroborated in 2019: `98.17%` of every class-9 point in the 1 km tile
falls inside the clipped Ahja polygon.

The 2019 three-metre outer shore collar contains `4,019` class-2 bank candidates:
`1,587` on one centreline-normal side and `2,432` on the other. At least one side
has four or more points at `449/493` stations; both sides do at `277/493`.
Among every both-supported station:

- the final 2019 water profile is never more than `0.25 m` above both Huber banks;
- it is never more than `0.25 m` above either individual Huber bank;
- the same zero-violation result holds against each side's tenth percentile.

This passes the contract's **inconsistency** gate where both banks are observed.
It does not turn missing banks into observations.

At the anomaly, the low side has `26` class-2 points within `3 m`; its Huber bank
height is `37.676835 m` and tenth percentile is `37.580 m`, consistent with the
`37.585156 m` water profile. The cliff side has no 2019 class-2 return within
`8 m`. In 2016, 2021, and 2023, sparse class-2 returns appear roughly `5-8 m`
from that shore at about `63-64 m`. Those observations corroborate the protected
steep cliff/top, not the missing wall or water-contact shape.

Required handling is consequently asymmetric but not scene-specific:

- preserve the mapped continuous shoreline and qualified water elevation;
- solve the sampled low bank from direct ground support;
- mark the unobserved steep side as protected/uncertain and do not interpolate a
  smooth bank through the missing vertical face;
- remember that a single-valued heightfield cannot reproduce that wall's true
  topology.

## 10. Executable authorization

### Allowed

1. Treat 2019 class 9 from source `19317` as the single-snapshot water-surface
   authority over the clipped Ahja reach after the exact station, Huber, Hampel,
   PAVA, and Gaussian operations recorded above.
2. Treat 2016, 2017, and 2024 class 9 as corroboration only. They may detect a
   gross 2019 failure but may not be averaged into the 2019 height.
3. Use the frozen ETAK area and principal centreline for water plan geometry,
   exact watercover, station coordinates, and cross-sectional extension.
4. Mark the mapped-water DTM bridge as `interpolation_invalid` where it conflicts
   with the qualified direct water observations. This explicitly permits
   `base_corrected != raw_dtm`.
5. Join dry structural terrain to the accepted shoreline using supported bank
   observations and uncertainty/protected-feature masks.
6. Use only the confidence-zero `0.10 m`
   `unknown_bathymetry_render_envelope` under water.

### Rejected or abstained

1. Raw DTM-only flowing-water profiling on this reach.
2. Class 2 as water-surface height, including its apparently low 2011 return.
3. 2011 or 2015 as water-profile authority; 2021 or 2023 as authority because
   their gaps exceed `20 m`.
4. 2016, 2017, or 2024 as the selected authority while the stronger 2019 snapshot
   exists.
5. Any mean, median, or fitted surface pooled across years; the real water stage
   changes between campaigns.
6. Any raw class-9 interpolation without robust station rejection.
7. ALS-derived bed depth, thalweg, sediment form, cliff wall, overhang, or
   centimetre morphology.
8. Smooth interpolation through the missing cliff-side bank observations.
9. Extrapolation beyond the retained tile as if this audit had observed the
   neighbouring reaches.

## 11. Final gate statement

**The Ahja repair is authorized, with a source-hierarchy correction.** The
class-9-based flowing profile passes support, continuity, orientation,
monotonicity, smoothing, and observed-bank plausibility for the retained reach.
The known anomaly is especially well supported and lies far from the evidence
edges.

**The literal DTM-only formulation does not pass scientific qualification.** It
passes its own numerical continuity test while retaining a false coherent bridge,
which demonstrates that continuity and Hampel rejection are necessary but not
sufficient. Implementation must record the 2019 class-9 observations as
higher-priority direct water evidence, or record the contradictory DTM cells as
interpolation-invalid before invoking the DTM estimator.

This authorization fixes water surface and shoreline contact only. It leaves the
unobserved cliff face and all bathymetry in explicit abstention, exactly as the
evidence requires.
