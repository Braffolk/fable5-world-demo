# Stage 1 Executable Microtopography Method Contract

**Date:** 2026-07-13
**Status:** superseded by the final integrated audit and `docs/specs/terrain/MICROTOPOGRAPHY.md`; retained as a rejected-contract record, not implementation authority
**Historical intent:** attempted to close `FINAL-SPEC-CRITIQUE.md` C1-C3 and method-facing H1-H3; final audit rejected that closure
**Does not claim:** a national morphology method, Estonia peat validity, real bathymetry without survey, or a solution for non-single-valued cliff faces

> **Do not implement this contract.** Final adversarial review found that its Moore
> task conditions on unavailable true 0.25 m input, mislabels a packed-null delta
> as the `F4` analysis band, lacks transferable per-band target error and replicated
> real-vs-real strata, and cannot support an owner decision. The corrected main
> spec makes Moore an evidence/engineering screen with a known
> `target_evidence_insufficient` ownership result, separates M0/M1 and packed/F4
> operators, and makes Taevaskoda structural repair plus acquisition closure the
> cheapest Stage 1 proof. Where this file conflicts with the main spec, this file
> is historical only.

## 1. Decision And Scope

Stage 1 has two deliberately separate outputs.

1. **Taevaskoda structural proof:** deterministic repair from one immutable evidence snapshot. It publishes ordinary absolute height at LOD `-2`, a decoded-child-derived LOD `-1`, corrected ordinary LOD0, and every affected ancestor through LOD4. Its LOD `-2` surface contains resolved structure only. The unresolved morphology contribution is exactly absent.
2. **Moore peat research bakeoff:** a foreign, local, `0.25 m -> 0.0625 m` morphology experiment. It may compare specialist families only after the target band is qualified. It cannot activate an Estonia regime, supply long-range mire organization, or publish national geometry.

This contract does not choose a national specialist. It makes the cheapest proof executable without weakening the final quality target or forcing a winner from inadequate evidence.

The following remain frozen:

- all synthesis is cook-side and packed;
- runtime synthesis is forbidden;
- LOD0 remains `1 m / 2,048 m`;
- LOD `-1` remains `0.25 m / 512 m`;
- LOD `-2` remains `0.0625 m / 128 m`;
- the accepted u16, 2D-delta, deflate payload and runtime path remain unchanged;
- height stores dry terrain plus submerged bed;
- `waterY` stores visible water elevation;
- `watercover` stores anti-aliased occupancy;
- raw Maa-amet DTM is evidence, not an infallible packed parent;
- no morphology residual is permitted on open water;
- no scene-specific Taevaskoda rule or constant is permitted.

## 2. Stage 1 Artifact And API Contract

The current `process/microtopo/api.py` residual/exemplar facade is not preserved. It must be replaced by an absolute-surface, typed boundary. Names below describe semantics, not runtime provenance.

### 2.1 `ObservationEvidence`

`ObservationEvidence` is immutable and pre-repair. It contains:

- `artifact_sha256` and `snapshot_sha256`;
- EPSG:3301 extent, grid identity, sample-center convention, and validity extent;
- raw DTM values, source tile/vintage identifiers, no-data, interpolation, seam, water, occlusion, object, suspected-error, and protected-feature masks;
- observation class and confidence for every usable sample;
- ETAK geometry IDs and snapshot hash for water polygon, shoreline, centerline, escarpment, ditch, road, quarry, and other active hard structures;
- orthophoto RGB/CIR source, date, registration error, availability, canopy, object, shadow, water, snow, seam, and temporal-confidence masks;
- soil, geology, land-cover, hydrology, and derived-condition source IDs, support scales, confidence, and missingness;
- every raster/vector resampling operator and source CRS transform;
- human rejection/abstention annotations and reasons.

Unknown is never encoded as a category value or zero-valued observation.

### 2.2 `StructuralAuthority`

`StructuralAuthority` contains:

- absolute corrected dry ground on the canonical structural grid;
- water-surface model, water occupancy, and submerged-bed model as separate arrays/records;
- composed absolute dry-plus-bed structural height;
- correction delta from the raw observation;
- correction class, observation support, correction confidence, and uncertainty;
- hard invalid, forbidden-morphology, and non-heightfield masks;
- protected-feature IDs and before/after preservation checks;
- solver configuration, convergence record, rejected components, and reasons;
- complete evidence and algorithm digests.

No field called `residual` is permitted in this artifact.

### 2.3 `SpecialistOutput`

`SpecialistOutput` contains either:

- absolute metric height on a named canonical grid; or
- a metric delta with all of `authority_sha256`, `authority_grid_sha256`, `analysis_operator_id`, `reconstruction_operator_id`, physical band, and allowed-domain mask.

It also contains specialist/config/checkpoint digests, input-condition digest, PRF identity, window/fusion identity, validity/confidence, and a typed abstention reason. A delta that omits its exact authority and operators is rejected.

### 2.4 `FinalMaster`

`FinalMaster` contains one absolute float32 EH2000 surface at LOD `-2` sample centers, plus typed ownership/provenance/confidence arrays. Composition order is structural authority, validated long-range form, validated local realization, and then protected/forbidden constraints. Two owners of the same phenomenon and band are a configuration error.

### 2.5 Required entry points

The replacement facade exposes these conceptual calls; concrete Python dataclass names may match them exactly.

```python
collect_observation_evidence(request, snapshot) -> ObservationEvidence
repair_structure(evidence, repair_recipe) -> StructuralAuthority
run_specialist(authority, conditions, specialist_recipe) -> SpecialistOutput
compose_final_master(authority, outputs, compose_recipe) -> FinalMaster
```

`cook/micro_synth_cook.py` orchestrates these records instead of reconstructing a raw parent and asking for a residual. `cook/layers_cook.py` participates in the same dependency transaction because water, watercover, slope-derived understory, and debris depend on the corrected base.

## 3. Coordinates, Grids, And Extents

### 3.1 Coordinate oracle

All height samples use `assetgen.height_geom.sample_center_en_units`. The world anchor is `E=368640`, `N=6635520`. Integer geometry units are `1/32 m`. Mathematical floor division defines parents, including negative chunk coordinates.

No implementation may derive a second half-cell formula. Unit tests compare any vector/raster sampler directly to the oracle.

### 3.2 Stage 1 structural grids

- Repair unknowns are on LOD `-1` sample centers at `0.25 m`.
- The published structural master is on LOD `-2` sample centers at `0.0625 m`.
- Water and shoreline geometry are evaluated in world coordinates, not rasterized once per output chunk.
- Storage chunks are crops from one reconciled domain. Chunk ID never changes a value.

The cook plans three extents independently:

- **model/repair support:** all evidence and the `8 m` deterministic repair collar;
- **hierarchy support:** east, south, and southeast samples needed for complete parent aprons;
- **publication:** exactly one complete LOD `-1` parent and its 16 LOD `-2` children for this proof.

The corrected base computation supplies canonical transient LOD `-1` children across the complete affected LOD0 chunk and its hierarchy support. Only the selected parent is indexed at negative LOD. This prevents a fine island from being reduced against unrelated raw LOD0 content.

## 4. One-Snapshot Water And Height Order

### 4.1 Normative semantics

- `height(E,N)` is dry ground outside water and submerged bed inside water.
- `waterY(E,N)` is visible water-surface elevation. Dry encoded samples use the existing reserved `q=0` behavior only.
- `watercover(E,N)` is water-area occupancy in `[0,1]`, not depth and not height.
- shoreline, bank, water surface, and bed share the same ETAK/evidence snapshot and geometry digest.

Optical intensity, color, reflection, or texture may locate a candidate boundary when its registration/visibility gate passes. It may never determine bed relief or water height.

Evaluate accepted water polygons with one world-coordinate geometry object over the complete snapshot. Fine-height wet membership uses polygon `covers()` at each LOD `-2` sample center. On the existing `2 m` watercover lattice, compute occupancy as exact polygon/cell intersection area divided by `4 m2`, then encode `np.rint(255*occupancy)` as u8 (IEEE ties-to-even). Do not supersample separately per chunk. The existing coarser watercover rung is the fixed-order area mean of decoded child occupancy.

### 4.2 Required cook order

1. Freeze and hash one `ObservationEvidence` snapshot.
2. Resolve water polygon, shoreline, centerline, bank side, mapped escarpment, and water-surface model.
3. Solve dry structural ground and shoreline contact constraints.
4. Resolve surveyed or conservative-unknown submerged bed.
5. Compose dry-plus-bed structural height before any fine composition or parent reduction.
6. Build the structural-only LOD `-2` master.
7. Quantize/decode LOD `-2`; derive/quantize/decode LOD `-1`; derive corrected LOD0 and LOD1-L4 from decoded immediate children.
8. Cook matching `waterY` and `watercover` from the same snapshot.
9. Recook affected slope-derived understory/debris and any other asset whose declared input digest changed.
10. Publish corrected format-1 base first, then the format-2 negative overlay pinned to that base manifest.

Any bed, shore, or water-level change after step 5 invalidates every affected height child, parent, apron, water product, and dependent-layer digest. A post-pass that mutates only LOD0 is forbidden.

### 4.3 Water-surface model

For Stage 1, water surfaces are deterministic and low-dimensional.

**Sea:** `Y=0 m EH2000`.

**Standing water:** use the Huber location of DTM samples at least `2 m` inside the ETAK polygon after excluding object, spike, no-data, and interpolation-invalid samples. These samples are legal only as water-surface observations; they never observe the bed. Normalize residuals by the tile noise scale from Section 5.5 and use Huber cutoff `1.345`. Require at least 64 valid 1 m samples and robust spread `1.4826*MAD <=0.15 m`; otherwise abstain until a surveyed/official elevation is supplied. One polygon gets one level.

Every Huber location in this contract uses the same deterministic solve: initialize at the median, estimate scale as `max(1.4826*MAD, 0.02 m)`, update the weighted mean with weights `min(1, 1.345/abs((x-mu)/scale))`, and stop at change `<1e-8 m` or 50 iterations. Failure to converge is abstention.

**Flowing water:**

1. Densify the ETAK centerline to `1 m` stations.
2. At each station, collect valid DTM samples inside the water polygon and within a `3 m` along-channel station interval; take their Huber location.
3. Reject a station with fewer than four samples. Apply an `11 m` Hampel window and reject values beyond `3 * 1.4826 * MAD`.
4. Infer downstream only when the endpoint `21 m` Huber locations differ by more than `max(0.05 m, 2 * endpoint_MAD)`. Otherwise retain an unoriented profile.
5. For an oriented reach, apply unweighted deterministic pool-adjacent-violators isotonic regression so water does not rise downstream; merge the first violating adjacent pair encountered in upstream-to-downstream order and represent a pool by its arithmetic mean. For an unoriented reach, skip this step and mark lower confidence.
6. Smooth the accepted station sequence with a normalized positive Gaussian kernel, sigma `4 m`, truncated at `4 sigma`, nearest endpoint extension. A positive convolution preserves an isotonic sequence.
7. Linearly interpolate along the centerline and extend across each cross-section at constant elevation.

Missing station spans longer than `20 m`, branched/ambiguous reach topology, or a surface inconsistent with both banks by more than `0.25 m` causes reach abstention. The proof does not silently replace an abstention with raw rippled DTM water.

### 4.4 Submerged bed

Surveyed bathymetry, when present, is a typed observation with its own error and support. Stage 1 has no such evidence at Taevaskoda.

For unknown bathymetry, use a deliberately conservative render envelope, not inferred morphology:

```text
d = distance inside the accepted shoreline polygon
u = clamp(d / 1.0 m, 0, 1)
depth_unknown = 0.10 m * (6*u^5 - 15*u^4 + 10*u^3)
bed = waterY - depth_unknown
```

This bed is continuous with the shore, contains no along-channel noise, and never claims the old width-based carve as measured bathymetry. It is labeled `unknown_bathymetry_render_envelope`, confidence zero, forbidden to all morphology specialists, and excluded from realism statistics. If `0.10 m` clearance is insufficient for the existing renderer in the exact boot, that is a representation failure requiring review, not permission to invent deeper bed relief.

## 5. Typed Deterministic Structural Repair

### 5.1 Purpose and output

The repair reconstructs resolved structure. It does not synthesize underdetermined texture. Its output is absolute EH2000 height on the `0.25 m` structural grid plus the typed water products above.

### 5.2 Constraint priority

Constraints are applied in this order:

1. non-heightfield and forbidden-domain masks;
2. surveyed ground/bathymetry and legally fixed engineering elevations;
3. accepted ETAK water/shore/structure geometry from the frozen snapshot;
4. protected real cliffs, escarpments, boulders, channels, and depressions;
5. valid high-confidence DTM observations;
6. lower-confidence DTM/interpolation observations;
7. regularization and reconstruction priors.

A lower priority never compromises a higher priority. Incompatible constraints fail the component and record both source IDs. They are not averaged.

### 5.3 Observation classes and operators

Let `z` be dry structural height at LOD `-1` sample centers. Rows increase south and columns east. Every linear sample below uses bilinear weights in world coordinates.

| Class | Observation operator | Loss / action |
|---|---|---|
| trusted exposed ground | `O_center z` at the Maa-amet 1 m sample center | confidence-weighted Huber against raw DTM |
| lower-confidence/interpolated ground | same `O_center` | Huber with confidence `0.25`; confidence `0` if interpolation support is unknown |
| no-data, isolated spike/pit, confirmed object/canopy leak | no DTM row | reconstruct from accepted boundary/neighbor evidence |
| open water | no DTM ground row | separate water/bed contract; morphology forbidden |
| surveyed ground/bathymetry | bilinear point/footprint operator matching the survey record | hard when its record declares control-grade; otherwise Huber by recorded sigma |
| shoreline contact | bilinear samples every `0.25 m` along accepted shoreline | hard `z=waterY` at the dry toe unless a surveyed vertical structure supersedes it |
| mapped escarpment | no invented target height | graph cut across the line, one-sided reconstruction, and protected-feature checks |
| scan seam/strip with verified offset | `O_center z + b_t` for source-strip bias `b_t`, plus paired same-landform rows across the seam | Huber data/pair loss; hard `sum_t b_t=0` within a source tile |
| orthophoto edge | geometry proposal only | never a height row; unavailable/canopy/water/shadow pixels contribute nothing |

`O_center` is the mean of the central `2 x 2` structural samples surrounding the 1 m sample center. This is bilinear evaluation at that center, not a `4 x 4` area-mean claim about the Maa-amet sensor.

Stage 1 evidence classification is deterministic:

- water and structure classes come from the frozen ETAK/survey geometry;
- no-data/interpolation comes from source metadata and raster validity; unknown interpolation history receives confidence zero;
- an isolated spike/pit candidate is a valid DTM sample whose protected-edge-aware `3 x 3` median residual exceeds `max(0.25 m, 6*sigma_tile)` and whose same-side `5 x 5` neighborhood contains at least 12 valid samples;
- object/canopy leakage requires the outlier rule plus independent building/nDSM/CHM/object support; the height difference alone is insufficient;
- a scan seam/strip requires a source-tile/flightline boundary and at least 32 same-landform paired observations supporting a common offset with pair-residual MAD `<=0.05 m`;
- any candidate within the Section 5.4 protected collar abstains unless higher-priority direct evidence classifies it.

Every automatic candidate retains both `suspected` and `accepted/rejected` state. Only an accepted typed class changes an observation row.

### 5.4 Protected escarpment reconstruction

An ETAK escarpment is a barrier, not a smoothing target.

1. Densify its line to `0.25 m` and compute a stable local normal from a `2 m` arclength fit.
2. Infer high/low side from valid same-class DTM samples in the `1-3 m` normal collars. Require side median difference greater than `max(0.20 m, 3 * pooled_MAD)`; otherwise retain the line as protected but mark side unknown.
3. Within `2 m` of the line, construct the initial surface from separate robust affine fits on each side using valid points within `3 m`. Require six points and design-matrix condition number `<1e4`; otherwise use same-side nearest bilinear support and mark low confidence.
4. Delete all smoothing graph edges that intersect the line. Do not interpolate a high-side sample from low-side observations.
5. Samples within `2 m` are immune to generic spike detection unless direct object, water, no-data, or survey evidence outranks the protection.

The result may form a steep single-valued wall between adjacent samples. Overhangs, undercuts, and the vertical cliff face remain explicitly non-heightfield.

Correction confidence is an auditable evidence label, not network self-confidence: `1.0` for control-grade surveyed hard observations, `0.75` for unchanged trusted DTM, `0.5` for accepted vector-constrained reconstruction supported by valid observations on both sides, `0.25` for inpainted samples within `2 m` of valid boundary support, and `0` for unknown-bed envelopes, abstentions, or farther unsupported inpaint. The artifact stores the contributing source IDs so these ordinal values are never mistaken for calibrated probabilities.

### 5.5 Noise and confidence calibration

For each raw DTM source tile and observation class, compute

```text
r = y - median_3x3(y), excluding hard structures and invalid samples
sigma_tile = clamp(1.4826 * MAD(r), 0.02 m, 0.25 m)
sigma_i = sigma_tile / sqrt(max(confidence_i, 1/16))
```

Trusted confidence is `1`; declared lower-confidence interpolation is `0.25`; invalid is `0`. These values are evidence weights, not roughness controls. A later target-pair calibration may replace them only through a new recipe and review.

The Huber loss is

```text
rho(r) = 0.5*r*r                         if abs(r) <= 1.345
       = 1.345*(abs(r) - 0.5*1.345)     otherwise
```

where `r` is normalized by `sigma_i`.

### 5.6 Repair domain and discretization

Rasterize each typed repair component on the `0.25 m` grid, union components connected by a shared hard constraint, and add an `8 m` collar. The outermost two sample rings are fixed to the accepted one-sided/bilinear initial surface. If invalid evidence reaches that ring, expand the support; never extrapolate through the fixed boundary.

Away from protected lines, the initial surface is bilinear interpolation of valid raw 1 m sample centers, equivalent to `scipy.ndimage.map_coordinates(order=1, mode="nearest", prefilter=False)` under the exact world-to-source index transform. Any stencil touching an invalid raw sample is left unknown for the solver; it is not filled by nearest data. Section 5.4 replaces this rule inside protected one-sided collars.

Use:

- four-neighbor forward gradients divided by `0.25 m`;
- the cut-aware five-point Laplacian divided by `(0.25 m)^2`, with diagonal equal to the number of retained neighbors and off-diagonal `-1` for each retained neighbor;
- graph edges removed at shore, escarpment, or other accepted barriers;
- symmetric assembly in float64;
- lexicographic row-major unknown ordering.

For free samples, minimize

```text
E(z) = sum_i rho((O_i*z - y_i) / sigma_i)
     + sum_j ((ell^2 * L_j*z) / sigma_ref)^2
```

where `L` is the cut-aware five-point Laplacian, `sigma_ref` is the median trusted `sigma_i` in the component, and `ell` is selected once for the Stage 1 recipe from `{0.5, 1, 2, 4} m`.

Select `ell` before inspecting the Taevaskoda result. Deterministically mask 10 percent of trusted, non-protected observations outside the review camera footprint using the PRF in Section 9, simulate the observed repair-component size distribution, and minimize the median normalized held-out height error. Break a tie within 1 percent in favor of the smaller `ell`. Bind the selected value and calibration mask digest into the recipe. This is interpolation calibration, not a global morphology knob.

Hard rows from Section 5.3 are equality constraints `Cz=d`. Verified strip biases are appended to the unknown vector; all other scalar nuisance variables are forbidden in Stage 1.

### 5.7 Solver

Use Huber IRLS in float64:

1. initialize with the one-sided/bilinear surface;
2. form IRLS weight `min(1, 1.345 / max(abs(r), 1e-12))`;
3. with normalized data matrix `A`, IRLS diagonal `W`, and normalized Laplacian matrix `B`, solve `[[A.T*W*A+B.T*B, C.T],[C,0]] [z,lambda] = [A.T*W*y,d]` using `scipy.sparse.linalg.minres`; use the inverse clamped diagonal of the primal block and identity multiplier block as an SPD block-Jacobi preconditioner, diagonal clamp `1e-12`, `rtol=1e-10`, and at most 5,000 iterations;
4. repeat at most 20 IRLS iterations;
5. stop when maximum free-sample change is `<1e-5 m` and relative objective change is `<1e-8` for two consecutive iterations.

The normalized KKT residual is `norm(K*x-b,2)/max(norm(b,2),1)`.

Reject, do not publish, a component if any of these holds:

- MINRES does not converge;
- hard-row maximum error exceeds `1e-4 m`;
- normalized KKT residual exceeds `1e-8`;
- any value is nonfinite;
- at a trusted sample not superseded by a higher-priority constraint, `abs(O_center*z-y) > max(0.10 m, 4*sigma_i)`;
- an unobserved repaired sample leaves the min/max of its valid component boundary by more than `max(0.25 m, 4*sigma_ref)` without a higher-priority constraint;
- a protected-feature height jump changes by more than `max(0.02 m, 5 percent)` outside the explicitly corrected water toe;
- the result creates a new closed depression or dam above `0.05 m` in a connected drainage path unless a mapped structure supports it.

### 5.8 Stage 1 optical use and falsification

The accepted Stage 1 geometry uses ETAK shoreline as the direct boundary. RGB/CIR is fetched and registered for QA and for a separately scored boundary-refinement challenger only. The challenger may replace an ETAK segment only if its human-QA mask is clear, registration p95 is `<=0.25 m`, the imagery date is compatible with the water state, and the proposed shift is `<=1 m`.

Before any optical boundary method can become accepted, run:

- no-image and availability-mask ablation;
- image shuffled between sites;
- registration offsets of `+/-0.25`, `+/-0.5`, and `+/-1.0 m` in each axis;
- RGB/CIR date mismatch;
- image-coverage boundary crossing;
- canopy, water reflection, shadow, object, snow, and seam challenge masks.

It fails if albedo/object edges become height, missing imagery acts like bare ground, or a perturbation moves protected macrogeometry outside the registered error envelope.

## 6. Structural-Only Fine Master And Hierarchy

### 6.1 Mean-preserving structural prolongation

Outside vector overrides, prolong the corrected `0.25 m` structural surface to `0.0625 m` without inventing a morphology band.

1. Apply tensor-product Keys cubic interpolation with `a=-0.5` to the structural grid.
2. For each `0.25 m` cell, compute `d = parent_value - mean_4x4(interpolated_children)`.
3. At fine phase `u_r=(r+0.5)/4`, define

```text
b(u) = u^3 * (1-u)^3
b_norm(u_r) = b(u_r) / mean_r(b(u_r))
B(r,s) = b_norm(u_r) * b_norm(u_s)
```

4. Add `d * B(r,s)` to that cell's 16 fine samples.

The bubble has zero value and first derivative at the mathematical parent-cell boundary and discrete `4 x 4` mean one. Therefore it preserves cubic cross-cell value/slope continuity and the exact structural parent mean. This operation is a reconstruction operator, not terrain detail.

At the accepted shoreline, evaluate wet/dry membership and the bed equation directly at LOD `-2` sample centers after prolongation. Reapply shoreline contact and protected constraints. Those resolved vector overrides are allowed to change the subsequently derived LOD `-1` mean.

### 6.2 Publication order

For the one-parent proof:

1. produce one reconciled structural LOD `-2` master over the 16 published children plus model and hierarchy support;
2. crop, quantize at `0.002 m`, and browser-equivalent decode every published/support LOD `-2` chunk;
3. assemble decoded children plus east/south/southeast support with `micro_hierarchy.py`;
4. derive LOD `-1` using its fixed-order float64 `4 x 4` area mean;
5. quantize/decode LOD `-1` at `0.005 m`;
6. use this decoded parent in the canonical transient LOD `-1` set for the containing LOD0 chunk;
7. derive corrected LOD0 from decoded LOD `-1` children, then derive every affected LOD1-L4 ancestor from decoded immediate children;
8. include west, north, and northwest apron-file closure at every changed rung;
9. publish the corrected format-1 base and then its pinned format-2 overlay.

For exact wire qscale `s=float(float32(requested_qscale))` and decoded float32 `d`, every encode passes `max(abs(float64(d)-float64(h))) <= s/2 + 2*abs(spacing(float32(max(abs(d)))))`, no clipping, finite values, decoded value/gradient/normal seams, and browser-equivalent child-to-parent rederivation. `waterY` and `watercover` remain on their existing `2 m` layer lattice and existing published rungs; this contract changes their compatible content and ordering, not their binary format. The current one-parent shared qoffset policy remains local to this proof; qoffset closure follows the fixed-point changed-core/apron policy in the main spec Section 14.4.

Stage 3 is no longer a prerequisite for this structural publication. A later validated morphology specialist may replace the same fine master under a new recipe.

## 7. Exact Analysis And Reconstruction Pyramid

Analysis bands are not packed rungs. They are diagnostics/training targets evaluated after the same composition and, for release metrics, after browser-equivalent quantization/decode.

### 7.1 Analysis low-pass `F4`

The 1D factor-four analysis filter is the threefold convolution of a four-sample box:

```text
c = [1, 3, 6, 10, 12, 12, 10, 6, 3, 1] / 64
F4(x)[p] = sum(k=0..9, c[k] * x[4*p - 3 + k])
```

Apply it separably in easting then northing, with float64 accumulation. Its center is between fine indices `4p+1` and `4p+2`, exactly the coarser sample center. Its 1D transfer magnitude is

```text
abs(sin(2*w) / (4*sin(w/2)))^3
```

with the linear phase implied by the half-sample-centered support. The DC limit is one.

### 7.2 Boundary and invalid-data policy

- Each `F4` application requires three input samples before and six after its nominal four-sample parent footprint. Allocate a symmetric six-input-sample halo for one stage. Computing both `A1` and `A0` from `H2` requires a symmetric 30-finest-sample (`1.875 m`) halo before cropping valid `B1/B2` diagnostics.
- Where a real domain boundary exists, reflect about the exterior sample-center plane; never wrap or clamp.
- Reflected samples are permitted only to compute context. Any target/loss output whose filter footprint touches a plot boundary, NaN, interpolation, or human-invalid cell is invalid and excluded.
- Storage/publication boundaries are never analysis boundaries; expand the source domain instead.

### 7.3 Reconstruction `R4` and exact bands

`R4` is separable Keys cubic convolution with `a=-0.5`. For fractional position `t` between coarse samples, use:

```text
w[-1] = -0.5*t + t^2 - 0.5*t^3
w[ 0] = 1 - 2.5*t^2 + 1.5*t^3
w[+1] = 0.5*t + 2*t^2 - 1.5*t^3
w[+2] = -0.5*t^2 + 0.5*t^3
```

Fine phases relative to a parent center are `-3/8`, `-1/8`, `+1/8`, and `+3/8` parent cells. Use reflected support under Section 7.2.

For finest surface `H2`:

```text
A1 = F4(H2)
A0 = F4(A1)
B2 = H2 - R4(A1)
B1 = A1 - R4(A0)
H2 = R4(R4(A0) + B1) + B2
```

The last equality is exact by construction, up to declared float64 arithmetic. Report signal, target error, model error, and packing error separately for `B1` and `B2`.

### 7.4 Target acquisition transfer

The Moore DEM is a `0.01 m` natural-neighbor raster already mean-filtered over `0.03 m`. Do not pretend its nominal grid is an unfiltered surface. Resample it to `0.0625 m` by exact separable source-cell/output-cell overlap area, treating each finite source cell as a constant area sample. Divide the finite height-overlap sum by finite overlap area. Accept an output cell only when finite overlap is at least 95 percent of its area. Bind the original `0.03 m` smoothing and reported validation errors into the target record, then apply `F4` to the resampled surface.

## 8. Stage 1 Acquisition And Condition Contract

### 8.1 Taevaskoda snapshot acquisition

The Stage 1 structural proof binds these exact current artifacts before repair:

- DTM sheet `54472`, official endpoint `https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=54472&andmetyyp=dem_1m_geotiff&dl=1&f=54472_dtm_1m.tif&page_id=614`, current SHA-256 `a70ddfeeec08f278201c6479205ca661e247b1ec6b0151597390aa1c726d5239`;
- whole-country ETAK GeoPackage from the existing `fetch/etak.py` endpoint ending `andmetyyp=ETAK&dl=1&f=ETAK_EESTI_GPKG.zip&page_id=609`, current extracted GeoPackage SHA-256 `e19084c4bff0a6096a7dd38cefa12402bdce98d59c73d5b35c30dda60682f5f9`;
- official download workbook `library/data/maaamet/tomba_etak_avaandmed.xlsx`, SHA-256 `fbef1433eff6116174cd1bbeca6e6550e093eeed794c14d583af77e03970b4b5`;
- RGB `54472_OF_RGB_GeoTIFF_2025_07_18.zip`, expected compressed bytes `207351332`, using the workbook URL with `andmetyyp=ortofoto_eesti_rgb`;
- CIR `54472_OF_CIR_GeoTIFF_2024_05_22.zip`, expected compressed bytes `133805748`, using the workbook URL with `andmetyyp=ortofoto_eesti_cir`.

RGB/CIR cache path is `asset-gen/data/in/orthophoto/{rgb,cir}/54472/<archive-name>`, with extraction under a sibling directory named by archive SHA-256. Download to `<name>.part`, resume only when the server accepts byte ranges and the retained prefix length matches, apply the existing configured five retries/politeness interval, validate expected content length, ZIP CRC, GeoTIFF readability, CRS EPSG:3301, transform, bounds, band count/type, and then atomically rename. Record archive and extracted-file SHA-256 values. An HTTP HTML/error body, changed workbook row, changed size, missing CRS, or unexpected schema fails closed; it never becomes a zero/missing-image tensor silently.

Validate ETAK before use. All six required geometry columns are `geom` with `gpkg_geometry_columns.srs_id=3301`. Required fields are:

- `E_201_meri_a`: `etak_id,kood,kood_t,muutmisaeg,geom_muutmisaeg,valjavote`;
- `E_202_seisuveekogu_a`: those fields plus `tyyp,tyyp_t,nimetus`;
- `E_203_vooluveekogu_a`: base fields plus `nimetus`;
- `E_203_vooluveekogu_j`: base fields plus `tyyp,tyyp_t,telje_tyyp,telje_tyyp_t,laius,laius_t,telje_staatus,telje_staatus_t,nimetus`;
- `E_204_kaldajoon_j`: base fields plus `tyyp,tyyp_t,kalda_veekogu_tyyp,kalda_veekogu_tyyp_t`;
- `E_102_nolv_j`: base fields plus `tyyp,tyyp_t,kaldaastang,kaldaastang_t`.

Persist exact layer/field/type inventory and feature IDs used in the proof. A missing, renamed, type-changed, or wrongly projected field is schema drift and fails closed.

All outputs carry Maa- ja Ruumiamet attribution and the repository's declared CC BY 4.0-equivalent terms. RGB 2025 and CIR 2024 are separate dated observations; the one-year mismatch is explicit condition metadata, not a co-temporal composite claim.

### 8.2 Conditions

The structural proof consumes only conditions with a declared causal role:

- raw DTM, source/error masks, and one-sided derivatives at `1, 4, and 16 m` support;
- ETAK water, shore, centerline, bank side, and escarpment signed distances at `0.25 m`, with source IDs;
- aspect/slope at `1, 4, and 16 m`;
- curvature/topographic position at `4, 16, and 64 m`;
- flow direction and contributing area from corrected structure at `1 and 4 m`;
- wetness index at `4, 16, and 64 m`;
- distance and side to shore/escarpment/ditch/road/quarry structures;
- dated RGB/CIR plus explicit registration/availability/canopy/water/shadow/object masks;
- soil/geology/land-cover values only at their real mapped support, with unknown/coverage masks.

The Moore bakeoff has none of the Estonia condition fabric. Its legal conditions are only the degraded `0.25 m` parent, derivatives of that parent, target validity, and documented plot surface semantic. Site identity is retained for splitting/provenance but is not a model input. Supplying guessed peat type, soil, water table, north orientation, or Estonia covariates is forbidden.

## 9. PRF, Serialization, Windows, And Fusion

### 9.1 PRF

Use keyed BLAKE2b-256 from Python `hashlib`, not Python `hash()`, NumPy global RNG, chunk seeds, or framework-default seeds.

- key: exactly 32 recipe-seed bytes;
- personalization: ASCII `laas-micro-prf1`;
- digest size: 32 bytes;
- all integers: fixed-width little-endian two's-complement;
- all text-to-ID conversion: UTF-8 NFC followed by SHA-256;
- no floats or variable-length strings enter the PRF record.

The 112-byte record is:

| Bytes | Field |
|---:|---|
| 0-1 | version `u16=1` |
| 2 | purpose enum `u8` |
| 3 | stage enum `u8` |
| 4-35 | recipe SHA-256 bytes |
| 36-51 | specialist UUID bytes |
| 52-83 | physical-domain SHA-256 bytes |
| 84-91 | easting/local-x units `i64` |
| 92-99 | northing/local-y units `i64` |
| 100-103 | channel `u32` |
| 104-111 | counter `u64` |

Purpose enums are `1=sample_uniform`, `2=sample_normal`, `3=event`, `4=window`, `5=training_mask`, `6=augmentation`.

World sample coordinates use the exact `1/32 m` integer coordinates from `sample_center_en_units`. Research plots use a domain-local origin snapped down to `1/32 m`; the plot artifact digest prevents collisions. Events are keyed by their owning lattice cell and event counter, then derive continuous offsets from PRF uniforms. Rounded continuous event positions never become a new random key.

Uniform float32 uses the high 24 bits of the next little-endian u64 digest word: `(bits24 + 0.5) / 2^24`. A normal pair uses digest u64 words 0 and 1 as the two open uniforms in Box-Muller; additional pairs increment `counter`. The recipe binds Python, NumPy/framework, CPU/GPU backend, and math implementation; a different stack is a different recipe.

### 9.2 Moore inference lattice

The bounded peat experiment uses:

- support window `2.0 m x 2.0 m` (`32 x 32` finest samples);
- valid output core `1.0 m x 1.0 m` (`16 x 16`);
- center stride `0.5 m` (`8` finest samples);
- domain-local lattice anchor from the snapped plot origin;
- output only where every required support sample is target-valid.

For point `p`, enumerate every lattice window whose half-open core contains `p`, independent of requested crop. Sort windows by `(center_y_units, center_x_units)`.

For local core coordinate `s` in meters, use

```text
w1(s) = cos(pi*s / 1.0 m)^2   for abs(s) < 0.5 m
      = 0                     otherwise
w(x,y) = w1(x) * w1(y)
```

Accumulate numerator and denominator in float64 in the sorted order. Reject a sample if total weight is zero.

### 9.3 Fusion quantity

Exemplar, simulation, event/process, regressor, GAN, and non-diffusion hybrid candidates fuse their **metric delta against the named structural reconstruction**, never normalized latent values, categories, or independently shifted absolute heights. Add the fused delta once, then reapply protected/forbidden constraints.

Diffusion uses one coordinate-indexed initial noise field for the whole expanded domain. At every denoising step, evaluate all influencing windows, fuse predicted `v` in metric-normalized band space with the same weights/order, update one global state once, and proceed. Final-output-only feathering is not the registered diffusion candidate.

Changing AOI, chunk request, worker count, batch order, retry order, or adjacent requested windows must produce identical float32 master bytes on the locked stack.

## 10. Moore Target Qualification

### 10.1 Evidence boundary

The Moore archive contains 68 disconnected foreign peat plots and `309.1387 m2` of finite `0.01 m` moss-surface DEM. Vascular vegetation was clipped; the surface was natural-neighbor interpolated and mean-filtered over `0.03 m`. Reported average DEM differences were about `0.004 m` for lab and `0.018 m` for field validation.

The experiment may test only:

- the `0.0625-0.25 m` local analysis band;
- local moss/peat surface height, gradient, normal, curvature, spectral, variogram, extrema, and connected-form distributions;
- local response conditioned on a supplied true `0.25 m` parent;
- tiling, repetition, determinism, packing survival, and rendering.

It may not test or claim:

- `0.25-10 m` hummock/hollow/lawn or ridge/pool organization;
- `1 m -> 0.25 m` source correction;
- 128 m support or transitions;
- Estonia soil, mire type, hydrology, season, or transfer;
- production ownership.

### 10.2 Site groups and splits

Group by acquisition locality/campaign, never random patch:

- all 50 `REC_*` plots: Red Earth Creek;
- Alpha, Beta, Gamma, Epsilon, Zeta, Eta, Iota, Kappa, Theta, Lambda: Nobel;
- WET, INT, DRY: Seney campaign;
- Maine/Caribou Bog;
- James Bay;
- Limerick;
- Puslinch;
- Sweden/Rodmossen.

Use eight outer leave-one-group-out folds. Within each outer training set, select any registered validation choice by leave-one-group-out again. The outer group remains untouched until configuration and seed selection are frozen.

### 10.3 Qualification gates

Before training a candidate:

1. verify archive and per-record hashes, units, orientation, finite mask, `0.01 m` spacing, surface semantic, and group IDs;
2. reproduce finite area totals within `0.01 m2`;
3. propagate Puslinch/lab validation residuals through the exact resampler and `F4` to estimate per-band measurement error;
4. require at least 75 percent of eligible plots and at least six of eight site groups to have `B2` robust RMS greater than twice the corresponding propagated error RMS;
5. require at least 200 valid `2 m` support windows across at least six groups after invalid collars;
6. freeze descriptors, folds, model configurations, seeds, budgets, and render views before seeing outer-fold results.

If validation residuals cannot support a per-band error estimate, or any gate fails, Track A fails and Track C makes no quality conclusion. That is a valid decisive Stage 1 result.

## 11. Preregistered Peat Candidates

### 11.1 Shared task

For each valid window:

- target `H` is the area-resampled `0.0625 m` Moore surface;
- input `P` is the fixed-order `4 x 4` area mean at `0.25 m`;
- base `S` is the mean-preserving prolongation in Section 6.1;
- metric target is `R=H-S`;
- input tensors are `S`, x/y gradient of `S`, Laplacian of `S`, local relief at `0.5 and 1 m`, and validity;
- subtract the support-window mean height and define `scale=max(0.02 m, median_training_window(p95(S)-p5(S)))`; divide height, relief, and residual by `scale`, multiply gradient by `0.25 m/scale`, and multiply Laplacian by `(0.25 m)^2/scale`; output is converted back to meters before fusion;
- loss and metrics use only the central `1 m` core and valid analysis collar;
- no exact post-hoc projection to raw/coarse means is applied. The packed parent is derived from decoded output.

Use D4 rotations/reflections only because the archive does not supply reliable geographic north. Apply the same augmentation stream to all candidates. Record original and transformed identity.

### 11.2 Candidate E: contextual exemplar

1. Store every training support window and its `R`.
2. Form a descriptor from the `8 x 8` parent, parent gradients, relief, and Laplacian; standardize on the training fold.
3. Fit 32-component PCA on training descriptors only.
4. Retrieve the eight nearest training descriptors by Euclidean distance, forbidding the held-out site.
5. Select one with PRF categorical probability proportional to `exp(-(d-d_min)/max(median(d-d_min),1e-6))`.
6. Return its metric residual after the registered D4 orientation transform; fuse by Section 9.

No patch blending outside the fixed fusion contract and no same-site exemplar is allowed.

### 11.3 Candidate S: conditional geostatistical simulation

1. Fit the training-fold residual mean as one shared pointwise ridge regression from the shared channels in a reflected `5 x 5` finest-grid neighborhood, ridge `alpha` selected from `{1e-4,1e-3,1e-2}` by inner folds.
2. Fit an isotropic Matern covariance to the remaining residual using robust binned semivariograms at `0.0625 m` lags through `1 m`; select `nu` from `{0.5,1.5,2.5}` and fit range/sill/nugget by bounded SciPy least squares.
3. Generate a coordinate-PRF Gaussian field by circulant embedding over the expanded plot domain, never per window.
4. Condition the residual's aligned `4 x 4` block means to zero by linear Gaussian conditioning using the exact area operator; solve with Cholesky plus jitter `1e-8 * sill`.
5. Add the learned conditional mean and fuse/crop normally.

Failure of positive-definite embedding after doubling each FFT dimension twice is candidate abstention, not permission to use fBm.

### 11.4 Candidate P: calibrated marked-form model

This is an empirical morphology challenger, not a proven peat-growth simulator.

1. Form `R_low=R4(F4(R))` and compute its positive/negative extrema.
2. Keep extrema whose prominence exceeds twice propagated target error; segment their basins by watershed on signed residual.
3. Fit each accepted form with the signed elliptical Wendland C2 profile `A*(1-r)^4*(4*r+1)` for `0<=r<1` and zero otherwise, recording amplitude, major/minor radii, orientation relative to parent gradient, overlap, and nearest-neighbor spacing.
4. Fit separate empirical mark distributions for positive and negative forms and a Matern type-II hard-core distance equal to the training-fold fifth percentile of same-sign center spacing.
5. Generate form centers from coordinate-PRF candidates on a `0.125 m` owner lattice, accept in deterministic priority order, draw marks by inverse empirical CDF, and condition orientation only on parent gradient.
6. Sum accepted forms once in domain coordinates. Fit one scalar amplitude for the complete outer-fold model by least squares over all inner-training central cores, freeze it before the outer group, and fuse/crop normally.

No unmodeled roughness remainder is added. If fewer than 200 accepted forms exist across six groups, this candidate is ineligible.

### 11.5 Candidate R: deterministic regressor

Use a three-level residual U-Net: channels `(48,96,192)`, two `3 x 3` residual blocks per level, GroupNorm with eight groups, SiLU, stride-2 convolution down, nearest-neighbor plus `3 x 3` convolution up, reflected one-pixel padding at every `3 x 3` convolution, no attention, and one metric residual output. Gradient and Laplacian losses use the metric finite-difference operators from Section 5.6. Loss is:

```text
Huber(delta=0.01 m) height
+ 0.25 * L1 gradient-vector
+ 0.10 * L1 five-point Laplacian
```

No random/noise input is allowed. This candidate represents the conditional-mean ceiling and is expected to lose multimodal variance rather than hide stochasticity.

### 11.6 Candidate G: conditional GAN

Generator is Candidate R plus one coordinate-PRF normal channel. Discriminator is a four-level `(48,96,192,256)` spectral-normalized `4 x 4` PatchGAN conditioned on the shared input. Use hinge GAN loss. Generator loss is Candidate R loss plus `0.02 * adversarial`; discriminator loss has weight one. No RGB perceptual loss, style loss, or pretrained image VAE is allowed.

One GAN optimizer update means one discriminator update followed by one generator update on the same batch identities.

### 11.7 Candidate D: pixel-space conditional diffusion

Use Candidate R's U-Net with sinusoidal timestep embedding projected to 192 channels and FiLM in every residual block. Train `v` prediction with the Nichol-Dhariwal cosine cumulative-alpha schedule `s=0.008`, 1,000 training timesteps, and:

```text
MSE(v)
+ 0.10 * reconstructed-x0 gradient-vector L1
+ 0.05 * reconstructed-x0 Laplacian L1
```

Inference uses 50 deterministic DDIM steps at integer timesteps obtained by rounding a descending linear spacing from 999 through 0, deduplicating while preserving order; `eta=0`, one coordinate-PRF normal field, and per-step fusion from Section 9. No latent RGB model or ControlNet transfer is allowed.

### 11.8 Candidate H: process-plus-diffusion hybrid

Generate Candidate P first. Train Candidate D on `R - P` with `P` as an additional condition. The composer adds `P` and the fused diffusion remainder exactly once. If Candidate P is ineligible, Candidate H is ineligible; it does not silently become Candidate D.

### 11.9 Shared optimization and budget

For R/G/D/H:

- float32 training, float64 metric conversion;
- AdamW, learning rate `2e-4`, betas `(0.9,0.999)`, weight decay `1e-4`;
- batch 32, gradient norm clip 1;
- 50,000 optimizer updates per outer fold and seed;
- training seeds `{17,31,47}` selected only by inner site folds;
- at most 3.5 million trainable parameters per network;
- no pretrained weights;
- identical accepted training windows and augmentation counts;
- report training GPU-hours, peak memory, inference seconds/km2, support multiplier, and packed entropy.

E/S/P receive the same folds and input windows and report CPU-hours/memory. Equal wall-clock is not imposed across fundamentally different algorithms; configuration and compute are frozen and all cost is reported before blind quality inspection.

## 12. Bakeoff Decision

Every stochastic candidate emits eight PRF realizations per outer held-out site; the registered production seed is not selected by visual preference. Report per site and aggregate:

- height, gradient vector, normal, multi-radius curvature, `F4` band energy, PSD, variogram, extrema/form size-spacing-orientation, and connected-component distributions;
- real-vs-real held-out variability as the non-inferiority envelope;
- source-parent deviation after the actual quantize/decode/hierarchy path;
- window/chunk periodicity, nearest-neighbor copy score, and train-patch attribution;
- p50/p95/p99 packed bytes and complete cook cost;
- fixed-camera ground-level geometry, silhouettes, contact, parallax, moving-light response, material/vegetation grounding, and LOD-motion review.

Discard a candidate for target qualification failure, measurement-error-scale output, generic random bumpiness, copied held-out/training identity, repeated forms, window/chunk seams, forbidden water relief, barely visible/shading-only detail, failure to survive the actual terrain DAG, or unreported cost.

For each descriptor family, compute the registered distance between generated and held-out distributions per outer site. Build the real-vs-real reference by 10,000 site-block bootstrap resamples of held-out plots. A candidate is non-inferior only when its one-sided 95 percent bootstrap upper confidence bound is no greater than the real-vs-real 95th-percentile distance plus propagated measurement-error distance. Apply Holm-Bonferroni at family-wise `alpha=0.05` across descriptor families. The bootstrap PRF purpose is `training_mask`, with replicate number in `counter`.

A candidate is eligible only when every registered descriptor family passes that rule and blind rendered review prefers it to structural-only without a hard failure. Among equivalent survivors, choose the simpler/cheaper one.

`No candidate survives` and `target evidence is insufficient` are valid, decisive results. Any winner remains `research / foreign Moore local B2 band` and cannot activate Estonia peat or satisfy a national release row.

## 13. Stage 1 Exit Criteria

Stage 1 succeeds only when:

1. the Taevaskoda structural proof publishes structural-only LOD `-2`, derived LOD `-1`, corrected LOD0, and affected ancestors from one snapshot;
2. the 1 m shoreline staircase and diagnosed river bumps are visibly absent without flattening the mapped bank/escarpment;
3. water, bed, shore, terrain, vegetation, materials, grass, and dependent slope layers agree in the exact real-WebGPU boot;
4. all encode, hierarchy, apron, value/gradient/normal seam, and parent-closure gates pass;
5. Moore Track A either passes its declared qualification or records a decisive evidence failure;
6. if Track A passes, Track C either selects a research-only local-band survivor or records `no_candidate_survives`/`tie_no_owner`; if Track A fails, Track C is not trained, viewed, or ranked and records `target_evidence_insufficient`;
7. no result is described as Estonia-wide morphology, real bathymetry, or a solved vertical cliff face.

The structural proof is independent of a peat winner. The peat quality conclusion is forbidden if Track A fails. No later morphology implementation may change this contract silently; it requires a new reviewed recipe/sub-specification.
