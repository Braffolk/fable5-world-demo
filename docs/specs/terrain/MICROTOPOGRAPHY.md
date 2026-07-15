# SPEC: Evidence-Grounded Cooked Microtopography

**Project:** `laas`, a 1:1 streamed Estonia open world

**Date:** 2026-07-13

**Status:** reviewed architecture with zero-external-budget evidence amendments; bounded weak-surface and R0 process research-preview paths are authorized, while production morphology remains gated by public-data qualification

**Decision:** implement one evidence-gated typed hybrid surface pipeline in `asset-gen`. Deterministic evidence fusion owns source repair and mapped structure. A deterministic regime graph owns eligibility, transitions, and abstention. Production unresolved morphology is produced only by a per-regime specialist that wins a preregistered target-scale bakeoff. Before such targets exist, evidence-tiered research specialists may be developed and packed only into immutable, explicitly non-production browser previews under Sections 9.6-9.9. A deterministic composer produces one canonical absolute finest surface, and the existing packed-height cook derives and publishes the hierarchy. The browser performs no synthesis.

**Quality target:** beautiful, realistic, non-repeating geometric variance whose forms and distributions change for defensible physical reasons: substrate, surficial material, soil profile, hydrology, slope position, vegetation and disturbance history, land use, management state, and direct local evidence.

---

## 1. Purpose

This document replaces every earlier microtopography implementation specification. It is standalone: an implementer may assume the repository exists, but may not assume knowledge of the preceding failed synthesis.

The requested result is not smoother interpolation, fewer visible triangles, a noise layer, or exact reconstruction of an unknowable surface. Maa- ja Ruumiamet's 1 m DTM does not contain the location of every decimeter stone, clod, root mound, erosion lip, joint, hollow, or rut. The product must:

1. correct supported source defects instead of preserving them for a zero-error score;
2. retain real measured and mapped structure;
3. add unresolved geometry whose morphology and variance match real target-scale surfaces from the same physical regime;
4. remain deterministic, seamless, parent-consistent, packed, and streamable;
5. fail closed in production where the target prior or representation is unsupported, while permitting explicitly labeled research-only previews to expose bounded weak-evidence hypotheses without claiming truth or transfer;
6. pass a ground-level rendered beauty review, not only internal numerical gates.

The final terrain remains synthesized below the reliable source support. Metadata and product descriptions must not call it recovered survey truth or 1:1 centimeter reconstruction.

## 2. Non-Negotiable Laws

1. **All synthesis is cook-side.** Every final height is generated and packed in `asset-gen`. The browser may demand, fetch, decode, cache, interpolate, morph, sample, cull, and render packed data. It may not invent terrain.
2. **Store final geometry, not synthesis parameters.** The output is ordinary absolute EH2000 height data in the accepted payload. No runtime decoder network, procedural parameters, residual texture, fBm, or browser displacement is permitted.
3. **Keep the canonical shared grid.** Height LOD 0 remains 1 m in 2,048 m chunks. LOD -1 is 0.25 m in 512 m chunks. LOD -2 is 0.0625 m in 128 m chunks. Every rung retains a `2049 x 2049` payload including apron. Do not redefine global LOD 0 as 128 m.
4. **Keep the accepted representation path.** The next proof may change cooked height contents, recipe identity, release planning, and verification. It may not add a new LAC payload, index encoding, server protocol, runtime synthesis path, shader path, or provenance branch in `TerrainField`. One representation-only runtime defect remains: streamed trees/boulders and understory/debris currently ground at LOD0 rather than the packed fine/morphed surface. Section 19.5 authorizes only the minimal shared packed-surface sampling correction required to close that defect; it may not synthesize or reinterpret geometry.
5. **Beauty is the objective; delivery correctness is a constraint.** Quantization, determinism, seams, hierarchy consistency, and clean boot are mandatory. None proves realistic morphology.
6. **The raw DTM is fallible evidence.** Exact agreement is required only where the calibrated observation model supports it. Known water, interpolation, vegetation, object, scan, and gridding errors may and should be corrected.
7. **The corrected packed hierarchy is authoritative.** Once evidence selects a corrected surface, all parents must agree with that accepted surface. Raw defects must not reappear during LOD fallback.
8. **No decorative noise.** White noise, generic fBm, chunk-local random fields, sine/checker fixtures, one global roughness multiplier, and nearest-class fallback are forbidden as shipped morphology.
9. **No broad-class style lookup.** `forest`, `field`, `bog`, `sand`, or a soil code alone may not select final geometry. Physical factors remain separate until a supported regime and event state are established.
10. **One canonical finest surface.** LOD -2 is generated once in world coordinates. LOD -1, corrected LOD 0, and affected ancestors derive from the accepted decoded child hierarchy; rungs are never synthesized independently.
11. **Storage chunks are not physical domains.** Feature ownership, phase, conditions, and stochastic state depend on stable world coordinates and physical domains, never final chunk identity, cook order, worker, request sequence, or AOI enclosure.
12. **Unsupported means no invented production residual.** It does not mean generic detail, a foreign analogue, or the visually nearest class in any production, `pilot`, `national`, or `latest` asset. A research specialist may emit only inside an explicit research-preview domain that passes Sections 9.6-9.9 and 14.6; that geometry remains non-authoritative and cannot substitute for an unsupported production row.
13. **A heightfield claim is single-valued.** Vertical faces, undercuts, caves, overhangs, detached blocks, and root plates are not solved by finer sampling. This spec marks them unsupported; a separate structural representation requires a separate specification.
14. **All surface consumers agree.** Terrain mesh, grass roots, trees/plants, materials, normals, probes, collision, and water boundaries must follow the accepted packed surface and transition policy. Fine terrain may not disable or strand inherited content.
15. **No scene-specific fixes.** Suur Taevaskoda is a fixed generalization review site, not a coordinate special case, hand-authored patch, or training/tuning site.
16. **No unsupported implementation.** The user reviewed the architecture on
    2026-07-13 and rejected paid acquisition plus a separately funded cliff
    project. Implementation may proceed only through the zero-budget public-data
    and staged gates in this document; review does not authorize invented evidence.
17. **Research preview is not release authority.** On 2026-07-14 the user
    explicitly authorized relaxing survey-truth gates for bounded visual/model
    research. This permits weak, probabilistic surface supervision and immutable
    browser previews only. It does not qualify a target, select a production owner,
    establish Estonia transfer, change a regime release row to `pilot` or
    `national`, or authorize publication to `latest`.

## 3. Premise Audit

The following instruction is included verbatim and governs this design:

> "When a problem resists solving, or a result disappoints, do NOT start varying your approach to the problem — that stays inside the problem. Go up one level first, to the context the problem lives in: thesurrounding system, the upstream decisions, the goal that made this a problem, and everything you've been treating as the fixed environment it sits inside. What you filed under 'given' is the prime suspect,precisely because you filed it under 'given' and never looked at it. Put the background on trial: What generates this problem? Is that context flawed in a way that PRODUCES this failure? Can I feasibly change the context instead of out-thinking a problem that shouldn't have existed? Only if the context is genuinely fixed, or genuinely sound, drop back down and solve the problem where it sits."

### 3.1 The false given

The failed design treated a literal fine surface whose every 1 m mean equals the raw DTM as the goal. That context produced the failure:

- a stepped 1 m shoreline was preserved exactly;
- false river bumps and interpolated water artifacts remained authoritative;
- the generator was confined to mean-null bumpiness;
- numerical hierarchy gates passed while visible terrain remained wrong;
- a foreign forest residual was mistaken for a national morphology prior.

### 3.2 The real fixed context

The actual fixed requirements are:

- the user must see convincing near-ground geometry;
- detail must be generated and packed offline;
- the existing negative height rungs, payloads, serving, and runtime path are accepted;
- the result must be deterministic and streamable;
- the renderer must not synthesize or add a second geometric terrain surface.

Raw source values, correction policy, conditioning fidelity, training data, generator family, physical processing domain, and hierarchy authority are not fixed. They are precisely the upstream systems that must change.

### 3.3 Consequence

The design separates four artifacts:

1. **raw observation:** what the sensor product reports;
2. **corrected structural authority:** the evidence-fused single-valued ground/bed;
3. **unresolved realization:** a supported sample of physically plausible missing morphology;
4. **final packed surface:** the constrained, quantized, parent-closed surface the user sees.

Exact raw agreement is replaced by calibrated observation fidelity. Exact packed-hierarchy consistency remains mandatory after the accepted surface is chosen.

## 4. Verified Repository Baseline

### 4.1 Physical lattice

The frozen world anchor is `E=368640`, `N=6635520`. Game coordinates are
`X=E-368640`, `Z=6635520-N`, and `Y` is EH2000 height. Signed height geometry uses
integer `1/32 m` units. `height_geom.sample_center_en_units` is the coordinate
oracle: every sample is at the exact half-cell center, and apron index `2048`
duplicates the east/south neighbor's first sample. Parent coordinates use
mathematical floor division, including negative chunk coordinates; one parent owns
`4 x 4` immediate children.

In geometry units, `U=32 units/m`, `A_Eu=11796480`, `A_Nu=212336640`, and
`R=2048`. For rung `l`, `T_l=32*4^l` when `l>=0`, `T_-1=8`, `T_-2=2`, and
`F_l=R*T_l`. Chunk `(l,cx,cz)` and payload `(row,col)` use:

```text
O_E = A_Eu + cx*F_l
O_N = A_Nu - cz*F_l
E_u = O_E + col*T_l + T_l/2
N_u = O_N - row*T_l - T_l/2
parent(l,cx,cz) = (l+1, floor(cx/4), floor(cz/4))
```

Convert to meters only after the integer calculation. Python `//` is required;
truncation toward zero is wrong for negative coordinates.

| LOD | Texel | Chunk footprint | Payload | Height qscale |
|---:|---:|---:|---:|---:|
| -2 | 0.0625 m | 128 m | `2049 x 2049` | 0.002 m |
| -1 | 0.25 m | 512 m | `2049 x 2049` | 0.005 m |
| 0 | 1 m | 2,048 m | `2049 x 2049` | 0.01 m |
| 1 | 4 m | 8,192 m | `2049 x 2049` | 0.01 m |
| 2 | 16 m | 32,768 m | `2049 x 2049` | 0.01 m |
| 3 | 64 m | 131,072 m | `2049 x 2049` | 0.25 m |
| 4 | 256 m | 524,288 m | `2049 x 2049` | 1 m |

The grid is defined by `asset-gen/config/base.toml`, `asset-gen/src/assetgen/grid.py`, and `asset-gen/src/assetgen/height_geom.py`. The finest pitch has a nominal two-sample wavelength of 0.125 m; it does not authorize millimeter grains, grass blades, leaf litter, or features that do not survive triangulation and filtering.

### 4.2 Accepted infrastructure

The accepted implementation baseline is commit `0ef75cf` as contained in current
branch commit `51d1e1e`. This identifies reusable infrastructure, not accepted
morphology and not an instruction to rewrite runtime code.

Keep:

- signed negative height LOD transport and LAC2 decoding;
- near-only demand, plane fill, cache, sampling, and packed-level morph;
- ordinary quant16, 2D delta, deflate encoding;
- checked non-clipping quantization and browser-equivalent decode;
- decoded-child parent derivation in `cook/micro_hierarchy.py`;
- content-addressed staged build/publication and strict boot tooling;
- fine terrain, material, and grass surface agreement;
- compact raster allocations and existing shader behavior.

Do not claim complete vegetation agreement from this baseline. Streamed trees and
boulders fetch their record key's LOD0 height in `src/nanite/world/ChunkContent.ts`;
understory and debris request LOD0 in `src/gpu/passes/UnderstoryScatter.ts`. They
therefore miss negative-rung geometry and the moving packed-level morph. This is a
known representation-sampling defect, not permission for runtime synthesis and
not a reason to redesign the format.

The bounded current coverage contract publishes one LOD -1 parent and 16 LOD -2
children. Nine additional east/south/southeast LOD -2 chunks exist only to supply
decoded parent-apron support.

Rejected as a quality method:

- the LUKE residual bank as a production Estonia prior;
- patch quilting as the national generator;
- raw-LOD0 mean-null projection as the beauty objective;
- the current `measured-synthesis-pilot` morphology and its generator-specific verification gates.

The accepted transport commit is not evidence that the current payload content is realistic.

### 4.3 Current cook conflict

`cook/micro_synth_cook.py` currently asks for a fine residual, reconstructs the pinned raw LOD0, projects the residual into its one-meter mean-null space, stages only LOD -2/-1, and verifies agreement with the raw base. Source repair cannot fit through that boundary.

The replacement adapter must return an absolute canonical surface and corrected-authority evidence. The next release must publish a corrected ordinary format-1 base and pin the unchanged format-2 negative overlay to it.

### 4.4 Heightfield scope

The Suur Taevaskoda reference contains a vertical/undercut sandstone face. A single-valued `H(E,N)` cannot represent it. This spec covers:

- cliff top and shoulder where single-valued;
- bank, shoreline, talus, slopes, ledges that remain single-valued;
- source errors and microtopography in those supported regions.

It does not cover the vertical wall, caves, or undercuts. Do not add a structural-cluster format or runtime path in this pass.

## 5. Research Verdict

No audited paper or repository solves realistic Estonia-wide `1 m -> 0.0625 m` ground synthesis.

| Evidence family | What transfers | What does not transfer |
|---|---|---|
| DDNM and inverse-problem diffusion | explicit linear-observation consistency | a Maa-amet degradation model or terrain prior |
| MultiDiffusion/infinite tiling | overlap consensus and coordinate-stable generation ideas | slope, drainage, seam, or physical-continuity guarantees |
| Guérin 2016 sparse amplification | paired low/high exemplar transfer for similar terrains | national coherence, source repair, unseen regimes, guaranteed small-scale realism |
| Argudo 2017 multilayer exemplars | contextual joint matching | centimeter Estonia transfer or complete structured networks |
| Argudo 2018 optical FCN | registered imagery can help a 15 m-to-2 m task | native-image detail as ground height; canopy/shadow/water safety |
| GATA/GAN terrain amplification | stochastic sharp terrain variation at a demonstrated macro scale | target data, checkpoint, physical Estonia classes, centimeter output |
| pixel-space terrain diffusion | high-ceiling metric stochastic generation concept | a trained centimeter prior or acceptable national cost |
| process/event methods | causal shape, topology, orientation in supported processes | a universal material generator |
| conditional geostatistics | calibrated covariance and uncertainty | recognizable objects, events, drainage, or multimodal layouts by itself |
| Moore peat data | foreign peat grids exported at 1 cm after interpolation/3 cm filtering, plus measured plot-scale classifications | demonstrated 1 cm effective truth, transferable band error, Estonia, national context, or other regimes |
| Pawlik forest evidence | 0.025 m TLS shows 1 m ALS misses pit-mound form | open target data or national parameters |
| agricultural roughness studies | true scale breaks and management dependence | a ready Estonia target corpus |

The source audit also corrects two inherited errors:

- Argudo et al. 2017 is the multilayer dictionary paper; the orthophoto FCN is Argudo, Chica, and Andujar 2018.
- HuHoLa classifies existing mire relief; it is not a generator.

Consequently:

- no universal diffusion backbone is approved;
- no universal dictionary/process backbone is approved;
- pixel-space conditional diffusion remains the highest-ceiling learned specialist challenger;
- per-regime owner selection must be empirical and target-supported;
- a bounded weak-evidence research screen may now train and visually inspect
  research specialists without declaring a target, winner, production owner, or
  Estonia transfer, under the separate state machine in Section 12.6;
- public raw-data conversion and target/analogue qualification are release
  dependencies, not optional polish; paid acquisition is out of scope.

### 5.1 Architecture evidence trace

The following is the minimum primary-source trace for the architecture decision.
The normalized machine ledger in Section 23 remains authoritative for artifact
hashes and provenance. "Pixel-space conditional diffusion is the highest-ceiling
learned challenger" is a project architecture judgment, not a result established
by an Estonia target-scale paper.

| Primary work | Demonstrated scale, data, degradation, and evaluation | Code/data/checkpoint and access | Direct finding | Limitation and permitted transfer |
|---|---|---|---|---|
| Eric Guérin, Julie Digne, Eric Galin, Adrien Peytavie (2016), *Sparse Representation of Terrains for Procedural Modeling* | Paired low/high terrain patches; results up to `8192 x 8192`; exemplar and input must contain similar terrain; timing and visual evaluation | Full paper local; linked MATLAB repository snapshot revision `5b83d65315e9845401df78f6a104c21f9bd473d5`, MIT, officialness not yet verified and code not yet read line-by-line; no national dataset or learned checkpoint | Coupled sparse coefficients can transfer exemplar detail while conditioning on low terrain | It does not supply material/geology semantics or guaranteed geomorphological/small-scale coherence; use only as a contextual-exemplar challenger/component |
| Oscar Argudo, Albert Chica, Carlos Andujar (2018), *Terrain Super-resolution through Aerial Imagery and Fully Convolutional Networks* | `15 m` DEM plus `1 m` orthophoto to `2 m` DEM on selected Pyrenees/Tyrol alpine terrain, roughly `400 m` training tiles; Euclidean height loss and metre-scale errors | Full paper local; no official code/checkpoint found; paper access does not license an implementation | Registered optical imagery can improve structural terrain inference | Vegetation and shadows produce false height, and the output scale is far above 6.25 cm; transfer only the masked optical-conditioning hypothesis |
| Yinhuai Wang, Jiwen Yu, Jian Zhang (2023), *Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model* | Natural-image inverse problems with explicit linear operator `A`; super-resolution uses synthetic average pooling/replication on ImageNet/CelebA | Full paper local; linked repository snapshot revision `00b58eac7843a4c99114fd8fa42da7aa2b6808af`, MIT, officialness not yet verified and code not yet read line-by-line; pretrained priors are natural-image, not terrain | Range/null-space projection can enforce a known linear observation exactly | Maa-amet is not known block averaging and contains correctable error; transfer soft observation-likelihood reasoning only, never exact raw-DTM preservation or its image prior |
| Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel (2023), *MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation* | Overlapping Stable Diffusion image/latent crops fused through weighted least-squares pixel consensus; panorama/control evaluation | Full paper local; linked repository snapshot revision `69bcdcef437dfdbf48c53624d6bf6f397b5f4894`, officialness not yet verified and code not yet read line-by-line; audited snapshot has no license file | Per-step overlap consensus can reconcile local denoiser predictions | It guarantees neither height gradients, normals, curvature, drainage nor hierarchy; transfer only as a terrain-native tiling challenger after validation |
| Hugo Schott, Eric Galin, Eric Guérin, Axel Paris, Adrien Peytavie (2024), *Terrain Amplification Using Multi-scale Erosion* | Hydraulic/thermal amplification through multiscale refinement to `8192 x 8192`; supported terrain examples compared with selected learned/procedural methods | Full author manuscript local; implementation and license must be normalized before reuse; no Estonia target data/checkpoint | Multiscale process simulation can add coherent erosion/deposition and drainage structure | It is one process family with whole-map dependency and boundary/cost risk; use only as an erodible-slope/process challenger |
| Paul A. Moore, Maxwell C. Lukenbach, Dan K. Thompson, Nick Kettridge, Gustaf Granath, James M. Waddington (2019), *Assessing the Peatland Hummock-Hollow Classification Framework Using High-Resolution Elevation Models* | 68 foreign clipped-moss plots, `3.2-10.1 m2`; `0.01 m` grid with `0.03 m` mean filter; lab RMSE below `0.01 m`, field median absolute difference `0.018 m`; plot/site morphology statistics | Full CC BY 4.0 paper and data local; archive SHA-256 `044413bb87171776b409172d29fbde16341b228dc43b694cffe9f02a88640a67`; no generative checkpoint | Real peat microforms are multi-class and measurable | Only `309.1387 m2`, disconnected, foreign, with vascular vegetation clipped; transfer only plot-local foreign method validation under Section 12.7 |
| Philip Marzahn, Moritz Seidel, Ralf Ludwig (2012), *Decomposing Dual Scale Soil Surface Roughness for Microwave Remote Sensing Applications* | Photogrammetric `2 mm` grids over `6-22 m2` worked fields; geostatistical decomposition separates seedbed-row and wheel-track scales | Full CC BY 3.0 paper local; no production geometry dataset/checkpoint in the ledger | Agricultural roughness has distinct operation-caused scales, directions, and states | Not Estonia synthesis evidence; transfer causal decomposition, descriptors, and acquisition requirements for agriculture challengers |
| Alexander Goslin (2026), *InfiniteDiffusion: Bridging Learned Fidelity and Procedural Utility for Open-World Terrain Generation* | MERIT/ETOPO/climate near `90 m`; `512 x 512` 90 m patches; hierarchical diffusion and overlap fusion; FID/latency/visual evaluation; roughly two weeks on RTX 3090 Ti | Full preprint local; linked terrain repository snapshot revision `82a0431281f21a6ec3d691a12ee61525de5b0790`, officialness not yet verified and code not yet read line-by-line; data/checkpoint/license state must be normalized | Coordinate/seed-stable overlapping diffusion is a credible unbounded-generation mechanism | Demonstrated geometry is tens-of-metres scale and runtime generation violates this project law; transfer only deterministic cook-side domain/tiling concepts, not weights or a quality claim |

The zero-budget public-data amendment adds these primary-source-audited raw
observation candidates; `R+` means high-priority conversion input, never ready
target truth:

| Public source | Verified value | Status and limit |
|---|---|---|
| Hovi et al. 2024 forest TLS, DOI `10.23729/9a8d90cd-73e2-438d-9230-94e10e61adc9` | 13 Järvselja Estonia plus 28 Hyytiälä Finland plots under one Leica P40 protocol; full scans/transforms and 2 cm-thinned LAZ; CC BY 4.0 | `R+`; strongest Estonia/neighbor paired forest source, but surface semantics/support/error still require conversion |
| Evo 2024 TLS, DOI `10.23729/fd-5a800660-8bd8-35ef-ac9f-ac5c45f7fa77` | 55 leaf-off, nine-station 32 m plots; 39.3 GiB; CC BY 4.0 | `R+`; compact boreal conversion corpus, not a pure ground class |
| FORWARD Sweden, DOI `10.71540/89rs-s553` | rough boulder/till forest, high-density helicopter LiDAR, UAV and machine-operation records; 1.07 TiB; CC BY 4.0 | `R+`; selectively download strata; total-return density is not ground support |
| Biała Góra, DOI `10.18150/BHH1RC` | ten raw UAV-LiDAR southern-Baltic cliff epochs, 27.402 GB; CC BY 4.0 | raw coastal/till/process corpus; unclassified and no published independent target error |
| Taevaskoda tile `444679` | eight official ALS epochs; audited 2023 file has 5,234,018 points over 1 km2 and broad derivative rights | immediate structural-repair evidence only; far too sparse for 6.25 cm morphology |

The verified 2015/2024 Taevaskoda, nearby outcrop, and Selisoo survey archives are
permission leads rather than public dependencies. Full findings and canonical
links are in `review/NORDIC-PUBLIC-TARGET-DATA-HUNT.md`,
`review/BALTIC-PUBLIC-TARGET-DATA-HUNT.md`, and
`review/TAEVASKODA-ESTONIA-PUBLIC-SCAN-HUNT.md`.

No table row authorizes a national owner. Full-paper and official-code review is
mandatory before a consequential method change; difficult synthesis and
scientific/visual judgment use `sol` at high effort when selectable, while routine
ingestion, harness, and testing work uses normal effort.

## 6. Decisive Architecture

Adopt the **Evidence-Gated Typed Hybrid Surface** architecture.

### 6.1 Ownership

| Component | Sole owner | May not do |
|---|---|---|
| source and condition registry | input assembler | invent missing categories or hide support scale/date |
| mapped/direct source repair | deterministic typed reconstructor | hallucinate unresolved detail |
| ambiguous artifact assistance | calibrated learned detector/regressor with abstention | silently override direct constraints |
| regime eligibility and transitions | deterministic support-domain graph | pick a style from one broad class |
| unresolved morphology | the validated specialist registered for that regime | run in unsupported cells or alter protected structure |
| final surface | deterministic constrained composer | average conflicting experts without ownership |
| packed hierarchy | existing height cook/release path | project back to known raw errors |
| runtime | existing packed-height path | synthesize, correct, or branch on generator provenance |

This is one architecture, not a decision menu. Different specialist families may win different physical regimes because the phenomena and evidence differ. The fixed owner-selection rule, not taste or implementation convenience, decides.

### 6.2 Processing order

1. Bind raw sources, dates, licenses, hashes, coverage, registration, and uncertainty.
2. Build a full-fidelity cook-only condition fabric.
3. Reconstruct the corrected structural authority.
4. Build overlapping physical regime/support domains and release status.
5. Generate long-range process/event organization inside its physical domain.
6. Generate target-supported unresolved within-form morphology.
7. Compose without overlapping ownership, then reapply protected/forbidden constraints.
8. Emit one absolute 0.0625 m master surface.
9. Quantize/decode LOD -2 and derive every accepted parent from decoded children.
10. Publish corrected base plus unchanged negative-rung overlay transactionally.

## 7. Artifact And API Contracts

The implementation may use dataclasses or equivalent typed records, but these semantics are mandatory.

### 7.1 `RawObservation`

Contains:

- absolute EH2000 DTM samples;
- EPSG:3301 bounds and sample convention;
- source tile, acquisition/vintage, hash, no-data, interpolation, and classification metadata;
- point-support or source-quality evidence where available;
- derived DTM gradient/curvature only as diagnostics, never as separate truth.

### 7.2 `ConditionField`

Every field carries:

- value array or vector geometry;
- CRS and transform;
- native/effective support scale;
- acquisition/reference date;
- source and license identity;
- availability mask;
- confidence/uncertainty;
- allowed causal roles;
- resampling/interpolation method.

Zero is never overloaded as both a real value and missing data.

### 7.3 `ObservationEvidence`

This immutable pre-repair artifact contains registered measured-support,
interpolation, water/shore, occlusion, canopy, building/object, seam, and
suspected-error fields; correction confidence; protected-feature counterevidence;
explicit unknown state; and per-field source provenance. Repair consumes this
artifact. `RawObservation` is never mutated to hide an accepted correction.

### 7.4 `ResearchSurfaceEvidence`

This artifact is permitted only for the research branch in Sections 9.6-9.8. It
contains metric surface hypotheses derived from immutable raw observations, never
target truth. Every sample records:

- source point/view identities or a reconstructable index;
- `height_m_f64` where a single-valued hypothesis exists;
- direct-support distance, footprint, view count, cross-view disagreement, and
  interpolation distance;
- source-calibrated semantic probabilities, `p_unknown`,
  `p_semantic_subclass_unknown`, heightfield-valid probability, dynamic/water
  probability, and explicit forbidden state;
- separate epistemic and repeatability uncertainty; neither is called total error;
- `total_surface_error`, which is `null` for weak geometry without independent
  total-error evidence;
- per-band research eligibility and its evidence;
- train/development/audit role and leakage group;
- evidence tier, recipe, code, environment, raw-artifact, and license hashes.

Forbidden, non-heightfield, interpolated, or hard ownership-unknown samples carry
no training/evaluation weight. For `R1_weak_surface` only, a directly observed
disjoint-view-consensus sheet may carry bounded weak-geometry weight while its
semantic subclass remains unknown, provided Section 9.6's hard exclusions and
confidence operator pass. It records `p_semantic_subclass_unknown=1` and
`total_surface_error=null`; neither field becomes evidence of ground, target
truth, total error, or transfer.

### 7.5 `StructuralAuthority`

Contains:

- absolute corrected 0.25 m surface over complete affected LOD0 chunks plus support;
- raw-observation confidence;
- correction delta and correction class;
- evidence provenance per corrected area;
- protected-feature mask;
- water surface, terrain bed, shoreline, bank side, and escarpment as separate typed fields;
- forbidden-residual and non-heightfield masks;
- uncertainty and abstention mask.

It is a deterministic recipe artifact.

### 7.6 `RegimeState`

Contains overlapping factors, not one label:

- substrate and surficial deposit;
- complete soil mixture/profile state;
- hydrology and terrain position;
- land cover plus actual management/disturbance state;
- exposure and vegetation/visibility;
- physical-domain identifier;
- compatible and exclusive phenomena;
- transition support and confidence;
- per-band release status: `unsupported`, `research`, `pilot`, or `national`.

### 7.7 `SpecialistOutput`

Every specialist returns:

- metric absolute height on the canonical grid, or a residual whose record names
  the exact corrected-authority digest, canonical grid, analysis band, and
  reconstruction operator;
- support/valid mask;
- claimed phenomena and physical scale;
- uncertainty or ensemble evidence;
- long-range state identity;
- target-registry, model/process/exemplar, code, environment, and stochastic identities;
- explicit abstention/rejection reason, correction confidence where applicable,
  and protected-feature check result;
- diagnostics for morphology coverage, repetition, and boundary behavior.
- authority class `research_only` or `production_candidate`; research output also
  names its research-candidate state, weak-evidence bundle, eligible bands, and
  immutable preview restriction.

An unnamed residual or a delta against the raw DTM is invalid.

### 7.8 `FinalMaster`

Contains:

- absolute float32 EH2000 height at 0.0625 m;
- complete model and hierarchy support;
- deterministic provenance;
- protected/forbidden constraint audit;
- expert ownership map used only as cook evidence;
- authority class and, for research, a null production-owner field plus explicit
  `target_truth=false`, `estonia_transfer=none`, and `latest_eligible=false`;
- no runtime parameters.

The cook crops storage chunks from this reconciled master. It never asks each storage chunk to generate itself.

## 8. Conditioning Fabric

The synthesis cook reads full-resolution source data directly or from a versioned cook-only cache. It must not rely on the lossy runtime biome/soil planes as its sole evidence.

### 8.1 Existing mandatory inputs

- Maa- ja Ruumiamet 1 m DTM as preferred ground observation;
- 1 m nDSM and CHM as object/vegetation/visibility evidence, never replacement ground;
- national ETAK GeoPackage;
- national Mullastikukaart source vectors;
- forest registry and available management/stand state;
- current water, road, building, land-cover, and terrain processing outputs where needed for dependency checks.

Derive registered multiscale conditions at fixed physical supports of `1, 4, 16,
64, and 256 m` where source resolution permits: gradient magnitude, aspect,
profile/plan curvature, topographic position, contributing area, flow direction,
wetness proxy, and signed distance plus side to water, shoreline, bank,
escarpment, ditch, road, and other typed structure. Every derived field records
its source authority and effective support. It diagnoses or conditions morphology;
it is not new truth.

### 8.2 Soil contract

Preserve at minimum:

- `Sif1..Sif4` and `Osa1..Osa4`;
- `Loimis1` and `Loimis2` layered full textures;
- `Lihtloimis`;
- `Huumus`;
- `Kivisus`;
- `Boniteet`;
- original text plus parsed normalized representation;
- polygon identity, support scale, and boundary uncertainty.

The current five-plane `process/soil.py` output discards mixture weights, full layered textures, and humus. It may remain for runtime rendering, but it is not the synthesis condition contract.

At the review point the source mixture is `LkI / 70` plus `L(k)I / 30`, with layered `l90-100/ls1`, simplified `l/ls`, humus `0/15-20 2₁/6-8`, and boniteet `37`. A generator that sees only `LkI` is using incomplete evidence.

### 8.3 ETAK typed structure

Consume, where present:

- water polygons and river centerlines;
- explicit shoreline and shoreline type;
- natural escarpment/slope lines;
- landform features;
- ditches and drainage;
- road/track classes;
- quarries, cuts, fill, and disturbed ground;
- buildings/objects;
- wetlands and peat fields;
- mapped boulders;
- land-cover polygons.

ETAK geometry constrains structure within its mapping uncertainty. Coarse polygon edges must not print a hard relief seam.

At `58.107506 N, 27.050242 E` (EPSG:3301 `679763.082, 6444796.565`; game `311123.082, 190723.435`), ETAK places:

- a natural shoreline escarpment 5.32 m away;
- the Ahja water polygon boundary 1.85 m away;
- an explicit shoreline 1.85 m away;
- the river centerline 15.69 m away.

These are direct reasons to reconstruct a smooth shoreline instead of retaining a 1 m staircase.

### 8.4 Geology and geomorphology

Use:

- EGT 1:50,000 bedrock, surficial, and geomorphology packages with explicit coverage masks;
- EGT 1:200,000 national bedrock and surficial fallback at low spatial authority;
- map scale, source date, code dictionary, and confidence.

The 1:50,000 packages do not cover Taevaskoda; the nearest mapped polygon boundary is about 7.05 km away. Missing fine geology is not a class. The 1:200,000 bedrock fallback at the point is Burtnieki Formation, sandstone with siltstone/clay interbeds. It changes a broad prior, not local feature placement.

### 8.5 Orthophoto

Fetch dated RGB and CIR for research and likely production. Use it for:

- visible shore/channel alignment;
- exposed material and outcrop cues;
- crop/tillage/track direction and recent disturbance;
- drainage and erosion structure;
- registration and condition-state checks.

Never use raw intensity, shadow, canopy, object edges, reflection, waves, snow, or albedo texture as displacement. Every optical condition requires water, canopy, object, shadow, season/snow, registration, and temporal-confidence masks.

Orthophoto use must survive an ablation against the same model without imagery. A model that merely embosses image edges fails.

Also require shuffled-sheet, deliberate registration perturbation, date-mismatch,
coverage-boundary, canopy, water, shadow, and no-image tests. A missing image has
`available=false`; it never becomes a zero-valued image confused with observed
bare ground.

Every national fetcher binds the official product/manifest URL, source schema and
code legend, CRS, capture/snapshot date, license/attribution, archive size and
hash, cache path, retry/resume state, and extractor version. Use `.part` downloads
and atomic rename. Schema/code drift and missing required attribution fail closed.
Use manifest-selected cache keys under
`data/in/orthophoto/<sheet>/<product>/<capture-date>/` and
`data/in/geology/<scale>/<snapshot-date>/`; never guess current input from a
filename.

### 8.6 Condition transition policy

- Map boundaries select changing distributions; they are not centimeter relief edges.
- Mixture weights vary through evidence-uncertainty collars or physical transition domains.
- Long-range state uses watersheds, connected mire units, forest stands/event fields, managed parcels, shore reaches, or outcrops as appropriate.
- Incompatible regimes are mutually exclusive.
- Compatible events may overlap only when scale and ownership are distinct.
- A condition with no target-supported expert yields abstention.

## 9. Public Target Data And Qualification

### 9.1 Current readiness

No audited source yet supplies a ready production target height surface. The
zero-purchase hunt nevertheless found substantial open raw observations that the
earlier inventory missed:

- Hovi 2024: 13 Järvselja, Estonia and 28 Hyytiälä, Finland forest TLS plots under
  one protocol, with full scans/transforms and selectable 2 cm-thinned LAZ;
- Evo 2024: 55 leaf-off nine-station Finnish TLS plots, 39.3 GiB total;
- FORWARD: Swedish high-density boulder/till forest and machine-disturbance data,
  selectively downloadable from a 1.07 TiB release;
- Biała Góra: ten open UAV-LiDAR Baltic coastal-cliff epochs, 27.402 GB compressed;
- ForestSemantic-MS: six manually annotated Finnish boreal clouds;
- Stordalen: one 5 cm subarctic peat DSM exemplar;
- eight open ALS epochs at exact Taevaskoda tile `444679`, including audited 2023
  LAZ SHA-256 `9c50c123f14841c717d0d123d2d08061b87a51baf6d806ef269fabf1036a9fb7`.

The exact 2015/2024 Suur Taevaskoda true-3D campaigns, Väike Taevaskoda/Härma
models, and Selisoo bog SfM campaigns are verified but not currently downloadable
with usable geometry rights. A no-cost permission request is allowed; planning may
not depend on success. Public raw observations eliminate paid acquisition as the
active path, but they still require surface extraction, support/error measurement,
semantics, and transfer qualification before becoming targets.

### 9.2 Production target truth gate

A production target must provide:

1. measured support at no worse than 0.03-0.05 m after ground/surface classification;
2. independent horizontal and vertical error distributions small relative to the claimed band's energy;
3. per-cell point/support, interpolation, confidence, and visibility masks;
4. explicit surface semantics: mineral soil, peat/moss, roots, clasts, litter, deadwood, vegetation, water, and objects;
5. registration to the temporally closest DTM, orthophoto, ETAK, soil, geology, hydrology, and management state;
6. raw data and immutable processing provenance;
7. a license covering training, derived artifacts, and intended model distribution;
8. complete site/campaign identity for geographic holdout.

It also records sensor/view/incidence geometry, human-QA masks, moisture, weather,
season, and explicit include/exclude decisions for stable roots, litter, crops,
deadwood, clasts, and other ambiguous surface semantics.

Nominal export pixel size and total point density are insufficient.

### 9.3 Minimum release gate per regime

Before a regime emits finest-band geometry, require at least three geographically
independent qualified sites across at least two campaigns, with two development
sites and one untouched site-level holdout. Sites may be in Estonia or a physically
analogous neighboring Baltic/Nordic region. Geographic proximity alone is not
evidence: a checked `physical-analogue-v1` dossier must compare substrate and soil
profile, climate/freeze-thaw, hydrology, vegetation/organic surface, land use and
disturbance, forming process, morphology scale, source observation process, and
the Estonia condition envelope. Any unmatched factor becomes OOD and abstains.

When a qualified Estonia target exists, keep it as the blind holdout. When none
exists, an all-foreign owner is limited to `pilot` with transfer ceiling
`neighbor_analogue_only`; it requires
at least two independently acquired neighboring-region sites, a held-out campaign,
condition-stratified OOD bounds over the claimed Estonia extent, fixed Estonia
visual/structural sentinels, and explicit user acceptance. It cannot silently
become national. Patches from one site are never independent sites. Hold out sensor
and campaign when feasible. Taevaskoda remains outside training, hyperparameter
selection, and candidate selection.

Before downloading/training, check in a zero-budget
`PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md` plus machine-readable condition and
target schemas. It freezes exact public artifact/file selections, regime IDs and
condition strata, physical-analogue dossiers, context/microplot counts, conversion
and semantic policy, target transfer/error threshold, power/stopping rule, OOD
bound, campaign holdouts, licenses, QA, storage ceiling, and selective-download
order. Paid capture, proprietary data, new hardware, contractors, and cloud-scale
services are excluded unless the user explicitly changes the project budget.

### 9.4 Nested observation requirements

Each qualifying public site must contain or permit derivation of:

- a contiguous context survey covering physical organization, transitions, drainage, and the real inference halo;
- nested target-scale microplots or dense ground returns for the 0.25-0.0625 m band.

Acceptable source modalities include:

- close-range/low-altitude SfM for visible bare soil, fields, peat, rock, and beaches with metric control;
- multi-position TLS or validated MLS for forest/scrub;
- oblique TLS/SfM plus nadir context for cliffs/talus, preserving true 3D invalid-heightfield masks;
- LiDAR plus ground checks for wetland and shore;
- repeated dated observations for agriculture and recent disturbance.

Lowest-point filtering alone is forbidden as target creation.

### 9.5 Target registry

Create a versioned machine-readable registry. Every record includes:

- `site_id`, `campaign_id`, country, EPSG bounds, date, season;
- regime factors and management/disturbance state;
- raw and derived artifact hashes;
- sample support, effective resolution, vertical/horizontal error;
- valid/interpolated/visibility/semantic masks;
- paired condition-source identities;
- license and attribution;
- role: `production_target`, `exemplar`, `raw_candidate`, or `calibration_only`;
- train/development/blind-test assignment;
- transfer restrictions.

Foreign data may pretrain, provide exemplars, and qualify a `pilot` with transfer
ceiling `neighbor_analogue_only` under Section 9.3. It may not fill unmatched Estonia
conditions, erase OOD/abstention, or claim national transfer from geographic
similarity alone.

The production gate in Sections 9.2-9.5 is unchanged by the research authorization
below. Research evidence never upgrades a target-registry role, satisfies an
independent-error field, or supplies one of Section 9.3's qualified sites.

### 9.6 Research evidence and surface tiers

Research may proceed without production-grade survey truth only through these
explicit tiers. A record carries one tier per role and band; evidence does not
inherit the strongest tier of another artifact from the same dataset.

| Tier | Meaning | Permitted use | Forbidden claim |
|---|---|---|---|
| `R0_raw` | immutable licensed observations with known site/campaign identity but no accepted surface | converter development, support diagnosis, unsupervised representation learning | height supervision, model comparison as truth |
| `R1_weak_surface` | a single-valued probabilistic surface hypothesis with reconstructable raw support and explicit unknown/forbidden masks, but incomplete semantics and/or total error | confidence-weighted research training and development diagnostics in an eligible band | target truth, real-vs-real non-inferiority, production owner, Estonia transfer |
| `R2_research_audit` | an artifact held out from the component it audits, such as semantic labels or an independent sensor/campaign surface hypothesis | frozen research audit after the relevant recipe/threshold freeze | independent metric truth unless it separately passes Sections 9.2-9.5 |
| `qualified_target_B1/B2` | a band-qualified target from the public-target protocol | production bakeoff for the exact qualified band and stratum | broader bands, regimes, or geography than its checked claim |

`R1` and `R2` are evidence tiers, not accuracy ranks. Cross-view agreement,
repeatability, or a second sensor can expose instability without bounding total
surface error. An `R2` failure rejects or narrows a research candidate; an `R2`
pass does not promote it to `qualified_target_B1/B2`.

Every weak surface validates as `microtopography-research-surface/1.0.0` and binds
the `ResearchSurfaceEvidence` fields in Section 7.4. It carries separate
probabilities for each included/excluded semantic class, static attachment,
heightfield validity, dynamic/water contamination, and observation support.
Source semantic probabilities must be calibrated on the named development source
and checked on its frozen audit split; they are not assumed calibrated after
transfer to Hovi/Evo. Cross-source semantic output is negative evidence only: a
calibrated forbidden detection may reject a cell, but absence of a detection never
positively labels it or contributes confidence. A fired transfer-shift detector or
missing required feature is a hard ownership unknown and sets `p_unknown=1`.

For `R1_weak_surface` only, uncertainty among mineral soil, persistent organic
surface, ground-attached moss, settled litter, ground-bonded root, and embedded
clast does not itself hard-reject a repeatable observed sheet. Such a sample keeps
`p_semantic_subclass_unknown=1`; `p_unknown` instead records uncertainty that the
sheet is usable weak geometry after hard exclusions. Compute it without a tuned
cutoff, separately for each fixed 8 m context `C` and analysis band:

```text
M_C = cells directly observed by both frozen view groups after only external
      water/dynamic/non-heightfield masks, before sheet selection
G_C = subset of M_C with one connected unique sheet, no interpolation, complete
      F4 support, common-feature in-domain, and no positive forbidden/object/
      multi-sheet evidence
c_area = area(G_C)/area(M_C), or 0 when area(M_C)=0

b_A,b_B = independent F4 band residuals on the common eroded support of G_C
S_C = max(0, mean(b_A*b_B))
N_C = 0.5*mean((b_A-b_B)^2)
c_band = S_C/(S_C+N_C), or 0 when S_C+N_C=0

d_i = 0.5*(b_A[i]-b_B[i])^2
c_local[i] = N_C/(N_C+d_i), with (N_C,d_i)=(0,0) defined as 1
c_view[i] = min(n_A[i],n_B[i])/max(n_A[i],n_B[i]), or 0 when both are 0
w_i = min(c_area,c_band,c_local[i],c_view[i])
p_unknown[i] = 1-w_i
```

These are observed evidence fractions combined by a conservative bottleneck, not
multiplied independent probabilities. Any interpolation, positive forbidden,
water/dynamic, non-heightfield, multi-sheet, missing-feature, transfer-shift, or
hard ownership-unknown state sets `w_i=0` and `p_unknown=1`. Otherwise the robust
weak-supervision loss weight is exactly `w_i`; there is no `p_unknown` threshold.
Persist `M_C`, `G_C`, every factor and reason bit, `p_semantic_subclass_unknown=1`,
and `total_surface_error=null`. Publisher plot cover fractions describe
composition only and may not label cells. No classifier may force an unrepresented
or ambiguous class to its nearest known class.

Each band receives exactly one research eligibility state:

- `research_ineligible`: no directly supported, heightfield-valid surface or no
  semantically admissible cells for that band; it supplies no geometry loss;
- `research_diagnostic`: supported cells exist, but view independence, band
  coherence, semantic transfer, valid-area coverage, or repeatability is
  insufficient for supervision; diagnostics and abstention tests only;
- `research_trainable`: a frozen manifest demonstrates direct non-interpolated
  support, at least two contributing view groups where available, positive
  cross-view band coherence and signal above within-campaign disagreement,
  calibrated forbidden-class rejection, and the frozen operator above. Its
  first-forest-bundle gate uses a 4 m x 4 m input/support window, central 2 m x
  2 m valid target, batch size 8, and complete partition halo
  `max(1.875 m two-stage F4 analysis halo,1.0 m model input-to-valid halo)=1.875 m`.
  It requires at least 32 m2 weighted effective train area, Kish effective count
  `(sum(w)^2/sum(w^2)) >= 8` over non-overlapping valid windows, and at least one
  complete nonzero-weight valid window in each development and internal-audit
  split. This first-forest-bundle amendment authorizes only B1 research
  trainability; B2 remains `research_ineligible` and structural-only until a new
  preregistration explicitly qualifies it;
- `production_qualified`: only the `qualified_target_B1/B2` result from Sections 9.2-9.5.

The manifest reports eligibility independently for `B1` and `B2`, including valid
area, window count, source groups, coherence, disagreement, confidence calibration,
and every exclusion. The research threshold is a feasibility screen, not the
production transfer/error test. Missing independent XY/Z error, acquisition MTF,
or total-error bounds remains prominently `unknown` even when a band is
`research_trainable`.

### 9.7 First forest research bundle

The first and only authorized bundle for the initial weak-evidence specialist
screen is `forest-weak-research-bundle/1`. Its roles are fixed:

| Artifact | Fixed role | Explicit limit |
|---|---|---|
| Hovi `HY_SPRUCE4` full scans | primary `R1_weak_surface` geometry supervisor | only directly supported probabilistic surface cells may train; no target truth or site independence claim |
| Evo plots `1086` and `1065` | `R2_research_audit` independent-sensor/campaign geometry evidence | evaluate frozen cross-campaign geometry behavior; existing converter inspection means they are not blind metric targets |
| ForestSemantic-MS `train1`-`train4` | semantic model development | semantics only; no height, absolute registration, or Hovi-truth claim |
| ForestSemantic-MS `test1`-`test2` | frozen `R2_research_audit` of semantic behavior | never train, select thresholds, or tune after inspection; its sparse classes and source shift remain reported |
| Hovi `HY_PINE2` | stress-only OOD/abstention case | no training, target, owner-selection, or positive-transfer credit; correct abstention is a pass |
| Hovi Järvselja `JS_SPRUCE1` | sealed Estonia research generalization audit | no byte/geometry inspection until the complete research recipe is frozen; opening it consumes its untouched status for any later changed recipe |
| historical LUKE artifacts | excluded | no training, validation, exemplar bank, morphology baseline, or accepted-owner comparison |
| Suur Taevaskoda | fixed visual/structural sentinel | excluded from training, thresholds, hyperparameters, candidate selection, and scene-specific repair; never metric morphology truth |

No other Hovi, Evo, FORWARD, Biała Góra, Moore, OPARA, or replacement-search
artifact may silently enter this bundle. Adding or changing an artifact, role,
surface contract, semantic class, split, or band creates a new bundle and recipe.

Before model training, freeze a research preregistration containing the exact raw
and derived hashes, surface contract, included/excluded semantics, ForestSemantic
feature/label mapping, probability calibration, unknown policy, B1/B2 eligibility,
spatial blocks, input tensors, candidate families/configurations, budgets, seeds,
metrics, candidate state transitions, pack extent, cameras, and rejection rules.
Development may use only ForestSemantic `train1`-`train4` and spatially separated
`HY_SPRUCE4` training/development blocks. Scan/view-derived samples from one block
remain one leakage group; overlapping receptive fields, augmentations, thinned
clouds, and full scans inherit that group.

Freeze semantic thresholds before opening ForestSemantic `test1`-`test2`. Freeze
the complete synthesis recipe, checkpoint, seed, candidate choice, and visual rule
before opening `JS_SPRUCE1` or evaluating the final Evo/HY_PINE2 audits. After any
frozen audit result is viewed, no tuning is allowed. A changed candidate requires
a genuinely uninspected audit group; reusing the same data with a new seed or
threshold is leakage. If no such group remains, the result stays development-only.

### 9.8 Research anti-memorization and generalization gate

Every learned, exemplar, or dictionary research specialist must:

- hold out contiguous physical blocks larger than its complete receptive and
  fusion halo; random point/window splits are forbidden;
- exclude absolute plot IDs and local coordinates unless a preregistered ablation
  proves they do not encode site identity; world coordinates are never available
  in these local foreign plots as a shortcut;
- report nearest training surface/feature patch, exact/near duplicate hashes,
  exemplar/atom attribution, effective sample count, and spatial reuse heatmaps;
- compare training-block, held-out `HY_SPRUCE4`, Evo cross-campaign, HY_PINE2
  stress, and sealed-Järvselja behavior without pooling them into one score;
- pass coordinate, condition, semantic-confidence, and stochastic-seed ablations;
- reject copied relief, repeated stamps, phase locking, texture leakage, or
  confidence that remains high on HY_PINE2/other declared OOD evidence;
- preserve a corrected-only control and show that any visible addition survives
  the actual packing and cluster/DAG path.

Järvselja and Taevaskoda are generalization sentinels, not optimization data. A
candidate that fails after either is viewed is `research_rejected`; the scene may
not be patched and the same sentinel may not be used to tune a replacement. A
candidate that abstains on unsupported cells is preferable to generic coverage.

### 9.9 Erodible-slope R0 research bundle

`erodible-slope-research-bundle/1` authorizes one deterministic process/event
challenger for `fluvial.rill_gully_seep_spring` research development on dry,
single-valued erodible mineral bank and slope surfaces. It does not qualify a
target, supply measured morphology truth, establish Estonia transfer, select a
production owner, or change the regime row from `unsupported`.
Sandstone-derived substrate and compatible colluvial toes are conditions only
where the bound soil/geology evidence supports them; neither is a default.

Its fixed roles are:

The replacement-role authority is selection recipe
`aa59aca7a5209b9504be5c3778b6764d6ae6f5c3bf3de26eab967a9274d7365d`
at
`asset-gen/data/work/microtopography/erodible-slope/site-selection/sha256/aa59aca7a5209b9504be5c3778b6764d6ae6f5c3bf3de26eab967a9274d7365d/selection.json`
(file SHA-256
`977b336b65c75c0548be74e18ee1eb16c61b19989e8650597cfdbe152111e359`).
It froze identities from official metadata and exact canonical LOD -2 chunks
before human pixel or detailed source-geometry inspection. Superseded selection
artifacts have no authority.

The condition authority is recipe
`10a5ea53b007f3e6e2e878ccebd70a9712716042139ca35178310655fcb85b8b`
at
`asset-gen/data/work/microtopography/erodible-slope/conditions/sha256/10a5ea53b007f3e6e2e878ccebd70a9712716042139ca35178310655fcb85b8b/bundle.json`
(file SHA-256
`9d417818c90b7e8e4db757a70cd2703df5a2e4045ff54ada9248f700bede38f9`).
No other condition bundle may supply Development A, B, or C inputs to this
research bundle.

| Artifact | Fixed role | Explicit limit |
|---|---|---|
| ETAK escarpment `1826743`, `E680551.76 N6444450.80`, sheet `54481` | Development A | R0 condition-bound process development only |
| ETAK ditch-adjacent slope `1826691`, `E679692.03 N6442784.16`, sheet `54472` | rejected Development B strict-abstention evidence | missing Huumus is unknown; no synthesis, imputation, material default, positive credit, or second Development role |
| ETAK bank slope `9688719`, `E680752.8418668837 N6441183.486937005`, sheet `54481`, LOD -2 chunk `(2438,1518)`, bounds `[680704,6441088,680832,6441216]` | Development C | second R0 condition-development role; exact target line `100.155104 m`, solve-domain coverage `9,713/16,384` (`59.2834%`), target/solve intersection `122/122`, and material support `13,346/16,384` (`81.4575%`) are frozen by the bound condition artifact |
| Orajõgi bank slope `9688685`, `E681429.34 N6443825.32`, sheet `54481` | consumed and disqualified former OOD | inspected before full recipe freeze; never audit, candidate-selection, threshold, morphology, transfer, or generalization evidence |
| ETAK bank slope `9688702`, `E681105.772168 N6442476.398258`, sheet `54481`, LOD -2 chunk `(2441,1508)`, bounds `[681088,6442368,681216,6442496]` | sealed OOD2 abstention stress | still uninspected; bounded abstention is its only allowed pass and it supplies no positive morphology, transfer, target, or generalization credit |
| Schott 2024 paper and pinned public implementation | operator-sequence and ablation reference | no code, constants, boundary conditions, hardness behavior, or scale transfer |
| OPARA R0 negative-form diagnostic `441eaf9194cd...` | qualitative process-existence diagnostic | no width, depth, amplitude, spacing, scale, or geometry transfer |
| Suur Taevaskoda | final visual/structural sentinel | no training, tuning, threshold selection, candidate selection, or scene-specific repair |

Development A and C are one conservative leakage group unless a frozen physical-
domain audit proves disjoint catchments and inference halos. They are never counted
as Section 9.3 independent qualified sites. Development B remains strict-abstention
development evidence only. OOD2 is not a comparable blind positive audit: its only
passing result is bounded abstention wherever its frozen condition predicate or
support is unknown or outside the Development A/C envelope. Orajõgi remains consumed
and may not substitute for OOD2.

Development C's superseded LOD -2 chunk `(2438,1519)`, bounds
`[680704,6440960,680832,6441088]`, and candidate recipe
`98c131d4826761e09746c4bd3f02ab34d423f8d7bb94cca2a79422437a898b52`
are retained only as rejected condition-abstention evidence. That crop contained
only `725/16,384` solve-domain cells and `22/118` target/solve cells; its zero C1
residual is the correct fail-closed result, not a solver failure, morphology
measurement, or basis for changing the solver. OOD2 remains unchanged and sealed.

Before any R0 generation, the machine preregistration must bind the complete soil
mixtures/layers/humus, authoritative decoded 1:200k `Q_Litoloogia_200` and
`Q_Genees_200` domains plus coverage/unknown masks, ETAK geometry/distances,
whole-domain drainage and real outlets, dated masked RGB/CIR, accepted corrected
base, water/dynamic/object/non-heightfield/protected-feature masks, physical
processing domains, code/environment, candidate/control configs, units, event
distributions, budgets, seeds, metrics, thresholds, and rejection rules. A null,
unhashed, inferred, or substituted required input blocks generation; the 1:50k
domains may not replace the 1:200k domains.

The only initial candidates are corrected-only `C0` and one rederived `C1` with
continuous or rotation-qualified routing, canonical world samples, whole-domain
upstream flux and outlets, declared runoff/seep/material fields, conservative
erosion/deposition accounting, typed hard masks, and one absolute master cropped
only after the solve. Development output is R0 descriptive evidence. Development C
may enter C1 only through the exact bound condition artifact above and remains
subject to its frozen domain, outlet, collar, mask, material-support, and invariance
gates.
Before opening OOD2, freeze the complete C1 recipe, seed, condition envelope,
abstention rule, metrics, thresholds, preview extent, cameras, and visual rule.
Any post-OOD2 change terminates that recipe and requires a genuinely untouched
replacement stress site.

A frozen C1 may reach `research_preview_candidate` only after it passes its
preregistered Development A/C safety, topology, conservation, partition, grid-
leakage, typed-form, and packing-survival gates and OOD2 abstention stress.
This permits one immutable `research-microtopography-preview-v1` manifest under
Section 14.6. The preview remains an R0 process hypothesis: attractive output or
an OOD-abstention pass supplies no measured morphology-fit, positive blind-audit,
production, owner, or transfer evidence.

## 10. Typed Structural Reconstruction

Source repair precedes unresolved synthesis and is evaluated separately.

### 10.1 Solver contract

Use absolute EH2000 unknowns on LOD -1 sample centers over each connected repair
component plus an 8 m collar. Union components sharing a hard constraint. Fix the
outer two sample rings to the accepted one-sided/bilinear initial surface; if bad
evidence reaches the ring, expand support. Assemble float64 sparse matrices in
lexicographic row-major order with 0.25 m four-neighbor gradients and a cut-aware
five-point Laplacian. Remove graph edges crossing accepted shore, escarpment, or
other hard barriers.

Constraint priority is: non-heightfield/forbidden masks; control-grade survey or
legally fixed elevation; accepted ETAK water/shore/structure; protected real
cliffs/escarpments/boulders/channels/depressions; trusted DTM; lower-confidence
DTM/interpolation; regularization. Conflicts at the same/higher level reject the
component and record both source IDs; lower priority never averages them away.

Observation rows are exact:

| Class | Operator/action |
|---|---|
| trusted exposed ground | bilinear evaluation at the 1 m DTM center, equal to the mean of its central `2 x 2` LOD -1 samples; normalized Huber loss using source/support-calibrated uncertainty |
| lower-confidence/interpolated | same operator, confidence 0.25; unknown interpolation support has confidence zero and no row |
| no-data, independently confirmed isolated spike/pit, confirmed object/canopy leak | no DTM row; reconstruct from accepted same-side/boundary evidence |
| open water | no DTM ground row; use Section 10.2 and forbid morphology |
| surveyed ground/bathymetry | bilinear point/footprint operator; hard only when the record declares control-grade, otherwise Huber by recorded sigma |
| shoreline corridor | continuous positional barrier/membership evidence with recorded horizontal and temporal uncertainty; no height row unless a co-temporal survey establishes the toe/elevation |
| mapped escarpment | no invented target height; delete crossing graph edges and reconstruct each side independently |
| verified scan strip | add strip bias `b_t` plus paired same-landform rows; Huber data/pair losses and hard `sum b_t=0` within a tile |
| orthophoto edge | boundary proposal only; never a height row |

Never estimate observation noise from `y-median_3x3(y)`: that residual contains
real banks, clasts, roots, rills, and source-scale relief. Every usable observation
instead joins a hashed `repair-calibration-v1` record keyed by source product,
acquisition/flightline, observation class, point/TIN support class, landform
counterexample stratum, and repair class. `sigma_i` comes only from stated
control-survey error, repeat/overlap differences on unchanged same-landform
surfaces, or leave-one-ground-return-out reconstruction error with raw classified
support and control checks. Pair-difference scale is divided by `sqrt(2)` only
when the two acquisitions have comparable independent error. Spatial covariance,
bias, sample count, geography, support distribution, and confidence interval are
stored; one marginal MAD is insufficient. A record with no defensible calibration
is `uncalibrated` and cannot authorize automatic removal or a production
correction. It may retain an observation conservatively or abstain when evidence
conflicts.

Huber uses cutoff 1.345 on the normalized residual. An isolated spike/pit requires
an outlier against a same-side protected-edge-aware neighborhood *and* independent
support such as raw return/TIN geometry, overlap disagreement, object evidence, or
control survey; local shape alone never deletes it. Object/canopy leakage
additionally needs independent building/nDSM/CHM/object support. A verified strip
needs its source/flightline boundary, at least 32 same-landform overlap pairs, and
calibrated pair covariance. Every nominee remains typed as suspected then
accepted/rejected; only accepted evidence removes or changes a row.

For an ETAK escarpment, densify to 0.25 m, fit its normal over 2 m arclength, and
infer high/low side from valid same-class samples 1-3 m away. Require side median
difference above `max(0.20 m,3*pooled_MAD)`; otherwise protect the line with unknown
side. Initialize within 2 m using separate robust affine fits within 3 m, requiring
six points and condition number below `1e4`; otherwise use same-side nearest
bilinear support and mark low confidence. Generic spike detection is forbidden in
that 2 m collar unless higher-priority direct evidence applies.

For free samples minimize:

```text
E(z)=sum_i Huber((O_i*z-y_i)/sigma_i)
    +sum_j ((ell^2*L_j*z)/sigma_ref)^2
```

`sigma_ref` is the median calibrated trusted sigma in the component. There is no
national or tile-global `ell`. Before viewing a review candidate, freeze a hashed
`repair-calibration-manifest.json` that enumerates every calibration AOI, source/
support stratum, repair class, connected-component shape/area/elongation bin,
protected counterexample, observation key, and PRF mask. For each repair class and
support stratum, require at least 30 geographically distributed shape-matched
contiguous pseudo-voids and 30 protected counterexamples outside all blind review
sites. Evaluate `ell in {0.5,1,2,4} m` by restoring the pseudo-voids from their
unchanged exterior evidence, not by hiding scattered pixels. Minimize equal-stratum
median normalized error subject to no protected counterexample exceeding its
predeclared change bound; ties within 1% choose smaller `ell`. If counts, independent
uncertainty, shape support, or counterexample safety are insufficient, that
repair-class/stratum is uncalibrated and automatic correction abstains. The
manifest records selected `ell`, all losses, rejects, and hashes. This is a
class-conditional interpolation hypothesis, never a terrain-roughness control or
evidence that source artifacts are truth.

Solve hard rows `Cz=d` and verified strip biases with float64 Huber IRLS. Initialize
from the one-sided/bilinear surface. At each iteration set
`w=min(1,1.345/max(abs(r),1e-12))` and solve the normalized KKT system with SciPy
MINRES, inverse-clamped primal diagonal plus identity multiplier block as an SPD
preconditioner, diagonal floor `1e-12`, `rtol=1e-10`, maximum 5,000 iterations.
Run at most 20 IRLS steps and stop only after max free-sample change below `1e-5 m`
and relative objective change below `1e-8` for two steps. Define normalized KKT
residual as `norm(K*x-b,2)/max(norm(b,2),1)`.

Reject a component on nonconvergence, nonfinite output, hard error above `1e-4 m`,
normalized KKT residual above `1e-8`, unexplained trusted-observation error above
`max(0.10 m,4*sigma_i)`, unobserved overshoot beyond boundary range by
`max(0.25 m,4*sigma_ref)`, protected-jump change above `max(0.02 m,5%)` outside the
water toe, or a new unsupported drainage dam/depression above 0.05 m. Emit
correction class, confidence, source IDs, protected checks, convergence, and every
rejection/abstention reason.

### 10.2 Water, shore, bank, and bed

Treat these as different surfaces/structures:

- `waterY`, the separately rendered visible water surface;
- packed `height`, which stores dry terrain plus submerged bed;
- `watercover`, the anti-aliased occupancy/shore fraction;
- the shoreline intersection shared by those products;
- terrestrial bank and shoulder;
- mapped escarpment.

All are solved from one versioned evidence snapshot and one continuous
world-coordinate polygon/line authority. ETAK supplies mapped plan geometry, not
a co-temporal surveyed toe or elevation. Each segment carries horizontal map/
registration uncertainty and temporal-state uncertainty. Within that corridor the
solver preserves possible bank discontinuities and reports ambiguity; only a
co-temporal control survey may collapse it to a hard height contact. LOD -2 wet membership uses polygon `covers()` at sample centers. On the
existing 2 m watercover lattice, occupancy is exact polygon/cell intersection area
divided by 4 m2, encoded with ties-to-even rounding of `255*occupancy`; coarser
occupancy is the fixed decoded-child area mean. Never supersample independently per
chunk.

Raster compatibility is a release gate, not an assumption. Rasterize the
continuous authority independently at 0.0625 m for reference. Through the actual
renderer-equivalent bilinear sampler, extract the 2 m `watercover=0.5` contour and
the visible water/terrain intersection; compare both with the continuous line and
with the fine bed wet-mask boundary. Require signed classification agreement at
every fine sample outside the declared uncertainty corridor, p95 contour distance
at most 0.5 m, maximum distance at most 1 m, and no dry gap or wet overlap wider
than 0.5 m on cross-sections every 0.25 m. Every mapped channel whose minimum
accepted width is at least 4 m must remain connected; narrower channels are
reported separately and may not be claimed as passed. Inspect acute bends,
islands, confluences, and one-cell necks. If the existing 2 m representation cannot
pass these limits, fail Stage 1 and open a separate representation review; do not
move the terrain line or call the products identical.

Every robust location here initializes at the median, uses scale
`max(1.4826*MAD,0.02 m)`, updates the weighted mean with
`min(1,1.345/abs((x-mu)/scale))`, and stops at change below `1e-8 m` or 50
iterations; failure abstains. Sea is 0 m EH2000. A DTM value inside a water polygon
is not automatically a water-surface observation. Standing-water fitting may use
DTM/point values only when raw source classification, point/TIN support, return
semantics, and independent QA establish that they measured the visible water
surface rather than a gap interpolation, bottom, bank, vegetation, ice, or noise.
Require 64 qualified samples, a source-support-calibrated uncertainty model, and
robust spread no greater than 0.15 m; otherwise require a compatible official or
surveyed level or abstain.

For each unbranched flowing reach:

1. densify the centerline to 1 m stations;
2. at each station, robust-locate only source-qualified water-surface observations in a 3 m along-channel interval; reject stations with fewer than four samples;
3. apply an 11 m Hampel window and reject beyond `3*1.4826*MAD`;
4. infer downstream only when the robust locations over the first/last 21 m differ by more than `max(0.05 m,2*endpoint_MAD)`;
5. if oriented, apply deterministic pool-adjacent-violators isotonic regression so water never rises downstream; otherwise preserve orientation uncertainty;
6. convolve accepted stations with a normalized positive Gaussian, sigma 4 m, truncated at 4 sigma, nearest-end extension;
7. interpolate along centerline and extend across sections at constant elevation.

A missing span over 20 m, branched/ambiguous topology, or profile inconsistent
with both bank observations by over 0.25 m abstains. Raw rippled water never becomes
fallback. Preserve bank side, shoulder, and escarpment; expand repair support
rather than taper a real structure at storage edges.

Surveyed bathymetry is a typed observation. Where none exists, use only this
conservative rendering envelope:

```text
d = distance inside accepted shoreline
u = clamp(d/1.0 m,0,1)
depth_unknown = 0.10 m*(6*u^5-15*u^4+10*u^3)
bed = waterY-depth_unknown
```

Label it `unknown_bathymetry_render_envelope`, confidence zero, exclude it from
realism statistics, and forbid morphology. It joins the shore without inversion
and has no along-channel detail. If 0.10 m clearance fails the existing renderer,
that is a representation-review failure, not permission to infer deeper relief
from color, reflection, waves, glare, or missing returns.

Unresolved morphology specialists emit no residual on open-water forbidden domains. Reapply that constraint after every learned/process stage.

Compose the accepted bed into structural height before canonical LOD -2
composition and before any parent reduction. Then cook matching `waterY` and
`watercover` from the same snapshot. Any later change to bed, shore, or occupancy
invalidates every mask-affected height artifact, parent window, and apron.

The known Ahja artifact near `E 679788.35, N 6444811.45` is a review case, not a coordinate patch.

### 10.3 Other correction classes

Support typed handling for:

- building/canopy leakage;
- isolated spikes and pits;
- strips, scan seams, and no-data interpolation;
- oversmoothed or bridged ditches/channels;
- paved and unpaved engineered corridors;
- quarry/cut/fill structure;
- protected boulders, depressions, cliffs, and rare landforms.

Every automatic correction class requires protected-feature counterexamples. A smoother result that removes a real cliff, bank, boulder, or depression fails.

### 10.4 Corrected authority extent

The Stage 1 anchor lies in fine parent `(-1,607,372)` and ordinary authority chunk
`(0,151,93)`, but the qualified downstream reach crosses the latter's east edge at
`E=679936` into `(0,152,93)`. Reconstruct both complete corrected LOD0 cores from
the canonical fine structural master. The evidence-driven transient sets are exact:

- LOD -1 reducer set: `cx=604..612`, `cz=372..376` (45 chunks);
- union LOD -2 reducer set: `cx=2416..2452`, `cz=1488..1508` (777 chunks);
- paired 0.25 m authority/baseline input support including the Keys halo:
  `cx=2415..2453`, `cz=1487..1509` (897 chunks);
- review publication core: LOD -1 `(607,372)` and its 16 LOD -2 cores
  `cx=2428..2431`, `cz=1488..1491`;
- review negative support: the east/south/southeast LOD -2 row/column through
  `cx=2432`, `cz=1492`, retained as transient unless the expectation publishes it;
- ordinary corrected LOD0 cores: `(0,151,93)` and `(0,152,93)`.

Then apply the masked hierarchy policy in Section 14.4. The planning artifact must
freeze separate canonical sets for corrected cores, support reads, transient fine
support, exact affected core and payload masks, apron-only replacements, and
affected ancestor windows before staging. The initial replacement candidates are
the west, north, and northwest published neighbors of every changed core, but only
their mask-induced apron samples may change. A qoffset failure is fail-closed and
cannot promote an unrelated core or recursively expand authority. The final
expectation is frozen independently rather than inferred from produced files.

### 10.5 Structural-only finest reconstruction

Outside vector overrides, prolong the corrected 0.25 m authority to 0.0625 m
without pretending to add morphology. Apply tensor-product Keys cubic convolution
with `a=-0.5`. For each 0.25 m parent cell, let `d` be the parent value minus the
mean of its interpolated 4x4 fine samples. For fine phase `u_r=(r+0.5)/4`, use:

```text
b(u)=u^3*(1-u)^3
b_norm(u_r)=b(u_r)/mean_r(b(u_r))
B(r,s)=b_norm(u_r)*b_norm(u_s)
fine[r,s] += d*B(r,s)
```

The bubble has zero value/first derivative at the mathematical parent-cell edge
and discrete 4x4 mean one. It preserves cubic cross-cell continuity and the exact
structural parent mean; it is a reconstruction operator, not terrain detail.
Evaluate shoreline membership and the bed equation directly again at fine sample
centers, then reapply shoreline/protected constraints. These world-vector
overrides may legitimately alter the subsequently decoded-child-derived LOD -1
mean. No unresolved residual exists in the structural proof.

## 11. Regime And Phenomenon Graph

The normative decomposition is `docs/deep-research/microtopography-generation/review/REGIME-PHENOMENON-MATRIX.md`. The production registry must at least distinguish:

- raised bog; fen/transitional mire; drained/cut peat;
- forest pit-and-mound; ordinary forest floor; managed forest/clear-cut;
- ploughed field; seedbed/harrowed/rolled field;
- pasture/meadow and yard/recreational turf;
- aeolian sand, beach sand, gravel/shingle shore;
- till/moraine, glaciofluvial/gravel/outwash;
- carbonate alvar, pavement/karst;
- sandstone and carbonate outcrop tops/slopes;
- colluvial slope/talus;
- floodplain, active bar/bed, rill/gully/seep;
- coastal wetland;
- roads/tracks, quarry/spoil, and technogenic ground;
- open water as a forbidden residual domain.

Pin the catalog as
`estonia-microtopography-phenomena/2026-07-13.1`, SHA-256
`6ca03d5bb552f6ad83e818da1d16d3fddf7d4006241990a90f6ffc21ab13ee4d`.
Changing a phenomenon, eligibility rule, required condition, scale claim, or
representation limit creates a new catalog version and invalidates every affected
row. The complete human-readable definitions and hard limits are in
`docs/deep-research/microtopography-generation/review/REGIME-PHENOMENON-MATRIX.md`;
the IDs and release meanings below are reproduced here so this spec is executable
without guessing.

### 11.1 Release-row schema

Every resolved row validates against
`microtopography-regime-release/1.0.0`, with `additionalProperties=false` and
these 13 required top-level groups. Omitted and unknown are not pass states.

| Required group | Mandatory content |
|---|---|
| `identity` | stable regime ID, display name, pinned catalog ID/hash, row revision |
| `physical_definition` | surface semantics, phenomena, claimed wavelength range, exclusions, narrowly worded transfer claim |
| `eligibility` | versioned three-valued predicate AST, required evidence, exclusive rows, and `unknown -> abstain` |
| `transition` | compatible mixtures, transition-domain owner/width/evidence, and boundary behavior |
| `structural_and_representation_constraints` | protected structure, forbidden masks, heightfield limits, composer ownership |
| `conditions` | source fields, supports, dates, confidence/missingness gates, valid ranges and counterfactuals |
| `target_evidence` | site IDs, surface semantics, effective resolution, measurement error, transfer function, support/QA masks, licenses, artifact hashes, transfer ceiling, qualification result |
| `geographic_splits` | grouping unit, train/development/blind/sensor-holdout IDs, leakage audit, and `taevaskoda_excluded=true` |
| `candidate_bakeoff` | preregistration, corrected-only baseline, candidate/config/budget IDs, required families, justified omissions, `allow_no_winner=true` |
| `band_evidence` | separately for `0.25-1 m` and `0.125-0.25 m`: signal/error RMS, confidence interval, evidence IDs and `supports_claim` |
| `morphology_and_visual_result` | frozen gate, measured/blind results, real-vs-real reference, failures, decision enum, rationale |
| `packing_cook_and_serving_cost` | measured state, bytes/km2, nearest-rank p50/p95/p99 entropy, train/inference/cook/scratch/retry/egress/recook ledger |
| `failure_abstention_and_release` | `unsupported/research/pilot/national`, per-band disposition, owner, frozen evidence, abstention codes, representation action, extent and approval |

Eligibility uses `laas-condition-ast/1` with recursive `all`, `any`, `not`, and
leaf `{field, op, value}` nodes. Operators are `eq`, `ne`, `in`, `not_in`, `lt`,
`lte`, `gt`, `gte`, `contains`, `intersects`, and `confidence_gte`. Missing input,
failed confidence, or schema/hash drift evaluates to `unknown`, never false-nearest
fallback. Mutually exclusive rows may not both resolve true. Compatible mixtures
must be explicit rows or explicit transition compositions with non-overlapping
phenomenon/band ownership.

The normative checked-in artifacts are:

- `review/contracts/regime-release-registry.schema.json`, SHA-256
  `a46d86c0a2960d0a9b58ae05bef3794bf055efb113fd21e8d4adfcdbdeaad495`;
- `review/contracts/regime-release-rows.initial.json`, SHA-256
  `74257a057d7aeaf196d0fdccff374cbcafd1a902efcc7723eea4c81fabac7b48`;
- `review/contracts/regime-initial-status.json`, SHA-256
  `f4ad9903077a736f254d378b5fc0e14c68c1be7dbdb3aeb43f03138db1f258d9`.

Paths are relative to
`docs/deep-research/microtopography-generation/`. The initial registry contains
exactly 26 complete rows: 25 unsupported and one foreign-Moore research-only row,
with zero Estonia targets, owners, release extents, approvals, or positive band
claims. Validate it before use with the command in
`review/REGIME-EVIDENCE-EXECUTABLE-CONTRACT.md`. The implementation may split
schema `$defs` but may not weaken required fields, enums, cross-field gates, or
`additionalProperties=false`. Cross-field gates are:

- `pilot` requires two independent development sites, one untouched blind site,
  passed target qualification and leakage audit, measurable band evidence, a
  `pilot_owner`, measured cost, bounded extent, and user approval. A
  `neighbor_analogue_only` pilot additionally requires Section 9.3's two-campaign
  physical-analogue/OOD/Estonia-sentinel contract and may not promote to national;
- `national` additionally requires an `estonia_national` transfer ceiling,
  `national_owner`, national coverage audit, national extent, and explicit user
  approval; the three-site minimum is a go/no-go floor, not sufficiency;
- `research` may use foreign data and name a `research_owner` for registry
  bookkeeping, but that name is never a production owner. Its bands remain
  `research_only`/`unsupported` and cannot enter production release assets. A
  Section 9.6 research specialist may enter only the separately typed immutable
  non-`latest` preview in Section 14.6;
- `target_evidenced_effectively_smooth` requires Estonia target evidence for the
  exact condition stratum whose measured transfer function and error can resolve a
  preregistered physically/perceptually meaningful residual threshold. A one-sided
  equivalence test must place the 95% upper confidence bound on recoverable true
  signal below that threshold. Large measurement error, missing/occluded samples,
  or smoothing never prove smoothness;
- `released` requires `supports_claim=true`, a non-null frozen owner, passed
  visual/morphology gates, measured cost, extent, and approval.

### 11.2 Initial Estonia registry

This is the exhaustive initial production status for the pinned catalog.
`research` does not authorize production packing. It authorizes research-preview
packing only through Sections 9.6-9.8 and 14.6, without changing the row's bands,
owner, extent, approval, or production status. Every ID must join one-to-one to a
complete 13-group row; an unknown, missing, duplicate, or newly discovered
phenomenon fails the build.

| Regime ID | Initial state | Both bands | Reason |
|---|---|---|---|
| `peat.raised_bog` | `research` | `research_only` | Moore supports only foreign plot-local clipped-moss research |
| `peat.fen_transitional` | `unsupported` | `unsupported` | no qualified Estonia targets; Moore does not span fen states |
| `peat.drained_cut` | `unsupported` | `unsupported` | no qualified dated target sites |
| `forest.pit_mound` | `unsupported` | `unsupported` | no qualified Estonia ground targets |
| `forest.floor_ordinary` | `unsupported` | `unsupported` | no qualified Estonia ground targets |
| `forest.managed_clearcut` | `unsupported` | `unsupported` | no qualified dated operation targets |
| `agriculture.ploughed` | `unsupported` | `unsupported` | foreign measurement evidence only |
| `agriculture.seedbed_harrowed_rolled` | `unsupported` | `unsupported` | foreign measurement evidence only |
| `grassland.pasture_meadow` | `unsupported` | `unsupported` | no qualified target sites |
| `grassland.yard_turf` | `unsupported` | `unsupported` | no qualified subtype targets |
| `sand.exposed` | `unsupported` | `unsupported` | no Estonia material-state targets |
| `sand.dune_aeolian` | `unsupported` | `unsupported` | no Estonia context/target pairs |
| `shore.beach_sand` | `unsupported` | `unsupported` | no dated Estonia target series |
| `shore.shingle_gravel_cobble` | `unsupported` | `unsupported` | no material-matched targets; heightfield limits |
| `glacial.till_plain` | `unsupported` | `unsupported` | no Estonia target-scale surfaces |
| `glacial.gravel_esker_outwash` | `unsupported` | `unsupported` | no Estonia target-scale surfaces |
| `carbonate.alvar_thin_soil` | `unsupported` | `unsupported` | no Estonia alvar height targets |
| `carbonate.pavement_karst` | `unsupported` | `unsupported` | unresolved measurements and heightfield limits |
| `outcrop.sandstone` | `unsupported` | `unsupported` | no lithology-matched Estonia targets; wall is non-heightfield |
| `outcrop.carbonate` | `unsupported` | `unsupported` | no lithology-matched Estonia height targets |
| `slope.colluvium_talus` | `unsupported` | `unsupported` | no source-to-toe targets and non-heightfield blocks |
| `fluvial.floodplain` | `unsupported` | `unsupported` | no dated Estonia flood target series |
| `fluvial.active_bar_bed` | `unsupported` | `unsupported` | no flow/bathymetry-conditioned targets |
| `fluvial.rill_gully_seep_spring` | `unsupported` | `unsupported` | no connected-network target sites |
| `coast.wetland` | `unsupported` | `unsupported` | no Baltic inundation-conditioned targets |
| `technogenic.disturbed` | `unsupported` | `unsupported` | no subtype-specific dated targets |

### 11.3 Complete coverage audit

Every claim first binds a `microtopography-claim-domain/1.0.0` manifest. It contains
the claim ID/type, exact EPSG:3301 polygon/multipolygon artifact and SHA-256,
catalog/condition/mask snapshot hashes, rasterization convention, expected LOD -2
interior chunk keys, expected unique sample count, water/hard-structure and
non-heightfield exclusion geometries, and whether the claim is
`structural_bounded`, `regime_pilot`, `multi_regime_pilot`, or `national`. The
expected key/sample set is generated from the claim geometry before reading a
release index and is itself hashed. A bounded proof uses the same contract; it may
not redefine its claim as whatever happened to cook successfully.

The release auditor enumerates every expected unique interior sample center first,
then joins the release index. A missing expected chunk/sample emits `U`; an
unexpected published negative key is an `A`. It uses the same world-coordinate
oracle, condition snapshot, masks, and dispatcher as synthesis and emits exactly
one terminal per claimed band:

| Code | Terminal |
|---|---|
| `R` | one `pilot`/`national` row owns the sample, every gate passes, and the band is `released` |
| `S` | the exact condition stratum passes the target-evidenced-effectively-smooth gate |
| `F` | a validated open-water or hard-structure mask owns the sample and residual is absent |
| `N` | a validated non-heightfield mask owns the sample; area is excluded and separately reported |
| `U` | required evidence is unknown, only research/unsupported rows match, or no released/smooth row exists |
| `A` | exclusive overlap, invalid mixture, missing catalog row, schema/hash drift, or inconsistent masks |

Reconcile expected keys, produced keys, and merged-release index keys byte-for-byte
before evaluating success. For every chunk in canonical `(z,x)` order, hash ordered terminal bytes plus row
revisions; record counts, extent, source snapshot, and apron ownership. Verify
east/south apron terminals against the neighbor interior, sum exact counts and
area by terminal/regime/band/condition/source, and repeat with changed worker
count/order/AOI. A full morphology claim fails on any `U` or `A`, any count gap or
overlap, or any changed hash. `F` and `N` are reported separately and never
inflate morphology success. Structural-only corrected terrain may publish under
an explicitly structural claim while its morphology coverage remains failed.

No implicit default or nearest-regime fallback exists.

Research previews are never inputs to this production coverage audit. For every
production claim, a cell covered only by research geometry remains `U`; browser
visibility does not convert it to `R`. A separate
`microtopography-research-preview-domain/1.0.0` manifest records exact bounds and,
per band, `research_generated`, `structural_only`, `forbidden`, or `unknown`.
`research_generated` means only that a frozen research candidate supplied the
preview sample. It is not a production terminal, coverage credit, or smoothness
claim.

## 12. Specialist Framework

### 12.1 Required challengers

For each production-data-supported regime, compare on the same
train/development/blind-test sites:

- corrected structural surface with no unresolved residual;
- contextual/multilayer exemplar transfer;
- calibrated conditional simulation;
- a calibrated process/event model when the phenomenon has one;
- learned deterministic regressor;
- conditional GAN;
- pixel-space conditional diffusion;
- justified hybrids with non-overlapping ownership.

The rejected LUKE quilt is not the quality bar. A family may be omitted only when a written pre-registration shows that its assumptions cannot represent the regime.

A Section 9.6 research screen may run a smaller, preregistered subset chosen for
maximum information under the existing-machine budget. That is method research,
not the required production bakeoff: omitted families remain untested, no
real-vs-real ranking exists, and a research preview cannot receive production
credit. Corrected-only is mandatory in both branches.

### 12.2 Learned-model rules

If GAN or diffusion is tested:

- generate metric height/residual in pixel space; do not use a natural-image RGB VAE;
- train on physically defined analysis bands from qualified measured targets for
  production, or on explicitly `research_trainable` weak surfaces with the
  Section 9.6 confidence/unknown weighting for research only;
- keep original/corrected height, derivatives, optical evidence, categorical factors, confidence, and missingness in separate encoders before fusion;
- do not initialize the height prior from Stable Diffusion;
- optical encoder pretraining requires ablation and leakage checks;
- condition on absolute coordinates only through non-memorizing large-scale context; hold out geography and test coordinate ablation;
- train with real fitted degradation, not only bicubic/box synthetic pairs;
- mask every target loss by measured support and semantics; weak research losses
  additionally preserve their evidence tier and may not be reported as target loss;
- report multiple samples during research, then freeze one recipe-bound realization for production;
- calibrate uncertainty on held-out sites for production. Research must instead
  distinguish model uncertainty, weak-supervisor disagreement, and unknown total
  survey error; a predicted confidence channel is not self-validating.

Pixel-space conditional diffusion is the default high-ceiling learned challenger, not the default production owner.

### 12.3 Process/event rules

A process/event specialist must define:

- eligible physical domain and boundary conditions;
- causal variables;
- event frequency, amplitude, orientation, spacing, age/decay, and co-occurrence distributions;
- which quantities are measured, inferred, or simulated;
- calibration sites and uncertainty;
- the scale it owns;
- an ablation proving it adds recognizable supported structure.

One ideal hummock, pit/mound, furrow, fracture, or rut primitive repeated with jitter is not an accepted distribution.

### 12.4 Exemplar rules

- Match physical regime and acquisition semantics, not only local slope.
- Hold out whole sites and exemplar groups for production. The first research
  bundle additionally uses the contiguous-block and cross-campaign restrictions
  in Section 9.8 and may not describe them as site-level target validation.
- Measure atom/patch reuse, nearest-training similarity, repetition, and transition behavior.
- Reject coverage outside the exemplar state space.
- Do not hide attenuation or variance loss through overlap blending.

### 12.5 Statistical simulation rules

- Fit production variograms/cross-covariance/anisotropy by regime and state from
  target surfaces. A research fit from `R1_weak_surface` is labeled weak-evidence
  and reports sensitivity to confidence thresholds and unknown-mask erosion.
- Use only for residual phenomena whose target diagnostics support local stationarity.
- Do not use Gaussian roughness as a substitute for typed events, channels, furrows, cracks, pits/mounds, or shores.
- A spectral match without recognizable morphology fails.

### 12.6 Research-candidate and production-owner state machines

Research candidates use this exact non-owner state machine:

```text
research_declared
  -> research_trained
  -> development_survivor
  -> research_recipe_frozen
  -> research_audit_pass
  -> research_preview_candidate
  -> research_preview_investigated | research_rejected

any pre-preview state -> research_rejected
```

`research_declared` requires the applicable bundle and preregistration in Sections
9.7-9.9. `research_trained` means only that the frozen training or deterministic
development-generation run completed.
`development_survivor` passes hard surface-safety, forbidden-domain, partition,
and cost ceilings plus its bundle-specific development gates: weak-evidence fit
and anti-copy for weak-surface candidates, or the Section 9.9 process gates for
the erodible-slope R0 candidate.
`research_recipe_frozen` binds code, environment, checkpoint, seed, inference
windows, composer, pack extent, metrics, and visual rule before any frozen audit.
`research_audit_pass` requires every bundle-specific frozen audit to pass its
preregistered role. For `forest-weak-research-bundle/1` these remain the
ForestSemantic test split, Evo evidence, HY_PINE2 stress behavior, and, when
opened, sealed Järvselja result. For `erodible-slope-research-bundle/1` it is the
sealed OOD2 abstention-only stress in Section 9.9. A pass permits one
`research_preview_candidate`; it does not declare a winner or owner. Any audit
failure, post-audit tuning, prohibited data access, copied/repeated relief,
unbounded confident OOD output, or failed hierarchy/runtime gate is terminal
`research_rejected` for that frozen recipe.

Research comparison may report descriptive weak-surface fit, cross-view
coherence, band energy, morphology descriptors, R0 process topology/conservation,
packing survival, and fixed visual preference against corrected-only. It may not report real-vs-real
non-inferiority, target RMSE, recoverable truth, production success, or Estonia
transfer. Candidate selection for preview is by the preregistered research rule;
cost may not rescue a visibly or geometrically failed candidate.

Production owner selection remains separate and unchanged:

Before any candidate training, check in and obtain user approval for a
target-specific scientific preregistration artifact and machine-readable candidate
configs. The preregistration must exactly define independent sampling units,
condition strata, tensor/channel order, grids/phases, observation operators,
training-window enumeration, augmentations, initialization, optimizer/checkpoint/
EMA policy, every seed as a replicate, descriptor stencils/radii, PSD taper and
bins, variogram estimator/lags/weights/bounds, form segmentation, distribution
distance/normalization, measurement-error propagation, plot/site weighting,
render support/context, compute budgets, and equivalence margins. Its hashes enter
the recipe. A prose family name or a promise to freeze these later is not an
executable quality decision.

Select the owner lexicographically:

1. discard every candidate that fails target-data, source-safety, heightfield, forbidden-domain, partition, hierarchy, repetition, or cost-accounting gates;
2. require statistical non-inferiority to a real-vs-real reference built only from replicated comparable condition strata, with geographic site/campaign as the top independent unit, for every registered descriptor family;
3. require a blind rendered preference win over corrected-only and, only when one exists, the current accepted morphology owner;
4. declare `tie_no_owner` when survivors are indistinguishable under the frozen quality margin; cost may break the tie only when the preregistration explicitly defines a quality-equivalent set and its cost rule before training;
5. freeze its regime row, target registry, configuration, checkpoint/code, environment, seed, and release extent.

The state machine is exact: failed target qualification yields
`target_evidence_insufficient`; qualified data but all candidates hard-failing
yields `no_candidate_survives`; indistinguishable survivors yield `tie_no_owner`;
only a preregistered quality win yields an owner. The first three states leave the
regime unsupported for production packing. They do not prevent a separately
labeled research-preview pack when the research state machine above passes, but
that preview supplies no evidence toward a later owner decision.

### 12.7 Stage 1 Moore evidence and method screen

The Moore archive is a `0.01 m` natural-neighbor-interpolated exported grid; its
published analysis then applies the archived `demsmooth.m` 0.03 m mean filter. It
is not demonstrated 1 cm effective truth. Most rasters
lack original point support; the two validation cases use special datum alignment
or custom 3D registration and do not provide a transferable per-cell error/MTF for
all campaigns. Moore also reports that 95% of plot variance is captured at scales
larger than roughly `0.37-0.90 m`. A true-0.25 m-parent `B2` experiment therefore
tests an oracle tail, not the project's `1 m -> 0.0625 m` problem or most visible
peat morphology.

Accordingly Stage 1 does **not** select a research owner from this archive. Its
historical evidence result remains `target_evidence_insufficient`, and the archive
still cannot rank or authorize a production winner. The exploratory screen
distinguishes:

- `M0`, a declared ideal synthetic `1 m -> 0.0625 m` task that includes both
  analysis bands but does not model real Maa-amet degradation; and
- `M1`, a true-target-derived 0.25 m oracle-input ablation that isolates an upper
  bound on the finest tail and is never called production-shaped.

Neither tests 0.25-10 m mire organization, 128 m context, Estonia covariates,
source correction, sensor error, or Estonia/production ownership.

#### 12.7.1 Narrow M0 research activation

The 2026-07-15 machine design at
`review/contracts/moore-m0-research-design.2026-07-15.json` authorizes only the
bounded source materialization and source-decimeter eligibility decision below.
It reserves `D`, `R`, and `E` plus corrected-only as the complete candidate subset,
but does not authorize any candidate training. A separately reviewed, hash-bound
candidate-selection supplement must first freeze the exact descriptor stencils,
distances, normalization, and input-only boundary/context operator for joint M0
B1 training. That supplement is not defined here. Any later selected candidate
must emit one joint absolute `Hhat` at `0.0625 m`; it is not a B1 model plus an
independently stamped B2 model. Its registered outputs are B1 and a
`derived_B2_hypothesis`. The latter name is mandatory because it describes only
repeatable behavior in the publisher's natural-neighbor-interpolated, 0.03 m
mean-filtered product. It is not direct support, recovered truth, an acquisition
MTF claim, total-error evidence, or a transferable peat prior.

Before any candidate trains, and again on every held-out candidate output, B2
must pass the frozen four-phase gate below. Compute
`B2_p=H_p-R4(F4(H_p))` for each Section 12.7 output phase on the common complete
valid footprint. Treat each B2 value as constant over its explicit half-open cell
and compare phases by exact polygon-overlap integration; interpolating one phase
onto another or selecting a favorable phase is forbidden. Geographic group is
the independent unit. At least six of the eight frozen groups must have usable
common support. For group `g` and phases `p,q`, define
`E_gp=mean_area(B2_gp^2)`, `E_g=(1/4) sum_p E_gp`,
`D_g=(1/6) sum_(p<q) mean_area((B2_gp-B2_gq)^2)`, and `R_g=D_g/E_g`.
Zero `E_g` fails. Define the population energy coefficient of variation as
`CV_g=sqrt((1/4) sum_p (E_gp-E_g)^2)/E_g`. Every following conjunct is mandatory:

- phase stability: `max_g R_g` is at most `0.25`, and population `CV_g` is at
  most `0.15` at the group median and `0.25` in every used group;
- signal and anti-alias behavior: median group B2 RMS is at least `0.01 m`, phase
  disagreement ratio is therefore bounded by `R_g`, and radial PSD energy in
  `7-8 cycles/m` is at most `0.15` at the group median and `0.25` in every used
  group relative to energy in `4-8 cycles/m`;
- morphology stability: enumerate every same-plot, same-sign family containing
  one signed watershed form from at least three phases, with prominence at least
  two times the plot-phase B2 MAD on common valid support, area-equivalent diameter
  `0.125-0.5 m`, and pairwise centroid distance at most `0.125 m`. Select a
  maximum-total-relief-volume set of globally disjoint families, breaking equal
  optima by the lexicographically lowest sorted canonical family-ID list. For each
  group separately, selected member relief volume divided by all eligible form
  relief volume across its four phases must be at least `0.70`; a zero denominator
  fails;
- browser-equivalent packing survival: decoded maximum height error is at most
  `0.005 m`, decoded B2 energy is `0.90-1.10` of the pre-pack value, at least
  `0.90` of persistent-form relief volume remains, parent/seam failures are zero,
  and the registered measured crop loses no detail to cluster or DAG discard.

The source-derived B2 gate is evaluated before training and is immutable
candidate-independent evidence. Its failure terminates Moore as a `<=0.1 m`
research-preview route. A later candidate's failure rejects that candidate. A
B1-only pack is not a substitute for either failure and may not be presented as
the requested preview.

The first executable checkpoint is the no-model materialization in the machine
design. It must hash and enumerate all 68 grids, bind the eight geographic groups,
reproduce exactly `3,091,387` finite 0.01 m cells and `309.1387 m2`, materialize
all four exact-overlap M0 phases and masks, enumerate the frozen
`2 m` support/`1 m` valid-core/`0.5 m` stride windows and splits, compute the
source-derived B1/B2 gate inputs, and emit the declared content-addressed manifest
and diagnostic PNG set. It decides only whether Moore's derived source is eligible
for a decimeter research attempt. Passing does not authorize `D`, `R`, or `E`;
training remains blocked until the separate hash-bound candidate-selection
supplement exists and passes review.

Use eight outer leave-one-geographic-group-out folds: all 50 `REC_*` plots as Red
Earth Creek; Alpha/Beta/Gamma/Epsilon/Zeta/Eta/Iota/Kappa/Theta/Lambda as Nobel;
WET/INT/DRY as Seney; Maine/Caribou Bog; James Bay; Limerick; Puslinch; and
Sweden/Rodmossen. Inner model selection is leave-one-remaining-group-out. The outer
group is untouched until configuration and seed are frozen. WET, INT, and DRY
retain their individual publisher plot labels and campaign metadata in every
artifact, but remain one conservative Seney leakage group and never cross a split.

The mandatory evidence audit must verify archive/per-record hashes, units,
orientation, exported-grid spacing, interpolation/filter code and boundary fill,
surface semantics, finite masks, group IDs, and total finite area within 0.01 m2.
It must separately reconstruct the Puslinch sign convention and median datum
alignment and the laboratory PLY rotation/translation registration, naming every
byte-level input, coordinate transform, fitted degree of freedom, invalid dilation,
residual covariance/PSD, and campaigns to which an estimate may transfer. It must
not transfer either validation residual to other campaigns without evidence. The
absence of per-cell point support and defensible replicated-stratum error bounds
therefore leaves the current archive `target_evidence_insufficient` for owner
selection. This result is known before candidate output and forbids an ownership
or non-inferiority claim.

For optional M0/M1 engineering screens, treat every finite source value as constant
over its explicit half-open 0.01 m source cell after applying archived
`demsmooth.m` (SHA-256
`729f54883e52360e57fdecee7a88aed4084973d40e0b31215f3612e28eaebddf`)
exactly once. Its exact emulation converts `z` to float64, records the original NaN
mask, pads two cells, fills padded NaNs by minimum squared Euclidean distance with
ties resolved to the lowest MATLAB column-major index in the padded array (or
replicate-pads when the source contains no NaNs), convolves once with a 3x3
float64 kernel whose nine weights are `1/9`, crops the two-cell pad, and restores
the original NaNs.
Nearest-filled or padded cells never become measured support. The source x/y
values are 0.01 m cell centers; the explicit cell-boundary origin is
`(min(x)-0.005 m,min(y)-0.005 m)` after the audited orientation is applied. Test
output-boundary offsets
`(dE,dN)={(0,0),(0.005,0),(0,0.005),(0.005,0.005)} m` relative to the source
cell-boundary origin; the corresponding 0.25/1 m synthetic observation grid shares
that output phase. Resample by exact polygon/cell overlap and require at least 95%
finite support after invalid dilation. Report every metric by phase and the maximum
phase change; no single favorable phase may be selected. Let `Q4` be the exact fixed-order area mean over aligned 4x4 cells.
For target `H`, M0 input is `O0=Q4(Q4(H))`; M1 oracle input is `O1=Q4(H)`.
Candidate output is one absolute `Hhat`. In M0 enforce
`Q4(Q4(Hhat))=O0`; in M1 enforce `Q4(Hhat)=O1`, after complete world-aligned
window fusion, using a deterministic null-space parameterization or final
world-aligned correction. This ideal synthetic constraint never authorizes exact
projection to fallible production DTM. Analyze, but never conflate, packed-null
deltas with Section 13.2's overlapping-filter bands
`B1=A1-R4(A0)` and `B2=H-R4(A1)` where `A1=F4(H)` and `A0=F4(A1)`.

Strict measured-only B1 support requires the complete two-stage F4/R4 footprint;
neither `demsmooth.m` padding/fill nor filter reflection supplies support. Small
plots can therefore have zero B1-valid output (known examples include
`REC_001_DEM.mat`). Materialization must report strict valid counts by plot and
group and preserve zero rather than manufacture support through padding,
reflection, interpolation, or a relaxed halo.

Shared inputs are the named synthetic observation's structural prolongation,
gradient, Laplacian, 0.5/1 m relief, and validity. Loss/metrics use only measured
support with invalid/filter-boundary dilation. D4 transforms are permitted because
reliable north is absent. The historical M1 and any M0 configuration outside the
narrow machine design report failure modes, parent preservation, band behavior,
tiling, packing survival, and compute only; no candidate wins. The narrow M0
materialization names no candidate. A later hash-bound supplement may authorize
selection of only a research-preview candidate under Section 12.6, never a winner
or owner.

The following seven families remain the required lower-bound comparison set for a
later target-specific preregistration; the prose recipes alone are not complete
machine configs and do not authorize training. The narrow M0 machine design
reserves only `D`, `R`, and `E` plus corrected-only; `S`, `P`, `G`, and `H` remain
inactive. Reservation is not training authorization; the hash-bound supplement
above remains mandatory. In a Moore engineering screen, write
`Delta=Hhat-prolong(observation)` for the metric candidate delta:

| ID | Executable recipe |
|---|---|
| `E` contextual exemplar | Store training support windows and `Delta`; Moore M0 descriptor height is an 8x8 array of exact 0.25 m area-pooled samples over the named 2 m structural prolongation, not an 8x8 array of 1 m `O0` cells, plus co-registered gradients/relief/Laplacian; standardize then use 32-component training-only PCA; retrieve eight Euclidean nearest from other sites; PRF-select proportional to `exp(-(d-dmin)/max(median(d-dmin),1e-6))`; return metric residual after registered D4 transform |
| `S` conditional simulation | 5x5 reflected-neighborhood pointwise ridge mean with alpha inner-selected from `{1e-4,1e-3,1e-2}`; fit remaining isotropic Matern variogram at 0.0625 m lags to 1 m, `nu in {0.5,1.5,2.5}`; coordinate-PRF circulant embedding over expanded plot; condition aligned 4x4 residual means to zero by Cholesky with jitter `1e-8*sill`; abstain if embedding remains non-positive after two dimension doublings |
| `P` marked forms | On `R4(F4(Delta))`, keep positive/negative extrema above a preregistered qualified-error prominence; watershed basins; fit signed elliptical Wendland C2 `A*(1-r)^4*(4r+1)` for `0<=r<1`; empirical sign-specific marks plus Matern-II hard-core at fifth-percentile same-sign spacing; PRF candidates on 0.125 m owner lattice, deterministic priority, orientation only from parent gradient; one inner-training least-squares global amplitude; ineligible below 200 forms/six groups; no generic remainder |
| `R` deterministic regressor | Three-level residual U-Net `(48,96,192)`, two 3x3 residual blocks/level, GroupNorm 8, SiLU, stride-2 down, nearest+3x3 up, reflected one-pixel padding, no attention/no noise, one metric output; `Huber(delta=.01m)+.25 L1 gradient+.10 L1 five-point Laplacian` |
| `G` conditional GAN | `R` generator plus one coordinate-PRF normal channel; spectral-normalized four-level `(48,96,192,256)` 4x4 conditional PatchGAN; hinge loss; generator uses `R` loss plus `.02*adversarial`; one discriminator then one generator update on same batch; no RGB perceptual/style/VAE loss |
| `D` pixel diffusion | `R` U-Net plus sinusoidal timestep projected to 192 and FiLM each residual block; v-prediction, Nichol-Dhariwal cosine alpha schedule `s=.008`, 1,000 timesteps; `MSE(v)+.10 x0-gradient L1+.05 x0-Laplacian L1`; 50 deterministic DDIM steps at deduplicated rounded descending linear 999..0, `eta=0`, global PRF noise and per-step `v` fusion; no latent RGB/ControlNet |
| `H` process-diffusion | Generate `P`; train `D` on `Delta-P` with `P` as condition; composer adds each once; ineligible if `P` is ineligible, with no silent fallback to `D` |

For R/G/D/H use float32 training, float64 metric conversion, AdamW `2e-4`, betas
`.9/.999`, weight decay `1e-4`, batch 32, gradient clip 1, 50,000 updates per outer
fold/seed, seeds `{17,31,47}` treated as replicate identities and all aggregated,
at most 3.5M parameters,
no pretrained weights, and identical accepted windows/augmentations. E/S/P use the
same folds/windows and report CPU time/memory. Freeze/report all compute before
blind quality inspection.

Every stochastic candidate emits eight PRF realizations per held-out group. Report
real-vs-real envelopes for height/gradient/normal/curvature, band energy, PSD,
variogram, extrema/form geometry/connectivity; parent deviation after actual pack;
periodicity, nearest-copy/train attribution; p50/p95/p99 bytes and cost; and fixed
geometric render gates. Target-scale noise, generic bumpiness, copying, repetition,
seams, forbidden water, barely visible/shading-only/DAG-discarded detail, or missing
cost is an engineering failure. Because the archive lacks replicated comparable
condition strata, no real-vs-real non-inferiority envelope, production survivor,
owner, or tie is declared. The narrow M0 activation may declare only the Section
12.6 research states frozen in its machine design. Report every geographic group
equally, expose the imbalance, and label all aggregate distances descriptive only.

Do not translate the Estonia camera rig to these tiny plots and do not repeat a
plot to fill a chunk. A hashed `moore-plot-screen-v1.json` must identify the valid
measured crop, a one-cell invalid dilation, neutral material/light, and four
plot-scaled cameras at azimuths 0/90/180/270 degrees, 55 degree FOV, eye 1.2 m
above a robust plot plane, aimed at the valid-crop centroid, with distance chosen
so the valid crop occupies 70% of the shorter viewport dimension. To exercise the
real cluster/DAG/packing path, place one plot once at the center of a structural
128 m master; outside the measured mask use only the robust tangent plane and mark
it context-only. Render and score through a crop/mask that contains measured
support only. No boundary feather, tangent plane, or surrounding area enters a
morphology statistic or visual claim.

## 13. Canonical Composition And Determinism

### 13.1 Ownership order

The composer resolves:

1. corrected direct structure;
2. validated long-range/process/event form;
3. validated within-form unresolved realization;
4. physically justified compatible cross-scale effects;
5. confidence-mediated transitions;
6. protected, water, hard-structure, and representation constraints.

Two specialists claiming the same phenomenon and scale is a configuration error. Do not average them.

For a research preview, `validated` in items 2-4 means only
`research_preview_candidate` under Section 12.6 and is recorded as provisional
research claimancy, never production ownership. Exactly one frozen candidate may
claim a preview sample/band. The composer still reapplies protected, water,
hard-structure, heightfield, and unknown masks after the candidate. Production
composition ignores every research claim and resolves the same cell as unsupported
until a qualified owner exists.

### 13.2 Analysis bands

Packed rungs are not model stages. Scientific analysis uses the separable
factor-four filter:

```text
c=[1,3,6,10,12,12,10,6,3,1]/64
F4(x)[p]=sum(k=0..9,c[k]*x[4*p-3+k])
```

Accumulate float64 east then north. The support center is between fine indices
`4p+1` and `4p+2`, exactly the coarser sample center; the DC limit is one. Each
stage needs three input samples before and six after its nominal footprint. Use a
symmetric six-sample input halo for one stage and 30 finest samples (1.875 m) for
two. Reflect about the exterior sample-center plane only at a real domain boundary;
never wrap or clamp. Reflection supplies context only: invalidate any target/loss
whose footprint touches plot boundary, NaN, interpolation, or human-invalid data.
Storage boundaries always expand support and are never analysis boundaries.

Reconstruction `R4` is separable Keys cubic, `a=-0.5`, with fine phases
`{-3/8,-1/8,+1/8,+3/8}` and weights for fractional `t`:

```text
w[-1]=-0.5*t+t^2-0.5*t^3
w[ 0]=1-2.5*t^2+1.5*t^3
w[+1]=0.5*t+2*t^2-1.5*t^3
w[+2]=-0.5*t^2+0.5*t^3
```

For finest surface `H2`, define:

```text
A1=F4(H2); A0=F4(A1)
B2=H2-R4(A1)
B1=A1-R4(A0)
H2=R4(R4(A0)+B1)+B2
```

This identity is exact by construction up to declared float64 arithmetic. Report
signal, target error, model error, and packing error separately for `B1` (0.25-1 m
registered band) and `B2` (0.125-0.25 m registered band). Cross-scale objects are
generated once then analyzed, not independently stamped into storage rungs.
Release metrics run after final composition and browser-equivalent quantization.
The packed hierarchy still uses the repository's fixed 4x4 decoded-child area
reducer, not `F4`.

For Moore, treat each finite 0.01 m source cell as a constant area sample and
resample to 0.0625 m by exact source/output cell-overlap area; require at least 95%
finite overlap. Bind its existing 0.03 m mean filter and validation errors, then
apply `F4`. Never treat nominal 0.01 m pitch as an unfiltered transfer function.

### 13.3 Global stochastic identity

Use keyed BLAKE2b-256 from Python `hashlib`, not Python `hash()`, global NumPy RNG,
chunk seeds, or framework defaults. The key is exactly 32 recipe-seed bytes;
personalization is ASCII `laas-micro-prf1`; digest is 32 bytes. Encode fixed-width
little-endian two's-complement only. Convert text IDs by UTF-8 NFC then SHA-256;
no floats or variable strings enter this 112-byte record:

| Bytes | Field |
|---:|---|
| 0-1 | version `u16=1` |
| 2 | purpose `u8`: uniform=1, normal=2, event=3, window=4, training-mask=5, augmentation=6 |
| 3 | stage enum `u8` |
| 4-35 | recipe SHA-256 |
| 36-51 | specialist UUID |
| 52-83 | physical-domain SHA-256 |
| 84-91 | easting/local-x `i64` in 1/32 m units |
| 92-99 | northing/local-y `i64` in 1/32 m units |
| 100-103 | channel `u32` |
| 104-111 | counter `u64` |

Uniform float32 is `(bits24+0.5)/2^24` using the high 24 bits of the next
little-endian u64 digest word. A normal pair uses digest u64 words 0/1 as open
uniforms in Box-Muller; increment counter for later pairs. World coordinates come
from `sample_center_en_units`. Research plots use a domain-local origin snapped
down to 1/32 m and include the plot artifact digest. Events are keyed by owner
lattice cell and event counter; rounded continuous positions never become keys.

Rules:

- the same world point samples the same base stochastic field in every support window;
- storage chunk coordinates are not keys;
- every inference window comes from a fixed world-aligned lattice;
- every output point uses the same complete set of influencing windows independent of requested AOI;
- overlap/fusion weights and scheduler are frozen and included in the recipe;
- long-range phenomena generate over their watershed/mire/stand/parcel/reach/outcrop domain before cropping;
- periodic boundaries are forbidden;
- worker count, order, batching, retry, adjacent requests, and enclosing AOI must not change accepted bytes on the locked stack.

MultiDiffusion-like overlap is an engineering candidate only. It passes only after height, gradient, normal, morphology, and physical-continuity gates.

The Moore proof freezes a 2x2 m support window (32x32 finest samples), 1x1 m valid
core (16x16), 0.5 m center stride (8 samples), and domain-local lattice anchor.
Output only where every required support sample is target-valid. Enumerate every
half-open core containing a point and sort by `(center_y_units,center_x_units)`.
For local core coordinate `s`, use `w1(s)=cos(pi*s/1m)^2` for `abs(s)<0.5m`, else
zero; `w(x,y)=w1(x)w1(y)`. Accumulate metric-delta numerator/denominator float64 in
that order. Non-diffusion candidates fuse only metric deltas against their named
authority. Diffusion uses one coordinate-indexed global initial-noise field and,
at every denoising step, fuses predicted `v` in metric-normalized band space before
one global state update. Final-output feathering is not the registered candidate.

### 13.4 ML environment

The default NumPy/SciPy cook remains usable without ML extras. If a learned specialist wins:

- add a pinned optional `micro-ml` environment under `asset-gen`;
- pin Python, framework, CUDA/Metal backend, driver compatibility, model code, checkpoint, and preprocessing;
- enable deterministic framework algorithms and disable nondeterministic kernels;
- bind hardware/backend identity into the recipe;
- fail closed when the required environment is unavailable;
- return NumPy float32 metric arrays to the existing cook adapter;
- never substitute a cheaper procedural fallback silently.

Bit identity is required on the locked production stack. A different stack creates a different recipe and must requalify.

## 14. Cook, Hierarchy, And Release

### 14.1 Extents

Plan three distinct extents:

| Extent | Meaning | Published |
|---|---|---:|
| model support | symmetric receptive/process/context domain | no |
| hierarchy support | east/south/southeast child/apron dependencies | no |
| publication coverage | complete runtime-addressable negative chunks | yes |

The current nine transient child chunks are hierarchy support, not a sufficient ML/process halo.

### 14.2 Canonical order

For every proof/release:

1. bind one evidence snapshot for terrain, water, shore, bank, and bed;
2. solve compatible `waterY`, `watercover`, shoreline, bank, and conservative submerged bed;
3. produce corrected structural LOD -1 candidates over complete affected LOD0 chunks and compose the bed into them;
4. produce one absolute LOD -2 final master over publication plus model/hierarchy support;
5. crop each LOD -2 artifact from the master;
6. quantize with `encode_quant16_checked`;
7. browser-equivalent decode and accept canonical decoded LOD -2;
8. assemble 16 decoded child cores plus apron support;
9. fixed-order 4x4 area-reduce to LOD -1;
10. quantize/decode LOD -1;
11. combine decoded-child-derived LOD -1 where fine exists with canonical structural LOD -1 elsewhere;
12. derive, quantize, decode, and publish corrected LOD0;
13. for LOD1-L4, splice decoded-child reductions only into transitively affected
    4x4 windows of the independently cooked inherited DTM chunks, then repair only
    the east/south/southeast apron samples made stale by those masks;
14. cook matching `waterY`/`watercover` and recook slope/height-dependent vegetation/debris from the same accepted snapshot;
15. publish a corrected ordinary format-1 base;
16. pin the existing format-2 negative-rung overlay to that corrected base.

Never replace an entire inherited LOD1-L4 core with a child-derived pyramid: those
rungs are independent DTM cooks, and doing so changes unrelated terrain that merely
shares an ancestor. Never project accepted fine geometry back to a known raw error.

### 14.3 Fine coverage

The next source-repair proof publishes one complete **structural-only** LOD -1
parent plus its 16 LOD -2 children, centered to contain the review coordinate.
Its unresolved morphology contribution is absent; fine samples carry the smooth
vector/evidence-backed shore, bank, and bed. Structural refinement does not wait
for a morphology prior. Fine publication requires:

- complete parent/child closure;
- corrected containing LOD0 chunk;
- complete ancestor/apron closure;
- all inherited non-height content retained;
- no fine region without terrain mesh parent coverage.

Before multi-parent or national fine publication, generalize the current one-parent `HeroCoverage`, release expectation, coverage metadata, auditor, and verifier. The LAC payload, v2 index, server, and runtime plane logic need not change.

A research preview may use the same one-parent publication core or another
preregistered parent-closed bounded extent. Its model and hierarchy support may be
larger, but every published negative chunk still requires complete parent, terrain
mesh, apron, corrected-base, and inherited-layer closure. Research status waives
no representation or consumer-agreement requirement.

### 14.4 Quantization

Retain initial qscales:

- LOD -2: 0.002 m;
- LOD -1: 0.005 m;
- LOD0-2: 0.01 m.

Quantization must:

- reject nonfinite values and u16 overflow;
- never clip;
- record qoffset/qscale and, for `s=float(float32(requested_qscale))` and decoded
  float32 `d`, require
  `max(abs(float64(d)-float64(h))) <= s/2 + 2*abs(spacing(float32(max(abs(d)))))`;
- use browser-equivalent decode for every parent input;
- ensure shared decoded aprons agree and decoded cross-seam gradient and normal
  differences remain inside predeclared quantization-derived tolerances.

The bounded one-parent proof may retain its shared fine qoffset domain over its 16
published LOD -2 children plus nine east/south/southeast support chunks. Read the
corresponding `640 x 640` LOD0 authority and use
`qoffset_fine=float(floor(min(domain_authority)-2.0))`; the checked encoder must
still prove fit. This is a local proof policy, not a national solution.

For a complete fine parent `q=(l+1,qx,qz)`, the decoded immediate-child read set is:

```text
D(q) = {(l,4*qx+i,4*qz+j) | i,j in {0,1,2,3,4}}
parent[r,c] = mean_float64(child_mosaic[4*r:4*r+4,4*c:4*c+4])
```

The north-west `4 x 4` children supply cores; the fifth column/row/corner are the
nine real east/south/southeast apron dependencies. Never duplicate a last core
sample. Use the existing fixed reshape and `mean(axis=(1,3),dtype=float64)` order.
This complete `D(q)` rule applies to ordinary fine-parent derivation. It does not
authorize replacing a legacy LOD1-L4 core, because those source rungs were cooked
independently rather than derived as this pyramid.

Corrected-base hierarchy closure is mask-based. At LOD0 initialize each corrected
core from its inherited browser-decoded values and splice the qualified target only
under exact mask `M_0`. For each affected child 4x4 sample window, reduce all 16
final decoded child values in the fixed order, set only the corresponding parent
sample, and define `M_l+1` by `any(M_l)` over that window. Initialize every LOD1-L4
parent from its independently cooked inherited decoded core; samples outside
`M_l+1` remain bit-identical. No full-core parent derivation or promotion is legal.

For each rung, the affected payload mask is the core mask plus east, south, and
southeast apron positions induced by the corresponding boundary samples of the
neighbor core masks. Stage only existing members of
`R_l = C_l union west(C_l) union north(C_l) union northwest(C_l)` whose affected
payload mask is nonempty. Apron-only files preserve their entire core. Persist a
packed exact affected mask and canonical row-run windows for every artifact, plus
their paths, sizes, SHA-256 identities, and sample counts. The plan also binds the
inherited decoded-core hash and every decoded-child window dependency.

Quantizer policy is
`preserve-inherited-or-retune-full-payload-outside-bitexact/1`. First preserve the
inherited qoffset/qscale and every code outside the affected payload mask; this is
mandatory whenever all changed core and apron values fit. On overflow only, choose
a qoffset from the final full payload, quantize the complete payload, and require
every unmasked decoded float32 value to remain bit-identical to the inherited
payload. Persist both outside-selection hashes and sample count. If that proof
fails, fail closed; never broaden the mask, promote an apron-only core, or propagate
an unrelated change to make quantization convenient.

At every changed rung, verify decoded east/south/corner value identity; reconstruct
the same `3 x 3` world stencil independently from both sides and require identical
runtime central gradients `dx=h_left-h_right`, `dz=h_north-h_south` and
`normalize(dx,2*t,dz)` normals for the shared-qoffset proof. Also evaluate the two
independent decoded bilinear assemblies at
`u=-1+k/8, k=0..16` across a two-texel seam strip and require identical values.
For a necessary changed/inherited boundary with distinct qoffsets, let `s` be the
larger exact wire qscale: require shared values within `s`, each reconstructed
gradient component within `s/t`, and normal angle within
`2*asin(min(1,sqrt(2)*s/(2*t)))`; record actual maxima. These are quantization
bounds, not permission for a visible seam, and the visual gate still applies. Do
not compare different one-sided slopes of a legitimately curved surface.

National qoffset domains require an explicit deterministic grouping/seam policy because one 0.002 m u16 span covers only about 131.07 m vertically.

### 14.5 Recipe identity

Hash:

- raw and condition sources, dates, licenses, and masks;
- target registry;
- research bundle, evidence-tier, research-surface, confidence calibration,
  unknown-mask, split, leakage, audit, and candidate-state manifests when present;
- correction algorithm/configuration;
- regime graph and release rows;
- specialist code/checkpoint/exemplar/process configuration;
- model environment and deterministic settings;
- world stochastic policy and seed;
- support/publication extents;
- reducer, qscales, qoffset policy, codec, and code revision;
- corrected base manifest.

An existing artifact is reusable only when its complete recipe and dependencies match.

### 14.6 Transactional publication

Current `release.py` cannot authorize this release: it pins the old base SHA,
accepts only `retention-fixture` and `measured-synthesis-pilot`, expects exactly the
old 17 negative keys, dispatches only the old verifiers, and refuses pilot
publication to `latest`. The implementation must replace those assumptions with
the following production two-artifact transaction and separately typed
research-preview transaction; bypassing an old check is forbidden.

The first recipe kind is `corrected-structural-base-v1`. It stages an ordinary
format-1 base manifest without changing `latest`. Its schema requires
`raw_base_manifest_sha256`, `observation_snapshot_sha256`,
`claim_domain_manifest_sha256`, `corrected_core_keys`, `support_read_keys`,
`apron_only_replacement_keys`, `affected_ancestor_keys`,
`water_snapshot_sha256`, expected qscale/flag per key, and the complete decoded
dependency map. Its verifier must enforce source-repair gates, `WATERBED_FLAG`,
masked decoded-child/apron closure, the versioned qoffset outside proof, exact
expected-versus-produced key sets, and unchanged inherited artifacts and decoded
samples outside each mask. Success materializes an immutable corrected
base and returns its manifest SHA; it does not publish.

The second recipe kind is `evidence-microtopography-v1`. It binds that exact
`corrected_base_manifest_sha256`, the regime/target/candidate identities,
`claim_domain_manifest_sha256`, `published_negative_keys`, `transient_support_keys`,
and dependency hashes. Its verifier checks the negative hierarchy, forbidden
domains, claim coverage, packing, runtime-consumer agreement, and the merged
format-2 manifest. A fixed 17-key expectation is legal only when the declared
claim and dependency manifests actually contain those exact keys.

Publication order is normative:

1. freeze both expectation documents and their hashes before cooking;
2. stage and independently verify `corrected-structural-base-v1`;
3. materialize its immutable format-1 manifest without touching `latest`;
4. bind that manifest SHA into and stage `evidence-microtopography-v1`;
5. independently verify the merged format-2 manifest and exact claim coverage;
6. run the manifest-bound runtime and visual gates;
7. after explicit user acceptance, atomically update `latest.json` exactly once to
   the already-verified merged manifest.

Any failed phase leaves `latest` unchanged. The expectation-bound corrected-base
SHA replaces every fixed old-base SHA. Diagnostic/research builds are immutable
and addressable but never publish to `latest`.

The only research recipe kind is `research-microtopography-preview-v1`. It binds:

- one already accepted immutable `corrected_base_manifest_sha256`;
- `research_bundle_id`, all raw/derived artifact hashes, evidence tiers, surface
  contract, per-band eligibility, confidence calibration, unknown/forbidden masks,
  and train/development/audit splits;
- research preregistration, code/environment/checkpoint/seed, candidate state, and
  all frozen-audit results;
- `microtopography-research-preview-domain/1.0.0`, exact published/transient keys,
  corrected-only control, cameras, cost ceiling, and dependency identities;
- literal metadata `authority=research_only`, `target_truth=false`,
  `production_owner=null`, `estonia_transfer=none`, `release_row_unchanged=true`,
  and `latest_eligible=false`.

Its verifier rejects any missing literal, any candidate not in
`research_preview_candidate`, any unbound/changed audit result, output outside
`research_generated` cells/bands, residual in forbidden/unknown cells, target or
owner wording, or production claim-domain/row mutation. It also runs the ordinary
absolute-master, quantization, decoded-child hierarchy, seam, parent closure,
inherited-layer, cost, and runtime-consumer checks. Weak evidence changes only
scientific authority, never packing correctness.

Research-preview materialization order is exact:

1. bind every applicable bundle-specific development input and completed
   development audit, then freeze the complete synthesis recipe and expectation
   before any sealed final audit; the forest bundle retains its exact
   ForestSemantic/Evo/HY_PINE2/Järvselja ordering from Section 9.7 and the
   erodible-slope bundle retains its Development A/C then sealed OOD2 ordering
   from Section 9.9;
2. run and bind every bundle-specific final audit; a failure terminates the recipe
   rather than retuning;
3. compose one absolute finest master against the accepted corrected base;
4. stage and independently verify `research-microtopography-preview-v1`;
5. materialize a content-addressed merged format-2 manifest without modifying the
   repository's canonical `latest.json` or any production release row;
6. serve that exact manifest through an explicit preview-server override, run the
   complete Section 16.7 real-WebGPU gate and research visual gate, and retain the
   evidence under the recipe hash;
7. provide the manifest-bound URL for visual investigation only.

No user preference, clean boot, or attractive screenshot can promote this manifest
or model. Promotion requires a future `qualified_target_B1/B2` bakeoff from the beginning;
research metrics and sentinel views may inform hypotheses but supply zero target,
owner, release, or Estonia-transfer credit. Release tooling must hard-reject any
attempt to pass a research manifest to the atomic `latest` update path.

## 15. Orthophoto Acquisition Decision

Orthophoto fetch is required for the research proof and likely production conditioning.

Official RGB/CIR/NGR data are about 20-40 cm nationally and 10-16 cm in dense settlements. Roughly half the country is refreshed yearly, so capture date is mandatory.

The official workbook contains 2,111 distinct 1:10,000 sheets. A sample-based newest-RGB estimate is about 367 GB compressed before extraction, history, CIR/NGR, masks, and working copies. Taevaskoda sheet `54472` has:

| Product | Exact archive | Acquisition | Compressed size |
|---|---|---:|---:|
| RGB | `54472_OF_RGB_GeoTIFF_2025_07_18.zip` | 2025-07-18 | 207,351,332 bytes |
| CIR | `54472_OF_CIR_GeoTIFF_2024_05_22.zip` | 2024-05-22 | 133,805,748 bytes |
| NGR | `54472_OF_NGR_GeoTIFF_2024_05_22.zip` | 2024-05-22 | 148,206,299 bytes |

### 15.1 Exact Stage 1 snapshot

Bind DTM sheet `54472` from
`https://geoportaal.maaamet.ee/index.php?lang_id=1&plugin_act=otsing&kaardiruut=54472&andmetyyp=dem_1m_geotiff&dl=1&f=54472_dtm_1m.tif&page_id=614`,
current SHA-256
`a70ddfeeec08f278201c6479205ca661e247b1ec6b0151597390aa1c726d5239`.
Bind the existing whole-country ETAK fetch ending
`andmetyyp=ETAK&dl=1&f=ETAK_EESTI_GPKG.zip&page_id=609`, current extracted
GeoPackage SHA-256
`e19084c4bff0a6096a7dd38cefa12402bdce98d59c73d5b35c30dda60682f5f9`.
Bind official workbook
`docs/deep-research/microtopography-generation/library/data/maaamet/tomba_etak_avaandmed.xlsx`,
SHA-256 `fbef1433eff6116174cd1bbeca6e6550e093eeed794c14d583af77e03970b4b5`.
Select its exact RGB/CIR archive names/dates/byte counts in the table above; RGB
uses `andmetyyp=ortofoto_eesti_rgb`, CIR `ortofoto_eesti_cir`.

Cache at `asset-gen/data/in/orthophoto/{rgb,cir}/54472/<archive-name>` and extract
under an archive-SHA sibling. Download to `.part`; resume only after byte-range and
prefix-length validation; use the configured five retries/politeness interval;
validate content length, ZIP CRC, GeoTIFF readability, EPSG:3301, transform,
bounds, band count/type, then atomic rename. Record archive/extracted hashes. HTML,
changed workbook row/size, missing CRS, or unexpected schema fails closed.

Validate `geom` and `gpkg_geometry_columns.srs_id=3301` plus these exact layers:
`E_201_meri_a`, `E_202_seisuveekogu_a`, `E_203_vooluveekogu_a`,
`E_203_vooluveekogu_j`, `E_204_kaldajoon_j`, `E_102_nolv_j`. All require
`etak_id,kood,kood_t,muutmisaeg,geom_muutmisaeg,valjavote`; also validate their
documented `tyyp`, `nimetus`, centerline `telje_*`/`laius`, shoreline
`kalda_veekogu_tyyp`, and slope `kaldaastang` fields as applicable. Persist exact
field/type inventory and used feature IDs; missing/renamed/type-changed/wrong-CRS
data fails. Carry Maa- ja Ruumiamet attribution. RGB 2025 and CIR 2024 remain two
dated observations, never a falsely co-temporal composite.

For Stage 1, the accepted ETAK continuous shoreline geometry is the boundary
authority with the positional/temporal uncertainty corridor in Section 10.2.
RGB/CIR is QA evidence and a separately scored refinement challenger, not an
automatic owner. A challenger segment is eligible only when its human-QA mask is
clear, image-to-ground registration p95 is at most 0.25 m, acquisition date is
compatible with the reviewed water state, and every proposed displacement from
ETAK is at most 1 m. It must improve held-out visible shoreline evidence without
copying shadow, canopy, object, water, or albedo edges. Otherwise ETAK remains the
accepted Stage 1 geometry and the optical proposal is retained only as rejected
evidence.

Before national download:

1. scan the complete official manifest for newest per-sheet product sizes;
2. define cache, crop, retention, retry, checksum, and attribution policy;
3. measure extracted/derived working-set size;
4. run visible-structure and morphology ablations;
5. reject any model that copies albedo edges into height.

## 16. Verification And Acceptance

### 16.1 Production data gates

For each enabled regime:

- target support and error pass Section 9;
- at least the Section 9.3 independent-site and physical-analogue requirements exist;
- train/development/blind-test are geographically disjoint;
- surface semantics and licenses are valid;
- signal energy exceeds measurement/interpolation error in every claimed band;
- unsupported factor combinations abstain.

These gates are unchanged and are required before production owner selection.

#### 16.1.1 Research-only data gates

A research specialist instead requires all of the following, without claiming a
production pass:

- exact applicable bundle artifacts, roles, licenses, hashes, identities, and no
  prohibited-data access: `forest-weak-research-bundle/1` follows Sections
  9.7-9.8 and `erodible-slope-research-bundle/1` follows Section 9.9;
- a validated `ResearchSurfaceEvidence` artifact with `R0/R1/R2` tier, separate
  B1/B2 eligibility, probabilistic semantics, explicit unknown/forbidden masks,
  and no fabricated total-error estimate for weak-surface candidates; or the
  complete hash-bound R0 process/condition contract in Section 9.9 for the
  erodible-slope process candidate;
- no geometry outside `research_trainable` weak-surface cells or the frozen
  erodible-slope R0 condition/mask domain, and no positive credit from
  `research_diagnostic`, R0 process references, or stress-only evidence;
- frozen contiguous spatial leakage groups and every applicable bundle-specific
  development, audit, stress, and sentinel role;
- candidate state `research_preview_candidate` before packing;
- literal non-production authority metadata and an immutable non-`latest` extent.

Unresolved license/provenance, forbidden-class leakage, confident output on OOD,
opening sealed inputs early, post-audit tuning, or missing unknown masks is a hard
research failure. Missing independent survey error is not a research failure; it
is mandatory evidence that production target/owner gates remain failed.

### 16.2 Source-repair gates

Measure separately from added morphology:

- raw/corrected delta by correction class;
- direct-constraint fit within source/map registration uncertainty;
- water reach continuity and no interior bridge/spike;
- smooth world-space shoreline without 1 m staircase;
- preserved bank side, shoulder, and escarpment;
- protected-feature false-correction rate;
- calibrated risk/coverage for learned assistance;
- correction-edge continuity into unchanged base.

Any destruction of a real protected cliff, boulder, bank, channel, or depression is a hard failure.

### 16.3 Morphology gates

Evaluate per regime and site:

- height, slope, normal, curvature, local relief;
- multiscale spectrum and variogram;
- typed form size, spacing, orientation, asymmetry, connectivity, and tails;
- distribution precision and coverage, not only mean error;
- nearest-training similarity and memorization;
- repeated-patch and periodicity detection;
- condition counterfactuals;
- transition strips and ordinary terrain, not only hero landmarks;
- multiple research seeds for diversity; one frozen production seed for delivery.

Set numerical thresholds from held-out real-vs-real variability before final candidate review. A low RMSE flat surface and a spectrally correct noise surface both fail.

For research-only candidates, report the same descriptors wherever weak evidence
supports them, but label every result by evidence tier and confidence/unknown-mask
threshold. Replace real-vs-real acceptance with preregistered descriptive ranges,
cross-view stability, Evo cross-campaign behavior, anti-copy checks, and visible
comparison against corrected-only. These can reject a candidate or authorize a
research preview; they cannot establish non-inferiority, production quality, or a
release owner.

### 16.4 One-meter grid leakage

Compare output residual statistics by phase inside the source 1 m cells. Reject:

- phase-locked amplitude, curvature, or orientation beyond held-out real-target variation;
- raster staircase shore/bank boundaries;
- repeated one-meter compensation bowls/ridges;
- visible change in pattern at LOD/chunk boundaries.

### 16.5 Hierarchy gates

Independently recompute:

- headers, CRC, hashes, index and content closure;
- finite/range/non-clipping quantization;
- decoded shared aprons;
- byte-exact decoded-child parent derivation under every affected 4x4 window;
- bit-exact inherited decoded values outside every corrected-base core/apron mask;
- corrected LOD0 mask and affected ancestor-window dependency closure;
- partition/AOI/worker/order deterministic bytes;
- water/bed/shore compatibility;
- inherited layer presence.

Raw-versus-corrected delta is reported, not failed merely for being nonzero.

### 16.6 Blind rendered beauty gate

Bind `reference/suur-taevaskoda1.jpg`, `1920 x 1278`, SHA-256
`f031ef585dd82779c8e3c8a8059d15aacce11d2e6c886364bee3eb7c2eac382e`.
It is a morphology/composition reference, not registered elevation truth; the
unrecoverable camera and vertical wall may not be treated as metric height truth.

Before blind inference, replace the pending review record in
`asset-gen/config/microtopography-review.toml` and materialize a hashed
`visual-review-v1.json` containing candidate/control manifest hashes, exact URLs,
source snapshot, held-out sites/regimes, every absolute camera and 30 Hz path
sample, FOV/viewport/light/render flags, expected layers, reviewer roster hash,
trial seed, questions, and decision rule. A changed candidate gets a new identity
and a new untouched blind set.

Use the real path with `scene=world`, `src=estonia`, `seed=1`, `freeze=1`,
`mschart=0`, `hud=0`, `dpr=1`, `shadowclipres=896`, `FOV=55 degrees`, and
`1920 x 1080` CSS pixels at device scale 1. Capture at `T=12.5` plus light checks
at `8.5` and `17.5`. Do not disable terrain, water, materials, trees, plants,
understory, debris, or grass. Boot a fresh real-WebGPU context, wait through cloud
bake and ready, settle 48 frames, then align `stats.frame % 1024 == 0`. Also
capture `nanite=1&nanitedbg=cluster` from the identical poses as diagnostic
evidence; beauty acceptance remains the normal full-production view.

Let `O=(x0,z0)=(311123.082,190723.435)`, define from the frozen corrected-only
control `G(x,z)=max(control_heightAt(x,z),control_waterAt(x,z)+0.05)`, and freeze
these control-ground-relative cameras after resolving them to absolute values:

```text
L = (x0,G(x0,z0)+0.50,z0)
yaw(camera,L) = atan2(-(L.x-camera.x),-(L.z-camera.z))
pitch(camera,L) = atan2(L.y-camera.y,hypot(L.x-camera.x,L.z-camera.z))
```

| ID | Position `(x,z)` | Height | Aim |
|---|---|---|---|
| `S0-establish` | `(x0+96,z0+96)` | `G+40.0` | `(x0,G(x0,z0)+0.50,z0)` |
| `S1-near-front` | `(x0+24,z0+28)` | `G+1.70` | same |
| `S2-near-west` | `(x0-28,z0+18)` | `G+1.70` | same |
| `S3-near-east` | `(x0+34,z0-16)` | `G+1.70` | same |
| `S4-grazing` | `(x0-18,z0-32)` | `G+1.70` | same |

Before seeing candidate output, sample `G` from the control at every path point,
resolve/freeze all absolute Y/yaw/pitch values, and then reuse them byte-for-byte:

- `P1-lateral`: 361 inclusive samples over 12 s from `(x0-48,z0+24)` to
  `(x0+48,z0+24)`, `y=G+1.70`, aimed at `L`;
- `P2-dolly`: 481 inclusive samples over 16 s from `(x0+10,z0+16)` to
  `(x0+10,z0+144)`, `y=G+2.00`, aimed at `L`;
- `P3-orbit`: 361 inclusive samples over 12 s at radius 30 m, counter-clockwise
  angle `0..pi` from `(x0+30,z0)`, `y=G+1.70`, aimed at `L`;
- `P4-light`: 301 inclusive samples over 10 s at the frozen absolute `S1` pose,
  setting time `8.5+9*i/300` before sample `i`.

Store PNG or FFV1, not temporally lossy video. Translate the same local rig to
every other blind-site anchor. Taevaskoda is mandatory but not sufficient.

Use nine independent reviewers who did not author a candidate or see blind
outputs. Show synchronized native-size finalist/control beauty, cluster, and
motion panels with deterministic hidden left/right order. The fixed reference is
shown in a separately labeled, non-metric context panel; reviewers answer whether
the representable shore, bank, top surface, and material/process character move
toward it without claiming wall/camera alignment. A tie is a non-win.

Track B compares raw/current base against corrected structural candidate and asks
which has the more credible representable shore/bank/bed while preserving the real
escarpment; it does not compare the candidate against itself as "corrected-only."
A morphology finalist must receive at least 8/9 wins against corrected-only. If a
declared previous accepted morphology owner exists, it must also receive 8/9
against that owner; if none exists, the comparison is absent, not replaced by the
rejected LUKE output. One comparison uses family alpha 0.05; two use
Holm/Bonferroni family-wise alpha 0.05. In either case
`P[X>=8|n=9,p=.5]=0.01953125`. Camera frames are correlated evidence, not
independent votes. The user is the separate final authority and may reject any
statistical pass.

Each reviewer also answers every visible hard-gate item below as `pass/fail` after
viewing all fixed still, cluster, motion, moving-light, and reference-context
panels. Every item requires at least 8/9 passes. Missing layers, failed WebGPU,
decoded seam/normal or parent-closure failure, visible static water displacement,
candidate detail absent in cluster view, or reproducible `P2` pop/crack is an
objective immediate failure. Preference cannot waive either kind of hard failure.

Pass requires:

- the applicable exact structural or morphology comparison rule above passes;
- explicit user acceptance at the fixed Estonia review;
- no visible generic noise, random bumpiness, repeated stamps, sine/quilt pattern, 1 m grid, polygon edge, chunk edge, apron line, coverage seam, or LOD motion;
- correct material/process character and meaningful variance;
- detail visibly changes actual geometric silhouettes, contacts, near-ground
  parallax, grounding, and moving-light response at normal FOV;
- intended forms survive the actual terrain cluster/DAG and are not shading-only;
- smooth corrected shore and absent river artifacts;
- terrain, water, materials, trees/plants, and grass present and grounded together.

Metrics cannot waive a visible hard failure.

Barely visible detail, shading-only detail, static water roughness, or detail
discarded by the actual terrain cluster/DAG is a hard failure.

#### 16.6.1 Research-preview visual gate

The production nine-reviewer/non-inferiority rule above remains mandatory for a
production owner and is not weakened. A research preview uses the same fixed
Taevaskoda cameras, paths, corrected-only control, normal/cluster views, materials,
water, trees, plants, understory, debris, grass, and hard visible-failure list, but
its question is narrower: does this frozen hypothesis add clearly geometric,
non-noise-like, non-repeating, materially plausible forest-floor detail worth
further research without damaging the accepted structural scene?

Taevaskoda is viewed only after candidate and recipe freeze. It is never used to
choose training patches, thresholds, amplitudes, conditions, masks, checkpoint, or
seed. If the preview exposes scene-specific failure, record and reject or park the
recipe; do not tune against Taevaskoda and resubmit it as an untouched test. The
same applies to opened Järvselja audit geometry.

Research visual acceptance requires the objective runtime/layer/continuity gates
and explicit user confirmation that the result is useful to investigate. It may
not use the words `truth`, `target`, `qualified`, `owner`, `pilot`, `national`, or
`accepted morphology` in its decision. A positive result changes only the
candidate to `research_preview_investigated`; a negative result changes it to
`research_rejected` or parks the track with its immutable evidence. Neither result
changes any production regime row or supplies credit to a future bakeoff.

### 16.7 Real runtime gate

For every materialized proof/release, freeze `PREVIEW_REL`,
`EXPECTED_MANIFEST_SHA256`, the complete URL below, and output paths in the review
record. Preflight that `asset-gen/data/out/$PREVIEW_REL` has the expected hash and
that the preview server's `latest.json` resolves to exactly `$PREVIEW_REL`. Start
the servers in separate terminals/process supervisors:

```bash
LAAS_PREVIEW_MANIFEST="$PREVIEW_REL" PORT=8787 node tools/serve-data.mjs
npm run dev
```

Then run this exact real-WebGPU gate after `tools/boot-smoke.ts` implements the
required `--manifest-sha` and `--evidence` assertions:

```bash
npx tsx tools/boot-smoke.ts \
  --url 'http://localhost:5173/?scene=world&src=estonia&dataurl=http%3A%2F%2Flocalhost%3A8787&x=311123.082&z=190723.435&seed=1&freeze=1&mschart=0&hud=0&dpr=1&shadowclipres=896' \
  --settle 48 --w 1920 --h 1080 \
  --manifest-sha "$EXPECTED_MANIFEST_SHA256" \
  --evidence shots/microtopography/boot-evidence.json \
  --out shots/microtopography/boot-smoke.png
```

The harness must fetch/record `latest.json`, the selected manifest bytes and
SHA-256, and fail unless the loaded manifest equals the expected hash. Evidence
JSON schema `laas-webgpu-boot-evidence/1.0.0` requires exact URL, UTC start/end,
git commit, browser/version/adapter identity, preview relative path, expected and
loaded manifest SHA-256, latest response, viewport/DPR, settle frames, final frame,
canvas sizes, screenshot path/hash, all console/page/request/response/WebGPU/TSL
diagnostics including empty arrays, and `pass`. Missing evidence is failure.

For `research-microtopography-preview-v1`, the preview server's `latest.json`
response is an explicit process-local selection shim only. Record the canonical
repository `latest.json` bytes/hash before and after the boot and require them to
remain identical; also require the materialized manifest's
`latest_eligible=false`. The URL and evidence must say `research_only`. A passing
server override is not publication.

Boot through readiness, cloud bake, and settled frames. Fail on page errors, TSL
invalid code, uncaptured WebGPU errors, shader/pipeline/bind errors, asset failures,
manifest mismatch, or missing frames. Then apply the frozen `cam=...,55` absolute
cameras/paths from `visual-review-v1.json`; query-string `cam` values are recorded
byte-for-byte in that artifact. Inspect terrain continuity, water/shore, material
presence, tree/plant/understory/debris presence, and all root grounding.

No shader change is authorized for this synthesis proof. Do not give the user a URL until this exact boot passes.

### 16.8 Testing discipline

Add focused tests only for stable contracts:

- condition/target registry validation;
- deterministic source-repair constraints on small fixtures;
- forbidden/protected masks;
- stochastic partition invariance;
- absolute-surface adapter;
- decoded-child hierarchy and corrected ancestor closure;
- release transaction and dependency identity;
- reproduced source/boot regressions.

Do not build a broad brittle suite around provisional model internals. Tests do not replace real WebGPU boot or visual review.

## 17. Resource And Serving Budget

The zero-runtime-synthesis law accepts stored geometry; it does not make storage free.

Approximate land-area lower-order math over 45,000 km2:

| Quantity | Before compression/replication |
|---|---:|
| LOD -2 samples | 11.52 trillion |
| raw LOD -2 u16 | 23.04 TB |
| LOD -2 chunks | about 2.75 million |
| raw LOD -1 u16 | 1.44 TB |
| raw LOD -2 plus -1 | 24.48 TB |

For area `A` and texel `t`, the no-apron sample estimate is `A/t^2`; exact
publication cost uses chunk count times `2049^2` samples. One raw u16 payload is
`2049^2 * 2 = 8,396,802` bytes plus a 56-byte header. One **negative-rung
one-parent published increment** is `17 * (8,396,802 + 56) = 142,746,586` bytes,
`136.13 MiB` raw; it excludes corrected/repacked LOD0-4 objects and transient work.
The full raw file is `8,396,858` bytes; core-only bytes are `8,388,608`; 4,097
apron samples consume 8,194 bytes. A v2 index record is exactly 21 bytes, so a
one-parent increment is 357 bytes before manifest/object/filesystem/CDN metadata.

The rejected LUKE diagnostic measured `47,181,413` published bytes for 17 chunks
and `23,708,353` additional bytes for nine transient support chunks. Those are
transport diagnostics, not accepted morphology or a compression forecast. At that
rejected-output ratio:

- an aligned `16.384 km` square has 1,024 LOD -1 plus 16,384 LOD -2 chunks,
  about 48.3 GB compressed and 146.2 GB raw;
- the current country rectangle has 386,880 LOD -1 plus 6,190,080 LOD -2
  chunks, about 18.25 TB compressed and 55.23 TB raw.

The corrected Stage 1 structural plan is larger than the old one-parent negative
diagnostic even though it publishes the same review core. Section 10.4's initial
canonical derivation set alone contains 777 transient LOD -2 payloads
(`6,524,315,154` raw bytes) and 45 transient LOD -1 payloads
(`377,856,090` raw bytes), plus 897 paired authority/baseline inputs, before masked
LOD0-4 replacements, source rasters, masters, solver memory, compression scratch,
and retained immutable objects. Measure deduplicated peak disk/RSS and final unique
object bytes; do not reuse the old nine-support/17-file diagnostic as Stage 1
working-set evidence.

The aligned pilot exact raw total is `146,172,504,064` bytes plus a 365,568-byte
**negative-rung index increment**. Its unique outer LOD -2 hierarchy support is
`4*32+4*32+1=257` chunks, `2,157,992,506` raw bytes, before model/higher-rung
scratch. The rectangle exact raw total is `55,225,799,191,680` bytes plus a
138,116,160-byte **negative-rung index increment**. Its outer support is
`4*744+4*520+1=5,057` chunks, `42,462,910,906` raw bytes. Complete release-index
size also includes inherited/base/layer records and must be measured. The rectangle
spans `380,928 x 266,240 m = 101,418.27072 km2`, so
it must not be conflated with the 45,000 km2 ideal land-only bound. The old roughly
8.3 TB estimate is unsafe because it mixed that ideal land area with rejected
smooth-output compression and omitted rectangular closure, aprons, indexes,
transients, replication, and serving. These are planning estimates; accepted
high-entropy detail may compress worse.

Every candidate/release must report:

- training and inference GPU/CPU hours;
- peak host/GPU memory;
- windows/chunks per hour;
- packed bytes per chunk and per regime;
- p50/p95/p99 packed bytes and entropy by regime and qscale;
- object count and index/metadata overhead;
- target/orthophoto working-set size;
- support/overlap multiplier, retries/checkpoints, scratch/transient bytes, and
  CPU/GPU hours per square kilometer;
- upload, retained-build replication, CDN/cache, egress, and recook cadence/cost;
- measured network demand at the existing runtime bands.

For the complete transaction, separate unique new immutable object bytes, inherited
referenced bytes, logical merged-release bytes, transient hierarchy-support bytes,
peak scratch/working-set bytes, and retained failed/retry bytes. The 17-file figure
may appear only in the negative-rung increment column.

For sorted actual observations `x[1..n]`, p50/p95/p99 mean nearest-rank
`x[ceil(p*n)]`; report `n`, mean, min, and max too. One packed observation is an
actual immutable chunk file including its header. Also report
`8*(file_bytes-56)/4,198,401` bits/sample, stratified by LOD and exact
regime/transition signature. Label p99 `insufficient-n` below 1,000 observations.
Totals use exact sums or mean times an explicit count, never p50 times count.

Emit a machine cost ledger separating identity; requested/chunk-snapped coverage;
training; inference; model-overlap multiplier; source repair, conditioning,
synthesis, composition, hierarchy, compression and verification phases; retry and
checkpoint waste; sources/targets/models/masters/memmaps/staging/scratch; packed
entropy; retained-build replication; upload/origin/CDN/browser egress; and partial
versus full recook cadence/cost. Bin compute into anchor-aligned 1 km cells and
apportion duplicated window work by output-core area; report nearest-rank
p50/p95/p99 plus total wall seconds, CPU core-seconds, accelerator seconds, source
bytes, scratch bytes, and output bytes per km2 by regime. Fixed cold/warm boots,
all review paths, and a preregistered 10 km traversal use the same nearest-rank
definition for p50/p95/p99 bytes, requests, latency, decode time, and cache hit rate
by layer/LOD; p95 needs 100 sessions and p99 1,000, otherwise label them
insufficient.

Quality gates precede cost. Cost breaks ties between quality-equivalent survivors; it may not excuse visibly worse terrain. National coverage shape and retained-build policy require explicit user approval before a national cook.

## 18. Staged Work After Spec Approval

### Stage 0: freeze evidence and interfaces

Deliver:

- final source/claim ledger and hashes;
- condition schema and target registry schema;
- regime matrix/release-row schema;
- absolute-surface specialist adapter;
- corrected-authority and hierarchy plan;
- preregistered proof cameras, metrics, and failure rules.

No morphology quality claim is made.

### Stage 1: cheapest decisive proof

Run Tracks A and B. Track C is optional research and is not part of the cheapest
acceptance path.

**Track A: public evidence closure and qualification specification**

- complete the Moore byte-level validation audit under Section 12.7 and record the
  known `target_evidence_insufficient` ownership result without training a model;
- write `PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md` and its condition/target/
  physical-analogue schemas; freeze selective files rather than downloading whole
  terabyte releases;
- register and audit all eight exact-site Taevaskoda ALS epochs for Track B, then
  qualify the Hovi Järvselja/Hyytiälä subset as the first raw-surface conversion
  candidate;
- freeze a qualified repair-calibration manifest for Track B or abstain on every
  unsupported correction class.

**Track B: Taevaskoda typed repair**

- fetch/version Taevaskoda RGB/CIR;
- reconstruct water surface/bed distinction, smooth shoreline, bank/shoulder, mapped escarpment, and diagnosed river bumps;
- publish structural-only LOD -2/-1, corrected full LOD0 authority, and affected
  ancestors through the existing negative-rung interface;
- keep unresolved morphology absent in this track;
- prove the stepped shore and source bumps are gone at ground level without flattening the real bank/escarpment.

**Track C: optional Moore engineering screen**

- materialize and gate the exact 2026-07-15 narrow M0 machine design before model
  work; if its source-derived B2 fails, produce no `<=0.1 m` Moore preview;
- after a separate reviewed hash-bound candidate-selection supplement authorizes
  training, exercise only `D`, `R`, and `E` against corrected-only; M1 and the
  other Section 12.7 families remain non-authorizing diagnostics;
- report parent preservation, subphase sensitivity, tiling, packing/DAG survival,
  compute, and qualitative failure modes through the plot-scaled protocol;
- make no morphology non-inferiority, owner, Estonia-transfer, or production claim.

Stage 1's decisive deliverable is Track B's visible structural repair plus the
public-evidence closure. Track C is optional and cannot delay or inflate that
proof. No **production** morphology owner selection begins until qualified Estonia
or Section 9.3 analogue targets and a user-approved target-specific
preregistration exist. The 2026-07-14 research authorization permits only Stage
2R below; it does not retroactively change Stage 1 evidence results.

### Stage 2R: first weak-evidence forest research specialist

This is the active bounded visual/model research path:

1. freeze `forest-weak-research-bundle/1` and one machine preregistration with the
   exact Section 9.7 roles, surface contract, bands, splits, candidates, budgets,
   seeds, metrics, failure rules, preview extent, and corrected-only control;
2. develop semantic probabilities and `p_unknown` using only ForestSemantic-MS
   `train1`-`train4`; freeze the classifier/thresholds, then run `test1`-`test2`
   once as the semantic audit;
3. construct `HY_SPRUCE4` `ResearchSurfaceEvidence` from its full scans with direct
   support, view/disagreement, semantic, unknown, forbidden, and B1/B2 eligibility
   fields; no filled or interpolated cell becomes supervision;
4. train the preregistered high-information candidate subset on only eligible
   `HY_SPRUCE4` blocks, preserving corrected-only and all anti-memorization gates;
5. select a development survivor by the frozen research rule, then freeze its
   complete recipe/checkpoint/seed before final Evo `1086`/`1065`, HY_PINE2, or
   Järvselja evaluation;
6. run Evo as independent-sensor/campaign geometry evidence, HY_PINE2 as OOD/
   abstention stress, and open `JS_SPRUCE1` exactly once as the sealed Estonia
   generalization audit. Any later changed recipe needs a new untouched site;
7. if and only if every role-specific audit passes, compose and pack one
   `research-microtopography-preview-v1` candidate over a bounded parent-closed
   Taevaskoda review extent, retaining accepted structure and all inherited layers;
8. independently verify, boot the exact immutable non-`latest` manifest in real
   WebGPU, emit relevant height/band/confidence/unknown/nearest-copy PNG evidence,
   and provide the research-only URL for user visual investigation.

Failure parks the exact recipe and evidence; it does not trigger scene tuning,
generic filler, another broad dataset hunt, or relaxation of production gates.
Success proves only that the hypothesis is worth investigating visually.

### Stage 2E: erodible-slope R0 process research

This separately authorized path does not alter or inherit the forest bundle:

1. resolve every blocker and freeze the machine preregistration under
   `asset-gen/config/microtopography/erodible-slope/` with the exact Section 9.9
   roles, sources, masks, physical domains, C0/C1 configs, units, budgets, gates,
   preview extent, cameras, and corrected-only control;
2. preserve rejected Development B as strict-abstention evidence; use only the
   bound Development A and corrected Development C condition domains for
   C0/C1, retain the old Development C crop/candidate as condition-abstention
   evidence, and crop only after each whole-domain solve;
3. reject candidates that violate protected features, typed masks, network/outlet
   topology, conservative accounting, rotation/partition identity, one-meter grid
   leakage, recognizable-form, cost, or packing-survival gates;
4. freeze the surviving recipe, seed, condition envelope, abstention rule,
   thresholds, and visual rule before opening OOD2; Orajõgi remains consumed and
   disqualified;
5. run OOD2 once as OOD abstention stress with zero positive-evidence credit;
   failure rejects the exact recipe and any change requires a new sealed site;
6. only after every frozen gate passes, materialize one immutable non-`latest`
   research preview through Section 14.6, complete the real-WebGPU gate, and
   provide its manifest-bound URL for visual investigation.

The output remains an R0 process hypothesis and the regime remains production
`unsupported`. No result in this stage satisfies Section 9.3 or selects an owner.

### Stage 2P: public qualification and first production regime

- convert/qualify the selective Hovi Järvselja/Hyytiälä forest subset first;
- qualify a small Evo 2024 holdout next, then selective FORWARD/Biała files only
  where their exact physical regime is required;
- send one no-cost, non-blocking permission request for the Kohv/Shlykova
  Taevaskoda/Väike/Härma/Selisoo archives; refusal or silence preserves abstention;
- pair every site to the full condition fabric;
- leave Taevaskoda outside production training, tuning, and candidate selection;
- rerun owner selection using Estonia targets or Section 9.3's explicit
  neighboring-analogue transfer contract;
- activate only the first regime that passes all gates.

If `JS_SPRUCE1` is opened in Stage 2R, it is no longer an untouched site for a
later changed production recipe. Stage 2P must either preregister it before that
opening without subsequent tuning or acquire another genuinely untouched Estonia
site. This cost does not waive Section 9.3's independent-site/campaign minimum.

### Stage 3: parent-closed Estonia production visual pilot

- replace the Stage 1 structural-only fine master with a morphology-bearing master
  only after a matching Estonia-supported regime exists;
- derive corrected LOD0 and splice only affected ancestor windows;
- recook dependent water/vegetation content;
- materialize immutable preview;
- boot and provide the verified live URL;
- pause for user visual judgment.

General improvements derived from the reference must apply to the same physical regime elsewhere. No Taevaskoda-specific constants or masks.

### Stage 4: regime expansion

Expand by national prevalence and visible value, not algorithm convenience. Each
new row repeats public-data qualification, bakeoff, blind holdout, packed cost,
and user-visible review.

### Stage 5: wide/national publication

Before cooking:

- close all common-regime gaps required by the product claim;
- generalize multi-parent coverage/release tooling;
- approve coverage geometry, storage, replication, and recook budget;
- run a bounded multi-parent pilot;
- verify demand/cache/network behavior without runtime synthesis;
- obtain explicit user authorization.

## 19. File-Level Implementation Map

This is a map, not implementation authorization.

### 19.1 Replace provisional synthesis internals

Under `asset-gen/src/assetgen/process/microtopo/`:

- replace the exemplar-bank/residual facade in `api.py` with typed
  `ObservationEvidence`, `ResearchSurfaceEvidence`, `StructuralAuthority`,
  `SpecialistOutput`, and `FinalMaster` absolute-surface contracts; no implicit
  raw-parent projection may remain;
- replace the rejected quilt implementation in `synthesis.py`;
- add condition, target-registry, repair, regime-graph, composer, and stochastic-field modules;
- add `experts/` adapters for exemplar, conditional simulation, process/event, regressor, GAN, diffusion, and hybrid candidates;
- add cohesive research-surface, confidence-calibration, leakage, audit, and
  candidate-state modules; keep them separate from production target qualification;
- keep research-only candidates unavailable to production. Even the same model
  code/checkpoint must pass a future qualified-target bakeoff and signed production
  row before it can become a production owner.

Do not retain `asset-gen/src/assetgen/process/micro_synth.py` as a fallback.

### 19.2 Cook and hierarchy

- change `cook/micro_synth_cook.py` from residual-only/raw-parent projection to absolute master orchestration;
- generalize `cook/micro_hierarchy.py` for complete decoded-child `-2 -> -1 -> 0`
  closure and masked affected-window splicing into independent inherited LOD1-L4;
- reuse `cook/encode.py` checked quantization and browser-equivalent decode;
- reuse `cook/pinned_height.py` against the newly corrected base;
- extend `cook/height_cook.py` or add a corrected-base cook without changing payload encoding;
- update `assetgen/cli.py` so recipe builds never invoke the legacy mutating
  `cook_waterbed` after height generation;
- retire/forbid the LOD0-mutating waterbed branch in `cook/layers_cook.py` for both
  new recipe kinds. Bed is composed into `StructuralAuthority` before the master
  and hierarchy; the later `waterY`/`watercover` cook is read-only with respect to
  height;
- set and verify `WATERBED_FLAG` on every corrected wet LOD0 payload so legacy
  tooling and future recooks cannot carve it twice; define/preserve that flag in
  `assetgen/manifest.py` and height header semantics without changing wire size;
- update the remaining `cook/layers_cook.py` water and slope-derived
  understory/debris dependencies so they consume the accepted corrected snapshot.

### 19.3 Inputs and configuration

- add official orthophoto and EGT fetch modules under `assetgen/fetch/`;
- make soil acquisition a real CLI option or remove the false advertised command;
- add full-fidelity cook-only soil/condition parsing without changing runtime layer formats;
- replace the current exemplar-only micro configuration with versioned source, regime, target, expert, and proof configuration;
- add a separately typed `forest-weak-research-bundle/1` configuration and
  `microtopography-research-surface/1.0.0` validation; do not overload the target
  registry with weak evidence;
- add the separately typed `erodible-slope-research-bundle/1` R0 process
  preregistration under `config/microtopography/erodible-slope/`; any unresolved
  required input must remain a machine-visible generation blocker;
- keep physical thresholds per regime/evidence; do not add one national roughness knob.

### 19.4 Recipe, verification, and release

- update `micro_recipe.py` to bind every new source/model/registry/environment artifact;
- replace raw-DTM and rejected-generator gates in `micro_verify.py`;
- replace the old recipe-kind/fixed-base/fixed-17 dispatch in `release.py` with the
  exact Section 14.6 production schemas, two-artifact transaction, and hard-isolated
  `research-microtopography-preview-v1` non-`latest` path;
- update `assetgen/manifest.py` and expectation parsing for claim-domain,
  corrected-core/support/apron/affected-ancestor masks, windows, outside proofs,
  and dependency identities;
- update `tools/serve-data.mjs` and `tools/boot-smoke.ts` only for immutable
  manifest selection/assertion and evidence output described in Section 16.7;
- require release publication code to reject `latest_eligible=false` manifests,
  and test that the canonical `latest.json` bytes remain unchanged around a
  research preview boot;
- later generalize one-parent coverage metadata/tooling before wide coverage;
- leave LAC payload bytes, v2 index bytes, data server, and runtime decoding unchanged.

### 19.5 Runtime

One minimal representation-sampling repair is authorized; no synthesis, shader,
format, decoder, cache, or `TerrainField` provenance branch is authorized.

- In `src/nanite/world/ChunkContent.ts`, replace LOD0-only grounding for streamed
  trees/boulders with the same generic packed finest/morph height sampler and
  coverage fallback used by terrain/grass.
- In `src/gpu/passes/UnderstoryScatter.ts`, remove hard-coded `{lod:0}` grounding
  and use that same representation sampler for understory/debris.
- Instance grounding must refresh or be evaluated when the packed-level morph
  center/ownership changes; caching one initial Y is invalid. A point exactly on a
  band boundary must receive the same height and transition weight as terrain at
  that world coordinate.
- Preserve deterministic instance identity, X/Z, density, materials, culling,
  scatter ownership, and existing no-fine-coverage fallback. Only root/base Y and
  any slope/normal derived from terrain may change.
- Prove terrain/grass/tree/boulder/understory/debris root agreement at fixed poses
  and while crossing both negative-rung transition rings. No consumer may float,
  sink, pop, or retain an elevated island.

If this cannot be implemented through the existing generic packed representation,
stop for a separate architecture review. Shader edits remain out of scope.

## 20. Explicitly Rejected Alternatives

- runtime procedural micro-displacement: violates zero-browser synthesis;
- global 128 m LOD0 rewrite: corrupts the shared world grid;
- exact raw-DTM DDNM projection: preserves a false observation model and diagnosed defects;
- universal diffusion backbone: no target corpus or demonstrated centimeter prior;
- universal regressor: averages one-to-many variation;
- universal GAN/theme model: sharpness and themes are not physical distribution coverage;
- universal sparse dictionary: similar exemplars and global coherence are missing;
- universal process library: partial models become ungrounded outside calibrated domains;
- generic geostatistical roughness: spectra alone produce noise;
- orthophoto displacement: albedo/objects/shadows/water are not height;
- two x4 networks mandated by storage rungs: packed layout is not model evidence;
- independent per-chunk generation or seeding: creates phase and seams;
- nearest supported-regime fallback: falsely fills unsupported Estonia;
- millimeter features because the grid is fine: below representable/evidenced support;
- heightfield-only Taevaskoda wall: topologically impossible;
- binary/runtime/shader redesign for the next proof: unnecessary and high risk.

## 21. User Decisions And Checkpoints

Defaults recommended by this spec:

- architecture: evidence-gated typed hybrid;
- orthophoto: fetch dated RGB/CIR for research and likely production;
- corrected authority: publish through ordinary corrected LOD0 and ancestors;
- finest geometry: stored LOD -2/-1, no runtime synthesis;
- first research morphology investigation: the exact
  `forest-weak-research-bundle/1` path in Stage 2R, with no truth, owner, transfer,
  or release claim;
- next R0 process investigation: the exact
  `erodible-slope-research-bundle/1` path in Stage 2E, with OOD2 reserved for
  abstention-only stress and no positive-evidence credit; Orajõgi is consumed;
- first production morphology proof: whichever exact regime the zero-budget
  public-target protocol qualifies first; Hovi forest remains a conversion
  candidate, not a preselected owner;
- first visible source proof: Taevaskoda shore/bank/river correction;
- vertical cliff face: explicitly rejected/out of scope, not a false heightfield claim.

The 2026-07-13 user review closed these decisions:

- paid acquisition, proprietary data, new hardware, vendors, and separately funded
  projects are rejected unless the user explicitly reopens budget;
- neighboring Baltic/Nordic and Estonia public data must be exhausted and may
  qualify regime-specific transfer under Section 9.3;
- no separate non-heightfield cliff project is commissioned.

The 2026-07-14 user review additionally closed these decisions:

- incomplete survey-truth/error evidence may be used for evidence-tiered,
  probabilistic research supervision and visual/model investigation;
- the exact first research bundle and roles are frozen by Section 9.7;
- research output may be packed only as an immutable, explicitly non-`latest`
  browser preview and cannot become target truth, a production owner, or evidence
  of Estonia transfer.

The 2026-07-15 review additionally authorizes the bounded Stage 2E R0 process
path in Section 9.9. It does not relax its explicit input blockers, qualify
erodible-slope morphology truth, or change the production regime row.

The remaining product decision is wide/national fine-coverage and retained-build
storage policy after measured pilot entropy.

Required checkpoints:

1. architecture review completed 2026-07-13;
2. user reviews the Stage 1 corrected-source live URL;
3. user reviews the first verified Stage 2R research-only live URL without that
   review changing production status;
4. user reviews any verified Stage 2E R0 research-only live URL without that
   review supplying positive morphology or production evidence;
5. user reviews the first public-target-qualified production morphology regime;
6. user explicitly approves any multi-parent/national cook.

## 22. Definition Of Done

The task is complete only when:

- every expected claim-domain cell has exactly one valid terminal: released
  supported morphology (`R`), equivalence-proven smooth (`S`), validated
  hard-structure/open-water exclusion (`F`), or declared non-heightfield exclusion
  (`N`); a full morphology claim has no `U` or `A`;
- every claimed common Estonia regime has target-supported release evidence;
- source errors are corrected through a published corrected hierarchy;
- the finest surface visibly adds realistic, varied, materially/process-correct geometry;
- unsupported regions contain no generic filler;
- water, shore, bank, bed, and escarpment remain correctly typed;
- one-meter grid, chunk, apron, regime-boundary, repetition, and LOD artifacts are absent;
- terrain, water, materials, trees/plants, and grass agree on the packed surface;
- the accepted runtime representation plus the narrowly authorized shared
  instance-grounding fix boots cleanly on the exact immutable Estonia URL;
- packed cost and national serving policy are approved;
- the user accepts the rendered result.

A cook that passes numerical gates but looks like noise, random bumpiness, a quilt, or a smoother triangulation is not done.

### 22.1 Research checkpoint completion

Stage 2R is complete only when the exact bundle, probabilistic surfaces, band
eligibility, preregistration, leakage audit, frozen candidate state, semantic and
cross-campaign/stress/Järvselja audits, anti-copy evidence, relevant labeled PNG
diagnostics, immutable pack, exact real-WebGPU boot, and user visual investigation
exist under one recipe hash. Its terminal is
`research_preview_investigated` or `research_rejected`, never project Definition
of Done. It leaves all production claim cells `U`, all production owner fields
null, all release rows unchanged, and canonical `latest.json` unchanged.

### 22.2 Erodible-slope R0 checkpoint completion

Stage 2E is complete only when the exact bundle/preregistration, resolved input
blockers, Development A/C whole-domain C0/C1 evidence, frozen recipe and
abstention rule, one-shot OOD2 stress result, relevant labeled PNG diagnostics,
immutable pack, exact real-WebGPU boot, and user visual investigation exist under
one recipe hash. Its terminal and production effects are identical to Section
22.1, but it carries no weak-surface, positive-audit, target-fit, or transfer claim.

## 23. Normative Research Record

### 23.1 Machine source-evidence gate

`docs/deep-research/microtopography-generation/library/sources.json` must validate
as `microtopography-source-ledger/2.0.0` before any literature record can authorize
a specialist owner. One `records` array uses the schema's exact kinds `paper`,
`dataset`, `repository`, `official_dataset`, `official_data_service`,
`official_license`, `data_description`, and `vendor_announcement`; related records
link by stable ID. Every record requires:

- complete citation authors/title/year/venue/DOI, with explicit unknown rather
  than inferred `et al.` expansion;
- canonical URL and related-record IDs;
- checked access date/status/result/full-text basis plus flags for `invalid_html`,
  `code_only`, `analogue`, `unresolved`, or `retracted`;
- exact review status/scope and whether consequential judgment is allowed;
- every local artifact path, SHA-256, media type, and validation result;
- independent license status/name/SPDX/URL/scope plus training and derivative-use
  state; a paper license never silently licenses data or code;
- software officialness, repository URL, exact 40-hex revision, snapshot path/hash,
  and code license;
- normalized data and checkpoint availability and artifact/record IDs;
- demonstrated task, geography/regime, metric input/output sample pitch/support/
  extent, degradation or sensor observation, training/split unit, and evaluation;
- separate direct finding, limitation, permitted project transfer, and prohibited
  inference; project architecture judgments are labeled inference.

Every local hash must match `metadata/SHA256SUMS`; every repository snapshot must
match its pinned revision/hash. `unknown`, `absent`, `paywalled`, `request_only`,
or `unavailable` are truthful values, but block any reuse that needs the missing
fact. Preserve the v1 ledger hash as migration provenance. The uncited Burren
clint/grike dimensions remain excluded; the two Argudo HTML masquerades remain
quarantined invalid artifacts rather than evidence.

The audited source-evidence snapshot contains 129 unified records including 21 pinned
repositories. All 141 checksum-manifest artifacts are linked exactly once as
canonical, duplicate, support, or quarantined evidence. Ledger SHA-256 is
`812988e5605daf15c4e0ace3f0615ee97741637450b12f072797a005f4fb17b6`;
`sources.schema.json` SHA-256 is
`8d6911d62602d087839f8aa5eb05cc392cb1b91df8df74ff661e9e1b41604e39`;
the 141-artifact integrity-manifest SHA-256 is
`30bd0c0ee4d0aa4a4f4f8a03a28844a982df4f44f62fdd29feeca6d6f1ab4aac`.
Only ten records currently allow consequential judgment; the other 119 do not.
Unknown authors/licenses/officialness/code-review/checkpoints/scales remain
blocking, not inferred success.

### 23.2 Required reading

The implementer must read these before code:

- `docs/deep-research/microtopography-generation/r1.md`;
- `docs/deep-research/microtopography-generation/r2.md`;
- `docs/deep-research/microtopography-generation/review/ROOT-CLAIM-AUDIT.md`;
- `docs/deep-research/microtopography-generation/review/ML-PAPER-NOTES.md`;
- `docs/deep-research/microtopography-generation/review/ML-SYNTHESIS-PROPOSAL.md`;
- `docs/deep-research/microtopography-generation/review/CRITIQUE-ML-PROPOSAL.md`;
- `docs/deep-research/microtopography-generation/review/deterministic-process-primary-source-audit.md`;
- `docs/deep-research/microtopography-generation/review/deterministic-process-proposal.md`;
- `docs/deep-research/microtopography-generation/review/DETERMINISTIC-PROPOSAL-CRITIQUE.md`;
- `docs/deep-research/microtopography-generation/review/ESTONIA-CONDITIONING-DATA-AUDIT.md`;
- `docs/deep-research/microtopography-generation/review/TARGET-SCALE-DATA-AND-REGIME-AUDIT.md`;
- `docs/deep-research/microtopography-generation/review/REGIME-PHENOMENON-MATRIX.md`;
- `docs/deep-research/microtopography-generation/review/COOK-CONTRACT-AND-HIERARCHY-AUDIT.md`;
- `docs/deep-research/microtopography-generation/review/COMPARATIVE-ARCHITECTURE-DOSSIER.md`;
- `docs/deep-research/microtopography-generation/review/REGIME-EVIDENCE-EXECUTABLE-CONTRACT.md`;
- `docs/deep-research/microtopography-generation/review/HIERARCHY-VISUAL-COST-EXECUTABLE-CONTRACT.md`;
- `docs/deep-research/microtopography-generation/review/STAGE1-EXECUTABLE-METHOD-CONTRACT.md` (superseded negative-design record; do not implement);
- `docs/deep-research/microtopography-generation/review/BALTIC-PUBLIC-TARGET-DATA-HUNT.md`;
- `docs/deep-research/microtopography-generation/review/NORDIC-PUBLIC-TARGET-DATA-HUNT.md`;
- `docs/deep-research/microtopography-generation/review/TAEVASKODA-ESTONIA-PUBLIC-SCAN-HUNT.md`;
- `docs/specs/terrain/PUBLIC-TARGET-QUALIFICATION-PROTOCOL.md`;
- `docs/deep-research/microtopography-generation/review/contracts/regime-release-registry.schema.json`;
- `docs/deep-research/microtopography-generation/review/contracts/regime-release-rows.initial.json`;
- `docs/deep-research/microtopography-generation/review/contracts/regime-initial-status.json`;
- `docs/deep-research/microtopography-generation/library/SOURCES.md`;
- `docs/deep-research/microtopography-generation/library/sources.json`;
- `docs/deep-research/microtopography-generation/library/sources.schema.json`;
- `docs/deep-research/microtopography-generation/library/metadata/SHA256SUMS`.

Key primary sources and boundaries:

- [Guérin et al. 2016](https://doi.org/10.1111/cgf.12821): paired sparse exemplar amplification, not a universal generator.
- [Argudo et al. 2017](https://doi.org/10.1007/s00371-017-1393-6): coherent multilayer exemplar synthesis, not the orthophoto FCN.
- [Argudo et al. 2018](https://doi.org/10.1111/cgf.13345): optical-guided 15 m-to-2 m inference with canopy/shadow limits.
- [Zhao et al. 2019](https://doi.org/10.1145/3355089.3356553): GATA x4 learned terrain themes, not centimeter Estonia.
- [Wang et al. 2023](https://arxiv.org/abs/2212.00490): DDNM consistency for an explicit operator, not a terrain prior.
- [Bar-Tal et al. 2023](https://arxiv.org/abs/2302.08113): overlap consensus, not physical terrain continuity.
- [Schott et al. 2024](https://doi.org/10.1145/3658200): strong multiscale hydraulic specialist, not universal material morphology.
- [Doane et al. 2024](https://doi.org/10.1029/2024AV001264): roughness event production/decay, not Estonia event parameters.
- [Moore et al. 2019](https://doi.org/10.5194/bg-16-3491-2019): measured peat surfaces, not national transfer.
- [Pawlik et al. 2024](https://doi.org/10.1016/j.geomorph.2024.109283): target-scale forest pit/mound evidence without an open national corpus.
- [Panangian and Bittner 2024](https://doi.org/10.5194/isprs-annals-X-2-2024-185-2024): real low-resolution elevation differs from ideal synthetic degradation.
- [Paris et al. 2019](https://doi.org/10.1145/3342765): implicit 3D terrain shows the heightfield topology limit.

These sources constrain architecture and evaluation. The three public-data hunt
reports are a reviewed zero-budget amendment to the Stage 0 ledger. Before any
candidate artifact can authorize a specialist owner, add its machine-readable
record, license, hashes, conversion provenance, and qualification result to the
source ledger. Public foreign targets may support only the bounded analogue
transfer in Section 9.3; none waives Estonia out-of-distribution checks, regime
abstention, or visible acceptance.
