# Deterministic/Process Track Proposal

Status: architecture proposal from the deterministic/process primary-source audit, 2026-07-13. It is not an implementation spec and does not authorize code changes.

## Recommendation

Use a **learned, dense, physically conditioned asset-gen synthesizer as the national backbone**, preceded by explicit confidence-aware source repair and followed only by narrow calibrated process/event modules. Do not build a deterministic dictionary/process-only replacement for the rejected patch quilt. The deterministic literature does not demonstrate realistic 0.0625 m coverage across heterogeneous national terrain; using it as the backbone would reproduce the same mistake with more elaborate procedural shapes.

The existing transport architecture remains fixed:

- LOD 0 stays the 1 m/2048 m canonical base;
- LOD -1 remains 0.25 m/512 m;
- LOD -2 remains 0.0625 m/128 m;
- the accepted fine-rung coverage, packing, streaming, decoding, and ordinary terrain sampling stay;
- all generation remains in `asset-gen`;
- runtime synthesis remains forbidden;
- the cooked output remains an ordinary packed height surface; no binary-format or runtime provenance path is required.

Only the cook's surface-generation method and its internal conditioning data change. In the current production boundary, the morphology owner is effectively `process.microtopo.synthesize_residual`; conservative reconstruction, hierarchy, quantization, and packing are separate. The replacement should keep that separation where possible, but the current hard one-meter mean-null constraint must not erase justified source correction.

## Why This Recommendation Follows From The Literature

The audited methods solve different, smaller problems:

- Guérin 2016 and Argudo 2017 transfer local exemplar patches when a similar exemplar already exists. They do not infer centimeter morphology from Estonia's soil/geology/hydrology and can repeat or transition sharply.
- Argudo 2018 proves that registered orthophoto can improve terrain inference, but only from 15 m to 2 m and with explicit canopy/shadow failures.
- Scott and Dodgson 2021 repair drainage pits in macro terrain, not fallible one-meter shorelines; global pit removal would erase valid bog, glacial, and karst depressions.
- Paris 2019 creates sparse volumetric cliffs/caves at about 10 cm, but its defining overhangs cannot survive the frozen heightfield and below-10-cm detail is future work.
- Cortial 2020 reaches about 50 cm with semantic GPU subdivision and generic random rules; it is not a centimeter raster cook.
- Schott 2024 is credible for hydraulic/thermal landforms but has whole-watershed dependencies, patch-boundary limitations, high iteration cost, and one process family.
- Moore, Pawlik, HuHoLa, and Ilyasov establish measured regime-specific structure. They do not provide a national generator.

The literature does support a learned backbone conditioned by causal evidence, because nearest-dictionary lookup and manual process tables cannot smoothly model the cross-product of substrate, parent material, soil profile, water, vegetation/disturbance, land use, and landform. "Learned" does not mean unconstrained image hallucination. It means learning the conditional distribution of measured geometric residuals while deterministic data and process logic control eligibility, confidence, continuity, and provenance.

## Proposed Cook-Side Surface Model

### Stage 0: assemble a lossless conditioning record

For every synthesis window, produce aligned condition tensors and confidence masks at their honest support scales. Coarse polygons must remain coarse priors; they must not print their edges into 6.25 cm geometry.

Required conditions:

1. Maa- ja Ruumiamet 1 m DTM plus local acquisition/source identity where available.
2. DTM/nDSM/CHM agreement and disagreement, missing-return/interpolation evidence, local grid/artifact descriptors, and a per-cell observation confidence.
3. ETAK water polygons, shoreline, river centerline, natural escarpment, ditch, road, quarry, wetland, peat field, land cover, and mapped boulder structures.
4. Full Mullastikukaart composition: `Sif1..Sif4`, `Osa1..Osa4`, `Loimis1`, `Loimis2`, `Lihtloimis`, `Huumus`, `Kivisus`, and `Boniteet`, preserving source typography/subscripts and unknown values.
5. EGT 1:50,000 bedrock/surficial/geomorphology with an explicit coverage mask.
6. EGT 1:200,000 bedrock/surficial fallback with low spatial authority and decoded official classes.
7. Dated RGB/CIR orthophoto with canopy, water, shadow, snow/season, and temporal-mismatch confidence.
8. Forest stand composition/age/disturbance evidence, agricultural/management state, and other land-use context.
9. Multi-scale DTM-derived landform descriptors and hydrological context.
10. Exemplar provenance, acquisition method, physical class, support scale, ground-filtering QA, and transfer-confidence label.

The current soil cook is insufficient because it discards secondary soil components and shares, layered texture, and humus thickness. Expanding this internal conditioning record is an `asset-gen` input change, not a streamed-format change.

### Stage 1: estimate a corrected structural base

Produce `base_corrected` and `base_confidence` on a synthesis master grid. This stage is allowed to disagree with the 1 m DTM where independent evidence says the observation is wrong.

It must address at least:

- grid-stepped shorelines and water interpolation;
- isolated spikes/bumps from bad returns or filtering;
- vegetation/building leakage into the DTM;
- holes and interpolated strips;
- bank/escarpment placement supported by ETAK and imagery;
- roads, ditches, quarries, and other mapped structures whose geometry should remain coherent.

Use hard constraints only for authoritative topology or placement, such as water occupancy and mapped linear structures with known confidence. Use soft constraints for uncertain offsets, slopes, and imagery cues. Preserve natural depressions unless a hydrological/vector constraint specifically contradicts them.

At Suur Taevaskoda, ETAK places a natural shoreline escarpment 5.32 m from the supplied point and a clear Ahja shoreline/river boundary 1.85 m away. That is direct evidence for reconstructing a smooth bank/shore structure instead of preserving the DTM's one-meter staircase. EGT 1:50,000 has no local polygon coverage there; the 1:200,000 Burtnieki sandstone/siltstone/clay-interbed class is only a broad material-family prior, not license to place local sandstone ledges from a polygon edge.

The source-repair stage may be learned, optimization-based, or hybrid. Its acceptance criterion is corrected visible structure and held-out target agreement, not zero one-meter error.

### Stage 2: synthesize the dense conditional surface residual

Train a model to sample `detail_residual` at 0.0625 m from the corrected base, conditions, confidence masks, and a deterministic global seed. The ML proposal should choose the final architecture, but this track imposes these requirements:

- output is metric height, not an RGB texture converted to height;
- model context must cover the physical structure controlling the output, not just one output chunk;
- conditions enter as continuous fields with confidence, not a single hard class ID per chunk;
- target training examples must genuinely resolve the claimed scale;
- the model learns joint morphology across bands instead of adding generic noise after interpolation;
- cross-condition transitions are learned/mediated over physical distances rather than hard polygon edges;
- deterministic global coordinates make recooks and overlapping windows identical for a fixed recipe;
- multiple stochastic samples are available during research, but the production recipe binds one accepted seed/checkpoint/input identity.

The dense model must represent every supported regime with actual structural variance. It should not collapse unsupported regions to smooth interpolation merely to avoid mistakes, nor fill them with decorative FBM. A regime without adequate target data is explicitly low-confidence and remains a blocker for national release.

### Stage 3: add only validated event/process modules

Event modules are not a way to compensate for a weak backbone. Add one only when measured Estonia/analogue data establish its geometry and eligibility.

Initial candidates:

- **Forest pit-mound:** paired, oriented pit/mound events conditioned on tree/stand history, slope, soil depth/texture/moisture, substrate, wind/disturbance likelihood, age, and decay. Calibrate from target-scale TLS, using Doane and Pawlik as form/process evidence.
- **Peat microforms:** likely learned/exemplar-based rather than hand-sprinkled. Condition on mire type, hydrology/water table, ridge/hollow hierarchy, peat depth/humus, and vegetation. Use Moore's 1 cm archive as foreign calibration/holdout evidence and HuHoLa/Ilyasov only as labels/diagnostics.
- **Erodible gullies/rills:** consider Schott-style multiscale erosion only inside eligible slopes/materials and with watershed-scale context. It is not part of the cheapest proof and must never affect water, bog, rock, field, or generic flat land by default.
- **Agricultural structure:** require management/event state and direction from imagery/parcel evidence. Do not substitute per-chunk sine furrows.
- **Exposed rock/weathering:** condition on observed exposure, bedrock/material, fracture evidence, aspect, moisture, and slope. A heightfield can add ledges and grooves but cannot claim Paris-style caves/overhangs.

Each module emits both geometry and an evidence record: source calibration set, eligibility probability, parameters/latent identity, seed, and rejection/fallback reason. No global roughness multiplier is a feature.

### Stage 4: compose one master surface, then derive the hierarchy

Compose `base_corrected + dense_residual + eligible_event_residuals` on one global-coordinate 0.0625 m master surface with sufficient halo/context. Crop packed LOD -2 chunks from that shared result and derive coarser fine rungs from the accepted master through the existing hierarchy path.

Chunk identity must never influence material class, orientation, phase, or feature selection. Chunking is an execution/storage partition only. Overlap consensus or latent tiling may be used inside the model, but acceptance includes value, gradient, normal, feature, and semantic continuity across output boundaries.

Do not independently generate LOD -1 and LOD -2. Do not synthesize per chunk with random class or orientation. Do not use periodic boundaries.

## Constraint Policy: Replace Exact Fidelity With Confidence-Weighted Fidelity

The current measured-pilot cook conservatively reconstructs each 1 m DTM cell and uses a smooth mean-null projector so synthetic detail cannot change its mean. That was useful for proving packing and hierarchy, but it is incompatible with the stated visual objective and the verified Taevaskoda shoreline evidence.

The replacement policy is:

1. Compute a corrected structural base first.
2. Preserve high-confidence DTM observations within uncertainty appropriate to source/acquisition, not an arbitrary zero-error rule.
3. Allow systematic correction in low-confidence cells and where authoritative structure contradicts the raster.
4. Constrain water topology, mapped shores/escarpments, and man-made structures through their own evidence rather than through the original cell mean.
5. Downsample the final fine master to produce coherent parents and audit differences against both corrected and raw DTM.

The existing quantization round-trip gate remains. Hierarchy and seam consistency remain necessary. What changes is which surface is considered the authority: the evidence-fused corrected surface, not every raw DTM cell mean.

This requires changing the cook's synthesis/projection policy but no packed format or runtime behavior. If preserving the existing `synthesize_residual` signature is operationally useful, the cook can treat the combined correction and detail as a residual relative to the old spline; it must not subsequently project the correction away.

## Orthophoto Decision

**Fetch and retain the 20-40 cm national orthophoto, including capture date and preferably RGB plus CIR where available.** It is warranted for source repair and conditioning.

High-value uses:

- sub-meter shore/water and exposed bank boundaries;
- bare rock, sediment, erosion, and recent disturbance;
- crop rows, vehicle tracks, drainage traces, and parcel state;
- occlusion/vegetation classification when combined with nDSM/CHM;
- selecting among plausible learned morphologies.

Forbidden uses:

- treating RGB intensity or texture edge as a direct height measurement;
- inventing ground structure under canopy, water, deep shadow, or occluded cliff faces;
- ignoring acquisition-date mismatch;
- copying albedo texture into relief because it looks detailed.

An ablation must prove that orthophoto improves target geometry and rendered beauty for the classes where it is enabled.

## Training And Exemplar Program

The generator cannot exceed its target evidence. Build a versioned registry rather than one foreign exemplar bank.

Each target surface needs:

- stable identifier and immutable source checksum;
- geographic extent and CRS;
- acquisition date/method and nominal/validated resolution;
- bare-ground filtering method and human QA;
- DTM/DSM status and vegetation/object removal notes;
- landform, substrate, surficial material, soil profile, hydrology, cover, management, and disturbance labels;
- scale-dependent uncertainty and valid mask;
- license and transfer limitations;
- split assignment by site, never random overlapping patches across train/validation/holdout.

Minimum stage-one physical classes:

- peatland/bog and poor-fen microrelief;
- mineral forest on till/sand with and without pit-mound disturbance;
- exposed/shallow carbonate or other rock where justified by Estonia data;
- cultivated field with known management state/direction;
- wetland/shore and smooth-water boundary repair;
- erodible sandstone escarpment/bank, with Taevaskoda held out from tuning.

Foreign analogues are allowed but carry an explicit transfer-confidence label and cannot be the only evidence for a national class. The current Lapinjärvi TLS bank is a useful infrastructure fixture/analogue, not a universal Estonia prior.

## Cheapest Decisive Proof

Do not start with national cooking or a broad test suite. Build one research harness around the existing synthesis output boundary and run a stratified held-out patch study.

### Proof corpus

Select target-scale measured patches for six regimes: peat, mineral forest/till, exposed rock/alvar, cultivated field, shore/wetland, and erodible sandstone bank. Hold out complete sites. Include known source defects and clean regions. Keep Suur Taevaskoda as a visual/generalization target, not a training patch and not a scene-specific rule.

### Compared systems

1. conservative interpolation/no detail;
2. current rejected quilt/procedural output as a failure control, not a candidate;
3. Guérin-style nearest paired patch baseline within correctly matched regimes;
4. one learned conditional backbone candidate from the ML track;
5. the learned candidate with confidence-aware structural correction and orthophoto/ETAK conditioning.

Do not implement Schott erosion or a catalog of event modules for this proof. They would obscure whether the backbone can learn dense material morphology.

### Required observations

- side-by-side ground-level renders with identical camera, light, materials, and vegetation;
- measured target comparison where a paired target exists;
- source-repair overlays at shorelines, spikes, water, and banks;
- regime-specific height/slope/curvature distributions, variograms/spectra by physical band, drainage/connectivity where applicable, and repetition detection;
- cross-boundary strips showing geometry, normals, and conditioning transitions;
- an orthophoto ablation and a geology/soil ablation;
- review by at least one geomorphology/remote-sensing/image-artifact expert plus the user's visual judgment.

### Pass condition

The learned conditioned system must visibly add plausible, non-repeating structure in every supported class, repair rather than preserve known DTM defects, keep water/shore geometry credible, and outperform the matched dictionary baseline without copying foreign exemplar identity. Passing aggregate RMSE or zero downsample error is neither necessary nor sufficient.

## Rejected Architectures

- **Current quilt plus more masks:** rejected; an unclassified foreign exemplar cannot become Estonia by suppressing it in more places.
- **Guérin/Argudo dictionary as national backbone:** rejected; missing exemplars, repetition, hard transitions, and no causal national inference.
- **Procedural per-class function library:** rejected; unsupported combinations and transition explosion lead back to generic noise and systemic knobs.
- **Global multiscale erosion:** rejected; wrong process outside eligible erosional terrain and unresolved watershed/chunk dependencies.
- **Global pit removal:** rejected; destroys valid closed forms and does not repair source-grid shorelines.
- **Paris implicit-volumetric port:** rejected for this pass; key geometry cannot be packed into the frozen heightfield and below-10-cm detail was not demonstrated.
- **Cortial runtime/procedural subdivision:** rejected; wrong representation, wrong execution boundary, and only approximately 50 cm demonstrated.
- **Grenier/Phasor/noise residual:** rejected; it is a refined version of the visible sine/pattern failure.
- **Runtime procedural displacement:** rejected; violates the zero-browser-synthesis law.

## Open Decisions For The Main Spec

The recommendation is not open. These implementation choices need evidence from the ML and critic tracks:

1. Which learned conditional model family gives the best centimeter geometry per cook cost and training-data volume: direct residual model, conditional latent diffusion, or another terrain-specific generative model?
2. How will paired/target-scale Estonia data be acquired for each release class, and which classes remain unsupported until then?
3. Should source repair be a separate trained model/optimization or a jointly trained branch with explicit structural losses and independent acceptance?
4. Which authoritative structures are hard constraints versus uncertainty-weighted conditions, especially where ETAK, DTM, orthophoto, and acquisition dates disagree?
5. Which event module, if any, is worth implementing only after the dense stage-one proof passes?

## Bottom Line

The deterministic/process literature is valuable because it prevents the learned design from becoming an unconstrained image generator. It supplies causal conditions, event mechanics, scale hierarchy, deterministic identity, diagnostics, and strong warnings about drainage and evaluation. It does not supply a credible deterministic national generator.

Proceed with a learned dense conditional surface model plus explicit evidence-fused source repair. Keep deterministic modules narrow, calibrated, and optional. Preserve the accepted packed-height infrastructure and ordinary runtime terrain path.
