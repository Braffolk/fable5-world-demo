# Cook Contract And Height-Hierarchy Audit

**Date:** 2026-07-13
**Status:** Read-only audit of commits `0ef75cf` and `51d1e1e`; input to the rewritten microtopography spec, not implementation authorization
**Scope:** Existing `asset-gen` height hierarchy, source repair, fine-height synthesis, packing, release closure, and the minimum contract that preserves the accepted binary/runtime infrastructure

## 1. Verdict

The accepted infrastructure can carry a substantially different, evidence-backed terrain surface without another binary format or another runtime synthesis path. The replacement generator may continue to emit ordinary absolute EH2000 height samples into the existing quant16/2D-delta/deflate height payload. LOD `-2` remains `0.0625 m` in `128 m` chunks, LOD `-1` remains `0.25 m` in `512 m` chunks, and LOD0 remains `1 m` in `2,048 m` chunks.

Source repair cannot, however, be expressed solely through the current `synthesize_residual` hook. The current cook reconstructs the pinned raw LOD0 DTM, forces every fine residual back into that raw DTM's one-meter mean-null space, publishes only LOD `-2/-1`, and then verifies that the decoded fine surface still agrees with the raw LOD0 cells. A repaired shoreline, water artifact, cliff placement, scan seam, or bad return is therefore either projected away or left to disagree with the LOD0 authority used by the runtime.

The required architecture is:

1. Build a corrected structural height hierarchy in `asset-gen`, beginning from the fallible DTM plus independent evidence.
2. Materialize that correction as ordinary replacement LOD0 height content in a new pinned format-1 base release.
3. Derive every affected LOD1-L4 ancestor from browser-equivalent decoded corrected children rather than re-reading the raw DTM independently.
4. Generate the final LOD `-2` surface against the corrected structural surface, quantize it, decode it, and derive LOD `-1` from those decoded children.
5. Where published fine coverage contributes to LOD0, derive the affected LOD0 cells from canonical decoded LOD `-1`; do not force them back to raw DTM means.
6. Publish the negative rungs as the existing format-2 overlay pinned to the new corrected base release.

This distinction is fundamental: **raw-observation fidelity is confidence-weighted and may be deliberately violated; packed-hierarchy consistency is still exact or quantization-bounded after the corrected surface has been selected.** The latter is a delivery invariant, not the visual objective and not a claim that the raw DTM was correct.

For the next one-parent Taevaskoda proof, this requires cook and verifier changes but no LAC container, index, server, demand, decode, terrain, material, vegetation, or shader change. A nationwide or multi-parent fine release is a separate issue: the runtime index/plane logic is already rectangle-capable, but the current format-2 release schema and auditor hard-code exactly one LOD `-1` parent and its 16 LOD `-2` children. Claims that national coverage requires "only different payload contents" contradict the current release code.

## 2. Audited Baseline

### 2.1 Frozen physical lattice

The canonical nonnegative grid remains:

| LOD | Texel | Footprint | Payload |
|---:|---:|---:|---:|
| `-2` | `0.0625 m` | `128 m` | `2049 x 2049` |
| `-1` | `0.25 m` | `512 m` | `2049 x 2049` |
| `0` | `1 m` | `2,048 m` | `2049 x 2049` |
| `1` | `4 m` | `8,192 m` | `2049 x 2049` |
| `2` | `16 m` | `32,768 m` | `2049 x 2049` |
| `3` | `64 m` | `131,072 m` | `2049 x 2049` |
| `4` | `256 m` | `524,288 m` | `2049 x 2049` |

Evidence:

- `asset-gen/config/base.toml:4-15` freezes `(chunk_m, chunk_res, lod_step) = (2048, 2048, 4)`, nonnegative LODs `0..4`, and the base height qscale.
- `asset-gen/src/assetgen/grid.py:8-14` defines the `4x4` hierarchy, the `2049^2` apron payload, and sample-center convention.
- `asset-gen/src/assetgen/height_geom.py:13-15` permits physical height LODs `-2..4` on integer `1/32 m` coordinate units.
- `asset-gen/src/assetgen/height_geom.py:38-59` produces exact `0.0625/0.25/1/4/... m` texels and corresponding footprints without redefining WorldGrid LOD0.
- `asset-gen/src/assetgen/height_geom.py:86-100` defines the four-by-four parent/child relationship.

No replacement synthesis method may change this lattice, call a `128 m` fine chunk "LOD0," or add a second base-grid interpretation.

### 2.2 Existing payload and quantization contract

Height remains encoding `1`: u16 quantization, 2D delta, then deflate.

- `asset-gen/src/assetgen/cook/encode.py:41-67` defines the existing encoding.
- `asset-gen/src/assetgen/cook/encode.py:70-108` provides checked fine-height quantization that rejects nonfinite values and u16 overflow instead of clipping.
- `asset-gen/src/assetgen/cook/encode.py:119-123` is the browser-equivalent float32 decode twin used by hierarchy derivation and verification.
- `asset-gen/src/assetgen/cook/chunkio.py:14-20` keeps both LAC1 and LAC2 at a 56-byte header; LAC2 changes the magic and makes physical LOD signed.
- `asset-gen/src/assetgen/cook/chunkio.py:70-97` writes LAC2 without changing the height payload encoding.

The accepted format-2 policy currently fixes:

| Rung | qscale | Maximum quantization error before float32 allowance | u16 vertical span |
|---:|---:|---:|---:|
| `-2` | `0.002 m` | `0.001 m` | `131.07 m` |
| `-1` | `0.005 m` | `0.0025 m` | `327.675 m` |
| `0..2` | `0.01 m` | `0.005 m` | `655.35 m` |
| `3` | `0.25 m` | `0.125 m` | `16,383.75 m` |
| `4` | `1 m` | `0.5 m` | `65,535 m` |

Evidence:

- `asset-gen/config/microtopography.toml:19-23` selects `0.002 m` and `0.005 m` for the fine rungs.
- `asset-gen/src/assetgen/release.py:198-209` currently enforces those two qscales in format-2 release audit.
- `asset-gen/config/base.toml:12-15` defines the nonnegative qscales.

These qscales are adequate for the next proof and require no wire change. They are not automatically optimal for national storage. Re-evaluation must happen through rendered and measured quantization trials, not by silently changing the qscale or weakening the checked encoder.

### 2.3 Current base height is raw-source-led at every rung

The legacy height cook reads the source stack independently at each requested LOD:

- `asset-gen/src/assetgen/cook/height_cook.py:46-56` reads one source window, quantizes it, and checks only encoding round-trip error against that same input array.
- `asset-gen/src/assetgen/cook/height_cook.py:68-91` invokes that operation independently for every configured LOD.
- `asset-gen/src/assetgen/process/mosaic.py:36-71` uses nearest reads at source resolution and average resampling for coarser reads.

Thus existing LOD1-L4 chunks are averages of the source mosaic before child quantization, not artifacts derived from browser-decoded child payloads. That is acceptable for the legacy raw base, but it is not the required contract once selected LOD0 cells are evidence-corrected and must remain coherent through runtime fallback.

### 2.4 Current fine cook is raw-LOD0-conservative

The accepted pilot's morphology entry point does not own the full surface:

- `asset-gen/src/assetgen/cook/micro_synth_cook.py:98-122` asks `synthesize_residual` for a residual only at LOD `-2`.
- `asset-gen/src/assetgen/cook/micro_synth_cook.py:79-95` fits a cubic spline to the pinned base release's LOD0 cells.
- `asset-gen/src/assetgen/cook/micro_synth_cook.py:147-160` conservatively reconstructs those raw authority cells and then projects the generated residual through the mean-null projector.
- `asset-gen/src/assetgen/cook/micro_synth_cook.py:171-172` forms `conservative + residual`; the generator never returns a corrected base or a complete absolute surface contract.
- `asset-gen/src/assetgen/process/micro_fixture.py:21-41` forces the fine reconstruction to exact discrete coarse-cell means.
- `asset-gen/src/assetgen/process/microtopo/projection.py:75-118` enforces zero one-meter residual means after masking.

The pinned reader itself accepts only a format-1 LOD0 authority:

- `asset-gen/src/assetgen/cook/pinned_height.py:68-99` audits and indexes a pinned base release.
- `asset-gen/src/assetgen/cook/pinned_height.py:105-132` decodes LOD0 only.

This makes the present synthesis boundary unsuitable for source repair. Treating a correction as a "larger residual" does not work while the subsequent projector removes its one-meter mean and the overlay leaves LOD0 untouched.

### 2.5 Existing decoded-child parent derivation is reusable

The hierarchy primitive is correctly separated from morphology:

- `asset-gen/src/assetgen/cook/micro_hierarchy.py:1-6` explicitly defines a synthesis-free packing/verifier boundary.
- `asset-gen/src/assetgen/cook/micro_hierarchy.py:53-137` assembles 16 decoded child cores plus the east/south/southeast support needed for the parent's full apron.
- `asset-gen/src/assetgen/cook/micro_hierarchy.py:140-169` performs fixed-order factor-four area means without materializing a four-dimensional reduction.
- `asset-gen/src/assetgen/cook/micro_synth_cook.py:487-504` derives the LOD `-1` payload from decoded LOD `-2` children.
- `asset-gen/src/assetgen/micro_verify.py:532-550` independently rederives and requires the parent payload to be byte-exact.

This primitive should be generalized from the one `HeroCoverage` case to arbitrary physical height parents, not replaced. The important rule is **quantize child, browser-equivalent decode child, fixed-order reduce, quantize parent**.

### 2.6 Accepted runtime behavior is representation-based

The runtime already interprets negative LODs as packed physical height, independent of the generating algorithm:

- `src/world/source/RemoteWorldSource.ts:151-162` validates physical height metadata (`baseTexelMeters`, `finestLod`, `authorityLod`) rather than a synthesis model.
- `src/nanite/world/PlaneFill.ts:151-169` maps format-2 height to physical-level geometry.
- `src/nanite/world/PlaneFill.ts:392-416` plans each indexed height LOD from its actual chunk rectangle.
- `src/nanite/world/PlaneFill.ts:546-548` enables the accepted packed-height morph whenever format 2 contains a negative height LOD; it does not branch on diffusion, dictionary, process, or other provenance.
- `src/nanite/world/TerrainField.ts:613-674` samples LOD `-2`, LOD `-1`, and the first nonnegative authority on CPU.
- `src/nanite/world/TerrainField.ts:677-750` applies the corresponding packed-level GPU selection/morph.
- `src/nanite/world/TerrainMorph.ts:9-12` fixes the camera and availability bands; `src/nanite/world/TerrainMorph.ts:25-48` fades against actual plane and published-coverage edges.

The replacement synthesis must therefore change height content, not add `learned*`, `corrected*`, `diffusion*`, or `synthetic*` runtime methods. The existing `cookedMicroHeight` name is historical representation handling already accepted in `0ef75cf`; it is not authorization for another provenance branch.

## 3. Exact Authority Conflict

### 3.1 Why raw LOD0 cannot remain authoritative in a repaired region

Suppose the cook moves a one-meter-grid shoreline to an ETAK/orthophoto-supported continuous boundary or removes a false river bump. If LOD `-2/-1` contain the repair but LOD0 remains the old DTM, the runtime's LOD `-1 -> 0` morph gradually restores the known error. Grass, probes, normals, and materials will follow the morphing packed surface correctly, but the surface itself is wrong during and outside the fine band.

The present verifier deliberately enforces the old behavior:

- `asset-gen/src/assetgen/micro_verify.py:702-750` downsamples decoded fine output and rejects disagreement with the pinned LOD0 authority.
- `asset-gen/src/assetgen/micro_verify.py:715-755` reconstructs the old conservative base and requires rejected-mask samples to equal it.
- `asset-gen/src/assetgen/micro_verify.py:767-779` records raw LOD0 authority agreement as a hard gate.

These gates prove retention and masking for the rejected pilot. They are invalid as production beauty/source-repair gates.

### 3.2 Correct interpretation of "authority"

The raw Maa-amet DTM remains the preferred elevation **observation**. It is not the final packed authority where independent evidence establishes a defect.

The release authority must be:

`accepted evidence-fused surface -> canonical quantized/decode hierarchy -> packed height artifacts`

The runtime authority at LOD0 must therefore be the corrected ordinary LOD0 artifact from that hierarchy. The raw DTM remains in recipe provenance, confidence maps, correction deltas, and audit reports. It must not be smuggled back into the rendered hierarchy as a zero-error constraint.

### 3.3 Source repair is upstream of fine stochastic detail

Source repair and underdetermined microdetail have different semantics:

| Component | Purpose | May disagree with raw DTM? | Must be stochastic? |
|---|---|---:|---:|
| Structural correction | Repair supported water, shore, object leakage, interpolation, spikes, scan seams, and mapped structures | Yes, with evidence/confidence | No |
| Resolved structural refinement | Express corrected banks, channels, slopes, and other structure between 1 m and 0.25 m | Yes | No requirement |
| Underdetermined detail | Sample plausible sub-observation morphology conditioned by material/process/context | Yes, within accepted structural/physical constraints | Often, but production seed is fixed |
| Packing hierarchy | Produce coherent runtime rungs | It follows the accepted surface, not the raw DTM | No |

No downstream mean-null operation may erase the structural correction. If a band decomposition uses mean-null residuals, their parent is the **corrected canonical parent**, not the raw observation.

## 4. Recommended Cook Contract

### 4.1 Inputs and immutable recipe identity

Every corrected/fine build must bind:

- the raw DTM tile identities and acquisition metadata;
- every independent correction/conditioning source and availability/confidence mask;
- model/checkpoint or deterministic algorithm identity;
- full preprocessing and inference environment identity;
- world-coordinate seed policy;
- output coverage, algorithm support halo, hierarchy support closure, and publication closure;
- reducer, qscales, qoffset policy, codec, and code revision;
- the corrected base-manifest hash to which the negative overlay is pinned.

The existing recipe hashing and immutable staged release machinery are reusable. The current recipe omits the replacement model and correction products and hard-pins one historical base manifest; those identities must be replaced, not bypassed.

### 4.2 Stage C0: plan three distinct extents

The cook must distinguish extents that the current pilot partially conflates:

| Extent | Meaning | Published? |
|---|---|---:|
| Model support | Symmetric context required by repair/synthesis, including receptive field, watershed/process support, and overlap consensus | No |
| Hierarchy support | East/south/southeast samples required to derive complete parent payloads after decode | No |
| Publication coverage | Complete rectangular negative-LOD chunks addressable by runtime demand | Yes |

The current nine transient LOD `-2` chunks are only hierarchy apron support (`asset-gen/src/assetgen/height_geom.py:103-119`). They are not an adequate ML receptive-field halo or a general process-domain halo. A replacement model must request its own symmetric physical context before cropping into the hierarchy closure.

### 4.3 Stage C1: produce corrected structural LOD `-1` candidates

Run the evidence-backed repair/structural method over complete affected LOD0 chunks plus algorithm halo and hierarchy support. Its canonical structural output is a `0.25 m` absolute EH2000 surface, not merely a mask and not a raw-DTM-preserving residual.

For each corrected LOD0 chunk, produce transient LOD `-1` candidates for all 16 child cores and the east/south/southeast support needed to derive the LOD0 apron. These candidates need not all be published. They exist so the corrected LOD0 artifact has one coherent submeter source across its whole payload.

The method must provide:

- absolute corrected height;
- raw-observation confidence and correction uncertainty;
- correction class/evidence provenance;
- hard invalid regions and support masks;
- a declared behavior for water surface, submerged bed, shore, cliffs, objects, and no-data.

If only a smaller research area has defensible correction evidence, the correction must converge continuously to the unchanged accepted base before the corrected-base publication boundary. Do not taper away a real shoreline or escarpment merely because a chunk ends; expand the correction/base-recook closure instead.

### 4.4 Stage C2: produce one canonical LOD `-2` master surface

Generate the final `0.0625 m` absolute surface in global coordinates over the fine publication coverage plus model support and hierarchy support. Chunk identity may not affect phase, class, orientation, feature ownership, or random state.

The generator may internally emit corrected base plus residual bands, but its cook adapter must return an absolute master height grid and validity/provenance evidence. This replaces the residual-only contract at `micro_synth_cook.py:98-122`.

Where LOD `-2` refines a structural LOD `-1` candidate, one of two internal implementations is allowed, with identical output contract:

1. The fine model is trained/constructed as a band whose area-reduced parent is the corrected LOD `-1` candidate.
2. The fine model emits an unconstrained absolute surface and the canonical LOD `-1` parent is replaced by the decoded-child reduction.

The second is the normative packing authority. No raw-DTM projection follows it.

### 4.5 Stage C3: canonicalize LOD `-2`

For every published and transient LOD `-2` artifact:

1. Crop from the reconciled global master, never independently generate a final chunk.
2. Quantize through `encode_quant16_checked` at the accepted fine qscale.
3. Decode through the browser-equivalent float32 decoder.
4. Verify finite values, u16 range, per-artifact round-trip bound, and decoded shared-apron agreement.
5. Bind artifact hashes into dependency evidence.

The qoffset may be shared per bounded synthesis/quantization domain as in the current pilot. National production must define how adjacent qoffset domains retain decoded seam tolerance. The current verifier requires one shared qoffset for the 25-chunk closure at `asset-gen/src/assetgen/micro_verify.py:456-500`; that is a pilot policy, not a viable nationwide global qoffset because `0.002 m * 65535` spans only about `131 m` vertically.

### 4.6 Stage C4: derive canonical LOD `-1`

For every published LOD `-1` parent:

1. Assemble all 16 decoded LOD `-2` child cores and factor-wide east/south/southeast support.
2. Apply the fixed-order `4x4` area mean.
3. Quantize the result at the accepted LOD `-1` qscale.
4. Decode it and make that decoded artifact the only parent input to the next rung.

Do not independently synthesize LOD `-1`. Do not compare it to the raw DTM as an acceptance objective. The reusable implementation is `micro_hierarchy.py:53-169` and the reusable independent gate is the byte-exact parent derivation at `micro_verify.py:536-550`.

For corrected LOD0 chunks that contain LOD `-1` children outside published fine coverage, quantize/decode the Stage C1 structural candidates as transient canonical children. Replace any Stage C1 child covered by LOD `-2` with the Stage C4 decoded-child-derived parent.

### 4.7 Stage C5: derive corrected LOD0

For every corrected LOD0 chunk:

1. Assemble its 16 canonical decoded LOD `-1` child cores plus the support required for the LOD0 apron.
2. Reduce by the same fixed-order factor-four area mean.
3. Quantize at the ordinary LOD0 qscale and decode.
4. Publish it as an ordinary LAC1 height replacement in a new format-1 base release.

This is the key repair step. It ensures that the runtime's LOD `-1 -> 0` fallback reaches the corrected structural surface rather than the old observation.

For a cheap experiment that does not yet generate structural LOD `-1` across an entire LOD0 chunk, a mixed LOD0 artifact is acceptable only if all these conditions hold:

- fine-covered LOD0 cells come from canonical decoded LOD `-1` reductions;
- other cells come from the evidence-corrected LOD0 structural solution, not blindly from raw DTM where repair is indicated;
- the splice lies outside the runtime availability fade plus sampling support;
- value, gradient, hydrology, and rendered motion gates show no transition;
- the recipe records the mixed derivation cell mask.

The recommended production contract is still complete canonical LOD `-1` support for every corrected LOD0 replacement.

### 4.8 Stage C6: derive every affected LOD1-L4 ancestor

Once a LOD0 core changes, every ancestor whose reduction footprint contains it must be replaced through LOD4. Each replacement is derived from decoded immediate children, then quantized and decoded before becoming the next source.

Aprons expand the dependency closure. If a child core changes, its west, north, and northwest neighbors may also need their east, south, or southeast apron repacked even when their cores remain unchanged. The planner must compute this decoded-artifact dependency closure rather than assuming "one changed child means one changed file per rung."

The current legacy height cook is not suitable for these replacement ancestors because it independently re-reads the source (`height_cook.py:46-56`). Generalize the decoded-artifact hierarchy primitive instead.

### 4.9 Stage C7: publish corrected base, then negative overlay

Publish in two immutable transactions:

1. **Corrected base release:** format 1, inheriting unchanged existing content and replacing the corrected LOD0/ancestor/apron dependency closure with ordinary LAC1 height chunks.
2. **Fine overlay release:** format 2, pinned to the corrected base-manifest SHA-256, adding the published LOD `-2/-1` LAC2 chunks.

`asset-gen/src/assetgen/release.py:913-917` already combines inherited base content with staged entries by key, so the content-addressed release model is compatible with replacement nonnegative chunks. The hard-coded historical base identity at `release.py:53`, `release.py:468-470`, and `release.py:595-600` must become recipe-bound corrected-base identity rather than an eternal constant.

The current format-2 overlay cannot itself stage replacement LOD0 chunks because `release.py:613-619` requires the staged key set to equal exactly the frozen negative-LOD set. Keeping corrected nonnegative content in the pinned base release is therefore the narrowest path with no manifest/runtime change for the next pilot.

## 5. Coverage And Transition Contract

### 5.1 Mandatory parent closure

For every published fine region:

- LOD `-2` coverage is a complete `4x4` child rectangle for each published LOD `-1` parent.
- Every published LOD `-1` parent has a corrected LOD0 authority chunk.
- Every corrected LOD0 authority has real corrected/derived ancestors through the coarsest released floor.
- No fine chunk is indexed without this parent chain.
- Transient model and hierarchy support is recipe-bound and verified even when not indexed.

The current one-parent release auditor enforces part of this at `asset-gen/src/assetgen/release.py:290-315`.

### 5.2 Coverage-edge behavior

The runtime currently fades LOD `-2` over a `2 m` availability margin and LOD `-1` over an `8 m` availability margin (`src/nanite/world/TerrainMorph.ts:9-12`). The cook must not treat those fades as permission to publish unrelated parent geometry.

Required edge behavior:

- Fine and parent content remain valid through the fade band plus all sampling/normal stencils.
- The LOD0 surface beneath the edge is the corrected structural authority.
- Correction or synthesis confidence transitions are physical/evidence-based and independent of storage chunk boundaries.
- Ground-level motion through camera and availability morphs is reviewed; a numerically smooth fade that visibly inflates/deflates terrain still fails.
- Grass, trees, plants, materials, probes, and terrain continue to use the same accepted packed-surface policy; no consumer-specific correction is allowed.

### 5.3 Current multi-parent limitation

The runtime can plan a rectangular chunk set at each indexed LOD (`PlaneFill.ts:392-416`) and derives coverage from indexes. The format-2 release tool cannot authorize more than one LOD `-1` parent:

- `asset-gen/src/assetgen/height_geom.py:103-119` plans one hero parent.
- `asset-gen/src/assetgen/release.py:296-306` requires one parent and exactly its 16 children, and rejects any other indexed negative key.
- `asset-gen/src/assetgen/release.py:613-619` requires the staged overlay to equal that one frozen set.
- `asset-gen/src/assetgen/micro_verify.py:401-415` reconstructs the same one-parent plan.

Therefore:

- The next Taevaskoda one-parent proof can meet a strict no-format/no-runtime-change contract.
- A wider or national release must generalize cook planning, format-2 coverage metadata/audit, and independent verification to a rectangle or set of parent-closed rectangles.
- LAC1/LAC2 chunk bytes, the v2 index record, server fetch, decode workers, physical geometry, and runtime height sampling do not need to change for that generalization.
- Calling the metadata/auditor generalization "only synthesis" would be inaccurate even though it is not a binary or shader redesign.

## 6. Water, Shore, And Dependent Cook Ordering

### 6.1 Height means rendered bed under water

The shipped height semantic includes a submerged bed under the water mask (`asset-gen/src/assetgen/manifest.py:43-46`). The corrected master and hierarchy must decide water surface, bank/shore, and bed together:

- `waterY` is the visible water surface.
- `watercover` carries anti-aliased occupancy/shore coverage.
- height carries dry terrain and the submerged bed.
- under-water microdetail is suppressed unless a separately validated bed model requires it; water-surface noise never enters height.

The current `cook_waterbed` is a LOD0-only post-pass and intentionally leaves coarse height as pure DTM (`asset-gen/src/assetgen/cook/layers_cook.py:271-283`). That ordering is incompatible with a fully derived corrected hierarchy. For corrected releases, bed composition must occur before Stage C5/C6 reduction, or every affected height ancestor must be rederived afterward.

If structural repair changes water levels or shore geometry, recook the compatible `water`, `watercover`, and height-bed content for the same evidence snapshot. Do not repair a terrain shoreline while leaving water occupancy on the old edge.

### 6.2 Other cook dependencies

Trees and mapped boulders carry no baked Y and are grounded by the runtime surface (`asset-gen/src/assetgen/manifest.py:79-86` and `asset-gen/src/assetgen/manifest.py:98-104`), so a height-only correction does not require regenerating their records solely to change elevation.

Understory and debris density do depend on slope from cooked LOD0:

- `asset-gen/src/assetgen/cook/layers_cook.py:419-428` decodes LOD0 to produce the 2 m slope field.
- `asset-gen/src/assetgen/cook/layers_cook.py:431-468` uses that slope for understory suitability.
- `asset-gen/src/assetgen/cook/layers_cook.py:473-497` uses it for debris suitability.

If a correction materially changes those slopes, recook the affected understory/debris content or explicitly prove the old values remain within their stable classification/suitability behavior. This is an asset dependency, not a runtime terrain exception.

## 7. Quantization, Seams, And Determinism

### 7.1 Quantization invariants

- No nonfinite sample reaches an encoder.
- No u16 saturation/clipping is allowed in fine or corrected hierarchy content.
- Every artifact passes its half-qscale plus declared float32 allowance.
- Parent construction consumes browser-equivalent decoded float32 children.
- A child is never reduced from its pre-quantized float master after a different child payload has been accepted.
- qoffset selection is deterministic, recipe-bound, and compatible across seam domains.

### 7.2 Seam invariants

- A shared world sample is generated once in the reconciled master domain.
- Final storage chunks are crops, never independent generation requests.
- Decoded adjacent apron/core values are equal where a shared qoffset domain makes equality possible; otherwise their difference is bounded by one declared fine quantum and must be visually/normal-safe.
- Value-only seam checks are insufficient. Gradient, normal, curvature, drainage, feature-continuation, and condition-transition diagnostics are required during synthesis research.
- Parent aprons are derived from actual neighbor child samples, never duplicated from the parent's last core sample.

The current pilot demonstrates exact decoded seams under one shared qoffset (`micro_verify.py:552-567`). National quantization-domain behavior remains unresolved and must be proven before rollout.

### 7.3 Determinism invariants

- Fixed recipe inputs produce immutable output hashes on the pinned accepted inference stack.
- Window order, process count, and crop request do not change a world sample.
- Released packed artifacts, not a promise of cross-driver floating-point identity, are runtime authority.
- A recook that intentionally changes a correction/model/input produces a new corrected base manifest and a new overlay recipe; it never mutates an inherited content object.

## 8. Honest Storage Math

### 8.1 Per-chunk and one-parent cost

One uncompressed `2049^2` u16 payload is `8,396,802` bytes, plus a 56-byte header. One parent-closed published fine tile contains 16 LOD `-2` chunks and one LOD `-1` chunk:

`17 * (8,396,802 + 56) = 142,746,586 bytes = 136.13 MiB raw`

The locally materialized accepted Taevaskoda cook at recipe `090747d...` measured:

- 16 published LOD `-2` files: `44,012,099` bytes;
- one published LOD `-1` file: `3,169,314` bytes;
- published total: `47,181,413` bytes, about `45.0 MiB` for `0.262144 km^2`;
- nine transient LOD `-2` hierarchy-support files: `23,708,353` additional cook bytes, not published.

That compression ratio belongs to the rejected LUKE-derived field and is not a reliable forecast for a learned high-entropy output. The raw number is the uncompressed-payload baseline and a near-worst-case planning bound; incompressible deflate can add slight framing/block overhead. Production must measure p50/p95/p99 compressed bytes by physical regime on the candidate generator.

### 8.2 Pilot and national implications

A `16 km x 16 km` request rounds to `32 x 32` LOD `-1` parents, covering `16.384 km x 16.384 km`:

- `1,024` LOD `-1` chunks;
- `16,384` LOD `-2` chunks;
- about `48.3 GB` at the rejected-pilot measured ratio;
- about `146.2 GB` raw in the accepted encoding.

The existing full-country AOI resolves to `380,928 m x 266,240 m` and `24,180` rectangular LOD0 chunks (`assetgen plan --aoi estonia`, from the cached official 1:10k sheet grid). A fully rectangular fine hierarchy over the same AOI would contain:

- `744 x 520 = 386,880` LOD `-1` chunks;
- `6,190,080` LOD `-2` chunks;
- `6,576,960` published negative-LOD chunks total;
- about `18.25 TB` at the rejected-pilot measured ratio;
- about `55.23 TB` raw.

Land/coverage masking can lower storage, but the current release requires rectangular parent closure and the legacy full-country height cook writes the chunk-snapped AOI. A national spec must not quote a tens-of-gigabytes number for all Estonia. Tens of gigabytes describes a roughly 16 km pilot; the national accepted-format cost is multi-terabyte even under favorable compression.

Corrected LOD0/coarse replacement chunks add comparatively little, but their content-addressed ancestors and dependency closure must still be budgeted.

## 9. Proposal Claims Versus Code

### 9.1 ML proposal

`ML-SYNTHESIS-PROPOSAL.md:19` and `:39-40` say the model can return ordinary float height arrays while existing quantization, manifests, and runtime remain unchanged. That is directionally correct for the final representation but false for the present cook API:

- the current hook returns a residual, not a complete surface (`micro_synth_cook.py:98-122`);
- the current cook projects it back to raw LOD0 means (`micro_synth_cook.py:147-160`);
- the current overlay publishes only negative keys (`release.py:613-619`);
- the current verifier rejects corrected LOD0 divergence (`micro_verify.py:702-750`).

`ML-SYNTHESIS-PROPOSAL.md:392` says the only integration result should be different fine payload contents. That cannot ship its own proposed source repair. The minimum correct integration also produces a new corrected base release and rederived ancestors, although the binary and runtime interfaces remain unchanged.

The two-stage `1 m -> 0.25 m -> 0.0625 m` decomposition maps well to the physical rungs, but rung alignment does not prove that two independent model stages are the best scientific architecture. This audit only establishes how any accepted model must deliver its hierarchy.

### 9.2 Deterministic/process proposal

`deterministic-process-proposal.md:18-20` correctly states that the current hard one-meter mean-null rule must not erase justified repair. `:104-126` correctly recommends one final master and confidence-weighted fidelity.

Its phrase "derive coarser fine rungs" at `:104-107` is incomplete for the actual runtime. Once the accepted master changes one-meter means, corrected LOD0 and every affected nonnegative ancestor must also be replaced. Merely deriving LOD `-1` and auditing a delta against raw LOD0 leaves the runtime fallback on the wrong surface.

Its statement at `:126` that combined correction/detail could remain a residual relative to the old spline is only safe if the raw mean-null projector and raw-authority verifier are removed and the corrected base hierarchy is published separately. Preserving the Python function name or arithmetic form does not preserve the current semantic boundary.

### 9.3 ML critique

`CRITIQUE-ML-PROPOSAL.md:251-257` correctly requires a chosen corrected authority and a runtime transition proof. `:328-329` explicitly requires deterministic typed reconstruction before generative synthesis and a corrected ordinary LOD0 or another proven corrected-parent hierarchy. This audit resolves the repository side of that requirement: publish a corrected format-1 base and derive its affected ancestors from decoded corrected children.

## 10. Allowed And Forbidden Changes

### 10.1 Allowed cook/content changes

- Replace the residual-only morphology generator with a complete corrected-surface adapter inside `asset-gen`.
- Add deterministic or learned source-repair preprocessing and evidence records.
- Generalize decoded-child hierarchy reduction to LOD `-1 -> 0 -> 1 -> 2 -> 3 -> 4`.
- Produce a corrected format-1 base release that inherits unchanged objects and replaces only the dependency closure.
- Replace the static approved-base hash with the recipe-bound corrected-base manifest hash.
- Replace raw-DTM fidelity gates with evidence-fused correction and canonical-hierarchy gates.
- Recook compatible water/watercover/bed and slope-dependent asset content when their inputs change.
- Change model, checkpoint, seed, correction evidence, and output payload values under a new immutable recipe.

### 10.2 Forbidden interface/runtime changes for the next proof

- No LAC3, new height encoding, detail-parameter layer, residual texture, or runtime decompression scheme.
- No redefinition of LOD0 or global chunk geometry.
- No runtime procedural synthesis, reconstruction, diffusion, patch placement, noise, or correction.
- No generator/provenance branch in `TerrainField`, shaders, grass, vegetation, probes, materials, or collision.
- No direct special-case Taevaskoda coordinates in the synthesis method.
- No shader edits for the replacement cook proof.
- No fine publication without corrected LOD0 and ancestor closure.
- No post-hoc projection to known raw-DTM errors for the sake of an internal consistency score.

### 10.3 Later metadata/tooling change required for wide coverage

Before more than one LOD `-1` parent can be released, generalize:

- `HeroCoverage` to a parent-closed rectangular/set coverage plan;
- expectation and staged-key validation;
- format-2 `coverage.height` metadata and release audit;
- transient-support dependency evidence;
- independent verifier iteration over all parents.

This does not require changing LAC payload bytes, v2 index records, the asset server, or runtime sampling, but it is still an explicit manifest/release-tool contract change and must be specified before national cooking.

## 11. Mandatory Invariants For The Rewritten Spec

| Invariant | Hard requirement |
|---|---|
| Final representation | Absolute EH2000 packed height; no runtime parameters or synthesis |
| Observation policy | Raw DTM is confidence-weighted evidence, not an exact target in diagnosed defect regions |
| Corrected authority | LOD0 in fine/repaired coverage is a corrected ordinary base artifact |
| Parent derivation | Immediate parent derives from browser-decoded canonical children in fixed order |
| Ancestor closure | Every affected ancestor and apron dependency through LOD4 is rederived/repacked |
| Fine coverage | Complete 16-child closure for each published LOD `-1` parent |
| Model tiling | Global/support-window generation; final storage chunks are crops only |
| Seams | Decoded value plus gradient/normal/feature continuity; chunk identity has no semantic effect |
| Quantization | Checked, non-clipping, recipe-bound qscale/qoffset; error recorded per artifact |
| Water | Water surface, cover, shoreline, and submerged bed are mutually compatible before final hierarchy derivation |
| Surface agreement | Terrain, grass, vegetation, materials, probes, and normals sample the accepted packed surface/morph |
| Verification | Raw correction delta is reported, not failed merely for being nonzero; canonical hierarchy and evidence are independently recomputed |
| Release | Corrected base and fine overlay are immutable, content-addressed, and hash-bound |
| Runtime | Existing accepted format-2 packed-height path only; no new shader or provenance path |

## 12. Verification Changes Required

Retain:

- CRC/header/index/content-hash validation;
- finite/range/checked-quantization gates;
- decoded child-to-parent independent derivation;
- apron seam checks;
- deterministic recipe and transient-dependency binding;
- exact real-WebGPU boot and visual surface-agreement review after a new release is materialized.

Replace:

- `asset-gen/src/assetgen/verify.py:58-69`, which fails any LOD0 source deviation beyond quantization error;
- `asset-gen/src/assetgen/micro_verify.py:702-750`, which treats raw LOD0 mean agreement as the hierarchy objective;
- the rejected generator-specific hard-mask identity at `micro_verify.py:570-585` and `:646-755`.

Add focused stable gates:

- correction evidence/confidence provenance and raw-versus-corrected delta reports by repair class;
- protected-feature false-correction checks for cliffs, boulders, banks, channels, and mapped structures;
- canonical `-2 -> -1 -> 0 -> ...` dependency rederivation;
- water/shore/bed compatibility;
- coverage-edge motion and grounding review through the actual runtime morph;
- storage-size distributions by physical regime and qscale candidate;
- ground-level blind visual comparison at Taevaskoda and held-out same-class sites.

Do not build a broad test suite around the provisional model internals. The stable contracts are representation, hierarchy, evidence binding, seams, release identity, and user-visible runtime behavior.

## 13. Unresolved Decisions

These remain for the research synthesis and rewritten spec; they do not weaken the recommended cook contract:

1. **Source-repair solver:** deterministic typed reconstruction, learned correction with abstention, or a hybrid. The output must still be the corrected structural hierarchy described above.
2. **Fine generator:** conditional diffusion, GAN, direct generative residual model, or another terrain-native method after the required bake-off. The output must still be one canonical absolute fine master.
3. **National coverage policy:** full sheet-grid rectangle, land/shore parent rectangles, or a coverage set with explicit holes. This controls multi-terabyte storage and the required generalized coverage metadata.
4. **National qoffset domains:** deterministic grouping and seam policy that avoids a global `131 m` vertical-span limit while keeping decoded boundaries within the declared fine tolerance.
5. **Corrected-base extent for research pilots:** complete affected LOD0 chunks are recommended; any smaller mixed derivation must satisfy Stage C5's explicit guard/mask/transition gates.
6. **Structural non-heightfield terrain:** vertical/undercut outcrops remain outside this height-hierarchy contract and must not be claimed as solved by a corrected 6.25 cm heightfield.

## 14. Recommended Next Proof Contract

The cheapest proof that exercises source repair without reopening runtime or format work is one existing Taevaskoda LOD `-1` publication parent:

1. Run the selected structural correction/Stage A over its entire containing LOD0 authority chunk plus model and hierarchy support.
2. Produce transient canonical `0.25 m` children sufficient to derive a corrected full LOD0 payload.
3. Run the candidate fine generator only for the published LOD `-1` parent plus its model/hierarchy support.
4. Quantize/decode LOD `-2`, derive the published LOD `-1`, derive corrected LOD0, and rederive the affected LOD1-L4/apron closure.
5. Publish a new corrected format-1 base and an otherwise unchanged one-parent format-2 overlay.
6. Independently verify correction evidence, canonical hierarchy, seams, water/shore/bed agreement, and immutable hashes.
7. Boot the exact Estonia/Taevaskoda URL through cloud bake and settled frames, then inspect shoreline smoothness, source-error removal, terrain continuity, materials, trees/plants, and grass grounding.

If that proof retains raw grid shorelines, restores them during the LOD `-1 -> 0` morph, changes water and terrain on different edges, or adds only texture-like bumpiness, the synthesis/repair method fails. The binary/runtime infrastructure is not the fallback explanation.
