# Microtopography Known Issues

Last updated: 2026-07-13

## Active Blockers

### LUKE extraction is approved only as a foreign-analogue infrastructure diagnostic

The first real-data pass admitted tree, trunk, deadwood, or crown returns as isolated terrain several metres above ground and remains rejected/archived. The corrected lowest-connected multiview extractor was rerun and visually approved on k11/k32/k36 plus k19 holdout; the calibration bank contains 149 direct measured patches and no elevated islands. LUKE supplies no validated substrate, texture, or geology labels, and unclassified TLS extraction lacks manually labeled ground checkpoints/repeat-scan calibration. It therefore proves ingest, packing, and low-confidence morphology diagnostics only; it cannot validate Estonia soil/geology-conditioned beauty or production amplitude.

### The first measured build is diagnostic-only

The production CLI, recipe-kind enforcement, content-addressed transaction, measured cook, and independent verifier are wired. Immutable diagnostic build `5f845ae2...` passed headers, coverage, parent closure, seams, masks, dependencies, determinism, and quantization gates, but used the forest bank on every dry surface and is not a beauty result.

### Real conditioning is incomplete

The currently selected corpus supports only a low-confidence foreign-analogue mesic mineral pine-forest matrix. It does not justify peat, exposed carbonate, agriculture, sandstone-cliff, or forest-event morphology. The first measured cook applied this matrix to every allowed dry sample; decoded diagnostics exposed that as causally wrong even though packing/masks passed. That immutable preview remains diagnostic-only. A context-restricted recook must enable it only for matching forest plus known mineral/non-peat soil and emit zero elsewhere.

The fail-closed selector is implemented and independently verified. It allows `53.3685%` of published hero samples and zeroes four complete chunks; texture is exclusion-only and no geology match is claimed. Corrected immutable recipe `090747d74911...` is now cooked and live for visual review.

### Measured quilting attenuation is fixed for the pilot

The rejected first quilt used a fixed 3 m normalized crossfade and attenuated one-axis/four-way overlap RMS by `17-26%`. Equal-power blending, normalized edge-shape cost, and balanced `top_k=12` now restore 64 m RMS to `4.831 cm` versus bank `4.762 cm`, p99.9 to `21.38 cm` versus `20.89 cm`, and effective diversity to `140.3/149` patches.

The overlap/selection defect is fixed in recipe `d46700e...`: equal-power blending, normalized edge-shape cost, and balanced `top_k=12` restore 64 m RMS to `4.831 cm` versus bank `4.762 cm`, p99.9 to `21.38 cm` versus `20.89 cm`, and effective diversity to `140.3/149`. That build exposed the independent projection defect below and remains diagnostic-only.

### Cell-local hierarchy projection is rejected and replaced

The inherited mask-aware C2 bubble divided mean corrections by tiny partial-cell support, creating decoded extrema as large as `-0.747 m` and a repeated 1 m phase signature. It is replaced by an AOI-wide smooth mean-null correction plus a 1 m dead fringe, 2 m quintic taper, and fail-closed envelope enforcement. The full real closure reports maximum mean error `8.83e-17 m`, hard residual exactly zero, overlap seam zero, output amplitude/slope inside the measured bank, and phase RMS CV reduced from `0.2126` to `0.1089`.

### Hard packed-level ownership steps are fixed

The first runtime selected the finest containing plane without a transition. At the actual Taevaskoda camera window this produced maximum pointwise switches of `14.96 cm` from LOD -2 to -1 and `34.89 cm` from -1 to 0, affecting terrain and grass roots together. Format-2 now recursively blends only already-cooked packed levels with camera-centered quintic rings, includes the blend derivative in normals/slopes, mirrors the same policy in CPU probes, and fades against both moving plane edges and immutable published coverage. Generated/format-1 worlds keep their original path.

### Raster prepass allocation is compacted

The prior fixed reservation allocated about `576 MiB` for projected vertices and `140 MiB` for cluster context, directly causing browser `ArrayBuffer allocation failed` errors. The clean implementation uses exact per-cluster reservations in a `120 MiB` projected-record pool and a `13.5 MiB` context domain. All projection consumers share the same cap and invalid-base sentinel; overflow is fail-closed and surfaced through counters/warnings. The exact Estonia URL now passes repeated clean WebGPU boots.

### Format-2 runtime displacement violated packed-geometry ownership

`NaniteFetch.terrainDispAt` previously added up to `0.55 m` of camera-faded procedural vertex displacement, and grass roots repeated it. The clean implementation compiles `TerrainDisp` out when `worldManifest.format === 2`, retaining the legacy path for generated/format-1 worlds. The centered live checkpoint boots with the packed format-2 surface, vegetation, and materials present; broader motion/viewpoint inspection remains part of synthesis review rather than a boot blocker.

### A heightfield cannot represent the Taevaskoda sandstone wall

The supplied reference contains vertical and undercut faces. A single-valued heightfield can only turn these into a steep ramp, regardless of `0.0625 m` spacing. The durable spec now requires a separate cook-produced, packed structural terrain cluster path for such outcrops, with a heightfield stitch collar, collision/material/vegetation ownership, proxy transition, and no runtime synthesis. Until that path has a 3D lithology-specific corpus, the cliff face is not a valid acceptance target for the height generator.

### Hard masks and morphology selector pass decoded verification

The measured pilot cook rasterizes ETAK open water, buildings, paved roads, forest, supported soil context, and buffered slope/cliff lines directly on the fine sample grid. Unpaved trails remain ground. The final independent verifier rerasterized all 25 fine chunks, compared `48,472,349` decoded rejected samples against an independently reconstructed and quantized conservative base, and measured exactly `0 m` rejected residual. The same run bounded decoded 1 m authority-mean error at `0.0003656 m` under a `0.0010153 m` limit and rederived the LOD -1 payload byte-for-byte.

## Visual/Data Investigations

### Estonia DTM may contain two tall bumps in the Ahja river

The user observed two anomalous tall bumps in the existing Estonia 1 m height source beside the main Suur Taevaskoda cliff, apparently where river should be. Audit of pinned base manifest `708478a57c2118ea`, raw Maa-amet `54472_dtm_1m.tif`, ETAK water geometry, and nDSM confirms a coherent LiDAR ground-model/TIN bridge across missing/occluded water returns beside the steep sandstone cliff. It is not DSM, quantization, micro synthesis, or mainly canopy. The main false interior component is about `225 m2`, centered near `E 679788.35, N 6444811.45`, reaches `52.22 m`, and extends `13.45 m` into the mapped channel. The current waterbed pass neither creates nor amplifies it, but preserves the corrupt absolute ramp and subtracts only normal hydraulic depth. The cliff-top anchor itself is dry and unchanged at about `60.02 m`.

The general upstream fix belongs in shared water cooking, not micro synthesis or runtime: estimate a robust longitudinal river surface from low/central transverse samples, reject coherent steep planar water-interior outliers, use that surface for both waterY and bed carving, and taper terrain correction to zero through a `2-3 m` uncertain shore strip so legitimate banks/cliffs remain untouched. The current immutable pilot base does not contain this correction; changing its LOD0 authority requires a separately verified base release rather than silently patching negative LODs.

### Existing fixture detail is diagnostic only

The sine/checkerboard-looking local fixture proves packed fine-height retention but is not a beauty result. It must not be used as user acceptance evidence.

### Fine surface alignment previously diverged by consumer

Grass previously sampled a flatter/coarser height than terrain, trees, and plants. The clean branch now uses the finest valid packed height for grass. The centered hero preview confirms grass, terrain, trees, plants, and materials are all enabled; detailed grounding quality remains a visual synthesis-review item.

### Runtime boot blockers are fixed

The accepted checkpoint removes the `TerrainField` ceiling failure, CPU `ArrayBuffer` allocation failure, invalid TSL `If`, oversized 64 KiB uniform bindings, non-uniform `workgroupBarrier`, writable storage alias, and mixed read-only/read-write synchronization-scope failures seen during integration. `tools/boot-smoke.ts` now fails on page errors, local asset/API failures, TSL invalid-code messages, uncaptured WebGPU errors, hook failures, or missing rendered frames. Three clean exact-URL runs passed after temporary diagnostics were removed.

## Integration Risks

- The root `estonia-asset-gen` worktree contains unrelated dirty and old-agent changes. Integration must preserve user work and must not copy the obsolete 128 m global-grid rewrite or hash/noise synthesizer.
- A production pilot preview may be materialized locally, but it must not update `latest.json` until realism, masks, and user-visible acceptance pass.
- The main spec now contains raw TLS extraction gates, executable factor-conditioning requirements, pilot-only scope, event/matrix separation, and reproducible Taevaskoda visual acceptance. Implementation of the broader factor model remains deferred until the measured mineral-forest pilot is sound.
