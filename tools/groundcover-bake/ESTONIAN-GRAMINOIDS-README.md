# Estonian graminoid profile authoring track

## Inspectable outcome

This track authors five independent deterministic GCRP/v2 mesh fixtures for exact native profile IDs `0..4` and bakes them through the existing periodic top-XZ Apple-Metal raster path. The deliverable is geometry and its first-hit depth/normal carrier, not a recolored copy of the analytic grass oracle. `EstonianGraminoids.ts`, its focused tests, and `run-estonian-graminoids.ts` are intentionally separate from the shared baker and runtime.

The production lattice is fixed at 16 azimuths by four elevations (`15, 35, 55, 75` degrees), azimuth-major/elevation-minor. Each slice has a `64x64` canonical interior and one exact wrapped gutter. The runner performs two localhost Chromium/WebGPU bakes per profile and rejects a changed packed hash, a browser/WebGPU diagnostic, a non-Apple adapter, periodic-address drift, or a failed selected-pixel CPU/raster comparison.

## Public source ledger and implementation bindings

The generated meshes ingest no external geometry, texture, photograph, or specimen scan. Public sources constrain the procedural architectural recipe only.

### Profile 0 — *Agrostis capillaris*

- Royal Botanic Gardens, Kew, Plants of the World Online / GrassBase: <https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:30197865-2/general-information>. Binding: caespitose perennial with elongated rhizomes and possible stolons; erect or ascending 10–70 cm culms; 1–5 mm ribbed blades; open oblong-to-ovate panicle; whorled lower primary branches; capillary branches; solitary 2–3.5 mm spikelets.
- Kew's range statement is Europe to northern China and Afghanistan. The local profile identity itself is the repository's approved zero-based native palette; the geometry does not assert abundance at a particular Estonian site.
- Geometry: eight small tufts are joined by explicit near-surface indexed rhizome tubes whose endpoint cap fans weld directly to the existing tuft hubs; every tuft has flat tapered leaves, a culm, and an open whorled panicle of independent capillary axes and spikelet bodies.

### Profile 1 — *Avenella flexuosa*

- Kew POWO / GrassBase under the synonym *Deschampsia flexuosa*: <https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:391947-1/general-information>. Binding: caespitose perennial; mostly basal, stiff, filiform, involute 0.3–0.8 mm leaves; wiry culms; open ovate-effuse panicle; capillary branches; 4–6 mm two-floret spikelets.
- eElurikkus synonym and Estonia record surface: <https://elurikkus.ee/app/taxonomy/taxon/4287>. This records *Avenella flexuosa* under the accepted local name *Deschampsia flexuosa* and exposes verified Estonian occurrences.
- Geometry: six separate dense tufts (no invented cross-tile rhizome network), strongly flexuous segmented basal ribbons, wiry culms, very open capillary panicles, and visibly larger spikelet bodies than *Agrostis*. The 64-texel carrier cannot preserve a literal 0.3 mm leaf over the chosen tile, so filiform blades are minimally widened and this is recorded as a carrier-resolution concession rather than a specimen-scale claim.

### Profile 2 — *Calamagrostis canescens*

- Kew POWO / GrassBase: <https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:77145450-1/general-information>. Binding: loosely clumped caespitose perennial with short rhizomes; erect 50–120 cm culms; flat or convolute, 3–6 mm pubescent blades; open lanceolate or oblong nodding 5–23 cm panicle; solitary pedicelled spikelets.
- Estonian occurrence context is independently present in Estonian University of Life Sciences vegetation work: <https://dspace.emu.ee/bitstream/handle/10492/1864/Indrek_Melts_DO2014.pdf?isAllowed=y&sequence=1>. It is not used to infer density, morphology, or a scene-specific placement rule.
- Geometry: six loose shoots joined by explicit short indexed rhizomes with nondegenerate welded endpoint fans; broader leaves than profiles 0–1; taller seven-sided culms; long, gently nodding lanceolate panicles with many short branches and spikelets. Pubescence and callus hairs remain below carrier resolution and are not claimed.

### Profile 3 — *Carex cespitosa*

- Kew POWO: <https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:298997-1/general-information>. Binding: accepted temperate-Eurasian perennial/rhizomatous geophyte in Cyperaceae.
- James, Jiménez-Mejías & Porter, “The occurrence in Britain of *Carex cespitosa*, a Eurasian sedge rare in western Europe”: <https://bsbi.org/wp-content/uploads/dlm_uploads/2022/04/YNJB_A_11754961_O.pdf>. Binding: dense tussocks without creeping rhizomes, height to 80 cm, bright yellowish-green foliage, round-backed purple-brown lower sheaths, one male spike, short stout cylindrical female spikes to 30 mm, and a lower bract shorter than the inflorescence.
- Direct eElurikkus Estonian occurrence: <https://elurikkus.ee/app/occurrences/occurrence/44434069>.
- Geometry: four separate dense tussocks; a basal sheath body; many arched blades; explicitly triangular culms; exactly one terminal male spike and two lateral female spikes per culm. Each spike is an indexed axis carrying repeated scale/perigynium-like bodies, not a smooth capsule.

### Profile 4 — *Eriophorum vaginatum*

- Kew POWO: <https://powo.science.kew.org/taxon/307418-1>. Binding: accepted perennial Cyperaceae species native across the subarctic and temperate Northern Hemisphere, including the Baltic States.
- Canadian Museum of Nature, Flora of the Canadian Arctic Archipelago: <https://nature.ca/aaflora/data/www/cyervg.htm>. Binding: dense compact tussocks; persistent basal sheaths; mostly basal 0.6–1.2 mm filiform leaves distinctly shorter than the culms; 1–3 bladeless inflated distal sheaths; a single erect terminal spike; 10 or more silky bristles elongating in fruit.
- eElurikkus accepted-name record: <https://elurikkus.ee/app/taxonomy/taxon/4617> and direct Estonia occurrence example: <https://elurikkus.ee/app/occurrences/occurrence/66627507>.
- Geometry: four separate compact tussocks; persistent sheath bodies; short basal leaves; triangular flowering culms with explicit inflated upper sheaths; exactly one terminal head per culm. Each fruiting head contains an indexed spike axis and 42 individually curved bristle ribbons. It is not a sphere or ellipsoid.

## Structural and claim boundaries

The focused tests require deterministic but unequal mesh hashes, unit normals, every vertex referenced by indexed triangles, strictly nonzero triangle area, exact component counts reached only through those nondegenerate faces at the plant/rhizome-network scale, profile-specific reproductive counts, grazing-angle periodic copy derivation, and periodic-address invariance under integer tile translations. These checks distinguish growth architecture instead of merely checking filenames.

The meshes are architectural flowering/fruiting profiles. They do not claim herbarium-grade micrometric reconstruction, calibrated color, reproductive timing, material chemistry, local abundance, one Estonian population's phenotype, wind state, or complete ecology. Grass spikelet internals, leaf pubescence, Carex utricle venation/stomatal surface, and Eriophorum individual floral organs are below this `64x64` geometric carrier's spatial ceiling. Those omissions must not be silently reinterpreted as absent botanical structures in a later higher-resolution profile.

Run the deterministic checks and real localhost bake with:

```sh
node --import tsx --test tools/groundcover-bake/EstonianGraminoids.test.ts
npx tsx tools/groundcover-bake/run-estonian-graminoids.ts --timeout 300000 --out /tmp/groundcover-gpu-bake-estonian-graminoids
```

Use `--profile 0` through `--profile 4` to run one profile without changing the recipe or output layout.
