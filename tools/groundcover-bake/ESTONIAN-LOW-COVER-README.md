# Estonian low-cover botanical profile track (IDs 6–11)

## Inspectable outcome and limits

This track authors six deterministic, species-distinct indexed meshes and bakes each through the existing
GCRP/v2 periodic top-XZ Apple-Metal path. The output target is six content-addressed binaries plus numbered
depth/normal QA images and a machine-readable acceptance index under `/tmp/groundcover-gpu-bake-lowcover/`.

The meshes are original procedural geometry. No downloaded mesh, image, texture, herbarium scan, or other
copyrighted asset is ingested. Geometry carries first-hit depth and normals only. It does not claim calibrated
material color, seasonal state, local abundance, age distribution, environmental plasticity, or complete
botanical anatomy.

The uncertain-track limit is two real end-to-end GPU attempts. After one failure, diagnosis, correction, and
one consolidated rerun are allowed. A second failure parks the affected profile with its code, diagnostic JSON,
exact blocker, and resume condition intact; it does not weaken the raster gate.

## Fixed bake and runtime contract

- Exact native-profile namespace: IDs `6..11` in the order below.
- Production direction bank: 16 azimuths at 22.5° spacing × elevations 15°, 35°, 55°, 75°.
- Stable order: azimuth-major/elevation-minor, `slice = azimuthIndex * 4 + elevationIndex`.
- Each slice: 64×64 canonical XZ interior plus one exactly wrapped gutter texel.
- Offline election: fixed-function rasterization with `depth32float`, then RGBA16 GCRP/v2 packing.
- Determinism: two GPU submissions must produce the same packed SHA-256.
- Runtime implication: four fixed direction-corner reads for one profile, or eight for the bounded A/B mixture.
  Offline mesh complexity, repeated source copies, branching depth, and leaf count add no runtime ray march,
  data-dependent loop, synchronization, or extra per-instance geometry.

Run focused structural tests and the real localhost bake with:

```sh
node --import tsx --test tools/groundcover-bake/EstonianLowCover.test.ts
npx tsx tools/groundcover-bake/run-estonian-low-cover.ts \
  --profile all --timeout 300000 --out /tmp/groundcover-gpu-bake-lowcover
```

One or more profiles can be selected with a comma-separated ID list such as `--profile 6,7`.

## Source ledger and geometry bindings

Sources were accessed on 2026-07-21. Public flora/consortium descriptions constrain growth form, stature,
branch/leaf organization, and discriminating silhouettes. eElurikkus binds the accepted taxon or an actual
Estonian occurrence; it is not silently treated as a quantitative abundance model.

### 6 — *Pleurozium schreberi*

- [British Bryological Society species account](https://www.britishbryologicalsociety.org.uk/learning/species-finder/pleurozium-schreberi/)
  and its linked field-guide sheet: pleurocarp forming dense patches; several-centimetre shoots; loose, neat,
  simple pinnation; concave oval leaves around 2–2.5 mm; smaller branch leaves.
- [eElurikkus “Meie naabrid” species list](https://elurikkus.ee/projects/meie-naabrid): recorded in Estonian
  urban forest under the Estonian name `palusammal`.
- Encoded form: 24 prostrate-to-ascending axes, 96 alternating lateral branches, and 480 explicit concave
  surface leaves. This is a low interwoven shoot mat, not rounded cushion primitives.

### 7 — *Cladonia rangiferina*

- [Consortium of Lichen Herbaria species account](https://lichenportal.org/portal/taxa/index.php?taxon=Cladonia+rangiferina):
  densely aggregated/caespitose secondary thallus, 50–120 mm high, 0.8–1.8 mm wide, branched podetia, and an
  evanescent primary thallus.
- [eElurikkus accepted taxon](https://elurikkus.ee/app/taxonomy/taxon/139351): accepted Estonian identity and
  common name `harilik põdrasamblik`.
- Encoded form: 30 upright terete podetial axes with repeated two-way crown forks. There are no cups and no
  persistent basal cap carpet. Tube diameter remains within the cited stipe range.

### 8 — *Oxalis acetosella*

- [World Flora Online, Flora of China description](https://www.worldfloraonline.org/taxon/wfo-0000387238):
  slender creeping rhizome, 3–15 cm petioles, three obcordate leaflets with a deep distal notch, and solitary
  nodding five-part flowers.
- [eElurikkus native/persistent urban-forest list](https://elurikkus.ee/projects/meie-naabrid): native/persistent
  classification and Estonian woodland record under `harilik jänesekapsas`.
- Encoded form: an edge-crossing creeping rhizome, 18 long petioles, exactly 54 heart-notched leaflets, and
  three sparse five-part flowers. The three-leaflet organization is a hard structural test.

### 9 — *Maianthemum bifolium*

- [World Flora Online, Flora of China description](https://www.worldfloraonline.org/taxon/wfo-0000691280):
  8–25 cm rhizomatous plants, usually two distal cordate leaves, 3–5 cm erect racemes with 10–25 flowers, and
  one-leaf vegetative shoots.
- [eElurikkus Estonia occurrence](https://elurikkus.ee/app/occurrences/occurrence/44461108): verified Estonian
  occurrence and accepted species identity.
- Encoded form: an edge-crossing rhizome, four one-leaf sterile shoots, eight two-leaf flowering shoots, broad
  cordate blades, and terminal four-part racemes. It is structurally unlike the trifoliate Oxalis profile.

### 10 — *Vaccinium myrtillus*

- [World Flora Online, Flora of China and Flora of North America descriptions](https://www.worldfloraonline.org/taxon/wfo-0000422209):
  rhizomatous, much-branched dwarf shrub; conspicuously three-angled green twigs; alternate ovate/elliptic,
  serrulate leaves around 1–3 cm.
- [eElurikkus Estonia occurrence](https://elurikkus.ee/app/occurrences/occurrence/66598395): verified Estonian
  occurrence and accepted species identity.
- Encoded form: nine explicit triangular-section primary axes, 36 angular branches, and 252 alternate serrated
  leaves. The profile intentionally omits flowers and berries rather than baking one reproductive phase into
  all cover.

### 11 — *Calluna vulgaris*

- [World Flora Online, Flora of North America and Flora Helvetica descriptions](https://www.worldfloraonline.org/taxon/wfo-0000580837):
  richly branched dwarf shrub; persistent 1–3.5 mm scale leaves, densely imbricate and arranged in four rows.
- [eElurikkus accepted taxon](https://elurikkus.ee/app/taxonomy/taxon/3309): accepted Estonian identity and
  common name `kanarbik`.
- Encoded form: 18 bushy axes, 144 first-order ascending branches, 288 explicit second-order branchlets,
  5,760 decussate nodes, and 23,040 tightly overlapping scale leaves in exactly four ranks. It does not reuse
  the broad-leaf Vaccinium construction.

## Structural and periodic acceptance

`EstonianLowCover.test.ts` checks deterministic reproduction, exact IDs/names, valid indexed triangles,
unit normals, substantial shared-edge topology, species-specific organ counts, all six distinct growth-form
fingerprints, and periodic address invariance under integer tile translations. It also asks the production
15° grazing projection for both forward and reverse azimuths and requires conservative source copies across
both X and Z seams.

The runner exercises larger positive and negative coordinates as well and accepts at most 32 binary64 epsilons
of address error (`7.105427357601002e-15`). This is the explicit arithmetic roundoff bound for multiplying
non-power-of-two metre tile sizes; stored one-texel gutters are still copied bit-for-bit from the opposite
canonical edge.

The runner additionally performs the existing selected-pixel CPU/raster validation, exact wrapped-gutter pack,
two-bake hash comparison, browser console/page/WebGPU diagnostic capture, and numbered PNG/index generation.
For thin leaves and podetia it applies the same submitted-f32, eight-fractional-bit raster transcription even
when the ideal continuous center ray has no hit. This closes the remaining form of the already-documented
fixed-function ownership case: a quantized triangle edge can legitimately cover a pixel center just outside
the unsnapped analytic triangle. Such samples are counted separately as `subpixelOnlyRasterHits`; a GPU value
that matches neither ideal nor transcribed raster ownership still fails.
