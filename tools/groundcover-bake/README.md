# Offline GPU ground-cover profile baker

This is the first production-oriented mesh-to-ray-profile path. It accepts an ordinary indexed triangle mesh, affinely projects the vertices into a fixed ray-origin frame, and lets hardware rasterization plus `depth32float` elect the first hit. The fragment payload is normalized first-hit depth plus an octahedral world normal. No geometry assumption is made by the GPU path; the ellipsoid in `run.ts` is an analytic validation fixture only.

Run the focused checks and the real WebGPU bake with:

```sh
node --import tsx --test tools/groundcover-bake/ProfileFormat.test.ts tools/groundcover-bake/PeriodicProfile.test.ts tools/groundcover-bake/SphagnumCapillifolium.test.ts
npx tsx tools/groundcover-bake/run.ts --timeout 300000 --out /tmp/groundcover-gpu-bake
```

The runner starts a throwaway Vite server on `localhost`, launches full Chromium WebGPU, submits the same indexed mesh for eight fixed azimuth/elevation directions, reads the float atlas back, and packs it twice. A run fails on any browser/WebGPU diagnostic, a non-identical packed repeat, or an analytic error outside the recorded tolerance. Its content-addressed output contains:

- `ellipsoid-profile.gcrp`: little-endian v1 binary with the profile id, atlas layout, affine frame and projection bounds for each slice, and RGBA16 texels (`depth`, `oct-x`, `oct-y`, `hit`).
- `qa/01-first-hit-depth.png` and `qa/02-first-hit-normal.png`: numbered visual diagnostics.
- `qa/index.json`: binary/image/recipe hashes, dimensions, interpretations, adapter identity, validation results, and the resource projection.

## Runtime contract and cost

This baker is offline and may rasterize arbitrarily complex meshes. Its runtime consumer remains fixed-cost O(1): a regular direction lattice uses four predetermined direction-corner taps for one profile, or at most eight predetermined taps for the existing bounded A/B cover mixture. There is no per-frame ray march and no data-dependent shader loop.

The first proof artifact uses `96x96 x 8` RGBA16 texels: 576 KiB of payload per profile. A fuller `64x64 x 16 azimuth x 4 elevation` bank is 2 MiB/profile, or 24 MiB for 12 profiles before guards/mips (approximately 30 MiB budgeted). All profiles belong in one filterable 2D-array/3D atlas plus one sampler, so profile count grows offline storage but not runtime binding count. Sampling is coherent inside profile/type regions; the fixed corner reads have ordinary atlas locality and no synchronization or barrier requirement.

The present artifact proves the arbitrary-mesh projection/readback/packing path. It is not a botanical species asset and makes no Estonia-native provenance claim.

## Periodic top-plane path

`PeriodicProfile.ts`, `periodic-page.ts`, and `run-periodic.ts` add the runtime-compatible projection
`rayOriginXZ = p.xz - d.xz * ((p.y - topH) / d.y)`. Every direction addresses the same canonical
world-XZ tile. Periodic source-copy ranges are derived from mesh XZ/height bounds, ray slope, and tile
dimensions; the packed GCRP/v2 slices have one wrapped texel gutter so bilinear taps cannot enter a
neighbor slice. The fixture is a connected/overlapping geometric cushion carpet and is explicitly not
a botanical or Sphagnum model.

The diagnostic surface-owner disagreement was traced to fixed-function 8-bit subpixel raster ownership:
the ideal ray oracle missed a silhouette triangle that the hardware correctly covered after window-edge
quantization. The CPU acceptance now keeps its ideal-ray strict-interior selection but transcribes the
submitted-f32 raster election for expected depth and normal. The corrected path passes twice
deterministically; `PERIODIC-KNOWN-ISSUE.md` retains the full failure and resolution evidence. The v1
orthogonal ellipsoid oracle above remains unchanged.

### Sphagnum capillifolium species fixture

The first botanical profile is original deterministic procedural indexed geometry for the vegetative
form of native `Sphagnum capillifolium`. It is selected explicitly so the geometric cushion proof remains
the default and cannot silently become a species claim:

```sh
npx tsx tools/groundcover-bake/run-periodic.ts --fixture sphagnum-capillifolium --timeout 300000 --out /tmp/groundcover-gpu-bake-sphagnum
```

The 24 cm periodic tile is one connected irregular hummock carpet carrying 88 non-grid capitula. Each
capitulum is assembled from a stem, central core, 9–12 explicit radial flattened branches, and forked
branchlets, retaining inter-branch and inter-head negative space at the 64 texel bake resolution. It is
not a union of smooth ellipsoid caps and ingests no external mesh or texture.

Species profile ID `5` is independent from functional cover ID `1` (`Moss`). The production lattice is
16 azimuths (`0..337.5` degrees in 22.5-degree steps) by four elevations (`15, 35, 55, 75` degrees), in
stable azimuth-major/elevation-minor order:
`sliceIndex = azimuthIndex * 4 + elevationIndex`. Each slice has a 64×64 canonical interior and a wrapped
one-texel gutter, producing a 66×66×64 RGBA16 payload. The QA index records the exact mesh recipe/hash,
direction table, adapter, two-bake deterministic hash, CPU/raster acceptance, numbered depth/normal
images, and the strict claim boundary. The binary carries geometry depth and normals only; later material
shading may use the recorded per-capitulum palette phase, but this asset makes no calibrated color,
reproductive-stage, chemistry, abundance, geographic-occurrence, or complete-ecology claim.
