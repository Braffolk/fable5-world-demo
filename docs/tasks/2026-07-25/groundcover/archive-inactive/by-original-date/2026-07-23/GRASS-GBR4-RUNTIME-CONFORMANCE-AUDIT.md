# GBR4 runtime conformance audit

**Date:** 2026-07-23  
**Scope:** the complete active isolated-Calamagrostis producer, container,
loader, decoder, terrain/control closure, invocation, and resolve path  
**Status:** **RED — the active path is not an implementation of the complete
exterior boundary-transfer mathematics.** It is an internally consistent,
fixed-cost decoder for the deliberately coarse `GBR4/v2` periodic-cap
container proof.

This audit compares the active implementation against two load-bearing
mathematical layers:

1. the continuous exterior carrier/boundary-transfer construction together
   with its terrain, motion, control, finite-patch, and invocation closure; and
2. the finite physically filtered codec/reconstruction contract.

The governing sources are:

- `GRASS-COMPLETE-MODEL-BOUNDARY.md` (only its still-binding finite-state and
  resource results; its pointed-origin obstruction is superseded);
- `GRASS-BOX-BOUNDARY-TRANSFER-MATH.md`;
- `GRASS-BOX-TERRAIN-MOTION-CLOSURE.md`; and
- `GRASS-GBR4-BRICK-CODEC.md`.

The active files were read in full:

- `src/nanite/grass/NaniteGrass.ts`;
- `src/nanite/shade/NaniteResolve.ts`;
- `src/nanite/frame/NaniteFrame.ts`;
- `src/nanite/groundcover/GroundCoverGbr4.ts`; and
- `tools/groundcover-bake/cook-gbr4-periodic-cap-assets.ts`.

The inspection also covered the stable `.gbr4` asset and its content-addressed
cook report.

## 1. Executive verdict

The active implementation must not be described as the general exterior
ground-cover solution. It is a **coarse isolated periodic-cap experiment**.

The narrow decoder mechanics are sound:

- one fixed 4D brick lookup;
- ten terminal texture reads;
- no loop, march, runtime geometry, candidate list, or per-species traversal;
- correct lossless block addressing;
- correct two-stratum premultiplied composition; and
- correct periodic later-copy truth after a valid cap entry.

The complete model is nevertheless RED because the runtime lacks or
contradicts load-bearing state:

- no `rho_P`/`rho_M` carrier first-passage field or finite-patch side entry;
- no fixed cooked terrain chart;
- no root/copy/plane/owner token and no exact winner-plane reintersection;
- no all-event firstness certificate;
- no certified regular versus filtered-mixed record classes;
- no live footprint/standoff state or footprint cascade;
- no full-screen sky/horizon support;
- exact horizontal/tangent directions are rejected;
- control is applied after selecting a winner;
- the finite horizon is tested against the wrong quantity; and
- an explicitly non-production `R=6`, one-footprint asset is used as visible
  grass.

The broad tan/green terrain-following mosaic is therefore not a texture-
resolution problem in isolation. The implementation interpolates coverage and
colour from several unrelated 4D histories, assigns that mixture one nearest
categorical depth/normal, and transports it through a camera/scene-dependent
terrain tangent plane. No final colour adjustment can turn that record into a
real plant surface.

The original proof documents already prohibit this publication state:

- `GRASS-COMPLETE-MODEL-BOUNDARY.md:166-179` says runtime remains held until a
  valid finite filter family passes;
- `GRASS-BOX-BOUNDARY-TRANSFER-MATH.md:605-628` records the actual-source finite
  macrobrick gate as rejected; and
- the asset report itself sets `visualProductionReady: false` and lists
  finite-patch side entry, production `R/VQ`, the scale cascade, held-out
  visual/translation gates, and runtime implementation as missing.

## 2. Verdict vocabulary

- **PASS:** the active producer and consumer satisfy the clause.
- **PARTIAL:** a special infinite-periodic-cap case is correct, but not the
  promised general domain.
- **FAIL-MATH:** active behavior contradicts a governing identity/domain rule.
- **MISSING-PRODUCER:** a required cooked field, token, certificate, or record
  class does not exist.
- **MISSING-FINAL-FEATURE:** required for generic ground cover, but not the
  cause of the current one-species visual corruption.
- **ACCOUNTING:** resource/read claims do not describe the active allocations.

## 3. Continuous carrier and exterior-domain audit

| Requirement | Active implementation | Verdict |
|---|---|---|
| Exterior quality domain is `o` outside the guarded carrier union; camera inside the whole guarded box/union may fade. | `outsideCarrier` tests only height relative to one extrapolated plane (`NaniteGrass.ts:1668-1675`). It has no horizontal footprint membership. | **FAIL-MATH** generally; **PARTIAL** for a full-tile infinite slab. |
| One common-slab lane is `U=P x I`. | Runtime has only `I`; it does not store/query `P`. | **MISSING-PRODUCER**. |
| `rho_P(q,omega)` obtains the first cap or side entry and skips arbitrarily many footprint gaps in one query. | Entry is always selected from top/bottom by `sign(kGround)` and divided by `kGround` (`1642-1653`). | **MISSING-PRODUCER / FAIL-MATH**. |
| Exact vertical rays use categorical footprint occupancy. | No footprint occupancy exists; vertical rays merely use the cap formula. | **MISSING-PRODUCER**. |
| Exact horizontal rays are the best-conditioned `rho_P` case. | `stableCap = abs(kGround)>1e-5` rejects them (`1673-1675`); divisions have already been formed. | **FAIL-MATH**. |
| Finite ecological patches use `rho_M`, a boundary-owner token, entry-state control, and live line reintersection. | None exists. The infinite periodic query is made first and control is sampled at the returned hit afterward (`1715-1738`). | **MISSING-PRODUCER / FAIL-MATH**. |
| First entry either yields a botanical hit before patch exit or carries a first-entry closure certificate proving no later patch can contribute. | No patch exit or closure certificate is encoded. | **MISSING-PRODUCER**. |
| Periodic copy identity is categorical and added only after selection. | The GBR4 categorical payload contains depth and normal only (`1040-1067`). | **MISSING-PRODUCER**. |
| Horizon accepts iff composed `t_U + tau <= R`. | The code rejects `tScene*|rd_xz| > RAY_END` (`1525-1527`) and later clips to underlying scene depth. It never evaluates the composed carrier-plus-transfer horizon. | **FAIL-MATH**. |
| Ties use a frozen face/owner/copy order and never blend categorical identity. | A nearest 4D lattice corner uses a deterministic half-cell rule, but no face/owner/copy token or certified visibility-cell tie exists. | **PARTIAL / MISSING-PRODUCER**. |
| Full-screen invocation covers every ray which may hit cover, including sky. | Compute dispatch covers `W*H`, but `tMax` without an elected scene pixel causes immediate return (`1472-1503`). Resolve also returns before the overlay for `elect==0` and explicitly declares sky unsupported (`NaniteResolve.ts:484-495,1563-1571`). | **FAIL-MATH**. |
| Scene depth is an optional cutoff, not the origin or existence proof for the cover query. | `tScene` is mandatory and becomes the terrain/carrier anchor. | **FAIL-MATH**. |
| No runtime grass geometry, carrier shell, floating plane, billboard, or mesh. | The GBR4 lane is a compute lookup plus resolve composition; no raised carrier geometry is wired. | **PASS**. The former 1.176 m floating geometry is absent. |

## 4. Terrain, root, motion, and control audit

| Requirement | Active implementation | Verdict |
|---|---|---|
| Terrain charts `(a_c,m_c)` are cooked, world-fixed, selected categorically, and never follow the camera/occluder. | The scene endpoint `O` is reconstructed per pixel; `groundO` and `gradO` are sampled there; that new per-pixel tangent plane becomes the carrier and transfer frame (`1472-1503,1596-1680`). | **FAIL-MATH**. |
| The resolved winner names source triangle/analytic chart, periodic copy, root, plane, domain, material/species, and deformation class. | Stored categorical data are only two distances and oct normals. Packed resolve metadata are normal/tip/profile (`1040-1067,1681-1708`). | **MISSING-PRODUCER**. |
| Terrain and deformation are evaluated at the actual winning root. | No root is returned, so no packed-terrain/control sample can occur at it. | **MISSING-PRODUCER**. |
| The transformed winner plane is reintersected with the live world ray. | The code transports a sampled relative distance using `t=tEntry+depth/metricSpeed` (`1660-1667,1710-1715`). That identity is valid for one chosen affine tangent chart only, not curved terrain without a root/plane correction. | **FAIL-MATH** generally; local affine algebra itself **PASS**. |
| Continued firstness is certified against every later event using support, incidence, barycentric, order, terrain, and deformation margins. | No certificate fields or margins are loaded or checked. | **MISSING-PRODUCER**. |
| Chart transitions are categorical or explicitly filtered, and regular records stay within an eroded chart interior. | No chart id, transition record, reach, or eroded-interior certificate exists. | **MISSING-PRODUCER**. |
| Geometry-affecting ecological state selects a complete jointly baked community before the query. | The already-selected infinite-periodic winner is rejected afterward by `hitControl` (`1715-1738`). Rejecting it does not reveal the correct later successor. | **FAIL-MATH**. |
| A cliff/ecology mask is a cooked availability decision, not a live correctness filter. | `|gradO|^2>1` hard-rejects the query (`1631-1636`). | **FAIL-MATH** as a general reconstruction rule. |
| Tier 1 uses two independent global affine layers, each inverse-transformed before querying and composed by true depth. | The isolated GBR4 path makes one static query and returns. | **MISSING-FINAL-FEATURE**. |
| Wind is root-fixed affine-in-height shear and smooth height/vigor is evaluated at the root. | Neither exists in the GBR4 path. | **MISSING-FINAL-FEATURE / MISSING-PRODUCER**. |
| Same-motion species are unioned before the query; independent motion groups use a fixed separate lane. | Only isolated Calamagrostis is available. There is no union-baked multi-species state, static moss lane, or cross-lane nearest-event composition. | **MISSING-FINAL-FEATURE**. |

## 5. Finite filtering and record-semantics audit

| Requirement | Active implementation | Verdict |
|---|---|---|
| A regular cell is certified to have one winner throughout it and returns one coupled owner/copy/plane/material payload. | No cell certification or regular-record class exists. | **MISSING-PRODUCER**. |
| A regular winner is reconstructed by live analytic plane intersection. | Only a sampled scalar distance is decoded. | **FAIL-MATH / MISSING-PRODUCER**. |
| An unresolved final mixed cell stores premultiplied appearance, conservative depth interval, and normal/material moments; it is never claimed to be one real surface. | All cells bilinearly interpolate two colours across up to sixteen 4D nodes, while one unrelated nearest node supplies depth/normal. Resolve then reconstructs and lights one fictitious world point (`sampleGbr4`, `NaniteResolve.ts:1621-1657`). | **FAIL-MATH — primary mosaic mechanism**. |
| Depth, normal, mark, owner, and copy remain coupled. | Depth and normal are coupled within the chosen categorical node, but not to the interpolated colour/coverage history. Owner/mark/copy are absent. | **PARTIAL locally; FAIL-MATH end-to-end**. |
| Categorical MISS never lends a cap depth to filtered neighbour coverage. | The new `categoricalHit = depth>0` gate prevents that narrow fabrication (`1726-1738`). | **PASS** narrowly. It creates holes where filtered coverage is real but the nearest categorical node is MISS because the format has no mixed-depth interval. |
| Macrobricks are classified as uniform regular/MISS, certified topology, or filtered mixed; optional correction is fixed and direct. | Current blocks are a lossless packing of a uniformly sampled 4D grid. There is no class bit, topology/palette, analytic payload, mixed VQ payload, or correction table. | **MISSING-PRODUCER**. |
| Mips/levels are independently integrated from the oracle at their physical footprint. | One LOD-0 field only; every texture read explicitly uses level zero. | **MISSING-PRODUCER**. |
| The live rank-two ray Jacobian selects a finest-field filter or finite standoff/covariance family. | Cook freezes one 4 m, 60°/1920, 2x2 footprint. Runtime uses that record at every actual standoff and has no Jacobian/scale selection. | **FAIL-MATH / MISSING-PRODUCER**. |
| Low-coverage values are preserved according to the declared physical tolerance. | Resolve discards alpha `<=1/255` without that threshold being part of the cook contract (`1625`). | **FAIL-CONTRACT**. |
| Mixed lighting is represented consistently. | Cook stores vertex albedo plus individual normals. Resolve lights the blended albedo with one representative normal and uses visibility at the underlying scene point. In general `E[a] L(N_rep) != E[a L(N)]`. | **FAIL-MATH** for mixed cells. |

## 6. GBR4 container, cooker, loader, and decoder audit

### 6.1 What is correct

1. The v2 cook traces one cap-origin ray through periodic successors to 155 m;
   later-tile hits are included.
2. Cook and runtime use the same rational compact slope chart.
3. Header offsets, section alignment, descriptor order, codeword layout, atlas
   flattening, and shared block-border nodes agree.
4. One descriptor plus four front-colour, four back-colour, and one
   categorical fetch equals ten terminal reads.
5. The terminal decode contains no loop, march, candidate, traversal, or
   runtime geometry.
6. The two cooked strata are disjoint subray integrals and the resolve equation
   `C0+C1+(1-a0-a1)Cscene` is correct for those integrals.
7. The selected categorical depth/normal pair comes from one real subray and
   is not numerically interpolated.
8. The affine inverse-transpose normal formula for one local tangent chart is
   correct.

### 6.2 What is not production-conformant

1. The active cook uses `R=6` on all four axes, `2x2` quadrature, and one 4 m
   footprint (`cook-gbr4-periodic-cap-assets.ts:32-40`). Its report explicitly
   says `visualProductionReady:false`. Binding it as visible grass contradicts
   its own publication status.
2. Six phase cells over 0.52 m are about 86.7 mm wide, while important plant
   elements are millimetric. This explains coarse fields, but increasing `R`
   alone does not repair the deeper semantic failures in Sections 3-5.
3. The rational square chart becomes extremely coarse near grazing. Measured
   `S=8` cells have median angular diameter about 31.5 degrees and p95/max near
   90 degrees; `S=16` is still about 16.2 degrees median and 43.2 degrees p95.
   The current `S=6` is worse. Boundary nodes also collapse to a few horizontal
   directions and are baked as MISS.
4. The selected certified regular/topology/mixed macrobrick codec was never
   produced. `GBR4/v2` is a different, uniform interpolation model.
5. The experimental direct-cap v3 cooker is unbound and does not solve the
   record-semantics, footprint, terrain, or finite-patch requirements. It must
   not replace v2 merely because it has a larger grid.
6. There is no held-out silhouette/RGB/depth/normal/mm-translation gate, no
   VQ/seam gate, and no simultaneous-residency result.

The bounded angular chart candidate for a future format is a Lambert
equal-area hemisphere disk:

\[
u={d_x\over\sqrt{1+|d_y|}},\qquad
v={d_z\over\sqrt{1+|d_y|}},
\]

with inverse inside the unit disk

\[
d_x=u\sqrt{2-r^2},\quad |d_y|=1-r^2,\quad
d_z=v\sqrt{2-r^2}.
\]

This is not authorized merely by appearing here: cap/side/corner chart
continuity and the full held-out gate remain required.

## 7. Read, bandwidth, and resident-memory audit

The advertised ten reads describe only `sampleGbr4`, not the complete active
pixel path.

The complete path additionally performs:

- the terrain guide samples needed to construct the moving local chart;
- guide/control sampling at `O`;
- one dependent hit-control sample per stratum;
- writes to two full-screen `RGBA16F` textures and one full-screen `RGBA32F`
  texture; and
- three full-screen texture reads in resolve.

The three overlay surfaces alone are 32 bytes per pixel:

- about 63.3 MiB at 1920x1080;
- about 112.5 MiB at 2560x1440; and
- about 253.1 MiB at 3840x2160.

That excludes `rayNrmTex`, the source assets, guide resources, and the rest of
the frame. It is not the intended low/mid-end fragment-path resource shape.

The loader's `residentBytes` is also wrong:

- descriptors expand from container `u16` to GPU `R32Uint`;
- categorical `RGBA16Uint` expands to GPU `RGBA32Uint`; but
- `residentBytes` charges three 8-byte atlases as though categorical remained
  8 bytes per texel (`GroundCoverGbr4.ts:473`).

For the current tiny asset the actual texture payload is roughly 295,040 bytes
rather than the reported 221,312 bytes. The relative error becomes material at
production scale. A later format can restore the intended 24-byte node by
packing two categorical `u16+oct16` records into `RG32Uint`, and can pack two
descriptor IDs per `R32Uint`, but those are format changes, not shader casts.

One conditional under-250-MiB production sketch exists — `P=256`, Lambert
`A=256`, `B=4`, a phase pyramid, and a shared roughly 11k-codeword VQ — at about
242.7 MiB before all missing carrier/control/work costs. It is not accepted:
its angular sufficiency depends on measured free-path statistics, its VQ has
not passed, and it leaves too little headroom for the complete product. A
scalar CPU cook would also be impractical; the final heavy cook must use the
existing GPU/offline path.

## 8. Exact source of the current visible mosaic

The observed image is consistent with four interacting violations:

1. a very coarse 4D field linearly interpolates broad premultiplied colour and
   coverage patches;
2. one nearest node supplies a depth/normal which does not own that mixture;
3. the sampled distance is lifted through a terrain plane built from the
   underlying scene endpoint; and
4. the result is clipped and composed only where an underlying scene election
   exists.

This produces a terrain-following coloured sheet/mosaic even though the RGB
values themselves came from the plant asset. It is neither a valid grass
surface nor literally a second copy of the terrain. The narrow depth-zero gate
removed one cap-plane fabrication case; it cannot repair the invalid mixed
record.

## 9. Required in-place replacement boundary

No legacy marching, GCRP projection, hardware shell, floating plane, billboard,
or grass mesh is to be restored. The current implementation work is preserved,
but the invalid model boundary must be replaced in this order:

1. **Freeze a conformant record model.** Regular cells carry a coupled
   root/copy/plane/material token and analytic plane intersection. Final mixed
   cells carry only filtered appearance plus conservative depth interval and
   moments; they never masquerade as one surface.
2. **Freeze a conformant carrier.** Produce guarded `P x I` carrier data,
   `rho_P`/finite-patch `rho_M`, exact cap/side/pole cases, boundary ownership,
   and the first-entry closure certificate.
3. **Freeze terrain/control closure.** Cook fixed terrain charts and transition
   records; return actual root and source plane; select a complete community
   state before querying; certify firstness under terrain/wind/height bounds.
4. **Freeze physical filtering.** Choose finest-field plus live correlated
   filtering or an explicitly charged finite standoff/covariance family. Bake
   every level independently from shared-origin pinhole truth.
5. **Fit the actual production codec.** Use a bounded angular chart, certified
   regular/topology/mixed bricks, honest resident accounting, and held-out
   spatial/angular/seam/mm-translation gates. A cook may fail rather than
   silently degrade.
6. **Transcribe one tiny fixed query path.** Support sky, exact horizontal and
   vertical rays, optional scene cutoff, finite horizon, and camera-in-box lane
   fade with no loop/march/candidate/runtime geometry. Avoid the three
   full-screen overlay allocations unless measured evidence justifies them.
7. **Add Tier 1 only after the single lane passes.** Two independent affines,
   root-fixed wind, same-motion multi-species union, and a separate static moss
   lane are fixed compositions of the accepted query, not new reconstruction
   approximations.

## 10. Acceptance boundary

The next user-visible URL is permitted only after all of the following are
true for the single Calamagrostis lane:

- fixed carrier and terrain charts are active;
- side/cap/horizontal/vertical cases are represented;
- no scene election is required for a possible cover hit;
- regular and mixed records have their declared distinct semantics;
- the live footprint state is selected correctly;
- root/plane/firstness data are consumed rather than omitted;
- the active asset is marked visual-production-ready by the held-out gate;
- complete resident bytes and reads are reported honestly;
- the exact URL passes a real WebGPU boot without validation/console errors;
  and
- visual acceptance remains the user's live flight review.

Until then, a clean compile or boot proves only plumbing, not mathematical or
visual conformance.
