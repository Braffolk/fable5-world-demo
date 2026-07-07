# PLAN — cut the projection whale (nanProjectVerts)

**Task #76 continuation. Date 2026-07-08. Follows docs/tasks/2026-07-07/PLAN-visbuffer-rewrite.md
(the vis-buffer rewrite — DONE, committed; world1 80→56).**

## The target (measured)
Frame profile after the rewrite: **`nanProjectVerts` = 30%** (the single most costly compute shader, ~2×
the next), **`nanMidRaster` = 14.5%**. Projection **looks memory-bound** (vertex reads + the 1.13 GB
`projVertBuf` writes). So: cut projection work → cut memory pressure → the whale shrinks.

## Lever 1 (bounded, do first): per-UNIQUE-VERT projection dispatch
The meshletize deduped the *storage* (`projVertBuf` @stride 512) but **`nanProjectVerts` still dispatches
per-tri-CORNER** (128 tris × 3 = 384 threads/cluster) — each corner projects its vert and writes the
deduped `canonVertSlot`, so a shared vert is **projected ~6× and written once**. Fix: dispatch per unique
vert — iterate `[0, uniqueCount)`, for mesh read `gpu.verts[vBase+slot]`, project once, write once.
Halves the mesh transforms **and** the vert-read + projVertBuf-write traffic (the mem-bound part).
Terrain-DAG stays per-corner (≤384, not the whale). The enumeration already exists (`canonVertSlot`).

## Lever 2 (bigger, harder): don't project cull-doomed verts
"Shouldn't do anything on already-culled stuff." clhw/voxel clusters are already skipped (clhw-default grew
that). The rest: verts whose tris all cull (backface/degenerate) or the far sub-pixel bulk (only needs a
splat centroid, not 3 stored corners). Needs the cull decision BEFORE projection (ordering problem) — this
is the plan-§6 compacted scheme (project only surviving verts → also shrinks projVertBuf toward ~700 MB).

## Carried-over open items (from the 07-07 audit §13)
- **`Classify.ts` extraction** (cleanliness) — classifier still in NaniteRaster.ts. Decide: extract or amend.
- **High-water readback** — not wired; the 511 clamp / 192K cap is the only overflow backstop. Wire it.
- **56→50** register polish; **frame A/B**; **append-contention**; per-kernel register captures.

## State to resume from
Branch `nanite-raster`, rewrite+clhw committed. world1=56, projVertBuf 1.13 GB flat-512 @192K cap,
meshletize storage-dedup done, clhw=1 default. Files: `src/nanite/raster/{Project,Classify-in-NaniteRaster,
Mid,Splat,Hw,ClusterCtx,Queues,Scanline,VisBuffer,index}`, `BuildDag.meshletizeDag`.
