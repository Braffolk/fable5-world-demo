/**
 * Off-thread DAG builder (N8-D1d, D-N30; extended 2026-07-04 for the cold-boot
 * arc). Runs in a module Worker so heavy CPU builds never block the boot
 * critical path / render loop:
 *  - 'height'    buildHeightGrid (terrain tiles — the original job)
 *  - 'mesh'      buildDag (QEM LOD DAG for vegetation heads)
 *  - 'aggregate' buildAggregateDag (leaf-crown area-preserving DAG)
 *  - 'crown'     prepareVoxelCrown (tri raster + MIP pyramid + block DAG),
 *                returned in the BootCache packed form (flat typed arrays)
 *  - 'rock'      generateRock (SDF-composed rock mesh bake, SPEC-ROCKS §G)
 * The whole build chain is three-free + typed-arrays in/out (VoxelizeCrown
 * imports VoxelBrickCore, not VoxelBrick), so this bundle carries no GPU/DOM
 * code. Output arrays are transferred back zero-copy; plain-number structs
 * ride the structured clone. Module knobs (cluster fill, agg errorK, voxlod
 * config, occupancy threshold) are forwarded PER REQUEST and applied to this
 * worker's module instances before each build — worker results must be
 * bit-identical to the synchronous fallback.
 */
/// <reference lib="webworker" />
import { buildHeightGrid } from './BuildHeightGrid';
import { buildDag } from './BuildDag';
import { buildAggregateDag, setAggLodErrorK } from './BuildAggregateDag';
import { setClusterFill } from './Clusterize';
import { packPreparedCrown } from '../world/BootCache';
import { prepareVoxelCrown, setVoxOccThreshold, setVoxlodConfig } from './VoxelizeCrown';
import { generateRock } from '../../vegetation/RockGen';
import type { ExplicitSource } from '../world/GeometryRegistry';
import type { DagReq, DagRes } from './DagWorkerTypes';

// DOM and webworker libs both define `self`; the cast pins the worker scope so
// postMessage takes a transfer list (the DOM overload would take targetOrigin).
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<DagReq>): void => {
  const req = e.data;
  try {
    if (req.kind === 'height') {
      const hfArgs = {
        heights: req.heights,
        gridN: req.gridN,
        cellSize: req.cellSize,
        originX: req.originX,
        originZ: req.originZ,
      };
      const b = buildHeightGrid(hfArgs, req.opts);
      const res: DagRes = {
        id: req.id,
        ok: true,
        kind: 'height',
        gridVerts: b.gridVerts,
        indices: b.indices,
        clusters: b.clusters,
        stats: b.stats,
      };
      ctx.postMessage(res, [b.gridVerts.buffer, b.indices.buffer]);
      return;
    }
    if (req.kind === 'mesh' || req.kind === 'aggregate') {
      setClusterFill(req.clusterFill);
      if (req.kind === 'aggregate') setAggLodErrorK(req.aggErrorK);
      const dag =
        req.kind === 'mesh'
          ? buildDag(req.verts, req.vertStride, req.indices, req.opts)
          : buildAggregateDag(req.verts, req.vertStride, req.indices, req.opts);
      const res: DagRes = { id: req.id, ok: true, kind: req.kind, dag };
      ctx.postMessage(res, [dag.verts.buffer, dag.indices.buffer]);
      return;
    }
    if (req.kind === 'crown') {
      setVoxOccThreshold(req.occThreshold);
      setVoxlodConfig(req.cfg);
      const src: ExplicitSource = {
        kind: 'mesh',
        positions: req.positions,
        normals: req.normals,
        uvs: req.uvs,
        vdata: req.vdata,
        indices: req.indices,
      };
      const pack = packPreparedCrown(prepareVoxelCrown(src, req.color, req.gridDim, req.voxlod));
      const transfer: Transferable[] = [
        pack.vox.grid.occupied.buffer,
        pack.vox.grid.words.buffer,
      ];
      for (const l of pack.vox.levels ?? []) {
        transfer.push(l.occupied.buffer, l.words.buffer);
      }
      const res: DagRes = { id: req.id, ok: true, kind: 'crown', pack };
      ctx.postMessage(res, transfer);
      return;
    }
    if (req.kind === 'rock') {
      const mesh = generateRock(req.archetype, req.variant, req.seed, req.gridRes, req.mod, req.domainScale);
      const res: DagRes = { id: req.id, ok: true, kind: 'rock', mesh };
      ctx.postMessage(res, [
        mesh.positions.buffer,
        mesh.normals.buffer,
        mesh.vdata.buffer,
        mesh.indices.buffer,
      ]);
      return;
    }
    const bad: DagRes = { id: (req as { id: number }).id, ok: false, error: `unknown kind` };
    ctx.postMessage(bad);
  } catch (err) {
    const res: DagRes = { id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) };
    ctx.postMessage(res);
  }
};
