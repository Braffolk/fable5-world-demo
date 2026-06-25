/**
 * voxlod occupancy-gate NO-HOLE unit check (CPU mirror of the GPU Phase-A mask build +
 * Phase-B gate in NaniteVoxelRaster.ts). Proves: for randomized bricks (center, half,
 * 4x4x4 occupancy) under a randomized yaw + perspective camera, EVERY screen pixel that
 * (a) lies in the brick's projected bbox AND (b) is covered by the projection of an OCCUPIED
 * sub-cell, has its Phase-B bucket bit SET in the mask => the gate never drops a covered
 * pixel (no hole). Run: npx tsx tools/voxlod-occgate-nohole.ts
 */
import { Matrix4, Vector3, Vector4 } from 'three';

const BRICK_DIM = 4;
const OCC_MASK_DIM = 4;
const OCC_GATE_MIN_AREA = OCC_MASK_DIM * OCC_MASK_DIM;
const W = 1512, H = 982;

function buildVP(camPos: Vector3, yaw: number, pitch: number): Matrix4 {
  const proj = new Matrix4();
  // standard perspective (fovy ~ 60deg), near 1 far 2000
  const f = 1 / Math.tan((60 * Math.PI / 180) / 2);
  const aspect = W / H;
  const near = 1, far = 2000;
  proj.set(
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
    0, 0, -1, 0,
  );
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const dir = new Vector3(sy * cp, sp, -cy * cp);
  const view = new Matrix4().lookAt(camPos, camPos.clone().add(dir), new Vector3(0, 1, 0));
  const rot = new Matrix4().extractRotation(view).transpose();
  const t = new Matrix4().makeTranslation(-camPos.x, -camPos.y, -camPos.z);
  return proj.multiply(rot).multiply(t);
}
function instXform(p: Vector3, yaw: number, scale: number, origin: Vector3): Vector3 {
  const ls = p.clone().multiplyScalar(scale);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  return new Vector3(ls.x * cy + ls.z * sy, ls.y, ls.z * cy - ls.x * sy).add(origin);
}
function project(vp: Matrix4, w: Vector3): { x: number; y: number; w: number } {
  const v = new Vector4(w.x, w.y, w.z, 1).applyMatrix4(vp);
  return { x: (v.x / v.w + 1) * 0.5 * W, y: (v.y / v.w + 1) * 0.5 * H, w: v.w };
}
interface Brick { center: Vector3; half: number; occ: boolean[]; }
function cellCenterLocal(b: Brick, cx: number, cy: number, cz: number): Vector3 {
  const cs = (2 * b.half) / BRICK_DIM, hd = BRICK_DIM * 0.5;
  return new Vector3(
    b.center.x + (cx + 0.5 - hd) * cs,
    b.center.y + (cy + 0.5 - hd) * cs,
    b.center.z + (cz + 0.5 - hd) * cs,
  );
}
function brickBbox(vp: Matrix4, b: Brick, yaw: number, scale: number, origin: Vector3) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9, anyFront = false;
  for (let s = 0; s < 8; s++) {
    const lp = new Vector3(
      b.center.x + ((s & 1) ? b.half : -b.half),
      b.center.y + ((s & 2) ? b.half : -b.half),
      b.center.z + ((s & 4) ? b.half : -b.half),
    );
    const sp = project(vp, instXform(lp, yaw, scale, origin));
    if (sp.w <= 1e-4) continue;
    anyFront = true;
    minX = Math.min(minX, sp.x); maxX = Math.max(maxX, sp.x);
    minY = Math.min(minY, sp.y); maxY = Math.max(maxY, sp.y);
  }
  if (!anyFront) return null;
  const x0 = Math.max(0, Math.floor(minX)), y0 = Math.max(0, Math.floor(minY));
  const ex = Math.min(W - 1, Math.ceil(maxX)), ey = Math.min(H - 1, Math.ceil(maxY));
  if (x0 > ex || y0 > ey) return null;
  return { x0, y0, w: ex - x0 + 1, h: ey - y0 + 1 };
}
// CONSERVATIVE mask: each occupied cell marks EVERY bucket its full screen bbox overlaps
// (project the cell's 8 corners -> screen bbox -> bucket range). Superset of true coverage =>
// provably no holes at ANY perspective (a cell that spans many buckets near-camera marks them
// all => near bricks stay solid; a small far cell marks ~1 bucket => far carves the interior).
function buildMask(vp: Matrix4, b: Brick, bb: { x0: number; y0: number; w: number; h: number }, yaw: number, scale: number, origin: Vector3): number {
  if (b.occ.every((o) => o)) return 0xffff;
  let mask = 0;
  for (let cz = 0; cz < BRICK_DIM; cz++)
    for (let cy = 0; cy < BRICK_DIM; cy++)
      for (let cx = 0; cx < BRICK_DIM; cx++) {
        if (!b.occ[cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM]) continue;
        const c = cellCenterLocal(b, cx, cy, cz), hc = b.half / BRICK_DIM;
        let mnX = 1e9, mnY = 1e9, mxX = -1e9, mxY = -1e9, front = false;
        for (let s = 0; s < 8; s++) {
          const lp = new Vector3(c.x + ((s & 1) ? hc : -hc), c.y + ((s & 2) ? hc : -hc), c.z + ((s & 4) ? hc : -hc));
          const sp = project(vp, instXform(lp, yaw, scale, origin));
          if (sp.w <= 1e-4) continue;
          front = true;
          mnX = Math.min(mnX, sp.x); mxX = Math.max(mxX, sp.x);
          mnY = Math.min(mnY, sp.y); mxY = Math.max(mxY, sp.y);
        }
        if (!front) continue;
        const u0 = Math.max(0, Math.min(OCC_MASK_DIM - 1, Math.floor(((mnX - bb.x0) / bb.w) * OCC_MASK_DIM)));
        const u1 = Math.max(0, Math.min(OCC_MASK_DIM - 1, Math.floor(((mxX - bb.x0) / bb.w) * OCC_MASK_DIM)));
        const v0 = Math.max(0, Math.min(OCC_MASK_DIM - 1, Math.floor(((mnY - bb.y0) / bb.h) * OCC_MASK_DIM)));
        const v1 = Math.max(0, Math.min(OCC_MASK_DIM - 1, Math.floor(((mxY - bb.y0) / bb.h) * OCC_MASK_DIM)));
        for (let v = v0; v <= v1; v++) for (let u = u0; u <= u1; u++) mask |= 1 << (v * OCC_MASK_DIM + u);
      }
  return mask === 0 ? 0xffff : mask;
}
function pixelPainted(mask: number, lx: number, ly: number, bb: { w: number; h: number }): boolean {
  const su = Math.min(Math.floor((lx * OCC_MASK_DIM) / bb.w), OCC_MASK_DIM - 1);
  const sv = Math.min(Math.floor((ly * OCC_MASK_DIM) / bb.h), OCC_MASK_DIM - 1);
  return ((mask >> (sv * OCC_MASK_DIM + su)) & 1) === 1;
}
function coveredByOcc(vp: Matrix4, b: Brick, x: number, y: number, yaw: number, scale: number, origin: Vector3): boolean {
  for (let cz = 0; cz < BRICK_DIM; cz++)
    for (let cy = 0; cy < BRICK_DIM; cy++)
      for (let cx = 0; cx < BRICK_DIM; cx++) {
        if (!b.occ[cx + cy * BRICK_DIM + cz * BRICK_DIM * BRICK_DIM]) continue;
        const c = cellCenterLocal(b, cx, cy, cz), hc = b.half / BRICK_DIM;
        let mnX = 1e9, mnY = 1e9, mxX = -1e9, mxY = -1e9, front = false;
        for (let s = 0; s < 8; s++) {
          const lp = new Vector3(c.x + ((s & 1) ? hc : -hc), c.y + ((s & 2) ? hc : -hc), c.z + ((s & 4) ? hc : -hc));
          const sp = project(vp, instXform(lp, yaw, scale, origin));
          if (sp.w <= 1e-4) continue;
          front = true;
          mnX = Math.min(mnX, sp.x); mxX = Math.max(mxX, sp.x);
          mnY = Math.min(mnY, sp.y); mxY = Math.max(mxY, sp.y);
        }
        if (front && x >= mnX && x <= mxX && y >= mnY && y <= mxY) return true;
      }
  return false;
}
function rng(seed: number) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 0xffffffff); }

let holes = 0, trials = 0, armed = 0, carvedBuckets = 0, totalBuckets = 0;
const r = rng(987654321);
for (let t = 0; t < 6000; t++) {
  const occ: boolean[] = [];
  const fill = 0.08 + r() * 0.6;
  for (let i = 0; i < 64; i++) occ.push(r() < fill);
  if (!occ.some((o) => o)) occ[Math.floor(r() * 64)] = true;
  const half = 0.3 + r() * 2.2;
  const b: Brick = { center: new Vector3((r() - 0.5) * 4, (r() - 0.5) * 8, (r() - 0.5) * 4), half, occ };
  const origin = new Vector3((r() - 0.5) * 20, 0, (r() - 0.5) * 20);
  const yaw = r() * Math.PI * 2, scale = 0.7 + r() * 0.6, dist = 8 + r() * 120;
  const camPos = new Vector3(origin.x + (r() - 0.5) * 10, 2 + r() * 6, origin.z + dist);
  const vp = buildVP(camPos, (r() - 0.5) * 0.4, -0.05 + (r() - 0.5) * 0.2);
  const bb = brickBbox(vp, b, yaw, scale, origin);
  if (!bb) continue;
  trials++;
  if (bb.w * bb.h < OCC_GATE_MIN_AREA) continue;
  armed++;
  const mask = buildMask(vp, b, bb, yaw, scale, origin);
  totalBuckets += 16; for (let k = 0; k < 16; k++) if (!((mask >> k) & 1)) carvedBuckets++;
  for (let ly = 0; ly < bb.h; ly++)
    for (let lx = 0; lx < bb.w; lx++) {
      const x = bb.x0 + lx, y = bb.y0 + ly;
      if (!coveredByOcc(vp, b, x, y, yaw, scale, origin)) continue;
      if (!pixelPainted(mask, lx, ly, bb)) {
        holes++;
        if (holes <= 3) console.error(`HOLE t=${t} px(${lx},${ly}) bb=${JSON.stringify(bb)} mask=0x${mask.toString(16)} half=${half.toFixed(2)} dist=${dist.toFixed(1)}`);
      }
    }
}
console.log(`trials(bbox-valid)=${trials} armed=${armed} HOLES=${holes} carvedBuckets=${carvedBuckets}/${totalBuckets} (${(100*carvedBuckets/Math.max(1,totalBuckets)).toFixed(1)}% buckets dropped => the savings)`);
process.exit(holes === 0 ? 0 : 1);
