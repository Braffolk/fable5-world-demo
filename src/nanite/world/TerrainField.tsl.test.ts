import assert from 'node:assert/strict';
import test from 'node:test';

import { vec2 } from 'three/tsl';
import type { NV2 } from '../../gpu/TSLTypes';
import type { FieldPlan, PlanePlan } from './PlaneFill';
import { TerrainField } from './TerrainField';

function heightPlan(lod: number, texel: number): PlanePlan {
  return {
    lod,
    res: 8,
    texel,
    wraps: false,
    originX: texel * 0.5,
    originZ: texel * 0.5,
    stride: 1,
    n0x: 0,
    n0z: 0,
    nMinX: 0,
    nMaxX: 7,
    nMinZ: 0,
    nMaxZ: 7,
  };
}

test('packed cold height samplers construct outside a TSL Fn stack', () => {
  const plan: FieldPlan = {
    cookedMicroHeight: true,
    height: [heightPlan(-2, 0.0625), heightPlan(-1, 0.25), heightPlan(0, 1)],
    biome: [],
    fields: [],
    water: null,
    waterFar: null,
    waterCover: null,
    waterCoverFar: null,
    soil: null,
    geology: null,
    groundCoverA: null,
    groundCoverB: null,
    groundCoverC: null,
    coverageBox: { minX: 0, minZ: 0, maxX: 8, maxZ: 8 },
    biomeHasCanopy: false,
  };
  const field = TerrainField.fromPlan(plan);
  const p = vec2(0.2, 0.2) as unknown as NV2;

  assert.ok(field.fieldHeightFinest(p));
  assert.ok(field.fieldHeightFinestNearest(p));
});
