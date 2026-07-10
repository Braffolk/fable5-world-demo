/** Scene registry — `?scene=` selects the boot scene (world | terrain | gallery | forest). */

import type { Engine } from '../core/Engine';
import type { LaasHooks } from '../core/Hooks';
import type { LaasParams } from '../core/Params';
import type { WorldSeed } from '../core/Seed';

export interface WorldContext {
  engine: Engine;
  params: LaasParams;
  seed: WorldSeed;
  hooks: LaasHooks;
  /** report build progress 0..1 */
  progress: (p: number, msg: string) => void;
}

export type SceneBuilder = (ctx: WorldContext) => Promise<void>;

const registry = new Map<string, SceneBuilder>();

export function registerScene(name: string, builder: SceneBuilder): void {
  registry.set(name, builder);
}

export async function buildScene(name: string, ctx: WorldContext): Promise<void> {
  const builder = registry.get(name);
  if (!builder) {
    const known = [...registry.keys()].join(', ');
    throw new Error(`Unknown scene "${name}". Known scenes: ${known}`);
  }
  await builder(ctx);
}
