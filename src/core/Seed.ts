/**
 * Deterministic seeding. The entire world derives from a single uint32 seed
 * (`?seed=N`). Independent subsystems get decorrelated streams via string-keyed
 * derivation so adding a consumer never shifts another stream's sequence.
 */

/** FNV-1a 32-bit over a string, for stream derivation. */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Final avalanche mix (murmur3 fmix32). */
export function mix32(h: number): number {
  h = h >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Combine two 32-bit values into one well-mixed 32-bit value. */
export function hashCombine(a: number, b: number): number {
  return mix32((Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca77)) >>> 0);
}

/**
 * sfc32-based PRNG — fast, solid statistical quality for procedural content.
 */
export class Rng {
  private a: number;
  private b: number;
  private c: number;
  private d: number;

  constructor(seed: number) {
    // splitmix-style state expansion from a single 32-bit seed
    let s = seed >>> 0;
    const next = (): number => {
      s = (s + 0x9e3779b9) >>> 0;
      return mix32(s);
    };
    this.a = next();
    this.b = next();
    this.c = next();
    this.d = next();
    for (let i = 0; i < 8; i++) this.u32();
  }

  /** uniform uint32 */
  u32(): number {
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    return t;
  }

  /** uniform [0,1) */
  float(): number {
    return this.u32() / 4294967296;
  }

  /** uniform [min,max) */
  range(min: number, max: number): number {
    return min + (max - min) * this.float();
  }

  /** uniform integer in [0,n) */
  int(n: number): number {
    return Math.floor(this.float() * n);
  }

  /** standard normal (Box–Muller) */
  gauss(): number {
    let u = 0;
    let v = 0;
    while (u === 0) u = this.float();
    while (v === 0) v = this.float();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  /** true with probability p */
  chance(p: number): boolean {
    return this.float() < p;
  }

  /** pick uniformly from a non-empty array */
  pick<T>(arr: readonly T[]): T {
    const item = arr[this.int(arr.length)];
    if (item === undefined && arr.length === 0) {
      throw new Error('Rng.pick on empty array');
    }
    return item as T;
  }

  /** derive an independent child stream */
  fork(label: string): Rng {
    return new Rng(hashCombine(this.u32(), hashString(label)));
  }

  /**
   * Snapshot this stream at its current position. The clone produces the exact
   * same subsequent sequence as the original, without advancing the original —
   * used by the crown-LOD ladder to REPLAY the identical foliage RNG stream for
   * every pruned level, so survivors are bit-identical to the full-density mesh.
   */
  clone(): Rng {
    const r = new Rng(0);
    r.a = this.a;
    r.b = this.b;
    r.c = this.c;
    r.d = this.d;
    return r;
  }
}

/** Root world seed container with named stream derivation. */
export class WorldSeed {
  readonly seed: number;

  constructor(seed: number) {
    this.seed = seed >>> 0;
  }

  /** stable, order-independent stream for a named subsystem */
  rng(stream: string): Rng {
    return new Rng(hashCombine(this.seed, hashString(stream)));
  }

  /** stable 32-bit sub-seed for GPU passes (uniform into shaders) */
  sub(stream: string): number {
    return hashCombine(this.seed, hashString(stream));
  }
}
