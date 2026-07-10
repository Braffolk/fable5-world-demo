/**
 * Shared TSL node type aliases. @types/three 0.184 types TSL nodes generically
 * (`Node<'vec3'>` carries swizzles/operators for vec3); bare `Node` has none.
 * Use these aliases for all TSL helper signatures.
 */

import type { Node } from 'three/webgpu';

export type NF = Node<'float'>;
export type NI = Node<'int'>;
export type NU = Node<'uint'>;
export type NB = Node<'bool'>;
export type NV2 = Node<'vec2'>;
export type NV3 = Node<'vec3'>;
export type NV4 = Node<'vec4'>;
export type NIV2 = Node<'ivec2'>;
export type NM4 = Node<'mat4'>;

// @types/three types exp/log as float-only; WGSL builtins are component-wise.
// Single deliberate cast point for vector transcendental math.
import { exp as tslExp } from 'three/tsl';
export function vexp3(v: NV3): NV3 {
  return tslExp(v as unknown as NF) as unknown as NV3;
}
