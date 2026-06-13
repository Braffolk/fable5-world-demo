/**
 * CachedCsmShadowNode — per-cascade shadow caching (Phase 7 perf directive:
 * cascade re-render was ~13–19 ms/frame at heavy bookmarks; the sun is
 * static between ToD edits and far-cascade content barely changes frame to
 * frame, so most of that is identical work redone).
 *
 * Cadence: cascade i re-fits + re-renders every PERIODS[i] frames (phases
 * staggered so far cascades never pile onto one frame). Between refreshes
 * BOTH the light pose and the map are frozen — the sampling matrix derives
 * from the light pose, so a moved light with a cached map would translate
 * every shadow on screen (swimming). Forced refresh when:
 *   - the sun direction changes (ToD edit) → all cascades,
 *   - the would-be fit center drifts > 4% of the cascade span (fast camera
 *     motion / teleport; the per-frame fit is a texel-snapped translation
 *     of a rotation-invariant square, so center drift captures all of it),
 *   - updateFrustums() runs (resize / camera change → extents change).
 *
 * Quality: near cascade refreshes every frame (wind sway in contact
 * shadows stays live); c1 at /2 (≥30 Hz at 60 fps), c2 /3, c3 /6 — far
 * cascades hold mostly-rigid content (impostor-band proxies, terrain) and
 * their texels are 1–10 m wide; a few frames of latency is sub-texel.
 */

import { Box3, Matrix4, Vector3 } from 'three';
import { CSMFrustum } from 'three/addons/csm/CSMFrustum.js';
import { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';
import type { Camera, Light, Object3D } from 'three/webgpu';
import {
  Fn,
  If,
  cameraViewMatrix,
  float,
  min,
  reference,
  renderGroup,
  shadowPositionWorld,
  uniform,
  vec2,
  vec4,
  viewZToOrthographicDepth,
} from 'three/tsl';
import type { NF, NV4 } from '../gpu/TSLTypes';

const PERIODS = [1, 2, 3, 6];
const PHASES = [0, 1, 2, 5];
/** fraction of the cascade span the fit center may drift before a forced refresh */
const DRIFT_FRAC = 0.04;

const _lightDirection = new Vector3();
const _lightOrientationMatrix = new Matrix4();
const _lightOrientationMatrixInverse = new Matrix4();
const _cameraToLightMatrix = new Matrix4();
const _lightSpaceFrustum = new CSMFrustum({ webGL: false });
const _bbox = new Box3();
const _center = new Vector3();
const _up = new Vector3(0, 1, 0);

interface ShadowLike {
  autoUpdate: boolean;
  needsUpdate: boolean;
  mapSize: { width: number; height: number };
  camera: { left: number; right: number; top: number; bottom: number };
}

export class CachedCsmShadowNode extends CSMShadowNode {
  private frameNo = 0;
  private frozenCenters: (Vector3 | undefined)[] = [];
  private lastSunDir = new Vector3();
  private lastFov = 0;
  private lastAspect = 0;

  constructor(light: Light, data?: ConstructorParameters<typeof CSMShadowNode>[1]) {
    super(light, data);
  }

  /** drop all cached fits — every cascade re-renders next frame */
  invalidate(): void {
    this.frozenCenters.length = 0;
  }

  /**
   * Cascade-select view depth derived from shadowPositionWorld instead of
   * three's `positionView.z`. For normal geometry the two are identical
   * (shadowPositionWorld defaults to positionWorld). For the nanite resolve
   * (a fullscreen triangle with `vertexNode`, N4/D-N18), `positionView`
   * reconstructs a NEAR-PLANE point (right direction, wrong magnitude —
   * Position.js fragment branch), which pinned every pixel to cascade 0;
   * the material supplies the true per-pixel world position via
   * `receivedShadowPositionNode`, which the shadow system assigns to
   * `shadowPositionWorld` BEFORE these bodies run (ShadowBaseNode contract).
   */
  private cascadeLinearDepth(shadowFar: NF): NF {
    type Grouped = { setGroup(g: unknown): unknown };
    const cameraNear = (reference('camera.near', 'float', this) as unknown as Grouped).setGroup(
      renderGroup,
    ) as unknown as NF;
    const spw4 = (vec4 as unknown as (a: unknown, b: number) => NV4)(shadowPositionWorld, 1);
    const viewZ = (cameraViewMatrix as unknown as { mul(v: NV4): NV4 }).mul(spw4).z;
    const ld = viewZToOrthographicDepth(viewZ, cameraNear, shadowFar) as unknown as {
      toVar(): NF;
    };
    return ld.toVar();
  }

  // _setupFade/_setupStandard are vendored from three 0.184's CSMShadowNode
  // with ONLY the linearDepth source swapped (cascadeLinearDepth above) —
  // verify against the addon source on any three upgrade (same caveat as
  // updateBefore below). The base methods read `positionView.z` directly.
  private _setupFadeShadowPos(): unknown {
    const self = this as unknown as {
      cascades: number;
      maxFar: number;
      camera: { far: number } | null;
      _shadowNodes: { oneMinus(): { mul(r: unknown): unknown } }[];
      setupShadowPosition(builder: unknown): void;
    };
    const cascades = (
      reference('_cascades', 'vec2', this) as unknown as {
        setGroup(g: unknown): { setName(n: string): unknown };
      }
    )
      .setGroup(renderGroup)
      .setName('cascades') as unknown as { element(i: number): NV4 };
    const shadowFar = (
      ((uniform as unknown as (t: string) => { setGroup(g: unknown): { setName(n: string): unknown } })('float'))
        .setGroup(renderGroup)
        .setName('shadowFar') as unknown as { onRenderUpdate(fn: () => number): NF }
    ).onRenderUpdate(() => Math.min(self.maxFar, self.camera?.far ?? self.maxFar));
    const linearDepth = this.cascadeLinearDepth(shadowFar);
    const lastCascade = self.cascades - 1;

    return Fn((builder: unknown) => {
      self.setupShadowPosition(builder);
      const ret = vec4(1, 1, 1, 1).toVar();
      const cascade = vec2().toVar();
      const cascadeCenter = float(0).toVar();
      const margin = float(0).toVar();
      const csmX = float(0).toVar();
      const csmY = float(0).toVar();
      for (let i = 0; i < self.cascades; i++) {
        const isLastCascade = i === lastCascade;
        cascade.assign(cascades.element(i) as unknown as Parameters<typeof cascade.assign>[0]);
        cascadeCenter.assign(cascade.x.add(cascade.y).div(2.0));
        const closestEdge = linearDepth.lessThan(cascadeCenter).select(cascade.x, cascade.y);
        margin.assign(float(0.25).mul(closestEdge.pow(2.0)));
        csmX.assign(cascade.x.sub(margin.div(2.0)));
        if (isLastCascade) {
          csmY.assign(cascade.y);
        } else {
          csmY.assign(cascade.y.add(margin.div(2.0)));
        }
        const inRange = linearDepth.greaterThanEqual(csmX).and(linearDepth.lessThanEqual(csmY));
        If(inRange, () => {
          const dist = min(linearDepth.sub(csmX), csmY.sub(linearDepth)).toVar();
          let ratio = dist.div(margin).clamp(0.0, 1.0);
          if (i === 0) {
            // don't fade at nearest edge
            ratio = linearDepth.greaterThan(cascadeCenter).select(ratio, 1) as typeof ratio;
          }
          const shadowNode = self._shadowNodes[i];
          if (shadowNode) {
            ret.subAssign(
              shadowNode.oneMinus().mul(ratio) as unknown as Parameters<typeof ret.subAssign>[0],
            );
          }
        });
      }
      return ret;
    })();
  }

  private _setupStandardShadowPos(): unknown {
    const self = this as unknown as {
      cascades: number;
      maxFar: number;
      camera: { far: number } | null;
      _shadowNodes: unknown[];
      setupShadowPosition(builder: unknown): void;
    };
    const cascades = (
      reference('_cascades', 'vec2', this) as unknown as {
        setGroup(g: unknown): { setName(n: string): unknown };
      }
    )
      .setGroup(renderGroup)
      .setName('cascades') as unknown as { element(i: number): NV4 };
    const shadowFar = (
      ((uniform as unknown as (t: string) => { setGroup(g: unknown): { setName(n: string): unknown } })('float'))
        .setGroup(renderGroup)
        .setName('shadowFar') as unknown as { onRenderUpdate(fn: () => number): NF }
    ).onRenderUpdate(() => Math.min(self.maxFar, self.camera?.far ?? self.maxFar));
    const linearDepth = this.cascadeLinearDepth(shadowFar);

    return Fn((builder: unknown) => {
      self.setupShadowPosition(builder);
      const ret = vec4(1, 1, 1, 1).toVar();
      const cascade = vec2().toVar();
      for (let i = 0; i < self.cascades; i++) {
        cascade.assign(cascades.element(i) as unknown as Parameters<typeof cascade.assign>[0]);
        If(
          linearDepth.greaterThanEqual(cascade.x).and(linearDepth.lessThanEqual(cascade.y)),
          () => {
            ret.assign(self._shadowNodes[i] as unknown as Parameters<typeof ret.assign>[0]);
          },
        );
      }
      return ret;
    })();
  }

  override setup(builder: Parameters<CSMShadowNode['setup']>[0]): ReturnType<CSMShadowNode['setup']> {
    const self = this as unknown as {
      camera: unknown;
      fade: boolean;
      _init(builder: unknown): void;
    };
    if (self.camera === null) self._init(builder);
    return (
      self.fade === true ? this._setupFadeShadowPos() : this._setupStandardShadowPos()
    ) as ReturnType<CSMShadowNode['setup']>;
  }

  override updateFrustums(): void {
    super.updateFrustums();
    this.invalidate();
  }

  // mirrors CSMShadowNode.updateBefore (three 0.184) with the per-cascade
  // freshness gate; verify against the addon source on any three upgrade
  override updateBefore(): boolean | undefined {
    const light = this.light as unknown as { parent: Object3D | null } & Light & {
      target: Object3D;
      position: Vector3;
    };
    const parent = light.parent;
    const camera = this.camera as (Camera & { matrixWorld: Matrix4 }) | null;
    const frustums = this.frustums;
    if (camera === null || parent === null) return;

    for (const lwLight of this.lights) {
      if (lwLight.parent === null) {
        parent.add(lwLight.target);
        parent.add(lwLight);
      }
    }

    _lightDirection.subVectors(light.target.position, light.position).normalize();
    if (_lightDirection.distanceToSquared(this.lastSunDir) > 1e-10) {
      this.lastSunDir.copy(_lightDirection);
      this.invalidate();
    }

    // frustum-shape changes (sprint FOV kick, resize races) refit cascades —
    // CSM extents derive from the camera frustum and would silently go stale
    const pc = camera as unknown as { fov?: number; aspect?: number };
    if (typeof pc.fov === 'number' && typeof pc.aspect === 'number') {
      if (pc.fov !== this.lastFov || pc.aspect !== this.lastAspect) {
        this.lastFov = pc.fov;
        this.lastAspect = pc.aspect;
        this.updateFrustums(); // calls invalidate() via our override
      }
    }

    _lightOrientationMatrix.lookAt(light.position, light.target.position, _up);
    _lightOrientationMatrixInverse.copy(_lightOrientationMatrix).invert();

    for (let i = 0; i < frustums.length; i++) {
      const lwLight = this.lights[i];
      const frustum = frustums[i];
      const shadow = lwLight?.shadow as unknown as ShadowLike | undefined;
      if (!lwLight || !frustum || !shadow) continue;
      const shadowCam = shadow.camera;
      // we own the cadence (ShadowNode renders on needsUpdate || autoUpdate)
      shadow.autoUpdate = false;

      const texelWidth = (shadowCam.right - shadowCam.left) / shadow.mapSize.width;
      const texelHeight = (shadowCam.top - shadowCam.bottom) / shadow.mapSize.height;
      _cameraToLightMatrix.multiplyMatrices(_lightOrientationMatrixInverse, camera.matrixWorld);
      frustum.toSpace(_cameraToLightMatrix, _lightSpaceFrustum);

      const nearVerts = _lightSpaceFrustum.vertices.near;
      const farVerts = _lightSpaceFrustum.vertices.far;
      _bbox.makeEmpty();
      for (let j = 0; j < 4; j++) {
        _bbox.expandByPoint(nearVerts[j] as Vector3);
        _bbox.expandByPoint(farVerts[j] as Vector3);
      }
      _bbox.getCenter(_center);
      _center.z = _bbox.max.z + this.lightMargin;
      _center.x = Math.floor(_center.x / texelWidth) * texelWidth;
      _center.y = Math.floor(_center.y / texelHeight) * texelHeight;
      _center.applyMatrix4(_lightOrientationMatrix);

      const frozen = this.frozenCenters[i];
      const span = shadowCam.right - shadowCam.left;
      const scheduled =
        (this.frameNo + (PHASES[i] ?? 0)) % (PERIODS[i] ?? 1) === 0;
      if (
        frozen !== undefined &&
        !scheduled &&
        frozen.distanceTo(_center) < span * DRIFT_FRAC
      ) {
        continue; // cached: light pose AND map stay frozen together
      }

      lwLight.position.copy(_center);
      lwLight.target.position.copy(_center).add(_lightDirection);
      shadow.needsUpdate = true;
      const slot = this.frozenCenters[i] ?? new Vector3();
      slot.copy(_center);
      this.frozenCenters[i] = slot;
    }
    this.frameNo++;
    return undefined;
  }
}
