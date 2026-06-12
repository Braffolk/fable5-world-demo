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
