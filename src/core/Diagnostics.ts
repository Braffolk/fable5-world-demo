/**
 * WebGPU capability probe + fail-loud reporting.
 * Spec: no WebGL fallback. If WebGPU is unavailable we stop and explain.
 */

import type { GpuDiagnostics } from './Hooks';

const INTERESTING_LIMITS: readonly (keyof GPUSupportedLimits & string)[] = [
  'maxTextureDimension2D',
  'maxTextureDimension3D',
  'maxTextureArrayLayers',
  'maxBindGroups',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupsPerDimension',
  'maxComputeInvocationsPerWorkgroup',
  'maxColorAttachments',
  'maxStorageBuffersPerShaderStage',
  'maxStorageTexturesPerShaderStage',
  'maxSampledTexturesPerShaderStage',
  'maxUniformBuffersPerShaderStage',
];

/**
 * Limits we ask the device for (clamped to what the adapter reports).
 * Compute passes bind many storage buffers — the default 8 is not enough.
 *
 * maxSampledTexturesPerShaderStage: the nanite resolve übershader samples the
 * terrain maps + probe GI + 4 CSM cascades + canopy + caustics + the bark
 * texture-array — 17, over the spec-DEFAULT 16. This adapter reports 48, so we
 * raise the request (D-N23/D-N25: storage buffers are the real cap at 10/F9,
 * "sampled textures are separate and plentiful"). Clamped to adapterMax below,
 * so lower hardware degrades to its own ceiling rather than failing the device.
 */
export function buildRequiredLimits(d: GpuDiagnostics): Record<string, number> {
  const want: Record<string, number> = {
    maxStorageBuffersPerShaderStage: 16,
    maxStorageTexturesPerShaderStage: 8,
    maxSampledTexturesPerShaderStage: 24,
    maxBufferSize: 1 << 30,
    maxStorageBufferBindingSize: 1 << 30,
  };
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(want)) {
    const adapterMax = d.limits[k];
    if (adapterMax !== undefined) out[k] = Math.min(v, adapterMax);
  }
  return out;
}

export async function probeWebGPU(): Promise<GpuDiagnostics> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    return {
      ok: false,
      reason: 'navigator.gpu is missing — this browser has no WebGPU. Use Chrome 113+.',
      features: [],
      limits: {},
    };
  }
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    return {
      ok: false,
      reason: `requestAdapter threw: ${e instanceof Error ? e.message : String(e)}`,
      features: [],
      limits: {},
    };
  }
  if (!adapter) {
    return {
      ok: false,
      reason:
        'requestAdapter returned null — WebGPU present but no adapter. If headless, check launch flags (e.g. --enable-unsafe-webgpu, ANGLE backend).',
      features: [],
      limits: {},
    };
  }
  const limits: Record<string, number> = {};
  for (const k of INTERESTING_LIMITS) {
    const v = adapter.limits[k];
    if (typeof v === 'number') limits[k] = v;
  }
  const info = adapter.info;
  return {
    ok: true,
    vendor: info?.vendor ?? 'unknown',
    architecture: info?.architecture ?? 'unknown',
    device: info?.device ?? 'unknown',
    description: info?.description ?? '',
    features: [...adapter.features].map(String).sort(),
    limits,
  };
}

let failShown = false;

/** Render an unmissable full-screen failure overlay and record it on hooks. */
export function failLoud(title: string, details: string[]): void {
  if (window.__laas) window.__laas.error = `${title}\n${details.join('\n')}`;
  // eslint-disable-next-line no-console
  console.error('[LAAS FATAL]', title, details);
  if (failShown) return;
  failShown = true;

  const boot = document.getElementById('boot');
  if (boot) boot.style.display = 'none';

  const el = document.createElement('div');
  el.id = 'laas-fatal';
  el.style.cssText = [
    'position:fixed', 'inset:0', 'z-index:99999', 'background:#140b0b',
    'color:#ff9d9d', 'font-family:ui-monospace,Menlo,monospace', 'padding:48px',
    'overflow:auto', 'white-space:pre-wrap', 'line-height:1.5',
  ].join(';');
  const h = document.createElement('div');
  h.textContent = `✕ ${title}`;
  h.style.cssText = 'font-size:24px;color:#ff5c5c;margin-bottom:20px;font-weight:bold';
  el.appendChild(h);
  const body = document.createElement('div');
  body.style.cssText = 'font-size:13px;color:#e0b0b0';
  body.textContent = details.join('\n');
  el.appendChild(body);
  document.body.appendChild(el);
}

/** Route uncaught errors to the overlay so failures are never silent. */
export function installGlobalErrorHooks(): void {
  window.addEventListener('error', (ev) => {
    failLoud('Uncaught error', [String(ev.message), ev.filename ? `at ${ev.filename}:${ev.lineno}` : '']);
  });
  window.addEventListener('unhandledrejection', (ev) => {
    const r: unknown = ev.reason;
    const msg = r instanceof Error ? `${r.message}\n${r.stack ?? ''}` : String(r);
    failLoud('Unhandled rejection', [msg]);
  });
}

export function describeDiagnostics(d: GpuDiagnostics): string[] {
  return [
    `adapter: ${d.vendor ?? '?'} / ${d.architecture ?? '?'} ${d.description ? `(${d.description})` : ''}`,
    `features: ${d.features.join(', ') || 'none reported'}`,
    ...Object.entries(d.limits).map(([k, v]) => `  ${k} = ${v}`),
  ];
}
