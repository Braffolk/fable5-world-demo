/**
 * EnableF16 — prepend the WGSL `enable f16;` directive to shaders that use f16 (task #76 fp16
 * wind).
 *
 * three r184's directive path does NOT reliably place `enable f16;` at the module top for
 * material / resolve shaders, so raw-WGSL f16 fails to parse ("'f16' type used without 'f16'
 * extension enabled"). We patch the device's createShaderModule and prepend the directive to the
 * FINAL WGSL string — as a DIRECTIVE ONLY (never a type alias: an alias is a global declaration,
 * and WGSL requires ALL directives — including three's own `enable subgroups;` — to precede every
 * declaration, so prepending an alias would break "directives must come before global
 * declarations"). The f16-vs-f32 choice is a JS-level branch in windOffset(), not a shader alias.
 * Install only when the device supports shader-f16 (Engine gates it). Idempotent per device.
 */
const patchedDevices = new WeakSet<GPUDevice>();

export function enableShaderF16Directive(device: GPUDevice): void {
  if (patchedDevices.has(device)) return;
  patchedDevices.add(device);
  const dev = device as unknown as {
    createShaderModule(desc: GPUShaderModuleDescriptor): GPUShaderModule;
  };
  const orig = dev.createShaderModule.bind(dev);
  dev.createShaderModule = (desc: GPUShaderModuleDescriptor): GPUShaderModule => {
    if (
      typeof desc.code === 'string' &&
      desc.code.includes('f16(') &&
      !desc.code.includes('enable f16;')
    ) {
      return orig({ ...desc, code: `enable f16;\n${desc.code}` });
    }
    return orig(desc);
  };
}
