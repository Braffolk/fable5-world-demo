import type { BakeResult, IndexedMesh, ProjectionSlice } from './ProfileFormat';

interface BakeRequest {
  mesh: IndexedMesh;
  slices: ProjectionSlice[];
  tileWidth: number;
  tileHeight: number;
  atlasColumns: number;
  atlasRows: number;
}

declare global {
  interface Window {
    __groundCoverBake: {
      ready: boolean;
      error: string | null;
      run: (request: BakeRequest) => Promise<BakeResult>;
    };
  }
}

const shader = /* wgsl */ `
struct Projection {
  axisU_uMin: vec4f,
  axisV_vMin: vec4f,
  ray_depthMin: vec4f,
  invSpan: vec4f,
};
@group(0) @binding(0) var<uniform> projection: Projection;

struct VertexIn {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
};
struct VertexOut {
  @builtin(position) clip: vec4f,
  @location(0) depth: f32,
  @location(1) normal: vec3f,
};

@vertex fn vertexMain(input: VertexIn) -> VertexOut {
  let u = dot(input.position, projection.axisU_uMin.xyz);
  let v = dot(input.position, projection.axisV_vMin.xyz);
  let d = dot(input.position, projection.ray_depthMin.xyz);
  let uv = vec2f(
    (u - projection.axisU_uMin.w) * projection.invSpan.x,
    (v - projection.axisV_vMin.w) * projection.invSpan.y
  );
  let depth = (d - projection.ray_depthMin.w) * projection.invSpan.z;
  var out: VertexOut;
  out.clip = vec4f(uv * 2.0 - 1.0, depth, 1.0);
  out.depth = depth;
  out.normal = input.normal;
  return out;
}

fn encodeOct(normalInput: vec3f) -> vec2f {
  let n = normalize(normalInput);
  var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  if (n.z < 0.0) {
    p = (vec2f(1.0) - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
  }
  return p * 0.5 + 0.5;
}

@fragment fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  return vec4f(input.depth, encodeOct(input.normal), 1.0);
}
`;

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

async function run(request: BakeRequest): Promise<BakeResult> {
  if (!navigator.gpu) throw new Error('navigator.gpu is unavailable');
  if (request.slices.length > request.atlasColumns * request.atlasRows) throw new Error('atlas does not fit slices');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const info = adapter.info;
  const device = await adapter.requestDevice({ label: 'groundcover-offline-baker' });
  const uncaptured: string[] = [];
  device.addEventListener('uncapturederror', (event) => uncaptured.push(event.error.message));
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');
  device.pushErrorScope('out-of-memory');

  const atlasWidth = request.tileWidth * request.atlasColumns;
  const atlasHeight = request.tileHeight * request.atlasRows;
  const color = device.createTexture({
    label: 'groundcover-first-hit-normal',
    size: [atlasWidth, atlasHeight],
    format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depth = device.createTexture({
    label: 'groundcover-first-hit-depth-test',
    size: [atlasWidth, atlasHeight],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const interleaved = new Float32Array((request.mesh.positions.length / 3) * 6);
  for (let vertex = 0; vertex < request.mesh.positions.length / 3; vertex++) {
    interleaved.set(request.mesh.positions.slice(vertex * 3, vertex * 3 + 3), vertex * 6);
    interleaved.set(request.mesh.normals.slice(vertex * 3, vertex * 3 + 3), vertex * 6 + 3);
  }
  const indices = new Uint32Array(request.mesh.indices);
  const vertexBuffer = device.createBuffer({
    label: 'groundcover-mesh-vertices',
    size: interleaved.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const indexBuffer = device.createBuffer({
    label: 'groundcover-mesh-indices',
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  const uniformBuffer = device.createBuffer({
    label: 'groundcover-projection',
    size: request.slices.length * 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, interleaved);
  device.queue.writeBuffer(indexBuffer, 0, indices);

  const module = device.createShaderModule({ label: 'groundcover-raster-project', code: shader });
  const pipeline = device.createRenderPipeline({
    label: 'groundcover-raster-project',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [{
        arrayStride: 24,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
        ],
      }],
    },
    fragment: { module, entryPoint: 'fragmentMain', targets: [{ format: 'rgba32float' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const bindGroups: GPUBindGroup[] = [];
  request.slices.forEach((slice, i) => {
    const b = slice.bounds;
    const values = new Float32Array([
      slice.axisU.x, slice.axisU.y, slice.axisU.z, b.uMin,
      slice.axisV.x, slice.axisV.y, slice.axisV.z, b.vMin,
      slice.direction.x, slice.direction.y, slice.direction.z, b.depthMin,
      1 / (b.uMax - b.uMin), 1 / (b.vMax - b.vMin), 1 / (b.depthMax - b.depthMin), 0,
    ]);
    device.queue.writeBuffer(uniformBuffer, i * 256, values);
    bindGroups.push(device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: i * 256, size: 64 } }],
    }));
  });

  const encoder = device.createCommandEncoder({ label: 'groundcover-bake-encoder' });
  request.slices.forEach((_slice, i) => {
    const pass = encoder.beginRenderPass({
      label: `groundcover-slice-${i}`,
      colorAttachments: [{
        view: color.createView(),
        clearValue: { r: -1, g: 0.5, b: 0.5, a: 0 },
        loadOp: i === 0 ? 'clear' : 'load',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depth.createView(),
        depthClearValue: 1,
        depthLoadOp: i === 0 ? 'clear' : 'load',
        depthStoreOp: 'store',
      },
    });
    const tileX = (i % request.atlasColumns) * request.tileWidth;
    const tileY = Math.floor(i / request.atlasColumns) * request.tileHeight;
    pass.setViewport(tileX, tileY, request.tileWidth, request.tileHeight, 0, 1);
    pass.setScissorRect(tileX, tileY, request.tileWidth, request.tileHeight);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroups[i] as GPUBindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint32');
    pass.drawIndexed(indices.length);
    pass.end();
  });

  const unpaddedBytesPerRow = atlasWidth * 16;
  const bytesPerRow = align(unpaddedBytesPerRow, 256);
  const readback = device.createBuffer({
    label: 'groundcover-readback',
    size: bytesPerRow * atlasHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyTextureToBuffer(
    { texture: color },
    { buffer: readback, bytesPerRow, rowsPerImage: atlasHeight },
    [atlasWidth, atlasHeight],
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readback.getMappedRange());
  const tight = new Uint8Array(unpaddedBytesPerRow * atlasHeight);
  for (let y = 0; y < atlasHeight; y++) {
    tight.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow), y * unpaddedBytesPerRow);
  }
  const pixels = Array.from(new Float32Array(tight.buffer));
  readback.unmap();

  const oom = await device.popErrorScope();
  const internal = await device.popErrorScope();
  const validation = await device.popErrorScope();
  const errors = [oom, internal, validation].filter((e): e is GPUError => e !== null).map((e) => e.message);
  errors.push(...uncaptured);
  color.destroy();
  depth.destroy();
  vertexBuffer.destroy();
  indexBuffer.destroy();
  uniformBuffer.destroy();
  readback.destroy();
  device.destroy();
  if (errors.length > 0) throw new Error(`WebGPU error(s): ${errors.join(' | ')}`);

  return {
    tileWidth: request.tileWidth,
    tileHeight: request.tileHeight,
    atlasColumns: request.atlasColumns,
    atlasRows: request.atlasRows,
    slices: request.slices,
    pixels,
    adapter: `${info.vendor || 'unknown'} ${info.architecture || ''} ${info.device || ''}`.trim(),
  };
}

window.__groundCoverBake = { ready: true, error: null, run };
