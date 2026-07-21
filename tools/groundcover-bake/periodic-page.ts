import type { IndexedMesh } from './ProfileFormat';
import type { PeriodicBakeResult, PeriodicSlice, PeriodicTile } from './PeriodicProfile';

interface PeriodicBakeRequest {
  mesh: IndexedMesh;
  tile: PeriodicTile;
  slices: PeriodicSlice[];
  tileWidth: number;
  tileHeight: number;
  atlasColumns: number;
  atlasRows: number;
}

declare global {
  interface Window {
    __groundCoverPeriodicBake: {
      ready: boolean;
      run: (request: PeriodicBakeRequest) => Promise<PeriodicBakeResult>;
    };
  }
}

const shader = /* wgsl */ `
struct Projection {
  ray_topH: vec4f,
  tileOrigin_size: vec4f,
  depth_copy: vec4f,
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
  let ray = projection.ray_topH.xyz;
  let p = input.position + vec3f(projection.depth_copy.z, 0.0, projection.depth_copy.w);
  let t = (p.y - projection.ray_topH.w) / ray.y;
  let rayOriginXZ = p.xz - ray.xz * t;
  let uv = (rayOriginXZ - projection.tileOrigin_size.xy) / projection.tileOrigin_size.zw;
  let depth = (t - projection.depth_copy.x) / (projection.depth_copy.y - projection.depth_copy.x);
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
  let incoming = projection.ray_topH.xyz;
  let candidate = normalize(input.normal);
  let facing = select(candidate, -candidate, dot(candidate, incoming) > 0.0);
  return vec4f(input.depth, encodeOct(facing), 1.0);
}
`;

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

async function run(request: PeriodicBakeRequest): Promise<PeriodicBakeResult> {
  if (!navigator.gpu) throw new Error('navigator.gpu is unavailable');
  if (request.slices.length > request.atlasColumns * request.atlasRows) throw new Error('atlas does not fit slices');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const info = adapter.info;
  const device = await adapter.requestDevice({ label: 'groundcover-periodic-offline-baker' });
  const uncaptured: string[] = [];
  device.addEventListener('uncapturederror', (event) => uncaptured.push(event.error.message));
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');
  device.pushErrorScope('out-of-memory');

  const atlasWidth = request.tileWidth * request.atlasColumns;
  const atlasHeight = request.tileHeight * request.atlasRows;
  const color = device.createTexture({
    label: 'groundcover-periodic-first-hit-normal',
    size: [atlasWidth, atlasHeight],
    format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depth = device.createTexture({
    label: 'groundcover-periodic-depth-election',
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
    label: 'groundcover-periodic-vertices',
    size: interleaved.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const indexBuffer = device.createBuffer({
    label: 'groundcover-periodic-indices',
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  const drawCount = request.slices.reduce((sum, slice) => sum + slice.copies.length, 0);
  const uniformBuffer = device.createBuffer({
    label: 'groundcover-periodic-projections',
    size: drawCount * 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, interleaved);
  device.queue.writeBuffer(indexBuffer, 0, indices);

  const module = device.createShaderModule({ label: 'groundcover-periodic-raster-project', code: shader });
  const pipeline = device.createRenderPipeline({
    label: 'groundcover-periodic-raster-project',
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

  const bindGroups: GPUBindGroup[][] = [];
  let drawIndex = 0;
  for (const slice of request.slices) {
    const groups: GPUBindGroup[] = [];
    for (const copy of slice.copies) {
      const values = new Float32Array([
        slice.direction.x, slice.direction.y, slice.direction.z, request.tile.topH,
        request.tile.originX, request.tile.originZ, request.tile.sizeX, request.tile.sizeZ,
        slice.depthMin, slice.depthMax, copy.ix * request.tile.sizeX, copy.iz * request.tile.sizeZ,
      ]);
      device.queue.writeBuffer(uniformBuffer, drawIndex * 256, values);
      groups.push(device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer, offset: drawIndex * 256, size: 48 } }],
      }));
      drawIndex++;
    }
    bindGroups.push(groups);
  }

  const encoder = device.createCommandEncoder({ label: 'groundcover-periodic-bake-encoder' });
  request.slices.forEach((_slice, sliceIndex) => {
    const pass = encoder.beginRenderPass({
      label: `groundcover-periodic-slice-${sliceIndex}`,
      colorAttachments: [{
        view: color.createView(),
        clearValue: { r: -1, g: 0.5, b: 0.5, a: 0 },
        loadOp: sliceIndex === 0 ? 'clear' : 'load',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depth.createView(),
        depthClearValue: 1,
        depthLoadOp: sliceIndex === 0 ? 'clear' : 'load',
        depthStoreOp: 'store',
      },
    });
    const tileX = (sliceIndex % request.atlasColumns) * request.tileWidth;
    const tileY = Math.floor(sliceIndex / request.atlasColumns) * request.tileHeight;
    pass.setViewport(tileX, tileY, request.tileWidth, request.tileHeight, 0, 1);
    pass.setScissorRect(tileX, tileY, request.tileWidth, request.tileHeight);
    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, 'uint32');
    for (const group of bindGroups[sliceIndex] as GPUBindGroup[]) {
      pass.setBindGroup(0, group);
      pass.drawIndexed(indices.length);
    }
    pass.end();
  });

  const unpaddedBytesPerRow = atlasWidth * 16;
  const bytesPerRow = align(unpaddedBytesPerRow, 256);
  const readback = device.createBuffer({
    label: 'groundcover-periodic-readback',
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
  const errors = [oom, internal, validation].filter((error): error is GPUError => error !== null).map((error) => error.message);
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
    tile: request.tile,
    slices: request.slices,
    pixels,
    adapter: `${info.vendor || 'unknown'} ${info.architecture || ''} ${info.device || ''}`.trim(),
  };
}

window.__groundCoverPeriodicBake = { ready: true, run };
