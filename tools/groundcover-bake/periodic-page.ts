import type { PeriodicBakeResult, PeriodicSlice, PeriodicTile } from './PeriodicProfile';

interface BakeMesh {
  positions: number[] | Float32Array;
  normals: number[] | Float32Array;
  colors?: number[] | Float32Array;
  indices: number[] | Uint32Array;
}

interface PeriodicBakeRequest {
  mesh: BakeMesh;
  tile: PeriodicTile;
  slices: PeriodicSlice[];
  tileWidth: number;
  tileHeight: number;
  atlasColumns: number;
  atlasRows: number;
}

interface PeriodicBakeTransport extends Omit<PeriodicBakeRequest, 'mesh'> {
  positionsUrl: string;
  normalsUrl: string;
  colorsUrl?: string;
  indicesUrl: string;
}

interface StoredBakeSummary extends Omit<PeriodicBakeResult, 'pixels' | 'correspondencePixels' | 'ownerPixels'> {
  pixelLength: number;
  correspondenceLength: number;
  ownerLength: number;
}

declare global {
  interface Window {
    __groundCoverPeriodicBake: {
      ready: boolean;
      run: (request: PeriodicBakeRequest) => Promise<PeriodicBakeResult>;
      runBinary: (transport: PeriodicBakeTransport) => Promise<PeriodicBakeResult>;
      runStored: (transport: PeriodicBakeTransport) => Promise<StoredBakeSummary>;
      readStored: (
        kind: 'pixels' | 'correspondencePixels' | 'ownerPixels',
        offset: number,
        count: number,
      ) => number[];
      clearStored: () => void;
    };
  }
}

const shader = /* wgsl */ `
struct Projection {
  ray_topH: vec4f,
  tileOrigin_size: vec4f,
  depth_copy: vec4f,
  copyIndex_pad: vec4u,
};
@group(0) @binding(0) var<uniform> projection: Projection;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;

struct VertexOut {
  @builtin(position) clip: vec4f,
  @location(0) depth: f32,
  @location(1) normal: vec3f,
  @location(2) color: vec3f,
  @location(3) worldPosition: vec3f,
  @location(4) @interpolate(flat) primitiveId: u32,
};

@vertex fn vertexMain(
  @builtin(vertex_index) corner: u32,
  @builtin(instance_index) primitiveId: u32,
) -> VertexOut {
  let vertex = indices[primitiveId * 3u + corner];
  let base = vertex * 3u;
  let position = vec3f(positions[base], positions[base + 1u], positions[base + 2u]);
  let normal = vec3f(normals[base], normals[base + 1u], normals[base + 2u]);
  let color = vec3f(colors[base], colors[base + 1u], colors[base + 2u]);
  let ray = projection.ray_topH.xyz;
  let p = position + vec3f(projection.depth_copy.z, 0.0, projection.depth_copy.w);
  let t = (p.y - projection.ray_topH.w) / ray.y;
  let rayOriginXZ = p.xz - ray.xz * t;
  let uv = (rayOriginXZ - projection.tileOrigin_size.xy) / projection.tileOrigin_size.zw;
  let depth = (t - projection.depth_copy.x) / (projection.depth_copy.y - projection.depth_copy.x);
  var out: VertexOut;
  out.clip = vec4f(uv * 2.0 - 1.0, depth, 1.0);
  out.depth = depth;
  out.normal = normal;
  out.color = color;
  out.worldPosition = p;
  out.primitiveId = primitiveId;
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

struct FragmentOut {
  @location(0) geometry: vec4f,
  @location(1) correspondence: vec4f,
  @location(2) owner: u32,
};

@fragment fn fragmentMain(
  input: VertexOut,
) -> FragmentOut {
  let incoming = projection.ray_topH.xyz;
  let candidate = normalize(input.normal);
  let facing = select(candidate, -candidate, dot(candidate, incoming) > 0.0);
  let faceCandidate = normalize(cross(dpdx(input.worldPosition), dpdy(input.worldPosition)));
  let faceFacing = select(faceCandidate, -faceCandidate, dot(faceCandidate, incoming) > 0.0);
  let rgb = clamp(input.color, vec3f(0.0), vec3f(1.0));
  let rgb565 =
    (u32(round(rgb.r * 31.0)) << 11u)
    | (u32(round(rgb.g * 63.0)) << 5u)
    | u32(round(rgb.b * 31.0));
  var out: FragmentOut;
  out.geometry = vec4f(input.depth, encodeOct(facing), 1.0);
  // W is populated after readback with the exact safe in-triangle radius.
  out.correspondence = vec4f(encodeOct(faceFacing), f32(rgb565) / 65535.0, 0.0);
  out.owner = input.primitiveId
    | (projection.copyIndex_pad.y << 22u)
    | (projection.copyIndex_pad.z << 27u);
  return out;
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
  if (adapter.limits.maxColorAttachmentBytesPerSample < 36) {
    throw new Error('groundcover owner bake requires 36 color-attachment bytes per sample');
  }
  const device = await adapter.requestDevice({
    label: 'groundcover-periodic-offline-baker',
    requiredLimits: { maxColorAttachmentBytesPerSample: 36 },
  });
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
  const correspondence = device.createTexture({
    label: 'groundcover-periodic-correspondence',
    size: [atlasWidth, atlasHeight],
    format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const owner = device.createTexture({
    label: 'groundcover-periodic-owner',
    size: [atlasWidth, atlasHeight],
    format: 'r32uint',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const depth = device.createTexture({
    label: 'groundcover-periodic-depth-election',
    size: [atlasWidth, atlasHeight],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const positions = new Float32Array(request.mesh.positions);
  const normals = new Float32Array(request.mesh.normals);
  const colors = new Float32Array(
    request.mesh.colors ?? Array.from({ length: request.mesh.positions.length }, () => 1),
  );
  const indices = new Uint32Array(request.mesh.indices);
  if (indices.length / 3 >= 1 << 22) {
    throw new Error('groundcover periodic triangle count exceeds packed owner bits');
  }
  const makeStorage = (label: string, bytes: ArrayBufferView): GPUBuffer => {
    const buffer = device.createBuffer({
      label,
      size: bytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      buffer,
      0,
      bytes.buffer as ArrayBuffer,
      bytes.byteOffset,
      bytes.byteLength,
    );
    return buffer;
  };
  const positionBuffer = makeStorage('groundcover-periodic-positions', positions);
  const normalBuffer = makeStorage('groundcover-periodic-normals', normals);
  const colorBuffer = makeStorage('groundcover-periodic-colors', colors);
  const indexBuffer = device.createBuffer({
    label: 'groundcover-periodic-indices',
    size: indices.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const drawCount = request.slices.reduce((sum, slice) => sum + slice.copies.length, 0);
  const uniformBuffer = device.createBuffer({
    label: 'groundcover-periodic-projections',
    size: drawCount * 256,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indices);

  const module = device.createShaderModule({ label: 'groundcover-periodic-raster-project', code: shader });
  const pipeline = device.createRenderPipeline({
    label: 'groundcover-periodic-raster-project',
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vertexMain',
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      targets: [
        { format: 'rgba32float' },
        { format: 'rgba32float' },
        { format: 'r32uint' },
      ],
    },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const bindGroups: GPUBindGroup[][] = [];
  let drawIndex = 0;
  for (const slice of request.slices) {
    const groups: GPUBindGroup[] = [];
    for (let copyIndex = 0; copyIndex < slice.copies.length; copyIndex++) {
      const copy = slice.copies[copyIndex]!;
      if (copyIndex >= 1024) throw new Error('groundcover periodic copy index exceeds packed owner bits');
      if (copy.ix < -16 || copy.ix > 15 || copy.iz < -16 || copy.iz > 15) {
        throw new Error('groundcover periodic copy offset exceeds packed signed 5-bit range');
      }
      const uniformBytes = new ArrayBuffer(64);
      const values = new Float32Array(uniformBytes);
      values.set([
        slice.direction.x, slice.direction.y, slice.direction.z, request.tile.topH,
        request.tile.originX, request.tile.originZ, request.tile.sizeX, request.tile.sizeZ,
        slice.depthMin, slice.depthMax, copy.ix * request.tile.sizeX, copy.iz * request.tile.sizeZ,
      ]);
      new Uint32Array(uniformBytes)[12] = copyIndex;
      new Uint32Array(uniformBytes)[13] = copy.ix + 16;
      new Uint32Array(uniformBytes)[14] = copy.iz + 16;
      device.queue.writeBuffer(uniformBuffer, drawIndex * 256, uniformBytes);
      groups.push(device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer, offset: drawIndex * 256, size: 64 } },
          { binding: 1, resource: { buffer: positionBuffer } },
          { binding: 2, resource: { buffer: normalBuffer } },
          { binding: 3, resource: { buffer: colorBuffer } },
          { binding: 4, resource: { buffer: indexBuffer } },
        ],
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
      }, {
        view: correspondence.createView(),
        clearValue: { r: 0.5, g: 0.5, b: 0, a: 0 },
        loadOp: sliceIndex === 0 ? 'clear' : 'load',
        storeOp: 'store',
      }, {
        view: owner.createView(),
        clearValue: { r: 0xffffffff, g: 0, b: 0, a: 0 },
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
    for (const group of bindGroups[sliceIndex] as GPUBindGroup[]) {
      pass.setBindGroup(0, group);
      pass.draw(3, indices.length / 3);
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
  const correspondenceReadback = device.createBuffer({
    label: 'groundcover-periodic-correspondence-readback',
    size: bytesPerRow * atlasHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const ownerBytesPerRow = align(atlasWidth * 4, 256);
  const ownerReadback = device.createBuffer({
    label: 'groundcover-periodic-owner-readback',
    size: ownerBytesPerRow * atlasHeight,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyTextureToBuffer(
    { texture: color },
    { buffer: readback, bytesPerRow, rowsPerImage: atlasHeight },
    [atlasWidth, atlasHeight],
  );
  encoder.copyTextureToBuffer(
    { texture: correspondence },
    { buffer: correspondenceReadback, bytesPerRow, rowsPerImage: atlasHeight },
    [atlasWidth, atlasHeight],
  );
  encoder.copyTextureToBuffer(
    { texture: owner },
    { buffer: ownerReadback, bytesPerRow: ownerBytesPerRow, rowsPerImage: atlasHeight },
    [atlasWidth, atlasHeight],
  );
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await Promise.all([
    readback.mapAsync(GPUMapMode.READ),
    correspondenceReadback.mapAsync(GPUMapMode.READ),
    ownerReadback.mapAsync(GPUMapMode.READ),
  ]);
  const mapped = new Uint8Array(readback.getMappedRange());
  const tight = new Uint8Array(unpaddedBytesPerRow * atlasHeight);
  for (let y = 0; y < atlasHeight; y++) {
    tight.set(mapped.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow), y * unpaddedBytesPerRow);
  }
  const pixels = new Float32Array(tight.buffer);
  const correspondenceMapped = new Uint8Array(correspondenceReadback.getMappedRange());
  const correspondenceTight = new Uint8Array(unpaddedBytesPerRow * atlasHeight);
  const ownerMapped = new Uint8Array(ownerReadback.getMappedRange());
  const ownerTight = new Uint8Array(atlasWidth * atlasHeight * 4);
  for (let y = 0; y < atlasHeight; y++) {
    correspondenceTight.set(
      correspondenceMapped.subarray(y * bytesPerRow, y * bytesPerRow + unpaddedBytesPerRow),
      y * unpaddedBytesPerRow,
    );
    ownerTight.set(
      ownerMapped.subarray(y * ownerBytesPerRow, y * ownerBytesPerRow + atlasWidth * 4),
      y * atlasWidth * 4,
    );
  }
  const correspondencePixels = new Float32Array(correspondenceTight.buffer);
  const owners = new Uint32Array(ownerTight.buffer);
  const point = (index: number, copyX: number, copyZ: number): [number, number, number] => [
    request.mesh.positions[index * 3]! + copyX,
    request.mesh.positions[index * 3 + 1]!,
    request.mesh.positions[index * 3 + 2]! + copyZ,
  ];
  const pointLineDistance = (
    p: readonly number[],
    a: readonly number[],
    b: readonly number[],
  ): number => {
    const ex = b[0]! - a[0]!;
    const ey = b[1]! - a[1]!;
    const ez = b[2]! - a[2]!;
    const px = p[0]! - a[0]!;
    const py = p[1]! - a[1]!;
    const pz = p[2]! - a[2]!;
    const cx = py * ez - pz * ey;
    const cy = pz * ex - px * ez;
    const cz = px * ey - py * ex;
    return Math.hypot(cx, cy, cz) / Math.max(1e-12, Math.hypot(ex, ey, ez));
  };
  for (let y = 0; y < atlasHeight; y++) {
    for (let x = 0; x < atlasWidth; x++) {
      const pixel = y * atlasWidth + x;
      const base = pixel * 4;
      if (pixels[base + 3]! <= 0.5) continue;
      const sliceX = Math.floor(x / request.tileWidth);
      const sliceY = Math.floor(y / request.tileHeight);
      const sliceIndex = sliceY * request.atlasColumns + sliceX;
      const slice = request.slices[sliceIndex];
      if (!slice) continue;
      const packedOwner = owners[pixel]!;
      const triangleId = packedOwner & 0x3fffff;
      const copyIx = ((packedOwner >>> 22) & 0x1f) - 16;
      const copyIz = ((packedOwner >>> 27) & 0x1f) - 16;
      if (triangleId * 3 + 2 >= request.mesh.indices.length) continue;
      const localX = x - sliceX * request.tileWidth;
      const localY = y - sliceY * request.tileHeight;
      const originX = request.tile.originX + ((localX + 0.5) / request.tileWidth) * request.tile.sizeX;
      const originZ = request.tile.originZ + (1 - (localY + 0.5) / request.tileHeight) * request.tile.sizeZ;
      const t = slice.depthMin + pixels[base]! * (slice.depthMax - slice.depthMin);
      const hit: [number, number, number] = [
        originX + slice.direction.x * t,
        request.tile.topH + slice.direction.y * t,
        originZ + slice.direction.z * t,
      ];
      const copyX = copyIx * request.tile.sizeX;
      const copyZ = copyIz * request.tile.sizeZ;
      const ia = request.mesh.indices[triangleId * 3]!;
      const ib = request.mesh.indices[triangleId * 3 + 1]!;
      const ic = request.mesh.indices[triangleId * 3 + 2]!;
      const a = point(ia, copyX, copyZ);
      const b = point(ib, copyX, copyZ);
      const c = point(ic, copyX, copyZ);
      const safeRadius = Math.min(
        pointLineDistance(hit, a, b),
        pointLineDistance(hit, b, c),
        pointLineDistance(hit, c, a),
      ) * 0.96;
      correspondencePixels[base + 3] = Number.isFinite(safeRadius) ? Math.max(0, safeRadius) : 0;
    }
  }
  readback.unmap();
  correspondenceReadback.unmap();
  ownerReadback.unmap();

  const oom = await device.popErrorScope();
  const internal = await device.popErrorScope();
  const validation = await device.popErrorScope();
  const errors = [oom, internal, validation].filter((error): error is GPUError => error !== null).map((error) => error.message);
  errors.push(...uncaptured);
  color.destroy();
  correspondence.destroy();
  owner.destroy();
  depth.destroy();
  positionBuffer.destroy();
  normalBuffer.destroy();
  colorBuffer.destroy();
  indexBuffer.destroy();
  uniformBuffer.destroy();
  readback.destroy();
  correspondenceReadback.destroy();
  ownerReadback.destroy();
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
    correspondencePixels,
    ownerPixels: owners,
    adapter: `${info.vendor || 'unknown'} ${info.architecture || ''} ${info.device || ''}`.trim(),
  };
}

async function fetchTyped(url: string, kind: 'float32' | 'uint32'): Promise<Float32Array | Uint32Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`groundcover periodic bake could not fetch ${url}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if ((bytes.byteLength & 3) !== 0) throw new Error(`groundcover periodic bake ${url} is not 32-bit aligned`);
  return kind === 'float32' ? new Float32Array(bytes) : new Uint32Array(bytes);
}

async function runBinary(transport: PeriodicBakeTransport): Promise<PeriodicBakeResult> {
  const [positions, normals, colors, indices] = await Promise.all([
    fetchTyped(transport.positionsUrl, 'float32') as Promise<Float32Array>,
    fetchTyped(transport.normalsUrl, 'float32') as Promise<Float32Array>,
    transport.colorsUrl
      ? fetchTyped(transport.colorsUrl, 'float32') as Promise<Float32Array>
      : Promise.resolve(undefined),
    fetchTyped(transport.indicesUrl, 'uint32') as Promise<Uint32Array>,
  ]);
  if (
    positions.length === 0
    || positions.length !== normals.length
    || (colors !== undefined && colors.length !== positions.length)
    || indices.length % 3 !== 0
  ) {
    throw new Error('groundcover periodic bake binary mesh attributes are incompatible');
  }
  return run({
    mesh: { positions, normals, colors, indices },
    tile: transport.tile,
    slices: transport.slices,
    tileWidth: transport.tileWidth,
    tileHeight: transport.tileHeight,
    atlasColumns: transport.atlasColumns,
    atlasRows: transport.atlasRows,
  });
}

let storedResult: PeriodicBakeResult | null = null;

async function runStored(transport: PeriodicBakeTransport): Promise<StoredBakeSummary> {
  storedResult = await runBinary(transport);
  const { pixels, correspondencePixels, ownerPixels, ...metadata } = storedResult;
  return {
    ...metadata,
    pixelLength: pixels.length,
    correspondenceLength: correspondencePixels?.length ?? 0,
    ownerLength: ownerPixels?.length ?? 0,
  };
}

function readStored(
  kind: 'pixels' | 'correspondencePixels' | 'ownerPixels',
  offset: number,
  count: number,
): number[] {
  if (!storedResult) throw new Error('no stored periodic bake');
  const source = storedResult[kind];
  if (!source) return [];
  const end = Math.min(source.length, offset + count);
  const output = new Array<number>(Math.max(0, end - offset));
  for (let index = offset; index < end; index++) output[index - offset] = source[index]!;
  return output;
}

function clearStored(): void {
  storedResult = null;
}

window.__groundCoverPeriodicBake = {
  ready: true,
  run,
  runBinary,
  runStored,
  readStored,
  clearStored,
};
