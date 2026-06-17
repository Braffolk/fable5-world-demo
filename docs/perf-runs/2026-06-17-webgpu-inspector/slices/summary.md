# Capture summary — frame 1244

- tool: webgpu_inspector 1.4.3  schema 1.1
- canvas: 1101x1450 bgra8unorm (1.60 Mpx)
- adapter: metal-3

## statistics
- apiCalls: 747
- draw: 19
- drawIndirect: 1
- dispatch: 56
- setVertexBuffer: 34
- setIndexBuffer: 1
- setBindGroup: 95
- uniformBuffers: 144
- storageBuffers: 280
- textures: 54
- samplers: 43
- setPipeline: 75
- vertexShaders: 19
- fragmentShaders: 19
- computeShaders: 56
- computePasses: 56
- renderPasses: 18
- colorAttachments: 19
- depthStencilAttachments: 3
- copyCommands: 2
- writeBuffer: 74
- bufferBytesWritten: 2408
- writeTexture: 0
- totalBytesWritten: 2408
- totalInstances: 18
- totalVertices: 6003
- totalTriangles: 2001
- totalLines: 0
- totalPoints: 0

## object counts
- TextureView: 342
- Buffer: 275
- RenderBundle: 248
- ShaderModule: 157
- BindGroup: 101
- ComputePipeline: 98
- Texture: 57
- RenderPipeline: 42
- PipelineLayout: 41
- BindGroupLayout: 34
- Sampler: 6
- Adapter: 1
- Device: 1
- CanvasContext: 1

## validationErrors
- GPUBuffer was garbage collected without being explicitly destroyed. These objects should explicitly destroyed to avoid GPU memory leaks.

## texture VRAM total: 858.8 MB across 57 textures
## buffer bytes total: 2212.2 MB across 275 buffers