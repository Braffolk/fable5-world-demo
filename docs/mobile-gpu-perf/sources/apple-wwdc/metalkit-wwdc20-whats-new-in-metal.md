<!-- source: https://metalkit.org/wwdc20-whats-new-in-metal/ (via Wayback Machine — live site returned connection error) -->

#### WWDC20 – What’s new in Metal and the Apple GPU

- July 3, 2020

WWDC20 has been an event of major changes. All operating systems had significant design changes. The new macOS named Big Sur is now version 11. The biggest announcement was the 2-yr plan for migrating from the Intel system on a chip (SoC) to the newly branded Apple Silicon SoC. The new SoC will replace the Intel integrated GPU with Apple’s own GPU in new upcoming Macs. There is no word yet on what will happen to the AMD discrete GPUs or whether eGPUs will continue to be supported in future or not. Bootcamp might still work with Apple Silicon based Macs since there is an ARM based distribution of Windows used by Microsoft Surface devices.

The migration from Intel to Apple Silicon will affect the operating system and the Apple frameworks as well, Metal included. Apple states that by leveraging a unified memory architecture (CPU+GPU), macOS apps will see great performance benefits from frameworks such as Metal which will be fine-tuned for Apple Silicon. Let’s start with looking at what this migration to Apple Silicon means for Metal.

#### Getting your apps ready for Apple Silicon Macs

If you remember, back in 2006, Apple developed a dynamic binary translator named Rosetta which served a similar purpose: to migrate from PowerPC CPUs to Intel CPUs. It was used over a period of 3 years, in OSX versions 10.4 – 10.6 (Tiger, Leopard, and Snow Leopard). This year, Apple released the second version of Rosetta to serve the transition from Intel to Apple Silicon. Rosetta will run Metal code directly on the Apple GPU.

Here is a high level description of the migration process:

Looking at the first stage, on Intel based Macs the apps will still run natively. On Apple Silicon based Macs the old apps will run via Rosetta without any modifications. To alleviate some of the tradeoffs Rosetta made to assure compatibility, you are strongly encouraged to start writing your app using the new macOS SDK. Finally, you should take advantage of the latest performance tools to make your app run as fast as possible on Apple Silicon.

The new GPUs on the Apple Silicon SoC features the efficient Tile Based Deferred Rendering (TBDR) architecture that you have been familiar with on iOS devices and will have support from both the Metal GPU Family Apple and Metal GPU Family Mac 2. In contrast, the GPUs that exist on the Intel-based Macs (AMD, Nvidia and Intel) feature an Immediate Mode Rendering architecture and will only have support from the Metal GPU Family Mac 2.

Comparing the two GPU architectures, TBDR has the following advantages:
- It drastically saves on memory bandwidth because of the unified memory architecture.
- Blending happens in-register facilitated by tile processing.
- Color, depth and stencil buffers don’t need to be re-fetched.

A best practice for Metal feature detection is to avoid querying devices by name and setting features based on that:

if deviceName.contains("AMD") 
{
    self.appleGPUFeatures = false
    self.simdgroupSize = 64
    self.isLowPower = false
}

```

Instead, you should query the device itself for its feature set:

self.appleGPUFeatures = metalDevice.supportsFamily(.apple5)
self.simdgroupSize = computePipeline.threadExecutionWidth
self.isLowPower = metalDevice.isLowPower

```

Note: The Apple Silicon GPU will return false when queried for isLowPower because its performance is at least as good as the discrete GPU performance in current Macs, while offering a much better performance/watt ratio.

A new feature is Position Invariance. Assume you have two shaders, one per rendering pass, each working on the same vertex position. This would not only be redundant work but it might also output unexpected values if not carefully used. In order to avoid this, you can use the preserveInvarianceproperty to signal the Metal library that a certain position will be invariant:

// API code
let options = MTLCompileOptions()
options.preserveInvariance = true
library = try device.makeLibrary(source: sourceString,
				options: options)
...				
// Shader code
struct VertexOut {
    float4 pos [[position, invariant]];
}

```

Another aspect to consider is Threadgroup Memory Synchronization. Normally, threads are organized in threadgroups that each have their own memory block shared between the threads inside the threadgroup. All threads inside a threadgroup are synchronized. You can synchronize the threads further on by using SIMDgroups.

Of course, if a SIMDgroup only has one threadgroup, no further synchronization is necessary. The size of a SIMDgroup is GPU-specific but many GPUs have a group size of 32. You should always check for the SIMDgroup size either in the API by querying the compute pipeline’s threadExecutionWidthproperty, or by querying [[threads_per_simdgroup]] inside the compute kernel.

Another aspect to consider is working with depth and stencil textures. Because of the shared system memory, race conditions might occur when a texture that is used as an attachment is also sampled in the same pass. You should snapshot these textures before sampling them:

#### The Apple GPU’s TBDR architecture

The Apple GPU shares the system memory but also has its own tile memory. However, there is no video memory as you would see in a discrete GPU:

Below is a high level representation of the Apple GPU rendering pipeline. It consists of two major parts delimited in red: Tiling and Rendering.

In the Tiling phase, the viewport is split into tiles, all vertices are shaded and the transformed primitives are grouped by tiles. Since resources are limited on the GPU, the output from the Tiling phase is stored in the Tiled Vertex Buffer which is shared with the Rasterizer for faster access.

In the Rendering phase, each tile is processed as follows: if the content exists, it is loaded for the current tile (clear, otherwise), then it is rasterized and visibility is determined, then all visible pixels are shaded, and finally the result either stored if needed later (don't care, otherwise).

The on-chip Depth Buffer facilitates the Hidden Surface Removal (HSR) stage which is a key factor in minimizing overdraw by keeping the frontmost layer for each pixel. The HSR output undergoes Fragment Processing next. The correct order for the visibility state is: draw opaque surfaces first, then those with alpha test/depth feedback such as foliage, and translucent surfaces last. For opaque surfaces the HSR will determine which pixel will be shaded, however, for translucent surfaces all the overlapping pixels need to be blended.

The TBDR architecture allows for Programmable Blending by providing access to the pixel directly from the tile memory. To achieve that you need to use MTLStorageModeMemoryless for memoryless render targets, thus avoiding system memory data traffic.

TBDR is also designed for very efficient MSAA. The GPU is tracking the primitive edges and it will blend the color per pixel is the current pixel doesn’t contain an edge, or otherwise it will blend per sample if the pixel contains edges.

The final two parts of the Rendering stage are the programmable Tile Compute block and the Local Image Block. Imageblocks are 2D data structures with depth information stored and they are accessible by both fragment and compute shaders. The main advantage for imageblocks is the ability to load/store a texture with one operation as opposed to loading/storing the texture pixel by pixel into the threadgroup memory. In order to do that, you will need to dispatch tile shaders (which are mid-render compute kernel) and interleave them with draw calls.

#### Optimizing performance for Apple Silicon Macs

Best practices for current/future Macs:
- Schedule work efficiently.
- run concurrently when there are no dependencies.
- avoid serial dependency chains and false dependencies.
- use separate and untracked resources.
- use Metal fences and events.
- reorder passes to start work as soon as possible.
- Minimize system bandwidth.
- minimize render passes in order to reduce the number of loads/stores on each pass.
- use parallel render command encoders instead of using parallel encoding with command buffers.
- avoid mismatched attachment configurations and instead preserve passes when load/store actions change.
- avoid attachment ping-ponging and instead use multiple render targets.
- avoid separate clear actions and instead defer clear until the render pass that needs it.
- avoid separate MSAA resolves and don’t store samples unless you need them later.
- Minimize overdraw
- maximize HSR by first drawing opaque (eg. walls), then feedback (eg. foliage) and last translucent surfaces (eg. smoke).
- use [[early_fragment_tests]] to maximize the rejection of discarded fragments.
- use write masking to restrict certain channels (eg. red) if they are not updates by the rendering pass.
- avoid partial attachment updates and instead write all render pass attachments on each fragment.
- current Macs use a depth pre-pass approach to render the scene geometry twice: depth-only for visibility and depth-test equal for shading, however, on Apple GPUs this approach is not required anymore because HSR achieves the same goal without additional cost.

Best practices for Apple Silicon Macs:
- Optimized deferred shading.
- multi-pass deferred shading is done in an optimized on-chip one pass for the TBDR architecture.
- the TBDR pipeline is more complex and features programmable blending.
- Mixing render and compute.
- render and compute encoders can be run at the same time by dispatching kernel threadgroups in the render encoder.
- tile shaders can read/write to imageblocks, as well as to threadgroup/device memory.
- the optimized TBDR does everything in one pass on tile memory, thus saving on the memory bandwidth traffic that multi-pass deferred renderers have.
- use memory-less attachments to save even more on the memory footprint.
- Repurposing tile memory.
- when a deferred rendering pipeline attachment is not needed anymore, the tile memory can be reused for other purposes, such as transitioning to a multi-layer alpha blending layout.
- Optimize for the Apple GPU shader core.
- the shader cores have a scalar ALU with vectorized load/store; they feature a constant execution and prefetch; ALUs are present in both 16-bit and 32-bit variants to ensure that the desired precision is performed at efficient cost.
- maximizing constant address space effectiveness (see the memory subsystem diagram below) will bring great performance benefits. while device memory is read-write and theoretically with no size limit, the constant memory is read-only and limited in size but highly optimized for reusable constant data between threads.
- choosing the correct address space has a great impact on performance: if the data size is unknown or if it is not reused much, the device address space should be used, otherwise the constant address space should be used.
- in constant address space buffers are likely preloaded, while load offset and size of arrays are known at compile time. to help the compiler, you should pass single struct argument by reference. you should also pass bounded arrays in a struct, rather than via pointers.
- using the correct ALU data types also has a great impact on performance. you should use half and short when possible because they will use the 16-bit ALUs and you will then have better occupancy and faster arithmetic. conversions from 32-bit to 16-bit types are typically free. you should also use ushort for local and global thread IDs when possible. similarly, use half when doing half-precision operations.
- optimizing memory accesses can be done by avoiding dynamically indexed non-constant stack arrays. when iterating over a loop, use signed offsets instead of uint because signed indexes facilitate vectorized loads for faster memory access. similarly, a batch vector loads/store is faster than individual scalar loads/stores.

#### Debugging and profiling with new Metal tools

This year, Apple is introducing the Enhanced Command Buffer Errors. In order to enhance those cryptic error messages that used to just say IOAF code 5, all you have to do is configure the options on your command buffer like this:

let desc = MTLCommandBufferDescriptor()
desc.errorOptions = .encoderExecutionStatus
let commandBuffer = commandQueue.makeCommandBuffer(descriptor: desc)

```

After that, you can parse the error state for details, like this:

if let error = commandBuffer.error as NSError? {
    if let infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey]
        		as? [MTLCommandBufferEncoderInfo] {
        for info in infos {
            print(info.label + info.debugSignposts.joined())
            if info.errorState == .faulted {
                print(info.label + " faulted!")
            }
        }
    }
}

```

The error state property MTLCommandEncoderErrorState is an enum with these possible values: completed, pending, faulted, affected, and unknown. Faulted is the error state that tells us the particular encoder was directly responsible for the command buffer fault.

Another addition this year is the Shader Validation tool. It is similar to the existing Metal API Validation layer, except Shader Validation is for the GPU. It can detect the following errors:
- out of bounds global (device and constant) memory access.
- out of bounds threadgroup memory access.
- attempting to use texturing functions on null texture objects.

In contrast, the Enhanced Command Buffer Errors can detect:
- infinite loops.
- resource residency errors.

To enable Shader Validation, in any Xcode project open Edit Scheme window and under the Diagnostics tab check the Shader Validation box. After that, all that’s left is for you to enable the Metal Diagnostics Breakpoint which tells Xcode to stop the execution of the program when a shader validation error occurs and show the recorded GPU/CPU backtrace for that error. Clicking the arrow to the right of Shader Validation will add the breakpoint. Once the break point has been added, you can find it in the debug navigator on the breakpoints tab. You can view the settings of this breakpoint by clicking on the blue arrow. First, make sure the breakpoint is enabled and then set the type to System Frameworks and enter Metal Diagnostics into the category field.

You can now use this feature in Xcode. Once you run your project, if you enabled the breakpoint, the execution will stop at the line causing issues. You will see some lines of feedback in the console but more importantly, you will see an annotation that you can click to expand and see what that particular line of code is causing.

You can also use Shader Validation in automation scripts by setting the following environment variables: MTL_DEBUG_LAYER=1 for API validation and MTL_SHADER_VALIDATION=1 for Shader Validation. Now you can send errors to the logs which you can parse later on. Here is the code to generate log output:

commandBuffer.addCompletedHandler { (commandBuffer) in
    for log in commandBuffer.logs {
        let encoderLabel = log.encoderLabel ?? "Unknown Label"
        print("Faulting encoder \"\(encoderLabel)\"")
        guard let debugLocation = log.debugLocation,
              let functionName = debugLocation.functionName
        else {
            return
        }
        print("Faulting function \(functionName):
        	\(debugLocation.line):\(debugLocation.column)")
    }
}

```

To parse the log, you would run this command in a terminal:

log stream --predicate "subsystem = 'com.apple.Metal' 
					and category = 'GPUDebug'"

```

Unlike the Enhanced Command Buffer Errors, the Shader Validation layer has a large impact on memory usage and performance so it is recommended to be only used during debugging stages.

This year, the Metal Debugger has a great new addition, the Summary View. As soon as you run your app and click Capture GPU Frame, the new Summary View shows center screen:

In the Overview section you can see information about command buffers, encoders and calls. There is also a button Show Dependencies that launches the Dependency Viewer tool. In the Performance section you get information about the frame time and about vertices. There is also a button Show Counters that launches the GPU Counters tool. In the Memory section you can see information about the textures, buffers and other objects using memory. There is also a button Show Memory that launches the Memory Viewer tool. Finally, in the Insights section you get suggestions on how to improve performance, bandwidth and memory usage. Each insight is accompanied by a short description of the issue, hints for fixing it and links to documentation.

The GPU Counters tool has been enhanced with more metrics to help you quickly spot where the GPU is spending time in your app. This year, GPU Counters is also available to non-Apple GPUs:

The Metal System Trace has been improved with new features as well. When capturing data on a device with A11 or newer GPUs, the MST now tracks start and end times for individual shaders. There is a new Shader Timeline tool for iOS that shows which shaders are running at any given time during the the MST recording. Also new this year for non-Apple GPUs is the Performance Limiterstool that lets you enable tracks for watching the values of the limiters over time. You can enable both these tools in Instruments under Recording Options:

The high level architecture of Apple GPU’s memory hierarchy looks like this:

Rendering a frame is a complex process that involves processing multiple passes running on multiple GPU cores, each processing multiple tasks on different hardware units and which all have different throughput metrics. That’s why there are over 150 GPU Counters available, all ready to help you see if your GPU is overloaded or underutilized and spot the exact areas that need attention.

Going a level deeper in the GPU architecture, we see how each shader code has multiple SIMD units with dedicated tile memory. Multiple threadgroups or tiles share the register memory. Each SIMD unit has 32 threads and each of these threads executes the same instruction:

A few ways to improve performance with an ALU Limiter is to replace complex calculations with approximations and look-up tables for textures, to replace floats with halfs, and to use the -ffast-math shader compilation flag.

Looking at the GPU memory hierarchy again, we see the Texture Unit which is responsible for reading Metal textures which are backed by the System/Device Memory and written by the Pixel Backend. The Texture Unit has its own dedicated L1 cache memory, support for multiple filtering modes and compression modes. You can improve performance with the Texture Sample Limiter by using mipmaps when minification occurs, by using bilinear instead of trilinear filtering, by lowering of the anisotropic sample count, by using smaller pixel sizes and by leveraging texture compression.

The Pixel Backend unit is responsible for writing texture data and it is optimized for coherent writes so it is better to avoid divergent writes. Performance can be improved using the Texture Write Limiter by considering smaller pixel sizes, by reducing the MSAA sample count and the number of small triangles, optimizing for coherent writes by improving spatial and temporal locality.

Another Performance Limiter is the Tile Memory which is a high-performance memory dedicated to accessing threadgroup and imageblock data, as well as to accessing color attachments or using programmable blending. The Threadgroup/Imageblock Limiter can be used to improve performance by reducing the number of threadgroup atomics, by aligning threadgroup memory allocations and accesses to 16 bytes, and by reordering the memory access patterns.

Buffer Read and Write is another Performance Limiter. Even though both Metal buffers and textures are backed by device Memory, the difference between them is that buffers are accessed by the Shader Cores only. They also have dedicated L1 cache memory and support different address spaces: Device (read-write, not cached) and Constant (read-only, cached and pre-fetched). The Buffer Read/Write Limiter can be used to improve performance by packing data more tightly, by vectorizing loads/stores, by avoiding device atomics, by avoiding register spills and by using textures to balance the workload.

GPU Last Level Cache is shared by all GPU Cores and is optimized for spatial and temporal locality. Performance can be improved with the GPU Last Level Cache Limiter by reducing size of working sets and optimizing texture/buffer limiters first if they show a high value, by refactoring code to use threadgroup atomics is the shaders use device atomics, and by improving spatial/temporal locality for memory accesses.

The Fragment Input Interpolator in Shader Cores has a fixed function, 32-bit precision and are responsible with interpolating the fragment inputs during the rendering stage. There is not much to do with this limiter except reduce some of the attributes passed from the vertex stage if the limiter shows high value.

Device Memory is backed by System Memory and cached by GPU Last Level Cache. It stores texture/buffer data and the tiled vertex buffer. The Memory Bandwidth GPU Counter measures the memory transfers between the GPU and System Memory. The GPU accesses System Memory when buffers/textures are accessed. Performance can be optimized by reducing size of working sets and optimizing texture/buffer limiters first if they show a high value, by leveraging texture compression, by only loading data needed in current pass and storing only data needed in future passes.

GPU hides latency by switching between available threads. GPU creates new threads when hardware resources are available and there are commands queued to run. The Occupancy GPU Counter measures the ratio of thread capacity used by the GPU in terms of compute, vertex and fragment occupancies. Low occupancy is fine if there is little work to do. However, if there is enough work to keep the GPU busy, try overlapping work that can run in parallel, such as independent compute and rendering pipelines.

Finally, the HSR GPU counters can be used to measure number of pixels rasterized, number of fragment shader invocations, number of pixels stored, pre-Z test fails. Overdraw is the ratio between fragment shader invocations and pixels stored. Try minimizing full-screen passes and blending. You can also use HSR efficiently by drawing meshes sorted by visibility state, by avoiding interleaving opaque and non-opaque meshes, as well as opaque meshes with different color attachment write masks.

#### New features of the Metal API

The traditional shader compilation model takes your .metal source code, compiles it into an .air(Apple Intermediate Representation) format, which is then either converted into an .a static library or an executable .metallib file (see diagram below). This model works fine for simple apps, however, for complex games, developers want to save on compilation time, share common subroutines, ship their games with GPU binaries and libraries as well as share them with other developers.

This year, the new MTLDynamicLibrary enables developers to dynamically link, load and share GPU binaries and library code without duplicate compilation. The dynamic libraries are callable and reusable between multiple compute pipelines. They cannot be used to create Metal functions, however, dynamic libraries can import existing Metal function from standard libraries.

The Metal developer toolset is expanded to include additional tools. The complete toolset looks like this now: metal, metal-libtool, metal-lld, metal-ar, metal-nm, metal-lipo, metal-readobj, metal-as, metal-size, metal-objdump. For example, you can use metal-nm to inspect symbols and metal-lipo to work with harvested slices from the library.

Also new this year are the Binary Archives which pass the control over shader pipeline caching entirely to the developer. This will help reusing pipeline caches between compatible devices, thus drastically improving the shader pipeline creation times. This tremendously improves the times on first-launch and cold-boot scenarios.

Finally, there is a new Developer Tool for Windows added to the Metal toolchain now.

#### Function Pointers

As you know, Metal restricts the use of pointers to only function arguments that are qualified with the Metal device, constant, threadgroup or threadgroup_imageblock address space attribute. This year, however, Function Pointers are also introduced this year for a wide range of use cases in Metal, possibly the best fit being raytracing where you could use function pointers for custom intersection functions.

Starting with Metal 2.3, the new attribute syntax is supported for the following function qualifiers:
- [[vertex]]
- [[fragment]]
- [[kernel]]
- [[intersection(..)]]
- [[visible]]

You can apply this syntax to function definitions and more importantly, you can now manipulate the functions from the Metal API. To check for this feature availability, query your device object:

device.supportsFunctionPointers

```

Visible Functions is a new addition to the Metal API this year. They allow you to add to your pipeline functions other than vertex, fragment or kernel functions, which could be in other .metalfiles or even in other Metal libraries. You need to first declare the function as visible on the GPU:

[[visible]]
Lighting Spot(...)
{
    //...
}

```

Then, on the CPU, use the new MTLLinkedFunctions class to add the function to the pipeline so you can refer to it later with a function pointer:

let linkedFunctions = MTLLinkedFunctions()
let spot = library.makeFunction(name: "Spot")
// assuming you now have six new visible functions to add to pipeline
linkedFunctions.functions = [area, spot, sphere, hair, glass, skin]
var descriptor = MTLComputePipelineDescriptor()
descriptor.linkedFunctions = linkedFunctions
let pipeline = try device.makeComputePipelineState(descriptor: descriptor,
                                                   options: [],
                                                   reflection: nil)

```

This model is called Single Compilation pipeline because there is one single object that represents the pipeline and all its visible functions:

However, sometimes you may need to have your function objects outside of the pipeline object, so you can call them instead of copying them into the pipeline. This is called Separate Compilationpipeline and it is useful for cases where multiple pipelines need to call the same function objects:

In order to compile the functions to binary functions, you need to use the new MTLFunctionDescriptor class:

let functionDescriptor = MTLFunctionDescriptor()
functionDescriptor.name = "Spot"
functionDescriptor.options = MTLFunctionOptions.compileToBinary
let spotBinaryFunc = try library.makeFunction(descriptor: functionDescriptor)
let linkedFunctions = MTLLinkedFunctions()
linkedFunctions.functions = [area, sphere, hair, glass, skin]
linkedFunctions.binaryFunctions = [spotBinaryFunc]
var descriptor = MTLComputePipelineDescriptor()
descriptor.linkedFunctions = linkedFunctions
let pipeline = try device.makeComputePipelineState(descriptor: descriptor,
                                                   options: [],
                                                   reflection: nil)

```

In this example, you tell the pipeline that your “spot” lighting function is going to be compiled to binary so we can call it later, rather than requesting for copying and specialization like for the other five functions. Contrasting the two compilation methods, the single one has the best performance while the separate one produces a smaller pipeline object and in a much shorter creation time.

A third option is the Incremental Compilation pipeline for cases where new binary functions need to be added later:

computeDescriptor.supportAddingBinaryFunctions = true
let functionDescriptor = MTLFunctionDescriptor()
functionDescriptor.name = "Water"
functionDescriptor.options = MTLFunctionOptions.compileToBinary
let water = try library.makeFunction(descriptor: functionDescriptor)
let newPipeline =
   try pipeline.makeComputePipelineStateWithAdditionalBinaryFunctions(
   		functions: [water])

```

Also new this year are the Visible Function Tables which are used to pass function pointer to the visible functions. On the GPU you set the tables like this:

using LightFunction = void(float, int);
kernel void Spot(uint tid [[thread_position_in_grid]],
 	         device float* buf [[buffer(0)]],
                 visible_function_table<LightFunction> table [[buffer(1)]])
{
    uint tsize = table.size();
    table[tid % tsize](buf[tid], tid);
}

```

On the CPU you would set the table up like this:

let descriptor = MTLVisibleFunctionTableDescriptor()
descriptor.functionCount = 3
let lightingFunctionTable = pipeline.makeVisibleFunctionTable(
					descriptor: descriptor)
let functionHandle = pipeline.functionHandle(function: spot)
lightingFunctionTable.setFunction(functionHandle, index: 0)
computeCommandEncoder.setVisibleFunctionTable(lightingFunctionTable, 
						bufferIndex: 1)
argumentEncoder.setVisibleFunctionTable(lightingFunctionTable, index: 1)

```

Also new this year are the Function Groups which are meant to group similar functions:

On the GPU you would modify the function like this:

float3 shade(...)
{
    LightingFunction *lightingFunction = lightingFunctions[light.index];
    [[function_groups("lighting")]] Lighting lighting = lightingFunction(
    				light, triangleIntersection);
    MaterialFunction *materialFunction = materialFunctions[material.index];
    [[function_groups("material")]] float3 result = materialFunction(
    				material, lighting, triangleIntersection);
    return result;
}

```

On the CPU you would configure function groups like this:

let linkedFunctions = MTLLinkedFunctions()
linkedFunctions.functions = [area, spot, sphere, hair, glass, skin]
linkedFunctions.groups = ["lighting" : [area, spot, sphere ],
                          "material" : [hair, glass, skin ] ]
computeDescriptor.linkedFunctions = linkedFunctions

```

When using function pointers recursively, you need to pay attention to thread divergence considerations and to the maximum call stack depth which can be configured on the pipeline descriptor like this:

computeDescriptor.maxCallStackDepth = 3

```

#### New API for Ray Tracing

The new Ray Tracing API allows for intersection testing directly in compute kernels. With the old API, the raytracing consisted of three separate compute kernels: ray generating, intersection finding and shading. It also required that rays and intersections are being passed through memory. With the new API, you can now compute intersections inside the compute kernel, thus letting you to combine the three compute kernels into just one:

Since there are no more separate compute kernels, the rays and intersection buffers can be completely eliminated, thus also eliminating the need to read/write them into memory. The acceleration structure (just like the intersector) was built in the MPS framework before. Now you can use the new API to create an acceleration structure like this:

let accelerationStructureDescriptor = 
				MTLPrimitiveAccelerationStructureDescriptor()
let geometryDescriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
geometryDescriptor.vertexBuffer = vertexBuffer
geometryDescriptor.triangleCount = triangleCount
accelerationStructureDescriptor.geometryDescriptors = [geometryDescriptor]
let sizes = device.accelerationStructureSizes(
				descriptor: accelerationStructureDescriptor)
let accelerationStructure =
    device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize,
                                      options: .storageModePrivate)
let commandEncoder = commandBuffer.makeAccelerationStructureCommandEncoder()
commandEncoder.build(accelerationStructure: accelerationStructure,
                     descriptor: accelerationStructureDescriptor,
                     scratchBuffer: scratchBuffer,
                     scratchBufferOffset: 0)

```

You can also bind the acceleration structure with an already existing command encoder:

computeEncoder.setAccelerationStructure(accelerationStructure, bufferIndex: 0)

```

Then, you simply pass the acceleration structure to an intersector on the GPU:

[[kernel]]
void raytracer(primitive_acceleration_structure structure [[buffer(0)]],
	       uint2 tid [[thread_position_in_grid]])
{
    // Generate ray
    ray r = generateCameraRay(tid);
    // Create an intersector
    intersector<triangle_data> intersector;
    // Intersect with scene
    intersection_result<triangle_data> intersection;
    intersection = intersector.intersect(r, structure);
    // shading...
}

```

The intersector takes in a ray as argument and traverses the accelerator structure (as its 2nd argument). It will continuously update the closest intersection and send the result to the shading stage:

The new intersection functions allows you to accept a found intersection early and skip traversing the rest of the acceleration structure. Alpha testing is a good such candidate for early intersection accepting. Here is how to write a custom triangle intersection function:

[[intersection(triangle, triangle_data)]]
bool alphaTestIntersectionFunction(
			uint primitiveIndex        [[primitive_id]],
			uint geometryIndex         [[geometry_id]],
			float2 barycentricCoords   [[barycentric_coord]],
			device Material *materials [[buffer(0)]])
{
    texture2d<float> alphaTexture = materials[geometryIndex].alphaTexture;
    float2 UV = interpolateUVs(materials[geometryIndex].UVs,
        		       primitiveIndex, barycentricCoords);
    float alpha = alphaTexture.sample(sampler, UV).x;
    return alpha > 0.5f;
}

```

Also new this year, [[intersection(..)]] is the only other new function qualifier along with [[visible]] which was mentioned earlier. Besides triangles, intersection functions can also be used for bounding box intersections, so you can enclose a sphere inside a bounding box, for example. Rendering a Bezier curve instead of a triangle will also look more realistic on surfaces like hair.

You can create a bounding box acceleration structure like this:

let accelerationStructureDescriptor = 
			MTLPrimitiveAccelerationStructureDescriptor()
let geometryDescriptor = 
			MTLAccelerationStructureBoundingBoxGeometryDescriptor()
geometryDescriptor.boundingBoxBuffer = boundingBoxBuffer
geometryDescriptor.boundingBoxCount = boundingBoxCount
accelerationStructureDescriptor.geometryDescriptors = [geometryDescriptor]

```

Then, you simple pass the acceleration structure to an intersector on the GPU:

struct BoundingBoxResult {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};
[[intersection(bounding_box)]]
BoundingBoxResult sphereIntersectionFunction(
				float3 origin            [[origin]],
				float3 direction         [[direction]],
				float minDistance        [[min_distance]],
				float maxDistance        [[max_distance]],
				uint primitiveIndex      [[primitive_id]],
				device Sphere *spheres   [[buffer(0)]],
				ray_data float3 & normal [[payload]])
{
    float distance;
    if (!intersectRaySphere(origin, direction, 
    			    spheres[primitiveIndex], &distance))
        return { false, 0.0f };
    if (distance < minDistance || distance > maxDistance)
        return { false, 0.0f };
    float3 intersectionPoint = origin + direction * distance;
    normal = normalize(intersectionPoint - spheres[primitiveIndex].origin);
    return { true, distance };
}

```

As you can see, if you want to return more data than just what the BoundingBoxResult lets you do, you can use the new [[payload]] attribute (in the ray_data address space, which is also new this year) to also return, for example, the surface normal at the intersection point so you can use it for shading. You should only modify the payload if you will accept the current intersection. To retrieve the payload value, you just need to change the call to the intersector to give it a one more argument:

[[kernel]]
void raytracer(...)
{
  // generate ray, create intersector... 
  float3 normal;
  intersection = intersector.intersect(ray, accelerationStructure, 
  				       functionTable, normal);
  // shading...
}

```

The intersector needs a way to be linked with the intersection functions so that various pieces of geometry are mapped to different intersection functions. The new Intersection Function Tableuses offsets to achieve that:

After linking all intersection functions into the compute pipeline state as showed before, you then need to create the table:

let descriptor = MTLIntersectionFunctionTableDescriptor()
descriptor.functionCount = intersectionFunctions.count
let functionTable = computePipeline.makeIntersectionFunctionTable(
						descriptor: descriptor)
for i in 0 ..< intersectionFunctions.count {
    let functionHandle = computePipeline.functionHandle(
    					function: intersectionFunctions[i])
    functionTable.setFunction(functionHandle, index: i)
}
functionTable.setBuffer(sphereBuffer, offset: 0, index: 0)
encoder.setIntersectionFunctionTable(functionTable, bufferIndex: 1)

```

Finally, you simply pass the function table (along with the acceleration structure) to an intersector on the GPU:

[[kernel]]
void raytracer(
	primitive_acceleration_structure accelerationStructure   [[buffer(0)]],
	intersection_function_table<triangle_data> functionTable [[buffer(1)]])
{
    // generate ray, create intersector...
    intersection = intersector.intersect(r, accelerationStructure, 
    					 functionTable);
    // shading...
}

```

Also new this year are the following functions and types: the primitive_acceleration_structureand instance_acceleration_structures types, the ray struct type, the intersector<>struct type, the intersection_result struct type, the intersection_function_table struct type and the intersector functions described in the MSL Specification (paragraph 6.16). They are all defined in <metal_raytracing> under the metal::raytracing namespace.

Finally, the following topics are also changed or new in macOS 11, however, not covered in this article:
- Metal Performance Shaders Graph
- ASTC-compressed 6×6 pixel format with HDR content
- Support for working with sparse textures
- MTLSamplerDescriptor property for LOD

For a complete list of new features consult the Metal API documentation website. For the latest additions to the Metal Shading Language consult the MSL 2.3 specification. The new WWDC20 sample code and the WWDC20 videos are available on the Apple website as well. All code snippets and images in this article belong to Apple.

Until next time!

#### Leave a Reply Cancel reply

Your email address will not be published. Required fields are marked *

Name * 

Email * 

Website 

Comment *

 Save my name, email, and website in this browser for the next time I comment.
