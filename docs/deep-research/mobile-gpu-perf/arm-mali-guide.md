
## References

1. [Arm GPU Best Practices Developer Guide](https://developer.arm.com/documentation/101897/latest/), [[backup](../pdf/arm-gpu_best_practices_developer_guide_3_4.pdf)]
2. [Arm GPU Datasheet](https://developer.arm.com/documentation/102849/latest/), [backup: [v3](../pdf/Arm-GPU_Datasheet_v3.pdf), [v7](../pdf/Arm-GPU_datasheet_v7.pdf), [v8](../pdf/Arm-GPU_Datasheet_v8.pdf)]
3. [AFBC](https://developer.arm.com/Architectures/Arm%20Frame%20Buffer%20Compression)
4. [(video playlist) Arm Mali GPU Training Series](https://www.youtube.com/playlist?list=PLKjl7IFAwc4QUTejaX2vpIwXstbgf8Ik7)
5. [Authoring Efficient Shaders for Optimal Mobile Performance](https://www.dropbox.com/s/ic0c6k0yzf2uk81/Authoring%20Efficient%20Shaders%20for%20Optimal%20Mobile%20Performance%20-%20GDC2022%20-Arm%20%26%20NaturalMotion.pdf?dl=0)
6. [Memory limits with Vulkan on Mali GPUs](https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/memory-limits-with-vulkan-on-mali-gpus)
7. [Forward Pixel Kill](https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/killing-pixels---a-new-optimization-for-shading-on-arm-mali-gpus), [[webarchive](https://web.archive.org/web/20240922023725/https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/killing-pixels---a-new-optimization-for-shading-on-arm-mali-gpus)]
8. [GDC2015: How to Optimize Your Mobile Game with ARM Tools and Practical Examples](https://www.gdcvault.com/play/1022397/How-to-Optimize-Your-Mobile) (Midgard architecture)
9. [Hidden Surface Removal in Immortalis-G925: The Fragment Prepass](https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/immortalis-g925-the-fragment-prepass), [[webarchive](https://web.archive.org/web/20241202033355/https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/immortalis-g925-the-fragment-prepass)]
10. [Vulkanised 2025: Vulkan Best Practices for Mobile Development](https://youtu.be/fWeR4ECVcW8)
11. [Constant data in Vulkan](https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/performance/constant_data)

**Guide**<br/>
1.1. [Accelerating 2D Applications](https://developer.arm.com/documentation/102524/0100/Overview)<br/>
1.2. [Tile-Based Rendering](https://developer.arm.com/documentation/102662/0100/Overview)<br/>
1.3. [Principles of High Performance guide](https://developer.arm.com/documentation/102544/0100/Overview)<br/>
1.4. [Workload Pipelining](https://developer.arm.com/documentation/102521/0100/Overview)<br/>
1.5. [GPU Processing Budget Approach to Game Development](https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/gpu-processing-budget-approach-to-game-development), [[webarchive](https://web.archive.org/web/20220810172906/https://community.arm.com/arm-community-blogs/b/graphics-gaming-and-vr-blog/posts/gpu-processing-budget-approach-to-game-development)]

**Real-time 3D Art Best Practices**<br/>
2.1. [Geometry Guide](https://developer.arm.com/documentation/102448/0200/Overview)<br/>
2.2. [Lighting Guide](https://developer.arm.com/documentation/102109/0100/Overview)<br/>
2.3. [Materials and Shaders Guide](https://developer.arm.com/documentation/102471/0200/Overview)<br/>
2.4. [Texturing Guide](https://developer.arm.com/documentation/102449/0200/Overview)<br/>

**Wiki/Specs**<br/>
3.1. [Wikipedia](https://en.wikipedia.org/wiki/Mali_(processor))
3.2. [Namu wiki](https://en.namu.wiki/w/ARM%20Mali%20GPU)


## Notes

* Arm Frame Buffer Compression (AFBC). [1][3]
	- In such cases it reduces the overall system level bandwidth and power cost of transferring spatially coordinated image data throughout the system by up to 50%.
	- Supported from the Mali-T760. Partial Vulkan support for AFBC is available from Mali-G71 onwards, and full support from Mali-G31, Mali-G51, and Mali-G76.
	- Use the texture() and texelFetch() functions in shaders to access textures and images that the GPU previously rendered as framebuffer attachments.
	- When packing data into color channels, to get the best compression rates, store the most volatile bits in the least significant bits of the channel.
	- Do not use imageLoad() or imageStore() to read or write into a texture or image that the GPU has rendered as a framebuffer attachment. Doing so triggers decompression.
	- Setting VkImageUsageFlags VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT or VkImageCreateFlags VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT stops AFBC use on GPUs before Valhall.
	- VkImageCompressionPropertiesEXT::imageCompressionFlags will contain VK_IMAGE_COMPRESSION_DEFAULT_EXT if AFBC is used.
	- Minimize use of storage images as they cannot use AFBC compression.

* Arm Fixed Rate Compression (AFRC).
	- Is a lossy image compression format. AFRC can be used for compressing external texture inputs and framebuffer outputs from the GPU. Configurable compression ratio provides guaranteed bandwidth reduction for such surfaces and memory footprint saving. [arm]
	- VkImageCompressionPropertiesEXT::imageCompressionFlags will contain VK_IMAGE_COMPRESSION_FIXED_RATE_EXPLICIT_EXT if AFRC is used. [1]
	- The main, and safest, use of AFRC is for display images. [1]

* Foveated Rendering allows VR application to specify the shading rate to be used in each part of the screen. This helps the developer to reduce the workload of VR applications by selectively define the part of the screen that require less work from the GPU due to distortion introduced by the VR headset lenses. (fragment density in Vulkan). [arm]

* How to optimize blending: [1]
	- Prefer blending on unorm formats, rather than floating-point values.
	- Monitor the number of blended layers that are being generated on a per-pixel basis. Even if the shaders are simple, high layer counts quickly consume cycles due to the number of fragments that must be processed.
	- Consider splitting large UI elements into opaque and transparent portions. The opaque and transparent portions can then be drawn separately, allowing either early ZS, or FPK, to remove the overdraw beneath the opaque parts.
	- Do not use blending on floating-point framebuffers.
	- Do not use blending on multisampled floating-point framebuffers.
	- Do not generalize the user interface rendering code so that blending is always enabled.
	- Do not just set alpha to 1.0 in the fragment shader to disable blending.

* **Transaction Elimination** spots the identical pixel blocks between two consecutive render targets and performs a partial update to the frame buffer with the changed pixel blocks only, which reduces memory bandwidth and thus energy. [arm]
	- TE works for whole tile (16x16 px).

* Smart Composition extends the concept of **Transaction Elimination** to every stage of UI composition. Identical pixel blocks of input surfaces are not read, not processed for composition and not written to final frame buffer. [arm]

* **Transaction Elimination** is used for an image if: [1]
	- The sample count is 1.
	- The mipmap level is 1.
	- The image uses COLOR_ATTACHMENT_BIT.
	- The image does not use TRANSIENT_ATTACHMENT_BIT.
	- The effective tile size is 16x16 pixels. Pixel data storage determines the effective tile size.

* To avoid unnecessary signature invalidation, use the safe image layouts for color attachments. [1]
	- These layouts are: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.

* sRGB: [1]
	- Both sampling from and rendering or blending to an sRGB surface comes at no performance cost.

* Storage image: [1]
	- Memory coherence across threads in a quad is important. Therefore, make sure to group four adjacent lanes to use addresses in the same 64 byte cache line.
	- Do not use imageLoad() in compute unless you must use coherent read and writes within the dispatch.

* Textures: [1]
	- Avoid textureGrad() unless it is essential. It is much slower than texture() and textureLod(). Instead, replace with trilinear or bilinear filtering where possible.
	- The texture unit is designed to give full-speed performance for both nearest sample and bilinear filtered, LINEAR_MIPMAP_NEAREST, texel sampling.
	- Cases where extra texture sampling cycles are required:
		* Trilinear, LINEAR_MIPMAP_LINEAR, filtering has a 2x cost.
		* 3D formats have a 2x cost.
		* FP32 formats have a 2x cost.
		* Depth formats have a 2x cost for Midgard GPUs, but only a 1x cost on Bifrost GPUs. For Valhall GPUs, Depth is generally a 1x cost. But, if there is a reference or comparison depth, then it is a 2x cost, for example Shadow Maps.
		* Cubemap formats have a 1x cost per cube face that is accessed.
		* For older Midgard, or first-generation Bifrost GPUs, YUV formats have an Nx cost, where N is the number of texture planes that are required. For the second-generation Bifrost and Valhall GPUs, meaning Mali-G51 onwards, YUV has a 1x cost, irrespective of the plane count.
	- Use mediump samplers. highp samplers can be half the speed due to their wider data path requirements.
	- If you need higher dynamic range, then consider using packed 32-bit formats. Such 32-bit formats include RGB10_A2 or RGB9_E5, as an alternative to FP16 or FP32 textures.
	- Consider using 2x bilinear AF in preference to isotropic trilinear filtering. 2x bilinear is faster, and has better image quality in regions of high anisotropy. Note, that by switching to bilinear filtering, you can see some visible seams at the decision point between mipmap levels.
	- Do not use higher levels of max anisotropy without reviewing performance. 8x bilinear AF costs eight times more GPU computational power than a simple bilinear filter.
	- Do not use trilinear AF without reviewing performance. 8x trilinear AF costs 16 times more than a simple bilinear filter.

* Workgroup sizes: [1]
	- The GPU hardware can split up, and then merge, workgroups during shader core resource scheduling. If barriers or shared memory are used, then GPUs cannot do this with workgroups.
	- Large workgroup sizes restrict the number of registers that are available to each work item in this scenario. In turn, forcing shader programs to use stack memory if insufficient registers are available.
	- Use 64 as a baseline workgroup size.
	- When working with images or textures, use a square execution dimension, for example 8x8, to exploit optimal 2D cache locality.
	- Do not assume that barriers with small workgroups are free from performance costs.

* Shared memory: [1]
	- Arm GPUs do not implement dedicated on-chip shared memory for compute shaders. The shared memory that is available to use is system RAM that is backed up by the load-store cache.
	- Keep your shared memory as small as possible, as it reduces the chance of thrashing the data cache.
	- Do not copy data from global memory to shared memory on Arm GPUs. Doing so pollutes the caches.
	- For barriers, smaller workgroups are less expensive.

* FP16: [1]
	- If using explicit FP16, make sure to vectorize the code by using f16vec2 or f16vec4. Modern GPU architectures use packed f16x2 instructions to improve arithmetic performance. Scalar float16_t does not gain the same benefit.

* Memory access: [1]
	- For memory accesses with a single thread, use vector data types.
	- Bifrost GPUs can run four neighboring threads in lock-step, which is known as a quad. You must access overlapping or sequential memory ranges across the four threads to allow load merging. Mali-G52 and Mali-G76 GPUs can do eight threads in a warp, and Valhall GPUs can do 16.
	- Do not use scalar loads if vector loads are possible.
	- Do not access divergent addresses across a thread quad where possible.

* Push constants: [1]
	- Push constants have no advantage over uniform buffers on Arm GPUs.
	- Vulkan push constants are equal to UBOs for Arm GPUs.

* UBO: [1]
	- Keep your uniform data small. 128 bytes is a good general rule for how much data can be promoted to registers in any given shader.
	- Avoid dynamic indexing where possible, whether using push constants, or uniform buffer objects.
	- Do not dynamically index into uniform arrays.
	- Do not over use instancing. Instanced uniforms that are indexed using gl_InstanceID count as being dynamically indexed and cannot use register mapped uniforms.

* Uniforms: [1]
	- Minimize the number of uniform-on-uniform or uniform-on-literal computations. Compute the result of the uniform subexpression on the CPU and then upload that as your uniform.
	- Do not use uniform values that parametrize control-flows. Instead, specialize shaders for each control path that is needed.

* Atomics: [1]
	- Atomics are efficient when a shader core controls the necessary cache line in its L1.
	- Consider spacing atomics 64 bytes apart to avoid multiple atomics contending on the same cache line.
	- Consider whether it is possible to amortize the contention by accumulating into a shared memory atomic (L2). Then, have one thread push the global atomic operation at the end of the workgroup.

* Bindless resources: [1]
	- A bindless descriptor set will work well with combining resources, allowing indexing into texture and buffer arrays.

* Render passes: [1]
	- Use a 128-bit G-buffer budget for color.
	- Set up any attachment that is only live during a single render pass as a TRANSIENT_ATTACHMENT that is backed by LAZILY_ALLOCATED memory.
	- Ensure that the contents of read-only attachments that you did not modify during the pass, but still need to keep, are not written at the end of the render pass using storeOp = STORE_OP_NONE.

* MC, MP: [[arm community](https://community.arm.com/support-forums/f/graphics-gaming-and-vr-forum/46737/i-want-to-know-about-the-gpu-line-up)]
	- MC is the number of physical cores.
	- MP mean the number of pixels per clock the GPU can generate.
	- MP on older GPU which are single pixel per core, it also gives the pixel per clock count.
	- Starting from Mali Bifrost: MP4 equal to MC2 with 2 pixel per clock, MP3 is MC2 with different cores: 2 pixel per clock + 1 pixel per clock.
	- Based on `VK_ARM_shader_core_builtins` they are same. [[az](https://github.com/azhirnov)]

* Forward Pixel Kill (FPK):
	- Calculations already in flight can be terminated at any time if we spot that a later thread will write opaque data to the same pixel location. Since each thread takes a finite time to complete, we have a window in time which  we can exploit to kill pixels already in the pipeline. [7]
	- Situations where FPK commonly fails include draw calls using: [1]
		* Alpha blended transparency
		* Shader programmatic framebuffer access.
		* Late ZS testing.
		* Small triangles.
	- Do not rely on the FPK optimization alone. An early ZS test is always more energy-efficient. [1]

* Fragment Pre-pass. What makes a draw call incompatible: [9]
	- A non-opaque draw call that writes Z or S.
		* If draw that does not fully overwrite all render targets that have previously been written to in the tile, then it is effectively transparent.
		* Draw calls that read the tile buffer are also considered transparent.
	- A ZS-only draw that follows a compatible non-opaque draw call is considered incompatible.
	- Fragment shader side effects: **read-write** access to something, atomics if used result value.
	- Anything where the rasterizer coverage is required in the fragment shader:
		* Centroid varyings
		* Reading the coverage mask
		* Checking if the lane is a helper lane
	- Reading from the tile buffer at sample positions for which the primitive doesn't have coverage.
	- Draw calls that write Z or Stencil where the fragment shader modifies coverage, but the shader explicitly states that ZS testing and update must happen early.

* Fragment Pre-pass benefits: [9]
	- No need to sort by depth any more.

* Transfer commands may be executed in compute and fragment stream. [10]
	- can cause false dependencies

* Avoid indexing in the shader if possible, such as dynamically indexing into buffer arrays or uniform arrays, as this can disable shader optimisations. [11]
	- on Mali it is not a recommend practice as it disables a compiler optimisation technique known as pilot shaders.
	- Pilot shaders are a technique that allows us to determine what calculations can be "piloted" into your GPU’s register so that when the data needs to be read it doesn’t take a full read cycle from the GPU RAM.

* Tiles are rendered in parallel: [[az](https://github.com/azhirnov)]
	- single tile 16x16 is at least 16x simd16 warps *()*
	- single core can process <= 32 warps (Valhall Gen2)
	- on multicore GPU *(MC12 for example)* some cores render first tile, some second tile, etc.
	- cores per tile depends on warps per core, which depends on register count (?).
	- Valhall supports 0-32 registers with max warps per core and 32-64 with half size *(similar to compute workgroup size)*.
	- some cores can execute vertex/binning or compute tasks in parallel with rasterization and fragment shading.
	- before Image Region Dependencies at the end of tile rendering core may have low number of active warps.
