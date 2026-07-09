<!-- source: https://developer.apple.com/videos/play/tech-talks/10580/ -->
<!-- Apple Tech Talk video. Transcript content extracted via WebFetch. -->

# Metal Compute on MacBook Pro (Apple Tech Talk 10580)

Speaker: Jason Fielder, GPU Software Engineering, Apple. Covers Metal compute optimization for M1 Pro / M1 Max.

## THE KEY CASE STUDY — kernel stuck at 16% occupancy from register spilling

> "With this kernel example, we can see that our occupancy is at 16 percent, and this is really low. Looking at the compiler statistics for this kernel shows the relative instruction costs of it, including the spilled bytes. This spill, along with the temporary registers, is likely the cause of our poor occupancy. We are exhausting the thread memory, and occupancy is reduced to free up more registers for the threads that will run."

**Occupancy definition:** "the measure of how many threads are currently active on the GPU, relative to the maximum that could be. When this figure is low, it is important to understand why, to determine if this is expected or signifies a problem."

**Register block granularity:** "Registers are allocated to a kernel in register blocks, and as such, you'll need to reduce the usage by up to the block size to see a potential increase in occupancy." (i.e. shaving a few registers may show no gain until you cross a whole block boundary.)

## The recovery levers (how they got occupancy back up)

1. **Prefer 16-bit types over 32-bit.** "Preferring 16bit types over 32bit types increases the number of registers available to other parts of the kernel. Conversion between these types to their 32bit counterparts is usually free."

2. **Reduce stack-allocated data.** "Reducing the data stored on the stack -- for example, large arrays or structs -- can consume a large number of registers and reducing them is an effective tool."

3. **Use the constant address space well.** "Look to tune your shader inputs to make the best use of the constant address space. This can drastically reduce the number of general purpose registers being used unnecessarily."

4. **Avoid dynamic indexing of stack arrays.** A stack array indexed by a value NOT known at compile time will likely spill to memory. If the index IS known at compile time, "the compiler will likely unroll the loop and be able to optimize away any spill."

5. **Reduce threadgroup (shared) memory.** "With a high thread-group memory usage, the only way to increase occupancy is to reduce the amount of shared memory used. Reducing thread-group memory can also help reduce the impact of thread-register pressure."

6. **Set maxThreadsPerThreadgroup / max_total_threads_per_threadgroup.** "There is scope for the compiler to spill registers more efficiently when the maximum thread count in a thread group is known at pipeline state creation time... Aim for a value that is the smallest multiple of the thread execution width that works for your algorithm."

## Profiling path used
- Xcode Metal debugger **compiler statistics** for the kernel: shows relative instruction costs INCLUDING spilled bytes and temporary registers — this is how the spill was diagnosed as the occupancy cause.

## GPU working-set limits (M1 Pro / Max)
- M1 Pro or M1 Max + 32GB RAM: GPU can access 21GB.
- M1 Max + 64GB RAM: GPU can access 48GB.
