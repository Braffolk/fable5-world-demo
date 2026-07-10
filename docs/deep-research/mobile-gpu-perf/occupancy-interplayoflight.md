# Shader Occupancy — what it is, why we care (reference copy)
Source: https://interplayoflight.wordpress.com/2020/11/11/what-is-shader-occupancy-and-why-do-we-care-about-it/
Author: Kostas Anagnostou. Fetched 2026-07-05 (WebFetch extraction, faithful).

---

## GPU Execution Model: Batching and Latency
GPUs batch 64 or 32 pixels/vertices during shader execution — **wavefronts** (AMD) / **warps** (NVIDIA). One instruction executes across all items in a batch simultaneously.

Memory fetch instructions carry significant latency (cache hierarchy, potential RAM access). GPU strategy:
- issue the memory instruction (request data)
- continue executing subsequent instructions
- check data availability only when needed
- if not ready → stall; if ready → proceed

## Case study (AMD GCN ISA)
The compiler reorders to MAXIMIZE distance between the instruction that REQUESTS memory and the one that USES it, and inserts `s_waitcnt` waits only when the data is actually needed. Latency hidden via instruction-level parallelism.

## VGPRs and register pressure
- Each Compute Unit has a fixed pool of Vector GPRs. GCN: 256 vector registers per SIMD.
- More VGPRs per shader ⇒ fewer wavefronts can be assigned to that SIMD.
- 32 VGPRs ⇒ 8 wavefronts per SIMD. >128 VGPRs ⇒ ZERO wavefronts schedulable.

## Occupancy definition
Occupancy = max number of wavefronts a SIMD can have assigned + ready at once, enabling rapid switching when the current batch stalls. GCN: max 10 wavefronts/SIMD.
- Low VGPR (32) → 8 wavefronts to swap → rarely stalls.
- High VGPR (>128) → 0 wavefronts → must wait for memory.

## Nuance — occupancy is NOT the whole story
**Low occupancy isn't always bad:** if the compiler schedules enough ALU between the memory request and use, latency is fully hidden without batch-swapping. Don't optimize on occupancy alone — profile for stalls on actual memory reads; raise occupancy only if memory-read stalls are the bottleneck.

**High occupancy isn't always good:**
- (1) Cache contention: many in-flight batches compete for limited cache; each can invalidate another's lines → higher effective latency.
- (2) Register serialization: to hit high occupancy the compiler may minimize VGPRs and SERIALIZE memory ops (fetch → wait → store → use → reuse the register). Fewer registers ⇒ worse fetch scheduling ⇒ higher latency. More VGPRs ⇒ multiple fetches issued upfront, results cached, more scheduling freedom.

## Takeaways
1. Occupancy = count of ready wavefronts on a SIMD.
2. Inversely tied to VGPR usage.
3. High occupancy hides latency via batch-switching.
4. Low occupancy can still be fine if ILP hides latency.
5. Too-high occupancy risks cache contention + register serialization.
6. PROFILE for real impact; occupancy is a proxy, not the goal.
