# Prior art: FreePipe (Liu, Huang, Liu, Wu — I3D 2010)

**Source:** "FreePipe: A Programmable Parallel Rendering Architecture for Efficient
Multi-Fragment Effects", Liu/Huang/Liu/Wu, ACM SIGGRAPH I3D 2010, pp.75–82.
DOI 10.1145/1730804.1730817.

**Fetch outcome:** The ACM PDF (dl.acm.org/doi/pdf/10.1145/1730804.1730817) is
paywalled — `curl` returned a 5.6KB HTML block page, not a PDF. UM repository
(`repository.um.edu.mo/handle/10692/5292`) explicitly hosts no local file ("There
are no files associated with this item"). kunzhou.net / ucsb mirrors 404'd. No free
author preprint exists. **However** FreePipe's architecture and — critically — its
no-64-bit-atomic depth trick are described precisely in three papers already present
in `sources/papers/`, which I deep-read instead:
- **CuRast** (`extra-0.pdf`, Fellner & Behnke), §2.2 + §3.1 — most detailed, gives
  the exact bit layout of FreePipe's dual-32-bit-atomic workaround.
- **Laine & Karras CudaRaster** (`laine2011-cudaraster.pdf`, HPG 2011), §2, §4.1,
  §6, Table 1 — reimplements FreePipe as a baseline, critiques its scheduling, and
  benchmarks it on a vegetation scene (San Miguel) directly analogous to our foliage.
- **LucidRaster** (`lucid.pdf`), §2 — one-line confirmation of the per-triangle focus.

This brief is sourced from those, not from a hallucinated reading of the original.

---

## What FreePipe is, and why it matters to US specifically

FreePipe is **the canonical pre-64-bit-atomic GPU software rasterizer**, and its core
scheduling model is *exactly what our system already does*:

> "FreePipe is a software rasterization pipeline... Scheduling is very simple: **each
> thread processes one input triangle, determines its pixel coverage and performs
> shading and blending sequentially for each pixel.**" — CudaRaster §2

> "The trivial solution is to **expand each triangle directly to pixels and shade them
> immediately, which is what FreePipe does.**" — CudaRaster §4.1

Our raster is the cluster-batched version of this: one workgroup per cluster, each
thread/iteration walks one triangle's bbox and does edge-test + barycentric-z +
atomic election per covered fragment. So FreePipe is less a menu of *new* techniques
to adopt and more a **mirror of our current architecture plus the literature's verdict
on its failure mode** — which happens to be precisely our failure mode (dense
overdraw of tiny tris). The transferable value is therefore split:
1. ONE direct, immediately-portable technique (the dual-32-bit-atomic depth election)
   that is the canonical answer to "how do you do a depth|payload election without
   64-bit atomics" — and it confirms our packed-32-bit `atomicMax` is the right idea.
2. The literature's measured indictment of the per-triangle/per-cluster scatter model
   on vegetation, pointing INDIRECTLY but loudly at sort-middle binning as the real
   structural win (detailed in the CudaRaster brief; summarized here for the chain).

---

## Raster architecture (FreePipe)

- **Single geometry pass, one thread per triangle.** No clipping/sorting stages; each
  thread rasterizes its triangle's screen bbox to pixels in a sequential inner loop.
- **No inter-thread communication.** Visibility is resolved purely by a global-memory
  **atomic-min on a packed depth word** — the atomic implicitly serializes concurrent
  writers to the same pixel, so there is no fragment queue and no sort
  (CuRast §3.1: "Because global atomics implicitly synchronize concurrent writes, this
  approach eliminates the need for inter-thread communication or fragment queuing and
  sorting"). This is the same insight our `atomicMax` election rests on.
- **No early-Z, no hierarchical Z, no tiling.** Pure brute-force scatter. Overdraw is
  paid in full at the atomic.
- Two multi-fragment variants in the paper (A-buffer-ish per-pixel linked lists and a
  fixed-depth array) layer on top, but those are for OIT/transparency and are
  irrelevant to our opaque-foliage visibility goal.

---

## The depth/visibility ELECTION and the NO-64-BIT-ATOMIC adaptation (THE key finding)

This is the single most relevant thing in the source for our central question.
FreePipe predates 64-bit atomics, so it faced *our exact constraint*. Its answer
(CuRast §2.2, verbatim):

> "Since CUDA was limited to 32 bit atomics back then, which did not allow atomically
> writing depth and color values with a single atomic operation, they suggested
> **performing two 32-bit atomic-min operations with the same 20-bit depth value but
> different 12 bits of the color value into the same pixel that is separated into two
> buffers.** Afterwards, they extract the 12-bit parts located in different buffers,
> and fuse them back into a 24-bit color value."

So FreePipe's packing was: each 32-bit atomic word = `[20-bit depth | 12-bit payload]`,
issued **twice** into two separate framebuffers carrying the *same* depth key but two
disjoint 12-bit halves of a 24-bit payload. Because both atomics use the identical
depth as the high bits, the *winner* of one min is the winner of the other, so the two
12-bit halves reassemble coherently in a resolve pass.

**How this maps to us / what to adapt:**
- Our scheme is strictly *better* than FreePipe's for the 25-bit cluster/tri id: we do
  ONE `atomicMax` on `[24b inverted-depth | 8b tiebreak]` and the **election winner
  then plain-stores** the full 25-bit id into a side buffer. FreePipe instead crammed
  the *whole* payload into the atomic word(s), which is why it needed two atomics to
  carry 24 bits. Our "atomic elects, winner stores" indirection is the more scalable
  pattern and removes any payload-width ceiling. **FreePipe confirms our design is
  sound and is the historically-validated way to do this without 64-bit atomics.**
- FreePipe's dual-atomic trick is therefore a *fallback we don't need* — we already
  decoupled payload from the atomic. Worth recording only as the canonical reference
  if we ever needed to carry payload *inside* the atomic (we don't).
- **Hazard FreePipe documents that applies to us** (CudaRaster §2): *"Conflicting cases
  where depth and color of two fragments are equal may be missed due to a race
  condition."* In FreePipe, equal-depth ties are non-deterministic. We mitigate this
  with our 8-bit tiebreak in the low bits of the atomic word — that tiebreak is exactly
  the thing FreePipe lacked, so it is doing real correctness work, not just packing. Do
  NOT remove it.

---

## Work distribution — the indictment of our current model (INDIRECT lever)

CudaRaster benchmarks its own sort-middle pipeline against a hand-optimized FreePipe on
five scenes (Table 1). The decisive datapoint for us is **SAN MIGUEL** — chosen because
it "includes a lot of vegetation that consists of very small triangles" (CudaRaster §6),
i.e. the closest published analog to our holey foliage:

| Scene (1024×768)   | Sort-middle (ms) | FreePipe (ms) | FreePipe : sort-middle |
|--------------------|------------------|---------------|------------------------|
| SAN MIGUEL (foliage)| 9.48            | 510.20        | **53.8× slower**       |
| SAN MIGUEL (2048²) | 15.44            | 1652.52       | **107× slower**        |
| CITY               | 3.13             | 251.86        | 80.5× slower           |
| STALKER            | 2.31             | 92.73         | 40× slower             |
| BUDDHA (large tris)| 2.66             | 3.08          | 1.16× (≈tie)           |

The pattern is unambiguous: FreePipe's per-triangle scatter is competitive *only* when
overdraw is low (BUDDHA). On **dense small-triangle vegetation it is catastrophically
worse — 50–100×** — and that gap is the cost of *not binning*. CudaRaster names the two
root causes (§2):
1. *"highly variable number of pixels in each triangle leads to poor thread utilization"*
   (less relevant to us — our clusters are well-filled, tris are uniformly ~1px), and
2. the structural one: every fragment hits **global-memory atomics with no spatial
   locality and no hierarchical/early Z to suppress overdraw before the per-fragment
   work.** Each of our ~36M visible fragments pays the full edge-test + bary-z + atomic
   load regardless of being later occluded.

**The transferable principle (INDIRECT):** the fix the entire post-FreePipe lineage
adopts is **sort-middle binning** — bin → coarse-tile → fine-pixel — so that (a) per-tile
work lands in a CTA with the framebuffer tile in fast shared memory, (b) coverage is
computed once per tile via a hierarchical mask, and (c) a per-tile **hierarchical-Z /
z_max** running in shared memory kills whole triangles/fragments *before* the
per-fragment loop. This directly attacks our DOMINANT 60% per-pixel cost by removing
fragments rather than making each one cheaper. **This is FreePipe pointing at the door,
not walking through it — the actual mechanism lives in the CudaRaster / LucidRaster /
cuRE briefs; do not re-derive it here.** (Note our refuted-deadend caveat: a *conservative*
HZB is ~0% on holey foliage; but CudaRaster's z_max is per-tile *and updated live during
the same pass as surfaces are drawn*, which is a different, finer instrument than a
pre-pass cluster-vs-HZB cull — see the CudaRaster brief for whether the live-tile variant
survives the holey-foliage objection.)

---

## Transferable techniques (ranked)

### DIRECT
1. **Atomic-min/max packed-depth election as the visibility primitive — already ours.**
   FreePipe is the origin of "use one packed atomic per pixel, no sort, no queue."
   Confirms our `atomicMax([24b depth|8b tiebreak])` is the textbook no-64-bit-atomic
   approach. Effort: none (validation only). Quality: lossless (it IS our method).
2. **Dual-atomic payload-splitting fallback** (two 32-bit atomic-mins, same depth key,
   disjoint payload halves, fuse in resolve). DIRECTLY portable to WGSL, but we **don't
   need it** — our "atomic elects, winner plain-stores the 25-bit id" already beats it
   and has no payload-width limit. Record as reference only. Not-a-win for us.
3. **Keep the low-bit tiebreak in the atomic word.** FreePipe's documented equal-depth
   race condition is exactly what our 8-bit tiebreak prevents. Keep it. Effort: none.

### INDIRECT (adapt-the-principle)
4. **Sort-middle binning instead of per-cluster/per-triangle scatter.** FreePipe's
   50–107× slowdown on vegetation is the measured cost of skipping this. The principle:
   bin triangles into screen tiles, process each tile in one workgroup with the
   framebuffer tile in shared memory, and run a **live per-tile z_max hierarchical kill**
   to drop occluded fragments before the per-fragment loop. Attacks the 60% per-pixel
   loop (fewer fragments) AND can amortize transform/setup. **Big refactor.** Mechanism
   detailed in CudaRaster brief, not here. Quality: lossless if z_max is exact per-tile.
5. **Hierarchical/early per-tile Z that is BUILT LIVE during the raster pass** (vs a
   conservative pre-pass HZB, which we refuted as ~0% on holey foliage). FreePipe's
   absence of any Z-kill is *why* it pays full overdraw; CudaRaster adds a live z_max in
   shared memory. The open question for our holey foliage is whether a *fine-grained
   live* z_max (per 8×8 tile, exact) suppresses meaningful overdraw where a coarse
   conservative cluster-HZB did not — pursue in the CudaRaster/Nanite analysis.

---

## Honest bottom line

FreePipe's *direct* contribution to our perf hunt is **confirmation, not novelty**: it
validates our packed-32-bit atomic election (and shows the historical dual-atomic
fallback we already out-design), and it confirms our tiebreak is load-bearing. Its real
value is **negative evidence**: FreePipe IS our current per-triangle/per-cluster scatter
architecture, and the literature measures it at 50–107× slower than sort-middle on
exactly the vegetation/overdraw workload we have. That is the strongest single
signal in the prior art that our 60% per-pixel bottleneck is *structural to the scatter
model* and that **binning + live per-tile hierarchical Z** (developed in the CudaRaster /
cuRE / LucidRaster briefs) is the lever, not any micro-optimization of the per-fragment
inner loop. No FreePipe technique relies on a hardware feature WGSL lacks; the only
64-bit-atomic dependency in the broader lineage (Schütz/Dreams-style interleaved
depth+payload `atomicMin`) is the thing FreePipe's own dual-atomic trick — and our
elect-then-store — explicitly route around.
