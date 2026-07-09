# Apple/mobile-GPU perf — implementation changelog (task #72/#73)

Changes land here as the audits (DEEP-FOREST-AUDIT, MASTER-AUDIT, WEBGPU-FEATURES) surface levers.
Order = simplest-change-biggest-win first. Complex pipeline changes → QUEUE (bottom), user-discussed.

**Rules (user, 2026-07-05):** implementation subagents are CODE-ONLY — NO GPU probes / benches /
dev-server / playwright (verification is serial + done by the orchestrator, never parallel). New
WebGPU features must be runtime feature-DETECTED with graceful fallback. Keep code clean. Each
change atomic + documented here.

Deep-forest cost model (2026-07-05 ablation, deep-forest-breakdown.md): base 17-21ms (grass off) =
~13ms spread per-covered-pixel floor (terrain/bark/raster/resolve/cull) + leaves/crownlod0 ~4ms +
shadows ~0.5ms (cached). Grass ~5ms on top. `c.nanVisClear` big timestamp = OVERLAP ARTIFACT (not
a real clear cost). Apple render||compute OVERLAP → ablate for truth, never per-pass timestamps.

| # | change | subsystem | file:line | expected | status | verify |
|---|---|---|---|---|---|---|
| 1 | bake `widenT` into dead guide-ctx word `base+3`, unpack in march (kill per-step 2×pow+sqrt+div) | grass | NaniteGrass.ts :501(src)/:526(bake)/:931(march) | <1ms grass (occupancy relief) | ✅ VERIFIED (renders clean, 0 shader errors) | boot-sanity PASS; perf <1ms unmeasured (thermal-noisy), algebraically sound |

### L2 measured — SHELVED, do not flip (2026-07-05)
`?respass=0` (two-pass forest resolve) A/B/A at the eye-level deep pose, grass off: A default 16.9 /
B two-pass 21.0 / A2 default 21.4 — **drift A2−A = 4.5ms (thermal) > the 4.1ms "signal"**; adjacent
equal-thermal pair (B 21.0 vs A2 21.4) = 0.4ms = NOISE. And the übershader occupancy cliff the audit
cites is a *close-up-inside-crown* phenomenon (37.5ms in a crown), NOT triggered at this eye-level
pose. ⇒ NOT a clean win here; flipping the default is UNJUSTIFIED. Needs a thermal-controlled A/B at
an INSIDE-CROWN pose to see if the cliff-recovery is real — and even then the default-flip is a
pose-dependent tradeoff = USER CALL. Method note: cross-boot A/B needs between-boot GPU cooldown
(drift 4.5ms otherwise); measureFrames cooldownMs is per-frame only.

### Measurement scaffolding (2026-07-05)
`?visclear=0` DEBUG flag (NaniteRaster.ts:533 decl / :553 gate) — build-time skips the 2 hot
full-screen vis clears (payloadV + visBV) in `kVisClear` for the world path, so the orchestrator
can A/B whether the ~11ms `c.nanVisClear` timestamp is real store cost or the **L3 render‖compute
WAR-stall** artifact. Produces a DIRTY/garbage render — timing-only. Mirrors the existing
`skipDepthClear` build-time gate; default (flag absent or ≠'0') = byte-identical to today. NOT a
product change — this is the gating ablation named in L3 below ("serial no-op-clear A/B").

## Triaged lever queue (from DEEP-FOREST-AUDIT L1-L8 + WEBGPU-FEATURES)

**SIMPLE — do now (code-only + serial verify):**
- ✅ #1 widenT bake (grass) — IMPLEMENTED, verifying.
- ▶ #2 **L2 respass=0 forest default flip** — two-pass resolve strips the übershader worst-case VGPR
  → occupancy recovery on the >90% voxel-path pixels. HIGH/CONFIDENT. NaniteResolve.ts:1574-1591.
  Serial A/B first (default vs ?respass=0 at deep pose); flip default if it wins.
- #3 **L8 gate `trackTimestamp` behind `?gpuprof`** + stagger `meterRead` readbacks — free prod win
  (~0.5-1ms + p95). Engine.ts:86,231-251; NaniteFrame.ts:667-694. (⚠️ our harness needs ?gpuprof to
  read timestamps after this.)
- #4 **L7 shadow half rg32f→rg16f/packed** — Mali FP32-texture 2× cost. NaniteShadowHalf.ts:83-85,168-181.
- #5 **depthV lazy-alloc** (world path never reads it) — ~13MB mem, NaniteRaster.ts:203-218.

**MEDIUM — after simple (some need a gating ablation first):**
- ❌ L3 **ping-pong vis buffers** — design A (`?visping` resolve-snapshot) BUILT + GPU-VERIFIED =
  **PERF-DEAD** (~0.1ms recovery vs 2.5ms expected: 16.4 vs 16.5 default = noise); **REVERTED**. Root
  cause: the 2.9ms is the **atomicStore-per-pixel CLEAR's own compute cost** (not a WAR-stall /
  resolve-decoupling hazard) → real fix = **DMA `clearBuffer`** (replace the two `atomicStore`
  full-screen clears in `kVisClear` with a native buffer clear), NOT ping-pong. The earlier
  `?visclear=0` A/B (saves 2.5-2.9ms) proved the clear is EXPENSIVE but NOT that it's a hazard; design A
  decoupled the resolve from the clear (resolve reads a ping-ponged snapshot, next-frame clear writes
  the live set) and recovered ~0ms — isolating the cost to the clear kernel itself. (B) FULL DOUBLE not
  attempted. Code reverted to pre-visping HEAD (NaniteFrame.ts matches HEAD; typecheck clean).
  ⚠️ **SCOPE (2026-07-05 code audit) — the vis-buffer consumer map (kept below as reference).**
  The vis set (payloadV/visBV/depthV) is bound at BUILD time via `storage(attr)` into SIX camera-path
  consumers — you cannot "thread `cur`" per frame, each must be built against the set it uses:
  (1) raster kVisClear+world1+scar WRITE (NaniteRaster.ts:547-569,602-612); (2) grass emitPx/kRay R+W
  (NaniteGrass.ts:316-320,808,822); (3) HZB level-0 READ payloadV (NaniteFrame.ts:222→NaniteHzb.ts:158);
  (4) shadowHalf kHalf READ payloadV (NaniteShadowHalf.ts:99); (5) resolve tri+vox meshes READ
  visBV/payloadV (NaniteResolve.ts:401,424-425,1555); (6) probe READ depthV/payloadV, debug
  (NaniteFrame.ts:432,439). The ONLY cross-queue (fragment) reader = the RESOLVE; the other 5 are
  compute (same queue as the writes → no render‖compute WAR). Interlock: the cull bakes
  `hzb.sphereOccluded` (reads the SEPARATE hzbF) at NaniteFrame.ts:235 — doubling the HZB strands the
  cull on one parity ⇒ needs a shared-hzbF refactor. → TWO clean designs, USER/design call:
  - **(A) RESOLVE-SNAPSHOT (smallest/safest) — ❌ BUILT + GPU-VERIFIED PERF-DEAD, REVERTED:** kept the
    whole live vis set + all 5 compute consumers SINGLE and byte-identical; added a 2-entry snapshot of
    ONLY {payloadV,visBV} (NOT depthV — the resolve never reads it), one compute copy pass writing
    live→snapshot[frame&1] AFTER world1+grass+scar, and built the resolve TWICE (one per snapshot
    parity) toggled `.visible` by `frame&1`. Rendered clean but recovered **~0ms** — the "hazard" it
    decoupled wasn't real; the clear's cost is intrinsic (atomicStore per pixel), not a stall behind the
    resolve. Reverted. (Residual, now moot: shadowHalf tex + grass rayNrmTex are also single,
    fragment-read by the resolve.)
  - **(B) FULL DOUBLE:** ping-pong the whole set; build ALL 6 consumers per parity + share one hzbF
    (NaniteHzb refactor) + share the grass guide (else 2× the per-frame guide bake). Recovers the full
    2.9ms (no copy) but doubles the two HEAVIEST shaders (resolve, grass) + rewires hzb/grass/cull —
    large, interlocked, high broken-render risk under CODE-ONLY.
  BOTH require doubling the resolve material (2 pipelines) + a per-frame scene-mesh swap. **(A) was BUILT
  + GPU-verified PERF-DEAD and REVERTED** (see the ❌ L3 line above — the WAR-stall premise was wrong;
  the 2.9ms is the clear kernel's own atomicStore cost). (B) is now moot for the same reason — doubling
  the vis set cannot recover a cost that lives in the clear, not in a resolve-coupling hazard. The
  productive lever is **DMA `clearBuffer`** for the two `kVisClear` full-screen atomicStore clears.
- subgroup **queue-compaction in cull** (~32× atomic cut; subgroups stable Chrome 134, feature-gated).
- L5 **post: contact half-res + temporal GTAO**. Gate: post ablation at the deep pose first.
- L6 **shadow MOVING strip re-raster + shvox2 cadence** (the stutter). Gate: moving-pose ablation.

**COMPLEX / USER-GATED — QUEUE, discuss when back:**
- L1 **leaf-crown whale** (coarsen leaf-crown DAG / push voxnear in) — the ~8ms whale, but quality-
  fragile + crownlod0 is a user mandate + lushness law. Shotdiff-gated experiment first.
- L4 **voxel megakernel split** — gated on a Metal GPU capture (user's Xcode) confirming spill/occupancy.
- **fp16** hot kernels (grass march / voxel scatter / resolve) — shader-f16 IS available (Chrome 120,
  device auto-requests) but three TSL has no half node → needs raw-WGSL FunctionNode per kernel. Complex.
- **analytic-blade grass lane** (MASTER-AUDIT L7, 2.36ms) — lane redesign vs march directive + lushness law.

## Notes
- Verification protocol: typecheck (no GPU) after every change; a SINGLE serial GPU shotdiff/perf
  bench by the orchestrator only where visual/perf risk. Never parallel benches.
- Commits: atomic per change on the branch (no push), documented here.
