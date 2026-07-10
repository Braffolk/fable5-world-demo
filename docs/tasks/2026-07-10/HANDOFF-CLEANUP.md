# HANDOFF 00:05 2026-07-10 — perf arc CLOSED (scorecard below), CLEANUP ARC ACTIVE

## WHY (user, 07-10): the cleanup is the PREREQUISITE for the next arc
User verbatim: "after the whole cleanup is done, i will have the biggest refactor demand
yet. hence the cleanup. we couldnt go on with this absolute mess of a branching case and
half-repo-dead-code state into that. for optimal progress, we needed lean codebase!"
The refactor itself is not yet specified — do not speculate; finish the cleanup slices
(incl. S4b deep review, S6 nanite restructure, S7 scene unification) to the full-excision
bar, then ask for the refactor brief.

## THE ACTIVE TASK (user, ~00:00): serious cleanup of the codebase
User verbatim: "a long due cleanup of all unnecessary branching, esp dead, params. not the
ones that are actual quality knobs that are used for different quality settings. but all of
the junk that makes our code very unclean. and ALL of the dead files and non nanite 1 paths
(nanite 1 is the default)".

### Process (agreed shape — inventory → ONE user approval → staged execution)
1. **IN FLIGHT: two read-only inventory agents** (launched ~00:02, results pending):
   (a) URL-param census — every param read site classified QUALITY-KNOB (preset-used;
   verify against the pre-boot LOW/MED/HIGH picker) / TOOLING (cam, fly, census, profile,
   HUD) / LIVE-ESCAPE (this week: ?hwproj = flagship re-land base KEEP; ?culloverlap=0
   keep briefly) / DEAD-EXPERIMENT (delete param + branch: ?middz incremental path
   MEASURED WORSE — delete incl. Scanline.ts branch; ?hw1fetch 3-corner path) /
   TUNING-DEFAULT (fold to constant). Canonical-URL flags (nanodisp clhwmax ksplit fp16w
   ctxsm) = prime fold-to-default candidates. Payload per flag = the dead-side compiled
   branch (file+lines).
   (b) Dead files/exports + non-nanite paths — import-graph proof per dead file; every
   nanite=0 arm mapped (what renders, what dies); Spike*/grassgeo/GroundRing/CSM/splat-saga
   relic checks; forest scene = still the DEV scene, flag don't assume.
2. Merge into ONE kill-list table ranked by lines removed, judgment calls marked → present
   to user for a single approval round.
3. Execute STAGED with gates per slice: fold defaults (behavior-identical) → delete dead
   arms → delete dead files. Gates: tsc + headless smoke at worst pose (scratchpad/smoke.mjs
   pattern) + census counter parity (visTris/mid/clhw/hwTris/splat byte-match) per slice;
   one commit per slice for surgical revert.
4. **(user 07-10) THIRD inventory agent added: tools/ folder audit** — ~85 probe-*.ts
   one-offs + misc scripts; sort by last git edit oldest-first; KEEP-CORE (tools/profile/ +
   tools/perf/interleaved_ab.mjs are untouchable) / KEEP-NICHE / DELETE / JUDGMENT; feeds
   the same kill-list.
### User decisions already made (07-10, pre-approval-round)
- **`?preset` is REMOVED; `?quality` is the single quality axis.** The old
  `preset=low|high|ultra` (Params.ts:36 → WorldConst.qualityConfig() → Heightfield.ts:115,
  world-build grids heightRes/simRes/erosionIters/tileVerts) must be driven by the quality
  tier instead. Mapping: quality low→qualityConfig('low'), medium/high→'high' (today's
  default for everyone); DELETE the 'ultra' arm. ⚠️ low-tier users' world grids change
  (2048/1024/500/49 vs high) — that's the intent, but heightfield output feeds BootCache:
  verify the cache key carries the grid config or bump CACHE_REV.

5. **(user 07-10) FINAL cleanup stage — AFTER all deletion slices land: restructure
   src/nanite/** — reorganize the kept files into a hierarchical, readable folder structure.
   Do NOT start until every other cleanup task is done. Pure moves + import rewrites,
   behavior-identical; same gates (tsc + smoke + counter parity).

## PERF ARC CLOSE-OUT (2026-07-09 blitz — scorecard, all committed)
- nanite-raster HEAD **345426c**: 5e45ae7 round-1 + d9e56ff round-2 + 9bac513 flagship→
  ?hwproj=1 opt-in (trunk-tri regression: two-kernel routing skew; RE-LAND = slot-11 as the
  single routing authority) + 0177364 handoff + 345426c middz default OFF (measured +8%
  regression as ON; per-line: loop collapsed into one 26.6% serial-chain line).
- **Cross-trace method that survived user review** (they rejected raw %-table comparison):
  anchor-ratio normalization — boot-constant anchors (probeGather/PMREM ⇒ T_old/T_new≈1.19)
  + per-frame anchors (traverses/TRAA ⇒ ×frame-ratio ≈1.21) ⇒ divide %new/%old by 1.21,
  ±10% band. EXACT method when thermals differ (user's insight): replay BOTH raw traces
  back-to-back in ONE Xcode sitting (traces carry commands, timing is generated at replay)
  — this laptop can't run two Xcode replays concurrently, hence the anchor math.
- **Per-shader absolute verdicts** (vs 19:48 baseline, worst pose base config):
  clE vertex −55% (96r/80B-spill→56r/0; per-line glue 45.7%→10.9%; remaining fat = 29% ctx
  float decode = the flagship re-land target) · projectVerts −58% (the 32% per-corner
  fallback region GONE — vcompact-at-attach; no big lever left) · HalfResMRT −44% · soup
  vertex −46% · mid: round-1 wins confirmed structurally (ghost line-0 29.9%→gone, atomics
  16.5%→8.5%) but middz serial chain cost +8% net → REVERTED, expect mid well below baseline
  now (unmeasured) · vox pair −7% (scaffold gone, DDA `min` line now TOP at 10-12.7% ⇒
  bitmask-DDA revive condition MET) · resolve mesh+terr −5%, 208r→176/150, now honestly
  material-bound (mx hashes) · RTT +17% unexplained (untouched shader, small, flagged).
- Total frame ≈ −16-19% vs baseline; user live: ~55fps semi-decent full-config scenes
  (was ~40 earlier, CPU-thief-contaminated). 60fps-floor NOT yet proven — needs a cooled
  interleaved session (tools/perf/interleaved_ab.mjs; NEVER bench while anything heavy runs
  — my own run_all cooked one A/B).
- **Next perf levers, data-ranked**: (1) flagship re-land w/ slot-11 single authority
  (−29% of clE vertex + near-flicker cure candidate); (2) bitmask voxel-DDA (top line of
  both vox kernels); (3) USER's classify-time backface reject via sign(area2raw) for
  bark/rock/terrain (~half trunk tris + fixes inside-trunk view); (4) Dawn indirect-draw
  validation ~3.4% (consolidation); (5) shadow zero-VRAM cut-cadence (P3) — sun static
  in-session, needs the far-edge padding proof; P4 (+30MB) VETOED on memory.
- 21:30 near-trunk "specs" artifact = pre-existing, separate, still open
  (near-depth-flicker-bug; cure = flagship re-land's w=1 screen-linear depth).

## Housekeeping to do during/after cleanup
- Worktrees to remove when bisect no longer needed: /private/tmp/laas-bisect-r1 (round-1,
  vite :5174), /private/tmp/laas-bisect-r0 (2b63f2a, vite :5175); kill those vite servers.
- Branch estonia-asset-gen = other session's (stopped; has asset-gen commits + a stray
  duplicate of the hwproj fix); nanite-raster is canonical. asset-gen/ dir untracked — ignore.
- NEVER commit: docs/METAL-PROFILING.md (not ours), docs/todo-human-only.md, asset-gen/.
- Traces on disk (keep until user says drop): /private/tmp/laas_trace_{phase1,phase2,
  postblitz}-*.gputrace (+ -perf exports); results folders profile-results-20260709-
  {194832,204125,233127}.
