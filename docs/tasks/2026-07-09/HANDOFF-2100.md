# HANDOFF 21:00 2026-07-09 — deadline MIDNIGHT: baseline floor + re-enable & optimize shadows+grass

## GOAL (user): solid 90fps FLOOR (p95 ≤ 11.1ms) grass+shadows OFF, then re-enable BOTH and optimize
## them too — ALL BY MIDNIGHT. Judge ONLY by live p95 at the worst pose. NO more Xcode exports tonight
## (each costs the user ~30 min). No VRAM hogs. User grants autonomous implementation.

## Worst pose (all measurement): `?cam=-582.1,302.4,1006.1,2.5692,-0.0077` — last read 28ms p95 (base config).

## COMMITTED STATE (branch nanite-raster, HEAD 2b63f2a)
- f0cc6a6 easy-mid levers · 4e51603 crown-LOD (mid 13.3M→~3M) · 81e5659 census tooling · 54cb71e/f821dcb
  cap right-sizing (−594MB total) · 6ce1cdc+e245c1f voxel bug family SOLVED · 9cb3014 F3 cleanup ·
  a9a3a7a **HW vertex Phase 1** (ctx-read + _clE/_clT class split; 15.66→~13.3%, regs 96→80, spill stays) ·
  2b63f2a **Phase 2 REVERTED** (wash: projection ballooned 7.66→12.78% projecting per-corner terrain HW
  clusters while _clE only fell to 8.43% w/ spill intact — the compiled fallback kept the ceiling; retry
  conditions documented at the Project.ts slot-11 skip) + projectVert i32-saturation bug-fix kept.

## CURRENT PROFILE (post-Phase-2 capture; POST-REVERT expect ~: clE back ~12.9%, projection back ~7.7%)
mid 13.15% (64r, loop-dominated: line-0 occupancy 29.9%, scanline ~36%, election ~17%, setup only ~10%)
· projectVerts (7.7% post-revert, honest gather) · clE vertex ~12.9% (80r/80B spill = 63% glue; the whale)
· clE frag 7.81% · probeGather 7.09% (sample-latency) · HalfResMRT 4.72% (6127 instr) · voxScatterB1 4.50%
· resolve_tri 3.93% (208r!) · traverses ~5.3% · PMREM 2.47% (likely boot-in-window artifact, not steady)
· Dawn indirect-draw VALIDATION ≈5% of sampled (consolidation lever, someday) · parked: nanVoxScatterFan
78% tint-loop overhead; MaterialX Worley hot; bitmask-DDA idea (docs/tasks/2026-07-09/IDEAS.md).

## PENDING (user delivering pre-compact): live p95 at worst pose (1) FULL config (shadows+grass ON),
## (2) full config + `&middz=1` (mid incremental-depth A/B — flip default if wins + no artifacts).

## PROCESS FOR THE REMAINING TIME (user proposal, shaped):
User wants a parallel blitz: every shader >4% gets a proposal agent at once. AGREED with two corrections
learned TODAY (Phase-2 wash): (a) parallel = PROPOSALS ONLY (read-only, each agent gets its shader's
per-line file from profile-results-20260709-194832/204125 + MSL + source + the laws below); implementation
is SERIALIZED by me in dependency order after a fast audit — shared raster/ files WILL collide otherwise;
(b) every proposal must state its measured-cost mechanism (which per-line bucket it removes) — no
estimate-only washes. Fable-5 agents authorized by the user for these proposals.
LAWS to embed in every prompt: no VRAM hogs; no noise/dither; no new URL knobs (disable-only ok); no
refusion; world1 stays 56 regs; drop-old-behavior + git undo; p95 eyeball gates; tsc green; no commit
until measured.
TARGETS for the blitz (>4%): _clE vertex+frag pair (the whale — the known lever: Gate-2 soup-routing of
near-crossers to kill the compiled fallback = the audited Finding-B fix; do NOT re-add HW projection
without terrain dedup), nanMidRaster (loop+occupancy: middz A/B pending, occupancy 64r→?; trihzb gated),
probeGather, HalfResMRT, voxScatterB1(+B0, 145r), resolve_tri (208r), + SHADOWS and GRASS once the full-
config reading identifies their real cost (priors: shadows 5-8ms moving, grass ray +4.4-6.3ms; grass
levers listed in task #55; shadow levers in task #61/70).

## MEASUREMENT LOOP: user reads p95 live at the worst pose per change batch (10s each). Headless smoke
## (playwright pattern in this session: waitForFunction(fn, undefined, {timeout}) + screenshot) for
## correctness before asking. Census (?census=1) for tri-budget checks only if needed.

## 21:30 READINGS (pre-compact) — INSTRUMENT IS THERMALLY COOKED
- Full config: p95 30-32ms then 51ms on rerun (SAME pose) ⇒ thermal throttle, ±10ms noise/run. middz run: 45-50ms.
- ⇒ tonight: optimize against the STABLE Xcode profile %s + census tri-budget; p95 = coarse FINAL gate only
  (cooled machine, median of 3). NO per-change p95 decisions.
- middz: INCONCLUSIVE (noise) — stays default-off.
- User sees artifacts "super up close to tree trunks" BOTH runs — most likely the KNOWN long-standing
  near-camera depth-flicker (memory: near-depth-flicker-bug; Phase 2 was the candidate cure, reverted).
  IF user says they look NEW/different in character → bisect vs f0cc6a6 FIRST.
- Full-config tri budget: 18M submit / 1.8M culled-resid (vs 15M base) — shadows+grass add ~3M submit; the
  full-config frame cost split (shadows vs grass ms) is STILL UNMEASURED (profile was base config).
