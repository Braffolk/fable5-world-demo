/**
 * perf-review — a GENERIC, reusable multi-agent performance-review workflow (the SKELETON).
 *
 * It encodes only the PROCESS and the DISCIPLINE:
 *   measure → attribute → confirm → ideate → rank → synthesize,
 * with mirage-defense (ablation-not-raw-timers, proposer-never-grades, adversarial confirmation)
 * and structural novelty generation. It contains NO project/task specifics.
 *
 * The MEAT (what the system is, how to measure it, the target, the suspected-but-unconfirmed
 * hypotheses, the hard constraints, the quality bars) is supplied at run time via `args`:
 *
 *   Workflow({ name: 'perf-review', args: {
 *     context:       "<the system + how to measure it (probes, metric keys, dev server, hooks)>",
 *     target:        "<what 'fast enough' means — and the honest metric to judge it by>",
 *     investigation: "<what to find; suspected-but-UNCONFIRMED hypotheses — NOT conclusions>",
 *     constraints:   "<hard feasibility limits the ideas must respect>",
 *     qualityBars:   "<no-regression floors; what counts as 'noticeable quality loss'>",
 *     measure:       "<concrete measurement guidance: commands, sweeps, counters>",
 *     lenses?:       ["<override ideation lenses>", ...],   // optional
 *     analysts?:     5,                                      // optional
 *     measureEnabled?: true                                  // false ⇒ reasoning-only (no device)
 *   }})
 *
 * args may also be a single string (treated as the whole brief). With no args, it warns and
 * runs in reasoning-only mode. Every prompt below references `BRIEF`/`DISCIPLINE` only — the
 * skeleton never names a kernel, a flag, or a domain object.
 */

export const meta = {
  name: 'perf-review',
  description: 'Generic measurement-gated, adversarial multi-agent performance review: find the REAL bottleneck (do not assume one), then generate verified novel ideas. Defeats single-LLM mirage convergence.',
  whenToUse: 'Any deep performance investigation where the bottleneck must be MEASURED not guessed, and you want a ranked portfolio of real + novel levers. Pass the task as args (see header).',
  phases: [
    { title: 'Measure',     detail: '1 harvester runs a serial measurement battery (skipped if measureEnabled=false)' },
    { title: 'Attribute',   detail: 'N analysts decompose the corpus along generic cost axes' },
    { title: 'Confirm',     detail: 'adversarial falsification of each cost-driver claim' },
    { title: 'Ideate',      detail: 'diverse expert lenses generate ideas (novel + dismissed)' },
    { title: 'Rank',        detail: '3 rankers score; ideas that miss a confirmed driver are killed' },
    { title: 'Synthesize',  detail: 'cost map + ranked idea portfolio + red-team critic' },
  ],
}

// ---- assemble the MEAT from args -------------------------------------------------
const cfg = (args && typeof args === 'object') ? args : {}
const BRIEF = (typeof args === 'string' && args.trim())
  ? args
  : ([
      cfg.context       && `CONTEXT (the system + how to measure it):\n${cfg.context}`,
      cfg.target        && `TARGET (what success means + the honest metric to judge it by):\n${cfg.target}`,
      cfg.investigation && `INVESTIGATION (what to find; suspected-but-UNCONFIRMED hypotheses — do NOT treat as conclusions):\n${cfg.investigation}`,
      cfg.constraints   && `HARD CONSTRAINTS (feasibility filter — do not propose violating these):\n${cfg.constraints}`,
      cfg.qualityBars   && `QUALITY BARS (no-regression floors; what counts as "noticeable quality loss"):\n${cfg.qualityBars}`,
      cfg.measure       && `MEASUREMENT GUIDANCE (commands / metric keys / sweeps):\n${cfg.measure}`,
    ].filter(Boolean).join('\n\n')) || (args ? JSON.stringify(args) : '')

if (!BRIEF) log('⚠ perf-review invoked with NO brief — running reasoning-only with whatever the agents can infer. Pass args (see the workflow header) for a real investigation.')

const MEASURE = cfg.measureEnabled !== false && !!BRIEF
const N_ANALYSTS = cfg.analysts || 5

const DISCIPLINE = `MEASUREMENT DISCIPLINE (non-negotiable): prefer ABLATION/SWEEPS and noise-immune COUNTERS over any single raw timer — timers lie (vsync/refresh caps, overlapping spans, thermal drift, CPU-vs-device boundedness). Whoever PROPOSES a change never GRADES it. A WRONG diagnosis dooms everything, so every cost driver must be ADVERSARIALLY CONFIRMED before any idea targets it. Every idea MUST attack a CONFIRMED driver — ideas aimed at a non-bottleneck are killed. Respect the stated hard constraints + quality bars; flag quality-affecting ideas for human sign-off. SERIALIZE access to any single shared device — never run two measurements at once. Measurement happens ONLY in the dedicated Measure/Confirm phases, which run a SINGLE agent at a time.`

// Injected into EVERY parallel/fan-out phase. The single most important rule for correctness:
// concurrent agents must NOT measure, or they collide on the one shared device and everyone reads garbage.
const PARALLEL_RULE = `‼‼ CONCURRENCY — READ FIRST. You are ONE OF SEVERAL AGENTS RUNNING THIS PHASE CONCURRENTLY, RIGHT NOW, IN PARALLEL. You are STRICTLY FORBIDDEN from running ANY probe, benchmark, boot, profiler, or measurement of your own. The measurement device (GPU / dev server) is a SINGLE SHARED resource: if you measure while your sibling agents measure, you ALL collide and EVERY number is WRONG (this has happened — it is the reason for this rule). Work ONLY from the measurement corpus + source code already provided to you. Reading source files is fine; RUNNING anything that touches the device is NOT. If you think a measurement is missing, REQUEST it precisely (name the exact experiment) — the dedicated serial Confirm phase will run it. Do NOT run it yourself.`

// ---- schemas (generic) -----------------------------------------------------------
const CORPUS_SCHEMA = { type:'object', properties:{
  harnessOk:{type:'boolean'}, probeBuilt:{type:'string'},
  rows:{type:'array', items:{type:'object', properties:{ config:{type:'string'}, metrics:{type:'object'}, notes:{type:'string'} }, required:['config','metrics'] }},
  observations:{type:'array', items:{type:'object', properties:{ sweep:{type:'string'}, finding:{type:'string'} }, required:['sweep','finding'] }},
  anomalies:{type:'array', items:{type:'string'}}, failures:{type:'array', items:{type:'string'}}
}, required:['rows','observations'] }

const HYPO_SCHEMA = { type:'object', properties:{
  angle:{type:'string'},
  costDrivers:{type:'array', items:{type:'object', properties:{ name:{type:'string'}, mechanism:{type:'string'}, evidence:{type:'string'}, estimatedSharePct:{type:'string'}, confidence:{type:'string'}, howToFalsify:{type:'string'} }, required:['name','mechanism','evidence','howToFalsify'] }},
  openQuestions:{type:'array', items:{type:'string'}}
}, required:['angle','costDrivers'] }

const CONFIRM_SCHEMA = { type:'object', properties:{
  experimentsRun:{type:'array', items:{type:'string'}},
  confirmedDrivers:{type:'array', items:{type:'object', properties:{ name:{type:'string'}, verdict:{type:'string'}, sharePct:{type:'string'}, evidence:{type:'string'} }, required:['name','verdict','sharePct'] }},
  refuted:{type:'array', items:{type:'string'}}, notes:{type:'string'}
}, required:['confirmedDrivers'] }

const IDEA_SCHEMA = { type:'object', properties:{
  lens:{type:'string'},
  ideas:{type:'array', items:{type:'object', properties:{
    title:{type:'string'}, mechanism:{type:'string'}, attacksDriver:{type:'string'},
    expectedMagnitude:{type:'string'}, novelty:{type:'string'}, qualityImpact:{type:'string'},
    integrationCost:{type:'string'}, constraintsRespected:{type:'string'}, howToValidate:{type:'string'}, risk:{type:'string'}
  }, required:['title','mechanism','attacksDriver','novelty','qualityImpact'] }}
}, required:['lens','ideas'] }

const RANK_SCHEMA = { type:'object', properties:{
  rankerAngle:{type:'string'},
  ranked:{type:'array', items:{type:'object', properties:{ title:{type:'string'}, score:{type:'number'}, attacksConfirmedDriver:{type:'boolean'}, verdict:{type:'string'}, why:{type:'string'} }, required:['title','verdict','why'] }},
  topPicks:{type:'array', items:{type:'string'}}
}, required:['rankerAngle','ranked'] }

const FINAL_SCHEMA = { type:'object', properties:{
  whatsActuallyHittingPerf:{type:'string'},
  costMap:{type:'array', items:{type:'object', properties:{ driver:{type:'string'}, sharePct:{type:'string'}, evidence:{type:'string'} }, required:['driver','sharePct'] }},
  ideaPortfolio:{type:'array', items:{type:'object', properties:{ title:{type:'string'}, mechanism:{type:'string'}, attacksDriver:{type:'string'}, expectedMagnitude:{type:'string'}, novelty:{type:'string'}, qualityRisk:{type:'string'}, validateBy:{type:'string'}, rank:{type:'number'} }, required:['title','mechanism','attacksDriver','novelty','rank'] }},
  recommendedNextExperiments:{type:'array', items:{type:'string'}},
  openQuestionsForUser:{type:'array', items:{type:'string'}}
}, required:['whatsActuallyHittingPerf','costMap','ideaPortfolio','recommendedNextExperiments'] }

const CRITIC_SCHEMA = { type:'object', properties:{
  missing:{type:'array', items:{type:'string'}}, failureModes:{type:'array', items:{type:'string'}},
  strengtheningEdits:{type:'array', items:{type:'string'}}, verdict:{type:'string'}
}, required:['missing','strengtheningEdits','verdict'] }

// ---- MEASURE ---------------------------------------------------------------------
let corpus = { rows: [], observations: [], note: 'measurement skipped (measureEnabled=false or no brief)' }
if (MEASURE) {
  phase('Measure')
  corpus = await agent(
    `${BRIEF}\n\n${DISCIPLINE}\n\nYOU ARE THE MEASUREMENT HARVESTER. Using the system + measurement guidance above, build/extend the project's probe harness as needed and run a battery of measurements SERIALLY to characterize WHERE the time actually goes. Design sweeps that SEPARATE the cost axes: (a) fixed cost per work-item vs cost that scales with data/elements; (b) compute-bound vs memory/bandwidth-bound (e.g. vary resolution/size at fixed work); (c) serialization/contention; (d) wasted/over-produced work. Reproduce the known baseline first to prove the harness, then sweep. Emit a clean CORPUS: every row (config + metrics), per-sweep OBSERVATIONS (the TREND, not just numbers), anomalies, and failures. Be honest about run-to-run noise. If no usable harness exists, say so plainly and report what you could infer. Return structured.`,
    { label:'harvester', phase:'Measure', schema: CORPUS_SCHEMA })
  log(`Measure: ${corpus && corpus.rows ? corpus.rows.length : 0} rows`)
}
const corpusJson = JSON.stringify(corpus)

// ---- ATTRIBUTE -------------------------------------------------------------------
phase('Attribute')
const ANGLES = [
  { key:'fixed-vs-scaling', focus:'FIXED PER-UNIT OVERHEAD vs SCALES-WITH-DATA: is cost dominated by a fixed per-work-item overhead (proportional to item count, ~invariant to data size), or by work that scales with the data/pixels/elements? Quantify the per-item cost and the invariant fraction.' },
  { key:'compute-vs-bw', focus:'COMPUTE-BOUND vs MEMORY/BANDWIDTH-BOUND: does cost track arithmetic or data movement? Use size/resolution sweeps at fixed work-item count to separate them.' },
  { key:'serialization', focus:'SERIALIZATION / CONTENTION / SYNC: is throughput limited by contended shared state (atomics, locks), poor occupancy/parallel-utilization, or sync barriers?' },
  { key:'wasted-work', focus:'WASTED / OVER-PRODUCED WORK: is work computed then discarded, recomputed redundantly, or emitted beyond what the output needs? Is the quantity of work itself reducible at no quality cost?' },
  { key:'boundedness', focus:'MEASUREMENT TRUST + TRUE BINDING RESOURCE: is the headline metric honest (overlap, caps, thermal), and what ACTUALLY gates wall-time here (the device, the host/submit, a serial dependency)? Flag any way the diagnosis itself could be a mirage.' },
].slice(0, N_ANALYSTS)
const hypotheses = await parallel(ANGLES.map(a => () => agent(
  `${PARALLEL_RULE}\n\n${BRIEF}\n\n${DISCIPLINE}\n\nMEASUREMENT CORPUS (JSON):\n${corpusJson}\n\nYOUR ATTRIBUTION ANGLE: ${a.focus}\n\nGround in BOTH the corpus numbers (already measured for you — do NOT re-measure) AND the real implementation (READING source is allowed; running anything is not). Output cost-driver hypotheses, each with the EVIDENCE it rests on, an estimated % share of the hot cost, a confidence, and a CHEAP falsifying experiment to REQUEST (do not run it). Be quantitative. Return structured.`,
  { label:`attr:${a.key}`, phase:'Attribute', schema: HYPO_SCHEMA }
))).then(r => r.filter(Boolean))
log(`Attribute: ${hypotheses.length} angles`)
const hypoJson = JSON.stringify(hypotheses)

// ---- CONFIRM (adversarial) -------------------------------------------------------
phase('Confirm')
const confirmed = await agent(
  `${BRIEF}\n\n${DISCIPLINE}\n\nMEASUREMENT CORPUS:\n${corpusJson}\n\nANALYST HYPOTHESES (with falsification experiments):\n${hypoJson}\n\nYOU ARE THE ADVERSARIAL CONFIRMER. ${MEASURE ? 'Run the CHEAPEST DECISIVE falsifying experiments SERIALLY to' : 'Reason from the provided data/code to'} CONFIRM or REFUTE each candidate cost driver — your job is to KILL wrong diagnoses. Resolve analyst conflicts with evidence, not a vote. Output the CONFIRMED cost-driver map: for each driver a verdict (confirmed/refuted/inconclusive), an evidence-backed % share, and what the evidence showed. State plainly whether the headline metric is trustworthy and what truly binds wall-time. Return structured.`,
  { label:'confirmer', phase:'Confirm', schema: CONFIRM_SCHEMA })
log(`Confirm: ${confirmed && confirmed.confirmedDrivers ? confirmed.confirmedDrivers.length : 0} drivers`)
const confirmedJson = JSON.stringify(confirmed)

// ---- IDEATE (open-ended; no solution seeding) ------------------------------------
phase('Ideate')
const DEFAULT_LENSES = [
  'Hardware / runtime specialist — what does this specific platform/runtime reward (occupancy, memory hierarchy, contention avoidance, instruction mix)?',
  'Algorithms & data-structures researcher — recent SOTA techniques for this class of problem; better complexity, batching, dedup, sorting, hierarchy.',
  '"Do less" specialist — eliminate, approximate, cache, precompute, or amortize the work entirely; what need not be done at all?',
  'First-principles — question the architecture itself: if you started clean knowing the confirmed cost map, what shape would the hot path take?',
  'Cross-domain analogy — what do adjacent fields do for an analogous bottleneck, and what transfers under the stated constraints?',
  '"What have we dismissed?" — READ the project\'s own docs/notes for shelved/deferred/under-explored ideas that, given the confirmed cost map, may have been dropped for the wrong reason.',
  'Domain specialist for this exact problem — the deepest practitioner\'s tricks that survive the hard constraints and quality bars.',
]
const lensList = (cfg.lenses && cfg.lenses.length ? cfg.lenses : DEFAULT_LENSES)
const ideaSets = await parallel(lensList.map((focus, i) => () => agent(
  `${PARALLEL_RULE}\n\n${BRIEF}\n\n${DISCIPLINE}\n\nCONFIRMED COST-DRIVER MAP (what is ACTUALLY hitting perf — measured + adversarially verified):\n${confirmedJson}\n\nYOUR EXPERTISE LENS: ${focus}\n\nGenerate the strongest ideas to hit the target with NO noticeable quality loss. RULES: (1) Do NOT default to any single pre-existing plan — generate from the measured cost map. (2) Every idea MUST name which CONFIRMED cost driver it attacks; ideas attacking a non-bottleneck are killed. (3) Actively include UNTESTED/novel ideas and ones that may have been DISMISSED for the wrong reason. (4) Respect the hard constraints + quality bars. For each idea: mechanism, the driver it attacks, expected magnitude (grounded in the cost map), novelty (novel / known-but-ignored / on-roadmap), quality impact, integration cost, how to VALIDATE with the measurement harness. Up to ~4 ideas. Quality over quantity. Return structured.`,
  { label:`ideate:${i}`, phase:'Ideate', schema: IDEA_SCHEMA }
))).then(r => r.filter(Boolean))
const allIdeas = ideaSets.flatMap(s => (s.ideas||[]).map(i => ({ ...i, lens: s.lens })))
log(`Ideate: ${allIdeas.length} ideas from ${ideaSets.length} lenses`)
const ideasJson = JSON.stringify(allIdeas)

// ---- RANK ------------------------------------------------------------------------
phase('Rank')
const RANKERS = [
  { key:'ev', focus:'EXPECTED-VALUE: rank by expected cost reduction grounded STRICTLY in the confirmed cost map. KILL any idea whose mechanism does not attack a confirmed bottleneck. Reward attacking the largest confirmed share.' },
  { key:'novelty-quality', focus:'NOVELTY + QUALITY: reward genuinely new or unfairly-dismissed ideas over rehashes; penalize anything risking a quality-bar breach; flag ideas needing human sign-off.' },
  { key:'feasibility', focus:'FEASIBILITY + MIRAGE-RISK: rank by platform feasibility under the hard constraints, integration cost, and whether the win would be REAL + cleanly MEASURABLE vs hard-to-attribute.' },
]
const rankings = await parallel(RANKERS.map(r => () => agent(
  `${PARALLEL_RULE}\n\n${BRIEF}\n\n${DISCIPLINE}\n\nCONFIRMED COST MAP:\n${confirmedJson}\n\nIDEA PORTFOLIO (JSON):\n${ideasJson}\n\nYOU ARE A RANKER. ${r.focus}\n\nScore EVERY idea, give a verdict (promote / maybe / kill) with a concrete reason, set attacksConfirmedDriver true/false, and list your top picks. Be adversarial — most ideas should not be 'promote'. Return structured.`,
  { label:`rank:${r.key}`, phase:'Rank', schema: RANK_SCHEMA }
))).then(r => r.filter(Boolean))
const rankJson = JSON.stringify(rankings)

// ---- SYNTHESIZE + CRITIC ---------------------------------------------------------
phase('Synthesize')
const final = await agent(
  `${BRIEF}\n\n${DISCIPLINE}\n\nCONFIRMED COST MAP:\n${confirmedJson}\n\nMEASUREMENT CORPUS:\n${corpusJson}\n\nIDEA PORTFOLIO:\n${ideasJson}\n\nRANKINGS (3 diverse rankers):\n${rankJson}\n\nSynthesize the deliverable: (whatsActuallyHittingPerf) a crisp evidence-backed answer with the numbers; (costMap) confirmed drivers with % shares + evidence; (ideaPortfolio) the merged, de-duplicated, RANKED idea list — each flagged novel / known-ignored / on-roadmap, tied to the driver it attacks, with expected magnitude, quality risk, and how to validate; promote only ideas attacking a confirmed bottleneck; (recommendedNextExperiments) the 2-4 highest-EV things to build+measure NEXT; (openQuestionsForUser) genuine user decisions (esp. quality-vs-speed tradeoffs). Be honest about uncertainty. Return structured.`,
  { label:'synthesize', phase:'Synthesize', schema: FINAL_SCHEMA })

const critic = await agent(
  `${BRIEF}\n\n${DISCIPLINE}\n\nPROPOSED DELIVERABLE (JSON):\n${JSON.stringify(final)}\n\nMEASUREMENT CORPUS:\n${corpusJson}\n\nYou are the red-team critic. Be ruthless and specific. Is the DIAGNOSIS sound or could it be a measurement mirage? Does any promoted idea actually attack a NON-bottleneck? Does any idea quietly breach a quality bar? Is any expected-magnitude claim unsupported by the cost map? What is MISSING — a cost driver not measured, an idea-space not explored, a confound? Give concrete strengthening edits. Return structured.`,
  { label:'critic', phase:'Synthesize', schema: CRITIC_SCHEMA })
log(`Synthesize done — critic verdict: ${critic && critic.verdict ? critic.verdict : 'n/a'}`)

return { final, critic, confirmed, corpus, rankings, ideaCount: allIdeas.length }
