# Bog v5: organic hummock-hollow field — design brief (orchestrator, 2026-07-17)

Status: the FOURTH bog attempt, and a deliberate mechanism BREAK from v3/v4.
v3/v4 (Rietkerk/Eppinga anisotropic reaction-diffusion) are PERMANENTLY REJECTED as
the visible surface: they generate thin string RIDGE LINES at ~5 cm amplitude, which
render at ground level as glitchy dark dashes — nothing like a bog. User verdict
2026-07-17: "no proper macro or microdetail, random glitchy looking things." Do NOT
reuse, retune, or restart the Turing/reaction-diffusion family. This brief replaces it.

## What a raised bog actually looks like (ground the form in REAL reference FIRST)

MANDATORY first stage (user law: ground visual forms in real reference — never guess
shapes): look at real raised-bog surface evidence before writing any generator —
photographs of Estonian/Baltic raised-bog surfaces (hummock-hollow-lawn-pool
microtopography) AND any real high-res bog DEM available locally (Moore 2019 1 cm
hummock-hollow grids under `data/`; the Stordalen 5 cm peat DSM exemplar if retained;
search `data/work/microtopography` for peat/moore/stordalen). Extract the REAL
morphometry and record it in the preregistration:
- hummock height distribution (real Sphagnum hummocks are ~0.15-0.5 m tall, not 5 cm),
  width (~0.3-1.5 m), shape (rounded domes, irregular, often coalescing),
- hollow depth/width, lawn extent, and the hummock:hollow:lawn:pool area fractions,
- spacing/clustering statistics, and how the pattern ELONGATES into strings/ridges
  (perpendicular to slope/flow) only in patterned sections.
The visible DOMINANT texture is a dense, bumpy field of rounded mounds — a lumpy
quilt — NOT a network of lines.

## Mechanism (organic hummock field, NOT a pattern-formation PDE)

Primary: a HUMMOCK-HOLLOW HEIGHT FIELD built as a marked-point-process of organic
rounded mounds (or, preferred if it qualifies, real measured micro-relief patches
placed the way the ACCEPTED forest floor did — world-locked blue-noise sites + compact
Wendland-C2 partition; that method is already user-accepted, reuse its machinery from
`forest_exemplar`). Requirements:
1. Hummocks = rounded, irregular raised mounds (superposed smooth dome/Gaussian-ish
   kernels with per-hummock jittered radius/height/asymmetry so no two are identical;
   allow coalescence into larger complexes). NO thin ridges, NO periodic lattice, NO
   single characteristic wavelength.
2. Amplitude REALISTIC and form-attributed (07-17 exactness law: deviate from the 1 m
   parent where the form demands — the parent is fallible and ancestors rederive from
   the accepted fine surface). Target hummock relief ~0.15-0.4 m from the real
   morphometry, NOT the timid 5 cm carrier. Report the deviation field; taper to
   near-zero (p95 <= ~2 cm) over form-free/non-bog ground (DC-drift guard).
3. Wetness conditioning: hummock density/height keyed to a dryness proxy (dome
   position, distance-from-pool, drainage) — dense tall hummocks on drier microsites,
   grading to hollows/lawn to pool margins. Hollows are the low wet interstices.
4. Real ETAK pools (Laugas, already in the cooked water layer, relief-free) are the
   wettest SINKS: hummocks avoid them, hollows grade organically into pool margins
   (this is the pool-margin coupling the park condition wanted — to REAL pools now).
5. Patterned sections: where slope/flow supports it, ELONGATE and ALIGN hummocks into
   organic ridges (strings) perpendicular to flow with flark hollows between — but as
   rounded organic ridges, not thin lines. Isotropic hummock field elsewhere.
6. World-PRF (BLAKE2b `laas-micro-prf1`) for all stochastic placement; storage-chunk
   coords never keys; whole connected mire + >=64 m halo solve, crop core LAST;
   deterministic.

## Gates (freeze concrete numbers in `bundle-preregistration-bog-hummock-v5.json`)

- Safety exact: ETAK pools/open-water relief-free (0 residual), hard/unknown masks
  exact, sealed mire untouched, seams/hierarchy self-consistent.
- Amplitude realistic: hummock relief p50/p95 within the real-morphometry envelope you
  recorded (expected p95 ~0.2-0.4 m), NOT below ~0.1 m (the v4 failure was too flat).
- Organic (anti-artifact): NO periodic spectrum peak, NO single characteristic
  wavelength, NO thin-ridge signature, NO lattice; hummock size/spacing distributions
  match the recorded real morphometry (KS-style check).
- Pool coupling to REAL pools: hollows/low cells adjacent to ETAK Laugas margins;
  hummocks not carved over pools.
- Parent deviation reported (diagnostic, not gated), form-attributed, tapered off-bog.
- ⭐ GROUND-LEVEL QA IS THE REAL GATE (hillshade fooled the orchestrator 3x): the
  implementer MUST render at least 2 low-oblique/ground-level perspective-style QA
  images (sun low, ~1-3 m eye height look-across), not only top-down hillshade, and
  judge honestly whether it reads as a bumpy organic bog vs dashes/lattice. Include a
  before(base)/after(hummock) ground-level pair.

## Pipeline

Reuse the pool-bearing site + base that already works: mire
`etak-component-0004071135`, base manifest `7aa4d523ea99c37d...` (has real Laugas
water, full LOD column), pack core [540224,6429504,540352,6429632] authority (0,83,100)
parent (-1,335,402), preview recipe kind `research-peat-bog-network-preview-v1`
(packer/verifier already site-constant'd to this mire — reuse, just swap the synthesis
float). Freeze v5 preregistration, run synthesis, evaluate gates incl. ground-level QA,
then pin+pack+verify onto the SAME base and report the preview manifest for the
orchestrator to boot at ground level (grass=0).

## Tier & escalation

Design (this brief) is orchestrator/Fable. Implementation from this brief = opus
xhigh. If real bog morphometry says the mechanism here is wrong, STOP and escalate to
the orchestrator — do NOT silently revert to a pattern-formation PDE. A negative/
disappointing ground-level result triggers a premise-audit (is the amplitude right? is
it reading as domes or lines? is the reference real?), not another Turing tune.
