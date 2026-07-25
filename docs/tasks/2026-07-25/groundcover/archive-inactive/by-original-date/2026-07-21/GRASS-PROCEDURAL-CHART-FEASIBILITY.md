# Procedural botanical-chart feasibility gate

Date: 2026-07-22  
Decision: **reject and park as the all-angle live visibility carrier**  
Scope: offline mathematics and actual-mesh measurement only; no shader or
runtime file was changed.

## Question and hard contract

Can the deterministic procedural provenance in
`tools/groundcover-bake/EstonianGraminoids.ts` replace triangle identity with a
small stable botanical chart token, so that one or a few fixed reads identify
the exact live surface and a fixed closed-form transform evaluates it?

The proposed query had to satisfy all of the following at once:

1. no live mesh traversal, loop, march, candidate list, species loop, extra
   pass, or distance-dependent work;
2. correct angular disocclusion, not merely a good fit to source triangles;
3. correct successor when the camera moves along the same oriented line,
   including starts inside the cover and exact/near-horizontal rays;
4. no more than `51,121,152` resident bytes and one/few fixed reads; and
5. an offline union of overlapping species, populations, moss, and other cover,
   rather than one live query per species.

The gate fails conditions 2--4 on the accepted Calamagrostis source. The
procedural chart partition is real and reusable offline, but it is not a
complete visibility representation.

## Immutable inputs and reproduction

- Accepted GCRP/v4 source:
  `src/assets/groundcover/calamagrostis-canescens.gcrp`, SHA-256
  `2ed57f59d86e83762d779130874347085eaa9271bc4963a0046e0b7ff641b05c`.
- Procedural generator: `tools/groundcover-bake/EstonianGraminoids.ts`, SHA-256
  `b430c03c4c8d24e545923bd20fbeecf10525889df07c53dc62d299549c0531e9`.
- Accepted generator mesh SHA-256:
  `37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0`.
- Generator/source ledger:
  `tools/groundcover-bake/ESTONIAN-GRAMINOIDS-README.md`, SHA-256
  `4176c589d33f1d0d45cba47ced0dce0cf3602776a48f18295899a91d26d1ee6e`.
- Accepted QA index SHA-256:
  `7243b8cb6fee45da05ee2d578cff8501963be3f255d82151f3d38a704edc5387`.
- Evaluator:
  `tools/groundcover-bake/analyze-procedural-chart-feasibility.ts`, SHA-256
  `c715feb7d9d6eb5ccdeb6916f03ce5afc1458a5093b8cfd94519ac91cc761af0`.
- Recipe SHA-256:
  `cc8609cd49533b5bd03f3b9f23813e11c7bf28d74d457055947bbe2ebeea3215`.
- Final artifact root:
  `data/work/groundcover-procedural-chart-feasibility/2ed57f59d86e8376/cc8609cd49533b5b/`.
  `metrics.json` SHA-256 is
  `fac690a8e28218bd5429efad7e12ac3689c9e38fd11fcd30e6cd7af324f691ae`;
  `index.json` SHA-256 is
  `ded016a2532af2b5d89778a5c2d7e3b28a1b523b37a05867f2d6677fa65277eb`.

Reproduce from the repository root with:

```sh
npx tsx tools/groundcover-bake/analyze-procedural-chart-feasibility.ts
```

The generator was rerun deterministically and its emitted triangle indices were
compared with the packed GCRP: there were zero mismatches. The audit recognizes
the generator's exact leaf-root-to-tip colour interpolation before grouping;
the foliage result is therefore not an accidental palette split.

## What the botanical chart partition proves

Triangles were grouped by exact generator semantic family and shared indexed
edge connectivity. Shared anchor vertices do not merge separate primitive
calls. This yields `270,535` charts, every one a contiguous triangle range:

| procedural family | charts | triangles | triangles/chart |
|---|---:|---:|---:|
| foliage ribbon | 102 | 1,914 | 19 |
| culm tube | 6 | 336 | 56 |
| rhizome tube | 5 | 150 | 30 |
| panicle-axis tube | 20,969 | 450,280 | p50 24, max 200 |
| spikelet surface | 18,478 | 240,214 | 13 |
| callus hair or filament | 203,258 | 1,117,919 | p50 4, max 13 |
| anther surface | 27,717 | 360,321 | 13 |

These counts reconstruct the generator, rather than merely renaming triangles.
For example, `20,969 = 6` rachises `+ 2,485` recursive branches `+ 9,239`
pedicels `+ 9,239` spikelet axes; `18,478 = 2 * 9,239` paired glumes; and
`27,717 = 3 * 9,239` anther surfaces. The chart ID therefore denotes a real
procedural primitive instance. It requires 19 bits instead of the source's
triangle identity.

This is useful offline provenance. It does **not** imply that one chart remains
the visible winner under a change of ray.

## Decisive angular-disocclusion failure

Let `C(q)` be the exact first-hit chart of live ray `q`. Let `A(q)` be the set
of chart tokens stored at the four surrounding spatial/angular canonical
samples. Any one-token selector, separator, or chart-local evaluation using
only those records has the necessary condition

```text
C(q) in A(q).
```

When this condition is false, no interpolation of local chart coordinates can
produce the newly disoccluded chart: its identity and local surface record are
absent.

The actual-mesh audit proposed **more** directions than the current carrier:
16 azimuths at elevations `5, 15, 35, 55, 75` degrees plus vertical, 81
directions total. It tested exact live midpoint views at
`10, 25, 45, 65, 82.5` degrees on an `8 x 8` phase grid: 5,120 live rays and
2,508 exact truth hits.

- The exact live winner chart appeared among the four corner tokens for only
  **0.837320574%** of truth hits.
- The exact live winner triangle appeared among those corners for **0%**.
- Even the broader semantic family appeared for only `44.7368%`.
- The nearest token had the exact winner chart for only `0.279107%`.
- The chart support remained between `0.5051%` and `1.4706%` at every tested
  live elevation; this is not only a grazing-angle defect.

Thus the four-token field is missing the live categorical event on more than
99% of truth hits. Chart-local UVs, an intra-cell separator, a 5-degree view, or
a better analytic intersection cannot recover absent disocclusions. A larger
fixed candidate tail would merely reintroduce the rejected candidate-list
cost; it is not authorized by this result.

## Decisive same-line/camera-path failure

An exterior oriented line does not by itself answer a camera-inside query. For
an ordered marked event sequence `E(ell) = (e_0, e_1, ...)`, a camera at line
parameter `s` needs the successor

```text
S(ell, s) = first e_i with parameter greater than s.
```

Replacing triangle marks by chart marks does not remove the dependence on
`s`. Across 384 exact actual-mesh lines, including a 155 m horizontal horizon
and signed `0.1, 1, 5, 15` degree directions:

- events/line: p50 `10`, p95 `1,077`, maximum `4,481`;
- distinct charts/line: p50 `5`, p95 `481`, maximum `692`;
- chart runs/line: p50 `5`, p95 `723`, maximum `4,182`;
- at `0.1` degrees, distinct-chart p50 is `166` downward and `162` upward;
  p95 is `670` and `665` respectively.

The horizontal distinct-chart count is deceptively small because the periodic
tile repeats the same chart ID. It still reaches 4,182 chart runs: an exact
successor needs periodic-copy/root eligibility and event phase, not merely the
unmarked chart identity. The fifth origin/successor coordinate has not
disappeared.

## Fixed-cost evaluation and byte failure

An optimistic token containing a 19-bit chart ID, two quantized local
coordinates, and valid state was charged at eight bytes. At 81 directions its
atlas alone costs `43,133,472` bytes, leaving `7,987,680` bytes, or only
`29.5255` bytes per chart, under the `51,121,152`-byte cap.

- An optimistic 16-byte/chart descriptor table costs `4,328,560` bytes and
  fits, but cannot describe the exact primitives.
- A 32-byte/chart table costs `8,657,120` bytes and already exceeds the cap.
- Exact raw blade, hair, lanceolate, and tube recipes require at least 9--10
  floating parameters before type, parent, local panel, periodic mark, and
  variable tube control points are charged.

More importantly, an entire chart is not generally one invertible surface.
Blades use `sin` and noninteger powers; tubes contain variable polyline
sections and side panels; lanceolate and filament surfaces are piecewise
ribbons. Exact ray intersection requires selecting a local panel and, for the
nonlinear forms, solving an intersection. The baked local panel cannot be
reused at a new angle: the measured exact-panel support was zero. Selecting
segments or roots live becomes precisely the traversal/candidate problem the
contract excludes.

## Missing metadata and bounded recook

GCRP/v4 currently contains no chart record. Its reserved fourth triangle word
is zero for every triangle; it has no chart-local UV, analytic recipe table, or
per-chart species/population mark. If a future representation gives chart
provenance a valid role, the missing data can be added by a bounded offline
recook:

1. instrument `appendBlade`, `appendTube`, `appendLanceolateSurface`, and
   `appendHairFilament` with a `ChartRecipe` recorder;
2. write the stable chart ID into the currently zero fourth triangle `u32`;
3. append a versioned recipe table and optional quantized local panel/UV data;
4. record parent chart, species/community/population root, periodic-copy
   semantics, and primitive kind.

This recook is **not** implemented because the visibility gate failed first.
It is preserved as a finite implementation route, not a reason to promote the
carrier.

## Overlapping species, moss, and publication boundary

Future overlap can retain this provenance offline with the categorical mark

```text
(community, species, population/root, chart, local coordinate).
```

Grass, forb, moss, and lichen geometry must be unioned in one offline periodic
community truth field; the winning event carries colour, normal, and the full
mark. Unrelated winners are never numerically blended and runtime never queries
one field per species. This preserves O(1) copy/species behaviour. It also
creates more occlusion boundaries and successors, so it cannot repair the
single-species failure measured here.

The public botanical sources and their recipe bindings are recorded in
`ESTONIAN-GRAMINOIDS-README.md`. The chart construction, angular-support test,
same-line chart-run test, byte proof, and negative result in this report are
LAAS derivations and measurements, not claims made by those botanical sources
or by Sannikov. The immutable source/recipe/tool/result hashes above preserve
that distinction for a future paper.

## Park/resume decision

Do not integrate a procedural chart token as the live visibility carrier. Keep
the deterministic chart partition and proposed `ChartRecipe` metadata as
reusable offline provenance.

Resume only if a new representation supplies a mathematically complete
analytic sharing mechanism that:

- recovers disoccluded winning charts without a runtime candidate set;
- answers same-line inside successors including periodic-copy marks;
- evaluates the selected primitive with fixed closed-form work;
- proves complete resident bytes `<=51,121,152` and one/few fixed reads; and
- becomes no more expensive when multiple species and moss overlap.

Failure does not authorize traversal, marching, loops, candidates, more passes,
distance-dependent work, or a raised memory cap.
