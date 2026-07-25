# Inverse-authored F4/F6 ground-cover mathematics

Date: 2026-07-23  
Status: **F4 and complementary F6 measured RED; vertically cohorting F5 rejected analytically; the cross-laminated ruled/sheet basis is parked**

## 1. The upstream correction

The failed lateral-core experiment solved the wrong compilation problem.  It
required every emitted ruled sheet to remain within at most `2.5--10 mm` of
the arbitrary Calamagrostis triangle source through one entire global height
interval.  That is a useful exact-correspondence certificate, but it is not
the visual objective.  It retained only `379` short arc edges, then delegated
most opaque foliage and head transfer to four smooth analytic modes.  The
result was forced toward an almost-everywhere opaque field.

The corrected problem separates two statements which must never be conflated:

1. **reconstruction:** does the fixed live algebra reproduce one compiled
   community `G*` without view-dependent stretch, wedges, a carrier plane, or
   a missing ray dimension?;
2. **inverse authoring:** can a community inside the reconstructible class be
   made visually equivalent to the accepted botanical reference after the
   same physical pixel filter?

Reconstruction below is exact in the continuous model.  Inverse authoring is
the only approximation and is judged directly in images/ray transfer.  There
is no longer a demand for triangle identity or millimetric pointwise proximity
when neither quantity is visible.

This is an upstream source-encoding change, not a relaxed runtime budget.  It
adds no read, loop, march, candidate list, species evaluation, pass, geometry,
or resident-memory allowance.

## 2. Compiled class

The baseline uses four opaque chromatic ruled fields.  The user explicitly
permits a small read-count fluctuation when it buys a well-founded quality
increase, so the gate also measures a five-field variant.  Field `f` contains:

- a periodic compact arc network `Gamma_f` on the phase torus
  `T^2 = R^2/Lambda`;
- one field-global height interval `I_f=[h_f^-,h_f^+]`;
- one structured affine shear `beta_f in R^2`;
- categorical arc payloads (growth-form/species mark, material, two-sidedness,
  palette class and tangent convention); and
- a small field-global analytic height palette/ramp.  The ramp changes
  appearance only.  It never changes opacity or owner eligibility.

The exact world surface is

```text
S_f = { (q + beta_f h, h) : q in Gamma_f, h in I_f }.
```

`Gamma_f` may contain thousands of open arcs, loops, crossings and disconnected
components.  Its complexity, the number of repeated plants, and the number of
species do not enter the live work.  They are reduced offline to one
first-passage atlas for the field.  Open arcs are intentional: `arc x I_f` is
a cap-free ribbon, not the boundary of a filled mask.

The two Tier-1 population transforms reuse the same four atlases.  They add
world variance but do not duplicate resident bytes.

### 2.1 Exact live query

For a rest-space live ray `r(t)=o+t d`, transform out the field shear:

```text
q(t) = o_xz - beta_f o_y
     + t (d_xz - beta_f d_y)
     = q_0 + t p_f.
```

Clip the forward ray analytically against `I_f`, scene depth and the declared
cover horizon, obtaining `[t_a,t_b]`.  If `s_f=|p_f|>0`, define

```text
q_a   = q_0 + t_a p_f,
omega = p_f / s_f,
L     = s_f (t_b-t_a).
```

The intrinsic first passage is

```text
rho_f(q_a,omega,L)
  = min { ell in [0,L] : q_a + ell omega in Gamma_f }.
```

The exact world-ray hit is

```text
t_f = t_a + rho_f/s_f.
```

At the removable pole `s_f=0`, the result is `t_a` exactly when `q_a` belongs
to `Gamma_f`, otherwise miss.  Camera-on-surface is outside the exterior
quality contract; ordinary air gaps inside the overall plant-height slab are
not.

The normal follows from the two tangents

```text
(gamma'_f,0),  (beta_f,1),
```

and the categorical record supplies the coupled mark/palette.  No depth,
normal, colour or owner is blended across fields or angular owners.

### 2.2 Fixed resource law

One complete record read is issued for each `(population,field)` pair and one
existing control/material read remains:

```text
T(F) = 2 populations * F fields + 1 control.

T(4)=9,    T(5)=11.
```

The eight returned events are elected by a compile-time fixed branchless
minimum network.  This is the finite representation algebra, not a per-copy,
per-triangle, per-cell or per-species candidate list.  There is no traversal
and no data-dependent number of operations.

The F5 variant is permitted only when its extra categorical cohort produces a
clear ideal-field quality gain.  The two extra reads may not be repurposed as
view correction, brute-force candidates, residual modes, or a hidden species
loop.

For atlas resolution `P_x x P_z x N_omega`, complete-record size `B`, and mip
factor `mu`, resident bytes are

```text
M(F) = F P_x P_z N_omega B mu.
```

Both populations share this storage.  F4 must remain at or below the current
accepted ground-cover allocation.  F5 must publish its exact incremental
bytes and locality cost; the read relaxation is not an implicit memory-cap
relaxation.

## 3. Why four/five fields are not four/five plants

The fixed count limits independent affine/height cohorts, not botanical
detail.  Each `Gamma_f` is an arbitrarily detailed periodic community-wide arc
network.  Many species and cover types are cooked jointly into its categorical
payloads.  A nearer event wins regardless of species, so overlap has zero
per-species live cost.

For the current single-species Calamagrostis gate, the F4 allocation is:

1. lower green blades and basal foliage;
2. tall culms and upper leaves;
3. purple/brown panicle scaffold and glumes;
4. pale/purple micro-ribbons for the dense fluffy head population.

The two global population transforms supply a second orientation/phase of all
four cohorts.  This allocation is not frozen until the ideal-field optimizer
reports its Pareto curve.  A moss/low-sward community may spend the same four
cohorts differently; runtime layout and cost remain identical.

The F5 allocation splits foliage into lower, middle and tall cohorts, followed
by lower and upper reproductive cohorts.  This is the most defensible use of
two extra reads: it reduces the false vertical persistence of the source's
largest opaque area while keeping both coloured head cohorts categorical.

The fourth cohort deliberately keeps recognition-critical head colour and
cutoff structure categorical.  It does not ask a positive smooth residual to
imitate opaque cream/purple landmarks.

## 4. The two geometric controls that matter

For one arc parameterised by unit phase arclength, the ruled sheet is

```text
X(s,h)=(gamma(s)+beta h,h).
```

Its top-view projected area, counted with multiplicity, is

```text
A_top = |I_f| integral_(Gamma_f)
                    |det(gamma'(s),beta_f)| ds.
```

Thus top-down occupancy is controlled by shear, arc length and arc orientation;
it does not require a horizontal cap or floating plane.

For an exactly horizontal world ray, `p_f=d_xz`; its clear corridors are the
literal gaps of `Gamma_f`.  Top-view coverage and lateral air are therefore
separate authoring controls.  The failed positive Fourier fit coupled them
through an almost-everywhere density floor; this construction does not.

These equations become hard cook constraints.  They are not post-hoc visual
explanations.

## 5. No-wedge property of the ideal field

The continuous candidate is one fixed world-space surface union.  Every live
camera ray is transformed analytically and intersected with that same union.
Camera elevation is not sampled, flattened, clamped or substituted.  A camera
motion can change the elected event only at a true silhouette, occlusion swap,
arc endpoint or field intersection of `G*`.

Therefore the ideal model has no mechanism for a direction-cell error to
create a world wedge of width `r DeltaTheta`.  A source/candidate discrepancy
is a static authoring displacement and projects approximately as

```text
epsilon_pixels ~= epsilon_world / (D theta_pixel),
```

so it honestly shrinks with distance.  This statement does not excuse finite
atlas angular errors; those are gated separately after the continuous model is
green.

## 6. Direct inverse-authoring objective

Let `R*` be the accepted reference renderer and `R_G` the exact continuous
four-field ray oracle.  For identical anisotropic pixel bundles `B`, camera
set `V`, physically reachable prefix cutoffs `b`, and Tier-1 population
transforms on both sides, optimize the arc networks, field intervals, shears
and categorical payloads against the complete filtered transfer:

```text
min_G  sum_(v in V, B)
       [ w_A   |A_G-A*|
       + w_C   ||C_G-C*||_inf
       + w_S   silhouette_loss
       + w_X   transverse_event_error
       + w_M   categorical_colour/mark_loss ]
       + w_R complexity(G).
```

This is a cook-side inverse problem.  It may use expensive global search,
automatic differentiation, mutation and exact source ray queries.  None of
that work exists in the live shader.

The following are hard constraints rather than soft averages:

- phase-stratified exact-horizontal and low-oblique air cones selected from
  the source remain clear through their measured distance;
- purple/brown/cream head landmarks retain categorical colour and their
  filtered cutoff mass;
- root/terrain attachment and total height remain within the accepted plant
  envelope;
- no field-global interval creates a connected false plane/sheet region;
- both global population transforms are present in the reference and
  candidate queries;
- maximum and p95 intrinsic free path are charged to the finite-atlas memory
  law; and
- reference and candidate are filtered through the same distance-dependent
  physical footprint before visible-detail errors are judged.

The optimizer starts from the certified `379`-edge relay network, but it may
move, split, insert and delete arcs beyond the old `10 mm` corridor.  It is
optimizing the observable object, not pretending the arbitrary source mesh is
already in the reconstructible class.

## 7. Residual and minification boundary

The first ideal gate has no general smooth residual.  The proved identity
theorem matters: a nonnegative finite analytic density which is exactly zero
on an open swept family of air rays is zero everywhere, while a small four-mode
fit cannot simultaneously retain sparse opaque landmarks and long clear
corridors.  A future residual is admitted only for genuinely soft,
optically-thin fuzz after all visible categorical cutoff mass is already in the
four fields.

Near-field truth uses categorical first events.  At projected sub-pixel scale,
the finite atlas may change semantics to a jointly baked community transfer
LOD.  That LOD is an explicit filtered approximation, not independently
averaged field opacity.  It must pass translation/shimmer and colour gates and
must occupy the same fixed reads and atlas bytes.  No minification decision is
needed for the ideal continuous-field go/no-go.

## 8. Finite-atlas gate after ideal green

For an arc first hit with intrinsic free path `rho`, an angular perturbation
obeys locally

```text
|Delta q| <= rho |Delta omega| + O(Delta omega^2).
```

This is local only while owner/endpoint margins remain certified.  The bake
therefore reports `rho_p95`, the long-tail mass, unsafe owner cells and exact
air-corridor cells per field.  `N_omega` is then selected from the measured
filtered displacement tolerance, and the exact byte equation in Section 2.2
must pass.  Long-path unsafe cells may not be hidden with interpolation.

The finite gate is one complete attempt plus one diagnosis/fix rerun.  It
measures connected wrong regions and millimetric camera translations, not only
average image loss.  A red ideal field cannot be rescued by atlas resolution;
a green ideal field whose required `N_omega` exceeds the current allocation is
also red.

## 9. Decision sequence

1. **Ideal continuous inverse-authoring Pareto gate.** Emit F4 and F5
   four-direction/oblique QA images plus distance-stratified transfer metrics.
   This is the first user-inspectable result.  F5 advances only for a clear
   quality jump attributable to its additional categorical cohort.
2. **Finite atlas gate.** Only after ideal green, bake complete categorical
   records and prove read/byte/translation bounds.
3. **Runtime transcription.** Only after both gates are green.  Preserve the
   exact nine-read algebra and perform the mandatory real-WebGPU boot before
   any URL is sent.

If the ideal F4 gate is red after one coherent optimization/fix cycle, the
representation is parked.  The honest conclusion is then that the current
quality target needs either a visibly relaxed authored morphology or a larger
fixed family/read budget; no filter, direction row, smooth residual, or shader
brute force may disguise that result.

## 10. Provenance boundary

The invariant-ray lift and `1/s_f` reconstruction descend from Sannikov's
precomputed repeating-ray grass construction.  The amplification dichotomy,
cap-free open-arc generalization, relay compiler, exact-horizontal pole, and
the failed four-mode residual evidence are local results recorded in:

- `GRASS-EXACT-REPRESENTATION-THEORY.md`;
- `GRASS-LATERAL-CORE-ANALYTIC-RESIDUAL-MATH.md`; and
- `GRASS-LATERAL-CORE-RESIDUAL-CPU-GATE.md`.

The new contribution in this note is the explicit upstream replacement of
pointwise mesh compilation by constrained inverse authoring of the observable
filtered exterior transfer, while keeping the proved fixed-cost class and all
air/colour/memory constraints.

## 11. F4 result and the complementary F6 correction (2026-07-23)

### 11.1 What the first ideal gate actually disproved

The first F4 seed used real semantic source cross-sections but swept every
selected arc through a much broader field-global interval.  The continuous
ray oracle was exact for that compiled object; the object itself was wrong.
The bound artifact is:

```text
data/work/groundcover-inverse-authored-ideal-field-gate/
  37b0cf1d33f632b5/fb4951afa064444f/e61c21c02adc7859/
```

The positive result is structural: exact-horizontal air remained exactly
clear and the old camera-centred wedge/warp mechanism was absent.  The
negative result is decisive for the selected basis: top-view IoU was about
`0.316` at 1--8 m, oblique RGB p95 remained about `0.45`, and broad swept head
arcs produced metres of false early depth at grazing.

This is not repaired by dividing the reproductive height range into three
vertical cohorts.  The accepted Calamagrostis reproductive range is
approximately `410.5 mm`.  Three disjoint bands force at least one band of
width

```text
B >= 410.5 / 3 = 136.8 mm.
```

Measured source-primitive p90 vertical spans are only approximately `6.38 mm`
for panicle-axis pieces, `5.36 mm` for glumes and `4.01 mm` for hairs.  A
retained microfeature would therefore persist falsely for roughly `25--55x`
its real vertical support.  At view elevation `alpha`, the possible false
along-ray lead is

```text
(B-l)/sin(alpha),
```

which already exceeds `1.5 m` at 5 degrees and `7.5 m` at 1 degree.  Binary
thinning cannot change this ratio: for retain/delete variable `x`, true
support measure is proportional to `x l` and false extension to `x(B-l)`.
Deleting enough arcs to remove the false foreground also deletes the fluffy
head.  A different shear moves the false set but cannot remove it for every
exterior direction.

Therefore the old F5 allocation (two foliage plus three *vertical*
reproductive cohorts) is not a second quality attempt.  It is rejected by a
source-span lower bound before spending another end-to-end run.

### 11.2 Complementary exact fields

The failed basis tried to obtain every projection from surfaces whose
invariance axis was mostly vertical.  The upstream correction is to use two
complementary, individually exact periodic families.

An affine ruled field remains

```text
V_f = { (q + beta_f h, h) : q in Gamma_f, h in I_f }.
```

A horizontal masked sheet is

```text
H_j = { (q, h_j) : q in M_j subset T^2 },
```

where `M_j` is an arbitrary periodic, coupled colour/normal/mark mask baked
from a narrow source-height band.  For live ray `r(t)=o+td`, its exact query
is

```text
t_j = (h_j-o_y)/d_y,
q_j = o_xz+t_j d_xz,
hit iff t_j is inside the forward/scene interval and q_j in M_j.
```

At the exact pole `d_y=0`, the sheet is a miss for an exterior origin not on
the sheet.  There is no epsilon and no hidden horizontal view: the ruled
fields are the representation at that pole.  Conversely, vertical rays are
the best-conditioned sheet queries.  Every result is an intersection with
one fixed world-space union, and the first event is the categorical minimum
of the fixed family results.  No direction is sampled in the ideal model.

This pairing removes the specific lower bound above.  A reproductive source
primitive assigned to sheet `j` is displaced in height by at most half its
sheet band's width; it is **not extended through that band**.  Its projected
error is a static world displacement and therefore shrinks in pixels with
distance.  Sheet coverage also tends to zero with the correct foreshortening
as the view approaches horizontal, while ruled coverage remains.  No
view-dependent fade or geometry blend is introduced.

### 11.3 F6 allocation and why 13 reads are justified

The final ideal-field Pareto attempt compares:

1. lower/basal foliage ruled field;
2. tall/upper foliage ruled field;
3. lower-purple panicle scaffold ruled field, pruned to structural axes and
   recognition-bearing ribbons rather than plume hairs;
4. upper/alternate-orientation panicle scaffold ruled field;
5. lower/middle reproductive horizontal mask;
6. middle/upper reproductive horizontal mask.

With two Tier-1 population transforms and the existing control read,

```text
T = 2*6 + 1 = 13 reads.
```

The increase from 9 to 13 is not a correction iteration, direction sample,
filter, species query or runtime candidate list.  Four reads buy two complete
complementary geometric fields across the two populations, eliminating the
measured 25--55x vertical-persistence mechanism.  Horizontal masks are 2D;
they do not carry the `N_omega` dimension of a ruled first-passage atlas, so
their byte increase is far smaller than their read increase.  Exact bytes,
bindings and locality remain a gate and the two populations still share all
stored data.

An optional F5 ablation uses one horizontal mask and costs 11 reads.  It is a
cost point, not the quality target: one common reproductive plane risks a
synchronised height cue.  F6 is allowed to advance only if the two-sheet
separation visibly and metrically removes that cue.

### 11.4 Binding second-attempt gate

The second and final ideal attempt must use the same source/candidate pixel
bundles and report F5/F6 separately.  In addition to the existing metrics it
must report:

- ruled versus sheet first-visible mass by payload and view;
- false early depth attributable to each family type;
- top/18/10/5/1-degree results, including exact horizontal;
- sheet-height synchronisation as connected equal-depth regions;
- 1--4.5 mm translation stability;
- F5/F6 read counts and exact prospective atlas bytes.

If F6 remains recognisably cross-laminated, produces plane bands, or misses
the head from any exterior angle, the finite-atlas/runtime phases stay parked.
The objective resume condition is then a new invariant that localises finite
microfeatures in both height and horizontal phase without a sampled camera
direction, traversal, or per-element runtime work.  More vertical cohorts are
not that invariant.

## 12. Complementary F6 result and park decision (2026-07-23)

The second complete ideal-field attempt is bound here:

```text
data/work/groundcover-complementary-f6-ideal-field-gate/
  37b0cf1d33f632b5/bc8b67ff7684bdf2/2e17a0c43761211a/
metrics SHA-256:
  cb611796e483023ae21fcfcf9e1af39d1ff4b62d2262fa0d4b59277ea572c5bd
network SHA-256:
  5a0e46a91cb155282b5482e649d8c34219a6ac5f168f0a33b37dec5c07b0969c
```

It evaluated the exact continuous union represented by four ruled fields and
two horizontal masks, with two Tier-1 population transforms.  The candidate
contained `61,576` offline oracle triangles derived from `30,788` arc-edge
equivalents versus `2,171,134` source triangles.  These triangles are only an
exact CPU intersection oracle for the mathematical field; no runtime mesh or
shader path was introduced.  Its prospective live shape was six field reads
per population plus one control read, hence `13` fixed reads.

The two horizontal masks bought exactly the improvement they were intended to
buy, but not enough to make the representation correct.  At the one-metre
vertical view, F6 improved IoU from approximately `0.316` to `0.389`, RGB p95
from approximately `0.725` to `0.470`, and first-hit depth p95 from
approximately `1.03 m` to `0.179 m`.  The gain proves that localising head mass
in height was necessary.  It also proves that four extra reads are not by
themselves justified: every binding quality threshold remained red.

Representative F6 measurements are:

| exterior view | IoU | RGB p95 | first-hit depth p95 | early-depth error |
|---|---:|---:|---:|---:|
| vertical, 1 m | 0.389 | 0.470 | 0.179 m | 9.7% |
| vertical, 8 m | 0.368 | 0.334 | 0.168 m | 9.4% |
| 18 degrees, 1 m | 0.917 | 0.453 | 2.821 m | 72.6% |
| 10 degrees, 1 m | 1.000 | 0.399 | 3.717 m | 66.7% |
| 5 degrees, 1 m | 1.000 | 0.338 | 6.781 m | 83.9% |
| 1 degree, 1 m | 1.000 | 0.328 | 8.562 m | 97.9% |

The apparently perfect low-angle IoU is not a success: source and candidate
both saturate coverage while the candidate selects the wrong, much earlier
periodic successor.  That is why colour and first-hit depth remain badly
wrong.  The one-metre vertical case also changes categorical class on `22.6%`
and `28.0%` of otherwise stable reference pixels under `2.5 mm` and `4.5 mm`
translations.  Exact cardinal-horizontal air stayed exactly clear to `4 m`,
so this is not a hidden epsilon or horizontal-pole regression.

The numbered QA images show the same failure directly: the candidate remains
recognisably cross-laminated and becomes broad pale bands at low elevation,
rather than preserving the source plant structure.  The ruled families still
create false periodic successors; the sheets add correctly localised top mass
but cannot repair the ruled winner's depth, ownership, or colour at oblique
angles.

### 12.1 Park record

- **Effort spent:** two complete ideal-field attempts, F4 and complementary
  F6, plus the analytic rejection of vertically cohorting F5.
- **Reusable result:** a direction-analytic, exact-horizontal ruled-field
  oracle; exact horizontal-mask complement; identical-filter source/candidate
  gate; semantic Calamagrostis source partition; and the measured fact that
  head-height localisation materially improves vertical reconstruction.
- **Exact blocker:** finite microfeatures need coupled localisation in height,
  horizontal phase, and first-visible successor identity.  A fixed union of
  field-global ruled intervals and a few height sheets cannot preserve all
  three for sparse overlapping grass at every exterior angle.
- **Why focus moves:** F6 exhausted the declared two-real-attempt budget and
  still misses RGB `<=0.15`, depth `<=0.05 m`, top silhouette, and translation
  stability by large margins.  More cohorts or masks merely subdivide the
  same failed invariance and spend reads linearly.
- **Active fallback:** derive a mathematically distinct split in which a
  sparse structural core is reconstructed by exact direction-analytic fields
  and only genuinely sub-pixel plume mass is represented as filtered
  extinction/radiance transfer.
- **Objective resume condition:** a new invariant or proven bounded selector
  that couples finite support and successor identity without sampled camera
  direction, runtime traversal, per-element work, or unbounded memory.

No finite-atlas or runtime transcription follows from this RED result.
